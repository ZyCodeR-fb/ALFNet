"""The launcher, resume, and the audit/eval/collect gates.

The first launcher was serial, had no tmux, never passed `--resume`, and its
`--resume` would not have restored anything if it had. Each of those four is
asserted here, because "the queue ran" and "the queue ran the experiment the
plan registered, and can be interrupted without losing it" are different claims
and only the second one is worth anything.

Nothing in this file trains for long: the expensive property (resume equals
uninterrupted training) is checked on three capped epochs of the smaller model,
which exercises exactly the code paths a hundred-epoch run would.
"""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import train_v4, v4_launcher
from scripts.build_v4_development_plan import assert_plan_still_holds, load_plan
from scripts.v4_launcher import (
    LauncherRefused,
    assign_slots,
    parse_devices,
    plan_runs,
    run_state,
)

CONFIG = ROOT / "configs/paper/v4/PEMS08_MoGERNN.yaml"
TRAIN_KW = dict(device="cpu", seed=32, max_train_batches=3,
                max_validation_batches=3, batch_size=4, num_workers=0)


@pytest.fixture(scope="module")
def plan():
    return assert_plan_still_holds(load_plan())


# --- slots and the queue -------------------------------------------------------


def test_devices_parse_into_cuda_names():
    assert parse_devices("0,1,2") == ["cuda:0", "cuda:1", "cuda:2"]
    assert parse_devices("cuda:3") == ["cuda:3"]
    assert parse_devices("cpu") == ["cpu"]
    with pytest.raises(LauncherRefused, match="empty"):
        parse_devices("")


def test_one_job_per_gpu_unless_sharing_is_explicit():
    assert assign_slots(3, parse_devices("0,1,2")) == ["cuda:0", "cuda:1", "cuda:2"]
    assert assign_slots(1, parse_devices("0")) == ["cuda:0"]
    with pytest.raises(LauncherRefused, match="ALLOW_GPU_SHARING"):
        assign_slots(3, parse_devices("0"))
    # ... and with it, three slots on the one card, deliberately.
    assert assign_slots(3, parse_devices("0"), allow_sharing=True) == \
           ["cuda:0"] * 3
    with pytest.raises(LauncherRefused, match="at least 1"):
        assign_slots(0, parse_devices("0"))


def test_the_queue_is_exactly_twelve_unique_runs(plan, tmp_path):
    runs = plan_runs(tmp_path, plan)
    assert len(runs) == 12
    names = [run.name for run in runs]
    assert len(set(names)) == 12
    assert {run.seed for run in runs} == {32, 42, 52}
    assert {run.dataset for run in runs} == {"PEMS08", "PEMSBAY"}
    # PEMS03/PEMS04 are not reachable from the queue at all.
    assert not [run for run in runs if run.dataset in ("PEMS03", "PEMS04")]


def test_the_health_phase_is_every_seed_thirty_two_run(plan, tmp_path):
    runs = plan_runs(tmp_path, plan)
    health = [run for run in runs if run.phase == 0]
    later = [run for run in runs if run.phase == 1]
    assert len(health) == 4 and {run.seed for run in health} == {32}
    assert len(later) == 8 and {run.seed for run in later} == {42, 52}
    # Phase 0 comes first in the list, so no later seed can be dispatched
    # before the barrier.
    assert [run.phase for run in runs] == [0] * 4 + [1] * 8


def test_the_dry_run_starts_nothing(plan, tmp_path, capsys):
    v4_launcher.dry_run(tmp_path, plan, ["cuda:0", "cuda:1", "cuda:2"])
    output = capsys.readouterr().out
    assert "12 unique runs, no duplicates" in output
    # Four phase-0 lines, each naming a seed-32 run.
    health = [line for line in output.splitlines() if "/seed32 " in line]
    assert len(health) == 4, health
    assert "phase 0: health (seed 32)" in output
    assert not list(tmp_path.rglob("*.pt"))


# --- run state ------------------------------------------------------------------


def test_run_state_classifies_each_directory(plan, tmp_path):
    run = plan_runs(tmp_path, plan)[0]
    assert run_state(run) == "NEW"
    run.directory.mkdir(parents=True)
    (run.directory / "stray.txt").write_text("x", "utf-8")
    assert run_state(run) == "UNRESUMABLE"
    (run.directory / train_v4.RUN_STATE_NAME).write_text("x", "utf-8")
    assert run_state(run) == "RESUMABLE"
    (run.directory / train_v4.FAILED_MARKER).write_text("{}", "utf-8")
    assert run_state(run) == "FAILED"
    (run.directory / train_v4.COMPLETE_MARKER).write_text("{}", "utf-8")
    assert run_state(run) == "COMPLETE"


def test_a_locked_run_is_not_started_twice(plan, tmp_path):
    run = plan_runs(tmp_path, plan)[0]
    first = v4_launcher.acquire_lock(run)
    assert first is not None
    run.lock = first
    assert v4_launcher.acquire_lock(run) is None, "a second launcher took the lock"
    v4_launcher.release_lock(run)
    second = v4_launcher.acquire_lock(run)
    assert second is not None
    run.lock = second
    v4_launcher.release_lock(run)


def test_a_completed_run_is_skipped_and_a_failed_one_refused(plan, tmp_path):
    launcher = v4_launcher.Launcher(tmp_path, plan, ["cpu"])
    run = launcher.runs[0]
    run.directory.mkdir(parents=True, exist_ok=True)
    (run.directory / train_v4.FAILED_MARKER).write_text("{}", "utf-8")
    with pytest.raises(LauncherRefused, match="--retry-failed"):
        launcher.start(run, "cpu")
    (run.directory / train_v4.FAILED_MARKER).unlink()
    (run.directory / "stray.txt").write_text("x", "utf-8")
    with pytest.raises(LauncherRefused, match="--restart-unresumable"):
        launcher.start(run, "cpu")


# --- true resume ---------------------------------------------------------------


@pytest.fixture(scope="module")
def resume_pair(tmp_path_factory, plan):
    """Three epochs straight, and one-then-resume-to-three."""
    straight = tmp_path_factory.mktemp("straight")
    a = train_v4.train(CONFIG, output_root=straight, max_epochs=3, plan=plan,
                       **TRAIN_KW)
    split = tmp_path_factory.mktemp("split")
    train_v4.train(CONFIG, output_root=split, max_epochs=1, plan=plan, **TRAIN_KW)
    for marker in Path(split).rglob(train_v4.COMPLETE_MARKER):
        marker.unlink()                       # make it look interrupted
    b = train_v4.train(CONFIG, output_root=split, max_epochs=3, resume=True,
                       plan=plan, **TRAIN_KW)
    return a, b


def test_resume_reproduces_uninterrupted_training(resume_pair):
    a, b = resume_pair
    assert a["epochs_run"] == b["epochs_run"] == 3
    assert a["best_epoch"] == b["best_epoch"]
    for left, right in zip(a["history"], b["history"]):
        assert left["epoch"] == right["epoch"]
        assert left["train"]["base"] == right["train"]["base"]
        assert left["validation"]["life_avg_mae"] == right["validation"]["life_avg_mae"]


def test_resume_reproduces_the_model_and_optimizer_state(resume_pair):
    a, b = resume_pair
    left = torch.load(Path(a["run_directory"]) / train_v4.RUN_STATE_NAME,
                      weights_only=False, map_location="cpu")
    right = torch.load(Path(b["run_directory"]) / train_v4.RUN_STATE_NAME,
                       weights_only=False, map_location="cpu")
    assert set(left["model_state"]) == set(right["model_state"])
    for name in left["model_state"]:
        assert torch.equal(left["model_state"][name], right["model_state"][name]), name
    assert left["optimizer_state"]["param_groups"] == \
           right["optimizer_state"]["param_groups"]
    for key, state in left["optimizer_state"]["state"].items():
        for tensor in ("exp_avg", "exp_avg_sq"):
            assert torch.equal(state[tensor],
                               right["optimizer_state"]["state"][key][tensor])
    assert left["next_epoch"] == right["next_epoch"] == 3


def test_the_run_state_records_what_a_resume_needs(resume_pair):
    a, _ = resume_pair
    state = torch.load(Path(a["run_directory"]) / train_v4.RUN_STATE_NAME,
                       weights_only=False, map_location="cpu")
    for field in ("schema_version", "completed_epoch", "next_epoch", "best_epoch",
                  "best_validation_life_avg_mae", "patience_wait", "history",
                  "model_state", "optimizer_state", "scheduler_state", "rng_state",
                  "config_sha256", "plan_sha256", "code_sha256", "payload_sha256",
                  "unseen_sha256", "manifest_sha256", "static_asset_hash",
                  "train_exposure_hash", "validation_exposure_hash", "scaler",
                  "split_fractions", "teacher_forcing_policy", "shuffle_policy"):
        assert field in state, field
    assert state["shuffle_policy"] == "lcbt_v4_shuffle_v1"
    assert set(state["rng_state"]) >= {"python", "numpy", "torch_cpu"}


@pytest.mark.parametrize("field, value", [
    ("seed", 999), ("dataset", "PEMSBAY"), ("config_sha256", "0" * 64),
    ("plan_sha256", "0" * 64), ("payload_sha256", "0" * 64),
    ("shuffle_policy", "something_else"), ("parameter_count", 1),
])
def test_a_foreign_run_state_is_refused(tmp_path, plan, field, value):
    """Right shape, wrong experiment: resuming into it would splice two runs."""
    root = tmp_path / "root"
    train_v4.train(CONFIG, output_root=root, max_epochs=1, plan=plan, **TRAIN_KW)
    directory = next(root.rglob("seed32"))
    (directory / train_v4.COMPLETE_MARKER).unlink()
    path = directory / train_v4.RUN_STATE_NAME
    state = torch.load(path, weights_only=False, map_location="cpu")
    torch.save(dict(state, **{field: value}), path)
    with pytest.raises(train_v4.RunnerRefused, match="different run"):
        train_v4.train(CONFIG, output_root=root, max_epochs=2, resume=True,
                       plan=plan, **TRAIN_KW)


def test_a_corrupt_run_state_is_refused(tmp_path, plan):
    root = tmp_path / "root"
    train_v4.train(CONFIG, output_root=root, max_epochs=1, plan=plan, **TRAIN_KW)
    directory = next(root.rglob("seed32"))
    (directory / train_v4.COMPLETE_MARKER).unlink()
    (directory / train_v4.RUN_STATE_NAME).write_bytes(b"not a checkpoint")
    with pytest.raises(train_v4.RunnerRefused, match="could not be read"):
        train_v4.train(CONFIG, output_root=root, max_epochs=2, resume=True,
                       plan=plan, **TRAIN_KW)


def test_an_unresumable_directory_is_refused_not_overwritten(tmp_path, plan):
    root = tmp_path / "root"
    train_v4.train(CONFIG, output_root=root, max_epochs=1, plan=plan, **TRAIN_KW)
    directory = next(root.rglob("seed32"))
    (directory / train_v4.COMPLETE_MARKER).unlink()
    (directory / train_v4.RUN_STATE_NAME).unlink()
    with pytest.raises(train_v4.RunnerRefused, match="restart-incomplete"):
        train_v4.train(CONFIG, output_root=root, max_epochs=2, resume=True,
                       plan=plan, **TRAIN_KW)
    # restart archives rather than overwrites
    before = len(list((ROOT / "outputs_v4/development_aborted").glob("restart-*"))) \
        if (ROOT / "outputs_v4/development_aborted").exists() else 0
    train_v4.train(CONFIG, output_root=root, max_epochs=1, restart_incomplete=True,
                   plan=plan, **TRAIN_KW)
    after = len(list((ROOT / "outputs_v4/development_aborted").glob("restart-*")))
    assert after == before + 1


def test_a_completed_run_is_still_never_overwritten(tmp_path, plan):
    root = tmp_path / "root"
    train_v4.train(CONFIG, output_root=root, max_epochs=1, plan=plan, **TRAIN_KW)
    with pytest.raises(train_v4.RunnerRefused, match="marked COMPLETE"):
        train_v4.train(CONFIG, output_root=root, max_epochs=1, resume=True,
                       plan=plan, **TRAIN_KW)


# --- the shuffle ----------------------------------------------------------------


def test_the_shuffle_is_a_function_of_seed_and_epoch_only():
    from experiments.v4_runner import EpochSeededSampler
    from lib.lcbt_v4 import shuffle_seed

    data = list(range(50))
    first = EpochSeededSampler(data, 32)
    second = EpochSeededSampler(data, 32)
    # A fresh sampler at epoch 7 matches one that has walked there, so a resumed
    # run sees the order an uninterrupted one would have.
    walked = None
    for epoch in range(8):
        first.set_epoch(epoch)
        walked = list(iter(first))
    assert list(iter(second.set_epoch(7))) == walked
    assert list(iter(second.set_epoch(0))) != walked
    assert shuffle_seed(32, 7) != shuffle_seed(33, 7)
    assert shuffle_seed(32, 7) == shuffle_seed(32, 7)


def test_the_shuffle_policy_is_stable_across_processes():
    script = (f"import sys; sys.path.insert(0, {str(ROOT)!r})\n"
              "from lib.lcbt_v4 import shuffle_seed\n"
              "print(shuffle_seed(32, 7))\n")
    from lib.lcbt_v4 import shuffle_seed

    for salt in ("0", "12345"):
        result = subprocess.run([sys.executable, "-c", script], capture_output=True,
                                text=True, env={**os.environ,
                                                "PYTHONHASHSEED": salt})
        assert result.returncode == 0, result.stderr
        assert int(result.stdout.strip()) == shuffle_seed(32, 7)


# --- signals --------------------------------------------------------------------


def test_a_signal_stops_the_trainer_and_leaves_no_orphan(tmp_path, plan):
    """SIGTERM to the process group; the run stays resumable."""
    root = tmp_path / "root"
    command = [sys.executable, str(ROOT / "scripts/train_v4.py"),
               "--config", "configs/paper/v4/PEMS08_MoGERNN.yaml",
               "--device", "cpu", "--seed", "32",
               "--output_root", str(root), "--development_plan",
               # Capped so an epoch boundary arrives in seconds; the signal
               # path being exercised does not care how long the epoch was.
               "--max_train_batches", "2", "--max_validation_batches", "2"]
    process = subprocess.Popen(command, cwd=str(ROOT), start_new_session=True,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        deadline = time.time() + 240
        while time.time() < deadline and not list(root.rglob(train_v4.RUN_STATE_NAME)):
            if process.poll() is not None:
                pytest.fail("the trainer exited before writing a run state")
            time.sleep(2)
        assert list(root.rglob(train_v4.RUN_STATE_NAME)), "no epoch boundary reached"
        group = os.getpgid(process.pid)
        os.killpg(group, signal.SIGTERM)
        process.wait(timeout=120)
    finally:
        if process.poll() is None:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            process.wait(timeout=30)
    # No process from that group survives.
    survivors = subprocess.run(["pgrep", "-g", str(group)], capture_output=True,
                               text=True).stdout.split()
    assert not survivors, f"orphaned processes {survivors}"
    assert list(root.rglob(train_v4.RUN_STATE_NAME)), "last.pt was not preserved"
    assert not list(root.rglob(train_v4.COMPLETE_MARKER))


# --- the gates ------------------------------------------------------------------


def test_the_audit_refuses_an_incomplete_tree(tmp_path):
    from scripts.audit_v4_development_base import audit

    result = audit(str(tmp_path))
    assert result["complete"] == 0
    assert len(result["findings"]) == 12
    assert all("not COMPLETE" in finding for finding in result["findings"])


def test_the_collector_refuses_before_the_evaluations_exist(tmp_path):
    from scripts.collect_v4_development_base import collect

    with pytest.raises(SystemExit, match="audit has findings"):
        collect(str(tmp_path))


def test_the_evaluator_reads_validation_only():
    source = (ROOT / "scripts/v4_development_eval.py").read_text("utf-8")
    assert 'phase != "validation"' in source or "validation_context.phase" in source
    assert '"test"' not in source
    from experiments.v4_validation_collector import CollectorRefused, \
        ValidationCellCollector
    from lib.runtime_context import build_runtime_context

    with pytest.raises(CollectorRefused, match="validation only"):
        ValidationCellCollector(build_runtime_context("PEMS08", "test"),
                                {"mean": 0.0, "std": 1.0})


def test_the_plan_records_the_infrastructure_but_not_the_scheduling(plan):
    infrastructure = plan["infrastructure"]
    assert infrastructure["run_state_file"] == "last.pt"
    assert infrastructure["launcher_version"] == "v2_parallel_tmux"
    assert infrastructure["scheduling_is_not_part_of_run_identity"] is True
    assert plan["training"]["shuffle_policy"] == "lcbt_v4_shuffle_v1"
    # JOBS and the device assignment are scheduling, and must not be in the plan.
    serialised = json.dumps(plan)
    assert "JOBS" not in serialised and "cuda:" not in serialised
    for source in ("scripts/v4_launcher.py", "scripts/audit_v4_development_base.py",
                   "scripts/v4_development_eval.py",
                   "scripts/collect_v4_development_base.py"):
        assert source in plan["analysis_code"], source


def test_the_tmux_entry_point_dry_runs_without_tmux():
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/tmux_v4_development_base.sh"), "start",
         "--dry-run"],
        capture_output=True, text=True, cwd=str(ROOT),
        env={**os.environ, "PYTHON": sys.executable, "JOBS": "3",
             "DEVICES": "0,1,2"})
    assert result.returncode == 0, result.stderr
    assert "v4-dev-base" in result.stdout
    assert "console.log" in result.stdout
    assert "nothing started" in result.stdout
