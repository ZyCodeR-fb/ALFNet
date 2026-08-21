"""The plan the launcher loaded must be the plan the trainer runs under.

The bug this file exists for: `build_command` appended `--development_plan`
unconditionally. The five-dataset launcher enumerated METR-LA, PEMS03 and PEMS04
correctly and then handed each run to the two-dataset development plan, which
refused the config before CUDA. Nothing downstream noticed, because the launcher
was right and the trainer was right -- only the command between them was wrong.

So these tests take the real command the real launcher builds and feed it to the
real `train_v4.main()`. A helper-level check would have passed throughout the
period the queue was broken.
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import build_v4_five_dataset_plan as five
from scripts import train_v4, v4_launcher

FIVE_PLAN = "configs/paper/v4_five_dataset_plan.json"
DEVELOPMENT_PLAN = "configs/paper/v4_development_plan.json"


@pytest.fixture(scope="module")
def five_plan():
    return five.assert_plan_still_holds(five.load_plan())


def _command_for(plan, plan_path, name, output_root):
    launcher = v4_launcher.Launcher(output_root, plan, ["cpu"],
                                    plan_path=plan_path)
    run = next(r for r in launcher.runs if r.name == name)
    run.device = "cpu"
    return run, v4_launcher.build_command(run, output_root, launcher.plan_path)


# --- the command the launcher actually builds -----------------------------------


def test_the_five_dataset_command_carries_the_five_dataset_plan(five_plan,
                                                                tmp_path):
    _, command = _command_for(five_plan, FIVE_PLAN, "METRLA/ALFNet/seed32",
                              tmp_path)
    assert "--plan" in command
    assert command[command.index("--plan") + 1] == FIVE_PLAN
    assert "--development_plan" not in command
    assert "configs/paper/v4/METRLA_ALFNet.yaml" in command


def test_the_development_command_still_uses_the_development_entry_point(tmp_path):
    from scripts.build_v4_development_plan import assert_plan_still_holds, load_plan

    plan = assert_plan_still_holds(load_plan())
    _, command = _command_for(plan, None, "PEMS08/ALFNet/seed32", tmp_path)
    assert "--development_plan" in command
    assert "--plan" not in command


@pytest.mark.parametrize("name", ["METRLA/ALFNet/seed32", "PEMS03/MoGERNN/seed52",
                                  "PEMS04/ALFNet/seed42"])
def test_every_new_training_command_names_the_plan(five_plan, tmp_path, name):
    _, command = _command_for(five_plan, FIVE_PLAN, name, tmp_path)
    assert command[command.index("--plan") + 1] == FIVE_PLAN


def test_the_failure_record_and_a_resume_name_the_same_plan(five_plan, tmp_path):
    """A FAILED.json or a resume that named another plan would restart the bug."""
    run, base = _command_for(five_plan, FIVE_PLAN, "METRLA/ALFNet/seed32",
                             tmp_path)
    resume = v4_launcher.build_command(run, tmp_path, FIVE_PLAN, ["--resume"])
    assert resume[:len(base)] == base and resume[-1] == "--resume"

    v4_launcher.write_failure(run, 1, base)
    record = json.loads((run.directory / train_v4.FAILED_MARKER).read_text("utf-8"))
    assert f"--plan {FIVE_PLAN}" in record["command"]
    assert "--development_plan" not in record["command"]


# --- the real entry point, driven by that command -------------------------------


def test_train_v4_main_accepts_metrla_under_the_five_dataset_plan(tmp_path,
                                                                  five_plan):
    """The end-to-end path that was broken: launcher command -> main() -> run.

    Capped to one batch on the CPU, but it goes through argument parsing, plan
    resolution, readiness, config membership, the session and a real optimiser
    step, and it writes a checkpoint whose recorded plan must be this one.
    """
    _, command = _command_for(five_plan, FIVE_PLAN, "METRLA/ALFNet/seed32",
                              tmp_path)
    argv = command[2:]                       # drop the interpreter and script
    argv[argv.index("--output_root") + 1] = str(tmp_path)
    argv += ["--max_train_batches", "1", "--max_validation_batches", "1"]
    assert train_v4.main(argv) == 0

    checkpoint_paths = list(Path(tmp_path).rglob("best.pt"))
    assert checkpoint_paths, "no checkpoint was written"
    import torch

    checkpoint = torch.load(checkpoint_paths[0], weights_only=False,
                            map_location="cpu")
    assert checkpoint["plan_sha256"] == five_plan["plan_sha256"]
    assert checkpoint["dataset"] == "METRLA"
    assert checkpoint["parameter_count"] == 1_037_529
    state = list(Path(tmp_path).rglob(train_v4.RUN_STATE_NAME))
    assert state, "no run state was written"
    assert torch.load(state[0], weights_only=False,
                      map_location="cpu")["plan_sha256"] == five_plan["plan_sha256"]


def test_the_development_entry_point_still_accepts_pems08(tmp_path):
    from scripts.build_v4_development_plan import assert_plan_still_holds, load_plan

    plan = assert_plan_still_holds(load_plan())
    report = train_v4.train(ROOT / "configs/paper/v4/PEMS08_MoGERNN.yaml",
                            output_root=tmp_path, device="cpu", seed=32,
                            max_epochs=1, max_train_batches=1,
                            max_validation_batches=1, batch_size=4,
                            num_workers=0, plan=plan)
    assert report["plan_sha256"] == plan["plan_sha256"]


# --- fail closed, before CUDA ---------------------------------------------------


def test_a_config_outside_the_plan_is_refused(tmp_path, five_plan):
    """METR-LA under the development plan: the exact failure, now asserted."""
    from scripts.build_v4_development_plan import assert_plan_still_holds, load_plan

    development = assert_plan_still_holds(load_plan())
    with pytest.raises(train_v4.RunnerRefused, match="is not one of the .* configs"):
        train_v4.resolve_plan(ROOT / "configs/paper/v4/METRLA_ALFNet.yaml",
                              development)
    # ... and it names which plan refused, rather than always saying development.
    plan, entry = train_v4.resolve_plan(
        ROOT / "configs/paper/v4/METRLA_ALFNet.yaml", five_plan)
    assert plan["plan_sha256"] == five_plan["plan_sha256"]
    assert entry["path"] == "configs/paper/v4/METRLA_ALFNet.yaml"


def test_the_two_plan_flags_are_mutually_exclusive():
    with pytest.raises(SystemExit):
        train_v4.main(["--config", "configs/paper/v4/PEMS08_ALFNet.yaml",
                       "--development_plan", "--plan", FIVE_PLAN])


def test_an_unknown_plan_kind_is_refused(tmp_path):
    path = tmp_path / "other.json"
    path.write_text(json.dumps({"kind": "something_else", "plan_sha256": "x"}),
                    "utf-8")
    with pytest.raises(train_v4.RunnerRefused, match="declares kind"):
        train_v4.load_plan_authority(path)
    with pytest.raises(train_v4.RunnerRefused, match="does not exist"):
        train_v4.load_plan_authority(tmp_path / "absent.json")


def test_the_five_dataset_plan_requires_readiness(tmp_path, monkeypatch):
    """No permit, no run -- and the refusal precedes any session or CUDA."""
    from scripts import build_v4_readiness as readiness

    record = json.loads(readiness.ATTESTATION_PATH.read_text("utf-8"))
    record["readiness"]["training_ready"] = False
    record["attestation_sha256"] = readiness.attestation_hash(record)
    path = tmp_path / "attestation.json"
    path.write_text(json.dumps(record), "utf-8")
    monkeypatch.setattr(readiness, "ATTESTATION_PATH", path)

    touched = []
    monkeypatch.setattr(train_v4, "build_v4_session",
                        lambda *a, **k: touched.append("session"))
    with pytest.raises(train_v4.RunnerRefused, match="training_ready"):
        train_v4.load_plan_authority(ROOT / FIVE_PLAN)
    assert touched == [], "a session was built before the readiness gate"


def test_no_plan_flag_means_no_plan_is_silently_loaded():
    assert train_v4.load_plan_authority(None)["plan_sha256"]
    # ... but `train()` without a plan neither resolves nor records one.
    assert "plan_sha256" not in train_v4.train.__doc__
