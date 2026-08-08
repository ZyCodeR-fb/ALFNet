"""The frozen 18-run manifest must keep matching the configs it names.

`configs/paper/final_runs.json` is the single list that training, the audit, the
evaluation and the aggregation all read. That removes the risk of four
hand-maintained matrices disagreeing, and replaces it with one: the manifest
drifting away from the YAML files it points at. A stale `canonical_config_hash`
there would reject every finished run as "not this experiment", or -- worse, if
someone widened the check to compensate -- accept the wrong one.

So the hashes are recomputed here from the configs themselves.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from args import apply_output_root, initialize_args  # noqa: E402
from scripts.final_runs import load_manifest  # noqa: E402

MANIFEST = load_manifest()
RUNS = MANIFEST["runs"]


def _freeze(config, seed, batch_size):
    argv = sys.argv
    try:
        sys.argv = [
            "main.py", "-c", config,
            "--batch_size", str(batch_size),
            "--seed", str(seed),
            "--output_root", "./outputs",
            "--run_id", "manifest-check",
        ]
        args = initialize_args()
        apply_output_root(args)
        return args
    finally:
        sys.argv = argv


def test_the_matrix_is_two_models_three_datasets_three_seeds():
    assert len(RUNS) == 18
    assert sorted(MANIFEST["seeds"]) == [32, 42, 52]
    assert sorted(MANIFEST["models"]) == ["ALFNet", "MoGERNN"]
    assert sorted(MANIFEST["datasets"]) == ["METRLA", "PEMS08", "PEMSBAY"]


def test_every_run_id_and_hash_is_unique():
    ids = [run["run_id"] for run in RUNS]
    assert len(set(ids)) == len(ids)
    digests = [run["canonical_config_hash"] for run in RUNS]
    assert len(set(digests)) == len(digests), (
        "two runs share a canonical config hash -- they are the same experiment"
    )


def test_each_cell_has_all_three_seeds():
    cells = {}
    for run in RUNS:
        cells.setdefault((run["model"], run["dataset"]), set()).add(run["seed"])
    assert len(cells) == 6
    for cell, seeds in cells.items():
        assert seeds == {32, 42, 52}, f"{cell} has seeds {sorted(seeds)}"


def test_every_config_named_by_the_manifest_exists():
    for run in RUNS:
        assert (ROOT / run["config"]).exists(), f"{run['run_id']}: {run['config']}"


@pytest.mark.parametrize("run", RUNS, ids=[run["run_id"] for run in RUNS])
def test_frozen_hashes_still_match_the_configs(run):
    """The check that stops the manifest silently going stale."""
    args = _freeze(run["config"], run["seed"], run["batch_size"])
    assert args.canonical_config_hash == run["canonical_config_hash"], (
        f"{run['run_id']}: {run['config']} no longer hashes to the frozen value. "
        "Either the config changed (which invalidates results produced under it) "
        "or the hash definition moved."
    )
    assert args.legacy_canonical_config_hash == run["legacy_canonical_config_hash"]


@pytest.mark.parametrize("run", RUNS, ids=[run["run_id"] for run in RUNS])
def test_manifest_identity_matches_the_config(run):
    args = _freeze(run["config"], run["seed"], run["batch_size"])
    assert args.model == run["model"]
    assert args.model_variant == run["model_variant"]
    assert args.dloader_name == run["dataset"]
    assert args.seed == run["seed"]


def test_the_manifest_is_the_only_matrix_the_scripts_carry():
    """No script may hard-code a run list of its own."""
    for name in ("run_final_18.sh", "run_final_eval_18.sh",
                 "audit_final_18.py", "collect_final_18.py"):
        text = (ROOT / "scripts" / name).read_text(encoding="utf-8")
        for run in RUNS:
            # A run_id may appear in a usage example, but a script must not
            # enumerate the matrix; the plan always comes from final_runs.py.
            assert text.count(run["run_id"]) <= 1, (
                f"scripts/{name} mentions {run['run_id']} more than once -- "
                "it may be carrying its own copy of the matrix"
            )
        assert "final_runs" in text or "final_runs.py" in text, (
            f"scripts/{name} does not read the shared manifest"
        )


# --- the launcher ------------------------------------------------------------
#
# scripts/run_final_18.sh is the only piece of the frozen pipeline that is
# shell rather than Python, and the two things below are exactly the two that
# a reader cannot check by eye: a background subshell's exit status, and the
# default concurrency. Both were wrong at once, and the combination is the bad
# one -- a silent failure on a machine running three jobs on one GPU.

LAUNCHER = ROOT / "scripts" / "run_final_18.sh"

requires_bash = pytest.mark.skipif(
    shutil.which("bash") is None,
    reason="scripts/run_final_18.sh is the Linux server's launcher",
)


def _run_launcher(*argv, env_overrides=None, cwd=ROOT):
    env = dict(os.environ)
    env.pop("JOBS", None)
    env.pop("MAX_EPOCH", None)
    env.pop("SUFFIX", None)
    env.update(env_overrides or {})
    return subprocess.run(
        ["bash", str(LAUNCHER), *argv],
        cwd=str(cwd), env=env, capture_output=True, text=True, timeout=300,
    )


@requires_bash
def test_the_launcher_is_serial_unless_told_otherwise(tmp_path):
    """Concurrency is a property of the machine, not of the experiment.

    It defaults to 1 so that starting the frozen matrix on an unfamiliar box
    cannot silently put three jobs on one GPU before anyone has looked at the
    memory headroom.
    """
    result = _run_launcher(
        "--dry-run", env_overrides={"OUTPUT_ROOT": str(tmp_path)})
    assert result.returncode == 0, result.stderr
    assert "parallel    : 1" in result.stdout


@requires_bash
def test_only_selects_a_seed_slice_without_a_seeds_flag(tmp_path):
    """`--only` is a substring match on run_id, so it already slices by seed.

    The six-cell smoke is `--only s32`. This is asserted rather than adding a
    --seeds flag: a second way to filter is a second thing that can disagree
    with the manifest.
    """
    result = _run_launcher(
        "--dry-run", "--only", "s32",
        env_overrides={"OUTPUT_ROOT": str(tmp_path)})
    assert result.returncode == 0, result.stderr
    assert "planned 6 |" in result.stdout

    expected = {run["run_id"] for run in RUNS if run["seed"] == 32}
    assert len(expected) == 6
    for run_id in expected:
        assert f"-- run    {run_id}" in result.stdout
    for run in RUNS:
        if run["seed"] != 32:
            assert f"-- run    {run['run_id']}" not in result.stdout


@requires_bash
def test_a_failing_run_is_reported_as_failed_not_done(tmp_path):
    """The bug this pins: the job subshell used to end with `echo "$?"`.

    A subshell exits with the status of its last command, so the status was
    the `echo`'s -- 0, whatever the training run did. `wait` in reap_finished
    therefore printed "== done" for a run that had crashed. The status file
    still held the true code, so the final summary was right and the live
    output was wrong for the whole matrix, which is the worst combination:
    the operator watching the log sees success.

    So this asserts on both channels: the live line and the status file.
    """
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_python = fake_bin / "fakepython"
    # Fails main.py with a distinctive code, forwards everything else (the
    # plan comes from scripts/final_runs.py through the same PYTHON).
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        'if [[ "${1:-}" == *main.py ]]; then\n'
        '  echo "pretending to train, then failing" >&2\n'
        "  exit 37\n"
        "fi\n"
        f'exec "{sys.executable}" "$@"\n',
        encoding="utf-8",
    )
    fake_python.chmod(0o755)

    output_root = tmp_path / "outputs"
    result = _run_launcher(
        "--only", "alfnet-metrla-s32",
        env_overrides={"OUTPUT_ROOT": str(output_root),
                       "PYTHON": str(fake_python)},
    )

    assert result.returncode == 1, (
        "a failed training run must fail the launcher\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    combined = result.stdout + result.stderr
    assert "!! FAILED alfnet-metrla-s32 (exit 37)" in combined, (
        "the live reap line must carry the real exit code\n" + combined)
    assert "== done   alfnet-metrla-s32" not in combined, (
        "a crashed run was announced as done\n" + combined)
    assert "alfnet-metrla-s32 (exit 37)" in result.stdout, (
        "the final summary must list it under FAILED\n" + result.stdout)

    status_file = output_root / "_launcher" / "train" / "alfnet-metrla-s32.status"
    assert status_file.exists()
    assert status_file.read_text(encoding="utf-8").strip() == "37"


@requires_bash
@pytest.mark.parametrize(
    "verdict,expect_exit,expect_marker",
    [
        (0, 0, "-- skip   alfnet-metrla-s32: verified complete"),
        (1, 1, "!! REFUSE alfnet-metrla-s32"),
        (2, 2, "!! ABORT: the verifier could not run (exit 2)"),
    ],
    ids=["match", "mismatch", "unverifiable"],
)
def test_the_launcher_separates_mismatch_from_unverifiable(
        tmp_path, verdict, expect_exit, expect_marker):
    """Three verdicts, three behaviours -- and never an overwrite.

    Collapsing "this run is wrong" into "I could not check" is what makes a
    broken environment look like a corrupt matrix, so exit 2 aborts the whole
    script rather than refusing run after run. The shared assertion is the one
    that protects finished work: whatever the verdict, nothing is retrained on
    top of an existing directory.
    """
    output_root = tmp_path / "outputs"
    summary = (output_root / "ALFNet" / "alfnet-metrla-s32"
               / "checkpoints" / "training_summary.json")
    summary.parent.mkdir(parents=True)
    summary.write_text("{}", encoding="utf-8")

    fake_python = tmp_path / "fakepython"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        f'if [[ "${{1:-}}" == *audit_final_18.py ]]; then exit {verdict}; fi\n'
        'if [[ "${1:-}" == *main.py ]]; then\n'
        '  echo "TRAINING WAS LAUNCHED" >&2\n'
        "  exit 0\n"
        "fi\n"
        f'exec "{sys.executable}" "$@"\n',
        encoding="utf-8",
    )
    fake_python.chmod(0o755)

    result = _run_launcher(
        "--only", "alfnet-metrla-s32",
        env_overrides={"OUTPUT_ROOT": str(output_root),
                       "PYTHON": str(fake_python)},
    )
    combined = result.stdout + result.stderr
    assert result.returncode == expect_exit, combined
    assert expect_marker in combined, combined
    assert "TRAINING WAS LAUNCHED" not in combined, (
        "an existing run directory was trained over\n" + combined)
