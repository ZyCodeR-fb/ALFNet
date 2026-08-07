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
