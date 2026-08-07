"""The frozen 18-run list, and the one definition of "this run is reusable".

`configs/paper/final_runs.json` is the single matrix. Training, evaluation and
aggregation all read it from here rather than each carrying their own copy,
because three hand-maintained matrices is three chances to disagree about which
runs the paper is made of.

Reusability is defined once, in `verify_run`. The failure it exists to stop:
"`training_summary.json` exists, therefore skip" treats any leftover directory
as a finished run of *this* experiment. A run from a different seed, dataset,
protocol or model writes exactly the same filename, so the check has to compare
identity, not presence.

`--emit-plan` prints the matrix as TSV for the bash launchers.
"""

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MANIFEST = ROOT / "configs" / "paper" / "final_runs.json"

# A run that stopped for one of these reasons ran to completion. Anything else
# (or a missing value) means it was interrupted and must not be reused.
FINISHED_STOP_REASONS = {"early_stop", "max_epoch", "max_epoch_reached", "completed"}


def load_manifest(path=MANIFEST):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema") != 1:
        raise SystemExit(f"{path}: unsupported manifest schema {payload.get('schema')}")
    expected = len(payload["models"]) * len(payload["datasets"]) * len(payload["seeds"])
    if len(payload["runs"]) != expected:
        raise SystemExit(
            f"{path}: declares {len(payload['runs'])} runs but the matrix implies {expected}"
        )
    ids = [run["run_id"] for run in payload["runs"]]
    if len(set(ids)) != len(ids):
        raise SystemExit(f"{path}: duplicate run_id in the matrix")
    return payload


def run_dir(run, output_root):
    return Path(output_root) / run["output_subdir"] / run["run_id"]


def _finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def _structural_shim(names, config_path):
    """A bare module tree carrying one structural encoder per recorded name.

    Built so the *real* validator can run without constructing the full model,
    which for ALFNet costs seconds per run. `out_dim` is irrelevant here: the
    raw digest covers the adjacency and coordinates, never the MLP.
    """
    import yaml
    from torch import nn

    from lib.utils import load_project_adjacency, load_project_positions
    from layers.structural_node_encoder import StructuralNodeEncoder

    # The data path is read straight from the YAML rather than through
    # `args.load_args_from_config`, which pulls in the full training argument
    # stack (omegaconf, lightning) for one string. An audit must stay light
    # enough to run wherever a checkpoint is.
    config = yaml.safe_load(Path(ROOT / config_path).read_text(encoding="utf-8"))
    data_path = str((config.get("dataset") or {})["data_path"])
    adjacency = load_project_adjacency({"data_path": data_path})
    positions = load_project_positions({"data_path": data_path})
    root = nn.Module()
    for name in names:
        parts = name.split(".")
        node = root
        for part in parts[:-1]:
            if not hasattr(node, part):
                setattr(node, part, nn.Module())
            node = getattr(node, part)
        setattr(node, parts[-1],
                StructuralNodeEncoder(adjacency, positions, out_dim=8))
    return root


def verify_run(run, output_root, check_structure=True):
    """[] if this directory really is this manifest entry, else the reasons.

    Checked: the run finished; the checkpoint records the same model, variant
    and canonical config hash (which covers seed, dataset, config and every
    hyper-parameter, all of which enter it); the protocol hash is present and
    agrees between the summary and the checkpoint; the structural provenance
    passes the same validator the loader uses; and the validation metrics are
    finite.
    """
    import torch

    problems = []
    directory = run_dir(run, output_root)
    summary_path = directory / "checkpoints" / "training_summary.json"
    checkpoint_path = directory / "checkpoints" / "best_val_checkpoint.pth"

    if not summary_path.exists():
        return ["training_summary.json missing (run never finished)"]
    if not checkpoint_path.exists():
        return ["best_val_checkpoint.pth missing"]

    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        return [f"training_summary.json unreadable: {exc}"]

    stop_reason = summary.get("stop_reason")
    if stop_reason not in FINISHED_STOP_REASONS:
        problems.append(f"stop_reason={stop_reason!r} is not a completed run")

    try:
        blob = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception as exc:                                  # noqa: BLE001
        return [f"checkpoint unreadable: {exc}"]
    config = blob.get("config") or {}

    for field in ("model", "model_variant"):
        found, expected = config.get(field), run[field]
        if found != expected:
            problems.append(f"{field}: checkpoint={found!r} manifest={expected!r}")

    found = config.get("canonical_config_hash")
    if not found:
        problems.append("checkpoint records no canonical_config_hash")
    elif found not in (run["canonical_config_hash"], run["legacy_canonical_config_hash"]):
        # This is the check that catches a wrong seed or a wrong dataset: both
        # enter the canonical hash, so a mismatch here means the directory holds
        # a different experiment wearing the right name.
        problems.append(
            f"canonical_config_hash: checkpoint={str(found)[:16]}... "
            f"manifest={run['canonical_config_hash'][:16]}... "
            "(seed, dataset, config and hyper-parameters all enter this hash)"
        )

    protocol = summary.get("training_protocol_hash")
    checkpoint_protocol = blob.get("training_protocol_hash") or config.get(
        "training_protocol_hash"
    )
    if not protocol:
        problems.append("training_summary.json records no training_protocol_hash")
    elif checkpoint_protocol and checkpoint_protocol != protocol:
        problems.append(
            "training_protocol_hash disagrees between summary and checkpoint"
        )

    metrics = summary.get("best_val_lifecycle_metrics") or {}
    if not metrics:
        problems.append("no best_val_lifecycle_metrics")
    else:
        bad = sorted(k for k, v in metrics.items() if not _finite(v))
        if bad:
            problems.append(f"non-finite validation metrics: {bad[:6]}")
    if not _finite(summary.get("best_val_metric")):
        problems.append("best_val_metric is missing or non-finite")

    if check_structure:
        recorded = config.get("structural_fingerprints") or {}
        if recorded:
            # The validator lives in layers/, not main, precisely so this needs
            # no training-entry-point dependencies. Import errors are raised, not
            # folded into `problems`: "the verifier would not load" is not
            # evidence about the graph, and reporting it as a provenance failure
            # rejects perfectly good runs.
            from layers.structural_node_encoder import (
                validate_structural_fingerprints,
            )

            shim = _structural_shim(sorted(recorded), run["config"])
            try:
                validate_structural_fingerprints(blob, shim, log=None)
            except ValueError as exc:
                problems.append(f"structural provenance: {exc}")

    return problems


# The four tables a completed `--mode test_async` must leave behind, plus the
# manifest copy that records which protocol produced them.
EVAL_ARTEFACTS = (
    "lce_main_summary.json",
    "lce_phase_summary.json",
    "lce_trajectory_summary.json",
    "ples_effects_summary.json",
    "protocol_manifest.json",
)


def eval_result_dirs(run, output_root):
    """Every `predictions/lifecycle-v3-*` directory this run has produced."""
    predictions = run_dir(run, output_root) / "predictions"
    if not predictions.exists():
        return []
    return sorted(p for p in predictions.glob("lifecycle-v3-*") if p.is_dir())


def verify_eval(run, output_root):
    """([] , dir) if this run has exactly one complete, consistent result set.

    More than one result directory means the run was evaluated under two
    different LCE protocols and there is no way to know which the paper should
    quote, so it is refused rather than silently resolved.
    """
    directories = eval_result_dirs(run, output_root)
    if not directories:
        return ["no predictions/lifecycle-v3-* directory (never evaluated)"], None
    if len(directories) > 1:
        return ([
            "more than one result directory: "
            f"{[d.name for d in directories]} -- evaluated under two protocols"
        ], None)

    directory = directories[0]
    problems = []
    for name in EVAL_ARTEFACTS:
        if not (directory / name).exists():
            problems.append(f"missing {name}")
    if problems:
        return problems, directory

    try:
        manifest = json.loads(
            (directory / "protocol_manifest.json").read_text(encoding="utf-8")
        )
    except (ValueError, OSError) as exc:
        return [f"protocol_manifest.json unreadable: {exc}"], directory

    summary_path = run_dir(run, output_root) / "checkpoints" / "training_summary.json"
    if summary_path.exists():
        try:
            trained = json.loads(summary_path.read_text(encoding="utf-8")).get(
                "training_protocol_hash"
            )
        except (ValueError, OSError):
            trained = None
        found = manifest.get("training_protocol_hash")
        if trained and found and trained != found:
            problems.append(
                "result was produced under a different training_protocol_hash "
                f"({str(found)[:12]}...) than the checkpoint was trained on "
                f"({str(trained)[:12]}...)"
            )
    return problems, directory


def emit_plan(payload):
    """TSV for the bash launchers: one line per run, tab separated."""
    for run in payload["runs"]:
        print("\t".join([
            run["run_id"], run["config"], str(run["seed"]),
            run["output_subdir"], run["model"], str(run["batch_size"]),
        ]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emit-plan", action="store_true")
    parser.add_argument("--manifest", default=str(MANIFEST))
    options = parser.parse_args()
    payload = load_manifest(options.manifest)
    if options.emit_plan:
        emit_plan(payload)
        return 0
    print(f"{payload['name']}: {len(payload['runs'])} runs, "
          f"seeds {payload['seeds']}, datasets {payload['datasets']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
