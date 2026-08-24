#!/usr/bin/env python
"""Re-score every result slot in a plan on validation, at its own seed.

Plan-driven throughout: the datasets, models, seeds and directories come from
`--plan`, so a plan that names five datasets is scored on five. Nothing here
knows that the development plan had twelve runs or that this one has thirty.

It handles both kinds of slot without distinguishing them in the result: a
`train` slot is scored from the directory this plan wrote, a `reuse` slot from
the directory the source plan wrote. What is checked is identical either way.

The seed is passed explicitly. Its absence is what made the first evaluator
score seeds 42 and 52 against seed 32's node split, so it is asserted after the
session is built rather than trusted to the call site.

The scaler comes from the checkpoint. Refitting one on validation would make the
number depend on the split being scored.
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.v4_runner import build_v4_session  # noqa: E402
from experiments.v4_validation_collector import (  # noqa: E402
    METRICS,
    cell_count,
    collect_validation_table,
)
from scripts.v4_utils import sha256_file  # noqa: E402
from scripts.train_v4 import COMPLETE_MARKER  # noqa: E402

EVAL_SCHEMA = 1
EVAL_NAME = "plan_validation_eval.json"
PARTIAL_NAME = "plan_validation_eval_partial.json"
TOLERANCE = 1e-6

IDENTITY_FIELDS = ("dataset", "model", "model_variant", "seed", "config_sha256",
                   "payload_sha256", "unseen_sha256", "manifest_sha256",
                   "static_asset_hash", "train_exposure_hash",
                   "validation_exposure_hash")


class PlanEvalRefused(ValueError):
    """This slot will not be scored in this session."""


def load_plan(path):
    from scripts import build_v4_five_dataset_plan as five

    return five.assert_plan_still_holds(five.load_plan(path))


def resolve_runs(plan, output_root=None):
    from scripts.v4_launcher import plan_runs

    root = Path(output_root or ROOT / plan["output_root"] / plan["plan_sha256"])
    return plan_runs(root, plan), root


def expected_identity(run, session, plan):
    entry = plan["configs"][f"{run.dataset}_{run.model}"]
    data = plan["data"][run.dataset]
    return {
        "dataset": run.dataset, "model": run.model,
        "model_variant": getattr(session.args, "model_variant", None),
        "seed": run.seed,
        "config_sha256": entry["sha256"],
        "payload_sha256": data["payload_sha256"],
        "unseen_sha256": data["unseen_sha256"],
        "manifest_sha256": data["manifest_sha256"],
        "static_asset_hash": session.train_context.static_asset_hash(),
        "train_exposure_hash": session.train_context.exposure_hash(),
        "validation_exposure_hash": session.validation_context.exposure_hash(),
    }


def assert_identity(checkpoint, expected, where):
    """Present on both sides and equal. Absence is a refusal, not a match."""
    problems = []
    for field in IDENTITY_FIELDS:
        want, got = expected.get(field, KeyError), checkpoint.get(field, KeyError)
        if want is KeyError:
            problems.append(f"{field}: nothing to compare against")
        elif got is KeyError:
            problems.append(f"{field}: absent from the checkpoint")
        elif want is None or got is None:
            problems.append(f"{field}: None on one side ({want!r} vs {got!r})")
        elif want != got:
            problems.append(f"{field}: expected {want!r}, checkpoint has {got!r}")
    if problems:
        raise PlanEvalRefused(f"{where} identity mismatch:\n  - "
                              + "\n  - ".join(problems))


def atomic_write_json(path, payload):
    path = Path(path)
    text = json.dumps(payload, indent=1, sort_keys=True, allow_nan=False) + "\n"
    if path.exists():
        if path.read_text("utf-8") == text:
            return path
        raise PlanEvalRefused(
            f"{path} exists with different content; refusing to overwrite an "
            "earlier evaluation")
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=str(path.parent), suffix=".partial")
    os.close(handle)
    try:
        Path(temporary).write_text(text, "utf-8")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return path


def evaluate_run(run, plan, device="cuda:0", max_batches=None):
    directory = run.directory
    if not (directory / COMPLETE_MARKER).exists():
        raise PlanEvalRefused(f"{run.name}: not COMPLETE at {directory}")
    checkpoint_path = directory / "best.pt"
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location="cpu")

    session = build_v4_session(run.config, device=device, run_seed=run.seed)
    if session.run_seed != run.seed:
        raise PlanEvalRefused(
            f"{run.name}: session seed {session.run_seed} is not {run.seed}")
    if session.validation_context.phase != "validation":
        raise PlanEvalRefused(f"{run.name}: not a validation context")
    assert_identity(checkpoint, expected_identity(run, session, plan), run.name)

    # The checkpoint's own scaler, never a fresh fit.
    stored_scaler = checkpoint.get("scaler") or {}
    if stored_scaler.get("fitted_on") != "train window, training-known nodes only":
        raise PlanEvalRefused(
            f"{run.name}: checkpoint scaler was fitted on "
            f"{stored_scaler.get('fitted_on')!r}")
    for key in ("mean", "std"):
        if abs(float(stored_scaler[key]) - float(session.scaler[key])) > 0:
            raise PlanEvalRefused(
                f"{run.name}: the session's scaler {key} does not match the "
                "checkpoint's; the evaluation would use a refitted scaler")

    session.model.load_state_dict(checkpoint["model_state"])
    session.model.to(device)
    table = collect_validation_table(session, epoch=int(checkpoint["best_epoch"]),
                                     max_batches=max_batches)
    if cell_count(table) != 33:
        raise PlanEvalRefused(f"{run.name}: {cell_count(table)} cells, expected 33")

    stored_path = directory / "development_validation_table.json"
    differences, worst = [], 0.0
    if stored_path.exists():
        stored = json.loads(stored_path.read_text("utf-8"))
        for section in ("main", "phase"):
            for row in stored[section]:
                for metric in METRICS:
                    a = float(stored[section][row][metric])
                    b = float(table[section][row][metric])
                    worst = max(worst, abs(a - b))
                    if abs(a - b) > TOLERANCE:
                        differences.append(f"{section}/{row}/{metric}: "
                                           f"training {a!r} recomputed {b!r}")
    partial = max_batches is not None
    record = {
        "schema_version": EVAL_SCHEMA,
        "plan_validation_only": True,
        "split": "validation",
        "partial_diagnostic_only": partial,
        "plan_sha256": plan["plan_sha256"],
        "run": run.name, "dataset": run.dataset, "model": run.model,
        "seed": run.seed, "run_seed_used": session.run_seed,
        "execution_action": run.execution_action,
        "source_plan_sha256": run.entry.get("source_plan_sha256"),
        "run_directory": str(directory.relative_to(ROOT)),
        "device": device,
        "best_epoch": int(checkpoint["best_epoch"]),
        "scaler": {"mean": float(stored_scaler["mean"]),
                   "std": float(stored_scaler["std"]),
                   "source": "checkpoint", "refit": False},
        "max_abs_cell_difference": worst,
        "tolerance": TOLERANCE,
        "compared_against_training_table": stored_path.exists(),
        "agrees": (not differences) and not partial,
        "differences": differences,
        "identity": expected_identity(run, session, plan),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "evaluator_sha256": sha256_file(Path(__file__)),
        "recomputed": table,
    }
    atomic_write_json(directory / (PARTIAL_NAME if partial else EVAL_NAME), record)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--run", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--devices", default=None,
                        help="accepted for symmetry; this process uses --device")
    parser.add_argument("--jobs", type=int, default=1,
                        help="accepted for symmetry; run several processes to "
                             "parallelise")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--max-batches", type=int, default=None,
                        help="diagnostic only; writes a separate partial file")
    arguments = parser.parse_args(argv)

    plan = load_plan(arguments.plan)
    runs, _ = resolve_runs(plan, arguments.output_root)
    if arguments.run:
        runs = [run for run in runs if run.name == arguments.run]
        if not runs:
            raise SystemExit(f"no run named {arguments.run}")

    failures = 0
    for run in runs:
        record = evaluate_run(run, plan, arguments.device, arguments.max_batches)
        status = ("PARTIAL" if record["partial_diagnostic_only"]
                  else ("AGREES" if record["agrees"] else "DIFFERS"))
        print(f"{status:8s} {record['execution_action']:6s} {run.name:26s} "
              f"seed={record['run_seed_used']:3d} "
              f"max|diff|={record['max_abs_cell_difference']:.3e}", flush=True)
        failures += 0 if record["agrees"] else 1
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
