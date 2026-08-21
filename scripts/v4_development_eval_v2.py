#!/usr/bin/env python
"""Re-score each development run's `best.pt` on validation -- with its own seed.

**Why v2 exists.** `v4_development_eval.py` built its session as
`build_v4_session(run.config, device=device)`, and `run_seed` defaults to the
config's `seed: 32`. So every seed-42 and seed-52 run was re-scored against
seed 32's node split and validation schedule. The signature is exactly that: all
four seed-32 runs agreed at max difference 0, and all eight seed-42/52 runs
disagreed on all 36 cells. Those eight v1 files are wrong, not the checkpoints.

v1 is left on disk and its outputs are marked `invalid_wrong_seed` rather than
deleted, because the old plan recorded v1's code hash and rewriting either would
destroy the trail. This is a **post-registration analysis correction**: the
model, the data, the endpoint and the aggregation are untouched -- only the seed
the evaluator passes when rebuilding the context.

Identity is checked before any forward pass, and a field that is missing on
either side is a refusal rather than a match: `.get() == .get()` returns True
when both are absent, which is precisely the case where nothing was verified.
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
from scripts.build_v4_development_plan import (  # noqa: E402
    assert_plan_still_holds,
    load_plan,
    sha256_file,
)
from scripts.train_v4 import CODE_SOURCES, COMPLETE_MARKER  # noqa: E402
from scripts.v4_launcher import plan_runs  # noqa: E402

EVAL_SCHEMA = 2
EVAL_NAME = "development_validation_eval_v2.json"
PARTIAL_NAME = "development_validation_eval_v2_partial.json"
V1_NAME = "development_validation_eval.json"
# Recomputation runs the same operations on the same numbers, so this is a
# float-summation-order tolerance, not room for a different answer.
TOLERANCE = 1e-6

# Every field must be present on both sides and equal. Absence is a refusal.
IDENTITY_FIELDS = (
    "dataset", "model", "model_variant", "seed", "config_path", "config_sha256",
    "plan_sha256", "payload_sha256", "unseen_sha256", "manifest_sha256",
    "static_asset_hash", "train_exposure_hash", "validation_exposure_hash",
)


class EvalRefused(ValueError):
    """This checkpoint will not be scored in this session."""


def expected_identity(run, session, plan):
    registered = plan["configs"][f"{run.dataset}_{run.model}"]
    data = plan["data"][run.dataset]
    return {
        "dataset": run.dataset,
        "model": run.model,
        "model_variant": getattr(session.args, "model_variant", None),
        "seed": run.seed,
        "config_path": registered["path"],
        "config_sha256": registered["sha256"],
        "plan_sha256": plan["plan_sha256"],
        "payload_sha256": data["payload_sha256"],
        "unseen_sha256": data["unseen_sha256"],
        "manifest_sha256": data["manifest_sha256"],
        "static_asset_hash": session.train_context.static_asset_hash(),
        "train_exposure_hash": session.train_context.exposure_hash(),
        "validation_exposure_hash": session.validation_context.exposure_hash(),
    }


def assert_identity(checkpoint, expected, where):
    """Refuse unless every field is present on both sides and equal."""
    problems = []
    for field in IDENTITY_FIELDS:
        want = expected.get(field, KeyError)
        got = checkpoint.get(field, KeyError)
        if want is KeyError:
            problems.append(f"{field}: nothing to compare against")
        elif got is KeyError:
            problems.append(f"{field}: absent from the checkpoint")
        elif got is None or want is None:
            problems.append(f"{field}: None on one side ({want!r} vs {got!r})")
        elif got != want:
            problems.append(f"{field}: expected {want!r}, checkpoint has {got!r}")
    if problems:
        raise EvalRefused(f"{where} identity mismatch:\n  - " + "\n  - ".join(problems))


def atomic_write_json(path, payload):
    """Idempotent on identical content; never silently different content."""
    path = Path(path)
    text = json.dumps(payload, indent=1, sort_keys=True, allow_nan=False) + "\n"
    if path.exists():
        if path.read_text("utf-8") == text:
            return path
        raise EvalRefused(
            f"{path} already exists with different content; refusing to "
            "overwrite an earlier evaluation")
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
        raise EvalRefused(f"{run.name}: not COMPLETE; nothing to evaluate")
    checkpoint_path = directory / "best.pt"
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location="cpu")

    # The fix. Without `run_seed` the context is rebuilt at the config's seed 32,
    # so seeds 42 and 52 were scored against another run's node split.
    session = build_v4_session(run.config, device=device, run_seed=run.seed)
    if session.run_seed != run.seed:
        raise EvalRefused(
            f"{run.name}: session seed {session.run_seed} is not the run's "
            f"{run.seed}")
    if session.validation_context.phase != "validation":
        raise EvalRefused("the eval session did not resolve a validation context")
    assert_identity(checkpoint, expected_identity(run, session, plan), run.name)

    session.model.load_state_dict(checkpoint["model_state"])
    session.model.to(device)
    table = collect_validation_table(session, epoch=int(checkpoint["best_epoch"]),
                                     max_batches=max_batches)
    if cell_count(table) != 36:
        raise EvalRefused(f"{run.name}: recomputed {cell_count(table)} cells")

    stored = json.loads(
        (directory / "development_validation_table.json").read_text("utf-8"))
    differences, worst = [], 0.0
    for section in ("main", "phase"):
        for row in stored[section]:
            for metric in METRICS:
                a = float(stored[section][row][metric])
                b = float(table[section][row][metric])
                worst = max(worst, abs(a - b))
                if abs(a - b) > TOLERANCE:
                    differences.append(f"{section}/{row}/{metric}: "
                                       f"stored {a!r} recomputed {b!r}")
    partial = max_batches is not None
    record = {
        "schema_version": EVAL_SCHEMA,
        "development_validation_only": True,
        "partial_diagnostic_only": partial,
        "correction": "post-registration analysis correction: v1 passed no run "
                      "seed, so seeds 42 and 52 were scored on seed 32's split",
        "run": run.name,
        "dataset": run.dataset, "model": run.model, "seed": run.seed,
        "run_seed_used": session.run_seed,
        "device": device,
        "best_epoch": int(checkpoint["best_epoch"]),
        "max_abs_cell_difference": worst,
        "tolerance": TOLERANCE,
        "agrees": (not differences) and not partial,
        "differences": differences,
        "identity": expected_identity(run, session, plan),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "evaluator_sha256": sha256_file(Path(__file__)),
        "code_sha256": {name: sha256_file(ROOT / name) for name in CODE_SOURCES},
        "recomputed": table,
    }
    atomic_write_json(directory / (PARTIAL_NAME if partial else EVAL_NAME), record)
    return record


def mark_v1_invalid(runs):
    """Leave v1 on disk, but say what it is. Never delete it."""
    marked = []
    for run in runs:
        path = run.directory / V1_NAME
        if not path.exists():
            continue
        record = json.loads(path.read_text("utf-8"))
        if record.get("invalid_wrong_seed"):
            continue
        record["invalid_wrong_seed"] = True
        record["invalid_reason"] = (
            "produced by v4_development_eval.py, which rebuilt the context "
            "without the run's seed; seeds 42 and 52 were scored against seed "
            "32's node split and validation schedule. Superseded by "
            f"{EVAL_NAME}; kept for traceability and never aggregated.")
        path.write_text(json.dumps(record, indent=1, sort_keys=True,
                                   allow_nan=False) + "\n", "utf-8")
        marked.append(run.name)
    return marked


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--max-batches", type=int, default=None,
                        help="diagnostic only; writes a separate partial file "
                             "the collector refuses")
    arguments = parser.parse_args(argv)

    plan = assert_plan_still_holds(load_plan())
    root = Path(arguments.output_root or
                ROOT / "outputs_v4/development_base" / plan["plan_sha256"])
    runs = plan_runs(root, plan)
    if arguments.run:
        runs = [run for run in runs if run.name == arguments.run]
        if not runs:
            raise SystemExit(f"no run named {arguments.run}")
    else:
        marked = mark_v1_invalid(plan_runs(root, plan))
        if marked:
            print(f"marked {len(marked)} v1 evaluation(s) invalid_wrong_seed "
                  "(kept on disk)", flush=True)

    failures = 0
    for run in runs:
        record = evaluate_run(run, plan, arguments.device, arguments.max_batches)
        status = ("PARTIAL" if record["partial_diagnostic_only"]
                  else ("AGREES" if record["agrees"] else "DIFFERS"))
        print(f"{status:8s} {run.name:26s} seed={record['run_seed_used']:3d} "
              f"best_epoch={record['best_epoch']:3d} "
              f"max|diff|={record['max_abs_cell_difference']:.3e}", flush=True)
        for line in record["differences"][:5]:
            print(f"    {line}")
        failures += 0 if record["agrees"] else 1
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
