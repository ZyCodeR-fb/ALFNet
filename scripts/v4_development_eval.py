#!/usr/bin/env python
"""Re-score each development run's `best.pt` on validation, and check it agrees.

A table written during training is a table produced by the same process that
produced the weights. Recomputing it from `best.pt` in a fresh process is the
cheapest available check that the number on disk is a property of the checkpoint
and not of the run that happened to write it -- a mismatch means the table was
computed against different weights, a different scaler or a different schedule.

Validation only, in every sense: it builds no test context, the collector refuses
a non-validation context, and the teacher-forcing mask is asserted empty.

    python scripts/v4_development_eval.py --jobs 3 --devices 0,1,2
    python scripts/v4_development_eval.py --run PEMS08/ALFNet/seed32 --device cuda:0
"""

import argparse
import json
import sys
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
)
from scripts.train_v4 import COMPLETE_MARKER  # noqa: E402
from scripts.v4_launcher import plan_runs  # noqa: E402

EVAL_NAME = "development_validation_eval.json"
# Recomputation on the same weights and the same data is deterministic, so the
# tolerance is for float summation order alone, not for a different result.
TOLERANCE = 1e-6


def evaluate_run(run, device="cuda:0", max_batches=None):
    directory = run.directory
    if not (directory / COMPLETE_MARKER).exists():
        raise SystemExit(f"{run.name}: not COMPLETE; nothing to evaluate")
    checkpoint = torch.load(directory / "best.pt", weights_only=False,
                            map_location="cpu")
    session = build_v4_session(run.config, device=device)
    if session.validation_context.phase != "validation":
        raise SystemExit("the eval session did not resolve a validation context")
    if session.model.static_asset_hash != checkpoint["static_asset_hash"]:
        raise SystemExit(f"{run.name}: checkpoint static assets do not match")
    session.model.load_state_dict(checkpoint["model_state"])
    session.model.to(device)
    table = collect_validation_table(session, epoch=int(checkpoint["best_epoch"]),
                                     max_batches=max_batches)
    if cell_count(table) != 36:
        raise SystemExit(f"{run.name}: recomputed {cell_count(table)} cells")

    stored = json.loads(
        (directory / "development_validation_table.json").read_text("utf-8"))
    differences, worst = [], 0.0
    for section in ("main", "phase"):
        for row in stored[section]:
            for metric in METRICS:
                a = float(stored[section][row][metric])
                b = float(table[section][row][metric])
                gap = abs(a - b)
                worst = max(worst, gap)
                if gap > TOLERANCE:
                    differences.append(f"{section}/{row}/{metric}: "
                                       f"stored {a!r} recomputed {b!r}")
    record = {
        "schema_version": 1,
        "development_validation_only": True,
        "run": run.name, "device": device,
        "best_epoch": int(checkpoint["best_epoch"]),
        "max_abs_cell_difference": worst,
        "tolerance": TOLERANCE,
        "agrees": not differences,
        "differences": differences,
        "recomputed": table,
    }
    (directory / EVAL_NAME).write_text(
        json.dumps(record, indent=1, sort_keys=True, allow_nan=False) + "\n", "utf-8")
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default=None,
                        help="a single run, e.g. PEMS08/ALFNet/seed32")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    arguments = parser.parse_args(argv)

    plan = assert_plan_still_holds(load_plan())
    root = Path(arguments.output_root or
                ROOT / "outputs_v4/development_base" / plan["plan_sha256"])
    runs = plan_runs(root, plan)
    if arguments.run:
        runs = [run for run in runs if run.name == arguments.run]
        if not runs:
            raise SystemExit(f"no run named {arguments.run}")

    failures = 0
    for run in runs:
        record = evaluate_run(run, arguments.device, arguments.max_batches)
        status = "AGREES" if record["agrees"] else "DIFFERS"
        print(f"{status:8s} {run.name:26s} best_epoch={record['best_epoch']:3d} "
              f"max|diff|={record['max_abs_cell_difference']:.3e}", flush=True)
        for line in record["differences"]:
            print(f"    {line}")
        failures += 0 if record["agrees"] else 1
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
