#!/usr/bin/env python
"""Aggregate a plan's test results. Never before the test audit passes.

Per dataset: 36 cells, three seeds, mean and sample std with ddof=1. Across
datasets: only dimensionless quantities. Three datasets are veh/5min and two are
mph, so a pooled absolute error would carry no unit -- and PEMS03/PEMS04 are
labelled confirmatory while the other three are supportive, which is a
distinction a single averaged number would erase.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.v4_validation_collector import METRICS, aggregate_seeds  # noqa: E402
from scripts.audit_v4_plan_test import audit  # noqa: E402
from scripts.collect_v4_plan_validation import compare  # noqa: E402
from scripts.eval_v4_plan_test import TEST_EVAL_NAME  # noqa: E402
from scripts.eval_v4_plan_validation import load_plan, resolve_runs  # noqa: E402

SUMMARY_NAME = "plan_test_summary.json"


class CollectRefused(SystemExit):
    """Not enough verified test evidence to aggregate."""


def collect(plan_path, output_root=None, skip_gates=False):
    plan = load_plan(plan_path)
    runs, root = resolve_runs(plan, output_root)
    if not skip_gates:
        result = audit(plan_path, output_root)
        if result["findings"]:
            raise CollectRefused(
                "the test audit has findings; refusing to aggregate:\n  - "
                + "\n  - ".join(result["findings"][:20]))
    tables = {}
    for run in runs:
        record = json.loads((run.directory / TEST_EVAL_NAME).read_text("utf-8"))
        tables.setdefault(run.dataset, {}).setdefault(
            run.model, {})[run.seed] = record["cells"]
    models, seeds = tuple(plan["models"]), tuple(int(s) for s in plan["seeds"])
    per_dataset = {d: compare(tables[d], models, seeds) for d in plan["datasets"]}
    left = models[0]
    summary = {
        "schema_version": 1, "split": "test", "read_once": True,
        "plan_sha256": plan["plan_sha256"],
        "models": list(models), "seeds": list(seeds), "n": len(seeds),
        "std_ddof": 1,
        "datasets": per_dataset,
        "exposure": {d: plan["data"][d]["test_exposure"]["status"]
                     for d in plan["datasets"]},
        "units": {d: plan["data"][d]["unit"] for d in plan["datasets"]},
        "cross_dataset": {
            "absolute_error_is_never_pooled": True,
            "reason": ("flow and speed do not share a unit, and confirmatory and "
                       "supportive datasets do not share a status"),
            f"{left.lower()}_cells_better_per_dataset": {
                d: per_dataset[d]["cells_better"][left] for d in plan["datasets"]},
            "confirmatory_datasets": [d for d in plan["datasets"]
                                      if plan["data"][d]["test_exposure"]["status"]
                                      == "confirmatory"],
        },
        "every_dataset_reported": True,
    }
    path = Path(root) / SUMMARY_NAME
    text = json.dumps(summary, indent=1, sort_keys=True, allow_nan=False) + "\n"
    if path.exists() and path.read_text("utf-8") != text:
        raise CollectRefused(f"{path} exists with different content")
    path.write_text(text, "utf-8")
    return summary, path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output-root", default=None)
    arguments = parser.parse_args(argv)
    summary, path = collect(arguments.plan, arguments.output_root)
    print(f"wrote {path}")
    left = summary["models"][0]
    for dataset, entry in summary["datasets"].items():
        print(f"  {dataset:8s} [{summary['exposure'][dataset]:12s}] "
              f"{left} better in {entry['cells_better'][left]}/36")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
