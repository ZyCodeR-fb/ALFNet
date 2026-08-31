#!/usr/bin/env python
"""Collect the fair TF40 PEMS03 ALFNet validation candidates."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import build_alfnet_tf40_candidates as build  # noqa: E402
from scripts.v4_utils import sha256_file  # noqa: E402

METRICS = ("MAE", "RMSE", "MAPE")
SECTIONS = ("main", "phase")


def cells(table):
    return {(section, row, metric): float(values[metric])
            for section in SECTIONS for row, values in table[section].items()
            for metric in METRICS}


def main():
    plan = build.assert_plan_still_holds(build.load_plan())
    root = ROOT / plan["output_root"] / plan["plan_sha256"]
    tables, provenance = {}, {}
    for entry in plan["runs"]:
        digest = plan["configs"][entry["candidate"]]["sha256"][:16]
        directory = root / entry["dataset"] / entry["model"] / digest / f"seed{entry['seed']}"
        if not (directory / "COMPLETE").exists():
            raise SystemExit(f"{entry['candidate']} is not COMPLETE: {directory}")
        path = directory / "development_validation_table.json"
        table = json.loads(path.read_text("utf-8"))
        if table.get("split") != "validation" or not table.get(
                "development_validation_only"):
            raise SystemExit(f"{path} is not a validation-only table")
        tables[entry["candidate"]] = table
        provenance[entry["candidate"]] = {
            "path": str(path.relative_to(ROOT)), "sha256": sha256_file(path)}
    base = cells(tables["BASE"])
    comparisons = {}
    for name, table in tables.items():
        values = cells(table)
        comparisons[name] = {
            "wins_vs_rerun_BASE": sum(values[key] < base[key] for key in base),
            "losses_vs_rerun_BASE": sum(values[key] > base[key] for key in base),
            "ties_vs_rerun_BASE": sum(values[key] == base[key] for key in base),
            "lifeavg": {metric: values[("main", "LifeAvg", metric)]
                        for metric in METRICS},
            "unseen": {metric: values[("main", "Unseen", metric)]
                       for metric in METRICS},
        }
    ranked = sorted(plan["candidates"], key=lambda name: (
        -comparisons[name]["wins_vs_rerun_BASE"],
        comparisons[name]["lifeavg"]["MAE"],
    ))
    report = {
        "schema_version": 1, "development_validation_only": True,
        "split": "validation", "dataset": "PEMS03", "seed": 32,
        "plan_sha256": plan["plan_sha256"],
        "training_contract": plan["training_contract"],
        "selection_rule": plan["selection_rule"], "ranking": ranked,
        "comparisons": comparisons, "tables": tables,
        "input_tables": provenance,
    }
    path = root / "candidate_validation_summary.json"
    path.write_text(json.dumps(report, indent=1, sort_keys=True,
                               allow_nan=False) + "\n", "utf-8")
    print(f"wrote {path}")
    for name in ranked:
        item, life = comparisons[name], comparisons[name]["lifeavg"]
        print(f"{name:8s} BASE {item['wins_vs_rerun_BASE']:2d}W/"
              f"{item['losses_vs_rerun_BASE']:2d}L  LifeAvg "
              f"{life['MAE']:.3f}/{life['RMSE']:.3f}/{life['MAPE']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
