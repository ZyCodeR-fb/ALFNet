#!/usr/bin/env python
"""Aggregate a plan's validation slots. Descriptive, plan-driven, unit-aware.

Refuses until the audit passes and every slot has a full, agreeing, own-seed
re-evaluation, because a summary is where numbers stop being checked and start
being quoted.

Flow and speed are never pooled. Three datasets are veh/5min and two are mph, so
a single `avg_mae` across them is a number with no unit. The cross-dataset
section carries only dimensionless quantities.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data_provider.ples_protocol import MAIN_TABLE_ROWS, PHASE_ORDER  # noqa: E402
from experiments.v4_validation_collector import METRICS, aggregate_seeds  # noqa: E402
from scripts.audit_v4_plan import audit  # noqa: E402
from scripts.eval_v4_plan_validation import (  # noqa: E402
    EVAL_NAME, EVAL_SCHEMA, PARTIAL_NAME, TOLERANCE, load_plan, resolve_runs,
)

SUMMARY_NAME = "plan_validation_summary.json"


class CollectRefused(SystemExit):
    """Not enough verified evidence to aggregate."""


def load_tables(runs):
    tables, missing = {}, []
    for run in runs:
        path = run.directory / EVAL_NAME
        if not path.exists():
            missing.append(f"{run.name}: no {EVAL_NAME}")
            continue
        record = json.loads(path.read_text("utf-8"))
        for reason, bad in (
            (f"schema {record.get('schema_version')} is not v{EVAL_SCHEMA}",
             record.get("schema_version") != EVAL_SCHEMA),
            ("a partial diagnostic is not a result",
             bool(record.get("partial_diagnostic_only"))),
            (f"evaluated at seed {record.get('run_seed_used')}, not {run.seed}",
             int(record.get("run_seed_used", -1)) != int(run.seed)),
            ("its re-evaluation disagrees", not record.get("agrees")),
            (f"max cell difference {record.get('max_abs_cell_difference')} "
             f"exceeds {TOLERANCE}",
             float(record.get("max_abs_cell_difference", 1.0)) > TOLERANCE),
            ("did not use the checkpoint scaler",
             record.get("scaler", {}).get("refit") is not False),
            ("a partial file sits beside the full one",
             (run.directory / PARTIAL_NAME).exists()),
        ):
            if bad:
                missing.append(f"{run.name}: {reason}")
                break
        else:
            tables.setdefault(run.dataset, {}).setdefault(
                run.model, {})[run.seed] = record["recomputed"]
    return tables, missing


def compare(dataset_tables, models, seeds):
    summaries = {model: aggregate_seeds(dataset_tables[model]) for model in models}
    left_name, right_name = models
    cells, wins = {}, {left_name: 0, right_name: 0, "ties": 0}
    by_metric = {m: {left_name: 0, right_name: 0, "ties": 0} for m in METRICS}
    by_section = {"main": 0, "phase": 0}
    worst = {"cell": None, "delta": None}
    unanimous = {left_name: [], right_name: []}
    for section, rows in (("main", MAIN_TABLE_ROWS), ("phase", PHASE_ORDER)):
        cells[section] = {}
        for row in rows:
            cells[section][row] = {}
            for metric in METRICS:
                left = summaries[left_name][section][row][metric]
                right = summaries[right_name][section][row][metric]
                delta = left["mean"] - right["mean"]
                seed_wins = sum(1 for s in seeds
                                if left["per_seed"][str(s)] < right["per_seed"][str(s)])
                ties = sum(1 for s in seeds
                           if left["per_seed"][str(s)] == right["per_seed"][str(s)])
                cells[section][row][metric] = {
                    f"{left_name.lower()}_mean": left["mean"],
                    f"{left_name.lower()}_std": left["std"],
                    f"{right_name.lower()}_mean": right["mean"],
                    f"{right_name.lower()}_std": right["std"],
                    "delta_left_minus_right": delta,
                    f"{left_name.lower()}_seed_wins": seed_wins,
                    "seed_ties": ties,
                    "valid_target_counts_agree":
                        summaries[left_name][section][row]["valid_target_count"]
                        == summaries[right_name][section][row]["valid_target_count"],
                }
                if delta < 0:
                    wins[left_name] += 1
                    by_metric[metric][left_name] += 1
                    by_section[section] += 1
                elif delta > 0:
                    wins[right_name] += 1
                    by_metric[metric][right_name] += 1
                else:
                    wins["ties"] += 1
                    by_metric[metric]["ties"] += 1
                if worst["delta"] is None or delta > worst["delta"]:
                    worst = {"cell": f"{section}/{row}/{metric}", "delta": delta}
                if seed_wins == len(seeds):
                    unanimous[left_name].append(f"{section}/{row}/{metric}")
                elif seed_wins == 0 and ties == 0:
                    unanimous[right_name].append(f"{section}/{row}/{metric}")
    return {"per_model": summaries, "cells": cells,
            "cells_better": wins, "total_cells": 36,
            "main_cells_left_better": by_section["main"],
            "phase_cells_left_better": by_section["phase"],
            "by_metric": by_metric, "worst_positive_delta": worst,
            "unanimous_across_seeds": unanimous}


def collect(plan_path, output_root=None, skip_gates=False):
    plan = load_plan(plan_path)
    runs, root = resolve_runs(plan, output_root)
    if not skip_gates:
        result = audit(plan_path, output_root)
        if result["findings"]:
            raise CollectRefused(
                "the audit has findings; refusing to aggregate:\n  - "
                + "\n  - ".join(result["findings"][:20]))
    tables, missing = load_tables(runs)
    if missing and not skip_gates:
        raise CollectRefused(
            "validation evidence is incomplete; refusing to aggregate:\n  - "
            + "\n  - ".join(missing[:20]))
    models, seeds = tuple(plan["models"]), tuple(int(s) for s in plan["seeds"])
    for dataset in plan["datasets"]:
        for model in models:
            present = sorted(tables.get(dataset, {}).get(model, {}))
            if present != list(seeds):
                raise CollectRefused(
                    f"{dataset}/{model} has seeds {present}, expected {list(seeds)}")

    per_dataset = {d: compare(tables[d], models, seeds) for d in plan["datasets"]}
    left = models[0]
    summary = {
        "schema_version": 1,
        "plan_validation_only": True,
        "split": "validation",
        "plan_sha256": plan["plan_sha256"],
        "note": ("Descriptive three-seed validation summary. Not a significance "
                 "claim and not a test result."),
        "models": list(models), "seeds": list(seeds), "n": len(seeds),
        "std_ddof": 1,
        "datasets": per_dataset,
        "cross_dataset": {
            "units_differ": True,
            "absolute_error_is_never_pooled": True,
            "reason": ("three datasets are veh/5min and two are mph; a mean over "
                       "them would carry no unit"),
            f"{left.lower()}_cells_better_per_dataset": {
                d: per_dataset[d]["cells_better"][left] for d in plan["datasets"]},
            "datasets_where_left_wins_every_cell": [
                d for d in plan["datasets"]
                if per_dataset[d]["cells_better"][left] == 36],
        },
    }
    path = root / SUMMARY_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text("utf-8") != json.dumps(
            summary, indent=1, sort_keys=True, allow_nan=False) + "\n":
        raise CollectRefused(f"{path} exists with different content")
    path.write_text(json.dumps(summary, indent=1, sort_keys=True,
                               allow_nan=False) + "\n", "utf-8")
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
        print(f"\n{dataset}: {left} better in "
              f"{entry['cells_better'][left]}/36 "
              f"(main {entry['main_cells_left_better']}/18, "
              f"phase {entry['phase_cells_left_better']}/18)")
        print(f"  worst cell for {left}: {entry['worst_positive_delta']['cell']} "
              f"{entry['worst_positive_delta']['delta']:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
