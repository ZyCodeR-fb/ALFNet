#!/usr/bin/env python
"""Aggregate the twelve development runs into one validation-only summary.

Refuses to run until the audit passes and all twelve validation evaluations
agree, because a summary is the point at which numbers stop being checked and
start being quoted. Descriptive only: three seeds, mean and sample std, per-cell
deltas and seed win counts. No p-value, no significance, no confirmatory claim,
and the word `final` appears nowhere in what it writes.

    python scripts/collect_v4_development_base.py
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
from scripts.audit_v4_development_base import audit  # noqa: E402
from scripts.build_v4_development_plan import (  # noqa: E402
    assert_plan_still_holds,
    load_plan,
)
from scripts.v4_development_eval import EVAL_NAME  # noqa: E402
from scripts.v4_launcher import plan_runs  # noqa: E402

SUMMARY_NAME = "development_validation_summary.json"
DATASETS = ("PEMS08", "PEMSBAY")
MODELS = ("ALFNet", "MoGERNN")
SEEDS = (32, 42, 52)


def load_tables(runs):
    """{dataset: {model: {seed: table}}}, from the *re-evaluated* tables."""
    tables = {dataset: {model: {} for model in MODELS} for dataset in DATASETS}
    missing = []
    for run in runs:
        path = run.directory / EVAL_NAME
        if not path.exists():
            missing.append(f"{run.name}: no {EVAL_NAME}")
            continue
        record = json.loads(path.read_text("utf-8"))
        if not record.get("agrees"):
            missing.append(f"{run.name}: its re-evaluation disagrees with training")
            continue
        tables[run.dataset][run.model][run.seed] = record["recomputed"]
    return tables, missing


def compare(dataset_tables):
    """Cell by cell: ALFNet minus MoGERNN, negative meaning ALFNet is better."""
    summaries = {model: aggregate_seeds(dataset_tables[model]) for model in MODELS}
    cells, wins = {}, {"ALFNet": 0, "ties": 0, "MoGERNN": 0}
    per_metric = {metric: {"ALFNet": 0, "MoGERNN": 0, "ties": 0}
                  for metric in METRICS}
    per_section = {"main": 0, "phase": 0}
    worst = {"cell": None, "delta": None}
    unanimous = {"ALFNet": [], "MoGERNN": []}
    for section, rows in (("main", MAIN_TABLE_ROWS), ("phase", PHASE_ORDER)):
        cells[section] = {}
        for row in rows:
            cells[section][row] = {}
            for metric in METRICS:
                left = summaries["ALFNet"][section][row][metric]
                right = summaries["MoGERNN"][section][row][metric]
                delta = left["mean"] - right["mean"]
                seed_wins = sum(1 for seed in SEEDS
                                if left["per_seed"][str(seed)]
                                < right["per_seed"][str(seed)])
                seed_ties = sum(1 for seed in SEEDS
                                if left["per_seed"][str(seed)]
                                == right["per_seed"][str(seed)])
                counts_agree = (
                    summaries["ALFNet"][section][row]["valid_target_count"]
                    == summaries["MoGERNN"][section][row]["valid_target_count"])
                cells[section][row][metric] = {
                    "alfnet_mean": left["mean"], "alfnet_std": left["std"],
                    "mogernn_mean": right["mean"], "mogernn_std": right["std"],
                    "delta_alfnet_minus_mogernn": delta,
                    "alfnet_seed_wins": seed_wins, "seed_ties": seed_ties,
                    "valid_target_counts_agree": counts_agree,
                }
                if delta < 0:
                    wins["ALFNet"] += 1
                    per_metric[metric]["ALFNet"] += 1
                    per_section[section] += 1
                elif delta > 0:
                    wins["MoGERNN"] += 1
                    per_metric[metric]["MoGERNN"] += 1
                else:
                    wins["ties"] += 1
                    per_metric[metric]["ties"] += 1
                if worst["delta"] is None or delta > worst["delta"]:
                    worst = {"cell": f"{section}/{row}/{metric}", "delta": delta}
                if seed_wins == len(SEEDS):
                    unanimous["ALFNet"].append(f"{section}/{row}/{metric}")
                elif seed_wins == 0 and seed_ties == 0:
                    unanimous["MoGERNN"].append(f"{section}/{row}/{metric}")
    return {
        "per_model": summaries, "cells": cells,
        "alfnet_cells_better": wins["ALFNet"], "mogernn_cells_better": wins["MoGERNN"],
        "tied_cells": wins["ties"], "total_cells": 36,
        "main_cells_alfnet_better": per_section["main"],
        "phase_cells_alfnet_better": per_section["phase"],
        "by_metric": per_metric,
        "worst_positive_delta": worst,
        "unanimous_across_seeds": unanimous,
    }


def collect(output_root=None, skip_gates=False):
    plan = assert_plan_still_holds(load_plan())
    root = Path(output_root or
                ROOT / "outputs_v4/development_base" / plan["plan_sha256"])
    runs = plan_runs(root, plan)

    if not skip_gates:
        result = audit(str(root))
        if result["findings"]:
            raise SystemExit(
                "the audit has findings; refusing to aggregate:\n  - "
                + "\n  - ".join(result["findings"]))
    tables, missing = load_tables(runs)
    if missing and not skip_gates:
        raise SystemExit(
            "validation re-evaluation is incomplete; refusing to aggregate:\n  - "
            + "\n  - ".join(missing))
    for dataset in DATASETS:
        for model in MODELS:
            present = sorted(tables[dataset][model])
            if present != list(SEEDS):
                raise SystemExit(
                    f"{dataset}/{model} has seeds {present}, expected {list(SEEDS)}")

    summary = {
        "schema_version": 1,
        "development_validation_only": True,
        "split": "validation",
        "note": ("Descriptive three-seed development diagnostic on the "
                 "validation split. Not a significance claim and not a result."),
        "plan_sha256": plan["plan_sha256"],
        "seeds": list(SEEDS), "n": len(SEEDS), "std_ddof": 1,
        "datasets": {dataset: compare(tables[dataset]) for dataset in DATASETS},
    }
    path = root / SUMMARY_NAME
    path.write_text(json.dumps(summary, indent=1, sort_keys=True,
                               allow_nan=False) + "\n", "utf-8")
    return summary, path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=None)
    arguments = parser.parse_args(argv)
    summary, path = collect(arguments.output_root)
    print(f"wrote {path}")
    for dataset in DATASETS:
        entry = summary["datasets"][dataset]
        print(f"\n{dataset}: ALFNet better in {entry['alfnet_cells_better']}/36 "
              f"(main {entry['main_cells_alfnet_better']}/18, "
              f"phase {entry['phase_cells_alfnet_better']}/18), "
              f"MoGERNN better in {entry['mogernn_cells_better']}, "
              f"tied {entry['tied_cells']}")
        print(f"  worst cell for ALFNet: {entry['worst_positive_delta']['cell']} "
              f"delta {entry['worst_positive_delta']['delta']:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
