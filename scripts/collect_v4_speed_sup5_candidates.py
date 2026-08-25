#!/usr/bin/env python
"""Audit the bounded speed SUP5 runs and apply the frozen decision gates."""

import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.v4_validation_collector import cell_count  # noqa: E402
from scripts import build_v4_speed_sup5_plan as builder  # noqa: E402
from scripts.v4_launcher import plan_runs  # noqa: E402
from scripts.v4_utils import sha256_file  # noqa: E402

METR_REFERENCE_NAMES = {
    "BASE": "BASE", "SUP6": "SUP", "LF_SUP6": "LF_SUP",
    "CONTROL": "CONTROL",
}


def _cells(table):
    result = {}
    for section in ("main", "phase"):
        for row, values in table[section].items():
            for metric in ("MAE", "RMSE", "MAPE"):
                value = float(values[metric])
                if not math.isfinite(value):
                    raise ValueError(f"non-finite {section}/{row}/{metric}")
                result[f"{section}/{row}/{metric}"] = value
    if len(result) != 33 or cell_count(table) != 33:
        raise ValueError(f"expected the frozen 33 cells, got {len(result)}")
    return result


def _load_table(path, expected_sha):
    if sha256_file(path) != expected_sha:
        raise ValueError(f"{path} changed after plan freeze")
    table = json.loads(path.read_text("utf-8"))
    return {"table": table, "cells": _cells(table)}


def _wins(left, right):
    keys = sorted(left["cells"])
    wins = sum(left["cells"][key] < right["cells"][key] for key in keys)
    losses = sum(left["cells"][key] > right["cells"][key] for key in keys)
    return {"wins": wins, "losses": losses, "ties": 33 - wins - losses}


def _gates(candidate, base, control):
    candidate_main = candidate["table"]["main"]
    base_main = base["table"]["main"]
    versus_base = _wins(candidate, base)
    versus_control = _wins(candidate, control)
    gates = {
        "lifeavg_mae_not_above_base": (
            candidate_main["LifeAvg"]["MAE"] <= base_main["LifeAvg"]["MAE"]),
        "lifeavg_rmse_not_above_base": (
            candidate_main["LifeAvg"]["RMSE"] <= base_main["LifeAvg"]["RMSE"]),
        "lifeavg_mape_within_one_percent_of_base": (
            candidate_main["LifeAvg"]["MAPE"]
            <= 1.01 * base_main["LifeAvg"]["MAPE"]),
        "unseen_mae_within_one_percent_of_base": (
            candidate_main["Unseen"]["MAE"]
            <= 1.01 * base_main["Unseen"]["MAE"]),
        "unseen_rmse_within_one_percent_of_base": (
            candidate_main["Unseen"]["RMSE"]
            <= 1.01 * base_main["Unseen"]["RMSE"]),
        "at_least_17_of_33_wins_vs_base": versus_base["wins"] >= 17,
        "at_least_18_of_33_wins_vs_control": versus_control["wins"] >= 18,
    }
    return {
        "passed": all(gates.values()), "gates": gates,
        "cells_vs_base": versus_base, "cells_vs_control": versus_control,
    }


def main():
    plan = builder.assert_plan_still_holds(builder.load_plan())
    root = ROOT / plan["output_root"] / plan["plan_sha256"]
    tables = {"METRLA": {}, "PEMSBAY": {}}
    evidence = {}

    reference = plan["reference_evidence"]["METRLA"]
    for report_name, reference_name in METR_REFERENCE_NAMES.items():
        item = reference["records"][reference_name]
        tables["METRLA"][report_name] = _load_table(
            ROOT / item["table_path"], item["table_sha256"])
        evidence[f"METRLA/{report_name}"] = {
            "source": "frozen_reference", **item,
        }

    runs = plan_runs(root, plan)
    if len(runs) != 6:
        raise SystemExit(f"expected six planned runs, got {len(runs)}")
    for run in runs:
        complete = run.directory / "COMPLETE"
        checkpoint_path = run.directory / "best.pt"
        table_path = run.directory / "development_validation_table.json"
        if not complete.exists() or not checkpoint_path.exists() or not table_path.exists():
            raise SystemExit(f"{run.name} is not COMPLETE with checkpoint and table")
        checkpoint = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
        if checkpoint.get("plan_sha256") != plan["plan_sha256"]:
            raise SystemExit(f"{run.name}: checkpoint belongs to another plan")
        if int(checkpoint.get("seed", -1)) != 32:
            raise SystemExit(f"{run.name}: checkpoint seed is not 32")
        table = json.loads(table_path.read_text("utf-8"))
        tables[run.dataset][run.candidate] = {
            "table": table, "cells": _cells(table),
        }
        evidence[f"{run.dataset}/{run.candidate}"] = {
            "source": "new_run", "run": run.name,
            "config": str(run.config.relative_to(ROOT)),
            "config_sha256": sha256_file(run.config),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "table_sha256": sha256_file(table_path),
            "best_epoch": int(checkpoint["best_epoch"]),
        }

    expected = {
        "METRLA": {"BASE", "SUP6", "LF_SUP6", "SUP5", "LF_SUP5", "CONTROL"},
        "PEMSBAY": {"BASE", "SUP5", "LF_SUP5", "CONTROL"},
    }
    for dataset, names in expected.items():
        if set(tables[dataset]) != names:
            raise SystemExit(f"{dataset} inventory differs: {set(tables[dataset])}")

    decisions = {}
    for candidate in ("SUP5", "LF_SUP5"):
        decisions[candidate] = {
            dataset: _gates(tables[dataset][candidate], tables[dataset]["BASE"],
                            tables[dataset]["CONTROL"])
            for dataset in ("METRLA", "PEMSBAY")
        }
        decisions[candidate]["passed_both_datasets"] = all(
            decisions[candidate][dataset]["passed"]
            for dataset in ("METRLA", "PEMSBAY"))

    if decisions["SUP5"]["passed_both_datasets"]:
        decision = "unified_SUP5"
    elif decisions["LF_SUP5"]["passed_both_datasets"]:
        decision = "unified_LF_SUP5"
    else:
        decision = "unified_BASE"

    report = {
        "schema_version": 1,
        "plan_sha256": plan["plan_sha256"],
        "datasets": ["METRLA", "PEMSBAY"],
        "seed": 32, "split": "validation", "test_accessed": False,
        "evidence": evidence,
        "lifeavg": {
            dataset: {
                name: tables[dataset][name]["table"]["main"]["LifeAvg"]
                for name in sorted(tables[dataset])
            }
            for dataset in ("METRLA", "PEMSBAY")
        },
        "candidate_decisions": decisions,
        "decision": decision,
    }
    output = root / "speed_sup5_validation_summary.json"
    if output.exists():
        raise SystemExit(f"{output} already exists; refusing to overwrite")
    output.write_text(
        json.dumps(report, indent=1, sort_keys=True, allow_nan=False) + "\n",
        "utf-8")
    print(f"wrote {output}")
    for dataset in ("METRLA", "PEMSBAY"):
        print(dataset)
        for name, row in report["lifeavg"][dataset].items():
            print(f"  {name:8s} MAE={row['MAE']:.6f} "
                  f"RMSE={row['RMSE']:.6f} MAPE={row['MAPE']:.6f}")
    print(f"decision: {decision}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
