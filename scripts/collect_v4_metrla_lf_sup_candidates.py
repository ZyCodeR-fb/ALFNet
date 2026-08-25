#!/usr/bin/env python
"""Audit and summarise the five METR-LA validation-only candidate runs."""

import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.v4_validation_collector import cell_count  # noqa: E402
from scripts import build_v4_metrla_lf_sup_plan as candidate_plan  # noqa: E402
from scripts.v4_launcher import plan_runs  # noqa: E402
from scripts.v4_utils import sha256_file  # noqa: E402


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


def main():
    plan = candidate_plan.assert_plan_still_holds(candidate_plan.load_plan())
    root = ROOT / plan["output_root"] / plan["plan_sha256"]
    runs = plan_runs(root, plan)
    tables, evidence = {}, {}
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
        tables[run.candidate] = {"table": table, "cells": _cells(table)}
        evidence[run.candidate] = {
            "run": run.name,
            "config": str(run.config.relative_to(ROOT)),
            "config_sha256": sha256_file(run.config),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "best_epoch": int(checkpoint["best_epoch"]),
        }
    expected = set(plan["candidates"])
    if set(tables) != expected:
        raise SystemExit(f"candidate inventory differs: {set(tables)} != {expected}")

    sup, lf_sup = tables["SUP"], tables["LF_SUP"]
    wins = sum(lf_sup["cells"][key] < sup["cells"][key] for key in sup["cells"])
    ties = sum(lf_sup["cells"][key] == sup["cells"][key] for key in sup["cells"])
    lifeavg_better = (lf_sup["table"]["main"]["LifeAvg"]["MAE"]
                      < sup["table"]["main"]["LifeAvg"]["MAE"])
    unseen_mae_ok = (lf_sup["table"]["main"]["Unseen"]["MAE"]
                     <= 1.01 * sup["table"]["main"]["Unseen"]["MAE"])
    unseen_rmse_ok = (lf_sup["table"]["main"]["Unseen"]["RMSE"]
                      <= 1.01 * sup["table"]["main"]["Unseen"]["RMSE"])
    gates = {
        "lifeavg_mae_strictly_better": bool(lifeavg_better),
        "unseen_mae_within_one_percent": bool(unseen_mae_ok),
        "unseen_rmse_within_one_percent": bool(unseen_rmse_ok),
        "at_least_17_of_33_cell_wins": bool(wins >= 17),
    }
    adopt = all(gates.values())
    report = {
        "schema_version": 1,
        "plan_sha256": plan["plan_sha256"],
        "dataset": "METRLA",
        "seed": 32,
        "split": "validation",
        "test_accessed": False,
        "evidence": evidence,
        "lifeavg": {
            name: tables[name]["table"]["main"]["LifeAvg"]
            for name in plan["candidates"]
        },
        "lf_sup_minus_sup": {
            key: lf_sup["cells"][key] - sup["cells"][key]
            for key in sorted(sup["cells"])
        },
        "cell_wins": {"LF_SUP": wins, "SUP": 33 - wins - ties, "ties": ties},
        "decision_gates": gates,
        "decision": ("unified_LF_SUP" if adopt
                     else "retain_flow_LF_SUP_speed_SUP"),
    }
    output = root / "candidate_validation_summary.json"
    if output.exists():
        raise SystemExit(f"{output} already exists; refusing to overwrite")
    output.write_text(
        json.dumps(report, indent=1, sort_keys=True, allow_nan=False) + "\n", "utf-8")
    print(f"wrote {output}")
    for name in plan["candidates"]:
        row = report["lifeavg"][name]
        print(f"{name:8s} LifeAvg MAE={row['MAE']:.6f} "
              f"RMSE={row['RMSE']:.6f} MAPE={row['MAPE']:.6f}")
    print(f"LF_SUP vs SUP cells: {wins} wins, {33 - wins - ties} losses, {ties} ties")
    print(f"decision: {report['decision']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
