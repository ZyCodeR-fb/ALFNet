#!/usr/bin/env python
"""Collect the final cumulative structure validation-only ablation."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import build_masked_node_candidates as build  # noqa: E402


def main():
    plan = build.assert_plan_still_holds(build.load_plan())
    root = ROOT / plan["output_root"] / plan["plan_sha256"]
    reference = ROOT / plan["reference_control"]["path"]
    tables = {"AR_BASE": json.loads(
        (reference / "development_validation_table.json").read_text("utf-8"))}
    reports = {"AR_BASE": json.loads(
        (reference / "run_report.json").read_text("utf-8"))}
    for entry in plan["runs"]:
        name = entry["candidate"]
        digest = plan["configs"][name]["sha256"][:16]
        directory = root / "PEMS03/ALFNet" / digest / "seed32"
        if not (directory / "COMPLETE").exists():
            raise SystemExit(f"{name} is not COMPLETE: {directory}")
        tables[name] = json.loads(
            (directory / "development_validation_table.json").read_text("utf-8"))
        reports[name] = json.loads((directory / "run_report.json").read_text("utf-8"))

    base = tables["AR_BASE"]
    keys = [(section, row, metric) for section in ("main", "phase")
            for row in base[section] for metric in ("MAE", "RMSE", "MAPE")]
    comparisons, eligible = {}, []
    for name in build.CANDIDATES:
        table = tables[name]
        wins = sum(table[s][r][m] < base[s][r][m] for s, r, m in keys)
        losses = sum(table[s][r][m] > base[s][r][m] for s, r, m in keys)
        gates = {
            "lifeavg_mae": table["main"]["LifeAvg"]["MAE"]
                < base["main"]["LifeAvg"]["MAE"],
            "lifeavg_rmse": table["main"]["LifeAvg"]["RMSE"]
                < base["main"]["LifeAvg"]["RMSE"],
            "unseen_mae": table["main"]["Unseen"]["MAE"]
                < base["main"]["Unseen"]["MAE"],
            "unseen_rmse": table["main"]["Unseen"]["RMSE"]
                < base["main"]["Unseen"]["RMSE"],
            "cell_majority": wins >= 17,
        }
        comparisons[name] = {
            "wins_vs_ar_base": wins, "losses_vs_ar_base": losses,
            "ties_vs_ar_base": len(keys) - wins - losses,
            "mandatory_gates": gates, "passes": all(gates.values()),
        }
        if all(gates.values()):
            eligible.append(name)
    selected = (min(eligible, key=lambda name:
                    tables[name]["main"]["LifeAvg"]["MAE"])
                if eligible else "AR_BASE")
    output = {
        "schema_version": 1, "development_validation_only": True,
        "plan_sha256": plan["plan_sha256"], "comparisons": comparisons,
        "selected_by_preregistered_rule": selected,
        "reports": {name: {
            "best_epoch": reports[name]["best_epoch"],
            "epochs_run": reports[name]["epochs_run"],
            "parameter_count": reports[name]["parameter_count"],
            "best_lifeavg_mae": reports[name]["best_validation_life_avg_mae"],
            "model_method_contract": reports[name].get("model_method_contract"),
        } for name in ("AR_BASE",) + build.CANDIDATES},
        "tables": tables,
    }
    path = root / "final_structure_validation_summary.json"
    path.write_text(json.dumps(
        output, indent=1, sort_keys=True, allow_nan=False) + "\n", "utf-8")
    print(f"wrote {path}")
    for name in ("AR_BASE",) + build.CANDIDATES:
        row = tables[name]["main"]["LifeAvg"]
        unseen = tables[name]["main"]["Unseen"]
        print(f"{name:17s} LifeAvg={row['MAE']:.6f}/{row['RMSE']:.6f}/"
              f"{row['MAPE']:.6f}  Unseen={unseen['MAE']:.6f}/"
              f"{unseen['RMSE']:.6f}/{unseen['MAPE']:.6f}")
    print("comparisons", comparisons)
    print("selected", selected)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
