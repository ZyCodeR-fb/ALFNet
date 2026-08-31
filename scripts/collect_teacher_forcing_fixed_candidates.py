#!/usr/bin/env python
"""Compare the two fixed-50 validation-only decoder-policy candidates."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import build_teacher_forcing_fixed_candidates as build  # noqa: E402


def main():
    plan = build.assert_plan_still_holds(build.load_plan())
    root = ROOT / plan["output_root"] / plan["plan_sha256"]
    tables, reports = {}, {}
    for entry in plan["runs"]:
        name = entry["candidate"]
        digest = plan["configs"][name]["sha256"][:16]
        directory = root / "PEMS03/ALFNet" / digest / "seed32"
        if not (directory / "COMPLETE").exists():
            raise SystemExit(f"{name} is not COMPLETE")
        tables[name] = json.loads(
            (directory / "development_validation_table.json").read_text("utf-8"))
        reports[name] = json.loads((directory / "run_report.json").read_text("utf-8"))
    names = list(build.CANDIDATES)
    left, right = (tables[name] for name in names)
    keys = [(section, row, metric) for section in ("main", "phase")
            for row in left[section] for metric in ("MAE", "RMSE", "MAPE")]
    comparison = {
        "left": names[0], "right": names[1],
        "left_wins": sum(left[s][r][m] < right[s][r][m] for s, r, m in keys),
        "right_wins": sum(left[s][r][m] > right[s][r][m] for s, r, m in keys),
        "ties": sum(left[s][r][m] == right[s][r][m] for s, r, m in keys),
    }
    output = {"schema_version": 1, "development_validation_only": True,
              "plan_sha256": plan["plan_sha256"], "comparison": comparison,
              "reports": {name: {"best_epoch": reports[name]["best_epoch"],
                                  "epochs_run": reports[name]["epochs_run"],
                                  "best_lifeavg_mae": reports[name][
                                      "best_validation_life_avg_mae"]}
                          for name in names},
              "tables": tables}
    path = root / "fixed50_validation_summary.json"
    path.write_text(json.dumps(output, indent=1, sort_keys=True,
                               allow_nan=False) + "\n", "utf-8")
    print(f"wrote {path}")
    print(comparison)
    for name in names:
        row = tables[name]["main"]["LifeAvg"]
        print(f"{name}: best_epoch={reports[name]['best_epoch']}  "
              f"LifeAvg={row['MAE']:.6f}/{row['RMSE']:.6f}/{row['MAPE']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
