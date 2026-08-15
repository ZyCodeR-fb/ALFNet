"""Independent recomputation of the v3 tables, straight from the row files.

``scripts/collect_final_18.py`` reads each run's ``lce_*_summary.json`` and
averages across seeds. Those summaries were written by the evaluator, so the
collector cannot detect an evaluator that aggregated wrongly -- it would average
a consistent set of wrong numbers and report them cleanly.

This script recomputes every cell from ``ples_paired_rows.csv`` instead, using
``data_provider.ples_protocol`` directly, and compares:

  * recomputed per-run metrics against that run's frozen summary JSON;
  * recomputed three-seed means against ``final_18_tables.json``.

It is a *checker*: it never writes into the frozen v3 artefacts. Its own output
goes to a separate file, and it adds the reporting the tables lack -- the
ALFNet-minus-MoGERNN delta, relative improvement, and per-cell seed agreement --
without touching the frozen aggregation semantics, which are correct and are not
to be adjusted to move a win count.

    python scripts/verify_collector_recompute.py --output-dir ./outputs/reports/final
"""

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_provider.ples_protocol import (  # noqa: E402
    summarize_lce_rows,
    summarize_paired_rows,
)
from scripts.final_runs import load_manifest, verify_eval  # noqa: E402

MAIN_ROWS = ("Unseen", "Steady", "Join", "Fault", "LifeAvg", "WorstGroup")
PHASE_ROWS = ("Cold", "Warm", "Ready", "Onset", "Decay", "Outage")
METRICS = ("MAPE", "MAE", "RMSE")
PLES_EFFECTS = ("JoinLift", "FaultPenalty", "OutageFaultPenalty")
MODELS = ("ALFNet", "MoGERNN")

INT_FIELDS = ("origin_id", "cohort_id", "tau", "count")
FLOAT_FIELDS = ("treated_ape_sum", "control_ape_sum", "treated_ae_sum",
                "control_ae_sum", "treated_se_sum", "control_se_sum")


def _read_rows(directory):
    path = directory / "ples_paired_rows.csv"
    if not path.exists():
        raise SystemExit(f"missing row file: {path}")
    rows = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            row = dict(raw)
            for key in INT_FIELDS:
                row[key] = int(row[key])
            for key in FLOAT_FIELDS:
                row[key] = float(row[key])
            rows.append(row)
    if not rows:
        raise SystemExit(f"row file is empty: {path}")
    return rows


def _finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def recompute_run(directory):
    """Every published cell for one run, from its rows and its own manifest."""
    manifest = json.loads((directory / "protocol_manifest.json").read_text("utf-8"))
    resolved = manifest.get("resolved") or {}
    input_len = resolved.get("input_len")
    main_cohort_ids = resolved.get("main_cohort_ids")
    if not input_len or not main_cohort_ids:
        raise SystemExit(f"{directory}: manifest declares no input_len/main_cohort_ids")

    rows = _read_rows(directory)
    lce = summarize_lce_rows(rows, int(input_len), main_cohort_ids)
    ples = summarize_paired_rows(rows, int(input_len), main_cohort_ids)
    return {
        "main": lce["main_metrics"],
        "phase": lce["phase_metrics"],
        "ples": ples["paired_effects"],
        "input_len": int(input_len),
        "main_cohort_ids": [int(v) for v in main_cohort_ids],
    }


def _compare(label, got, expected, worst, problems, tolerance):
    if not _finite(got) or not _finite(expected):
        problems.append(f"{label}: non-finite value got={got!r} expected={expected!r}")
        return worst
    difference = abs(got - expected)
    if difference > tolerance:
        problems.append(f"{label}: recomputed {got!r} vs frozen {expected!r} "
                        f"(|d|={difference:.3e} > {tolerance:.1e})")
    return max(worst, difference)


def verify(payload, output_root, tolerance):
    problems = []
    worst_run = 0.0
    per_run = defaultdict(dict)

    for run in payload["runs"]:
        run_problems, directory = verify_eval(run, output_root)
        if run_problems:
            problems.append(f"{run['run_id']}: " + "; ".join(run_problems))
            continue
        directory = Path(directory)

        frozen_main = json.loads((directory / "lce_main_summary.json").read_text("utf-8"))
        frozen_phase = json.loads((directory / "lce_phase_summary.json").read_text("utf-8"))
        frozen_ples = json.loads((directory / "ples_effects_summary.json").read_text("utf-8"))

        # Identity, before any number is compared.
        for label, found, expected in (
            ("model", frozen_main.get("model"), run["model"]),
            ("seed", frozen_main.get("seed"), run["seed"]),
            ("dataset", (frozen_main.get("dataset") or {}).get("name"), run["dataset"]),
        ):
            if found != expected:
                problems.append(f"{run['run_id']}: {label}={found!r}, manifest says {expected!r}")
        if not frozen_main.get("formal_protocol_match", False):
            problems.append(f"{run['run_id']}: formal_protocol_match is false")

        got = recompute_run(directory)
        for row in MAIN_ROWS:
            for metric in METRICS:
                worst_run = _compare(
                    f"{run['run_id']} main/{row}/{metric}",
                    got["main"][row][metric],
                    (frozen_main["metrics"].get(row) or {}).get(metric),
                    worst_run, problems, tolerance)
        for row in PHASE_ROWS:
            for metric in METRICS:
                worst_run = _compare(
                    f"{run['run_id']} phase/{row}/{metric}",
                    got["phase"][row][metric],
                    (frozen_phase["metrics"].get(row) or {}).get(metric),
                    worst_run, problems, tolerance)
        for metric in ("MAPE", "MAE"):
            for effect in PLES_EFFECTS:
                worst_run = _compare(
                    f"{run['run_id']} ples/{effect}/{metric}",
                    got["ples"][metric][effect],
                    (frozen_ples["paired_effects"].get(metric) or {}).get(effect),
                    worst_run, problems, tolerance)

        per_run[(run["dataset"], run["model"])][run["seed"]] = got

    if problems:
        raise SystemExit("RECOMPUTATION FAILED:\n" + "\n".join(f"  - {p}" for p in problems))
    return per_run, worst_run


def _mean_std(values):
    return {
        "mean": float(statistics.fmean(values)),
        "std": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
        "n": len(values),
    }


def build_report(per_run, payload):
    """Three-seed means plus the comparison columns the frozen tables omit."""
    report = {}
    for dataset in payload["datasets"]:
        tables = {}
        for table, rows in (("main", MAIN_ROWS), ("phase", PHASE_ROWS)):
            tables[table] = {}
            for row in rows:
                tables[table][row] = {}
                for metric in METRICS:
                    per_model, per_seed = {}, {}
                    for model in MODELS:
                        cell = per_run.get((dataset, model), {})
                        values = [cell[seed][table][row][metric] for seed in sorted(cell)]
                        per_model[model] = _mean_std(values)
                        per_seed[model] = dict(zip(sorted(cell), values))
                    seeds = sorted(per_seed["ALFNet"])
                    wins = sum(1 for s in seeds
                               if per_seed["ALFNet"][s] < per_seed["MoGERNN"][s])
                    alf, mog = per_model["ALFNet"]["mean"], per_model["MoGERNN"]["mean"]
                    tables[table][row][metric] = {
                        **per_model,
                        # every LCE metric is lower-is-better
                        "delta_alfnet_minus_mogernn": float(alf - mog),
                        "relative_improvement_pct": float((mog - alf) / mog * 100.0)
                        if mog else None,
                        "alfnet_seed_wins": wins,
                        "seed_count": len(seeds),
                        "seed_unanimous": wins in (0, len(seeds)),
                        "per_seed": {m: per_seed[m] for m in MODELS},
                    }
        report[dataset] = tables
    return report


def _win_counts(report):
    out = {}
    for dataset, tables in report.items():
        counts = {}
        for table in ("main", "phase"):
            counts[table] = sum(
                1
                for row in tables[table]
                for metric in METRICS
                if tables[table][row][metric]["delta_alfnet_minus_mogernn"] < 0
            )
        counts["total"] = counts["main"] + counts["phase"]
        out[dataset] = counts
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="./outputs")
    parser.add_argument("--output-dir", default="./outputs/reports/final")
    parser.add_argument("--tolerance", type=float, default=1e-9,
                        help="max accepted |recomputed - frozen| per cell")
    args = parser.parse_args()

    payload = load_manifest()
    per_run, worst = verify(payload, args.output_root, args.tolerance)
    report = build_report(per_run, payload)
    counts = _win_counts(report)

    print(f"recomputed {len(payload['runs'])} runs from ples_paired_rows.csv")
    print(f"max |recomputed - frozen| over every published cell = {worst:.3e}")
    print()
    print(f"{'dataset':<10}{'main':>10}{'phase':>10}{'total':>10}   descriptive mean-wins")
    total_main = total_phase = 0
    for dataset, entry in counts.items():
        total_main += entry["main"]
        total_phase += entry["phase"]
        print(f"{dataset:<10}{entry['main']:>7}/18{entry['phase']:>7}/18{entry['total']:>7}/36")
    print(f"{'TOTAL':<10}{total_main:>7}/54{total_phase:>7}/54"
          f"{total_main + total_phase:>7}/108")
    print()
    print("These are descriptive mean comparisons over derived and primitive")
    print("cells, not 108 independent significant results.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "final_18_recompute_check.json"
    destination.write_text(json.dumps({
        "check": "independent recomputation of the v3 tables from row files",
        "source": "ples_paired_rows.csv via data_provider.ples_protocol",
        "max_abs_difference_vs_frozen": worst,
        "tolerance": args.tolerance,
        "descriptive_mean_win_counts": counts,
        "note": "descriptive means over derived and primitive cells; "
                "not independent significance tests",
        "datasets": report,
    }, indent=2), encoding="utf-8")
    print(f"\n-> {destination}")


if __name__ == "__main__":
    main()
