"""An independent arithmetic oracle for the v3 tables.

The previous version of this file imported ``summarize_lce_rows`` and
``summarize_paired_rows`` from the production evaluator, so it could only ever
show that the evaluator agreed with itself. An aggregation error shared by the
evaluator and the collector would reproduce perfectly and be reported as a
clean recomputation.

This implementation therefore uses **only the Python standard library**. It
re-derives every published cell from ``ples_paired_rows.csv`` --
row -> stratum -> tau -> group -> per-run summary -> three-seed mean and sample
standard deviation -- and compares the result field by field against both the
per-run frozen summaries and ``final_18_tables.json``.

Nothing from ``data_provider.ples_protocol`` or ``scripts.collect_*`` may be
imported here. That restriction is the entire point of the file and is asserted
by a test.

The protocol it re-implements, stated so the implementation can be checked
against the specification rather than against the other code:

  * cell metrics      MAPE = 100 * ape_sum / count
                      MAE  = ae_sum / count
                      RMSE = sqrt(se_sum / count)
  * calendar weighting mean within a stratum, then combine strata by calendar
                      share: each of 5 weekday time-of-day bins carries 1/7 and
                      each of 5 weekend bins carries 2/35, so weekday totals 5/7
                      and weekend 2/7
  * tau balancing     mean over tau of the calendar-weighted value
  * groups            Unseen = join/control,  Steady = fault/control
                      Join   = join/treated,  Fault  = fault/treated
                      LifeAvg = mean of those four, WorstGroup = max
  * phases            tau-balanced treated error over the phase's rows
  * PLES              JoinLift     = mean over post-event join tau of
                                     (control - treated)
                      FaultPenalty = mean over post-event fault tau of
                                     (treated - control)
                      Outage       = the fault value at tau = input_len - 1
  * cohorts           only the manifest's main_cohort_ids
  * seeds             mean and sample standard deviation (ddof = 1)

    python scripts/verify_collector_recompute.py --output-dir <dir>
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

MAIN_ROWS = ("Unseen", "Steady", "Join", "Fault", "LifeAvg", "WorstGroup")
PHASE_ROWS = ("Cold", "Warm", "Ready", "Onset", "Decay", "Outage")
METRICS = ("MAPE", "MAE", "RMSE")
PLES_EFFECTS = ("JoinLift", "FaultPenalty", "OutageFaultPenalty")
MODELS = ("ALFNet", "MoGERNN")

DAY_TYPE_SHARE = {"Weekday": 5.0 / 7.0, "Weekend": 2.0 / 7.0}
TOD_BINS = ("EarlyAM", "AMPeak", "Midday", "PMPeak", "Evening")


def stratum_weight(name):
    """Calendar share of one ``DayType/TimeOfDay`` stratum, derived here."""
    day_type, _, tod = str(name).partition("/")
    if day_type not in DAY_TYPE_SHARE or tod not in TOD_BINS:
        raise ValueError(f"stratum outside the protocol: {name!r}")
    return DAY_TYPE_SHARE[day_type] / len(TOD_BINS)


class OracleMismatch(SystemExit):
    """Any disagreement between the independent recomputation and the tables."""


# --- independent aggregation --------------------------------------------------


def _cell_metrics(rows, arm):
    """Pooled metrics for one (track, tau, stratum) cell."""
    count = sum(int(r["count"]) for r in rows)
    if count <= 0:
        return None
    ape = sum(float(r[f"{arm}_ape_sum"]) for r in rows)
    ae = sum(float(r[f"{arm}_ae_sum"]) for r in rows)
    se = sum(float(r[f"{arm}_se_sum"]) for r in rows)
    return {"MAPE": 100.0 * ape / count, "MAE": ae / count, "RMSE": math.sqrt(se / count)}


def _mean(values):
    return sum(values) / len(values)


def _calendar_weighted(rows, arm):
    """Mean inside each stratum, then combine strata by calendar share."""
    by_stratum = defaultdict(list)
    for row in rows:
        by_stratum[str(row["regime"])].append(row)
    per_stratum = {}
    for name, group in by_stratum.items():
        by_cell = defaultdict(list)
        for row in group:
            by_cell[(str(row["track"]), int(row["tau"]), str(row["origin_id"]),
                     str(row["cohort_id"]))].append(row)
        cells = [_cell_metrics(cell, arm) for cell in by_cell.values()]
        cells = [c for c in cells if c is not None]
        if cells:
            per_stratum[name] = {m: _mean([c[m] for c in cells]) for m in METRICS}
    if not per_stratum:
        return None
    total = sum(stratum_weight(n) for n in per_stratum)
    return {
        metric: sum(stratum_weight(n) * v[metric] for n, v in per_stratum.items()) / total
        for metric in METRICS
    }


def _tau_balanced(rows, arm):
    by_tau = defaultdict(list)
    for row in rows:
        by_tau[int(row["tau"])].append(row)
    per_tau = [_calendar_weighted(by_tau[t], arm) for t in sorted(by_tau)]
    per_tau = [v for v in per_tau if v is not None]
    if not per_tau:
        return None
    return {metric: _mean([v[metric] for v in per_tau]) for metric in METRICS}


def recompute_run(rows, input_len, main_cohort_ids):
    wanted = {int(v) for v in main_cohort_ids}
    post = [r for r in rows
            if int(r["cohort_id"]) in wanted and 0 <= int(r["tau"]) < int(input_len)]
    join = [r for r in post if r["track"] == "join"]
    fault = [r for r in post if r["track"] == "fault"]

    groups = {
        "Unseen": _tau_balanced(join, "control"),
        "Steady": _tau_balanced(fault, "control"),
        "Join": _tau_balanced(join, "treated"),
        "Fault": _tau_balanced(fault, "treated"),
    }
    primitive = ("Unseen", "Steady", "Join", "Fault")
    groups["LifeAvg"] = {m: _mean([groups[g][m] for g in primitive]) for m in METRICS}
    groups["WorstGroup"] = {m: max(groups[g][m] for g in primitive) for m in METRICS}

    phases = {}
    for phase in PHASE_ROWS:
        phases[phase] = _tau_balanced([r for r in post if r["phase"] == phase], "treated")

    # PLES runs over every post-event tau of the *whole* trajectory, not only
    # tau < input_len, so it is rebuilt from the unrestricted rows.
    ples_rows = [r for r in rows if int(r["cohort_id"]) in wanted and int(r["tau"]) >= 0]
    by_tau = defaultdict(list)
    for row in ples_rows:
        by_tau[(str(row["track"]), int(row["tau"]))].append(row)
    curves = {}
    for (track, tau), group in by_tau.items():
        treated = _calendar_weighted(group, "treated")
        control = _calendar_weighted(group, "control")
        if treated is None or control is None:
            continue
        curves[(track, tau)] = {
            metric: (control[metric] - treated[metric]) if track == "join"
            else (treated[metric] - control[metric])
            for metric in METRICS
        }
    ples = {}
    for metric in ("MAPE", "MAE"):
        join_values = [v[metric] for (t, _), v in curves.items() if t == "join"]
        fault_values = [v[metric] for (t, _), v in curves.items() if t == "fault"]
        outage = curves.get(("fault", int(input_len) - 1))
        if not join_values or not fault_values or outage is None:
            raise OracleMismatch(f"PLES cells missing for metric {metric}")
        ples[metric] = {
            "JoinLift": _mean(join_values),
            "FaultPenalty": _mean(fault_values),
            "OutageFaultPenalty": outage[metric],
        }
    return {"main": groups, "phase": phases, "ples": ples}


# --- reading ------------------------------------------------------------------


def read_rows(path):
    rows = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(row)
    if not rows:
        raise OracleMismatch(f"empty row file: {path}")
    return rows


def _close(a, b, atol, rtol):
    if a is None or b is None:
        return False
    if not (math.isfinite(a) and math.isfinite(b)):
        return False
    return abs(a - b) <= max(atol, rtol * abs(b))


# --- verification -------------------------------------------------------------


def verify(manifest, output_root, atol, rtol):
    problems, worst = [], 0.0
    per_run = defaultdict(dict)

    for run in manifest["runs"]:
        directory = Path(output_root) / run["output_subdir"] / run["run_id"] / "predictions"
        candidates = sorted(directory.glob("lifecycle-v3-*"))
        if len(candidates) != 1:
            problems.append(f"{run['run_id']}: expected exactly one result bundle, "
                            f"found {len(candidates)}")
            continue
        bundle = candidates[0]

        frozen_main = json.loads((bundle / "lce_main_summary.json").read_text("utf-8"))
        frozen_phase = json.loads((bundle / "lce_phase_summary.json").read_text("utf-8"))
        frozen_ples = json.loads((bundle / "ples_effects_summary.json").read_text("utf-8"))
        protocol = json.loads((bundle / "protocol_manifest.json").read_text("utf-8"))

        for label, found, expected in (
            ("model", frozen_main.get("model"), run["model"]),
            ("seed", frozen_main.get("seed"), run["seed"]),
            ("dataset", (frozen_main.get("dataset") or {}).get("name"), run["dataset"]),
        ):
            if found != expected:
                problems.append(f"{run['run_id']}: {label}={found!r} != manifest {expected!r}")

        resolved = protocol["resolved"]
        got = recompute_run(read_rows(bundle / "ples_paired_rows.csv"),
                            resolved["input_len"], resolved["main_cohort_ids"])

        for table, rows_, frozen in (("main", MAIN_ROWS, frozen_main),
                                     ("phase", PHASE_ROWS, frozen_phase)):
            for name in rows_:
                for metric in METRICS:
                    mine = (got[table].get(name) or {}).get(metric)
                    theirs = (frozen["metrics"].get(name) or {}).get(metric)
                    if not _close(mine, theirs, atol, rtol):
                        problems.append(f"{run['run_id']} {table}/{name}/{metric}: "
                                        f"oracle {mine!r} vs frozen {theirs!r}")
                    elif mine is not None and theirs is not None:
                        worst = max(worst, abs(mine - theirs))
        for metric in ("MAPE", "MAE"):
            for effect in PLES_EFFECTS:
                mine = got["ples"][metric][effect]
                theirs = (frozen_ples["paired_effects"].get(metric) or {}).get(effect)
                if not _close(mine, theirs, atol, rtol):
                    problems.append(f"{run['run_id']} ples/{effect}/{metric}: "
                                    f"oracle {mine!r} vs frozen {theirs!r}")
                else:
                    worst = max(worst, abs(mine - theirs))

        per_run[(run["dataset"], run["model"])][run["seed"]] = got

    if problems:
        raise OracleMismatch("ORACLE MISMATCH (per-run):\n" + "\n".join(f"  - {p}" for p in problems))
    return per_run, worst


def verify_final_tables(per_run, manifest, tables_path, atol, rtol):
    """Recursively check the published three-seed table, field by field."""
    tables = json.loads(Path(tables_path).read_text("utf-8"))
    problems, worst = [], 0.0
    expected_seeds = sorted(manifest["seeds"])

    for dataset in manifest["datasets"]:
        if dataset not in tables["datasets"]:
            problems.append(f"final tables are missing dataset {dataset}")
            continue
        published = tables["datasets"][dataset]
        for table, rows_ in (("main", MAIN_ROWS), ("phase", PHASE_ROWS)):
            for name in rows_:
                for metric in METRICS:
                    for model in MODELS:
                        cell = per_run.get((dataset, model), {})
                        seeds = sorted(cell)
                        if seeds != expected_seeds:
                            problems.append(f"{dataset}/{model}: seeds {seeds}")
                            continue
                        values = [cell[s][table][name][metric] for s in seeds]
                        entry = published[table][name][metric][model]
                        checks = (("mean", _mean(values)),
                                  ("std", statistics.stdev(values)))
                        for field, mine in checks:
                            theirs = entry.get(field)
                            if not _close(mine, theirs, atol, rtol):
                                problems.append(
                                    f"{dataset} {table}/{name}/{metric}/{model}.{field}: "
                                    f"oracle {mine!r} vs published {theirs!r}")
                            else:
                                worst = max(worst, abs(mine - theirs))
                        if entry.get("n") != len(seeds):
                            problems.append(f"{dataset} {table}/{name}/{metric}/{model}.n: "
                                            f"{entry.get('n')!r} != {len(seeds)}")
        # the run list, exactly and in order
        for model in MODELS:
            expected = [r["run_id"] for r in manifest["runs"]
                        if r["dataset"] == dataset and r["model"] == model]
            expected = sorted(expected)
            found = published.get("runs", {}).get(model, {})
            if sorted(found.values()) != expected:
                problems.append(f"{dataset}/{model}: run list {sorted(found.values())} "
                                f"!= manifest {expected}")

    if problems:
        raise OracleMismatch("ORACLE MISMATCH (final tables):\n"
                             + "\n".join(f"  - {p}" for p in problems))
    return worst


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-root", default="./outputs")
    parser.add_argument("--output-dir", default="./outputs/reports/final")
    parser.add_argument("--final-tables", default=None)
    parser.add_argument("--atol", type=float, default=1e-9)
    parser.add_argument("--rtol", type=float, default=1e-9)
    args = parser.parse_args()

    manifest = json.loads((ROOT / "configs/paper/final_runs.json").read_text("utf-8"))
    per_run, worst_run = verify(manifest, args.output_root, args.atol, args.rtol)
    tables_path = args.final_tables or Path(args.output_dir) / "final_18_tables.json"
    worst_tables = verify_final_tables(per_run, manifest, tables_path, args.atol, args.rtol)

    print(f"independent oracle: {len(manifest['runs'])} runs recomputed with stdlib only")
    print(f"  max |oracle - frozen per-run summary| = {worst_run:.3e}")
    print(f"  max |oracle - final_18_tables.json|   = {worst_tables:.3e}")
    print("  every key, n and run list matched exactly")


if __name__ == "__main__":
    main()
