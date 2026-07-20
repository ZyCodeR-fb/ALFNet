import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


PAIR_KEYS = ("origin_id", "cohort_id", "track", "tau", "phase")
SUM_FIELDS = (
    "treated_ape_sum",
    "control_ape_sum",
    "treated_ae_sum",
    "control_ae_sum",
    "treated_se_sum",
    "control_se_sum",
)
PHASE_COLORS = {
    "Cold": "#DCEAF4",
    "Warm": "#F4DFB5",
    "Ready": "#DCEBDC",
    "Onset": "#F1D5C2",
    "Decay": "#E8D8C7",
    "Outage": "#DDD9E8",
}


def _resolve_result_path(value, filename):
    path = Path(value)
    return path / filename if path.is_dir() else path


def _read_summary(value):
    path = _resolve_result_path(value, "paired_event_summary.json")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle), path


def _read_rows(summary_path):
    path = summary_path.parent / "paired_event_rows.csv"
    rows = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            row = dict(raw)
            for key in ("origin_id", "cohort_id", "tau", "count"):
                row[key] = int(row[key])
            for key in SUM_FIELDS:
                row[key] = float(row[key])
            rows.append(row)
    if not rows:
        raise ValueError(f"PLES paired rows are empty: {path}")
    return rows


def _index_rows(rows):
    indexed = {}
    for row in rows:
        key = tuple(row[field] for field in PAIR_KEYS)
        if key in indexed:
            raise ValueError(f"Duplicate PLES paired row: {key}")
        indexed[key] = row
    return indexed


def _aggregate(rows):
    count = sum(int(row["count"]) for row in rows)
    if count <= 0:
        raise ValueError("PLES comparison cell has no valid targets")
    totals = {field: sum(float(row[field]) for row in rows) for field in SUM_FIELDS}
    treated_mape = 100.0 * totals["treated_ape_sum"] / count
    control_mape = 100.0 * totals["control_ape_sum"] / count
    return {
        "count": count,
        "treated_mape": treated_mape,
        "control_mape": control_mape,
        "ace_mape": treated_mape - control_mape,
    }


def compare_rows(ours_rows, baseline_rows):
    ours_index = _index_rows(ours_rows)
    baseline_index = _index_rows(baseline_rows)
    if set(ours_index) != set(baseline_index):
        raise ValueError("PLES model exports do not contain the same origin/cohort/track/tau cells")
    for key in ours_index:
        if ours_index[key]["count"] != baseline_index[key]["count"]:
            raise ValueError(f"PLES valid-target counts differ for paired cell {key}")

    grouped = defaultdict(lambda: {"ours": [], "baseline": []})
    for key in sorted(ours_index):
        row = ours_index[key]
        cell = grouped[(row["track"], row["tau"], row["phase"])]
        cell["ours"].append(row)
        cell["baseline"].append(baseline_index[key])

    curves = []
    for (track, tau, phase), group in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        ours = _aggregate(group["ours"])
        baseline = _aggregate(group["baseline"])
        curves.append(
            {
                "track": track,
                "tau": tau,
                "phase": phase,
                "count": ours["count"],
                "ours_ace_mape": ours["ace_mape"],
                "baseline_ace_mape": baseline["ace_mape"],
                "attenuation_gain_mape": baseline["ace_mape"] - ours["ace_mape"],
            }
        )
    return curves


def _track_gains(curves):
    gains = {}
    for track, metric_name in (("join", "Join"), ("fault", "Fault")):
        values = [row["attenuation_gain_mape"] for row in curves if row["track"] == track and row["tau"] >= 0]
        if not values:
            raise ValueError(f"PLES comparison has no post-event {track} cells")
        gains[metric_name] = float(np.mean(values))
    gains["LifeMacro"] = float(np.mean([gains["Join"], gains["Fault"]]))
    return gains


def bootstrap_gains(ours_rows, baseline_rows, samples=2000, seed=2026):
    point = _track_gains(compare_rows(ours_rows, baseline_rows))
    if samples <= 0:
        return {key: {"point": value, "low": None, "high": None} for key, value in point.items()}

    ours_units = defaultdict(lambda: defaultdict(list))
    baseline_units = defaultdict(lambda: defaultdict(list))
    for row in ours_rows:
        ours_units[row["origin_id"]][row["cohort_id"]].append(row)
    for row in baseline_rows:
        baseline_units[row["origin_id"]][row["cohort_id"]].append(row)
    if set(ours_units) != set(baseline_units):
        raise ValueError("PLES model exports do not share the same bootstrap origins")

    rng = np.random.default_rng(seed)
    origin_ids = sorted(ours_units)
    draws = {key: np.empty(samples, dtype=np.float64) for key in point}
    for sample_index in range(samples):
        sampled_ours = []
        sampled_baseline = []
        for sampled_origin_id, origin_id in enumerate(
            rng.choice(origin_ids, size=len(origin_ids), replace=True),
            start=1,
        ):
            cohort_ids = sorted(ours_units[int(origin_id)])
            for sampled_cohort_id, cohort_id in enumerate(
                rng.choice(cohort_ids, size=len(cohort_ids), replace=True),
                start=1,
            ):
                for row in ours_units[int(origin_id)][int(cohort_id)]:
                    sampled_ours.append(
                        {**row, "origin_id": sampled_origin_id, "cohort_id": sampled_cohort_id}
                    )
                for row in baseline_units[int(origin_id)][int(cohort_id)]:
                    sampled_baseline.append(
                        {**row, "origin_id": sampled_origin_id, "cohort_id": sampled_cohort_id}
                    )
        gains = _track_gains(compare_rows(sampled_ours, sampled_baseline))
        for key, value in gains.items():
            draws[key][sample_index] = value
    return {
        key: {
            "point": point[key],
            "low": float(np.percentile(values, 2.5)),
            "high": float(np.percentile(values, 97.5)),
        }
        for key, values in draws.items()
    }


def _write_curves(path, curves):
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(curves[0]))
        writer.writeheader()
        writer.writerows(curves)


def _phase_segments(rows):
    segments = []
    start = rows[0]["tau"]
    phase = rows[0]["phase"]
    previous = start
    for row in rows[1:]:
        if row["phase"] != phase:
            segments.append((start, previous, phase))
            start = row["tau"]
            phase = row["phase"]
        previous = row["tau"]
    segments.append((start, previous, phase))
    return segments


def plot_comparison(curves, output_path, ours_label, baseline_label, dpi=260):
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8))
    for axis, track, title in zip(axes, ("join", "fault"), ("Sensor joining", "Sensor fault")):
        rows = sorted((row for row in curves if row["track"] == track), key=lambda row: row["tau"])
        tau = np.asarray([row["tau"] for row in rows])
        ours = np.asarray([row["ours_ace_mape"] for row in rows])
        baseline = np.asarray([row["baseline_ace_mape"] for row in rows])
        y_min = float(min(0.0, ours.min(), baseline.min()))
        y_max = float(max(0.0, ours.max(), baseline.max()))
        axis.plot(tau, np.zeros_like(ours), color="#555555", linewidth=0.9)
        axis.plot([0, 0], [y_min, y_max], color="#222222", linestyle="--", linewidth=0.9)
        axis.plot(tau, baseline, color="#8A8A8A", linewidth=2.0, label=baseline_label)
        axis.plot(tau, ours, color="#176C5B", linewidth=2.4, label=ours_label)
        axis.fill_between(tau, ours, baseline, where=baseline >= ours, color="#70AE99", alpha=0.22)
        for start, end, phase in _phase_segments(rows):
            axis.axvspan(start, end + 1, color=PHASE_COLORS.get(phase, "#EEEEEE"), alpha=0.18)
        axis.set_title(title, fontsize=13, weight="bold")
        axis.set_xlabel("Event-relative time (steps)")
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.7)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    axes[0].set_ylabel("ACE MAPE (percentage points)")
    axes[1].legend(frameon=False, loc="best")
    fig.suptitle("PLES Counterfactual Attenuation", fontsize=16, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Compare two models under one frozen PLES manifest.")
    parser.add_argument("--ours", required=True, help="PLES result directory or paired_event_summary.json")
    parser.add_argument("--baseline", required=True, help="PLES result directory or paired_event_summary.json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--ours-label", default="ALFNet")
    parser.add_argument("--baseline-label", default="MoGERNN")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    args = parser.parse_args()

    ours_summary, ours_path = _read_summary(args.ours)
    baseline_summary, baseline_path = _read_summary(args.baseline)
    if ours_summary["protocol_hash"] != baseline_summary["protocol_hash"]:
        raise ValueError("PLES comparison requires identical protocol_manifest.json hashes")
    ours_rows = _read_rows(ours_path)
    baseline_rows = _read_rows(baseline_path)
    curves = compare_rows(ours_rows, baseline_rows)
    intervals = bootstrap_gains(
        ours_rows,
        baseline_rows,
        samples=args.bootstrap_samples,
        seed=args.bootstrap_seed,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_curves(output_dir / "paired_model_comparison.csv", curves)
    summary = {
        "protocol": "Paired Lifecycle Event Study",
        "protocol_hash": ours_summary["protocol_hash"],
        "definition": "AttenuationGain = ACE_baseline - ACE_ours; positive means ours is better",
        "ours": args.ours_label,
        "baseline": args.baseline_label,
        "attenuation_gain": intervals,
    }
    with (output_dir / "paired_model_comparison_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    plot_comparison(
        curves,
        output_dir / "paired_model_comparison.png",
        args.ours_label,
        args.baseline_label,
    )
    for key in ("Join", "Fault", "LifeMacro"):
        interval = intervals[key]
        if interval["low"] is None or interval["high"] is None:
            print(f"{key} AttenuationGain={interval['point']:.4f} | 95% CI=NA")
        else:
            print(
                f"{key} AttenuationGain={interval['point']:.4f} | "
                f"95% CI=[{interval['low']:.4f}, {interval['high']:.4f}]"
            )


if __name__ == "__main__":
    main()
