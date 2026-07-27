import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_provider.ples_protocol import (  # noqa: E402
    PLES_PROTOCOL_VERSION,
    overlap_blocks,
    trajectory_span_from_manifest,
)


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


def _read_rows(summary_path, filename="paired_event_rows.csv"):
    path = summary_path.parent / filename
    rows = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            row = dict(raw)
            for key in ("origin_id", "cohort_id", "tau", "count"):
                if row.get(key) not in {None, ""}:
                    row[key] = int(row[key])
            for key in SUM_FIELDS:
                if row.get(key) not in {None, ""}:
                    row[key] = float(row[key])
            rows.append(row)
    if not rows:
        raise ValueError(f"PLES paired rows are empty: {path}")
    return rows


def _read_result_bundle(value):
    path = Path(value)
    if path.is_dir():
        v2_path = path / "ples_effects_summary.json"
        if v2_path.exists():
            summary_path = v2_path
            version = 2
        else:
            summary_path = path / "paired_event_summary.json"
            version = 1
    else:
        summary_path = path
        version = 2 if path.name == "ples_effects_summary.json" else 1
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    version = int(summary.get("schema_version", version))
    # v2 and v3 both write ples_paired_rows.csv; only the original v1 export
    # used the older filename.
    rows_filename = "paired_event_rows.csv" if version < 2 else "ples_paired_rows.csv"
    return summary, summary_path, _read_rows(summary_path, rows_filename), version


def _trajectory_span(summary, summary_path):
    """PLES trajectory length as declared by the run that produced the rows.

    Read the effects summary first, then the manifest written beside it. Older
    exports carry the span in neither, and silently treating every origin as an
    independent draw is exactly the interval-shrinking mistake this function
    exists to prevent -- so that case raises instead of degrading. The
    reconstruction uses pred_len for the horizon, not a second copy of
    input_len, which is only equivalent when the two happen to match.
    """
    declared = summary.get("bootstrap")
    if isinstance(declared, dict) and declared.get("trajectory_span"):
        return int(declared["trajectory_span"])
    manifest_path = Path(summary_path).parent / "protocol_manifest.json"
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        span = trajectory_span_from_manifest(manifest)
        if span:
            return int(span)
    raise ValueError(
        f"{summary_path} declares no PLES trajectory span, and no manifest beside "
        "it supplies one. Refusing to fall back to independent per-origin "
        "resampling, which would report intervals that are too narrow."
    )


def _paired_trajectory_span(ours, ours_path, baseline, baseline_path):
    """Both sides must agree on the resampling unit, and both must declare it.

    Resolving only one side and assuming the other matches is not safe: the
    protocol hash covers the evaluation design, not the horizon a particular
    export was written with, so two bundles can share a hash and still disagree.
    """
    ours_span = _trajectory_span(ours, ours_path)
    baseline_span = _trajectory_span(baseline, baseline_path)
    if ours_span != baseline_span:
        raise ValueError(
            f"PLES trajectory spans differ between the two models: "
            f"{ours_span} ({ours_path}) vs {baseline_span} ({baseline_path}); "
            "the paired bootstrap has no common resampling unit"
        )
    return ours_span


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


def _cell_mape_v2(row, prefix):
    count = int(row["count"])
    if count <= 0:
        raise ValueError("PLES v3 comparison cell has no valid targets")
    return 100.0 * float(row[f"{prefix}_ape_sum"]) / count


def _calendar_balanced_mape_v2(rows, prefix):
    by_regime = defaultdict(list)
    for row in rows:
        by_regime[str(row.get("regime", "all"))].append(_cell_mape_v2(row, prefix))
    if not by_regime:
        raise ValueError("PLES v3 comparison has no calendar cells")
    return float(np.mean([np.mean(values) for values in by_regime.values()]))


def _effect_curves_v2(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(str(row["track"]), int(row["tau"]), str(row["phase"]))].append(row)
    curves = []
    for (track, tau, phase), cell_rows in sorted(
        grouped.items(), key=lambda item: (item[0][0], item[0][1])
    ):
        treated = _calendar_balanced_mape_v2(cell_rows, "treated")
        control = _calendar_balanced_mape_v2(cell_rows, "control")
        if track == "join":
            effect_name = "JoinLift"
            effect = control - treated
        elif track == "fault":
            effect_name = "FaultPenalty"
            effect = treated - control
        else:
            raise ValueError(f"Unsupported PLES v3 track: {track}")
        curves.append(
            {
                "track": track,
                "tau": tau,
                "phase": phase,
                "effect_name": effect_name,
                "effect_mape": effect,
            }
        )
    return curves


def compare_rows_v2(ours_rows, baseline_rows):
    ours_index = _index_rows(ours_rows)
    baseline_index = _index_rows(baseline_rows)
    if set(ours_index) != set(baseline_index):
        raise ValueError("PLES v3 model exports do not contain the same paired cells")
    for key in ours_index:
        if ours_index[key]["count"] != baseline_index[key]["count"]:
            raise ValueError(f"PLES v3 valid-target counts differ for paired cell {key}")

    ours_curves = {
        (row["track"], row["tau"], row["phase"]): row
        for row in _effect_curves_v2(ours_rows)
    }
    baseline_curves = {
        (row["track"], row["tau"], row["phase"]): row
        for row in _effect_curves_v2(baseline_rows)
    }
    if set(ours_curves) != set(baseline_curves):
        raise ValueError("PLES v3 model curves do not share the same track/tau cells")

    curves = []
    for key in sorted(ours_curves, key=lambda value: (value[0], value[1])):
        ours = ours_curves[key]
        baseline = baseline_curves[key]
        if ours["effect_name"] == "JoinLift":
            advantage = ours["effect_mape"] - baseline["effect_mape"]
        else:
            advantage = baseline["effect_mape"] - ours["effect_mape"]
        curves.append(
            {
                **ours,
                "ours_effect_mape": ours["effect_mape"],
                "baseline_effect_mape": baseline["effect_mape"],
                "advantage_mape": advantage,
            }
        )
    return curves


def _advantages_v2(curves):
    post = [row for row in curves if int(row["tau"]) >= 0]
    join = [row for row in post if row["effect_name"] == "JoinLift"]
    fault = [row for row in post if row["effect_name"] == "FaultPenalty"]
    if not join or not fault:
        raise ValueError("PLES v3 comparison requires post-event Join and Fault cells")
    return {
        "JoinLiftGain": float(np.mean([row["advantage_mape"] for row in join])),
        "FaultPenaltyReduction": float(np.mean([row["advantage_mape"] for row in fault])),
        "OutageFaultPenaltyReduction": float(
            max(row["baseline_effect_mape"] for row in fault)
            - max(row["ours_effect_mape"] for row in fault)
        ),
    }


def bootstrap_advantages_v2(ours_rows, baseline_rows, samples=2000, seed=2026,
                            trajectory_span=None):
    """Paired advantage intervals on the PLES v3 resampling unit.

    This used to draw each origin independently and nest the cohort draw inside
    it. Both parts disagreed with the design the manifest declares: adjacent
    origins share trajectory data, so they are not independent draws, and the
    frozen cohorts are replayed at every origin, so they are crossed with
    origins rather than nested in them. Independent origins shrink the interval;
    nesting a crossed factor distorts it.
    """
    point = _advantages_v2(compare_rows_v2(ours_rows, baseline_rows))
    if samples <= 0:
        return {name: {"point": value, "low": None, "high": None} for name, value in point.items()}

    ours_units = defaultdict(lambda: defaultdict(list))
    baseline_units = defaultdict(lambda: defaultdict(list))
    origin_times, origin_stratum = {}, {}
    for row in ours_rows:
        origin = int(row["origin_id"])
        ours_units[origin][int(row["cohort_id"])].append(row)
        origin_times[origin] = int(row.get("event_time_index", origin))
        origin_stratum[origin] = str(row.get("regime", "all")).split("/")[0]
    for row in baseline_rows:
        baseline_units[int(row["origin_id"])][int(row["cohort_id"])].append(row)
    if set(ours_units) != set(baseline_units):
        raise ValueError("PLES v3 model exports do not share bootstrap origins")

    if not trajectory_span:
        raise ValueError(
            "bootstrap_advantages_v2 needs the PLES trajectory span; without it "
            "each origin would be resampled independently even though adjacent "
            "origins share data, and the interval would come out too narrow"
        )
    assignment = overlap_blocks(origin_times, trajectory_span)
    blocks = defaultdict(list)
    for origin, block in assignment.items():
        blocks[block].append(origin)
    block_stratum = {block: origin_stratum[min(members)] for block, members in blocks.items()}
    blocks_by_stratum = defaultdict(list)
    for block, stratum in sorted(block_stratum.items()):
        blocks_by_stratum[stratum].append(block)

    rng = np.random.default_rng(seed)
    cohort_ids = sorted({int(row["cohort_id"]) for row in ours_rows})
    draws = {name: np.empty(int(samples), dtype=np.float64) for name in point}
    for sample_index in range(int(samples)):
        sampled_ours = []
        sampled_baseline = []
        # One global cohort draw per sample, applied to every sampled block.
        cohort_repeats = defaultdict(int)
        for cohort_id in rng.choice(cohort_ids, size=len(cohort_ids), replace=True):
            cohort_repeats[int(cohort_id)] += 1
        slot = 0
        for _stratum, stratum_blocks in sorted(blocks_by_stratum.items()):
            for block in rng.choice(stratum_blocks, size=len(stratum_blocks), replace=True):
                for origin_id in blocks[int(block)]:
                    slot += 1
                    for cohort_id, repeats in sorted(cohort_repeats.items()):
                        for copy in range(repeats):
                            replacement_ids = {
                                "origin_id": slot,
                                "cohort_id": cohort_id * 1000 + copy,
                            }
                            sampled_ours.extend(
                                {**row, **replacement_ids}
                                for row in ours_units[origin_id][cohort_id]
                            )
                            sampled_baseline.extend(
                                {**row, **replacement_ids}
                                for row in baseline_units[origin_id][cohort_id]
                            )
        values = _advantages_v2(compare_rows_v2(sampled_ours, sampled_baseline))
        for name, value in values.items():
            draws[name][sample_index] = value
    return {
        name: {
            "point": point[name],
            "low": float(np.percentile(values, 2.5)),
            "high": float(np.percentile(values, 97.5)),
        }
        for name, values in draws.items()
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


def plot_comparison_v2(curves, output_path, ours_label, baseline_label, dpi=260):
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8))
    panels = (
        ("join", "Sensor joining", "JoinLift (MAPE points)", True),
        ("fault", "Sensor fault", "FaultPenalty (MAPE points)", False),
    )
    for axis, (track, title, ylabel, higher_is_better) in zip(axes, panels):
        rows = sorted((row for row in curves if row["track"] == track), key=lambda row: row["tau"])
        tau = np.asarray([row["tau"] for row in rows])
        ours = np.asarray([row["ours_effect_mape"] for row in rows])
        baseline = np.asarray([row["baseline_effect_mape"] for row in rows])
        y_min = float(min(0.0, ours.min(), baseline.min()))
        y_max = float(max(0.0, ours.max(), baseline.max()))
        axis.axhline(0.0, color="#555555", linewidth=0.9)
        axis.axvline(0.0, color="#222222", linestyle="--", linewidth=0.9)
        axis.plot(tau, baseline, color="#8A8A8A", linewidth=2.0, label=baseline_label)
        axis.plot(tau, ours, color="#176C5B", linewidth=2.4, label=ours_label)
        better = ours >= baseline if higher_is_better else ours <= baseline
        axis.fill_between(tau, ours, baseline, where=better, color="#70AE99", alpha=0.22)
        for start, end, phase in _phase_segments(rows):
            axis.axvspan(start, end + 1, color=PHASE_COLORS.get(phase, "#EEEEEE"), alpha=0.18)
        axis.set_ylim(y_min, y_max if y_max > y_min else y_min + 1.0)
        axis.set_title(title, fontsize=13, weight="bold")
        axis.set_xlabel("Event-relative time (steps)")
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.7)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    axes[1].legend(frameon=False, loc="best")
    fig.suptitle("PLES v3 Paired Lifecycle Effects", fontsize=16, weight="bold")
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

    ours_summary, ours_path, ours_rows, ours_version = _read_result_bundle(args.ours)
    baseline_summary, baseline_path, baseline_rows, baseline_version = _read_result_bundle(args.baseline)
    if ours_version != baseline_version:
        raise ValueError("PLES comparison requires both models to use the same protocol version")
    if ours_summary["protocol_hash"] != baseline_summary["protocol_hash"]:
        raise ValueError("PLES comparison requires identical protocol_manifest.json hashes")
    if ours_version < PLES_PROTOCOL_VERSION:
        raise ValueError(
            f"PLES schema v{ours_version} predictions predate the maintained protocol "
            f"(v{PLES_PROTOCOL_VERSION}); re-run inference instead of replotting them"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if ours_version >= 2:
        curves = compare_rows_v2(ours_rows, baseline_rows)
        intervals = bootstrap_advantages_v2(
            ours_rows,
            baseline_rows,
            samples=args.bootstrap_samples,
            seed=args.bootstrap_seed,
            trajectory_span=_paired_trajectory_span(
                ours_summary, ours_path, baseline_summary, baseline_path
            ),
        )
        summary = {
            "schema_version": 3,
            "protocol": "PLES v3 paired model comparison",
            "protocol_hash": ours_summary["protocol_hash"],
            "definitions": {
                "JoinLiftGain": "JoinLift_ours - JoinLift_baseline; higher is better",
                "FaultPenaltyReduction": "FaultPenalty_baseline - FaultPenalty_ours; higher is better",
                "OutageFaultPenaltyReduction": "OutageFaultPenalty_baseline - OutageFaultPenalty_ours; higher is better",
            },
            "ours": args.ours_label,
            "baseline": args.baseline_label,
            "advantages": intervals,
        }
        plot_comparison_v2(
            curves,
            output_dir / "paired_model_comparison.png",
            args.ours_label,
            args.baseline_label,
        )
        print_keys = tuple(intervals)
    else:
        curves = compare_rows(ours_rows, baseline_rows)
        intervals = bootstrap_gains(
            ours_rows,
            baseline_rows,
            samples=args.bootstrap_samples,
            seed=args.bootstrap_seed,
        )
        summary = {
            "protocol": "Paired Lifecycle Event Study",
            "protocol_hash": ours_summary["protocol_hash"],
            "definition": "AttenuationGain = ACE_baseline - ACE_ours; positive means ours is better",
            "ours": args.ours_label,
            "baseline": args.baseline_label,
            "attenuation_gain": intervals,
        }
        plot_comparison(
            curves,
            output_dir / "paired_model_comparison.png",
            args.ours_label,
            args.baseline_label,
        )
        print_keys = ("Join", "Fault", "LifeMacro")
    _write_curves(output_dir / "paired_model_comparison.csv", curves)
    with (output_dir / "paired_model_comparison_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    for key in print_keys:
        interval = intervals[key]
        label = key if ours_version >= 2 else f"{key} AttenuationGain"
        if interval["low"] is None or interval["high"] is None:
            print(f"{label}={interval['point']:.4f} | 95% CI=NA")
        else:
            print(
                f"{label}={interval['point']:.4f} | "
                f"95% CI=[{interval['low']:.4f}, {interval['high']:.4f}]"
            )


if __name__ == "__main__":
    main()
