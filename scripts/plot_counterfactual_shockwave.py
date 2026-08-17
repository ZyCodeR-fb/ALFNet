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

from scripts.analysis_identity import (  # noqa: E402
    assert_shared_design,
    load_manifest,
    verify_role,
)
from data_provider.ples_protocol import (  # noqa: E402
    PLES_PROTOCOL_VERSION,
    build_curve_rows,
    main_cohort_rows,
    overlap_blocks,
    paired_effect_arms,
    paired_effect_cells,
    paired_effects,
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


def assert_comparable_bundles(ours, baseline):
    """Refuse a comparison that is not the same experiment on both sides.

    A shared ``protocol_hash`` only says both runs were evaluated under one
    evaluation design. It says nothing about which dataset, seed or model each
    side actually is, so it accepted ALFNet METR-LA seed 32 against MoGERNN
    METR-LA seed **52** -- a comparison that silently mixes a model difference
    with a training-seed difference and attributes the whole thing to the model.

    The two sides must therefore agree on everything the protocol determines and
    differ only in the model.
    """
    problems = []
    for field in ("protocol_hash", "training_protocol_hash", "lce_protocol_hash",
                  "ples_protocol_hash"):
        left, right = ours.get(field), baseline.get(field)
        if left != right:
            problems.append(f"{field}: {left} vs {right}")

    ours_dataset = (ours.get("dataset") or {})
    baseline_dataset = (baseline.get("dataset") or {})
    for field in ("name", "fingerprint", "adjacency_hash", "timestamps_hash",
                  "test_values_hash"):
        left, right = ours_dataset.get(field), baseline_dataset.get(field)
        if left != right:
            problems.append(f"dataset.{field}: {left} vs {right}")

    if ours.get("seed") != baseline.get("seed"):
        problems.append(
            f"seed: {ours.get('seed')} vs {baseline.get('seed')} -- a cross-seed "
            "comparison confounds the model difference with training randomness"
        )
    if ours.get("model") == baseline.get("model"):
        problems.append(
            f"both sides are {ours.get('model')!r}; a paired comparison needs two models"
        )
    if problems:
        raise ValueError(
            "PLES comparison refused; the two bundles are not the same experiment:\n  - "
            + "\n  - ".join(problems)
        )


def _protocol_plan(summary, summary_path):
    """``input_len`` and the main plan's cohort ids, from the frozen manifest.

    Both decide *which cells* an effect is computed over, so neither may be
    guessed. Outage is the fault cell at ``tau = input_len - 1``, and the main
    plan excludes the saturated cohorts. Raise rather than default: a wrong
    input_len silently reports a different tau as Outage, and a missing cohort
    list silently widens every endpoint to cohorts the main table never used.
    """
    manifest_path = Path(summary_path).parent / "protocol_manifest.json"
    if not manifest_path.exists():
        raise ValueError(
            f"PLES comparison needs the frozen manifest beside the summary: {manifest_path}"
        )
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    resolved = manifest.get("resolved") or {}
    input_len = resolved.get("input_len")
    main_cohort_ids = resolved.get("main_cohort_ids")
    if not input_len:
        raise ValueError(f"{manifest_path} declares no resolved.input_len")
    if not main_cohort_ids:
        raise ValueError(f"{manifest_path} declares no resolved.main_cohort_ids")
    return int(input_len), [int(value) for value in main_cohort_ids]


def _paired_protocol_plan(ours_summary, ours_path, baseline_summary, baseline_path):
    ours = _protocol_plan(ours_summary, ours_path)
    baseline = _protocol_plan(baseline_summary, baseline_path)
    if ours != baseline:
        raise ValueError(
            "PLES comparison requires both models to declare the same input_len "
            f"and main cohort plan: {ours} vs {baseline}"
        )
    return ours


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


# Effect direction. JoinLift is a benefit, so a larger value is better and our
# advantage is ours - baseline; both Fault penalties are harms, so a smaller
# value is better and the advantage is baseline - ours.
HIGHER_IS_BETTER = {"JoinLift": True, "FaultPenalty": False, "OutageFaultPenalty": False}
ADVANTAGE_NAME = {
    "JoinLift": "JoinLiftGain",
    "FaultPenalty": "FaultPenaltyReduction",
    "OutageFaultPenalty": "OutageFaultPenaltyReduction",
}


def _curves_for(rows, main_cohort_ids):
    """Frozen curve construction: main-plan cohorts, then calendar weighting.

    Both steps come from data_provider.ples_protocol. This script used to carry
    its own copy of each and all three parts had drifted from the protocol it
    claims to plot: it kept the saturated cohorts the main plan excludes, it
    weighted calendar regimes equally instead of by their 5/7 and 2/7 shares,
    and it read Outage off the maximum of the fault curve.
    """
    return build_curve_rows(main_cohort_rows(rows, main_cohort_ids))


def compare_rows_v2(ours_rows, baseline_rows, main_cohort_ids=None):
    ours_index = _index_rows(ours_rows)
    baseline_index = _index_rows(baseline_rows)
    if set(ours_index) != set(baseline_index):
        raise ValueError("PLES v3 model exports do not contain the same paired cells")
    for key in ours_index:
        if ours_index[key]["count"] != baseline_index[key]["count"]:
            raise ValueError(f"PLES v3 valid-target counts differ for paired cell {key}")

    ours_curves = {
        (row["track"], row["tau"], row["phase"]): row
        for row in _curves_for(ours_rows, main_cohort_ids)
    }
    baseline_curves = {
        (row["track"], row["tau"], row["phase"]): row
        for row in _curves_for(baseline_rows, main_cohort_ids)
    }
    if set(ours_curves) != set(baseline_curves):
        raise ValueError("PLES v3 model curves do not share the same track/tau cells")

    curves = []
    for key in sorted(ours_curves, key=lambda value: (value[0], value[1])):
        ours = ours_curves[key]
        baseline = baseline_curves[key]
        ours_effect = ours["effect"]["MAPE"]
        baseline_effect = baseline["effect"]["MAPE"]
        if HIGHER_IS_BETTER[ours["effect_name"]]:
            advantage = ours_effect - baseline_effect
        else:
            advantage = baseline_effect - ours_effect
        curves.append(
            {
                "track": ours["track"],
                "tau": ours["tau"],
                "phase": ours["phase"],
                "effect_name": ours["effect_name"],
                "effect_mape": ours_effect,
                "ours_effect_mape": ours_effect,
                "baseline_effect_mape": baseline_effect,
                "advantage_mape": advantage,
                "ours_treated_mape": ours["treated"]["MAPE"],
                "ours_control_mape": ours["control"]["MAPE"],
                "baseline_treated_mape": baseline["treated"]["MAPE"],
                "baseline_control_mape": baseline["control"]["MAPE"],
            }
        )
    return curves


def arm_status(delta, interval, margin):
    """Classify one absolute-error arm against its interval and a margin.

    Both arms are absolute errors, so a negative delta (ours - baseline) is an
    improvement. Deciding this on the point estimate alone -- which an earlier
    version did -- cannot separate "the control arm degraded" from "the control
    arm moved by less than the noise", and it classified an exactly-zero delta
    as a degradation. Four outcomes, from the interval:

      improve     the whole interval is better than -margin
      degrade     the whole interval is worse than +margin
      equivalent  the whole interval lies inside [-margin, +margin]
      uncertain   the interval spans more than one of those
    """
    if interval is None:
        return "uncertain"
    low, high = float(interval["low"]), float(interval["high"])
    if not all(np.isfinite(value) for value in (delta, low, high)):
        return "uncertain"
    if high < -margin:
        return "improve"
    if low > margin:
        return "degrade"
    if low >= -margin and high <= margin:
        return "equivalent"
    return "uncertain"


# A contrast may only be credited when neither arm is established as degraded.
NON_DEGRADED = {"improve", "equivalent"}


def pareto_status(treated, control):
    """Joint verdict over the two arm classifications.

    The gate this exists for: a PLES contrast improves when the treated arm gets
    better *or* when the control arm gets worse, and only the first is a better
    model. ``rejected_control_degraded`` is the case a contrast-only reader
    would have scored as a win.
    """
    if treated == "improve" and control in NON_DEGRADED:
        return "accepted_treated_improves"
    if control == "degrade":
        return "rejected_control_degraded"
    if treated == "degrade":
        return "rejected_treated_degraded"
    if treated in NON_DEGRADED and control in NON_DEGRADED:
        return "no_material_change"
    return "uncertain"


def _advantages_v2(ours_rows, baseline_rows, input_len, main_cohort_ids=None):
    """Paired contrast plus both absolute arms, on the frozen cell selection."""
    ours_curves = _curves_for(ours_rows, main_cohort_ids)
    baseline_curves = _curves_for(baseline_rows, main_cohort_ids)
    out = {}
    for metric in ("MAPE", "MAE"):
        ours_effect = paired_effects(ours_curves, input_len, metric)
        baseline_effect = paired_effects(baseline_curves, input_len, metric)
        ours_arms = paired_effect_arms(ours_curves, input_len, metric)
        baseline_arms = paired_effect_arms(baseline_curves, input_len, metric)
        for name in ours_effect:
            advantage = (
                ours_effect[name] - baseline_effect[name]
                if HIGHER_IS_BETTER[name]
                else baseline_effect[name] - ours_effect[name]
            )
            treated_delta = ours_arms[name]["treated"] - baseline_arms[name]["treated"]
            control_delta = ours_arms[name]["control"] - baseline_arms[name]["control"]
            out[f"{ADVANTAGE_NAME[name]}/{metric}"] = {
                "advantage": float(advantage),
                "ours_effect": float(ours_effect[name]),
                "baseline_effect": float(baseline_effect[name]),
                "ours_treated": float(ours_arms[name]["treated"]),
                "ours_control": float(ours_arms[name]["control"]),
                "baseline_treated": float(baseline_arms[name]["treated"]),
                "baseline_control": float(baseline_arms[name]["control"]),
                "treated_delta": float(treated_delta),
                "control_delta": float(control_delta),
                "n_evaluable_cells": len(cells_for(ours_curves, input_len, name)),
            }
    return out


def cells_for(curves, input_len, name):
    return paired_effect_cells(curves, input_len)[name]


DEFAULT_ARM_MARGINS = {"MAE": 0.05, "MAPE": 0.25}


def bootstrap_advantages_v2(ours_rows, baseline_rows, input_len, samples=2000, seed=2026,
                            trajectory_span=None, main_cohort_ids=None,
                            arm_margins=None):
    """Paired advantage intervals on the PLES v3 resampling unit.

    This used to draw each origin independently and nest the cohort draw inside
    it. Both parts disagreed with the design the manifest declares: adjacent
    origins share trajectory data, so they are not independent draws, and the
    frozen cohorts are replayed at every origin, so they are crossed with
    origins rather than nested in them. Independent origins shrink the interval;
    nesting a crossed factor distorts it.
    """
    # Restrict to the main plan's cohorts once, here, and resample inside that
    # set. Two reasons, and the first is not optional: the resampler relabels
    # cohorts as `cohort_id * 1000 + copy` to keep repeated draws distinct, so a
    # cohort-id filter applied downstream of it matches nothing and every draw
    # comes back empty. The second is that the point estimate is a main-plan
    # quantity, so an interval that resampled the saturated cohorts too would
    # not be an interval for it.
    ours_rows = main_cohort_rows(ours_rows, main_cohort_ids)
    baseline_rows = main_cohort_rows(baseline_rows, main_cohort_ids)
    point = _advantages_v2(ours_rows, baseline_rows, input_len)
    # Interval on the contrast and on both arms: a Pareto verdict read off point
    # estimates alone cannot tell "the control arm degraded" from "the control
    # arm moved by less than the noise".
    resampled_fields = ("advantage", "treated_delta", "control_delta")
    arm_margins = DEFAULT_ARM_MARGINS if arm_margins is None else arm_margins
    if samples <= 0:
        # No interval means no arm can be classified; say uncertain rather than
        # falling back to a point-estimate verdict.
        return {
            name: {**entry,
                   **{f"{field}_ci": None for field in resampled_fields},
                   "arm_margin": arm_margin_for(name, arm_margins),
                   "treated_status": "uncertain", "control_status": "uncertain",
                   "pareto": "uncertain"}
            for name, entry in point.items()
        }

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
    draws = {
        (name, field): np.empty(int(samples), dtype=np.float64)
        for name in point
        for field in resampled_fields
    }
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
        values = _advantages_v2(sampled_ours, sampled_baseline, input_len)
        for name, entry in values.items():
            for field in resampled_fields:
                draws[(name, field)][sample_index] = entry[field]
    resolved = {}
    for name, entry in point.items():
        intervals = {
            f"{field}_ci": {
                "low": float(np.percentile(draws[(name, field)], 2.5)),
                "high": float(np.percentile(draws[(name, field)], 97.5)),
            }
            for field in resampled_fields
        }
        margin = arm_margin_for(name, arm_margins)
        treated = arm_status(entry["treated_delta"], intervals["treated_delta_ci"], margin)
        control = arm_status(entry["control_delta"], intervals["control_delta_ci"], margin)
        resolved[name] = {
            **entry, **intervals,
            "arm_margin": margin,
            "treated_status": treated,
            "control_status": control,
            "pareto": pareto_status(treated, control),
        }
    return resolved


def arm_margin_for(name, arm_margins):
    """Margin for this endpoint's metric. MAPE is a percentage, MAE is not."""
    metric = name.rsplit("/", 1)[-1]
    return float(arm_margins.get(metric, arm_margins["MAE"]))


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
    # After the schema check on purpose: identity fields are only meaningful once
    # the schema is one this script understands, and a retired bundle should be
    # refused for being retired rather than for missing a field it never carried.
    #
    # Anchored to configs/paper/final_runs.json rather than to the other side:
    # comparing the two bundles to each other accepted a swap, an arbitrary
    # model string and an ablation standing in for the mainline.
    manifest = load_manifest()
    verify_role(ours_summary, "ours", manifest)
    verify_role(baseline_summary, "baseline", manifest)
    assert_shared_design(ours_summary, baseline_summary)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if ours_version >= 2:
        input_len, main_cohort_ids = _paired_protocol_plan(
            ours_summary, ours_path, baseline_summary, baseline_path
        )
        curves = compare_rows_v2(ours_rows, baseline_rows, main_cohort_ids)
        intervals = bootstrap_advantages_v2(
            ours_rows,
            baseline_rows,
            input_len,
            samples=args.bootstrap_samples,
            seed=args.bootstrap_seed,
            trajectory_span=_paired_trajectory_span(
                ours_summary, ours_path, baseline_summary, baseline_path
            ),
            main_cohort_ids=main_cohort_ids,
        )
        summary = {
            "schema_version": 4,
            "protocol": "PLES v3 paired model comparison",
            "protocol_hash": ours_summary["protocol_hash"],
            "plan": {
                "input_len": input_len,
                "main_cohort_ids": main_cohort_ids,
                "outage_definition": f"FaultPenalty at tau=input_len-1={input_len - 1}",
                "source": "frozen protocol_manifest.json; not redefined here",
            },
            "definitions": {
                "JoinLiftGain": "JoinLift_ours - JoinLift_baseline; higher is better",
                "FaultPenaltyReduction": "FaultPenalty_baseline - FaultPenalty_ours; higher is better",
                "OutageFaultPenaltyReduction": "OutageFaultPenalty_baseline - OutageFaultPenalty_ours; higher is better",
                "treated_delta": "ours_treated - baseline_treated; absolute error, negative is better",
                "control_delta": "ours_control - baseline_control; absolute error, negative is better",
                "treated_status/control_status": "improve | degrade | equivalent | "
                    "uncertain, decided from the arm's 95% CI against arm_margin, "
                    "not from the point estimate",
                "pareto": "joint arm verdict; a contrast is credited only when the "
                          "treated arm improves and the control arm is not degraded",
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
        if ours_version >= 2:
            # v3 reports the contrast and both absolute arms, because a contrast
            # that improved only by degrading its control arm is not a better
            # model and the contrast alone cannot show that.
            confidence = interval["advantage_ci"]
            span = ("95% CI=NA" if confidence is None
                    else f"95% CI=[{confidence['low']:.4f}, {confidence['high']:.4f}]")
            print(
                f"{key}={interval['advantage']:+.4f} | {span} | "
                f"treatedD={interval['treated_delta']:+.4f}({interval['treated_status']}) "
                f"controlD={interval['control_delta']:+.4f}({interval['control_status']}) "
                f"| {interval['pareto']}"
            )
        elif interval["low"] is None or interval["high"] is None:
            print(f"{key} AttenuationGain={interval['point']:.4f} | 95% CI=NA")
        else:
            print(
                f"{key} AttenuationGain={interval['point']:.4f} | "
                f"95% CI=[{interval['low']:.4f}, {interval['high']:.4f}]"
            )


if __name__ == "__main__":
    main()
