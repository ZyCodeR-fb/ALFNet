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
# The status semantics are defined once, in the multiseed script, and imported
# here. This file previously carried a second copy that had already drifted:
# its verdict never read the contrast, so "both arms improved but the contrast
# clearly worsened" still came back as a clean win. Parity tests pin the two
# together.
from scripts.multiseed_paired_inference import (  # noqa: E402
    ALGEBRA_TOLERANCE,
    CONTRAST_SIGN,
    EFFECT_DEFINITION,
    MIN_BOOTSTRAP_SAMPLES,
    NonFiniteValue,
    arms_pareto_status,
    combined_status,
    margin_for,
)
from scripts.paired_model_bootstrap import read_rows as strict_read_rows  # noqa: E402
from scripts.multiseed_paired_inference import _arm_status as _multiseed_arm_status  # noqa: E402
from scripts.multiseed_paired_inference import (  # noqa: E402
    _interval_status as _multiseed_interval_status,
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


def _read_result_bundle(value):
    """Read a bundle's summary, then its rows through the strict loader.

    The schema version is resolved *before* any row is parsed. A retired bundle
    is refused here rather than being fed to the maintained loader, which would
    report a confusing column error instead of "this export predates the
    protocol". There is deliberately no fallback to the retired
    ``paired_event_rows.csv``: a maintained bundle that lacks
    ``ples_paired_rows.csv`` is broken, not old.
    """
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
    if version < PLES_PROTOCOL_VERSION:
        raise ValueError(
            f"PLES schema v{version} predictions predate the maintained protocol "
            f"(v{PLES_PROTOCOL_VERSION}); re-run inference instead of replotting them"
        )
    # One loader for both analysis entry points. This file used to carry its own
    # permissive parser, which skipped the closed-header, finite/non-negative and
    # track/phase checks entirely -- so a corrupted row reached the statistics.
    rows, rows_path = strict_read_rows(summary_path.parent)
    return summary, summary_path, rows, version, rows_path


# Fields the frozen protocol fixes. They are model independent, so a difference
# between the two sides means the rows did not come from one evaluation design.
PAIRED_PROTOCOL_FIELDS = (
    "dataset", "origin_id", "calendar_day", "regime", "event_time_index",
    "event_timestamp", "cohort_id", "cohort_fraction", "cohort_style",
    "track", "tau", "phase", "node_count", "count",
)


def assert_rows_match_bundles(ours_rows, ours_summary, baseline_rows, baseline_summary):
    """The rows must be the runs their summaries claim, and one shared design.

    A strict parser proves each row is well formed. It cannot prove the file
    belongs to this bundle, nor that the two sides describe the same events --
    only the error sums and the model label may differ between them.
    """
    def _identity(rows, summary, side):
        model = summary.get("model")
        dataset = (summary.get("dataset") or {}).get("name")
        for row in rows:
            if row["model"] != model:
                raise ValueError(
                    f"{side}: row model {row['model']!r} != summary model {model!r} "
                    f"[origin_id={row['origin_id']} cohort_id={row['cohort_id']} "
                    f"track={row['track']} tau={row['tau']}]")
            if row["dataset"] != dataset:
                raise ValueError(
                    f"{side}: row dataset {row['dataset']!r} != summary dataset "
                    f"{dataset!r} [origin_id={row['origin_id']}]")

    _identity(ours_rows, ours_summary, "ours")
    _identity(baseline_rows, baseline_summary, "baseline")

    def _indexed(rows, side):
        indexed = {}
        for row in rows:
            key = (row["origin_id"], row["cohort_id"], row["track"], row["tau"])
            if key in indexed:
                raise ValueError(f"{side}: duplicate paired row {key}")
            indexed[key] = row
        return indexed

    ours_index = _indexed(ours_rows, "ours")
    baseline_index = _indexed(baseline_rows, "baseline")
    if set(ours_index) != set(baseline_index):
        missing = sorted(set(ours_index) - set(baseline_index))[:5]
        extra = sorted(set(baseline_index) - set(ours_index))[:5]
        raise ValueError(
            f"the two row files do not describe the same paired cells; "
            f"only in ours {missing}, only in baseline {extra}")

    for key, ours_row in ours_index.items():
        baseline_row = baseline_index[key]
        for field in PAIRED_PROTOCOL_FIELDS:
            if ours_row[field] != baseline_row[field]:
                raise ValueError(
                    f"paired cell {key}: {field} differs between the two models "
                    f"({ours_row[field]!r} vs {baseline_row[field]!r}); the "
                    "protocol fixes this field, so the rows are not one design")


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

    Thin adapter over the multiseed implementation: this file used to carry its
    own copy of the whole status machine and the two drifted, so the single-seed
    one classified the arms without ever reading the contrast. Only the calling
    convention differs here -- an ``{low, high}`` mapping instead of two scalars.
    """
    if interval is None:
        return "uncertain"
    return _multiseed_arm_status(delta, interval["low"], interval["high"], margin)


def contrast_status(advantage, interval, margin):
    """Classify the oriented contrast. Positive advantage favours ALFNet.

    ``advantage`` is already oriented so that larger is better on every
    endpoint, so it is classified as a higher-is-better quantity.
    """
    if interval is None:
        return "uncertain"
    return _multiseed_interval_status(advantage, interval["low"], interval["high"],
                                      margin, True)


# A contrast may only be credited when neither arm is established as degraded.
NON_DEGRADED = {"improve", "equivalent"}


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
            }
    return out


def cells_for(curves, input_len, name):
    return paired_effect_cells(curves, input_len)[name]


COUNT_SEMANTICS = (
    "Descriptive audit of the observed design, not statistical sample sizes. "
    "The bootstrap's independent unit is the overlap-connected origin block, "
    "crossed with a global cohort factor; rows, target elements and tau curves "
    "are not independent samples of each other. "
    "paired_valid_target_element_count_sum adds target elements across the "
    "selected rows and therefore counts one physical episode once per tau and "
    "track, so it is neither an independent sample count nor a degrees-of-"
    "freedom figure. No effective sample size is computed or implied. "
    "The single-seed interval reflects event resampling at one fixed training "
    "seed and excludes training randomness."
)


def _endpoint_row_selector(effect, input_len):
    """The rows one endpoint is computed over -- the same selection as its cells."""
    if effect == "JoinLift":
        return lambda row: row["track"] == "join" and row["tau"] >= 0
    if effect == "FaultPenalty":
        return lambda row: row["track"] == "fault" and row["tau"] >= 0
    if effect == "OutageFaultPenalty":
        return lambda row: row["track"] == "fault" and row["tau"] == input_len - 1
    raise NonFiniteValue(f"no row selector declared for effect {effect!r}")


def _design_counts(rows):
    origins = {row["origin_id"] for row in rows}
    cohorts = {row["cohort_id"] for row in rows}
    episodes = {(row["origin_id"], row["cohort_id"]) for row in rows}
    taus = {row["tau"] for row in rows}
    return {
        "unique_origin_count": len(origins),
        "unique_cohort_count": len(cohorts),
        "unique_origin_cohort_episode_count": len(episodes),
        "unique_tau_count": len(taus),
        "paired_valid_target_element_count_sum": sum(int(row["count"]) for row in rows),
    }


def endpoint_sample_counts(main_rows, curves, input_len):
    """Per-endpoint counts, on exactly the rows and cells that endpoint uses."""
    cells = paired_effect_cells(curves, input_len)
    out = {}
    for effect in EFFECT_DEFINITION:
        selector = _endpoint_row_selector(effect, input_len)
        rows = [row for row in main_rows if selector(row)]
        if not rows:
            raise NonFiniteValue(f"{effect}: no source rows in the main plan")
        group = cells[effect]
        counts = {
            "source_row_count": len(rows),
            **_design_counts(rows),
            "tau_curve_count": len(group),
        }
        # Cross-check the direct row count against the aggregated curves. Two
        # independent paths to the same figure; a disagreement means the counts
        # describe a different selection than the statistic does.
        cell_total = sum(int(curve["cell_count"]) for curve in group)
        target_total = sum(int(curve["target_count"]) for curve in group)
        if cell_total != counts["source_row_count"]:
            raise NonFiniteValue(
                f"{effect}: curve cell_count sum {cell_total} != source rows "
                f"{counts['source_row_count']}")
        if target_total != counts["paired_valid_target_element_count_sum"]:
            raise NonFiniteValue(
                f"{effect}: curve target_count sum {target_total} != summed row "
                f"count {counts['paired_valid_target_element_count_sum']}")
        if effect == "OutageFaultPenalty" and counts["unique_tau_count"] != 1:
            raise NonFiniteValue(
                f"Outage must use only tau={input_len - 1}, found "
                f"{counts['unique_tau_count']} tau values")
        out[effect] = counts
    return out


def top_level_sample_counts(raw_rows, main_rows, trajectory_span):
    """Shared design counts, after main-cohort filtering and before resampling."""
    origin_times = {row["origin_id"]: int(row["event_time_index"]) for row in main_rows}
    blocks = overlap_blocks(origin_times, trajectory_span)
    per_day_type = defaultdict(set)
    for row in main_rows:
        day_type = str(row["regime"]).split("/")[0]
        per_day_type[day_type].add(blocks[row["origin_id"]])

    design = _design_counts(main_rows)
    counts = {
        "input": {
            "raw_row_count_per_model": len(raw_rows),
            "main_plan_row_count_per_model": len(main_rows),
            "excluded_non_main_row_count_per_model": len(raw_rows) - len(main_rows),
        },
        "paired_design": {
            "paired_row_count": len(main_rows),
            "unique_origin_count": design["unique_origin_count"],
            "unique_cohort_count": design["unique_cohort_count"],
            "unique_origin_cohort_episode_count":
                design["unique_origin_cohort_episode_count"],
            "track_count": len({row["track"] for row in main_rows}),
            "tau_value_count": design["unique_tau_count"],
            "min_tau": min(row["tau"] for row in main_rows),
            "max_tau": max(row["tau"] for row in main_rows),
        },
        "bootstrap_units": {
            "unique_origin_count": design["unique_origin_count"],
            "overlap_block_count": len(set(blocks.values())),
            "cohort_count": design["unique_cohort_count"],
            "day_type_strata": sorted(per_day_type),
            "blocks_per_day_type": {k: len(v) for k, v in sorted(per_day_type.items())},
        },
        "targets": {
            "paired_valid_target_element_count_sum":
                design["paired_valid_target_element_count_sum"],
        },
    }
    for group, entries in counts.items():
        for field, value in entries.items():
            if isinstance(value, int) and value < 0:
                raise NonFiniteValue(f"sample_counts.{group}.{field} is negative")
    if counts["input"]["raw_row_count_per_model"] <= 0:
        raise NonFiniteValue("sample_counts: no raw rows")
    if counts["input"]["main_plan_row_count_per_model"] <= 0:
        raise NonFiniteValue("sample_counts: no main-plan rows")
    for field in ("unique_origin_count", "overlap_block_count", "cohort_count"):
        if counts["bootstrap_units"][field] <= 0:
            raise NonFiniteValue(f"sample_counts.bootstrap_units.{field} must be positive")
    return counts


# The six per-model quantities behind every endpoint. They are resampled on the
# same replicate as the comparison so that a raw interval is never produced by a
# second, independent bootstrap.
RAW_FIELDS = ("ours_effect", "baseline_effect",
              "ours_treated", "baseline_treated",
              "ours_control", "baseline_control")
# Which raw pair each comparison quantity is the difference of.
DELTA_SOURCES = {
    "contrast_delta": ("ours_effect", "baseline_effect"),
    "treated_delta": ("ours_treated", "baseline_treated"),
    "control_delta": ("ours_control", "baseline_control"),
}
# ADVANTAGE_NAME maps a PLES effect to its reported name; invert it so a record
# can recover the effect and its orientation.
EFFECT_OF_ADVANTAGE = {value: key for key, value in ADVANTAGE_NAME.items()}

# The endpoint inventory is derived from the frozen effect and metric
# definitions, never from whatever the run happened to produce. Taking
# "expected" from the observed keys makes the completeness check vacuous: a
# dropped endpoint agrees with itself.
EXPECTED_METRICS = ("MAPE", "MAE")
EXPECTED_ENDPOINTS = tuple(
    f"{ADVANTAGE_NAME[effect]}/{metric}"
    for effect in EFFECT_DEFINITION
    for metric in EXPECTED_METRICS
)
# Every point field the current statistics consume.
EXPECTED_POINT_FIELDS = RAW_FIELDS + ("advantage", "treated_delta", "control_delta")


def _fail_inventory(message):
    raise NonFiniteValue(f"PLES inventory: {message}")


def assert_point_inventory(point):
    """Exact endpoint and field sets on one ``_advantages_v2`` result.

    Checked before the resampling loop as well as inside the record builder: the
    loop calls ``_advantages_v2`` once per replicate and indexes its fields
    directly, so a missing field would otherwise surface as a bare ``KeyError``
    from deep inside the sampler rather than as a named inventory failure.
    """
    expected_endpoints = set(EXPECTED_ENDPOINTS)
    found = set(point)
    if found != expected_endpoints:
        _fail_inventory(f"point endpoints missing {sorted(expected_endpoints - found)}, "
                        f"extra {sorted(found - expected_endpoints)}")

    expected_fields = set(EXPECTED_POINT_FIELDS)
    for name in EXPECTED_ENDPOINTS:
        fields = set(point[name])
        if fields != expected_fields:
            _fail_inventory(
                f"{name}: point fields missing {sorted(expected_fields - fields)}, "
                f"extra {sorted(fields - expected_fields)}")


def _assert_inventory(point, draws, resampled_fields):
    """Exact endpoint and field sets, before any percentile or classification."""
    def _fail(message):
        _fail_inventory(message)

    assert_point_inventory(point)
    expected_fields = set(EXPECTED_POINT_FIELDS)

    if set(resampled_fields) != set(EXPECTED_POINT_FIELDS):
        _fail(f"resampled fields {sorted(set(resampled_fields))} != expected "
              f"{sorted(expected_fields)}")

    expected_draws = {(name, field)
                      for name in EXPECTED_ENDPOINTS for field in resampled_fields}
    found_draws = set(draws)
    if found_draws != expected_draws:
        _fail(f"draws missing {sorted(expected_draws - found_draws)[:5]}, "
              f"extra {sorted(found_draws - expected_draws)[:5]}")


def _derived_points(entry, higher_is_better, label):
    """Comparison points, derived from the raw model points rather than trusted.

    The draw level already derives its deltas by subtraction, but the point
    level took ``treated_delta``/``control_delta``/``advantage`` on faith. That
    is not fail-closed: shifting both arm deltas by the same amount leaves
    ``treated - control`` unchanged, so the comparison algebra residual stays
    zero while neither point is the raw model difference any more.
    """
    derived = {
        "contrast_delta": entry["ours_effect"] - entry["baseline_effect"],
        "treated_delta": entry["ours_treated"] - entry["baseline_treated"],
        "control_delta": entry["ours_control"] - entry["baseline_control"],
    }
    derived["advantage"] = (derived["contrast_delta"] if higher_is_better
                            else -derived["contrast_delta"])
    for field in ("treated_delta", "control_delta", "advantage"):
        drift = abs(entry[field] - derived[field])
        if not np.isfinite(drift) or drift > ALGEBRA_TOLERANCE:
            raise NonFiniteValue(
                f"{label}/{field}: reported point {entry[field]!r} is not the raw "
                f"model difference {derived[field]!r} (drift {drift:.3e} > "
                f"{ALGEBRA_TOLERANCE:g})")
    return derived


# Contrast margins and arm margins are separate settings with separate defaults.
# They share units, which is exactly why one silently became the other before.
# Both are v3 post-hoc reporting aids, never acceptance thresholds.
DEFAULT_ARM_MARGINS = {"MAE": 0.05, "MAPE": 0.25}
DEFAULT_CONTRAST_MARGINS = {"MAE": 0.05, "MAPE": 0.25}


def _validate_samples(samples):
    """Refuse too few draws rather than reporting a degraded interval."""
    if isinstance(samples, bool) or not isinstance(samples, int):
        raise NonFiniteValue(f"bootstrap samples must be an integer, got {samples!r}")
    if samples < MIN_BOOTSTRAP_SAMPLES:
        raise NonFiniteValue(
            f"bootstrap samples must be at least {MIN_BOOTSTRAP_SAMPLES} to report "
            f"a 95% interval; got {samples}")


def _checked_draws(values, label):
    """One-dimensional, non-empty, all finite -- with the endpoint named."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise NonFiniteValue(f"{label}: draws must be one-dimensional, got {array.shape}")
    if array.size == 0:
        raise NonFiniteValue(f"{label}: draws are empty")
    if not np.all(np.isfinite(array)):
        raise NonFiniteValue(f"{label}: draws contain a non-finite value")
    return array


def bootstrap_advantages_v2(ours_rows, baseline_rows, input_len, samples=2000, seed=2026,
                            trajectory_span=None, main_cohort_ids=None,
                            arm_margins=None, contrast_margins=None):
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
    # Before the loop: the sampler indexes these fields directly, so a missing
    # one must be named here rather than escaping as a KeyError from inside it.
    assert_point_inventory(point)
    # Interval on the contrast and on both arms: a Pareto verdict read off point
    # estimates alone cannot tell "the control arm degraded" from "the control
    # arm moved by less than the noise".
    # Every raw model quantity is resampled on the same replicate as the
    # comparison, so a raw interval never comes from a second bootstrap.
    resampled_fields = RAW_FIELDS + ("advantage", "treated_delta", "control_delta")
    arm_margins = DEFAULT_ARM_MARGINS if arm_margins is None else arm_margins
    contrast_margins = DEFAULT_CONTRAST_MARGINS if contrast_margins is None else contrast_margins
    # Too few draws is refused, not degraded into "uncertain". A summary that
    # silently reports every endpoint as uncertain looks like a measurement.
    _validate_samples(samples)

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
    endpoint_counts = endpoint_sample_counts(
        ours_rows, _curves_for(ours_rows, None), input_len)
    return _build_ples_records(point, draws, resampled_fields, input_len,
                               arm_margins, contrast_margins, endpoint_counts)


def _series(point_value, draw_values, label):
    """point plus a nominal 95% percentile interval, fully validated."""
    values = _checked_draws(draw_values, label)
    if not np.isfinite(point_value):
        raise NonFiniteValue(f"{label}: point estimate is not finite")
    low = float(np.percentile(values, 2.5))
    high = float(np.percentile(values, 97.5))
    if not (np.isfinite(low) and np.isfinite(high)):
        raise NonFiniteValue(f"{label}: interval ends are not finite")
    if low > high:
        raise NonFiniteValue(f"{label}: interval low {low} > high {high}")
    return {"point": float(point_value),
            "nominal_95_percentile_interval": {"low": low, "high": high}}


def _algebra(effect, contrast_point, contrast_draws, treated_point, treated_draws,
             control_point, control_draws, label):
    """``contrast - sign * (treated - control)`` at the point and per draw.

    Applied to each model's own raw quantities as well as to the comparison
    deltas. The comparison check alone is not enough: if both models' contrasts
    are broken the same way the error cancels in the difference, leaving a
    comparison residual of exactly zero over two wrong effects.
    """
    sign = CONTRAST_SIGN[effect]
    point_residual = contrast_point - sign * (treated_point - control_point)
    if not np.isfinite(point_residual):
        raise NonFiniteValue(f"{label}: algebra point residual is not finite")
    draw_residual = np.abs(contrast_draws - sign * (treated_draws - control_draws))
    if not np.all(np.isfinite(draw_residual)):
        raise NonFiniteValue(f"{label}: algebra residual over draws is not finite")
    worst = max(abs(point_residual), float(np.max(draw_residual)))
    if worst > ALGEBRA_TOLERANCE:
        raise NonFiniteValue(
            f"{label}: the reported contrast is not the signed difference of its "
            f"arms (worst residual {worst:.3e} > {ALGEBRA_TOLERANCE:g}); the "
            "contrast and the arms did not come from the same draw")
    return {
        "formula": ("contrast = control - treated" if sign < 0
                    else "contrast = treated - control"),
        "point_residual": float(point_residual),
        "max_abs_per_draw_residual": float(np.max(draw_residual)),
        "tolerance": ALGEBRA_TOLERANCE,
    }


def _build_ples_records(point, draws, resampled_fields, input_len,
                        arm_margins, contrast_margins, endpoint_counts=None):
    """The single canonical record per endpoint: both models, both arms, deltas.

    There is deliberately no second, arms-free structure. A bare contrast is
    gameable -- degrading the control arm improves it -- so a consumer must not
    be able to read one without the arms that qualify it.
    """
    _assert_inventory(point, draws, resampled_fields)
    records = []
    for name in EXPECTED_ENDPOINTS:
        entry = point[name]
        advantage_name, metric = name.rsplit("/", 1)
        effect = EFFECT_OF_ADVANTAGE[advantage_name]
        higher_is_better = HIGHER_IS_BETTER[effect]
        derived_point = _derived_points(entry, higher_is_better, name)

        column = {field: _checked_draws(draws[(name, field)], f"{name}/{field}")
                  for field in resampled_fields}
        sizes = {field: array.size for field, array in column.items()}
        if len(set(sizes.values())) != 1:
            raise NonFiniteValue(f"{name}: draw lengths differ {sizes}")

        # Comparison draws are the raw model draws subtracted within the
        # replicate, never a separately computed quantity.
        derived = {}
        for delta_name, (ours_field, baseline_field) in DELTA_SOURCES.items():
            derived[delta_name] = column[ours_field] - column[baseline_field]
        for delta_name in ("treated_delta", "control_delta"):
            drift = float(np.max(np.abs(derived[delta_name] - column[delta_name])))
            if drift > ALGEBRA_TOLERANCE:
                raise NonFiniteValue(
                    f"{name}/{delta_name}: the reported delta is not ALFNet minus "
                    f"MoGERNN (worst {drift:.3e}); it was computed separately")
        oriented = derived["contrast_delta"] if higher_is_better else -derived["contrast_delta"]
        drift = float(np.max(np.abs(oriented - column["advantage"])))
        if drift > ALGEBRA_TOLERANCE:
            raise NonFiniteValue(
                f"{name}/advantage: the oriented advantage is not the signed "
                f"contrast delta (worst {drift:.3e})")

        contrast_delta_point = derived_point["contrast_delta"]
        models = {}
        for label, prefix in (("ALFNet", "ours"), ("MoGERNN", "baseline")):
            models[label] = {
                "contrast": _series(entry[f"{prefix}_effect"],
                                    column[f"{prefix}_effect"], f"{name} {label} contrast"),
                "treated_absolute_error": _series(
                    entry[f"{prefix}_treated"], column[f"{prefix}_treated"],
                    f"{name} {label} treated"),
                "control_absolute_error": _series(
                    entry[f"{prefix}_control"], column[f"{prefix}_control"],
                    f"{name} {label} control"),
                "algebra_identity": _algebra(
                    effect, entry[f"{prefix}_effect"], column[f"{prefix}_effect"],
                    entry[f"{prefix}_treated"], column[f"{prefix}_treated"],
                    entry[f"{prefix}_control"], column[f"{prefix}_control"],
                    f"{name} {label} raw"),
            }

        comparison = {
            "contrast_delta": _series(contrast_delta_point, derived["contrast_delta"],
                                      f"{name} contrast delta"),
            "treated_delta": _series(derived_point["treated_delta"], derived["treated_delta"],
                                     f"{name} treated delta"),
            "control_delta": _series(derived_point["control_delta"], derived["control_delta"],
                                     f"{name} control delta"),
            "contrast_oriented_advantage": _series(derived_point["advantage"], oriented,
                                                   f"{name} oriented advantage"),
        }
        comparison_algebra = _algebra(
            effect, contrast_delta_point, derived["contrast_delta"],
            derived_point["treated_delta"], derived["treated_delta"],
            derived_point["control_delta"], derived["control_delta"],
            f"{name} comparison")

        arm_margin = _margin_for_endpoint(name, arm_margins)
        contrast_margin = _margin_for_endpoint(name, contrast_margins)
        # Each family reads only its own margin. Sharing one dict is how the
        # contrast threshold silently became the arm threshold.
        advantage_interval = comparison["contrast_oriented_advantage"][
            "nominal_95_percentile_interval"]
        treated = arm_status(derived_point["treated_delta"],
                             comparison["treated_delta"]["nominal_95_percentile_interval"],
                             arm_margin)
        control = arm_status(derived_point["control_delta"],
                             comparison["control_delta"]["nominal_95_percentile_interval"],
                             arm_margin)
        contrast = contrast_status(derived_point["advantage"], advantage_interval,
                                   contrast_margin)

        records.append({
            "effect": effect,
            "metric": metric,
            "definition": EFFECT_DEFINITION[effect]["formula"],
            "orientation": EFFECT_DEFINITION[effect]["orientation"],
            "models": models,
            "comparison": comparison,
            "comparison_algebra_identity": comparison_algebra,
            "contrast_margin": contrast_margin,
            "arm_margin": arm_margin,
            "contrast_status": contrast,
            "treated_status": treated,
            "control_status": control,
            "arms_pareto_status": arms_pareto_status(treated, control),
            "combined_descriptive_status": combined_status(contrast, treated, control),
        })
        if endpoint_counts is not None:
            records[-1]["sample_counts"] = dict(endpoint_counts[effect])

    if endpoint_counts is not None:
        # MAE and MAPE share one row/cell selection, so their counts must agree.
        by_effect = defaultdict(list)
        for record in records:
            by_effect[record["effect"]].append(record["sample_counts"])
        for effect, entries in by_effect.items():
            if any(entry != entries[0] for entry in entries):
                raise NonFiniteValue(
                    f"{effect}: MAE and MAPE report different sample counts, so "
                    "they are not computed over the same selection")
    return records


def _margin_for_endpoint(name, margins):
    """Margin for this endpoint's metric. MAPE is a percentage, MAE is not."""
    return margin_for(name.rsplit("/", 1)[-1], margins)


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
    # Two independent margin families. They share units, which is precisely why
    # one silently became the other when a single dict served both.
    parser.add_argument("--post-hoc-contrast-margin-mae", type=float, default=0.05)
    parser.add_argument("--post-hoc-contrast-margin-mape", type=float, default=0.25)
    parser.add_argument("--post-hoc-arm-margin-mae", type=float, default=0.05)
    parser.add_argument("--post-hoc-arm-margin-mape", type=float, default=0.25)
    args = parser.parse_args()
    for name, value in (
        ("--post-hoc-contrast-margin-mae", args.post_hoc_contrast_margin_mae),
        ("--post-hoc-contrast-margin-mape", args.post_hoc_contrast_margin_mape),
        ("--post-hoc-arm-margin-mae", args.post_hoc_arm_margin_mae),
        ("--post-hoc-arm-margin-mape", args.post_hoc_arm_margin_mape),
    ):
        if not np.isfinite(value) or value < 0:
            parser.error(f"{name} must be finite and >= 0, got {value!r}")
    if args.bootstrap_samples < MIN_BOOTSTRAP_SAMPLES:
        parser.error(f"--bootstrap-samples must be at least {MIN_BOOTSTRAP_SAMPLES} "
                     f"to report a 95% interval; got {args.bootstrap_samples}")

    ours_summary, ours_path, ours_rows, ours_version, ours_rows_path = \
        _read_result_bundle(args.ours)
    baseline_summary, baseline_path, baseline_rows, baseline_version, baseline_rows_path = \
        _read_result_bundle(args.baseline)
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
    # A strict parser proves each row is well formed; it cannot prove the file
    # belongs to this bundle or that both sides describe the same events.
    assert_rows_match_bundles(ours_rows, ours_summary, baseline_rows, baseline_summary)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Separate dict objects on purpose: passing one object twice is how the two
    # families silently became one.
    contrast_margins = {"MAE": args.post_hoc_contrast_margin_mae,
                        "MAPE": args.post_hoc_contrast_margin_mape, "RMSE": None}
    arm_margins = {"MAE": args.post_hoc_arm_margin_mae,
                   "MAPE": args.post_hoc_arm_margin_mape, "RMSE": None}

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
            arm_margins=arm_margins,
            contrast_margins=contrast_margins,
        )
        summary = {
            # 7: formal sample_counts added and the row loader contract
            # tightened. 6 replaced the flat `advantages` mapping with canonical
            # `ples_records`. Both incompatible, so the version moves with them.
            "schema_version": 7,
            "sample_counts": top_level_sample_counts(
                ours_rows, main_cohort_rows(ours_rows, main_cohort_ids),
                _paired_trajectory_span(ours_summary, ours_path,
                                        baseline_summary, baseline_path)),
            "count_semantics": COUNT_SEMANTICS,
            "protocol": "PLES v3 paired model comparison",
            "post_hoc_diagnostic_only": True,
            "inferential_status": "descriptive_not_formal",
            "conditional_on_one_trained_seed": True,
            "trained_seed": ours_summary.get("seed"),
            "interval_scope": (
                "overlap-block and cohort resampling of evaluation events at a "
                "single fixed training seed. It does not include training "
                "randomness, so it is not an interval for the model difference."
            ),
            "post_hoc_contrast_margins": contrast_margins,
            "post_hoc_arm_margins": arm_margins,
            "margin_status": (
                "v3 post-hoc reporting aids chosen after the test split was read; "
                "not pre-registered and not acceptance thresholds"
            ),
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
                "contrast_status": "favourable | degraded | equivalent | uncertain, "
                    "from the advantage interval against contrast_margin",
                "treated_status/control_status": "improve | degrade | equivalent | "
                    "uncertain, decided from the arm's 95% CI against arm_margin, "
                    "not from the point estimate",
                "arms_pareto_status": "how the two absolute-error arms moved",
                "combined_descriptive_status": "all_conditions_met requires contrast "
                    "favourable AND treated improve AND control non-degraded; it "
                    "describes which conditions held and adjudicates nothing",
            },
            "ours": args.ours_label,
            "baseline": args.baseline_label,
            # One canonical structure. The old flat `advantages` mapping is gone:
            # keeping both would let a consumer read a bare contrast and skip the
            # arms that qualify it.
            "ples_records": intervals,
        }
        plot_comparison_v2(
            curves,
            output_dir / "paired_model_comparison.png",
            args.ours_label,
            args.baseline_label,
        )
        print_keys = None
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
        json.dump(summary, handle, ensure_ascii=False, indent=2, allow_nan=False)
    if ours_version >= 2:
        # v3 reports the contrast and both absolute arms, because a contrast that
        # improved only by degrading its control arm is not a better model and
        # the contrast alone cannot show that.
        for record in intervals:
            advantage = record["comparison"]["contrast_oriented_advantage"]
            span = advantage["nominal_95_percentile_interval"]
            treated = record["comparison"]["treated_delta"]["point"]
            control = record["comparison"]["control_delta"]["point"]
            print(
                f"{record['effect']}/{record['metric']}="
                f"{advantage['point']:+.4f} | "
                f"95% CI=[{span['low']:.4f}, {span['high']:.4f}] | "
                f"treatedD={treated:+.4f}({record['treated_status']}) "
                f"controlD={control:+.4f}({record['control_status']}) "
                f"| contrast={record['contrast_status']}"
                f" | {record['combined_descriptive_status']}"
            )
    else:
        for key in print_keys:
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
