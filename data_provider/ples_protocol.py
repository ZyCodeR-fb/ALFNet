import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np


LCE_PROTOCOL_NAME = "Lifecycle Coverage Evaluation"
LCE_PROTOCOL_VERSION = 2
PLES_V2_PROTOCOL_NAME = "Paired Lifecycle Event Study v2"
PLES_V2_PROTOCOL_VERSION = 2
TRACK_PHASES = {
    "join": ("Cold", "Warm", "Ready"),
    "fault": ("Onset", "Decay", "Outage"),
}
PHASE_ORDER = TRACK_PHASES["join"] + TRACK_PHASES["fault"]
CALENDAR_REGIMES = ("AM", "PM", "OffPeak", "Weekend")
PLES_V2_EFFECT_NAMES = ("JoinLift", "FaultPenalty", "PeakFaultPenalty")


def _stable_hash(value):
    payload = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _array_hash(value):
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _as_datetime(value):
    if hasattr(value, "to_pydatetime"):
        return value.to_pydatetime()
    if isinstance(value, datetime):
        return value
    if isinstance(value, np.datetime64):
        microseconds = value.astype("datetime64[us]").astype(np.int64)
        return datetime.fromtimestamp(
            float(microseconds) / 1_000_000.0,
            tz=timezone.utc,
        ).replace(tzinfo=None)
    text = str(value).replace("Z", "+00:00")
    return datetime.fromisoformat(text)


def _timestamp_text(value):
    return _as_datetime(value).isoformat(sep=" ")


def _calendar_regime(value):
    timestamp = _as_datetime(value)
    if timestamp.weekday() >= 5:
        return "Weekend"
    if 7 <= timestamp.hour < 10:
        return "AM"
    if 16 <= timestamp.hour < 19:
        return "PM"
    return "OffPeak"


def _time_step_minutes(timestamps):
    if len(timestamps) < 2:
        return None
    seconds = [
        (_as_datetime(timestamps[index + 1]) - _as_datetime(timestamps[index])).total_seconds()
        for index in range(len(timestamps) - 1)
    ]
    return float(np.median(seconds) / 60.0)




def weighted_graph_degree(adjacency):
    adjacency = np.asarray(adjacency, dtype=np.float64)
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("adjacency must be a square matrix")
    weights = np.nan_to_num(np.abs(adjacency), nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(weights, 0.0)
    return weights.sum(axis=0) + weights.sum(axis=1)




def phase_for_tau_v2(track, tau, input_len):
    """Inclusive event clock: tau=0 contributes one event-affected input point."""
    if track not in TRACK_PHASES:
        raise ValueError(f"Unsupported PLES track: {track}")
    if input_len <= 2:
        raise ValueError("PLES v2 requires input_len > 2")
    if tau < 0:
        return "Pre"
    initial, transition, settled = TRACK_PHASES[track]
    if tau == 0:
        return initial
    if tau < input_len - 1:
        return transition
    return settled




def select_event_origins_v2(
    timestamps,
    timeline_length,
    input_len,
    pred_len,
    pre_steps,
    post_steps,
):
    """Select at most one inclusive-clock origin per calendar day and stratum."""
    timeline_length = int(timeline_length)
    min_event_index = int(input_len) + int(pre_steps) - 1
    max_event_index = timeline_length - int(post_steps) - int(pred_len) - 1
    if min_event_index > max_event_index:
        return []
    candidates = np.arange(min_event_index, max_event_index + 1, dtype=np.int64)
    if timestamps is None:
        trajectory_span = int(input_len) + int(pred_len) + int(pre_steps) + int(post_steps)
        origins = []
        for origin_id, event_index in enumerate(
            candidates[:: max(1, trajectory_span)],
            start=1,
        ):
            origins.append(
                {
                    "origin_id": int(origin_id),
                    "event_time_index": int(event_index),
                    "event_timestamp": None,
                    "calendar_day": f"block-{origin_id:04d}",
                    "regime": "Block",
                }
            )
        return origins

    timestamps = np.asarray(timestamps)
    if len(timestamps) != timeline_length:
        raise ValueError("timestamps must align with the evaluation timeline")
    grouped = {}
    for event_index in candidates:
        timestamp = timestamps[int(event_index)]
        day = _as_datetime(timestamp).date().isoformat()
        regime = _calendar_regime(timestamp)
        grouped.setdefault((day, regime), []).append(int(event_index))
    origins = []
    for origin_id, ((day, regime), indices) in enumerate(sorted(grouped.items()), start=1):
        event_index = int(indices[len(indices) // 2])
        origins.append(
            {
                "origin_id": int(origin_id),
                "event_time_index": event_index,
                "event_timestamp": _timestamp_text(timestamps[event_index]),
                "calendar_day": str(day),
                "regime": str(regime),
            }
        )
    return origins


def _allocate_integer_counts(total, weights):
    weights = np.asarray(weights, dtype=np.float64)
    exact = int(total) * weights / weights.sum()
    counts = np.floor(exact).astype(np.int64)
    for index in np.argsort(-(exact - counts), kind="stable")[: int(total) - int(counts.sum())]:
        counts[index] += 1
    return counts


def _select_v2_cohort(nodes, adjacency, degree, size, style, seed, seed_stratum):
    nodes = np.asarray(sorted({int(node) for node in nodes}), dtype=np.int64)
    size = min(len(nodes), max(1, int(size)))
    rng = np.random.default_rng(int(seed))
    ordered = nodes[np.lexsort((nodes, degree[nodes]))]
    strata = [np.asarray(part, dtype=np.int64) for part in np.array_split(ordered, 3)]
    if style == "scattered":
        counts = _allocate_integer_counts(size, [len(part) for part in strata])
        selected = []
        for part, count in zip(strata, counts):
            shuffled = part[rng.permutation(len(part))]
            selected.extend(int(node) for node in shuffled[: int(count)])
        return sorted(selected)
    if style != "clustered":
        raise ValueError(f"unsupported cohort style: {style}")

    nonempty_strata = [part for part in strata if len(part)]
    seed_candidates = nonempty_strata[int(seed_stratum) % len(nonempty_strata)]
    seed_node = int(seed_candidates[int(seed) % len(seed_candidates)])
    selected = [seed_node]
    available = set(int(node) for node in nodes) - {seed_node}
    weights = np.nan_to_num(np.abs(np.asarray(adjacency, dtype=np.float64)), nan=0.0)
    while len(selected) < size and available:
        candidates = np.asarray(sorted(available), dtype=np.int64)
        connectivity = weights[np.ix_(candidates, np.asarray(selected, dtype=np.int64))].max(axis=1)
        jitter = rng.random(len(candidates))
        order = np.lexsort((candidates, jitter, -connectivity))
        chosen = int(candidates[order[0]])
        selected.append(chosen)
        available.remove(chosen)
    return sorted(selected)


def build_lifecycle_cohorts_v2(
    adjacency,
    known_nodes,
    test_unseen_nodes,
    cohort_fractions=(0.025, 0.05, 0.10),
    cohort_weights=(0.25, 0.50, 0.25),
    seed=2026,
):
    adjacency = np.asarray(adjacency)
    num_nodes = int(adjacency.shape[0])
    if len(cohort_fractions) != len(cohort_weights):
        raise ValueError("cohort fractions and weights must align")
    # Eight cohorts realize 0.25/0.50/0.25 exactly and split styles 50/50.
    fraction_counts = _allocate_integer_counts(4, cohort_weights)
    fraction_plan = []
    for fraction, count in zip(cohort_fractions, fraction_counts):
        fraction_plan.extend([float(fraction)] * int(count))
    degree = weighted_graph_degree(adjacency)
    cohorts = []
    cohort_id = 1
    for style_index, style in enumerate(("clustered", "scattered")):
        for fraction_index, fraction in enumerate(fraction_plan):
            requested_size = max(1, int(round(num_nodes * fraction)))
            join_nodes = _select_v2_cohort(
                test_unseen_nodes,
                adjacency,
                degree,
                requested_size,
                style,
                seed=int(seed) + cohort_id * 17,
                seed_stratum=fraction_index,
            )
            fault_nodes = _select_v2_cohort(
                known_nodes,
                adjacency,
                degree,
                requested_size,
                style,
                seed=int(seed) + cohort_id * 17 + 1,
                seed_stratum=fraction_index + style_index,
            )
            cohorts.append(
                {
                    "cohort_id": int(cohort_id),
                    "fraction": float(fraction),
                    "style": style,
                    "requested_size": int(requested_size),
                    "join_nodes": join_nodes,
                    "fault_nodes": fault_nodes,
                    "join_degree_mean": float(np.mean(degree[join_nodes])),
                    "fault_degree_mean": float(np.mean(degree[fault_nodes])),
                }
            )
            cohort_id += 1
    return cohorts


def build_protocol_manifest(
    dataset_name,
    timestamps,
    adjacency,
    known_nodes,
    master_unseen_nodes,
    validation_unseen_nodes,
    test_unseen_nodes,
    input_len,
    pred_len,
    training_protocol_hash,
    test_values=None,
    input_features=None,
    target_features=None,
    pre_steps=12,
    post_steps=36,
    cohort_fractions=(0.025, 0.05, 0.10),
    cohort_weights=(0.25, 0.50, 0.25),
    seed=2026,
    bootstrap_samples=2000,
    bootstrap_seed=2026,
    lce_protocol_version=2,
):
    adjacency = np.asarray(adjacency)
    num_nodes = int(adjacency.shape[0])
    timestamps = None if timestamps is None else np.asarray(timestamps)
    known_nodes = sorted({int(node) for node in known_nodes})
    master_unseen_nodes = sorted({int(node) for node in master_unseen_nodes})
    validation_unseen_nodes = sorted({int(node) for node in validation_unseen_nodes})
    test_unseen_nodes = sorted({int(node) for node in test_unseen_nodes})
    if adjacency.shape != (num_nodes, num_nodes):
        raise ValueError("adjacency must be square")
    if set(known_nodes) & set(master_unseen_nodes):
        raise ValueError("known and master unseen pools must be disjoint")
    if set(known_nodes) | set(master_unseen_nodes) != set(range(num_nodes)):
        raise ValueError("known and master unseen pools must partition all nodes")
    if set(validation_unseen_nodes) & set(test_unseen_nodes):
        raise ValueError("validation and test unseen pools must be disjoint")
    if set(validation_unseen_nodes) | set(test_unseen_nodes) != set(master_unseen_nodes):
        raise ValueError("validation/test unseen pools must partition master unseen nodes")
    timeline_length = len(timestamps) if timestamps is not None else int(np.asarray(test_values).shape[0])
    if test_values is not None:
        test_values = np.asarray(test_values)
        if test_values.shape[:2] != (timeline_length, num_nodes):
            raise ValueError("test_values must align with timeline and graph")

    origins = select_event_origins_v2(
        timestamps,
        timeline_length,
        input_len,
        pred_len,
        pre_steps,
        post_steps,
    )
    if not origins:
        raise ValueError("evaluation timeline has no valid LCE/PLES v2 origins")
    cohorts = build_lifecycle_cohorts_v2(
        adjacency,
        known_nodes,
        test_unseen_nodes,
        cohort_fractions=cohort_fractions,
        cohort_weights=cohort_weights,
        seed=seed,
    )
    timestamp_text = (
        [_timestamp_text(value) for value in timestamps]
        if timestamps is not None
        else None
    )
    dataset_record = {
        "name": str(dataset_name),
        "num_nodes": num_nodes,
        "time_steps": int(timeline_length),
        "time_step_minutes": _time_step_minutes(timestamps) if timestamps is not None else None,
        "first_timestamp": timestamp_text[0] if timestamp_text else None,
        "last_timestamp": timestamp_text[-1] if timestamp_text else None,
        "adjacency_hash": _array_hash(adjacency),
        "timestamps_hash": _stable_hash(timestamp_text) if timestamp_text else None,
        "test_values_hash": _array_hash(test_values) if test_values is not None else None,
        "input_features": input_features,
        "target_features": target_features,
    }
    dataset_record["fingerprint"] = _stable_hash(dataset_record)
    if int(lce_protocol_version) != 2:
        raise ValueError("the maintained absolute evaluation protocol is LCE v2")
    lce = {
        "protocol": f"{LCE_PROTOCOL_NAME} v2",
        "event_clock": "input_end=event_time+tau+1",
        "anchors": ["Unseen", "Steady"],
        "phase_names": list(PHASE_ORDER),
        "main_rows": ["Unseen", "Steady", "Join", "Fault", "LifeAvg"],
        "aggregation": {
            "cell": "origin_x_cohort_x_tau",
            "strata": "equal calendar-stratum mean",
            "Join": "equal mean over k=1..input_len",
            "Fault": "equal mean over k=1..input_len",
            "LifeAvg": "mean(Unseen,Steady,Join,Fault)",
        },
        "phase_rule": {
            "Cold": "join tau=0; one post-join input point",
            "Warm": "join 1<=tau<input_len-1",
            "Ready": "join tau=input_len-1",
            "Onset": "fault tau=0; one failed input point",
            "Decay": "fault 1<=tau<input_len-1",
            "Outage": "fault tau=input_len-1",
        },
        "absolute_tau_range": [0, int(input_len) - 1],
        "long_post_event_usage": "PLES only",
    }
    ples = {
        "protocol": PLES_V2_PROTOCOL_NAME,
        "pre_event_usage": "implementation check only",
        "post_event_effects": ["JoinLift", "FaultPenalty", "PeakFaultPenalty"],
        "bootstrap": {
            "hierarchy": ["origin", "cohort"],
            "samples": int(bootstrap_samples),
            "seed": int(bootstrap_seed),
            "confidence": 0.95,
        },
        "causal_claim": False,
    }
    manifest = {
        "schema_version": 2,
        "protocol": "LCBT v3 + LCE v2 + PLES v2",
        "dataset": dataset_record,
        "nodes": {
            "known": known_nodes,
            "master_unseen": master_unseen_nodes,
            "validation_unseen": validation_unseen_nodes,
            "test_unseen": test_unseen_nodes,
        },
        "requested": {
            "calendar_regimes": list(CALENDAR_REGIMES),
            "cohort_fractions": [float(value) for value in cohort_fractions],
            "cohort_weights": [float(value) for value in cohort_weights],
            "cohort_styles": ["clustered", "scattered"],
            "pre_steps": int(pre_steps),
            "post_steps": int(post_steps),
            "seed": int(seed),
        },
        "resolved": {
            "origin_count": int(len(origins)),
            "cohort_count": int(len(cohorts)),
            "episode_count_per_track": int(len(origins) * len(cohorts)),
            "input_len": int(input_len),
            "pred_len": int(pred_len),
            "tau_range": [-int(pre_steps), int(post_steps)],
        },
        "training_protocol_hash": training_protocol_hash,
        "lce": lce,
        "ples": ples,
        "tracks": {
            "join": {
                "control": "test-unseen cohort remains offline",
                "treatment": "same cohort joins at event origin",
            },
            "fault": {
                "control": "known cohort remains steadily online",
                "treatment": "same cohort fails at event origin",
            },
        },
        "origins": origins,
        "cohorts": cohorts,
    }
    manifest["lce_protocol_hash"] = _stable_hash(lce)
    manifest["ples_protocol_hash"] = _stable_hash(ples)
    manifest["protocol_hash"] = _stable_hash(manifest)
    return manifest


def _row_metrics_v2(row, prefix):
    count = int(row.get("count", 0))
    if count <= 0:
        return None
    return {
        "MAPE": float(row[f"{prefix}_ape_sum"]) / count * 100.0,
        "MAE": float(row[f"{prefix}_ae_sum"]) / count,
        "RMSE": float(np.sqrt(float(row[f"{prefix}_se_sum"]) / count)),
    }


def _mean_metrics_v2(metrics):
    if not metrics:
        return None
    return {
        name: float(np.mean([entry[name] for entry in metrics]))
        for name in ("MAPE", "MAE", "RMSE")
    }


def _calendar_balanced_metrics_v2(rows, prefix):
    by_regime = defaultdict(list)
    for row in rows:
        metrics = _row_metrics_v2(row, prefix)
        if metrics is not None:
            by_regime[str(row.get("regime", "all"))].append(metrics)
    regime_metrics = [_mean_metrics_v2(values) for values in by_regime.values() if values]
    return _mean_metrics_v2(regime_metrics)


def _tau_balanced_metrics_v2(rows, prefix):
    by_tau = defaultdict(list)
    for row in rows:
        by_tau[int(row["tau"])].append(row)
    tau_metrics = [
        _calendar_balanced_metrics_v2(by_tau[tau], prefix)
        for tau in sorted(by_tau)
    ]
    return _mean_metrics_v2([metrics for metrics in tau_metrics if metrics is not None])


def summarize_lce_rows(rows, input_len):
    """Summarize absolute accuracy with cell, calendar, phase, then group equality."""
    if int(input_len) <= 2:
        raise ValueError("LCE v2 requires input_len > 2")
    post_rows = [row for row in rows if 0 <= int(row["tau"]) < int(input_len)]
    phase_metrics = {}
    for phase in PHASE_ORDER:
        phase_rows = [row for row in post_rows if row.get("phase") == phase]
        metrics = _tau_balanced_metrics_v2(phase_rows, "treated")
        if metrics is None:
            raise ValueError(f"LCE phase has no valid cells: {phase}")
        phase_metrics[phase] = metrics

    join_rows = [row for row in post_rows if row.get("track") == "join"]
    fault_rows = [row for row in post_rows if row.get("track") == "fault"]
    unseen = _tau_balanced_metrics_v2(join_rows, "control")
    steady = _tau_balanced_metrics_v2(fault_rows, "control")
    join = _tau_balanced_metrics_v2(join_rows, "treated")
    fault = _tau_balanced_metrics_v2(fault_rows, "treated")
    life_avg = _mean_metrics_v2([unseen, steady, join, fault])
    trajectory_metrics = {"Join": [], "Fault": []}
    for label, track in (("Join", "join"), ("Fault", "fault")):
        track_rows = [row for row in post_rows if row.get("track") == track]
        for tau in range(int(input_len)):
            metrics = _calendar_balanced_metrics_v2(
                [row for row in track_rows if int(row["tau"]) == tau],
                "treated",
            )
            if metrics is None:
                raise ValueError(f"LCE v2 {track} is missing tau={tau}")
            trajectory_metrics[label].append(
                {"k": tau + 1, "visibility": (tau + 1) / float(input_len), **metrics}
            )
    return {
        "phase_metrics": phase_metrics,
        "trajectory_metrics": trajectory_metrics,
        "main_metrics": {
            "Unseen": unseen,
            "Steady": steady,
            "Join": join,
            "Fault": fault,
            "LifeAvg": life_avg,
        },
    }


def _mean_optional_v2(values):
    finite = [float(value) for value in values if value not in {None, ""} and np.isfinite(value)]
    return float(np.mean(finite)) if finite else None


def _curve_rows_v2(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(str(row["track"]), int(row["tau"]), str(row["phase"]))].append(row)

    curves = []
    for (track, tau, phase), cell_rows in sorted(
        grouped.items(),
        key=lambda item: (item[0][0], item[0][1]),
    ):
        treated = _calendar_balanced_metrics_v2(cell_rows, "treated")
        control = _calendar_balanced_metrics_v2(cell_rows, "control")
        if treated is None or control is None:
            continue
        if track == "join":
            effect = {metric: control[metric] - treated[metric] for metric in treated}
            effect_name = "JoinLift"
        else:
            effect = {metric: treated[metric] - control[metric] for metric in treated}
            effect_name = "FaultPenalty"
        graph_weight = _mean_optional_v2([row.get("graph_weight") for row in cell_rows])
        curves.append(
            {
                "track": track,
                "tau": tau,
                "phase": phase,
                "cell_count": len(cell_rows),
                "target_count": int(sum(int(row["count"]) for row in cell_rows)),
                "treated": treated,
                "control": control,
                "effect_name": effect_name,
                "effect": effect,
                "graph_weight": graph_weight,
                "patch_residual_weight": None if graph_weight is None else 1.0 - graph_weight,
            }
        )
    return curves


def _paired_effects_v2(curves):
    post = [curve for curve in curves if int(curve["tau"]) >= 0]
    join_lift = [curve["effect"]["MAPE"] for curve in post if curve["track"] == "join"]
    fault_penalty = [curve["effect"]["MAPE"] for curve in post if curve["track"] == "fault"]
    if not join_lift or not fault_penalty:
        raise ValueError("PLES v2 requires post-event Join and Fault cells")
    return {
        "JoinLift": float(np.mean(join_lift)),
        "FaultPenalty": float(np.mean(fault_penalty)),
        "PeakFaultPenalty": float(np.max(fault_penalty)),
    }


def summarize_paired_rows_v2(rows):
    curves = _curve_rows_v2(rows)
    pre_differences = [
        abs(float(curve["treated"]["MAPE"]) - float(curve["control"]["MAPE"]))
        for curve in curves
        if int(curve["tau"]) < 0
    ]
    return {
        "paired_effects": _paired_effects_v2(curves),
        "pre_event_max_abs_mape_difference": max(pre_differences, default=0.0),
        "curves": curves,
    }


def bootstrap_paired_effects_v2(rows, samples=2000, seed=2026):
    point = summarize_paired_rows_v2(rows)["paired_effects"]
    if samples <= 0:
        return {
            name: {"point": point[name], "low": None, "high": None}
            for name in PLES_V2_EFFECT_NAMES
        }

    units = defaultdict(lambda: defaultdict(list))
    for row in rows:
        units[int(row["origin_id"])][int(row["cohort_id"])].append(row)
    origin_ids = sorted(units)
    if not origin_ids:
        raise ValueError("cannot bootstrap empty PLES v2 rows")

    rng = np.random.default_rng(seed)
    draws = {
        name: np.empty(int(samples), dtype=np.float64)
        for name in PLES_V2_EFFECT_NAMES
    }
    for sample_index in range(int(samples)):
        sampled_rows = []
        for origin_id in rng.choice(origin_ids, size=len(origin_ids), replace=True):
            cohort_ids = sorted(units[int(origin_id)])
            for cohort_id in rng.choice(cohort_ids, size=len(cohort_ids), replace=True):
                sampled_rows.extend(units[int(origin_id)][int(cohort_id)])
        effects = _paired_effects_v2(_curve_rows_v2(sampled_rows))
        for name in PLES_V2_EFFECT_NAMES:
            draws[name][sample_index] = effects[name]

    return {
        name: {
            "point": point[name],
            "low": float(np.percentile(draws[name], 2.5)),
            "high": float(np.percentile(draws[name], 97.5)),
        }
        for name in PLES_V2_EFFECT_NAMES
    }
