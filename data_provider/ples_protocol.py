import hashlib
import json
from datetime import datetime

import numpy as np


PROTOCOL_NAME = "Paired Lifecycle Event Study"
PROTOCOL_VERSION = 1
TRACK_PHASES = {
    "join": ("Cold", "Warm", "Ready"),
    "fault": ("Onset", "Decay", "Outage"),
}
PHASE_ORDER = TRACK_PHASES["join"] + TRACK_PHASES["fault"]
CALENDAR_REGIMES = ("AM", "PM", "OffPeak", "Weekend")


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
        return datetime.utcfromtimestamp(float(microseconds) / 1_000_000.0)
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


def resolve_time_blocks(timestamps, min_event_index, max_event_index, temporal_blocks=4):
    timestamps = np.asarray(timestamps)
    if temporal_blocks <= 0:
        raise ValueError("temporal_blocks must be positive")
    min_event_index = max(0, int(min_event_index))
    max_event_index = min(len(timestamps) - 1, int(max_event_index))
    if min_event_index > max_event_index:
        return []

    valid_indices = np.arange(min_event_index, max_event_index + 1, dtype=np.int64)
    blocks = [block for block in np.array_split(valid_indices, temporal_blocks) if len(block)]
    return [
        {
            "block_id": block_id,
            "start_time_index": int(block[0]),
            "end_time_index": int(block[-1]),
            "time_steps": int(len(block)),
            "start_timestamp": _timestamp_text(timestamps[int(block[0])]),
            "end_timestamp": _timestamp_text(timestamps[int(block[-1])]),
        }
        for block_id, block in enumerate(blocks, start=1)
    ]


def resolve_cohort_plan(num_nodes, unknown_count, known_count, requested_cohorts=5, cohort_fraction=0.05):
    if num_nodes <= 0:
        raise ValueError("num_nodes must be positive")
    if unknown_count <= 0 or known_count <= 0:
        raise ValueError("PLES requires non-empty known and unknown node pools")
    if requested_cohorts <= 0:
        raise ValueError("requested_cohorts must be positive")
    if not 0 < cohort_fraction <= 1:
        raise ValueError("cohort_fraction must be in (0, 1]")

    cohort_count = min(int(requested_cohorts), int(unknown_count), int(known_count))
    cohort_size = min(
        max(1, int(round(num_nodes * cohort_fraction))),
        int(unknown_count) // cohort_count,
        int(known_count) // cohort_count,
    )
    if cohort_size <= 0:
        raise ValueError("node pools are too small to build disjoint PLES cohorts")
    return {"cohort_count": cohort_count, "cohort_size": cohort_size}


def weighted_graph_degree(adjacency):
    adjacency = np.asarray(adjacency, dtype=np.float64)
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("adjacency must be a square matrix")
    weights = np.nan_to_num(np.abs(adjacency), nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(weights, 0.0)
    return weights.sum(axis=0) + weights.sum(axis=1)


def _representative_rank_indices(pool_size, selected_count):
    if selected_count > pool_size:
        raise ValueError("selected_count cannot exceed pool_size")
    if selected_count == pool_size:
        return np.arange(pool_size, dtype=np.int64)
    positions = np.linspace(0, pool_size - 1, num=selected_count)
    indices = np.rint(positions).astype(np.int64)
    if len(np.unique(indices)) != selected_count:
        available = list(range(pool_size))
        indices = np.array(
            sorted(
                available,
                key=lambda index: min(abs(index - position) for position in positions),
            )[:selected_count],
            dtype=np.int64,
        )
    return indices


def build_balanced_cohorts(nodes, degree, cohort_count, cohort_size, seed=None):
    nodes = sorted({int(node) for node in nodes}, key=lambda node: (float(degree[node]), node))
    selected_count = int(cohort_count) * int(cohort_size)
    if cohort_count <= 0 or cohort_size <= 0:
        raise ValueError("cohort_count and cohort_size must be positive")
    if selected_count > len(nodes):
        raise ValueError("node pool is too small for the requested disjoint cohorts")

    if seed is None:
        selected_indices = _representative_rank_indices(len(nodes), selected_count)
    else:
        rng = np.random.default_rng(int(seed))
        rank_strata = np.array_split(np.arange(len(nodes), dtype=np.int64), selected_count)
        selected_indices = np.asarray([rng.choice(stratum) for stratum in rank_strata], dtype=np.int64)
    representative = [nodes[index] for index in selected_indices]
    cohorts = [[] for _ in range(cohort_count)]
    for rank, node in enumerate(representative):
        cohorts[rank % cohort_count].append(int(node))
    return [sorted(cohort) for cohort in cohorts]


def select_event_origins(
    timestamps,
    min_event_index,
    max_event_index,
    temporal_blocks=4,
    min_gap=72,
):
    timestamps = np.asarray(timestamps)
    if min_gap < 0:
        raise ValueError("min_gap must be non-negative")
    blocks = resolve_time_blocks(
        timestamps,
        min_event_index=min_event_index,
        max_event_index=max_event_index,
        temporal_blocks=temporal_blocks,
    )
    selected = []
    selected_indices = []
    for block in blocks:
        block_indices = range(block["start_time_index"], block["end_time_index"] + 1)
        midpoint = 0.5 * (block["start_time_index"] + block["end_time_index"])
        for regime in CALENDAR_REGIMES:
            candidates = [index for index in block_indices if _calendar_regime(timestamps[index]) == regime]
            candidates.sort(key=lambda index: (abs(index - midpoint), index))
            event_index = next(
                (
                    index
                    for index in candidates
                    if all(abs(index - previous) >= min_gap for previous in selected_indices)
                ),
                None,
            )
            if event_index is None:
                continue
            selected_indices.append(event_index)
            selected.append(
                {
                    "block_id": block["block_id"],
                    "regime": regime,
                    "event_time_index": event_index,
                    "event_timestamp": _timestamp_text(timestamps[event_index]),
                }
            )
    selected.sort(key=lambda item: item["event_time_index"])
    for origin_id, origin in enumerate(selected, start=1):
        origin["origin_id"] = origin_id
    return selected


def phase_for_tau(track, tau, input_len):
    if track not in TRACK_PHASES:
        raise ValueError(f"Unsupported PLES track: {track}")
    if input_len <= 0:
        raise ValueError("input_len must be positive")
    if tau < 0:
        return "Pre"
    initial, transition, settled = TRACK_PHASES[track]
    if tau == 0:
        return initial
    if tau < input_len:
        return transition
    return settled


def _mean_metrics(metric_dicts):
    if not metric_dicts:
        return None
    keys = metric_dicts[0].keys()
    return {key: float(np.mean([metrics[key] for metrics in metric_dicts])) for key in keys}


def summarize_phase_metrics(phase_metrics):
    missing = [phase for phase in PHASE_ORDER if phase not in phase_metrics]
    if missing:
        raise ValueError(f"Missing PLES phase metrics: {', '.join(missing)}")
    join_metrics = _mean_metrics([phase_metrics[phase] for phase in TRACK_PHASES["join"]])
    fault_metrics = _mean_metrics([phase_metrics[phase] for phase in TRACK_PHASES["fault"]])
    life_macro = _mean_metrics([phase_metrics[phase] for phase in PHASE_ORDER])
    return {"Join": join_metrics, "Fault": fault_metrics, "LifeMacro": life_macro}


def build_protocol_manifest(
    dataset_name,
    timestamps,
    adjacency,
    known_nodes,
    unknown_nodes,
    input_len,
    pred_len,
    test_values=None,
    input_features=None,
    target_features=None,
    pre_steps=12,
    post_steps=36,
    temporal_blocks=4,
    requested_cohorts=5,
    cohort_fraction=0.05,
    min_event_gap=72,
    seed=2026,
    bootstrap_samples=2000,
    bootstrap_seed=2026,
):
    timestamps = np.asarray(timestamps)
    adjacency = np.asarray(adjacency)
    known_nodes = sorted({int(node) for node in known_nodes})
    unknown_nodes = sorted({int(node) for node in unknown_nodes})
    if adjacency.ndim != 2:
        raise ValueError("adjacency must be a square matrix")
    num_nodes = int(adjacency.shape[0])
    if adjacency.shape != (num_nodes, num_nodes):
        raise ValueError("adjacency must be square")
    if input_len <= 0 or pred_len <= 0:
        raise ValueError("input_len and pred_len must be positive")
    if pre_steps < 1 or post_steps < input_len:
        raise ValueError("PLES requires positive pre_steps and post_steps >= input_len")
    if bootstrap_samples < 0:
        raise ValueError("bootstrap_samples must be non-negative")
    if set(known_nodes) & set(unknown_nodes):
        raise ValueError("known and unknown node pools must be disjoint")
    if any(node < 0 or node >= num_nodes for node in known_nodes + unknown_nodes):
        raise ValueError("known or unknown node index is out of range")
    if set(known_nodes) | set(unknown_nodes) != set(range(num_nodes)):
        raise ValueError("known and unknown node pools must partition all dataset nodes")
    if test_values is not None:
        test_values = np.asarray(test_values)
        if test_values.ndim < 2 or test_values.shape[:2] != (len(timestamps), num_nodes):
            raise ValueError("test_values must align with the test timeline and graph nodes")

    cohort_plan = resolve_cohort_plan(
        num_nodes,
        len(unknown_nodes),
        len(known_nodes),
        requested_cohorts=requested_cohorts,
        cohort_fraction=cohort_fraction,
    )
    degree = weighted_graph_degree(adjacency)
    join_cohorts = build_balanced_cohorts(
        unknown_nodes,
        degree,
        cohort_plan["cohort_count"],
        cohort_plan["cohort_size"],
        seed=seed,
    )
    fault_cohorts = build_balanced_cohorts(
        known_nodes,
        degree,
        cohort_plan["cohort_count"],
        cohort_plan["cohort_size"],
        seed=int(seed) + 1,
    )

    min_event_index = int(input_len) + int(pre_steps)
    max_event_index = len(timestamps) - int(post_steps) - int(pred_len)
    origins = select_event_origins(
        timestamps,
        min_event_index=min_event_index,
        max_event_index=max_event_index,
        temporal_blocks=temporal_blocks,
        min_gap=min_event_gap,
    )
    if not origins:
        raise ValueError("test timeline has no valid PLES event origins")
    time_blocks = resolve_time_blocks(
        timestamps,
        min_event_index=min_event_index,
        max_event_index=max_event_index,
        temporal_blocks=temporal_blocks,
    )
    for block in time_blocks:
        selected = [origin for origin in origins if origin["block_id"] == block["block_id"]]
        block["origin_count"] = len(selected)
        block["selected_regimes"] = [origin["regime"] for origin in selected]

    dataset_record = {
        "name": str(dataset_name),
        "num_nodes": num_nodes,
        "time_steps": int(len(timestamps)),
        "time_step_minutes": _time_step_minutes(timestamps),
        "first_timestamp": _timestamp_text(timestamps[0]) if len(timestamps) else None,
        "last_timestamp": _timestamp_text(timestamps[-1]) if len(timestamps) else None,
        "adjacency_hash": _array_hash(adjacency),
        "timestamps_hash": _stable_hash([_timestamp_text(value) for value in timestamps]),
        "known_nodes_hash": _stable_hash(known_nodes),
        "unknown_nodes_hash": _stable_hash(unknown_nodes),
        "test_values_hash": _array_hash(test_values) if test_values is not None else None,
        "input_features": input_features,
        "target_features": target_features,
    }
    dataset_record["fingerprint"] = _stable_hash(dataset_record)
    cohorts = [
        {
            "cohort_id": index + 1,
            "join_nodes": join_cohorts[index],
            "fault_nodes": fault_cohorts[index],
            "join_degree_mean": float(np.mean(degree[join_cohorts[index]])),
            "fault_degree_mean": float(np.mean(degree[fault_cohorts[index]])),
        }
        for index in range(cohort_plan["cohort_count"])
    ]
    manifest = {
        "schema_version": PROTOCOL_VERSION,
        "protocol": PROTOCOL_NAME,
        "dataset": dataset_record,
        "requested": {
            "temporal_blocks": int(temporal_blocks),
            "calendar_regimes": list(CALENDAR_REGIMES),
            "cohort_count": int(requested_cohorts),
            "cohort_fraction": float(cohort_fraction),
            "pre_steps": int(pre_steps),
            "post_steps": int(post_steps),
            "min_event_gap": int(min_event_gap),
            "seed": int(seed),
            "bootstrap_samples": int(bootstrap_samples),
            "bootstrap_seed": int(bootstrap_seed),
        },
        "resolved": {
            **cohort_plan,
            "origin_count": len(origins),
            "temporal_block_count": len(time_blocks),
            "time_blocks": time_blocks,
            "episode_count_per_track": len(origins) * cohort_plan["cohort_count"],
            "input_len": int(input_len),
            "pred_len": int(pred_len),
            "tau_range": [-int(pre_steps), int(post_steps)],
        },
        "metric_names": ["Join", "Fault", "LifeMacro"],
        "phase_names": list(PHASE_ORDER),
        "evaluation": {
            "model_state": "frozen; no calibration, adaptation, or online update",
            "slide_step": 1,
            "primary_targets": "directly changed cohort nodes",
            "primary_tau": [0, int(post_steps)],
            "phase_rule": {
                "Cold": "join tau = 0",
                "Warm": "join 1 <= tau < input_len",
                "Ready": "join tau >= input_len",
                "Onset": "fault tau = 0",
                "Decay": "fault 1 <= tau < input_len",
                "Outage": "fault tau >= input_len",
            },
            "main_metric_rule": {
                "Join": "equal mean of Cold, Warm, Ready",
                "Fault": "equal mean of Onset, Decay, Outage",
                "LifeMacro": "equal mean of all six phases",
            },
            "paired_effect": "ACE = treatment error - control error",
            "bootstrap": {
                "hierarchy": ["origin", "cohort"],
                "samples": int(bootstrap_samples),
                "seed": int(bootstrap_seed),
                "confidence": 0.95,
            },
        },
        "tracks": {
            "join": {
                "control": "selected unseen cohort remains offline",
                "treatment": "selected unseen cohort becomes online at the event origin",
            },
            "fault": {
                "control": "selected known cohort remains online",
                "treatment": "selected known cohort becomes offline at the event origin",
            },
        },
        "origins": origins,
        "cohorts": cohorts,
    }
    manifest["protocol_hash"] = _stable_hash(manifest)
    return manifest
