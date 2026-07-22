import hashlib
import json
from datetime import datetime, timezone

import numpy as np


LCBT_V2_PROTOCOL_NAME = "Lifecycle Coverage-Balanced Training v2"
LCBT_V2_PROTOCOL_VERSION = 2
LCBT_V2_PHASES = ("Cold", "Warm", "Ready", "Onset", "Decay", "Outage")
LCBT_V2_GROUPS = ("zero_history", "steady", "join", "fault")
LCBT_V2_REPORT_GROUPS = ("Unseen", "Steady", "Join", "Fault")
CALENDAR_REGIMES = ("AM", "PM", "OffPeak", "Weekend")


def stable_hash(value):
    payload = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def lcbt_manifest_hash(manifest):
    """Hash protocol semantics without machine-specific provenance fields."""
    payload = dict(manifest)
    payload.pop("protocol_hash", None)
    payload.pop("data_path", None)
    return stable_hash(payload)


def array_hash(value):
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(array.shape).encode("ascii"))
    if array.dtype.hasobject:
        for item in array.reshape(-1):
            digest.update(str(item).encode("utf-8"))
            digest.update(b"\0")
    else:
        digest.update(array.tobytes())
    return digest.hexdigest()


def _as_datetime(value):
    if hasattr(value, "to_pydatetime"):
        return value.to_pydatetime()
    if isinstance(value, datetime):
        return value
    if isinstance(value, np.datetime64):
        micros = value.astype("datetime64[us]").astype(np.int64)
        return datetime.fromtimestamp(
            float(micros) / 1_000_000.0,
            tz=timezone.utc,
        ).replace(tzinfo=None)
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def calendar_regime(value):
    timestamp = _as_datetime(value)
    if timestamp.weekday() >= 5:
        return "Weekend"
    if 7 <= timestamp.hour < 10:
        return "AM"
    if 16 <= timestamp.hour < 19:
        return "PM"
    return "OffPeak"


def resolve_lcbt_boundaries(
    total_steps,
    train_end_ratio=0.50,
    val_end_ratio=0.60,
    test_start_ratio=0.60,
    scaler_end_ratio=0.50,
):
    total_steps = int(total_steps)
    ratios = {
        "train_end_ratio": float(train_end_ratio),
        "val_end_ratio": float(val_end_ratio),
        "test_start_ratio": float(test_start_ratio),
        "scaler_end_ratio": float(scaler_end_ratio),
    }
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if not all(0.0 < value <= 1.0 for value in ratios.values()):
        raise ValueError("LCBT split ratios must be in (0, 1]")
    if ratios["train_end_ratio"] > ratios["val_end_ratio"]:
        raise ValueError("train_end_ratio cannot exceed val_end_ratio")
    if ratios["val_end_ratio"] > ratios["test_start_ratio"]:
        raise ValueError("val_end_ratio cannot exceed test_start_ratio")
    if ratios["scaler_end_ratio"] > ratios["train_end_ratio"]:
        raise ValueError("scaler_end_ratio cannot exceed train_end_ratio")

    train_end = int(total_steps * ratios["train_end_ratio"])
    val_end = int(total_steps * ratios["val_end_ratio"])
    test_start = int(total_steps * ratios["test_start_ratio"])
    scaler_end = int(total_steps * ratios["scaler_end_ratio"])
    if min(train_end, val_end - train_end, total_steps - test_start, scaler_end) <= 0:
        raise ValueError("LCBT split produced an empty train, validation, test, or scaler segment")
    return {
        "train_start": 0,
        "train_end": train_end,
        "val_start": train_end,
        "val_end": val_end,
        "test_start": test_start,
        "test_end": total_steps,
        "scaler_start": 0,
        "scaler_end": scaler_end,
    }


def _allocate_counts(total, weights):
    total = int(total)
    weights = np.asarray(weights, dtype=np.float64)
    if total < 0 or weights.ndim != 1 or len(weights) == 0:
        raise ValueError("invalid balanced-count request")
    if np.any(weights < 0) or float(weights.sum()) <= 0:
        raise ValueError("balanced-count weights must be non-negative with a positive sum")
    exact = total * weights / weights.sum()
    counts = np.floor(exact).astype(np.int64)
    remainder = total - int(counts.sum())
    order = sorted(range(len(weights)), key=lambda index: (-(exact[index] - counts[index]), index))
    for index in order[:remainder]:
        counts[index] += 1
    return counts


def _scenario_assignments(
    calendar_regimes,
    ratios,
    rng,
    scenario_names=LCBT_V2_GROUPS,
):
    calendar_regimes = np.asarray(calendar_regimes, dtype=object)
    num_samples = len(calendar_regimes)
    desired = _allocate_counts(num_samples, ratios)
    regime_names = [name for name in CALENDAR_REGIMES if np.any(calendar_regimes == name)]
    regime_names.extend(
        sorted({str(name) for name in calendar_regimes if str(name) not in regime_names})
    )
    regime_indices = [np.flatnonzero(calendar_regimes == name) for name in regime_names]
    allocation = np.stack([_allocate_counts(len(indices), ratios) for indices in regime_indices])

    difference = desired - allocation.sum(axis=0)
    while np.any(difference != 0):
        receiver = next(index for index, value in enumerate(difference) if value > 0)
        donor = next(index for index, value in enumerate(difference) if value < 0)
        candidates = [row for row in range(len(regime_indices)) if allocation[row, donor] > 0]
        if not candidates:
            raise RuntimeError("unable to rebalance LCBT scenarios across calendar regimes")
        row = min(
            candidates,
            key=lambda item: (
                abs((allocation[item, receiver] + 1) - len(regime_indices[item]) * ratios[receiver]),
                item,
            ),
        )
        allocation[row, receiver] += 1
        allocation[row, donor] -= 1
        difference[receiver] -= 1
        difference[donor] += 1

    assignments = np.empty(num_samples, dtype=object)
    for row, indices in enumerate(regime_indices):
        labels = []
        for scenario, count in zip(scenario_names, allocation[row]):
            labels.extend([scenario] * int(count))
        labels = np.asarray(labels, dtype=object)
        assignments[indices] = labels[rng.permutation(len(labels))]
    return assignments




def _phase_id_v2(scenario, evidence_steps, input_len):
    if scenario in ("zero_history", "steady"):
        return -1
    if not 1 <= int(evidence_steps) <= int(input_len):
        raise ValueError("LCBT v2 event evidence must be in [1, input_len]")
    if scenario == "join":
        if evidence_steps == 1:
            return 0
        return 2 if evidence_steps == input_len else 1
    if scenario == "fault":
        if evidence_steps == 1:
            return 3
        return 5 if evidence_steps == input_len else 4
    raise ValueError(f"unsupported LCBT v2 scenario: {scenario}")


def _weighted_degree_strata(adjacency):
    weights = np.nan_to_num(np.abs(np.asarray(adjacency, dtype=np.float64)), nan=0.0)
    np.fill_diagonal(weights, 0.0)
    degree = weights.sum(axis=0) + weights.sum(axis=1)
    order = np.lexsort((np.arange(len(degree)), degree))
    return weights, [np.asarray(part, dtype=np.int64) for part in np.array_split(order, 3)]


def _least_exposed(candidates, exposure, rng, count):
    candidates = np.asarray(candidates, dtype=np.int64)
    if count <= 0 or len(candidates) == 0:
        return []
    jitter = rng.random(len(candidates))
    order = np.lexsort((candidates, jitter, exposure[candidates]))
    return candidates[order[: min(int(count), len(candidates))]].tolist()


def _select_scattered_nodes(strata, exposure, rng, cohort_size):
    counts = _allocate_counts(cohort_size, [1.0, 1.0, 1.0])
    selected = []
    for stratum, count in zip(strata, counts):
        selected.extend(_least_exposed(stratum, exposure, rng, int(count)))
    if len(selected) < cohort_size:
        remaining = np.setdiff1d(np.arange(len(exposure)), selected, assume_unique=False)
        selected.extend(_least_exposed(remaining, exposure, rng, cohort_size - len(selected)))
    return sorted({int(node) for node in selected})


def _select_clustered_nodes(weights, strata, exposure, rng, cohort_size, seed_stratum):
    seed_candidates = strata[int(seed_stratum) % len(strata)]
    seed = _least_exposed(seed_candidates, exposure, rng, 1)[0]
    selected = [int(seed)]
    available = set(range(len(exposure))) - {int(seed)}
    while len(selected) < cohort_size and available:
        candidates = np.asarray(sorted(available), dtype=np.int64)
        connectivity = weights[np.ix_(candidates, np.asarray(selected, dtype=np.int64))].max(axis=1)
        jitter = rng.random(len(candidates))
        order = np.lexsort((candidates, jitter, exposure[candidates], -connectivity))
        chosen = int(candidates[order[0]])
        selected.append(chosen)
        available.remove(chosen)
    return sorted(selected)


def _balanced_values(size, values, weights, rng):
    counts = _allocate_counts(size, weights)
    output = []
    for value, count in zip(values, counts):
        output.extend([value] * int(count))
    output = np.asarray(output, dtype=object)
    return output[rng.permutation(len(output))].tolist()




def build_lcbt_v2_schedule(
    num_samples,
    input_len,
    adjacency,
    calendar_regimes,
    seed,
    epoch,
    scenario_ratios=(0.25, 0.25, 0.25, 0.25),
    cohort_fractions=(0.025, 0.05, 0.10),
    cohort_weights=(0.25, 0.50, 0.25),
    clustered_ratio=0.50,
    scenario_node_pools=None,
):
    """Build the frozen LCBT v2 four-group, six-phase training schedule."""
    num_samples = int(num_samples)
    input_len = int(input_len)
    adjacency = np.asarray(adjacency)
    calendar_regimes = [str(value) for value in calendar_regimes]
    if num_samples <= 0 or input_len <= 2:
        raise ValueError("LCBT v2 requires num_samples > 0 and input_len > 2")
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("adjacency must be square")
    if len(calendar_regimes) != num_samples:
        raise ValueError("calendar_regimes must align with num_samples")
    if len(scenario_ratios) != len(LCBT_V2_GROUPS):
        raise ValueError("LCBT v2 requires four scenario ratios")
    if not 0.0 <= float(clustered_ratio) <= 1.0:
        raise ValueError("clustered_ratio must be in [0, 1]")
    if scenario_node_pools is not None:
        scenario_node_pools = {
            str(name): np.asarray(sorted({int(node) for node in nodes}), dtype=np.int64)
            for name, nodes in scenario_node_pools.items()
        }
        missing_pools = [name for name in LCBT_V2_GROUPS if name not in scenario_node_pools]
        if missing_pools:
            raise ValueError(
                "LCBT v2 scenario_node_pools missing: " + ", ".join(missing_pools)
            )
        if any(len(nodes) == 0 for nodes in scenario_node_pools.values()):
            raise ValueError("LCBT v2 scenario node pools must be non-empty")

    rng = np.random.default_rng(int(seed) + int(epoch) * 1_000_003)
    assignments = _scenario_assignments(
        calendar_regimes,
        scenario_ratios,
        rng,
        scenario_names=LCBT_V2_GROUPS,
    )
    evidence_steps = np.zeros(num_samples, dtype=np.int64)
    for scenario in ("join", "fault"):
        middle_indices = []
        for regime in sorted(set(calendar_regimes)):
            indices = np.flatnonzero(
                (assignments == scenario)
                & (np.asarray(calendar_regimes, dtype=object) == regime)
            )
            shuffled = indices[rng.permutation(len(indices))]
            stage_counts = _allocate_counts(len(shuffled), (1.0, 1.0, 1.0))
            initial_end = int(stage_counts[0])
            middle_end = initial_end + int(stage_counts[1])
            evidence_steps[shuffled[:initial_end]] = 1
            middle_indices.extend(int(index) for index in shuffled[initial_end:middle_end])
            evidence_steps[shuffled[middle_end:]] = input_len
        if middle_indices:
            middle_values = _balanced_values(
                len(middle_indices),
                list(range(2, input_len)),
                np.ones(input_len - 2, dtype=np.float64),
                rng,
            )
            for index, value in zip(middle_indices, middle_values):
                evidence_steps[index] = int(value)

    fractions = _balanced_values(
        num_samples,
        cohort_fractions,
        cohort_weights,
        rng,
    )
    topology_modes = _balanced_values(
        num_samples,
        ("clustered", "scattered"),
        (float(clustered_ratio), 1.0 - float(clustered_ratio)),
        rng,
    )
    exposure = np.zeros(adjacency.shape[0], dtype=np.int64)

    schedule = []
    for sample_index in range(num_samples):
        scenario = str(assignments[sample_index])
        cohort_fraction = float(fractions[sample_index])
        topology_mode = str(topology_modes[sample_index])
        pool = (
            scenario_node_pools[scenario]
            if scenario_node_pools is not None
            else np.arange(adjacency.shape[0], dtype=np.int64)
        )
        cohort_size = min(
            len(pool),
            max(1, int(round(adjacency.shape[0] * cohort_fraction))),
        )
        pool_adjacency = adjacency[np.ix_(pool, pool)]
        weights, strata = _weighted_degree_strata(pool_adjacency)
        pool_exposure = exposure[pool].copy()
        if topology_mode == "clustered":
            selected_local = _select_clustered_nodes(
                weights,
                strata,
                pool_exposure,
                rng,
                cohort_size,
                seed_stratum=sample_index,
            )
        else:
            selected_local = _select_scattered_nodes(
                strata,
                pool_exposure,
                rng,
                cohort_size,
            )
        changed_nodes = sorted(int(pool[index]) for index in selected_local)
        exposure[np.asarray(changed_nodes, dtype=np.int64)] += 1

        evidence = int(evidence_steps[sample_index])
        schedule.append(
            {
                "scenario": scenario,
                "evidence_steps": evidence,
                "phase_id": _phase_id_v2(scenario, evidence, input_len),
                "changed_nodes": changed_nodes,
                "cohort_fraction": cohort_fraction,
                "topology_mode": topology_mode,
                "calendar_regime": calendar_regimes[sample_index],
            }
        )
    return schedule


def split_unseen_nodes(
    master_unseen_nodes,
    adjacency,
    validation_ratio=0.40,
    seed=2026,
):
    """Freeze disjoint validation/test unseen pools within degree strata."""
    nodes = np.asarray(sorted({int(node) for node in master_unseen_nodes}), dtype=np.int64)
    adjacency = np.asarray(adjacency)
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("adjacency must be square")
    if len(nodes) < 2:
        raise ValueError("at least two master unseen nodes are required")
    if np.any((nodes < 0) | (nodes >= adjacency.shape[0])):
        raise ValueError("master unseen nodes fall outside the adjacency")
    if not 0.0 < float(validation_ratio) < 1.0:
        raise ValueError("validation_ratio must be in (0, 1)")

    weights = np.nan_to_num(np.abs(adjacency.astype(np.float64)), nan=0.0)
    np.fill_diagonal(weights, 0.0)
    degree = weights.sum(axis=0) + weights.sum(axis=1)
    ordered_nodes = nodes[np.lexsort((nodes, degree[nodes]))]
    strata = [np.asarray(part, dtype=np.int64) for part in np.array_split(ordered_nodes, 3)]
    validation_count = int(round(len(nodes) * float(validation_ratio)))
    validation_count = min(len(nodes) - 1, max(1, validation_count))
    validation_counts = _allocate_counts(
        validation_count,
        [len(stratum) for stratum in strata],
    )
    rng = np.random.default_rng(int(seed))
    validation_nodes = []
    test_nodes = []
    for stratum, count in zip(strata, validation_counts):
        shuffled = stratum[rng.permutation(len(stratum))]
        validation_nodes.extend(int(node) for node in shuffled[: int(count)])
        test_nodes.extend(int(node) for node in shuffled[int(count) :])
    validation_nodes = np.asarray(sorted(validation_nodes), dtype=np.int64)
    test_nodes = np.asarray(sorted(test_nodes), dtype=np.int64)
    metadata = {
        "validation_ratio": float(validation_ratio),
        "degree_stratum_sizes": [int(len(stratum)) for stratum in strata],
        "validation_stratum_counts": [int(value) for value in validation_counts],
        "test_stratum_counts": [
            int(len(stratum) - count)
            for stratum, count in zip(strata, validation_counts)
        ],
    }
    return validation_nodes, test_nodes, metadata


def select_master_unseen_nodes(adjacency, fraction=0.25, seed=2026):
    """Create a deterministic degree-stratified master unseen pool."""
    adjacency = np.asarray(adjacency)
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("adjacency must be square")
    if not 0.0 < float(fraction) < 1.0:
        raise ValueError("fraction must be in (0, 1)")
    num_nodes = int(adjacency.shape[0])
    target = min(num_nodes - 1, max(2, int(round(num_nodes * float(fraction)))))
    weights = np.nan_to_num(np.abs(adjacency.astype(np.float64)), nan=0.0)
    np.fill_diagonal(weights, 0.0)
    degree = weights.sum(axis=0) + weights.sum(axis=1)
    order = np.lexsort((np.arange(num_nodes), degree))
    strata = [np.asarray(part, dtype=np.int64) for part in np.array_split(order, 3)]
    counts = _allocate_counts(target, [len(stratum) for stratum in strata])
    rng = np.random.default_rng(int(seed))
    selected = []
    for stratum, count in zip(strata, counts):
        shuffled = stratum[rng.permutation(len(stratum))]
        selected.extend(int(node) for node in shuffled[: int(count)])
    return np.asarray(sorted(selected), dtype=np.int64)


def _finite_mean(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) else float("nan")


def finalize_lifecycle_mae(cell_mae):
    """Equal-average cells into two anchors, two trajectories, and LifeAvg."""
    phase_mae = {
        name: _finite_mean(cell_mae.get(name, []))
        for name in ("Unseen", "Steady") + LCBT_V2_PHASES
    }
    join_mae = _finite_mean([phase_mae[name] for name in LCBT_V2_PHASES[:3]])
    fault_mae = _finite_mean([phase_mae[name] for name in LCBT_V2_PHASES[3:]])
    group_mae = {
        "unseen_mae": phase_mae["Unseen"],
        "steady_mae": phase_mae["Steady"],
        "join_mae": join_mae,
        "fault_mae": fault_mae,
    }
    complete = np.asarray(list(group_mae.values()), dtype=np.float64)
    group_mae["mae"] = (
        float(complete.mean()) if np.all(np.isfinite(complete)) else float("nan")
    )
    group_mae["worst_group_mae"] = (
        float(complete.max()) if np.all(np.isfinite(complete)) else float("nan")
    )
    group_mae.update({f"{name}_MAE": value for name, value in phase_mae.items()})
    return group_mae


def accumulate_lifecycle_mae(
    cell_state,
    y_true,
    y_pred,
    group_ids,
    phase_ids,
    target_cohort_mask,
    calendar_ids,
):
    """Accumulate one original-scale MAE per replay cell, not per tensor element."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    group_ids = np.asarray(group_ids).reshape(-1).astype(np.int64)
    phase_ids = np.asarray(phase_ids).reshape(-1).astype(np.int64)
    calendar_ids = np.asarray(calendar_ids).reshape(-1).astype(np.int64)
    target_cohort_mask = np.asarray(target_cohort_mask).astype(bool)
    if y_true.shape != y_pred.shape or y_true.ndim != 4:
        raise ValueError("lifecycle MAE expects matching [batch,time,node,feature] arrays")
    if target_cohort_mask.shape != (y_true.shape[0], y_true.shape[2]):
        raise ValueError("target_cohort_mask must have shape [batch,node]")

    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    errors = np.abs(y_pred - y_true)
    for sample_index in range(y_true.shape[0]):
        selected = valid[sample_index] & target_cohort_mask[sample_index][None, :, None]
        if not np.any(selected):
            continue
        group_id = int(group_ids[sample_index])
        if group_id == 0:
            label = "Unseen"
        elif group_id == 1:
            label = "Steady"
        elif group_id in (2, 3):
            phase_id = int(phase_ids[sample_index])
            if not 0 <= phase_id < len(LCBT_V2_PHASES):
                raise ValueError("event replay cell is missing a valid phase_id")
            label = LCBT_V2_PHASES[phase_id]
        else:
            raise ValueError(f"unsupported lifecycle validation group_id: {group_id}")
        calendar_id = int(calendar_ids[sample_index])
        cell_state.setdefault(label, {}).setdefault(calendar_id, []).append(
            float(errors[sample_index][selected].mean())
        )


def finalize_lifecycle_validation_mae(cell_state):
    """Equal-average replay cells within calendar strata, then semantic groups."""
    calendar_balanced = {}
    for label in ("Unseen", "Steady") + LCBT_V2_PHASES:
        calendar_balanced[label] = [
            _finite_mean(values)
            for _, values in sorted(cell_state.get(label, {}).items())
        ]
    return finalize_lifecycle_mae(calendar_balanced)


def compose_effective_lifecycle_features(
    raw_seq,
    lifecycle_mask,
    role_ids,
    append_role,
    append_age,
    append_observation_mask,
    state_age=None,
):
    raw_seq = np.asarray(raw_seq)
    lifecycle_mask = np.asarray(lifecycle_mask, dtype=bool)
    role_ids = np.asarray(role_ids, dtype=np.int64)
    if raw_seq.ndim != 3:
        raise ValueError("raw_seq must have shape [time, node, feature]")
    if lifecycle_mask.shape != raw_seq.shape[:2] or role_ids.shape != raw_seq.shape[:2]:
        raise ValueError("lifecycle_mask and role_ids must align with raw_seq")
    if np.any((role_ids < 0) | (role_ids > 3)):
        raise ValueError("role_ids must be in [0, 3]")

    natural_mask = np.isfinite(raw_seq).all(axis=-1)
    effective_mask = natural_mask & lifecycle_mask
    masked_raw = np.where(effective_mask[..., None], raw_seq, np.nan).astype(np.float32, copy=False)
    features = [masked_raw]
    if append_role:
        features.append(np.eye(4, dtype=np.float32)[role_ids])
    if append_age:
        if state_age is None:
            history_age = np.cumsum(effective_mask.astype(np.float32), axis=0)
            history_age /= float(raw_seq.shape[0])
        else:
            history_age = np.asarray(state_age, dtype=np.float32)
            if history_age.shape != raw_seq.shape[:2]:
                raise ValueError("state_age must align with raw_seq")
        features.append(history_age[..., None])
    if append_observation_mask:
        features.append(effective_mask.astype(np.float32)[..., None])
    return np.concatenate(features, axis=-1).astype(np.float32, copy=False), effective_mask
