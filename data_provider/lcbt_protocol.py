import hashlib
import json
from datetime import datetime

import numpy as np


LCBT_PROTOCOL_NAME = "Lifecycle Coverage-Balanced Training"
LCBT_PROTOCOL_VERSION = 1
LCBT_PHASES = ("Cold", "Warm", "Ready", "Onset", "Decay", "Outage")
LCBT_SCENARIOS = ("join", "fault", "steady")
CALENDAR_REGIMES = ("AM", "PM", "OffPeak", "Weekend")


def accumulate_phase_mae(
    phase_error_sums,
    phase_counts,
    y_true,
    y_pred,
    phase_ids,
    changed_node_mask,
):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    phase_ids = np.asarray(phase_ids).reshape(-1).astype(np.int64)
    changed_node_mask = np.asarray(changed_node_mask).astype(bool)
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    absolute_error = np.abs(y_pred - y_true)
    for phase_index in range(len(LCBT_PHASES)):
        sample_mask = phase_ids == phase_index
        if not np.any(sample_mask):
            continue
        node_mask = changed_node_mask[sample_mask]
        selected_valid = valid[sample_mask] & node_mask[:, None, :, None]
        count = int(selected_valid.sum())
        if count <= 0:
            continue
        phase_error_sums[phase_index] += float(
            absolute_error[sample_mask][selected_valid].sum()
        )
        phase_counts[phase_index] += count


def finalize_phase_mae(phase_error_sums, phase_counts):
    per_phase = {}
    finite_values = []
    for phase_index, phase_name in enumerate(LCBT_PHASES):
        if phase_counts[phase_index] <= 0:
            per_phase[f"{phase_name}_MAE"] = float("nan")
            continue
        value = phase_error_sums[phase_index] / phase_counts[phase_index]
        per_phase[f"{phase_name}_MAE"] = float(value)
        finite_values.append(value)
    per_phase["mae"] = (
        float(sum(finite_values) / len(finite_values))
        if len(finite_values) == len(LCBT_PHASES)
        else float("nan")
    )
    return per_phase


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
        return datetime.utcfromtimestamp(float(micros) / 1_000_000.0)
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


def _scenario_assignments(calendar_regimes, ratios, rng):
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
        for scenario, count in zip(LCBT_SCENARIOS, allocation[row]):
            labels.extend([scenario] * int(count))
        labels = np.asarray(labels, dtype=object)
        assignments[indices] = labels[rng.permutation(len(labels))]
    return assignments


def _interleave_regime_indices(indices, calendar_regimes, rng):
    queues = []
    for regime in CALENDAR_REGIMES:
        regime_indices = [index for index in indices if calendar_regimes[index] == regime]
        if regime_indices:
            shuffled = np.asarray(regime_indices, dtype=np.int64)[rng.permutation(len(regime_indices))]
            queues.append(list(shuffled))
    remaining = [index for index in indices if calendar_regimes[index] not in CALENDAR_REGIMES]
    if remaining:
        shuffled = np.asarray(remaining, dtype=np.int64)[rng.permutation(len(remaining))]
        queues.append(list(shuffled))

    ordered = []
    while any(queues):
        for queue in queues:
            if queue:
                ordered.append(int(queue.pop()))
    return ordered


def _phase_id(scenario, transition_steps, input_len):
    if scenario == "steady":
        return -1
    if scenario == "join":
        if transition_steps == 0:
            return 0
        return 2 if transition_steps == input_len else 1
    if scenario == "fault":
        if transition_steps == 0:
            return 3
        return 5 if transition_steps == input_len else 4
    raise ValueError(f"unsupported LCBT scenario: {scenario}")


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


def build_lcbt_schedule(
    num_samples,
    input_len,
    adjacency,
    calendar_regimes,
    seed,
    epoch,
    scenario_ratios=(0.40, 0.40, 0.20),
    cohort_fractions=(0.025, 0.05, 0.10),
    cohort_weights=(0.25, 0.50, 0.25),
    clustered_ratio=0.50,
):
    num_samples = int(num_samples)
    input_len = int(input_len)
    adjacency = np.asarray(adjacency)
    calendar_regimes = [str(value) for value in calendar_regimes]
    if num_samples <= 0 or input_len <= 0:
        raise ValueError("num_samples and input_len must be positive")
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("adjacency must be square")
    if len(calendar_regimes) != num_samples:
        raise ValueError("calendar_regimes must align with num_samples")
    if not 0.0 <= float(clustered_ratio) <= 1.0:
        raise ValueError("clustered_ratio must be in [0, 1]")

    rng = np.random.default_rng(int(seed) + int(epoch) * 1_000_003)
    assignments = _scenario_assignments(calendar_regimes, scenario_ratios, rng)
    transition_steps = np.full(num_samples, input_len, dtype=np.int64)
    for scenario in ("join", "fault"):
        indices = np.flatnonzero(assignments == scenario).tolist()
        ordered = _interleave_regime_indices(indices, calendar_regimes, rng)
        offset = int(rng.integers(0, input_len + 1))
        for rank, index in enumerate(ordered):
            transition_steps[index] = (rank + offset) % (input_len + 1)

    event_indices = np.flatnonzero(assignments != "steady").tolist()
    fractions = _balanced_values(len(event_indices), cohort_fractions, cohort_weights, rng)
    topology_modes = _balanced_values(
        len(event_indices),
        ("clustered", "scattered"),
        (float(clustered_ratio), 1.0 - float(clustered_ratio)),
        rng,
    )
    weights, strata = _weighted_degree_strata(adjacency)
    exposure = np.zeros(adjacency.shape[0], dtype=np.int64)
    event_rank = {sample_index: rank for rank, sample_index in enumerate(event_indices)}

    schedule = []
    for sample_index in range(num_samples):
        scenario = str(assignments[sample_index])
        changed_nodes = []
        cohort_fraction = 0.0
        topology_mode = "none"
        if scenario != "steady":
            rank = event_rank[sample_index]
            cohort_fraction = float(fractions[rank])
            topology_mode = str(topology_modes[rank])
            cohort_size = min(
                adjacency.shape[0],
                max(1, int(round(adjacency.shape[0] * cohort_fraction))),
            )
            if topology_mode == "clustered":
                changed_nodes = _select_clustered_nodes(
                    weights,
                    strata,
                    exposure,
                    rng,
                    cohort_size,
                    seed_stratum=rank,
                )
            else:
                changed_nodes = _select_scattered_nodes(strata, exposure, rng, cohort_size)
            exposure[np.asarray(changed_nodes, dtype=np.int64)] += 1

        steps = int(transition_steps[sample_index])
        schedule.append(
            {
                "scenario": scenario,
                "transition_steps": steps,
                "phase_id": _phase_id(scenario, steps, input_len),
                "changed_nodes": changed_nodes,
                "cohort_fraction": cohort_fraction,
                "topology_mode": topology_mode,
                "calendar_regime": calendar_regimes[sample_index],
            }
        )
    return schedule


def compose_effective_lifecycle_features(
    raw_seq,
    lifecycle_mask,
    role_ids,
    append_role,
    append_age,
    append_observation_mask,
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
        history_age = np.cumsum(effective_mask.astype(np.float32), axis=0)
        history_age /= float(raw_seq.shape[0])
        features.append(history_age[..., None])
    if append_observation_mask:
        features.append(effective_mask.astype(np.float32)[..., None])
    return np.concatenate(features, axis=-1).astype(np.float32, copy=False), effective_mask
