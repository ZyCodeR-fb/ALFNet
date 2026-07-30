import hashlib
import json

import numpy as np

from data_provider.calendar_strata import (
    STRATA,
    calendar_stratum,
    weighted_stratum_mean,
)


LCBT_PROTOCOL_NAME = "Lifecycle Coverage-Balanced Training v3.2"
# 5: the evidence-length quota. v3.1 allocated k=1..L inside each
# (scenario, calendar regime) cell independently, and `_allocate_counts` breaks
# its remainder ties by index -- with uniform weights every fractional part ties,
# so every cell handed its leftovers to the smallest k. Simulated over twenty
# realistic cells the counts ran monotonically 377 at k=1 down to 357 at k=12,
# a construction-level tilt rather than sampling noise, against a manifest that
# declares `event_evidence_distribution: uniform_k=1..L`.
#
# The tilt biased no reported number -- LCE aggregates `mean_k(Join)`, a mean of
# per-k means, which unequal counts do not distort -- so this is an
# implementation/manifest inconsistency, not a wrong metric. LCE and PLES are
# unchanged and stay at version 3.
LCBT_PROTOCOL_VERSION = 5
LCBT_PHASES = ("Cold", "Warm", "Ready", "Onset", "Decay", "Outage")
LCBT_GROUPS = ("zero_history", "steady", "join", "fault")


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
    scenario_names=LCBT_GROUPS,
):
    calendar_regimes = np.asarray(calendar_regimes, dtype=object)
    num_samples = len(calendar_regimes)
    desired = _allocate_counts(num_samples, ratios)
    regime_names = [name for name in STRATA if np.any(calendar_regimes == name)]
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

def _phase_id(scenario, evidence_steps, input_len):
    if scenario in ("zero_history", "steady"):
        return -1
    if not 1 <= int(evidence_steps) <= int(input_len):
        raise ValueError("LCBT event evidence must be in [1, input_len]")
    if scenario == "join":
        if evidence_steps == 1:
            return 0
        return 2 if evidence_steps == input_len else 1
    if scenario == "fault":
        if evidence_steps == 1:
            return 3
        return 5 if evidence_steps == input_len else 4
    raise ValueError(f"unsupported LCBT scenario: {scenario}")


def _weighted_degree(adjacency):
    weights = np.nan_to_num(np.abs(np.asarray(adjacency, dtype=np.float64)), nan=0.0)
    np.fill_diagonal(weights, 0.0)
    return weights.sum(axis=0) + weights.sum(axis=1)


def _degree_strata_for(graph_degree, pool):
    """Split pool members into low/mid/high thirds of their full-graph degree.

    Returned indices are positions inside ``pool``, which is what the selectors
    work in, but the ordering comes from the network-wide degree.
    """
    pool = np.asarray(pool, dtype=np.int64)
    local = np.arange(len(pool), dtype=np.int64)
    order = local[np.lexsort((pool, graph_degree[pool]))]
    return [np.asarray(part, dtype=np.int64) for part in np.array_split(order, 3)]


def _hop_distances(adjacency):
    """Hop distance on the graph the cohort is drawn from.

    Clustering has to be measured on the graph, not on the induced subgraph of
    the eligible pool: a pool of scattered unseen sensors has almost no internal
    edges, so pool-internal connectivity cannot tell clustered from scattered.
    """
    from scipy.sparse.csgraph import shortest_path

    weights = np.nan_to_num(np.abs(np.asarray(adjacency, dtype=np.float64)), nan=0.0)
    np.fill_diagonal(weights, 0.0)
    undirected = ((weights > 0) | (weights.T > 0)).astype(np.float64)
    return shortest_path(undirected, method="D", unweighted=True, directed=False)


def _least_exposed(candidates, exposure, rng, count):
    candidates = np.asarray(candidates, dtype=np.int64)
    if count <= 0 or len(candidates) == 0:
        return []
    jitter = rng.random(len(candidates))
    order = np.lexsort((candidates, jitter, exposure[candidates]))
    return candidates[order[: min(int(count), len(candidates))]].tolist()


def _degree_quotas(strata, cohort_size):
    return _allocate_counts(cohort_size, [max(len(stratum), 1) for stratum in strata])


def _select_scattered_nodes(strata, exposure, rng, cohort_size):
    selected = []
    for stratum, count in zip(strata, _degree_quotas(strata, cohort_size)):
        selected.extend(_least_exposed(stratum, exposure, rng, int(count)))
    if len(selected) < cohort_size:
        remaining = np.setdiff1d(np.arange(len(exposure)), selected, assume_unique=False)
        selected.extend(_least_exposed(remaining, exposure, rng, cohort_size - len(selected)))
    return sorted({int(node) for node in selected})


def _select_clustered_nodes(hops, strata, exposure, rng, cohort_size, seed_stratum):
    """Grow by hop proximity while honouring the same degree quotas as scattered.

    Both styles now respect the low/mid/high degree split, so the two differ in
    spatial compactness alone rather than in compactness *and* degree mix.
    """
    nonempty = [stratum for stratum in strata if len(stratum)]
    if not nonempty:
        raise ValueError("cannot build a clustered cohort from an empty node pool")
    quotas = _degree_quotas(strata, cohort_size)
    stratum_of = {
        int(node): index for index, stratum in enumerate(strata) for node in stratum
    }
    seed_candidates = nonempty[int(seed_stratum) % len(nonempty)]
    seed = int(_least_exposed(seed_candidates, exposure, rng, 1)[0])
    remaining = {index: int(quota) for index, quota in enumerate(quotas)}
    remaining[stratum_of[seed]] -= 1
    selected = [seed]
    available = set(range(len(exposure))) - {seed}
    while len(selected) < cohort_size and available:
        candidates = np.asarray(
            sorted(node for node in available if remaining.get(stratum_of[node], 0) > 0),
            dtype=np.int64,
        )
        if not len(candidates):
            candidates = np.asarray(sorted(available), dtype=np.int64)
        spread = hops[np.ix_(candidates, np.asarray(selected, dtype=np.int64))]
        spread = np.where(np.isfinite(spread), spread, hops.shape[0]).mean(axis=1)
        jitter = rng.random(len(candidates))
        chosen = int(candidates[np.lexsort((candidates, jitter, exposure[candidates], spread))[0]])
        selected.append(chosen)
        remaining[stratum_of[chosen]] = remaining.get(stratum_of[chosen], 0) - 1
        available.remove(chosen)
    return sorted(selected)


def global_evidence_quota(total, input_len, rotation=0):
    """A multiset of k=1..L whose per-k counts differ by at most one, globally.

    The v3 schedule allocated k inside each (scenario, calendar regime) cell
    independently. `_allocate_counts` breaks its remainder ties by index, so
    every cell handed its leftovers to the smallest k, and the bias accumulated:
    measured on METR-LA the Join counts ran 361 at k=1 down to 352 at k=12,
    which is a systematic tilt towards short evidence rather than the uniform
    coverage the protocol claims.

    Here the quota is computed once for the whole scenario, and the remainder
    walks a rotating offset so it does not always land on the same k.
    """
    total, input_len = int(total), int(input_len)
    if input_len <= 0:
        raise ValueError("input_len must be positive")
    counts = np.full(input_len, total // input_len, dtype=np.int64)
    remainder = total - int(counts.sum())
    for step in range(remainder):
        counts[(int(rotation) + step) % input_len] += 1
    values = []
    for index, count in enumerate(counts):
        values.extend([index + 1] * int(count))
    return np.asarray(values, dtype=np.int64)


def assign_global_evidence(indices_by_regime, input_len, rng, rotation=0):
    """Deal one global k quota across calendar regimes, interleaved.

    Interleaving matters: dealing the quota regime by regime would give the
    first regime all the low k. Round-robin over regimes keeps each regime's k
    distribution close to uniform while the *global* counts stay exact.
    """
    regimes = sorted(indices_by_regime)
    # A two-dimensional quota, regime x k. Each regime gets floor(n_r / L) of
    # every evidence length plus a run of `n_r % L` extras taken from a single
    # cursor that carries across regimes. That makes both margins exact at once:
    #
    #   within a regime  every k appears base_r or base_r + 1 times
    #   across regimes   the extras walk a contiguous rotation, so the global
    #                    counts also differ by at most one
    #
    # Dealing a shuffled global quota instead balances only in expectation
    # (measured range 8 across 12 lengths at 120 samples per regime), and
    # dealing cyclically pairs regime with k whenever the regime count divides
    # L -- three regimes at L=12 would give one regime only k in {1,4,7,10}.
    cursor = int(rotation) % input_len
    order, quota = [], []
    for regime in regimes:
        bucket = np.asarray(indices_by_regime[regime], dtype=np.int64)
        count = len(bucket)
        if not count:
            continue
        values = np.tile(np.arange(1, input_len + 1), count // input_len)
        extras = count % input_len
        if extras:
            values = np.concatenate([
                values,
                np.asarray(
                    [((cursor + step) % input_len) + 1 for step in range(extras)],
                    dtype=np.int64,
                ),
            ])
            cursor = (cursor + extras) % input_len
        # Shuffle the *assignment* inside the regime, not the counts: which
        # sample gets which k is arbitrary, how many get each k is not.
        order.extend(bucket[rng.permutation(count)].tolist())
        quota.extend(values[rng.permutation(count)].tolist())
    return np.asarray(order, dtype=np.int64), np.asarray(quota, dtype=np.int64)


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
    scenario_ratios=(0.25, 0.25, 0.25, 0.25),
    cohort_fractions=(0.025, 0.05, 0.10),
    cohort_weights=(0.25, 0.50, 0.25),
    clustered_ratio=0.50,
    scenario_node_pools=None,
    cohort_size_base=None,
    topology_adjacency=None,
    node_ids=None,
    stratify_by_evidence=True,
    hidden_fraction=0.0,
):
    """Build the frozen LCBT schedule with uniform evidence k=1..L.

    ``cohort_size_base`` fixes what a fraction is a fraction *of*. It defaults to
    the graph handed in, which used to make "5%" mean 8 nodes during training
    (157-node graph) and 10 nodes at validation/test (207-node graph). Passing
    the full node count makes one absolute size hold everywhere.

    ``topology_adjacency`` is the graph that defines degree and hop distance.
    Node degree is a property of the road network, not of which sensors happen
    to be visible in a split, so passing the full adjacency keeps "high-degree
    node" and "clustered cohort" meaning the same thing in training, validation
    and test. It defaults to ``adjacency`` for standalone use.

    ``node_ids`` maps a schedule node index onto its row in
    ``topology_adjacency``. Training runs on the known-node induced graph, whose
    indices are local, so the map is what stops a local index from being read as
    a full-graph node id.

    ``hidden_fraction`` turns each training window into an episode with its own
    held-out sensors. Without it the model trains on a graph where nothing is
    permanently missing and only a handful of cohort nodes are perturbed, then
    meets an evaluation graph where a quarter of the sensors have never been
    observed -- and its synthetic Join events sit on nodes it has seen in other
    windows, so they are not really unseen. With it, every window hides a
    rotating subset of the graph for its whole length, Join and ZeroHistory
    cohorts are drawn from that hidden subset, and Steady and Fault cohorts from
    the visible remainder. Set it to the evaluation hidden rate so the training
    and evaluation graphs are equally sparse.

    ``stratify_by_evidence`` controls how finely cohort size and topology are
    balanced. Training has thousands of samples per stratum, so balancing them
    inside each evidence length is free. The validation replay has roughly one
    sample per (stratum, evidence) cell and replays every k from one cohort, so
    balancing per evidence there would ask ``_allocate_counts`` to split a single
    sample across the size tiers, which always awards it to the heaviest tier and
    wipes the 2.5% cohorts out entirely.
    """
    num_samples = int(num_samples)
    input_len = int(input_len)
    adjacency = np.asarray(adjacency)
    calendar_regimes = [str(value) for value in calendar_regimes]
    if num_samples <= 0 or input_len <= 2:
        raise ValueError("LCBT requires num_samples > 0 and input_len > 2")
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("adjacency must be square")
    if len(calendar_regimes) != num_samples:
        raise ValueError("calendar_regimes must align with num_samples")
    if len(scenario_ratios) != len(LCBT_GROUPS):
        raise ValueError("LCBT requires four scenario ratios")
    if not 0.0 <= float(clustered_ratio) <= 1.0:
        raise ValueError("clustered_ratio must be in [0, 1]")
    if scenario_node_pools is not None:
        scenario_node_pools = {
            str(name): np.asarray(sorted({int(node) for node in nodes}), dtype=np.int64)
            for name, nodes in scenario_node_pools.items()
        }
        missing_pools = [name for name in LCBT_GROUPS if name not in scenario_node_pools]
        if missing_pools:
            raise ValueError(
                "LCBT scenario_node_pools missing: " + ", ".join(missing_pools)
            )
        if any(len(nodes) == 0 for nodes in scenario_node_pools.values()):
            raise ValueError("LCBT scenario node pools must be non-empty")

    rng = np.random.default_rng(int(seed) + int(epoch) * 1_000_003)
    assignments = _scenario_assignments(
        calendar_regimes,
        scenario_ratios,
        rng,
        scenario_names=LCBT_GROUPS,
    )
    evidence_steps = np.zeros(num_samples, dtype=np.int64)
    regime_array = np.asarray(calendar_regimes, dtype=object)
    for scenario_index, scenario in enumerate(("join", "fault")):
        indices_by_regime = {}
        for regime in sorted(set(calendar_regimes)):
            indices = np.flatnonzero(
                (assignments == scenario) & (regime_array == regime)
            )
            if len(indices):
                indices_by_regime[regime] = indices[rng.permutation(len(indices))]
        if not indices_by_regime:
            raise ValueError(
                f"LCBT: scenario {scenario!r} received no samples, so its "
                "evidence coverage cannot be built"
            )
        # The rotation differs per scenario and per epoch, so the remainder does
        # not sit on the same k every time either.
        rotation = (scenario_index + int(epoch)) % input_len
        order, quota = assign_global_evidence(
            indices_by_regime, input_len, rng, rotation=rotation
        )
        evidence_steps[order] = quota

    fractions = np.empty(num_samples, dtype=np.float64)
    topology_modes = np.empty(num_samples, dtype=object)
    calendar_array = np.asarray(calendar_regimes, dtype=object)
    for scenario in LCBT_GROUPS:
        for regime in sorted(set(calendar_regimes)):
            scenario_indices = np.flatnonzero(
                (assignments == scenario) & (calendar_array == regime)
            )
            evidence_values = (
                range(1, input_len + 1)
                if stratify_by_evidence and scenario in {"join", "fault"}
                else (None,)
            )
            for evidence in evidence_values:
                indices = (
                    scenario_indices
                    if evidence is None
                    else scenario_indices[evidence_steps[scenario_indices] == int(evidence)]
                )
                if not len(indices):
                    continue
                fractions[indices] = np.asarray(
                    _balanced_values(
                        len(indices), cohort_fractions, cohort_weights, rng
                    ),
                    dtype=np.float64,
                )
                topology_modes[indices] = np.asarray(
                    _balanced_values(
                        len(indices),
                        ("clustered", "scattered"),
                        (float(clustered_ratio), 1.0 - float(clustered_ratio)),
                        rng,
                    ),
                    dtype=object,
                )
    exposure = np.zeros(adjacency.shape[0], dtype=np.int64)
    size_base = int(cohort_size_base or adjacency.shape[0])
    topology = np.asarray(adjacency if topology_adjacency is None else topology_adjacency)
    topology_ids = (
        np.arange(adjacency.shape[0], dtype=np.int64)
        if node_ids is None
        else np.asarray(node_ids, dtype=np.int64)
    )
    if len(topology_ids) != adjacency.shape[0]:
        raise ValueError("node_ids must give one topology row per schedule node")
    if topology_ids.max(initial=-1) >= topology.shape[0]:
        raise ValueError("node_ids fall outside topology_adjacency")
    # Degree strata and hop distances both come from the topology graph and are
    # only *restricted* to the pool afterwards, never recomputed on the induced
    # subgraph: a pool of scattered unseen sensors has almost no internal edges,
    # so its induced degrees and distances describe the pool, not the network.
    graph_degree = _weighted_degree(topology)
    graph_hops = _hop_distances(topology)

    hidden_fraction = float(hidden_fraction)
    if hidden_fraction and scenario_node_pools is not None:
        raise ValueError(
            "hidden_fraction builds its own per-window pools; it cannot be combined "
            "with fixed scenario_node_pools"
        )
    if not 0.0 <= hidden_fraction < 1.0:
        raise ValueError("hidden_fraction must be in [0, 1)")
    all_nodes = np.arange(adjacency.shape[0], dtype=np.int64)
    hidden_size = int(round(hidden_fraction * adjacency.shape[0]))
    # A separate exposure counter so held-out duty rotates evenly over the graph
    # rather than repeatedly falling on the same sensors.
    hidden_exposure = np.zeros(adjacency.shape[0], dtype=np.int64)
    all_strata = _degree_strata_for(graph_degree, topology_ids[all_nodes])

    schedule = []
    for sample_index in range(num_samples):
        scenario = str(assignments[sample_index])
        cohort_fraction = float(fractions[sample_index])
        topology_mode = str(topology_modes[sample_index])

        hidden_nodes = []
        if hidden_size > 0:
            hidden_local = _select_scattered_nodes(
                all_strata,
                hidden_exposure,
                rng,
                hidden_size,
            )
            hidden_nodes = sorted(int(all_nodes[index]) for index in hidden_local)
            hidden_exposure[np.asarray(hidden_nodes, dtype=np.int64)] += 1

        if scenario_node_pools is not None:
            pool = scenario_node_pools[scenario]
        elif hidden_nodes:
            # Join and ZeroHistory must land on sensors this window never showed
            # the model; Steady and Fault on sensors it did.
            hidden_array = np.asarray(hidden_nodes, dtype=np.int64)
            pool = (
                hidden_array
                if scenario in {"join", "zero_history"}
                else np.setdiff1d(all_nodes, hidden_array, assume_unique=True)
            )
        else:
            pool = all_nodes
        cohort_size = min(
            len(pool),
            max(1, int(round(size_base * cohort_fraction))),
        )
        topology_pool = topology_ids[pool]
        strata = _degree_strata_for(graph_degree, topology_pool)
        pool_hops = graph_hops[np.ix_(topology_pool, topology_pool)]
        pool_exposure = exposure[pool].copy()
        if topology_mode == "clustered":
            selected_local = _select_clustered_nodes(
                pool_hops,
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
                "phase_id": _phase_id(scenario, evidence, input_len),
                "changed_nodes": changed_nodes,
                "hidden_nodes": hidden_nodes,
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


def _calendar_balanced_state_mean(cell_state, label):
    """Mean within each stratum, then the calendar-share weighted combination.

    Equal weight over four regimes used to hand a validation Weekend cell built
    from three origins a full quarter of the checkpoint metric.
    """
    calendar_values = cell_state.get(label, {})
    return weighted_stratum_mean(
        {
            STRATA[int(calendar_id)] if 0 <= int(calendar_id) < len(STRATA) else str(calendar_id):
            _finite_mean(values)
            for calendar_id, values in sorted(calendar_values.items())
        }
    )


def finalize_lifecycle_validation_mae(cell_state, input_len):
    """Equal-average calendar cells, evidence lengths, then four semantic groups."""
    input_len = int(input_len)
    if input_len <= 2:
        raise ValueError("LCBT v3 validation requires input_len > 2")
    unseen = _calendar_balanced_state_mean(cell_state, "Unseen")
    steady = _calendar_balanced_state_mean(cell_state, "Steady")
    join_by_k = [
        _calendar_balanced_state_mean(cell_state, f"Join@{k}")
        for k in range(1, input_len + 1)
    ]
    fault_by_k = [
        _calendar_balanced_state_mean(cell_state, f"Fault@{k}")
        for k in range(1, input_len + 1)
    ]
    join = _finite_mean(join_by_k)
    fault = _finite_mean(fault_by_k)
    group_mae = {
        "unseen_mae": unseen,
        "steady_mae": steady,
        "join_mae": join,
        "fault_mae": fault,
    }
    complete = np.asarray(list(group_mae.values()), dtype=np.float64)
    group_mae["mae"] = (
        float(complete.mean()) if np.all(np.isfinite(complete)) else float("nan")
    )
    group_mae["worst_group_mae"] = (
        float(complete.max()) if np.all(np.isfinite(complete)) else float("nan")
    )
    group_mae.update(
        {
            "Unseen_MAE": unseen,
            "Steady_MAE": steady,
            "Cold_MAE": join_by_k[0],
            "Warm_MAE": _finite_mean(join_by_k[1:-1]),
            "Ready_MAE": join_by_k[-1],
            "Onset_MAE": fault_by_k[0],
            "Decay_MAE": _finite_mean(fault_by_k[1:-1]),
            "Outage_MAE": fault_by_k[-1],
        }
    )
    for k, value in enumerate(join_by_k, start=1):
        group_mae[f"Join_k{k}_MAE"] = value
    for k, value in enumerate(fault_by_k, start=1):
        group_mae[f"Fault_k{k}_MAE"] = value
    return group_mae


def accumulate_lifecycle_mae(
    cell_state,
    y_true,
    y_pred,
    group_ids,
    phase_ids,
    target_cohort_mask,
    calendar_ids,
    evidence_steps,
):
    """Accumulate one original-scale MAE per replay cell, not per tensor element."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    group_ids = np.asarray(group_ids).reshape(-1).astype(np.int64)
    phase_ids = np.asarray(phase_ids).reshape(-1).astype(np.int64)
    calendar_ids = np.asarray(calendar_ids).reshape(-1).astype(np.int64)
    evidence_steps = np.asarray(evidence_steps).reshape(-1).astype(np.int64)
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
            evidence = int(evidence_steps[sample_index])
            if evidence <= 0:
                raise ValueError("event replay cell is missing valid evidence_steps")
            label = f"{'Join' if group_id == 2 else 'Fault'}@{evidence}"
        else:
            raise ValueError(f"unsupported lifecycle validation group_id: {group_id}")
        calendar_id = int(calendar_ids[sample_index])
        cell_state.setdefault(label, {}).setdefault(calendar_id, []).append(
            float(errors[sample_index][selected].mean())
        )


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
