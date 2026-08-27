"""Deterministic scenario, evidence-length and cohort schedule for v4."""

import numpy as np

from data_provider.calendar_strata import STRATA


LCBT_GROUPS = ("zero_history", "steady", "join", "fault")


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


def topology_statistics(adjacency):
    """Compute the immutable graph statistics reused by every schedule epoch."""
    adjacency = np.asarray(adjacency)
    degree = np.asarray(_weighted_degree(adjacency), dtype=np.float64)
    hops = np.asarray(_hop_distances(adjacency), dtype=np.float64)
    degree.setflags(write=False)
    hops.setflags(write=False)
    return degree, hops


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
    topology_degree=None,
    topology_hops=None,
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
    graph_degree = (_weighted_degree(topology) if topology_degree is None
                    else np.asarray(topology_degree, dtype=np.float64))
    graph_hops = (_hop_distances(topology) if topology_hops is None
                  else np.asarray(topology_hops, dtype=np.float64))
    if graph_degree.shape != (topology.shape[0],):
        raise ValueError("topology_degree does not match topology_adjacency")
    if graph_hops.shape != topology.shape:
        raise ValueError("topology_hops does not match topology_adjacency")

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
