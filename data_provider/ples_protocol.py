import hashlib
import json
from collections import defaultdict

import numpy as np

from data_provider.calendar_strata import (
    DAY_TYPES,
    STRATA,
    TOD_BINS,
    as_datetime,
    lce_episode_span,
    select_stratified_origins,
    stratum_weights,
    strip_diagnostics,
    summarize_origins,
    timestamp_text,
    weighted_stratum_mean,
)


LCE_PROTOCOL_NAME = "Lifecycle Coverage Evaluation"
LCE_PROTOCOL_VERSION = 3
PLES_PROTOCOL_NAME = "Paired Lifecycle Event Study v3"
PLES_PROTOCOL_VERSION = 3
TRACK_PHASES = {
    "join": ("Cold", "Warm", "Ready"),
    "fault": ("Onset", "Decay", "Outage"),
}
PHASE_ORDER = TRACK_PHASES["join"] + TRACK_PHASES["fault"]
PLES_EFFECT_NAMES = ("JoinLift", "FaultPenalty", "OutageFaultPenalty")
MAIN_TABLE_ROWS = ("Unseen", "Steady", "Join", "Fault", "LifeAvg", "WorstGroup")


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


def _time_step_minutes(timestamps):
    if len(timestamps) < 2:
        return None
    seconds = [
        (as_datetime(timestamps[index + 1]) - as_datetime(timestamps[index])).total_seconds()
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
        raise ValueError("PLES v3 requires input_len > 2")
    if tau < 0:
        return "Pre"
    initial, transition, settled = TRACK_PHASES[track]
    if tau == 0:
        return initial
    if tau < input_len - 1:
        return transition
    return settled




def select_event_origins(
    timestamps,
    timeline_length,
    input_len,
    pred_len,
    pre_steps,
    post_steps,
    per_stratum=1,
    values=None,
    min_observed_fraction=0.30,
):
    """Stratified low-discrepancy origins, shared with the validation replay."""
    if timestamps is not None and len(np.asarray(timestamps)) != int(timeline_length):
        raise ValueError("timestamps must align with the evaluation timeline")
    origins = select_stratified_origins(
        timestamps,
        timeline_length,
        input_len,
        pred_len,
        pre_steps=pre_steps,
        post_steps=post_steps,
        per_stratum=per_stratum,
        values=values,
        min_observed_fraction=min_observed_fraction,
    )
    # `regime` is kept as the row-level column name so the frozen CSV schema and
    # every downstream grouping keep working; its content is now the stratum.
    for origin in origins:
        origin["regime"] = origin["stratum"]
    return origins


def _allocate_integer_counts(total, weights):
    weights = np.asarray(weights, dtype=np.float64)
    exact = int(total) * weights / weights.sum()
    counts = np.floor(exact).astype(np.int64)
    for index in np.argsort(-(exact - counts), kind="stable")[: int(total) - int(counts.sum())]:
        counts[index] += 1
    return counts


def graph_hop_distances(adjacency):
    """All-pairs hop distance on the full graph, treating edges as undirected.

    Cohort compactness has to be judged on the full road network. The old
    implementation measured connectivity only among pool members, and the
    test-unseen pool has almost no internal edges, so every "clustered" Join
    cohort came out as sparse as a scattered one (1-7 internal edges against a
    possible 210). Distance on the full graph separates the two styles for both
    tracks.
    """
    from scipy.sparse.csgraph import shortest_path

    weights = np.nan_to_num(np.abs(np.asarray(adjacency, dtype=np.float64)), nan=0.0)
    np.fill_diagonal(weights, 0.0)
    undirected = ((weights > 0) | (weights.T > 0)).astype(np.float64)
    return shortest_path(undirected, method="D", unweighted=True, directed=False)


def _select_cohort(nodes, hops, degree, size, style, seed, seed_stratum):
    """Degree-stratified cohort selection; `clustered` also minimises hop spread."""
    nodes = np.asarray(sorted({int(node) for node in nodes}), dtype=np.int64)
    size = min(len(nodes), max(1, int(size)))
    rng = np.random.default_rng(int(seed))
    ordered = nodes[np.lexsort((nodes, degree[nodes]))]
    strata = [np.asarray(part, dtype=np.int64) for part in np.array_split(ordered, 3)]
    quotas = _allocate_integer_counts(size, [max(len(part), 1) for part in strata])

    if style == "scattered":
        selected = []
        for part, quota in zip(strata, quotas):
            shuffled = part[rng.permutation(len(part))]
            selected.extend(int(node) for node in shuffled[: int(quota)])
        return sorted(selected)
    if style != "clustered":
        raise ValueError(f"unsupported cohort style: {style}")

    # Grow from a seed by shortest hop distance, but keep the same per-degree
    # quotas the scattered style honours so the two differ only in compactness.
    nonempty = [part for part in strata if len(part)]
    seed_candidates = nonempty[int(seed_stratum) % len(nonempty)]
    seed_node = int(seed_candidates[int(seed) % len(seed_candidates)])
    remaining = {index: int(quota) for index, quota in enumerate(quotas)}
    stratum_of = {
        int(node): index for index, part in enumerate(strata) for node in part
    }
    selected = [seed_node]
    remaining[stratum_of[seed_node]] -= 1
    available = {int(node) for node in nodes} - {seed_node}
    while len(selected) < size and available:
        candidates = np.asarray(
            sorted(node for node in available if remaining.get(stratum_of[node], 0) > 0),
            dtype=np.int64,
        )
        if not len(candidates):
            candidates = np.asarray(sorted(available), dtype=np.int64)
        spread = hops[np.ix_(candidates, np.asarray(selected, dtype=np.int64))]
        spread = np.where(np.isfinite(spread), spread, hops.shape[0]).mean(axis=1)
        jitter = rng.random(len(candidates))
        chosen = int(candidates[np.lexsort((candidates, jitter, spread))[0]])
        selected.append(chosen)
        remaining[stratum_of[chosen]] = remaining.get(stratum_of[chosen], 0) - 1
        available.remove(chosen)
    return sorted(selected)


def cohort_compactness(nodes, hops):
    nodes = sorted(int(node) for node in nodes)
    if len(nodes) < 2:
        return 0.0
    block = hops[np.ix_(nodes, nodes)]
    finite = block[np.triu_indices(len(nodes), k=1)]
    finite = finite[np.isfinite(finite)]
    return float(finite.mean()) if len(finite) else float("inf")


# A cohort covering more than this share of its eligible pool stops behaving
# like a sample of that pool: the eight cohorts then overlap so heavily that
# resampling them captures almost no node-level variation.
POOL_SATURATION_LIMIT = 0.50


def cohort_size_for(num_nodes, fraction):
    """One size rule for train, validation and test."""
    return max(1, int(round(int(num_nodes) * float(fraction))))


def main_table_cohort_plan(
    num_nodes,
    pool_sizes,
    cohort_fractions=(0.025, 0.05, 0.10),
    cohort_weights=(0.25, 0.50, 0.25),
    limit=POOL_SATURATION_LIMIT,
):
    """Split the requested tiers into the pooled main table and the large-event row.

    The validation checkpoint and the LCE main table have to agree on which
    tiers they average, otherwise training optimises a target the paper never
    reports. Both call this, so the split is decided once from the pool sizes.

    Returns the retained fractions with weights renormalised over them, the
    dropped fractions, and the realised weight per retained tier so the manifest
    can state it instead of leaving it implicit in how many cohorts share a size.
    """
    pool_sizes = [int(value) for value in pool_sizes if int(value) > 0]
    if not pool_sizes:
        raise ValueError("cohort planning needs at least one non-empty pool")
    kept, kept_weights, dropped = [], [], []
    for fraction, weight in zip(cohort_fractions, cohort_weights):
        size = cohort_size_for(num_nodes, fraction)
        share = max(min(size, pool) / pool for pool in pool_sizes)
        if share > float(limit):
            dropped.append({"fraction": float(fraction), "max_pool_share": float(share)})
        else:
            kept.append(float(fraction))
            kept_weights.append(float(weight))
    if not kept:
        raise ValueError(
            "every requested cohort fraction saturates its pool; lower the fractions"
        )
    total = sum(kept_weights)
    normalised = [value / total for value in kept_weights]
    return {
        "fractions": tuple(kept),
        "weights": tuple(normalised),
        "dropped": dropped,
        "realised_weights": {f"{f:.4f}": w for f, w in zip(kept, normalised)},
        "pool_saturation_limit": float(limit),
    }


def build_lifecycle_cohorts(
    adjacency,
    known_nodes,
    test_unseen_nodes,
    cohort_fractions=(0.025, 0.05, 0.10),
    cohort_weights=(0.25, 0.50, 0.25),
    seed=2026,
):
    """Cohorts sized as a share of the full graph, audited against their pool.

    Sizes stay defined on the full graph because that is the physically
    meaningful quantity ("a 5% sensor outage"). What changes is that every
    cohort now also reports its share of the eligible pool, and any cohort above
    ``POOL_SATURATION_LIMIT`` is flagged so the reporting layer can keep it out
    of the pooled Join/Fault averages. On METR-LA the 10% cohort takes 21 of the
    30 test-unseen nodes, and the eight cohorts together cover 29 of them, so it
    cannot be treated as one draw among many.
    """
    adjacency = np.asarray(adjacency)
    num_nodes = int(adjacency.shape[0])
    if len(cohort_fractions) != len(cohort_weights):
        raise ValueError("cohort fractions and weights must align")
    # Tier weights are realised by how many cohorts carry each size, so the
    # count vector is the weighting; `main_table_cohort_plan` records the
    # resulting share explicitly in the manifest.
    fraction_counts = _allocate_integer_counts(4, cohort_weights)
    fraction_plan = []
    for fraction, count in zip(cohort_fractions, fraction_counts):
        fraction_plan.extend([float(fraction)] * int(count))
    degree = weighted_graph_degree(adjacency)
    hops = graph_hop_distances(adjacency)
    join_pool = len({int(node) for node in test_unseen_nodes})
    fault_pool = len({int(node) for node in known_nodes})

    cohorts = []
    cohort_id = 1
    for style_index, style in enumerate(("clustered", "scattered")):
        for fraction_index, fraction in enumerate(fraction_plan):
            requested_size = cohort_size_for(num_nodes, fraction)
            join_nodes = _select_cohort(
                test_unseen_nodes,
                hops,
                degree,
                requested_size,
                style,
                seed=int(seed) + cohort_id * 17,
                seed_stratum=fraction_index,
            )
            fault_nodes = _select_cohort(
                known_nodes,
                hops,
                degree,
                requested_size,
                style,
                seed=int(seed) + cohort_id * 17 + 1,
                seed_stratum=fraction_index + style_index,
            )
            join_share = len(join_nodes) / max(join_pool, 1)
            fault_share = len(fault_nodes) / max(fault_pool, 1)
            saturated = max(join_share, fault_share) > POOL_SATURATION_LIMIT
            cohorts.append(
                {
                    "cohort_id": int(cohort_id),
                    "fraction": float(fraction),
                    "style": style,
                    "requested_size": int(requested_size),
                    "graph_share": float(len(join_nodes)) / float(num_nodes),
                    "join_pool_share": float(join_share),
                    "fault_pool_share": float(fault_share),
                    "pool_saturated": bool(saturated),
                    "table": "large_event" if saturated else "main",
                    "join_nodes": join_nodes,
                    "fault_nodes": fault_nodes,
                    "join_degree_mean": float(np.mean(degree[join_nodes])),
                    "fault_degree_mean": float(np.mean(degree[fault_nodes])),
                    "join_compactness_hops": cohort_compactness(join_nodes, hops),
                    "fault_compactness_hops": cohort_compactness(fault_nodes, hops),
                }
            )
            cohort_id += 1
    if not any(cohort["table"] == "main" for cohort in cohorts):
        raise ValueError("every cohort saturates its pool; lower the cohort fractions")
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
    lce_protocol_version=3,
    origins_per_stratum=1,
    min_observed_fraction=0.30,
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

    origins = select_event_origins(
        timestamps,
        timeline_length,
        input_len,
        pred_len,
        pre_steps,
        post_steps,
        per_stratum=origins_per_stratum,
        values=test_values,
        min_observed_fraction=min_observed_fraction,
    )
    if not origins:
        raise ValueError("evaluation timeline has no valid LCE/PLES origins")
    origin_summary = summarize_origins(origins)
    origins = strip_diagnostics(origins)
    cohorts = build_lifecycle_cohorts(
        adjacency,
        known_nodes,
        test_unseen_nodes,
        cohort_fractions=cohort_fractions,
        cohort_weights=cohort_weights,
        seed=seed,
    )
    timestamps_text = (
        [timestamp_text(value) for value in timestamps]
        if timestamps is not None
        else None
    )
    dataset_record = {
        "name": str(dataset_name),
        "num_nodes": num_nodes,
        "time_steps": int(timeline_length),
        "time_step_minutes": _time_step_minutes(timestamps) if timestamps is not None else None,
        "first_timestamp": timestamps_text[0] if timestamps_text else None,
        "last_timestamp": timestamps_text[-1] if timestamps_text else None,
        "adjacency_hash": _array_hash(adjacency),
        "timestamps_hash": _stable_hash(timestamps_text) if timestamps_text else None,
        "test_values_hash": _array_hash(test_values) if test_values is not None else None,
        "input_features": input_features,
        "target_features": target_features,
    }
    dataset_record["fingerprint"] = _stable_hash(dataset_record)
    if int(lce_protocol_version) != 3:
        raise ValueError("the maintained absolute evaluation protocol is LCE v3")
    lce = {
        "protocol": f"{LCE_PROTOCOL_NAME} v3",
        "event_clock": "input_end=event_time+tau+1",
        "anchors": ["Unseen", "Steady"],
        "phase_names": list(PHASE_ORDER),
        "main_rows": list(MAIN_TABLE_ROWS),
        "primary_endpoint": "LifeAvg_MAE",
        "free_rows": ["Unseen", "Steady", "Cold", "Warm", "Ready", "Onset", "Decay", "Outage"],
        "derived_rows": {
            "Join": "(Cold + (input_len-2)*Warm + Ready) / input_len",
            "Fault": "(Onset + (input_len-2)*Decay + Outage) / input_len",
            "LifeAvg": "mean(Unseen, Steady, Join, Fault)",
            "WorstGroup": "max(Unseen, Steady, Join, Fault)",
        },
        "aggregation": {
            "cell": "origin_x_cohort_x_tau",
            "strata": "mean within stratum, then equal over tod inside a day type, day types by calendar share 5/7 and 2/7",
            "Join": "equal mean over k=1..input_len",
            "Fault": "equal mean over k=1..input_len",
            "cohort_set": "pooled rows use non-saturated cohorts only",
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
        "protocol": PLES_PROTOCOL_NAME,
        "pre_event_usage": "implementation assertion only; identical by construction, never reported as a score",
        "post_event_effects": list(PLES_EFFECT_NAMES),
        "effect_metrics": ["MAPE", "MAE"],
        # PeakFaultPenalty was a max over tau. The bootstrap distribution of a
        # maximum is badly behaved, so the settled-outage cell is reported
        # instead: it is the same substantive claim on a smooth statistic.
        "outage_definition": "FaultPenalty at tau=input_len-1",
        "bootstrap": {
            "design": "origins and cohorts are crossed, not nested",
            "outer_unit": "overlap-connected origin block, stratified by day_type",
            "outer_reason": "origins whose pre+input+post+horizon spans touch are chained into one block; day blocking missed 24 of 175 adjacent pairs that overlap across midnight",
            "trajectory_span": int(pre_steps) + int(input_len) + int(post_steps) + int(pred_len),
            "cohort_resampling": "one global draw per bootstrap sample, applied to every sampled block",
            "samples": int(bootstrap_samples),
            "seed": int(bootstrap_seed),
            "confidence": 0.95,
            "paired_across_models": "scripts/paired_model_bootstrap.py",
        },
        "causal_claim": False,
    }
    main_cohorts = [c for c in cohorts if c["table"] == "main"]
    large_event_cohorts = [c for c in cohorts if c["table"] == "large_event"]
    manifest = {
        "schema_version": 3,
        # The three version independently: only LCBT changed, so only it moved.
        "protocol": "LCBT v3.2 + LCE v3 + PLES v3",
        "dataset": dataset_record,
        "nodes": {
            "known": known_nodes,
            "master_unseen": master_unseen_nodes,
            "validation_unseen": validation_unseen_nodes,
            "test_unseen": test_unseen_nodes,
        },
        "requested": {
            "calendar_strata": list(STRATA),
            "day_types": list(DAY_TYPES),
            "tod_bins": [[name, start, end] for name, start, end in TOD_BINS],
            "stratum_weights": stratum_weights(),
            "cohort_fractions": [float(value) for value in cohort_fractions],
            "cohort_weights": [float(value) for value in cohort_weights],
            "cohort_size_base": num_nodes,
            "main_table_cohort_plan": main_table_cohort_plan(
                num_nodes,
                [len(known_nodes), len(test_unseen_nodes)],
                cohort_fractions=cohort_fractions,
                cohort_weights=cohort_weights,
            ),
            "cohort_styles": ["clustered", "scattered"],
            "pool_saturation_limit": POOL_SATURATION_LIMIT,
            "origins_per_stratum": int(origins_per_stratum),
            "min_observed_fraction": float(min_observed_fraction),
            "pre_steps": int(pre_steps),
            "post_steps": int(post_steps),
            "seed": int(seed),
        },
        "resolved": {
            "origin_count": int(len(origins)),
            "cohort_count": int(len(cohorts)),
            "main_cohort_ids": [int(c["cohort_id"]) for c in main_cohorts],
            "large_event_cohort_ids": [int(c["cohort_id"]) for c in large_event_cohorts],
            "episode_count_per_track": int(len(origins) * len(cohorts)),
            "input_len": int(input_len),
            "pred_len": int(pred_len),
            "tau_range": [-int(pre_steps), int(post_steps)],
            "origin_selection": {
                "rule": "stratified_low_discrepancy",
                "sequence": "golden_ratio_additive_recurrence",
                "stratum_order": "chronological",
                "min_spacing": lce_episode_span(input_len, pred_len),
                "min_spacing_scope": "global; LCE episodes (tau=0..input_len-1) are disjoint",
                "known_overlap": "PLES tails beyond tau=input_len-1 may overlap between origins",
                **origin_summary,
            },
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
    """Mean inside each stratum, then the calendar-share weighted combination."""
    by_stratum = defaultdict(list)
    for row in rows:
        metrics = _row_metrics_v2(row, prefix)
        if metrics is not None:
            by_stratum[str(row.get("regime", "all"))].append(metrics)
    if not by_stratum:
        return None
    stratum_metrics = {
        name: _mean_metrics_v2(values) for name, values in by_stratum.items() if values
    }
    combined = {
        metric: weighted_stratum_mean(
            {name: entry[metric] for name, entry in stratum_metrics.items()}
        )
        for metric in ("MAPE", "MAE", "RMSE")
    }
    return None if not all(np.isfinite(value) for value in combined.values()) else combined


def _tau_balanced_metrics_v2(rows, prefix):
    by_tau = defaultdict(list)
    for row in rows:
        by_tau[int(row["tau"])].append(row)
    tau_metrics = [
        _calendar_balanced_metrics_v2(by_tau[tau], prefix)
        for tau in sorted(by_tau)
    ]
    return _mean_metrics_v2([metrics for metrics in tau_metrics if metrics is not None])


def _worst_group(groups):
    return {
        metric: float(np.max([groups[name][metric] for name in ("Unseen", "Steady", "Join", "Fault")]))
        for metric in ("MAPE", "MAE", "RMSE")
    }


def summarize_lce_rows(rows, input_len, main_cohort_ids=None):
    """Absolute accuracy per cell, stratum, tau, then semantic group.

    ``main_cohort_ids`` restricts the pooled rows to cohorts that are still a
    sample of their eligible pool. Saturated cohorts are summarised separately
    as ``large_event_metrics`` instead of being folded into Join/Fault/LifeAvg,
    because averaging a near-census together with genuine samples produces an
    interval that resampling cannot justify.
    """
    if int(input_len) <= 2:
        raise ValueError("LCE v3 requires input_len > 2")
    post_rows = [row for row in rows if 0 <= int(row["tau"]) < int(input_len)]
    if main_cohort_ids is None:
        main_rows, large_rows = post_rows, []
    else:
        wanted = {int(value) for value in main_cohort_ids}
        main_rows = [row for row in post_rows if int(row["cohort_id"]) in wanted]
        large_rows = [row for row in post_rows if int(row["cohort_id"]) not in wanted]
    if not main_rows:
        raise ValueError("LCE has no rows from non-saturated cohorts")

    phase_metrics = {}
    for phase in PHASE_ORDER:
        phase_rows = [row for row in main_rows if row.get("phase") == phase]
        metrics = _tau_balanced_metrics_v2(phase_rows, "treated")
        if metrics is None:
            raise ValueError(f"LCE phase has no valid cells: {phase}")
        phase_metrics[phase] = metrics

    join_rows = [row for row in main_rows if row.get("track") == "join"]
    fault_rows = [row for row in main_rows if row.get("track") == "fault"]
    groups = {
        "Unseen": _tau_balanced_metrics_v2(join_rows, "control"),
        "Steady": _tau_balanced_metrics_v2(fault_rows, "control"),
        "Join": _tau_balanced_metrics_v2(join_rows, "treated"),
        "Fault": _tau_balanced_metrics_v2(fault_rows, "treated"),
    }
    groups["LifeAvg"] = _mean_metrics_v2(list(groups.values()))
    groups["WorstGroup"] = _worst_group(groups)

    trajectory_metrics = {"Join": [], "Fault": []}
    for label, track in (("Join", "join"), ("Fault", "fault")):
        track_rows = [row for row in main_rows if row.get("track") == track]
        for tau in range(int(input_len)):
            metrics = _calendar_balanced_metrics_v2(
                [row for row in track_rows if int(row["tau"]) == tau],
                "treated",
            )
            if metrics is None:
                raise ValueError(f"LCE v3 {track} is missing tau={tau}")
            trajectory_metrics[label].append(
                {"k": tau + 1, "visibility": (tau + 1) / float(input_len), **metrics}
            )

    large_event_metrics = {}
    if large_rows:
        for label, track in (("Join", "join"), ("Fault", "fault")):
            metrics = _tau_balanced_metrics_v2(
                [row for row in large_rows if row.get("track") == track],
                "treated",
            )
            if metrics is not None:
                large_event_metrics[label] = metrics
    return {
        "phase_metrics": phase_metrics,
        "trajectory_metrics": trajectory_metrics,
        "main_metrics": groups,
        "large_event_metrics": large_event_metrics,
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


def paired_effect_cells(curves, input_len):
    """The exact post-event cells each PLES endpoint is computed over.

    The contrast and the treated/control arms must be read off the *same* cells.
    A PLES contrast is gameable on its own -- ``JoinLift`` is
    ``control - treated``, so degrading the control arm inflates it, and the
    Fault penalties are ``treated - control``, so degrading the control shrinks
    them. The arms are what distinguish a real improvement from that, and they
    only do so if they are selected identically. Both callers take their
    selection from here rather than repeating the filter.
    """
    post = [curve for curve in curves if int(curve["tau"]) >= 0]
    cells = {
        "JoinLift": [curve for curve in post if curve["track"] == "join"],
        "FaultPenalty": [curve for curve in post if curve["track"] == "fault"],
        "OutageFaultPenalty": [
            curve
            for curve in post
            if curve["track"] == "fault" and int(curve["tau"]) == int(input_len) - 1
        ],
    }
    if not cells["JoinLift"] or not cells["FaultPenalty"]:
        raise ValueError("PLES requires post-event Join and Fault cells")
    if not cells["OutageFaultPenalty"]:
        raise ValueError(f"PLES is missing the settled outage cell tau={int(input_len) - 1}")
    return cells


def paired_effects(curves, input_len, metric="MAPE"):
    """Post-event effects on one metric.

    ``OutageFaultPenalty`` replaces the old ``PeakFaultPenalty``. Both answer
    "how bad does a settled outage get", but a maximum over tau is a
    non-smooth functional whose percentile bootstrap is unreliable, whereas the
    settled cell at ``tau = input_len - 1`` is a plain mean. A maximum is also
    not a paired quantity: the two models' maxima land at different tau, so the
    difference of two maxima compares two different cells.
    """
    cells = paired_effect_cells(curves, input_len)
    return {
        name: float(np.mean([curve["effect"][metric] for curve in group]))
        for name, group in cells.items()
    }


def paired_effect_arms(curves, input_len, metric="MAPE"):
    """Treated and control absolute error behind each effect, same cells.

    Reported beside the contrast so a change in the contrast can be attributed:
    an endpoint that improves only because its control arm got worse is not an
    improvement, and without the arms that case is invisible.
    """
    cells = paired_effect_cells(curves, input_len)
    return {
        name: {
            arm: float(np.mean([curve[arm][metric] for curve in group]))
            for arm in ("treated", "control")
        }
        for name, group in cells.items()
    }


def _main_cohort_rows(rows, main_cohort_ids):
    if main_cohort_ids is None:
        return rows
    wanted = {int(value) for value in main_cohort_ids}
    return [row for row in rows if int(row["cohort_id"]) in wanted]


# Public names for the two steps a reader outside this module needs in order to
# reproduce a frozen effect: restrict to the main plan's non-saturated cohorts,
# then build the calendar-weighted curves. They exist so no caller has to write
# its own copy -- scripts/plot_counterfactual_shockwave.py did, and its copy
# drifted on all three of cohort eligibility, calendar weighting and the Outage
# cell.
main_cohort_rows = _main_cohort_rows
build_curve_rows = _curve_rows_v2


def summarize_paired_rows(rows, input_len, main_cohort_ids=None):
    """Effects are estimated on the same non-saturated cohorts as the main table."""
    curves = _curve_rows_v2(_main_cohort_rows(rows, main_cohort_ids))
    pre_differences = [
        abs(float(curve["treated"]["MAPE"]) - float(curve["control"]["MAPE"]))
        for curve in curves
        if int(curve["tau"]) < 0
    ]
    return {
        "paired_effects": {
            metric: paired_effects(curves, input_len, metric) for metric in ("MAPE", "MAE")
        },
        "pre_event_max_abs_mape_difference": max(pre_differences, default=0.0),
        "curves": curves,
    }


def trajectory_span(pre_steps, input_len, post_steps, pred_len):
    """Steps of data one PLES trajectory touches: pre + input + post + horizon.

    Single definition on purpose. It used to be spelled out at each call site,
    and one of those spellings used ``input_len`` twice, which is only correct
    when input_len == pred_len -- true for the current 12->12 setting and wrong
    the moment the horizon changes.
    """
    return int(pre_steps) + int(input_len) + int(post_steps) + int(pred_len)


def trajectory_span_from_manifest(manifest, default_input_len=None):
    """Read a span from a frozen manifest, declared or reconstructed."""
    declared = manifest.get("ples", {}).get("bootstrap", {})
    if isinstance(declared, dict) and declared.get("trajectory_span"):
        return int(declared["trajectory_span"])
    requested = manifest.get("requested", {})
    resolved = manifest.get("resolved", {})
    input_len = resolved.get("input_len", default_input_len)
    pred_len = resolved.get("pred_len")
    if (
        "pre_steps" in requested
        and "post_steps" in requested
        and input_len is not None
        and pred_len is not None
    ):
        return trajectory_span(
            requested["pre_steps"], input_len, requested["post_steps"], pred_len
        )
    return None


def overlap_blocks(origin_times, span):
    """Group origins whose trajectories overlap into transitively closed blocks.

    Calendar days are the intuitive block, but they do not contain the
    dependence: with origins spaced ``lce_episode_span`` apart and a PLES tail
    reaching ``pre + input + post + horizon``, neighbouring origins share data
    across midnight too. On METR-LA 24 of the 175 adjacent origin pairs overlap
    across a day boundary, so day blocking would treat correlated origins as
    independent and shrink the interval. Chaining any pair closer than ``span``
    into one block is the unit the data actually supports.
    """
    ordered = sorted(origin_times.items(), key=lambda item: item[1])
    assignment, block_index = {}, 0
    previous = None
    for origin_id, time_index in ordered:
        if previous is not None and time_index - previous >= int(span):
            block_index += 1
        assignment[origin_id] = block_index
        previous = time_index
    return assignment


def bootstrap_units(rows, span=None):
    """Resampling units that match how the design was actually sampled.

    Cohorts are *crossed* with origins, not nested inside them: the same frozen
    cohorts are replayed at every origin. They therefore have to be resampled
    once per draw and applied to every sampled block, not redrawn inside each
    origin, which would treat one factor as if it were nested in the other.
    """
    origin_times, origin_stratum, cohort_ids = {}, {}, set()
    for row in rows:
        origin = int(row["origin_id"])
        origin_times[origin] = int(row.get("event_time_index", origin))
        origin_stratum[origin] = str(row.get("regime", "all")).split("/")[0]
        cohort_ids.add(int(row["cohort_id"]))

    if span:
        assignment = overlap_blocks(origin_times, span)
        label = {origin: f"overlap-{index:04d}" for origin, index in assignment.items()}
    else:  # no trajectory span given: fall back to calendar days
        label = {
            int(row["origin_id"]): str(row.get("calendar_day", "all")) for row in rows
        }

    blocks, block_stratum = defaultdict(list), {}
    for row in rows:
        block = label[int(row["origin_id"])]
        blocks[block].append(row)
        # A block spanning midnight is labelled by its first origin's day type.
        block_stratum.setdefault(block, origin_stratum[int(row["origin_id"])])
    by_stratum = defaultdict(list)
    for block, stratum in sorted(block_stratum.items()):
        by_stratum[stratum].append(block)
    return {
        "blocks": {name: list(items) for name, items in sorted(blocks.items())},
        "blocks_by_stratum": {name: sorted(items) for name, items in sorted(by_stratum.items())},
        "cohort_ids": sorted(cohort_ids),
    }


def _draw_rows(units, rng):
    """One stratified block draw crossed with one global cohort draw."""
    cohort_ids = units["cohort_ids"]
    sampled_cohorts = collections_counter(
        rng.choice(cohort_ids, size=len(cohort_ids), replace=True)
    )
    sampled = []
    for _stratum, blocks in units["blocks_by_stratum"].items():
        for block in rng.choice(blocks, size=len(blocks), replace=True):
            for row in units["blocks"][str(block)]:
                repeats = sampled_cohorts.get(int(row["cohort_id"]), 0)
                if repeats:
                    sampled.extend([row] * repeats)
    return sampled


def collections_counter(values):
    counts = defaultdict(int)
    for value in values:
        counts[int(value)] += 1
    return counts


def bootstrap_paired_effects(rows, input_len, samples=2000, seed=2026, main_cohort_ids=None,
                             trajectory_span=None):
    rows = _main_cohort_rows(rows, main_cohort_ids)
    point = summarize_paired_rows(rows, input_len)["paired_effects"]
    metrics = ("MAPE", "MAE")
    if samples <= 0:
        return {
            metric: {
                name: {"point": point[metric][name], "low": None, "high": None}
                for name in PLES_EFFECT_NAMES
            }
            for metric in metrics
        }

    units = bootstrap_units(rows, span=trajectory_span)
    if not units["blocks"]:
        raise ValueError("cannot bootstrap empty PLES rows")

    rng = np.random.default_rng(seed)
    draws = {
        metric: {name: np.empty(int(samples), dtype=np.float64) for name in PLES_EFFECT_NAMES}
        for metric in metrics
    }
    for sample_index in range(int(samples)):
        curves = _curve_rows_v2(_draw_rows(units, rng))
        for metric in metrics:
            effects = paired_effects(curves, input_len, metric)
            for name in PLES_EFFECT_NAMES:
                draws[metric][name][sample_index] = effects[name]

    return {
        metric: {
            name: {
                "point": point[metric][name],
                "low": float(np.percentile(draws[metric][name], 2.5)),
                "high": float(np.percentile(draws[metric][name], 97.5)),
            }
            for name in PLES_EFFECT_NAMES
        }
        for metric in metrics
    }
