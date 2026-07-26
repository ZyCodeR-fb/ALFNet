"""Minimal tests for the stage-0 offline paired bootstrap script."""

import collections

import numpy as np
import pytest

from data_provider.ples_protocol import (
    phase_for_tau_v2,
    summarize_lce_rows,
    summarize_paired_rows,
)
from scripts.paired_model_bootstrap import (
    RowTensor,
    _holm,
    _pair_keys,
    assert_paired_bundles,
    assert_pre_event_identity,
    classify,
    compute_statistics,
    paired_bootstrap,
)


INPUT_LEN = 12
HASHES = {
    "protocol_hash": "p",
    "training_protocol_hash": "t",
    "lce_protocol_hash": "l",
    "ples_protocol_hash": "e",
}


def _rows(scale=1.0, origins=(1, 2), cohorts=(1, 2), strata=None, origin_offset=0.0):
    """A complete, protocol-shaped row set small enough to reason about.

    The default strata deliberately mix a Weekday and a Weekend cell: they carry
    different calendar weights. ``origin_offset`` makes the two strata differ in
    error level, which is what lets an equal-weight aggregation be told apart
    from the calendar-share one.
    """
    strata = strata or ("Weekday/AMPeak", "Weekend/PMPeak")
    rows = []
    for origin_id in origins:
        regime = strata[origin_id % len(strata)]
        calendar_day = f"2026-01-{origin_id:02d}"
        for cohort_id in cohorts:
            for track in ("join", "fault"):
                for tau in range(-1, INPUT_LEN + 1):
                    treated = scale * (
                        1.0 + 0.1 * tau + 0.5 * cohort_id + origin_offset * origin_id
                    )
                    control = 1.0 + 0.05 * tau
                    if tau < 0:  # pre-event arms are identical by construction
                        treated = control
                    count = 6
                    rows.append(
                        {
                            "dataset": "TOY",
                            "origin_id": origin_id,
                            "calendar_day": calendar_day,
                            "regime": regime,
                            "event_time_index": 100 * origin_id,
                            "event_timestamp": f"2026-01-0{origin_id} 08:30:00",
                            "cohort_id": cohort_id,
                            "cohort_fraction": 0.05,
                            "cohort_style": "scattered",
                            "track": track,
                            "tau": tau,
                            "phase": phase_for_tau_v2(track, tau, INPUT_LEN),
                            "node_count": 3,
                            "count": count,
                            "graph_weight": None,
                            "patch_residual_weight": None,
                            "treated_ape_sum": treated * count / 100.0,
                            "control_ape_sum": control * count / 100.0,
                            "treated_ae_sum": treated * count,
                            "control_ae_sum": control * count,
                            "treated_se_sum": treated**2 * count,
                            "control_se_sum": control**2 * count,
                        }
                    )
    return rows


def _run(model):
    return {"model": model, "model_variant": "v", "seed": 32, "dataset": {"name": "TOY"}, "hashes": dict(HASHES)}


def _tensor(rows):
    return RowTensor(rows, _pair_keys(rows), INPUT_LEN)


def test_vectorised_kernel_reproduces_production_aggregation():
    rows = _rows(scale=1.3)
    tensor = _tensor(rows)
    fast = compute_statistics(tensor, np.ones(tensor.n_rows))
    lce = summarize_lce_rows(rows, input_len=INPUT_LEN)
    ples = summarize_paired_rows(rows, input_len=INPUT_LEN)["paired_effects"]["MAPE"]

    for name in ("Unseen", "Steady", "Join", "Fault", "LifeAvg"):
        for metric in ("MAPE", "MAE", "RMSE"):
            assert fast[("main", name, metric)] == pytest.approx(
                lce["main_metrics"][name][metric], rel=1e-12
            )
    for phase in ("Cold", "Warm", "Ready", "Onset", "Decay", "Outage"):
        for metric in ("MAPE", "MAE", "RMSE"):
            assert fast[("phase", phase, metric)] == pytest.approx(
                lce["phase_metrics"][phase][metric], rel=1e-12
            )
    for label in ("Join", "Fault"):
        for cell in lce["trajectory_metrics"][label]:
            assert fast[("trajectory", f"{label}_k{cell['k']}", "MAE")] == pytest.approx(
                cell["MAE"], rel=1e-12
            )
    for name in ("JoinLift", "FaultPenalty", "OutageFaultPenalty"):
        assert fast[("ples", name, "MAPE")] == pytest.approx(ples[name], rel=1e-12)


def test_kernel_uses_calendar_share_weights_not_equal_weights():
    """Weekday and Weekend carry 5/7 and 2/7, so equal weighting must not pass."""
    rows = _rows(scale=1.3, origin_offset=0.7)
    tensor = _tensor(rows)
    stats = compute_statistics(tensor, np.ones(tensor.n_rows))
    production = summarize_lce_rows(rows, input_len=INPUT_LEN)["main_metrics"]

    weekday = [row for row in rows if row["regime"].startswith("Weekday")]
    weekend = [row for row in rows if row["regime"].startswith("Weekend")]
    assert weekday and weekend, "fixture must span both day types"

    equal_weighted = np.mean(
        [
            summarize_lce_rows(part, input_len=INPUT_LEN)["main_metrics"]["Join"]["MAE"]
            for part in (weekday, weekend)
        ]
    )
    assert stats[("main", "Join", "MAE")] == pytest.approx(
        production["Join"]["MAE"], rel=1e-12
    )
    assert stats[("main", "Join", "MAE")] != pytest.approx(equal_weighted, rel=1e-9)


def test_bootstrap_blocks_by_calendar_day_and_crosses_cohorts():
    """Overlapping PLES tails make origins dependent; whole days are the unit.

    Cohorts are crossed with origins, so one global cohort draw must apply to
    every sampled block. A nested draw (redrawing cohorts inside each origin)
    would let the same bootstrap sample use different cohort sets per day.
    """
    rows = _rows(origins=(1, 2, 3, 4))
    tensor = _tensor(rows)
    rng = np.random.default_rng(3)

    assert set(tensor.blocks_by_stratum) == {"Weekday", "Weekend"}
    day_sizes = {name: len(blocks) for name, blocks in tensor.blocks_by_stratum.items()}

    for _ in range(30):
        weights = _draw(tensor, rng)
        # Day-type composition is preserved by the stratified block draw.
        drawn_days = collections.Counter()
        for stratum, blocks in tensor.blocks_by_stratum.items():
            for block in blocks:
                indices = tensor.block_rows[block]
                per_row = weights[indices] / np.maximum(
                    _cohort_multiplicity(tensor, weights)[tensor.row_cohort[indices]], 1e-12
                )
                drawn_days[stratum] += float(np.max(per_row))
        for stratum, size in day_sizes.items():
            assert drawn_days[stratum] == pytest.approx(size, rel=0.5)

        # One cohort draw for the whole sample: the multiplicity a cohort gets
        # must be identical in every block it appears in.
        for cohort_id in tensor.cohort_ids:
            ratios = set()
            for block, indices in tensor.block_rows.items():
                mask = tensor.row_cohort[indices] == cohort_id
                block_total = weights[indices][mask].sum()
                if block_total:
                    ratios.add(round(float(block_total) / int(mask.sum()), 9))
            assert len(ratios) <= len(tensor.block_rows), "cohort draw must be global"


def _cohort_multiplicity(tensor, weights):
    """Recover the per-cohort draw multiplicity implied by a weight vector."""
    out = np.zeros(int(max(tensor.cohort_ids)) + 1, dtype=np.float64)
    for cohort_id in tensor.cohort_ids:
        mask = tensor.row_cohort == cohort_id
        total = weights[mask].sum()
        out[cohort_id] = total / max(int(mask.sum()), 1)
    return out


def test_derived_rows_are_exact_linear_functions_of_the_free_rows():
    """Join/Fault/LifeAvg carry no information beyond the eight free rows."""
    tensor = _tensor(_rows())
    stats = compute_statistics(tensor, np.ones(tensor.n_rows))
    for metric in ("MAPE", "MAE", "RMSE"):
        for group, phases in (
            ("Join", ("Cold", "Warm", "Ready")),
            ("Fault", ("Onset", "Decay", "Outage")),
        ):
            recomposed = (
                stats[("phase", phases[0], metric)]
                + 10 * stats[("phase", phases[1], metric)]
                + stats[("phase", phases[2], metric)]
            ) / 12.0
            assert stats[("main", group, metric)] == pytest.approx(recomposed, rel=1e-12)
        assert stats[("main", "LifeAvg", metric)] == pytest.approx(
            np.mean([stats[("main", name, metric)] for name in ("Unseen", "Steady", "Join", "Fault")]),
            rel=1e-12,
        )


def test_self_comparison_gives_exactly_zero_difference_in_every_draw():
    rows = _rows()
    tensor_a, tensor_b = _tensor(rows), _tensor(list(rows))
    draws = paired_bootstrap(tensor_a, tensor_b, samples=25, seed=7)
    assert draws
    for key, values in draws.items():
        assert np.allclose(values, 0.0), f"shared resample must cancel for {key}"


def test_paired_draws_are_tighter_than_treating_models_as_independent():
    """The point of pairing: shared resampling removes the common origin/cohort noise."""
    rows_a, rows_b = _rows(scale=1.0), _rows(scale=1.4)
    tensor_a, tensor_b = _tensor(rows_a), _tensor(rows_b)
    paired = paired_bootstrap(tensor_a, tensor_b, samples=200, seed=11)[("main", "Join", "MAE")]

    rng_a, rng_b = np.random.default_rng(11), np.random.default_rng(4242)
    independent = np.array(
        [
            compute_statistics(tensor_a, _draw(tensor_a, rng_a))[("main", "Join", "MAE")]
            - compute_statistics(tensor_b, _draw(tensor_b, rng_b))[("main", "Join", "MAE")]
            for _ in range(200)
        ]
    )
    assert np.std(paired) < np.std(independent)


def _draw(tensor, rng):
    from scripts.paired_model_bootstrap import draw_weights

    return draw_weights(tensor, rng)


def test_mismatched_bundles_raise_instead_of_silently_intersecting():
    rows_a = _rows()
    rows_b = [row for row in _rows() if not (row["origin_id"] == 2 and row["tau"] == 5)]
    with pytest.raises(ValueError, match="refusing to intersect"):
        assert_paired_bundles(_run("A"), _run("B"), rows_a, rows_b)


def test_hash_and_seed_mismatches_are_rejected():
    rows = _rows()
    other = _run("B")
    other["hashes"]["training_protocol_hash"] = "different"
    with pytest.raises(ValueError, match="training_protocol_hash differs"):
        assert_paired_bundles(_run("A"), other, rows, list(rows))

    other = _run("B")
    other["seed"] = 42
    with pytest.raises(ValueError, match="seed differs"):
        assert_paired_bundles(_run("A"), other, rows, list(rows))


def test_protocol_determined_fields_must_agree_row_by_row():
    rows_a = _rows()
    rows_b = _rows()
    rows_b[3]["count"] = rows_b[3]["count"] + 1
    with pytest.raises(ValueError, match="protocol-determined field count differs"):
        assert_paired_bundles(_run("A"), _run("B"), rows_a, rows_b)


def test_pre_event_identity_is_asserted_not_scored():
    rows = _rows()
    assert assert_pre_event_identity(rows, "A") == sum(1 for row in rows if row["tau"] < 0)
    broken = _rows()
    target = next(row for row in broken if row["tau"] < 0)
    target["treated_ae_sum"] += 1e-6
    with pytest.raises(ValueError, match="pre-event treated/control diverge"):
        assert_pre_event_identity(broken, "A")


def test_verdicts_split_significance_from_direction():
    assert classify(-1.0, -2.0, -0.5, lower_is_better=True) == "显著胜出"
    assert classify(-1.0, -2.0, 0.5, lower_is_better=True) == "仅方向胜出"
    assert classify(1.0, 0.5, 2.0, lower_is_better=True) == "显著落后"
    assert classify(1.0, -0.5, 2.0, lower_is_better=True) == "仅方向落后"
    # JoinLift is the one quantity where larger is better.
    assert classify(1.0, 0.5, 2.0, lower_is_better=False) == "显著胜出"


def test_holm_is_monotone_and_never_below_the_raw_p_value():
    raw = [0.001, 0.02, 0.04, 0.9]
    adjusted = _holm(raw)
    assert all(a >= r for a, r in zip(adjusted, raw))
    assert adjusted == sorted(adjusted)
    assert adjusted[0] == pytest.approx(0.004)
    assert all(value <= 1.0 for value in adjusted)
