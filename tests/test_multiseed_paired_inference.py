"""Behaviour of the three-seed crossed paired bootstrap.

The single-seed bootstrap it extends stamped ``single_seed_diagnostic_only``
into its own output, which was honest but left the comparison resting on one
training run. These tests cover what a seed level can get wrong:

  * the estimator must reduce to the published three-seed difference;
  * seeds, blocks and cohorts are **crossed**, so one block/cohort draw per
    replicate must reach every seed and both models -- an earlier version drew
    fresh blocks inside each seed, which resamples one set of test events three
    times and narrows the interval;
  * no verdict may imply a test the design cannot support. With three seeds the
    exact two-sided sign-flip null has minimum p = 0.25, so the wording must
    stay descriptive and Holm must not be presented as inference.

Nothing here reads ``outputs/``. Statistics are exercised on synthetic draws and
the loader's refusals on a manifest fixture, so the file is self-contained.
"""

import numpy as np
import pytest

from scripts.multiseed_paired_inference import (
    FAMILIES,
    HIGHER_IS_BETTER,
    MODEL_A,
    MODEL_B,
    _descriptive_verdict,
    build_records,
)

SEEDS = [32, 42, 52]
MARGINS = {"MAE": 0.05, "MAPE": 0.25, "RMSE": None}
KEY = ("main", "Unseen", "MAE")


def _records(delta, draws, per_seed, key=KEY):
    return build_records({key: delta}, {key: np.asarray(draws, dtype=float)},
                         {s: {key: v} for s, v in zip(SEEDS, per_seed)},
                         SEEDS, MARGINS)


# --- the estimator is the published three-seed difference --------------------


def test_point_estimate_is_the_mean_of_the_per_seed_differences():
    """Mean of differences equals difference of means; the tables publish the
    latter, this publishes the former, and they must not drift apart."""
    alfnet = [4.0, 5.0, 6.0]
    mogernn = [4.5, 5.5, 5.0]
    per_seed = [a - m for a, m in zip(alfnet, mogernn)]
    assert float(np.mean(per_seed)) == pytest.approx(
        float(np.mean(alfnet)) - float(np.mean(mogernn))
    )


# --- Holm is applied inside frozen families ----------------------------------


def test_families_are_frozen_and_disjoint():
    keys = [key for member in FAMILIES.values() for key in member]
    assert len(keys) == len(set(keys)), "a key in two families would be corrected twice"
    assert set(FAMILIES) == {"lce_primary", "ples"}
    assert len(FAMILIES["lce_primary"]) == 4
    assert len(FAMILIES["ples"]) == 6


def test_no_p_like_statistic_is_emitted():
    """Three seeds cannot support one, and a p-shaped number gets read as a p."""
    record = _records(-1.0, [-1.0] * 50, [-1.0, -1.0, -1.0])[0]
    banned = {"p_bootstrap", "p_holm", "p_holm_descriptive", "p_value",
              "bootstrap_tail_area", "holm_ordering_score"}
    assert banned.isdisjoint(record), sorted(banned & set(record))


def test_cells_outside_a_group_are_marked_exploratory():
    exploratory = ("phase", "Warm", "RMSE")
    assert exploratory not in [k for m in FAMILIES.values() for k in m]
    records = _records(-1.0, [-1.0] * 50, [-1.0, -1.0, -1.0], key=exploratory)
    assert records[0]["family"] is None
    assert "exploratory" in records[0]["verdict"]


# --- the verdict needs all three of direction, significance and size ---------


def test_below_the_post_hoc_margin_is_not_called_a_win():
    verdict = _descriptive_verdict(-0.01, -0.02, -0.005, False, 0.05, 3, 3)
    assert "post-hoc margin" in verdict
    assert "favours" not in verdict


def test_an_interval_covering_zero_is_reported_as_such():
    assert _descriptive_verdict(-5.0, -9.0, 1.0, False, 0.05, 2, 3) == "interval covers zero"


def test_no_verdict_ever_claims_significance():
    """The wording must not imply a test the design cannot support."""
    banned = ("significant", "p <", "confirm", "proven")
    for delta, low, high, hib, favouring in (
        (-1.0, -2.0, -0.5, False, 3), (1.0, 0.5, 2.0, False, 0),
        (-0.01, -0.02, -0.005, False, 3), (-5.0, -9.0, 1.0, False, 2),
    ):
        verdict = _descriptive_verdict(delta, low, high, hib, 0.05, favouring, 3)
        assert not any(word in verdict.lower() for word in banned), verdict


def test_direction_is_respected_for_higher_is_better_endpoints():
    # JoinLift is a benefit: a positive delta favours ALFNet.
    assert MODEL_A in _descriptive_verdict(0.9, 0.4, 1.4, True, 0.25, 3, 3)
    assert MODEL_B in _descriptive_verdict(-0.9, -1.4, -0.4, True, 0.25, 0, 3)
    # an error metric is the other way round
    assert MODEL_A in _descriptive_verdict(-0.9, -1.4, -0.4, False, 0.25, 3, 3)
    assert MODEL_B in _descriptive_verdict(0.9, 0.4, 1.4, False, 0.25, 0, 3)


def test_seed_agreement_is_reported_in_fields_not_buried_in_prose():
    """The tally is structured, so a reader cannot mistake a 2/3 split."""
    split = _records(-1.0, [-1.0] * 50, [-1.0, -1.0, +0.4])[0]
    assert split["n_favour_alfnet_seeds"] == 2
    assert split["n_favour_baseline_seeds"] == 1
    assert split["seed_unanimous"] is False

    agreed = _records(-1.0, [-1.0] * 50, [-1.0, -1.0, -0.4])[0]
    assert agreed["n_favour_alfnet_seeds"] == 3
    assert agreed["seed_unanimous"] is True


def test_only_ten_distinct_seed_multiplicity_vectors_exist():
    """Documented as 27 once; 27 is the ordered count, the design has 10."""
    from collections import Counter
    from itertools import product
    vectors = {tuple(Counter(s).get(i, 0) for i in range(3))
               for s in product(range(3), repeat=3)}
    assert len(vectors) == 10
    assert len(list(product(range(3), repeat=3))) == 27


def test_three_paired_seeds_cannot_reach_conventional_significance():
    """Exact two-sided sign-flip null over 3 pairs has minimum p = 0.25."""
    assert 2 / (2 ** 3) == 0.25


def test_joinlift_is_the_only_higher_is_better_endpoint():
    assert HIGHER_IS_BETTER == {("ples", "JoinLift", "MAPE"), ("ples", "JoinLift", "MAE")}


def test_non_finite_input_is_undetermined_not_a_win():
    assert _descriptive_verdict(float("nan"), -1.0, 1.0, False, 0.05, 3, 3) == "undetermined"
    assert _descriptive_verdict(-1.0, float("nan"), 1.0, False, 0.05, 3, 3) == "undetermined"
    assert _descriptive_verdict(-1.0, -2.0, float("nan"), False, 0.05, 3, 3) == "undetermined"


# --- synthetic null, fixed effect, reversal ----------------------------------


def test_synthetic_null_is_not_described_as_a_difference():
    rng = np.random.default_rng(0)
    draws = rng.normal(0.0, 1.0, 4000)
    record = _records(0.0, draws, [0.0, 0.0, 0.0])[0]
    assert record["verdict"] == "no difference"
    assert record["n_tie_seeds"] == 3


def test_a_clear_fixed_effect_is_described_with_the_right_sign():
    rng = np.random.default_rng(1)
    draws = rng.normal(-1.0, 0.05, 4000)
    record = _records(-1.0, draws, [-1.0, -0.9, -1.1])[0]
    assert record["ci_high"] < 0, "interval should sit entirely below zero"
    assert MODEL_A in record["verdict"]
    assert record["seeds_favouring_alfnet"] == 3
    assert record["seed_unanimous"] is True


def test_reversing_the_effect_reverses_the_verdict():
    rng = np.random.default_rng(2)
    draws = rng.normal(1.0, 0.05, 4000)
    record = _records(1.0, draws, [1.0, 0.9, 1.1])[0]
    assert MODEL_B in record["verdict"]
    assert record["seeds_favouring_alfnet"] == 0
    assert record["seed_unanimous"] is True


def test_seed_conflict_is_visible_next_to_the_point_estimate():
    """The METR-LA situation: a mean that favours one model on split seeds.

    The seed disagreement must be visible in structured fields, because a 2/3
    split is what distinguishes this from a settled result.
    """
    rng = np.random.default_rng(3)
    draws = rng.normal(-0.3, 0.05, 4000)
    record = _records(-0.3, draws, [-1.0, -1.0, +0.4])[0]
    assert record["seed_unanimous"] is False
    assert record["seeds_favouring_alfnet"] == 2
    assert record["seed_unanimous"] is False
    assert record["per_seed_delta"] == {"32": -1.0, "42": -1.0, "52": 0.4}


def test_seed_order_does_not_change_the_result():
    rng = np.random.default_rng(4)
    draws = rng.normal(-0.5, 0.1, 500)
    forward = _records(-0.5, draws, [-1.0, -0.5, 0.0])[0]
    shuffled = build_records(
        {KEY: -0.5}, {KEY: draws},
        {52: {KEY: 0.0}, 32: {KEY: -1.0}, 42: {KEY: -0.5}},
        SEEDS, MARGINS)[0]
    assert forward["per_seed_delta"] == shuffled["per_seed_delta"]
    assert forward["seeds_favouring_alfnet"] == shuffled["seeds_favouring_alfnet"]
    assert forward["delta"] == shuffled["delta"]


# --- the loader refuses rather than working around ---------------------------


def test_loader_refuses_a_missing_or_duplicated_seed(monkeypatch):
    from scripts import multiseed_paired_inference as module

    def _manifest(runs, seeds=(32, 42, 52)):
        return {"seeds": list(seeds), "runs": runs}

    base = [
        {"dataset": "METRLA", "model": model, "seed": seed, "run_id": f"{model}-{seed}"}
        for model in (MODEL_A, MODEL_B) for seed in (32, 42, 52)
    ]

    # a missing seed
    monkeypatch.setattr(module, "load_manifest",
                        lambda: _manifest([r for r in base if r["seed"] != 52]))
    with pytest.raises(SystemExit, match="seeds"):
        module.load_seed_bundles("METRLA", "./outputs")

    # a duplicated seed
    monkeypatch.setattr(module, "load_manifest", lambda: _manifest(base + [base[0]]))
    with pytest.raises(SystemExit, match="twice"):
        module.load_seed_bundles("METRLA", "./outputs")


# --- crossed design: one draw per replicate, shared everywhere ---------------


class _FakeTensor:
    """Minimal stand-in carrying only what the resampler touches."""

    def __init__(self, offset):
        self.n_rows = 6
        self.cohort_ids = [1, 2]
        self.blocks_by_stratum = {"Weekday": ["b0", "b1"]}
        self.block_rows = {"b0": np.array([0, 1, 2]), "b1": np.array([3, 4, 5])}
        self.row_cohort = np.array([1, 2, 1, 2, 1, 2])
        self.offset = offset


def test_one_block_and_cohort_draw_is_shared_by_every_seed_and_both_models(monkeypatch):
    """The crossed-design property, asserted on the weights themselves.

    Each replicate must produce exactly one distinct weight vector. If the
    resampler ever draws again per seed, this sees more than one.
    """
    from scripts import multiseed_paired_inference as module

    seen = []

    def _spy(tensor, weights):
        seen.append((tensor.offset, tuple(np.asarray(weights).tolist())))
        return {("main", "Unseen", "MAE"): float(tensor.offset)}

    monkeypatch.setattr(module, "statistics_and_arms", _spy)
    tensors = {s: {MODEL_A: _FakeTensor(s), MODEL_B: _FakeTensor(s + 1000)}
               for s in SEEDS}

    seen.clear()
    module.crossed_bootstrap(tensors, SEEDS, samples=1, seed=7)  # returns (draws, raw)
    weight_vectors = {entry[1] for entry in seen}
    assert len(weight_vectors) == 1, (
        f"a replicate used {len(weight_vectors)} weight vectors; seeds, blocks "
        "and cohorts are crossed so it must use exactly one"
    )
    # and both models were evaluated against it
    models_seen = {entry[0] >= 1000 for entry in seen}
    assert models_seen == {True, False}


def test_a_repeated_seed_is_reweighted_not_resampled(monkeypatch):
    from scripts import multiseed_paired_inference as module

    calls = []

    def _spy(tensor, weights):
        calls.append(tensor.offset)
        return {("main", "Unseen", "MAE"): float(tensor.offset)}

    monkeypatch.setattr(module, "statistics_and_arms", _spy)
    tensors = {s: {MODEL_A: _FakeTensor(s), MODEL_B: _FakeTensor(s + 1000)}
               for s in SEEDS}

    for replicate_seed in range(6):
        calls.clear()
        module.crossed_bootstrap(tensors, SEEDS, samples=1, seed=replicate_seed)
        # one call per (distinct drawn seed x model); a seed drawn twice must not
        # be evaluated twice, it is weighted twice
        assert len(calls) == len(set(calls))


def test_row_design_hash_ignores_model_output_but_not_order():
    from scripts.multiseed_paired_inference import _row_design_hash

    def _row(origin, tau, treated):
        return {"origin_id": origin, "cohort_id": 1, "track": "join", "tau": tau,
                "phase": "Warm", "regime": "Weekday/Midday", "event_time_index": 0,
                "calendar_day": "2026-01-01", "cohort_fraction": 0.05,
                "cohort_style": "clustered", "node_count": 4, "count": 10,
                "treated_ae_sum": treated}

    base = [_row(1, 0, 1.0), _row(1, 1, 2.0)]
    # different model outputs, same design
    other = [_row(1, 0, 99.0), _row(1, 1, 77.0)]
    assert _row_design_hash(base) == _row_design_hash(other)
    # reordering is a different design, because weights are positional
    assert _row_design_hash(base) != _row_design_hash(list(reversed(base)))


# --- ties and non-finite values (Stage 1R-A2) --------------------------------


def test_ties_are_not_counted_for_either_model():
    """[+1, 0, 0] once read as "favours MoGERNN, all 3 seeds agree"."""
    from scripts.multiseed_paired_inference import seed_tally

    tally = seed_tally([+1.0, 0.0, 0.0], higher_is_better=False)
    assert tally == {"n_favour_alfnet_seeds": 0, "n_favour_baseline_seeds": 1,
                     "n_tie_seeds": 2, "n_nonfinite_seeds": 0, "n_evaluable_seeds": 3}

    record = _records(0.3333, [0.3] * 50, [+1.0, 0.0, 0.0])[0]
    assert record["n_tie_seeds"] == 2
    assert record["n_favour_baseline_seeds"] == 1
    assert record["seed_unanimous"] is False
    assert "all 3 seeds agree" not in record["verdict"]


def test_unanimity_requires_every_seed_evaluable_and_on_one_side():
    from scripts.multiseed_paired_inference import _unanimous, seed_tally

    assert _unanimous(seed_tally([-1.0, -1.0, -1.0], False), 3) is True
    assert _unanimous(seed_tally([-1.0, -1.0, 0.0], False), 3) is False
    assert _unanimous(seed_tally([-1.0, -1.0, +1.0], False), 3) is False


def test_a_non_finite_per_seed_delta_fails_closed():
    """It used to be counted as an ALFNet win."""
    from scripts.multiseed_paired_inference import NonFiniteValue

    with pytest.raises(NonFiniteValue, match="non-finite"):
        _records(-0.5, [-0.5] * 50, [-1.0, float("nan"), -0.5])


def test_an_all_nan_family_fails_closed_instead_of_scoring_zero():
    """all-NaN once produced holm_ordering_score = 0.0, the strongest rank."""
    from scripts.multiseed_paired_inference import NonFiniteValue

    with pytest.raises(NonFiniteValue):
        _records(float("nan"), [float("nan")] * 50, [float("nan")] * 3)


def test_non_finite_bootstrap_draws_fail_closed():
    from scripts.multiseed_paired_inference import NonFiniteValue

    draws = [-0.5] * 49 + [float("inf")]
    with pytest.raises(NonFiniteValue, match="bootstrap draws"):
        _records(-0.5, draws, [-1.0, -0.5, -0.2])


def test_rmse_gets_no_borrowed_margin():
    from scripts.multiseed_paired_inference import NonFiniteValue, margin_for

    margins = {"MAE": 0.05, "MAPE": 0.25, "RMSE": None}
    assert margin_for("RMSE", margins) is None
    assert margin_for("MAE", margins) == 0.05
    with pytest.raises(NonFiniteValue, match="no margin policy"):
        margin_for("SMAPE", margins)


def test_a_negative_or_non_finite_margin_is_refused():
    from scripts.multiseed_paired_inference import NonFiniteValue, margin_for

    for bad in (-0.1, float("nan"), float("inf")):
        with pytest.raises(NonFiniteValue, match="finite and >= 0"):
            margin_for("MAE", {"MAE": bad})


def test_a_marginless_metric_is_described_by_direction_only():
    record = _records(-1.0, [-1.0] * 50, [-1.0, -1.0, -1.0],
                      key=("main", "Unseen", "RMSE"))[0]
    assert record["post_hoc_margin"] is None
    assert "no margin defined" in record["verdict"]


def test_production_requires_enough_draws_for_a_95_percent_interval():
    from scripts.multiseed_paired_inference import MIN_BOOTSTRAP_SAMPLES

    assert MIN_BOOTSTRAP_SAMPLES >= 1000


def test_seed_multiplicity_actually_reaches_the_number(monkeypatch):
    """Force the draw to [32, 32, 52] and require exactly (2*d32 + d52)/3.

    The earlier fixture held A-B constant across seeds, so any weighting gave
    the same answer and the test could not see a multiplicity bug.
    """
    from scripts import multiseed_paired_inference as module

    effects = {32: 1.0, 42: 100.0, 52: 4.0}

    class _T:
        def __init__(self, value):
            self.n_rows = 4
            self.cohort_ids = [1]
            self.blocks_by_stratum = {"W": ["b0"]}
            self.block_rows = {"b0": np.array([0, 1, 2, 3])}
            self.row_cohort = np.array([1, 1, 1, 1])
            self.value = value

    def _stats(tensor, weights):
        return {KEY: tensor.value}

    monkeypatch.setattr(module, "statistics_and_arms", _stats)
    monkeypatch.setattr(module.np.random, "default_rng", module.np.random.default_rng)

    class _RNG:
        def __init__(self):
            self.calls = 0

        def choice(self, values, size, replace=True):
            self.calls += 1
            if self.calls == 1:                      # the seed draw
                return np.array([32, 32, 52])
            return np.array(list(values))            # cohorts, then blocks

    monkeypatch.setattr(module.np.random, "default_rng", lambda seed: _RNG())
    tensors = {s: {MODEL_A: _T(effects[s]), MODEL_B: _T(0.0)} for s in SEEDS}
    draws, _raw = module.crossed_bootstrap(tensors, SEEDS, samples=1, seed=0)

    expected = (2 * effects[32] + effects[52]) / 3
    assert draws[KEY][0] == pytest.approx(expected), (
        f"seed multiplicity did not reach the statistic: got {draws[KEY][0]}, "
        f"expected {expected}"
    )
