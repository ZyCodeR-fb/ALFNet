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
from scripts.paired_model_bootstrap import _bootstrap_p_value, _holm

SEEDS = [32, 42, 52]
MARGINS = {"MAE": 0.05, "MAPE": 0.25}
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


def test_holm_is_monotone_and_never_below_the_raw_p():
    raw = [0.001, 0.02, 0.04, 0.5, 0.9]
    adjusted = _holm(raw)
    for a, r in zip(adjusted, raw):
        assert a >= r - 1e-12
    ordered = [adjusted[i] for i in sorted(range(len(raw)), key=lambda i: raw[i])]
    assert ordered == sorted(ordered), "Holm output must be monotone in the raw order"
    assert all(0.0 <= value <= 1.0 for value in adjusted)


def test_families_are_frozen_and_disjoint():
    keys = [key for member in FAMILIES.values() for key in member]
    assert len(keys) == len(set(keys)), "a key in two families would be corrected twice"
    assert set(FAMILIES) == {"lce_primary", "ples"}
    assert len(FAMILIES["lce_primary"]) == 4
    assert len(FAMILIES["ples"]) == 6


def test_cells_outside_a_family_get_no_adjusted_p():
    """Correcting across a set picked after seeing results is not a correction."""
    exploratory = ("phase", "Warm", "RMSE")
    assert exploratory not in [k for m in FAMILIES.values() for k in m]
    records = _records(-1.0, [-1.0] * 50, [-1.0, -1.0, -1.0], key=exploratory)
    assert records[0]["p_holm_descriptive"] is None
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


def test_seed_split_is_stated_in_the_verdict():
    split = _descriptive_verdict(-1.0, -2.0, -0.5, False, 0.05, 2, 3)
    assert "2/3" in split
    unanimous = _descriptive_verdict(-1.0, -2.0, -0.5, False, 0.05, 3, 3)
    assert "all 3 seeds agree" in unanimous


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


def test_synthetic_null_is_not_called_significant():
    rng = np.random.default_rng(0)
    draws = rng.normal(0.0, 1.0, 4000)
    assert _bootstrap_p_value(draws) > 0.05
    record = _records(0.0, draws, [0.0, 0.0, 0.0])[0]
    assert record["verdict"] == "interval covers zero"


def test_a_clear_fixed_effect_is_detected_with_the_right_sign():
    rng = np.random.default_rng(1)
    draws = rng.normal(-1.0, 0.05, 4000)
    record = _records(-1.0, draws, [-1.0, -0.9, -1.1])[0]
    assert record["p_bootstrap"] < 0.05
    assert record["p_holm_descriptive"] < 0.05
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


def test_seed_conflict_is_reported_even_when_the_mean_is_significant():
    """The METR-LA situation: a mean that favours one model on split seeds.

    The verdict may still be significant -- that is the estimator's business --
    but the seed disagreement must be visible next to it, because a 2/3 split is
    what distinguishes this from a settled result.
    """
    rng = np.random.default_rng(3)
    draws = rng.normal(-0.3, 0.05, 4000)
    record = _records(-0.3, draws, [-1.0, -1.0, +0.4])[0]
    assert "2/3" in record["verdict"]
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

    monkeypatch.setattr(module, "compute_statistics", _spy)
    tensors = {s: {MODEL_A: _FakeTensor(s), MODEL_B: _FakeTensor(s + 1000)}
               for s in SEEDS}

    seen.clear()
    module.crossed_bootstrap(tensors, SEEDS, samples=1, seed=7)
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

    monkeypatch.setattr(module, "compute_statistics", _spy)
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
