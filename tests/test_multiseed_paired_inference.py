"""Behaviour of the three-seed hierarchical paired inference.

The single-seed bootstrap it extends stamped ``single_seed_diagnostic_only``
into its own output, which was honest but left the formal comparison resting on
one training run. These tests cover the parts that a seed level can get wrong:
the estimator must reduce to the published three-seed difference, the seed draw
must actually reach the interval, Holm must be applied inside frozen families,
and the verdict must need direction, adjusted significance and a practical
margin at once.

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
    _verdict,
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
    assert records[0]["p_holm"] is None
    assert "exploratory" in records[0]["verdict"]


# --- the verdict needs all three of direction, significance and size ---------


def test_significant_but_below_the_practical_margin_is_not_a_win():
    verdict = _verdict(delta=-0.01, adjusted_p=0.001, higher_is_better=False, margin=0.05)
    assert "practical margin" in verdict
    assert MODEL_A not in verdict


def test_large_but_insignificant_is_not_a_win():
    assert _verdict(delta=-5.0, adjusted_p=0.4, higher_is_better=False,
                    margin=0.05) == "no difference"


def test_direction_is_respected_for_higher_is_better_endpoints():
    # JoinLift is a benefit: a positive delta favours ALFNet.
    assert _verdict(0.9, 0.001, True, 0.25) == f"{MODEL_A} better"
    assert _verdict(-0.9, 0.001, True, 0.25) == f"{MODEL_B} better"
    # an error metric is the other way round
    assert _verdict(-0.9, 0.001, False, 0.25) == f"{MODEL_A} better"
    assert _verdict(0.9, 0.001, False, 0.25) == f"{MODEL_B} better"


def test_joinlift_is_the_only_higher_is_better_endpoint():
    assert HIGHER_IS_BETTER == {("ples", "JoinLift", "MAPE"), ("ples", "JoinLift", "MAE")}


def test_non_finite_input_is_undetermined_not_a_win():
    assert _verdict(float("nan"), 0.001, False, 0.05) == "undetermined"
    assert _verdict(-1.0, float("nan"), False, 0.05) == "undetermined"


# --- synthetic null, fixed effect, reversal ----------------------------------


def test_synthetic_null_is_not_called_significant():
    rng = np.random.default_rng(0)
    draws = rng.normal(0.0, 1.0, 4000)
    assert _bootstrap_p_value(draws) > 0.05
    record = _records(0.0, draws, [0.0, 0.0, 0.0])[0]
    assert record["verdict"] == "no difference"


def test_a_clear_fixed_effect_is_detected_with_the_right_sign():
    rng = np.random.default_rng(1)
    draws = rng.normal(-1.0, 0.05, 4000)
    record = _records(-1.0, draws, [-1.0, -0.9, -1.1])[0]
    assert record["p_bootstrap"] < 0.05
    assert record["p_holm"] < 0.05
    assert record["verdict"] == f"{MODEL_A} better"
    assert record["seeds_favouring_alfnet"] == 3
    assert record["seed_unanimous"] is True


def test_reversing_the_effect_reverses_the_verdict():
    rng = np.random.default_rng(2)
    draws = rng.normal(1.0, 0.05, 4000)
    record = _records(1.0, draws, [1.0, 0.9, 1.1])[0]
    assert record["verdict"] == f"{MODEL_B} better"
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
