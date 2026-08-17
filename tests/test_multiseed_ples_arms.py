"""Three-seed PLES endpoints must report both arms, not the contrast alone.

A PLES contrast is gameable by construction:

    JoinLift           = control(Unseen) - treated(Join)      larger is better
    FaultPenalty       = treated(Fault)  - control(Steady)    smaller is better
    OutageFaultPenalty = the same, read at tau = input_len - 1

so a model that merely predicts the *control* arm worse improves JoinLift and
shrinks both penalties without forecasting anything better. The arms are what
separate that from a real improvement, and three orthogonal statuses keep the
distinction visible instead of collapsing it into one verdict:

  contrast_status              did the contrast move, against contrast_margin
  arms_pareto_status           how the two absolute-error arms moved
  combined_descriptive_status  which of the three conditions held

``combined_descriptive_status`` deliberately says ``all_conditions_met`` rather
than "accepted": v3 has no pre-registered gate, so it describes and adjudicates
nothing. Three marginal intervals also do not have joint 95% coverage, so no
joint claim is made anywhere.
"""

import numpy as np
import pytest

from scripts.multiseed_paired_inference import (
    ARM_METRICS,
    EFFECT_DEFINITION,
    PLES_EFFECTS,
    NonFiniteValue,
    _arm_status,
    _interval_status,
    arms_pareto_status,
    build_ples_records,
    combined_status,
)

SEEDS = [32, 42, 52]
MARGINS = {"MAE": 0.05, "MAPE": 0.25, "RMSE": None}
MODEL_A, MODEL_B = "ALFNet", "MoGERNN"


def _inputs(contrast, treated, control, effect="JoinLift", metric="MAE", spread=0.01):
    """All six endpoints, with the one under test carrying the given values.

    build_ples_records requires every endpoint to be present -- a missing one is
    a hard failure, not a reduced set -- so the fixture supplies them all and
    leaves the others neutral.
    """
    rng = np.random.default_rng(0)
    point, draws, seed_deltas, absolute = {}, {}, {s: {} for s in SEEDS}, {}
    for candidate_effect in PLES_EFFECTS:
        for candidate_metric in ARM_METRICS:
            selected = (candidate_effect == effect and candidate_metric == metric)
            values = {
                ("ples", candidate_effect, candidate_metric): contrast if selected else 0.0,
                ("arm", f"{candidate_effect}/treated", candidate_metric): treated if selected else 0.0,
                ("arm", f"{candidate_effect}/control", candidate_metric): control if selected else 0.0,
            }
            for key, value in values.items():
                point[key] = value
                draws[key] = rng.normal(value, spread if selected else 1e-6, 400)
                for seed in SEEDS:
                    seed_deltas[seed][key] = value
    for seed in SEEDS:
        absolute[seed] = {
            model: {("arm", f"{e}/{a}", m): 1.0
                    for e in PLES_EFFECTS for a in ("treated", "control")
                    for m in ARM_METRICS}
            for model in (MODEL_A, MODEL_B)
        }
    return point, draws, seed_deltas, absolute


def _record(contrast, treated, control, effect="JoinLift", metric="MAE", spread=0.01):
    point, draws, seed_deltas, absolute = _inputs(
        contrast, treated, control, effect, metric, spread)
    records = build_ples_records(point, draws, seed_deltas, absolute, SEEDS,
                                 MARGINS, MARGINS)
    return next(r for r in records if r["effect"] == effect and r["metric"] == metric)


# --- definitions and directions are stated, not guessed from a name ----------


def test_effect_definitions_are_explicit():
    assert EFFECT_DEFINITION["JoinLift"]["orientation"] == "higher_is_better"
    assert EFFECT_DEFINITION["FaultPenalty"]["orientation"] == "lower_is_better"
    assert EFFECT_DEFINITION["OutageFaultPenalty"]["orientation"] == "lower_is_better"
    assert "control(Unseen) - treated(Join)" in EFFECT_DEFINITION["JoinLift"]["formula"]
    assert "input_len - 1" in EFFECT_DEFINITION["OutageFaultPenalty"]["formula"]


def test_oriented_advantage_is_positive_for_alfnet_on_every_endpoint():
    """One sign convention across endpoints whose raw signs differ."""
    lift = _record(contrast=+1.0, treated=-1.0, control=-1.0, effect="JoinLift")
    assert lift["contrast_oriented_advantage"] > 0

    penalty = _record(contrast=-1.0, treated=-1.0, control=-1.0, effect="FaultPenalty")
    assert penalty["contrast_delta"] < 0          # penalty fell
    assert penalty["contrast_oriented_advantage"] > 0   # which favours ALFNet


def test_every_endpoint_and_metric_is_present():
    point, draws, seed_deltas, absolute = {}, {}, {s: {} for s in SEEDS}, {}
    rng = np.random.default_rng(1)
    for effect in PLES_EFFECTS:
        for metric in ARM_METRICS:
            for key in (("ples", effect, metric),
                        ("arm", f"{effect}/treated", metric),
                        ("arm", f"{effect}/control", metric)):
                point[key] = -0.5
                draws[key] = rng.normal(-0.5, 0.01, 200)
                for seed in SEEDS:
                    seed_deltas[seed][key] = -0.5
    for seed in SEEDS:
        absolute[seed] = {model: {("arm", f"{e}/{a}", m): 1.0
                                  for e in PLES_EFFECTS for a in ("treated", "control")
                                  for m in ARM_METRICS}
                          for model in (MODEL_A, MODEL_B)}
    records = build_ples_records(point, draws, seed_deltas, absolute, SEEDS,
                                 MARGINS, MARGINS)
    assert len(records) == len(PLES_EFFECTS) * len(ARM_METRICS)
    assert {(r["effect"], r["metric"]) for r in records} == {
        (e, m) for e in PLES_EFFECTS for m in ARM_METRICS}


# --- the four counterexamples ------------------------------------------------


def test_control_only_degradation_does_not_meet_the_conditions():
    """The exploit: JoinLift = control - treated, so a *worse* control arm
    (a larger control error) inflates the lift while the forecast on joined
    nodes is unchanged."""
    exploit = _record(contrast=+1.0, treated=0.0, control=+1.0)
    assert exploit["contrast_status"] == "favourable"
    assert exploit["control_status"] == "degrade"
    assert exploit["arms_pareto_status"] == "control_degraded"
    assert exploit["combined_descriptive_status"] == \
        "contrast_favourable_but_control_degraded"
    assert "all_conditions_met" not in exploit["combined_descriptive_status"]


def test_both_arms_improve_but_contrast_degrades_is_reported_as_such():
    """Control improves more than treated, so the contrast genuinely worsens."""
    record = _record(contrast=-1.0, treated=-0.5, control=-2.0)
    assert record["contrast_status"] == "degraded"
    assert record["treated_status"] == "improve"
    assert record["control_status"] == "improve"
    assert record["arms_pareto_status"] == "treated_improves_control_non_degraded"
    assert record["combined_descriptive_status"] == "contrast_degraded"


def test_treated_improves_control_non_degraded_and_contrast_improves():
    record = _record(contrast=+1.0, treated=-1.0, control=0.0)
    assert record["contrast_status"] == "favourable"
    assert record["treated_status"] == "improve"
    assert record["control_status"] in {"improve", "equivalent"}
    assert record["combined_descriptive_status"] == "all_conditions_met"


def test_an_interval_crossing_zero_is_uncertain():
    record = _record(contrast=+0.05, treated=-0.02, control=0.0, spread=2.0)
    assert record["contrast_status"] == "uncertain"
    assert record["combined_descriptive_status"].startswith("contrast_uncertain")


def test_an_exact_tie_is_equivalent_not_a_win():
    record = _record(contrast=0.0, treated=0.0, control=0.0, spread=1e-6)
    assert record["contrast_status"] == "equivalent"
    assert record["treated_status"] == "equivalent"
    assert record["arms_pareto_status"] == "no_material_arm_change"


# --- non-finite input fails closed -------------------------------------------


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_a_non_finite_draw_fails_closed(bad):
    point, draws, seed_deltas, absolute = _inputs(+1.0, -1.0, 0.0)
    key = ("arm", "JoinLift/treated", "MAE")
    draws[key] = np.append(draws[key][:-1], bad)
    with pytest.raises(NonFiniteValue, match="draws"):
        build_ples_records(point, draws, seed_deltas, absolute, SEEDS, MARGINS, MARGINS)


def test_a_non_finite_per_seed_contrast_fails_closed():
    point, draws, seed_deltas, absolute = _inputs(+1.0, -1.0, 0.0)
    seed_deltas[42][("ples", "JoinLift", "MAE")] = float("nan")
    with pytest.raises(NonFiniteValue, match="non-finite per-seed"):
        build_ples_records(point, draws, seed_deltas, absolute, SEEDS, MARGINS, MARGINS)


def test_a_missing_arm_endpoint_fails_closed():
    point, draws, seed_deltas, absolute = _inputs(+1.0, -1.0, 0.0)
    del draws[("arm", "JoinLift/control", "MAE")]
    with pytest.raises(NonFiniteValue, match="missing endpoint"):
        build_ples_records(point, draws, seed_deltas, absolute, SEEDS, MARGINS, MARGINS)


# --- status primitives -------------------------------------------------------


def test_interval_status_respects_orientation():
    # higher-is-better: an interval above +margin is favourable
    assert _interval_status(1.0, 0.5, 1.5, 0.25, True) == "favourable"
    assert _interval_status(-1.0, -1.5, -0.5, 0.25, True) == "degraded"
    # lower-is-better: the other way round
    assert _interval_status(-1.0, -1.5, -0.5, 0.25, False) == "favourable"
    assert _interval_status(1.0, 0.5, 1.5, 0.25, False) == "degraded"


def test_interval_status_without_a_margin_never_says_equivalent():
    """With no scale there is no basis for calling something negligible."""
    assert _interval_status(0.001, 0.0005, 0.0015, None, False) == "degraded"
    assert _interval_status(-0.001, -0.0015, -0.0005, None, False) == "favourable"


def test_arm_status_is_always_lower_is_better():
    assert _arm_status(-1.0, -1.5, -0.5, 0.05) == "improve"
    assert _arm_status(1.0, 0.5, 1.5, 0.05) == "degrade"
    assert _arm_status(0.0, -0.01, 0.01, 0.05) == "equivalent"
    assert _arm_status(0.0, -5.0, 5.0, 0.05) == "uncertain"


def test_no_status_uses_confirmatory_wording():
    """v3 has no pre-registered gate, so nothing here accepts or rejects."""
    banned = ("accept", "reject", "pass", "fail", "significant")
    values = set()
    for treated in ("improve", "degrade", "equivalent", "uncertain"):
        for control in ("improve", "degrade", "equivalent", "uncertain"):
            values.add(arms_pareto_status(treated, control))
            for contrast in ("favourable", "degraded", "equivalent", "uncertain"):
                values.add(combined_status(contrast, treated, control))
    for value in values:
        assert not any(word in value.lower() for word in banned), value
