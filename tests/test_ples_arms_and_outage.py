"""PLES effects must come from the frozen plan, and must show both arms.

Two independent failures motivate this file, both found by auditing
``scripts/plot_counterfactual_shockwave.py`` against the protocol it claims to
plot:

1. The script computed ``OutageFaultPenalty`` as the maximum of the fault curve.
   The frozen definition is the settled cell at ``tau = input_len - 1``. A
   maximum is not a paired quantity -- measured on the finished runs the two
   models' maxima landed at different tau in 12 of 18 cells, and the resulting
   contrast flipped sign in 2 of them.
2. It also kept the saturated cohorts the main plan excludes and weighted the
   calendar regimes equally instead of by their 5/7 and 2/7 shares.

The fix was to delete the parallel implementation and call
``data_provider.ples_protocol``. These tests hold that boundary: they check the
behaviour a second implementation would get wrong, not the text of the code.

Separately, a PLES contrast is gameable on its own -- ``JoinLift`` is
``control - treated``, so a model that merely predicts the control arm worse
scores better. The treated/control arms exist to catch that, and the Pareto
status names it.
"""

import numpy as np
import pytest

from data_provider.calendar_strata import stratum_weights
from data_provider.ples_protocol import (
    build_curve_rows,
    main_cohort_rows,
    paired_effect_arms,
    paired_effect_cells,
    paired_effects,
)
from scripts import plot_counterfactual_shockwave as plot

SUM_FIELDS = ("treated_ape_sum", "control_ape_sum", "treated_ae_sum",
              "control_ae_sum", "treated_se_sum", "control_se_sum")
INPUT_LEN = 4


def _phase(track, tau):
    join = ("Cold", "Warm", "Ready")
    fault = ("Onset", "Decay", "Outage")
    names = join if track == "join" else fault
    if tau == 0:
        return names[0]
    return names[2] if tau == INPUT_LEN - 1 else names[1]


def _row(track, tau, cohort, treated, control, regime="Weekday/Midday", origin=1):
    row = {
        "origin_id": origin, "cohort_id": cohort, "track": track, "tau": tau,
        "phase": _phase(track, tau), "count": 10, "regime": regime,
        "calendar_day": "2026-01-01", "event_time_index": origin * 1000,
    }
    for field in SUM_FIELDS:
        row[field] = (treated if field.startswith("treated") else control) * 10
    return row


def _rows(fault_treated_by_tau, control=1.0, cohorts=(1,), regime="Weekday/Midday",
          join_treated=1.0):
    """Both tracks scripted: fault treated per tau, join treated flat.

    ``join_treated`` is a parameter rather than a constant so a test can move
    the join treated arm; with it pinned, every JoinLift ``treated_delta`` is
    identically zero and an arm-movement assertion cannot fail.
    """
    rows = []
    for cohort in cohorts:
        for tau in range(INPUT_LEN):
            rows.append(_row("join", tau, cohort, join_treated, control, regime))
            rows.append(_row("fault", tau, cohort,
                             fault_treated_by_tau[tau], control, regime))
    return rows


# --- 1. Outage is the settled cell, never the maximum ------------------------


def test_outage_reads_tau_input_len_minus_one_not_the_curve_maximum():
    """The fault curve peaks at tau=1; the settled cell is tau=3."""
    treated = {0: 2.0, 1: 9.0, 2: 3.0, 3: 5.0}
    curves = build_curve_rows(_rows(treated))
    effects = paired_effects(curves, INPUT_LEN, "MAPE")

    # FaultPenalty is treated - control, control is 1.0, values are *100 for MAPE.
    assert effects["OutageFaultPenalty"] == pytest.approx((5.0 - 1.0) * 100)
    # and emphatically not the maximum, which sits at tau=1
    assert effects["OutageFaultPenalty"] != pytest.approx((9.0 - 1.0) * 100)


def test_outage_cell_selection_is_exactly_one_cell_at_the_settled_tau():
    curves = build_curve_rows(_rows({0: 2.0, 1: 9.0, 2: 3.0, 3: 5.0}))
    cells = paired_effect_cells(curves, INPUT_LEN)
    outage = cells["OutageFaultPenalty"]
    assert len(outage) == 1
    assert outage[0]["tau"] == INPUT_LEN - 1
    assert outage[0]["track"] == "fault"


def test_missing_settled_outage_cell_raises_rather_than_falling_back():
    rows = [row for row in _rows({0: 2.0, 1: 9.0, 2: 3.0, 3: 5.0})
            if not (row["track"] == "fault" and row["tau"] == INPUT_LEN - 1)]
    with pytest.raises(ValueError, match=f"tau={INPUT_LEN - 1}"):
        paired_effects(build_curve_rows(rows), INPUT_LEN, "MAPE")


# --- 2. effects and arms must be read off the same cells ---------------------


def test_arms_and_effects_share_one_cell_selection():
    """If these ever diverge the Pareto verdict describes different cells."""
    curves = build_curve_rows(_rows({0: 2.0, 1: 9.0, 2: 3.0, 3: 5.0}))
    cells = paired_effect_cells(curves, INPUT_LEN)
    arms = paired_effect_arms(curves, INPUT_LEN, "MAPE")
    effects = paired_effects(curves, INPUT_LEN, "MAPE")
    for name, group in cells.items():
        treated = float(np.mean([curve["treated"]["MAPE"] for curve in group]))
        control = float(np.mean([curve["control"]["MAPE"] for curve in group]))
        assert arms[name]["treated"] == pytest.approx(treated)
        assert arms[name]["control"] == pytest.approx(control)
        signed = control - treated if name == "JoinLift" else treated - control
        assert effects[name] == pytest.approx(signed)


# --- 3. a contrast that improves only by degrading the control arm -----------


def test_control_only_degradation_is_named_by_the_pareto_status():
    """The gameable case, built explicitly.

    Our JoinLift beats the baseline's, but only because our control arm is
    worse. The treated arm -- the actual forecast on joined nodes -- is
    identical. A contrast-only reader calls this an improvement.
    """
    flat = {tau: 1.0 for tau in range(INPUT_LEN)}
    baseline = _rows(flat, control=1.0)
    # identical treated arm, degraded control arm
    ours = _rows(flat, control=3.0)

    advantages = plot._advantages_v2(ours, baseline, INPUT_LEN)
    join = advantages["JoinLiftGain/MAPE"]

    # _advantages_v2 reports the arms; the verdict needs intervals and is made
    # in bootstrap_advantages_v2, exercised end to end further down.
    assert join["advantage"] > 0, "fixture must look like a win on the contrast"
    assert join["treated_delta"] == pytest.approx(0.0), "treated arm is unchanged"
    assert join["control_delta"] > 0, "control arm got worse"
    assert "pareto" not in join, "a verdict must not be made without an interval"


def test_both_arms_improving_is_distinguished_from_a_contrast_artifact():
    baseline = _rows({tau: 1.0 for tau in range(INPUT_LEN)},
                     control=3.0, join_treated=1.0)
    ours = _rows({tau: 0.5 for tau in range(INPUT_LEN)},
                 control=2.0, join_treated=0.5)

    advantages = plot._advantages_v2(ours, baseline, INPUT_LEN)
    for key in ("JoinLiftGain/MAPE", "FaultPenaltyReduction/MAPE",
                "OutageFaultPenaltyReduction/MAPE"):
        assert advantages[key]["treated_delta"] < 0, key
        assert advantages[key]["control_delta"] < 0, key
        assert advantages[key]["n_evaluable_cells"] >= 1, key


# --- arm classification is decided by the interval, not the point ------------


def _ci(low, high):
    return {"low": low, "high": high}


def test_arm_status_uses_the_interval_not_the_point_estimate():
    """A clearly negative point estimate whose interval spans zero is uncertain."""
    assert plot.arm_status(-0.9, _ci(-2.0, +0.5), 0.05) == "uncertain"
    assert plot.arm_status(-0.9, _ci(-2.0, -0.4), 0.05) == "improve"
    assert plot.arm_status(+0.9, _ci(+0.4, +2.0), 0.05) == "degrade"


def test_an_interval_inside_the_margin_is_equivalent_not_degraded():
    """The old point-estimate rule called an exactly-zero delta a degradation."""
    assert plot.arm_status(0.0, _ci(-0.01, +0.01), 0.05) == "equivalent"
    assert plot.arm_status(0.0, _ci(0.0, 0.0), 0.05) == "equivalent"


def test_a_missing_or_non_finite_interval_is_uncertain_never_a_win():
    assert plot.arm_status(-1.0, None, 0.05) == "uncertain"
    assert plot.arm_status(float("nan"), _ci(-2.0, -1.0), 0.05) == "uncertain"
    assert plot.arm_status(-1.0, _ci(float("nan"), -1.0), 0.05) == "uncertain"


def test_pareto_credits_a_contrast_only_when_treated_improves():
    assert plot.pareto_status("improve", "equivalent") == "accepted_treated_improves"
    assert plot.pareto_status("improve", "improve") == "accepted_treated_improves"
    # the gameable case: contrast looks better purely from a worse control arm
    assert plot.pareto_status("equivalent", "degrade") == "rejected_control_degraded"
    assert plot.pareto_status("improve", "degrade") == "rejected_control_degraded"
    assert plot.pareto_status("degrade", "improve") == "rejected_treated_degraded"
    assert plot.pareto_status("equivalent", "equivalent") == "no_material_change"
    assert plot.pareto_status("uncertain", "improve") == "uncertain"


def test_a_degraded_control_arm_is_never_accepted():
    """No arm classification may rescue a degraded control arm."""
    for treated in ("improve", "degrade", "equivalent", "uncertain"):
        assert not plot.pareto_status(treated, "degrade").startswith("accepted")


def test_margin_is_selected_per_metric():
    margins = {"MAE": 0.05, "MAPE": 0.25}
    assert plot.arm_margin_for("JoinLiftGain/MAE", margins) == 0.05
    assert plot.arm_margin_for("JoinLiftGain/MAPE", margins) == 0.25


# --- 4. saturated cohorts must not reach a main-plan number ------------------


def test_adding_a_saturated_cohort_does_not_move_the_main_plan_result():
    """The main plan names its cohorts; anything else is excluded by id."""
    main_ids = [1, 2]
    treated = {0: 2.0, 1: 9.0, 2: 3.0, 3: 5.0}
    plan_rows = _rows(treated, cohorts=(1, 2))
    # cohort 8 is a saturated large-event cohort with wildly different values
    polluted = plan_rows + _rows({tau: 99.0 for tau in range(INPUT_LEN)},
                                 control=50.0, cohorts=(8,))

    clean = paired_effects(build_curve_rows(main_cohort_rows(plan_rows, main_ids)),
                           INPUT_LEN, "MAPE")
    filtered = paired_effects(build_curve_rows(main_cohort_rows(polluted, main_ids)),
                              INPUT_LEN, "MAPE")
    assert clean == filtered

    # and without the filter the pollution is visible, so the test is not vacuous
    unfiltered = paired_effects(build_curve_rows(polluted), INPUT_LEN, "MAPE")
    assert unfiltered["FaultPenalty"] != pytest.approx(clean["FaultPenalty"])


# --- 5. calendar weighting is the protocol's, not equal-weight ---------------


def test_calendar_weights_are_five_sevenths_and_two_sevenths():
    weights = stratum_weights()
    assert sum(weights.values()) == pytest.approx(1.0)
    weekday = sum(v for k, v in weights.items() if k.startswith("Weekday/"))
    weekend = sum(v for k, v in weights.items() if k.startswith("Weekend/"))
    assert weekday == pytest.approx(5 / 7)
    assert weekend == pytest.approx(2 / 7)
    # equal within a day type
    for prefix in ("Weekday/", "Weekend/"):
        bins = [v for k, v in weights.items() if k.startswith(prefix)]
        assert len(set(round(v, 12) for v in bins)) == 1


def test_curves_use_calendar_shares_not_an_equal_regime_mean():
    """A weekday-heavy value must dominate; equal weighting would not show it."""
    rows = []
    for tau in range(INPUT_LEN):
        for track in ("join", "fault"):
            rows.append(_row(track, tau, 1, 10.0, 0.0, "Weekday/Midday", origin=1))
            rows.append(_row(track, tau, 1, 0.0, 0.0, "Weekend/Midday", origin=2))
    curves = build_curve_rows(rows)
    treated = curves[0]["treated"]["MAPE"]
    # weights inside one day type are equal, so this is 5/7 vs 2/7 across types
    equal_weighted = (10.0 + 0.0) / 2 * 100
    share_weighted = (10.0 * (5 / 7) + 0.0 * (2 / 7)) * 100
    assert treated == pytest.approx(share_weighted)
    assert treated != pytest.approx(equal_weighted)


# --- the gameable contrast, end to end through the bootstrap ------------------


def test_control_only_degradation_is_rejected_by_the_full_pipeline():
    """Build the exploit, run the real bootstrap, require a rejection.

    Treated arm identical, control arm clearly worse. The JoinLift contrast
    improves, and the verdict must still refuse it.
    """
    flat = {tau: 1.0 for tau in range(INPUT_LEN)}
    baseline = _rows(flat, control=1.0, join_treated=1.0)
    ours = _rows(flat, control=3.0, join_treated=1.0)
    for origin, rows in ((1, baseline), (1, ours)):
        for row in rows:
            row["event_time_index"] = row["origin_id"] * 1000

    result = plot.bootstrap_advantages_v2(
        ours, baseline, INPUT_LEN, samples=120, seed=11, trajectory_span=8,
        arm_margins={"MAE": 0.01, "MAPE": 0.05})
    join = result["JoinLiftGain/MAPE"]

    assert join["advantage"] > 0, "fixture must look like a win on the contrast"
    assert join["treated_status"] == "equivalent", join["treated_status"]
    assert join["control_status"] == "degrade", join["control_status"]
    assert join["pareto"] == "rejected_control_degraded"


def test_zero_samples_yields_uncertain_rather_than_a_point_verdict():
    flat = {tau: 1.0 for tau in range(INPUT_LEN)}
    result = plot.bootstrap_advantages_v2(
        _rows(flat, control=3.0), _rows(flat, control=1.0), INPUT_LEN, samples=0)
    for entry in result.values():
        assert entry["pareto"] == "uncertain"
        assert entry["treated_status"] == "uncertain"
