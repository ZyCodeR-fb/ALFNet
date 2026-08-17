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

import json

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


def _find(records, effect, metric):
    """bootstrap_advantages_v2 now returns a list of canonical records."""
    return next(r for r in records if r["effect"] == effect and r["metric"] == metric)


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


def test_single_seed_reuses_the_multiseed_status_semantics():
    """Parity, not a second implementation.

    This file used to carry its own status machine and it had drifted: it
    classified the arms without ever reading the contrast.
    """
    from scripts import multiseed_paired_inference as multiseed

    for treated in ("improve", "degrade", "equivalent", "uncertain"):
        for control in ("improve", "degrade", "equivalent", "uncertain"):
            assert plot.arms_pareto_status(treated, control) == \
                multiseed.arms_pareto_status(treated, control)
            for contrast in ("favourable", "degraded", "equivalent", "uncertain"):
                assert plot.combined_status(contrast, treated, control) == \
                    multiseed.combined_status(contrast, treated, control)


def test_arm_and_contrast_adapters_match_the_multiseed_functions():
    from scripts import multiseed_paired_inference as multiseed

    for delta, low, high in ((-1.0, -2.0, -0.5), (1.0, 0.5, 2.0),
                             (0.0, -0.01, 0.01), (0.0, -5.0, 5.0)):
        assert plot.arm_status(delta, {"low": low, "high": high}, 0.05) == \
            multiseed._arm_status(delta, low, high, 0.05)
        assert plot.contrast_status(delta, {"low": low, "high": high}, 0.25) == \
            multiseed._interval_status(delta, low, high, 0.25, True)


def test_no_confirmatory_wording_survives_in_the_status_values():
    banned = ("accept", "reject", "pass", "fail", "significant", "formal")
    values = set()
    for treated in ("improve", "degrade", "equivalent", "uncertain"):
        for control in ("improve", "degrade", "equivalent", "uncertain"):
            values.add(plot.arms_pareto_status(treated, control))
            for contrast in ("favourable", "degraded", "equivalent", "uncertain"):
                values.add(plot.combined_status(contrast, treated, control))
    for value in values:
        assert not any(word in value.lower() for word in banned), value


def test_margin_is_selected_per_metric():
    margins = {"MAE": 0.05, "MAPE": 0.25, "RMSE": None}
    assert plot._margin_for_endpoint("JoinLiftGain/MAE", margins) == 0.05
    assert plot._margin_for_endpoint("JoinLiftGain/MAPE", margins) == 0.25


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
        ours, baseline, INPUT_LEN, samples=1000, seed=11, trajectory_span=8,
        arm_margins={"MAE": 0.01, "MAPE": 0.05, "RMSE": None},
        contrast_margins={"MAE": 0.01, "MAPE": 0.05, "RMSE": None})
    join = _find(result, "JoinLift", "MAPE")

    assert join["comparison"]["contrast_oriented_advantage"]["point"] > 0
    assert join["contrast_status"] == "favourable"
    assert join["treated_status"] == "equivalent", join["treated_status"]
    assert join["control_status"] == "degrade", join["control_status"]
    assert join["arms_pareto_status"] == "control_degraded"
    assert join["combined_descriptive_status"] == \
        "contrast_favourable_but_control_degraded"


@pytest.mark.parametrize("samples", [0, -1, 999])
def test_too_few_samples_are_refused_not_degraded_to_uncertain(samples):
    """A summary that silently calls every endpoint uncertain looks measured."""
    flat = {tau: 1.0 for tau in range(INPUT_LEN)}
    with pytest.raises(plot.NonFiniteValue, match="at least 1000"):
        plot.bootstrap_advantages_v2(
            _rows(flat, control=3.0), _rows(flat, control=1.0), INPUT_LEN,
            samples=samples, trajectory_span=8)


def test_a_non_integer_sample_count_is_refused():
    flat = {tau: 1.0 for tau in range(INPUT_LEN)}
    with pytest.raises(plot.NonFiniteValue, match="must be an integer"):
        plot.bootstrap_advantages_v2(
            _rows(flat, control=3.0), _rows(flat, control=1.0), INPUT_LEN,
            samples=1000.5, trajectory_span=8)


def test_both_arms_improve_but_contrast_degrades_is_not_all_conditions_met():
    """Control improves more than treated, so the contrast genuinely worsens."""
    flat = {tau: 1.0 for tau in range(INPUT_LEN)}
    baseline = _rows(flat, control=3.0, join_treated=1.0)
    ours = _rows(flat, control=1.0, join_treated=0.9)

    result = plot.bootstrap_advantages_v2(
        ours, baseline, INPUT_LEN, samples=1000, seed=5, trajectory_span=8,
        arm_margins={"MAE": 0.01, "MAPE": 0.05, "RMSE": None},
        contrast_margins={"MAE": 0.01, "MAPE": 0.05, "RMSE": None})
    join = _find(result, "JoinLift", "MAPE")

    assert join["treated_status"] == "improve"
    assert join["control_status"] == "improve"
    assert join["arms_pareto_status"] == "treated_improves_control_non_degraded"
    assert join["contrast_status"] == "degraded"
    assert join["combined_descriptive_status"] == "contrast_degraded"
    assert join["combined_descriptive_status"] != "all_conditions_met"


def test_contrast_and_arm_margins_move_independently():
    flat = {tau: 1.0 for tau in range(INPUT_LEN)}
    ours = _rows(flat, control=1.02, join_treated=0.98)
    baseline = _rows(flat, control=1.0, join_treated=1.0)
    tight = {"MAE": 0.001, "MAPE": 0.005, "RMSE": None}
    wide = {"MAE": 1000.0, "MAPE": 1000.0, "RMSE": None}

    wide_contrast = _find(plot.bootstrap_advantages_v2(
        ours, baseline, INPUT_LEN, samples=1000, seed=7, trajectory_span=8,
        arm_margins=tight, contrast_margins=wide), "JoinLift", "MAPE")
    wide_arm = _find(plot.bootstrap_advantages_v2(
        ours, baseline, INPUT_LEN, samples=1000, seed=7, trajectory_span=8,
        arm_margins=wide, contrast_margins=tight), "JoinLift", "MAPE")

    # a wide contrast margin only flattens the contrast verdict
    assert wide_contrast["contrast_status"] == "equivalent"
    assert wide_contrast["treated_status"] != "equivalent"
    # and a wide arm margin only flattens the arms
    assert wide_arm["treated_status"] == "equivalent"
    assert wide_arm["control_status"] == "equivalent"
    assert wide_arm["contrast_status"] != "equivalent"


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_a_non_finite_draw_fails_closed_with_context(bad, monkeypatch):
    flat = {tau: 1.0 for tau in range(INPUT_LEN)}
    real = plot._advantages_v2
    state = {"n": 0}

    def _poison(*args, **kwargs):
        out = real(*args, **kwargs)
        state["n"] += 1
        if state["n"] == 3:
            for entry in out.values():
                entry["advantage"] = bad
        return out

    monkeypatch.setattr(plot, "_advantages_v2", _poison)
    with pytest.raises(plot.NonFiniteValue, match="non-finite"):
        plot.bootstrap_advantages_v2(
            _rows(flat, control=3.0), _rows(flat, control=1.0), INPUT_LEN,
            samples=1000, seed=3, trajectory_span=8)


# --- identity: the two bundles must be the same experiment -------------------


def _bundle(model, seed, dataset="METRLA", fingerprint="fp-1", protocol="p-1"):
    return {
        "model": model, "seed": seed,
        "protocol_hash": protocol, "training_protocol_hash": "t-1",
        "lce_protocol_hash": "l-1", "ples_protocol_hash": "e-1",
        "dataset": {"name": dataset, "fingerprint": fingerprint,
                    "adjacency_hash": "a-1", "timestamps_hash": "s-1",
                    "test_values_hash": "v-1"},
    }


def test_a_cross_seed_comparison_is_refused():
    """The live defect: ALFNet s32 vs MoGERNN s52 was accepted.

    A shared protocol hash says both were evaluated under one design; it does
    not say they are the same experiment. Comparing across seeds attributes a
    training-randomness difference to the model.
    """
    with pytest.raises(ValueError, match="seed"):
        plot.assert_comparable_bundles(_bundle("ALFNet", 32), _bundle("MoGERNN", 52))


def test_a_same_model_comparison_is_refused():
    with pytest.raises(ValueError, match="needs two models"):
        plot.assert_comparable_bundles(_bundle("ALFNet", 32), _bundle("ALFNet", 32))


def test_a_cross_dataset_comparison_is_refused():
    with pytest.raises(ValueError, match="dataset.name"):
        plot.assert_comparable_bundles(
            _bundle("ALFNet", 32, dataset="METRLA"),
            _bundle("MoGERNN", 32, dataset="PEMSBAY"))


def test_a_different_dataset_fingerprint_is_refused():
    """Same name, different bytes underneath."""
    with pytest.raises(ValueError, match="fingerprint"):
        plot.assert_comparable_bundles(
            _bundle("ALFNet", 32, fingerprint="fp-1"),
            _bundle("MoGERNN", 32, fingerprint="fp-2"))


def test_mismatched_protocol_hashes_are_refused():
    left = _bundle("ALFNet", 32)
    right = _bundle("MoGERNN", 32)
    right["lce_protocol_hash"] = "l-2"
    with pytest.raises(ValueError, match="lce_protocol_hash"):
        plot.assert_comparable_bundles(left, right)


def test_the_matched_pair_is_accepted():
    plot.assert_comparable_bundles(_bundle("ALFNet", 32), _bundle("MoGERNN", 32))


# --- single-seed main(): summary assembly and JSON write ---------------------


def _bundle_dir(tmp_path, name, model, variant, seed, rows):
    """A minimal maintained-schema PLES v3 bundle on disk."""
    import csv as _csv

    directory = tmp_path / name
    directory.mkdir(parents=True)
    (directory / "ples_effects_summary.json").write_text(json.dumps({
        "schema_version": 3, "model": model, "model_variant": variant, "seed": seed,
        "formal_protocol_match": True,
        "protocol_hash": "p" * 8, "training_protocol_hash": "t" * 8,
        "lce_protocol_hash": "l" * 8, "ples_protocol_hash": "e" * 8,
        "dataset": {"name": "METRLA", "fingerprint": "f", "adjacency_hash": "a",
                    "timestamps_hash": "s", "test_values_hash": "v"},
        "bootstrap": {"trajectory_span": 8},
    }), encoding="utf-8")
    (directory / "protocol_manifest.json").write_text(json.dumps({
        "resolved": {"input_len": INPUT_LEN, "main_cohort_ids": [1]},
    }), encoding="utf-8")
    fields = list(rows[0])
    with (directory / "ples_paired_rows.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = _csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return directory


def _single_seed_main(tmp_path, monkeypatch, samples=1000):
    import sys

    flat = {tau: 1.0 for tau in range(INPUT_LEN)}
    ours = _bundle_dir(tmp_path, "ours", "ALFNet", "gs", 32,
                       _rows(flat, control=3.0, join_treated=0.5))
    baseline = _bundle_dir(tmp_path, "base", "MoGERNN", "mg_l3", 32,
                           _rows(flat, control=3.0, join_treated=1.0))
    out = tmp_path / "out"

    # plotting is not what this test is about
    monkeypatch.setattr(plot, "plot_comparison_v2", lambda *a, **k: None)
    monkeypatch.setattr(sys, "argv", [
        "plot_counterfactual_shockwave.py", "--ours", str(ours),
        "--baseline", str(baseline), "--output-dir", str(out),
        "--bootstrap-samples", str(samples),
    ])
    plot.main()
    return json.loads((out / "paired_model_comparison_summary.json").read_text("utf-8"))


def test_single_seed_main_writes_the_current_schema(tmp_path, monkeypatch):
    payload = _single_seed_main(tmp_path, monkeypatch)

    assert payload["schema_version"] == 5
    assert payload["post_hoc_diagnostic_only"] is True
    assert payload["inferential_status"] == "descriptive_not_formal"
    assert payload["conditional_on_one_trained_seed"] is True
    assert payload["trained_seed"] == 32
    assert "does not include training randomness" in payload["interval_scope"]

    # the two margin families are separate top-level fields
    assert payload["post_hoc_contrast_margins"]["MAE"] == 0.05
    assert payload["post_hoc_arm_margins"]["MAE"] == 0.05

    assert "advantages" not in payload, "the old flat structure must be gone"
    records = payload["ples_records"]
    assert len(records) == 6
    assert {(r["effect"], r["metric"]) for r in records} == {
        (e, m) for e in ("JoinLift", "FaultPenalty", "OutageFaultPenalty")
        for m in ("MAE", "MAPE")}
    for entry in records:
        for field in ("contrast_status", "treated_status", "control_status",
                      "arms_pareto_status", "combined_descriptive_status"):
            assert field in entry, field
        assert "pareto" not in entry
        for model in ("ALFNet", "MoGERNN"):
            for quantity in ("contrast", "treated_absolute_error",
                             "control_absolute_error"):
                block = entry["models"][model][quantity]
                assert "point" in block
                assert set(block["nominal_95_percentile_interval"]) == {"low", "high"}
            assert "algebra_identity" in entry["models"][model]
        assert "comparison_algebra_identity" in entry


def test_single_seed_main_emits_no_confirmatory_wording(tmp_path, monkeypatch):
    payload = _single_seed_main(tmp_path, monkeypatch)
    statuses = []
    for entry in payload["ples_records"]:
        statuses += [entry["contrast_status"], entry["treated_status"],
                     entry["control_status"], entry["arms_pareto_status"],
                     entry["combined_descriptive_status"]]
    for value in statuses:
        assert not any(word in value.lower()
                       for word in ("accept", "reject", "pass", "fail",
                                    "significant", "formal")), value


def test_single_seed_main_output_has_no_non_finite_constant(tmp_path, monkeypatch):
    _single_seed_main(tmp_path, monkeypatch)
    raw = (tmp_path / "out" / "paired_model_comparison_summary.json").read_text("utf-8")

    def _reject(name):
        raise AssertionError(f"non-finite constant {name!r} in written JSON")

    json.loads(raw, parse_constant=_reject)


def test_single_seed_cli_refuses_too_few_samples(tmp_path, monkeypatch):
    with pytest.raises(SystemExit):
        _single_seed_main(tmp_path, monkeypatch, samples=999)


# --- raw model draws, same replicate, three-layer algebra --------------------


def _bootstrap(ours, baseline, samples=1000, seed=13, **kw):
    return plot.bootstrap_advantages_v2(
        ours, baseline, INPUT_LEN, samples=samples, seed=seed,
        trajectory_span=8, **kw)


def _varied(join_treated, control, fault_scale=1.0):
    """Two origins with different values so draws actually vary."""
    rows = []
    for origin, bump in ((1, 0.0), (2, 0.7)):
        for tau in range(INPUT_LEN):
            rows.append(_row("join", tau, 1, join_treated + bump, control + bump,
                             origin=origin))
            rows.append(_row("fault", tau, 1, fault_scale + bump, control + bump,
                             origin=origin))
    return rows


def test_every_record_carries_both_models_raw_contrast_and_arms():
    records = _bootstrap(_varied(0.5, 1.0), _varied(1.0, 1.0))
    assert len(records) == 6
    for record in records:
        for model in ("ALFNet", "MoGERNN"):
            block = record["models"][model]
            for quantity in ("contrast", "treated_absolute_error",
                             "control_absolute_error"):
                assert "point" in block[quantity]
                interval = block[quantity]["nominal_95_percentile_interval"]
                assert interval["low"] <= interval["high"]
        for quantity in ("contrast_delta", "treated_delta", "control_delta",
                         "contrast_oriented_advantage"):
            assert "point" in record["comparison"][quantity]


def test_the_oriented_advantage_follows_the_endpoint_orientation():
    records = _bootstrap(_varied(0.5, 1.0), _varied(1.0, 1.0))
    for record in records:
        delta = record["comparison"]["contrast_delta"]["point"]
        oriented = record["comparison"]["contrast_oriented_advantage"]["point"]
        if record["effect"] == "JoinLift":
            assert oriented == pytest.approx(delta)
        else:
            assert oriented == pytest.approx(-delta)


def test_the_contrast_delta_point_is_the_raw_model_difference():
    records = _bootstrap(_varied(0.5, 1.0), _varied(1.0, 1.0))
    for record in records:
        ours = record["models"]["ALFNet"]["contrast"]["point"]
        baseline = record["models"]["MoGERNN"]["contrast"]["point"]
        assert record["comparison"]["contrast_delta"]["point"] == \
            pytest.approx(ours - baseline)
        for arm, quantity in (("treated_delta", "treated_absolute_error"),
                              ("control_delta", "control_absolute_error")):
            assert record["comparison"][arm]["point"] == pytest.approx(
                record["models"]["ALFNet"][quantity]["point"]
                - record["models"]["MoGERNN"][quantity]["point"])


def test_all_three_algebra_identities_are_reported_and_within_tolerance():
    records = _bootstrap(_varied(0.5, 1.0), _varied(1.0, 1.2))
    for record in records:
        expected = ("contrast = control - treated"
                    if record["effect"] == "JoinLift" else "contrast = treated - control")
        checks = [record["comparison_algebra_identity"],
                  record["models"]["ALFNet"]["algebra_identity"],
                  record["models"]["MoGERNN"]["algebra_identity"]]
        for algebra in checks:
            assert algebra["formula"] == expected
            assert abs(algebra["point_residual"]) <= algebra["tolerance"]
            assert algebra["max_abs_per_draw_residual"] <= algebra["tolerance"]


def test_a_broken_raw_model_identity_fails_closed(monkeypatch):
    """Both models corrupted identically: the comparison residual stays zero.

    Only the per-model identities can see this, which is why the comparison
    check alone is insufficient.
    """
    real = plot._advantages_v2

    def _corrupt(*args, **kwargs):
        out = real(*args, **kwargs)
        for entry in out.values():
            entry["ours_effect"] += 1.0
            entry["baseline_effect"] += 1.0
        return out

    monkeypatch.setattr(plot, "_advantages_v2", _corrupt)
    with pytest.raises(plot.NonFiniteValue, match="not the signed difference"):
        _bootstrap(_varied(0.5, 1.0), _varied(1.0, 1.0))


def test_a_single_model_corruption_fails_closed(monkeypatch):
    real = plot._advantages_v2

    def _corrupt(*args, **kwargs):
        out = real(*args, **kwargs)
        for entry in out.values():
            entry["ours_effect"] += 1.0
        return out

    monkeypatch.setattr(plot, "_advantages_v2", _corrupt)
    with pytest.raises(plot.NonFiniteValue):
        _bootstrap(_varied(0.5, 1.0), _varied(1.0, 1.0))


def test_a_delta_that_is_not_the_raw_difference_fails_closed(monkeypatch):
    real = plot._advantages_v2

    def _corrupt(*args, **kwargs):
        out = real(*args, **kwargs)
        for entry in out.values():
            entry["treated_delta"] += 1.0
        return out

    monkeypatch.setattr(plot, "_advantages_v2", _corrupt)
    with pytest.raises(plot.NonFiniteValue, match="not ALFNet minus"):
        _bootstrap(_varied(0.5, 1.0), _varied(1.0, 1.0))


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_a_non_finite_raw_draw_fails_closed(bad, monkeypatch):
    real = plot._advantages_v2
    state = {"n": 0}

    def _poison(*args, **kwargs):
        out = real(*args, **kwargs)
        state["n"] += 1
        if state["n"] == 4:
            for entry in out.values():
                entry["ours_treated"] = bad
        return out

    monkeypatch.setattr(plot, "_advantages_v2", _poison)
    with pytest.raises(plot.NonFiniteValue):
        _bootstrap(_varied(0.5, 1.0), _varied(1.0, 1.0))


def test_outage_still_reads_only_the_settled_tau():
    """treated rises with tau, so Outage must not equal the Decay-wide mean."""
    ours = _rows({0: 1.0, 1: 2.0, 2: 3.0, 3: 9.0}, control=1.0)
    baseline = _rows({0: 1.0, 1: 2.0, 2: 3.0, 3: 4.0}, control=1.0)
    records = _bootstrap(ours, baseline)
    outage = _find(records, "OutageFaultPenalty", "MAE")
    penalty = _find(records, "FaultPenalty", "MAE")
    assert outage["models"]["ALFNet"]["treated_absolute_error"]["point"] != \
        pytest.approx(penalty["models"]["ALFNet"]["treated_absolute_error"]["point"])
