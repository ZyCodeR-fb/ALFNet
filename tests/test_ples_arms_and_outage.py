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


def _row(track, tau, cohort, treated, control, regime="Weekday/Midday", origin=1,
         model="ALFNet", dataset="METRLA"):
    """A row carrying the full closed schema the strict loader requires."""
    row = {
        "dataset": dataset, "model": model,
        "origin_id": origin, "cohort_id": cohort, "track": track, "tau": tau,
        "phase": _phase(track, tau), "count": 10, "regime": regime,
        "calendar_day": "2026-01-01", "event_time_index": origin * 1000,
        "event_timestamp": "2026-01-01 10:00:00",
        "cohort_fraction": 0.05, "cohort_style": "clustered", "node_count": 4,
        "graph_weight": "", "patch_residual_weight": "",
    }
    for field in SUM_FIELDS:
        row[field] = (treated if field.startswith("treated") else control) * 10
    return row


def _rows(fault_treated_by_tau, control=1.0, cohorts=(1,), regime="Weekday/Midday",
          join_treated=1.0, model="ALFNet"):
    """Both tracks scripted: fault treated per tau, join treated flat.

    ``join_treated`` is a parameter rather than a constant so a test can move
    the join treated arm; with it pinned, every JoinLift ``treated_delta`` is
    identically zero and an arm-movement assertion cannot fail.
    """
    rows = []
    for cohort in cohorts:
        for tau in range(INPUT_LEN):
            rows.append(_row("join", tau, cohort, join_treated, control, regime,
                             model=model))
            rows.append(_row("fault", tau, cohort,
                             fault_treated_by_tau[tau], control, regime,
                             model=model))
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
        assert "n_evaluable_cells" not in advantages[key], (
            "the misleading tau-curve count must not be a point statistic")


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
                       _rows(flat, control=3.0, join_treated=0.5, model="ALFNet"))
    baseline = _bundle_dir(tmp_path, "base", "MoGERNN", "mg_l3", 32,
                           _rows(flat, control=3.0, join_treated=1.0, model="MoGERNN"))
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

    assert payload["schema_version"] == 7
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
    # Caught at the point-derivation guard, which runs before the draw check.
    with pytest.raises(plot.NonFiniteValue, match="not the raw model difference"):
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


# --- exact inventory and point-level derivation (H2) -------------------------


def _corrupting(mutate):
    """Wrap _advantages_v2 so a test can perturb its output in place."""
    real = plot._advantages_v2

    def _wrapped(*args, **kwargs):
        out = real(*args, **kwargs)
        mutate(out)
        return out

    return _wrapped


def test_the_expected_inventory_is_six_endpoints_from_the_frozen_definitions():
    assert len(plot.EXPECTED_ENDPOINTS) == 6
    assert set(plot.EXPECTED_ENDPOINTS) == {
        f"{plot.ADVANTAGE_NAME[effect]}/{metric}"
        for effect in ("JoinLift", "FaultPenalty", "OutageFaultPenalty")
        for metric in ("MAE", "MAPE")}
    assert set(plot.EXPECTED_POINT_FIELDS) == {
        "ours_effect", "baseline_effect", "ours_treated", "baseline_treated",
        "ours_control", "baseline_control", "advantage", "treated_delta",
        "control_delta"}


def test_records_are_always_exactly_six_without_duplicates():
    records = _bootstrap(_varied(0.5, 1.0), _varied(1.0, 1.0))
    keys = [(r["effect"], r["metric"]) for r in records]
    assert len(records) == 6
    assert len(set(keys)) == 6


def test_a_missing_point_endpoint_fails_closed(monkeypatch):
    def _drop(out):
        out.pop("JoinLiftGain/MAE")

    monkeypatch.setattr(plot, "_advantages_v2", _corrupting(_drop))
    with pytest.raises(plot.NonFiniteValue, match="point endpoints missing"):
        _bootstrap(_varied(0.5, 1.0), _varied(1.0, 1.0))


def test_an_unknown_point_endpoint_fails_closed(monkeypatch):
    def _add(out):
        out["InventedGain/MAE"] = dict(out["JoinLiftGain/MAE"])

    monkeypatch.setattr(plot, "_advantages_v2", _corrupting(_add))
    with pytest.raises(plot.NonFiniteValue, match="extra"):
        _bootstrap(_varied(0.5, 1.0), _varied(1.0, 1.0))


def test_a_missing_point_field_fails_closed(monkeypatch):
    def _drop_field(out):
        out["JoinLiftGain/MAE"].pop("ours_treated")

    monkeypatch.setattr(plot, "_advantages_v2", _corrupting(_drop_field))
    with pytest.raises(plot.NonFiniteValue, match="point fields missing"):
        _bootstrap(_varied(0.5, 1.0), _varied(1.0, 1.0))


def test_an_unknown_point_field_fails_closed(monkeypatch):
    def _add_field(out):
        out["JoinLiftGain/MAE"]["smuggled_statistic"] = 1.0

    monkeypatch.setattr(plot, "_advantages_v2", _corrupting(_add_field))
    with pytest.raises(plot.NonFiniteValue, match="point fields"):
        _bootstrap(_varied(0.5, 1.0), _varied(1.0, 1.0))


def test_a_missing_draw_field_fails_closed():
    from scripts import plot_counterfactual_shockwave as module

    point = {name: {field: 1.0 for field in module.EXPECTED_POINT_FIELDS}
             for name in module.EXPECTED_ENDPOINTS}
    draws = {(name, field): np.ones(10)
             for name in module.EXPECTED_ENDPOINTS
             for field in module.EXPECTED_POINT_FIELDS}
    draws.pop((module.EXPECTED_ENDPOINTS[0], "ours_effect"))
    with pytest.raises(module.NonFiniteValue, match="draws missing"):
        module._build_ples_records(point, draws, module.EXPECTED_POINT_FIELDS,
                                   INPUT_LEN, {"MAE": 0.05, "MAPE": 0.25, "RMSE": None},
                                   {"MAE": 0.05, "MAPE": 0.25, "RMSE": None})


def test_an_extra_draw_field_fails_closed():
    from scripts import plot_counterfactual_shockwave as module

    point = {name: {field: 1.0 for field in module.EXPECTED_POINT_FIELDS}
             for name in module.EXPECTED_ENDPOINTS}
    draws = {(name, field): np.ones(10)
             for name in module.EXPECTED_ENDPOINTS
             for field in module.EXPECTED_POINT_FIELDS}
    draws[(module.EXPECTED_ENDPOINTS[0], "smuggled")] = np.ones(10)
    with pytest.raises(module.NonFiniteValue, match="draws missing|extra"):
        module._build_ples_records(point, draws, module.EXPECTED_POINT_FIELDS,
                                   INPUT_LEN, {"MAE": 0.05, "MAPE": 0.25, "RMSE": None},
                                   {"MAE": 0.05, "MAPE": 0.25, "RMSE": None})


def test_a_shifted_point_delta_fails_closed_even_when_algebra_cancels(monkeypatch):
    """The case the comparison algebra alone cannot see.

    Shifting treated_delta and control_delta by the same amount leaves
    ``treated - control`` unchanged, so the comparison residual stays zero --
    but neither point is the raw model difference any more.
    """
    def _shift_both(out):
        for entry in out.values():
            entry["treated_delta"] += 1.0
            entry["control_delta"] += 1.0

    monkeypatch.setattr(plot, "_advantages_v2", _corrupting(_shift_both))
    with pytest.raises(plot.NonFiniteValue, match="not the raw model difference"):
        _bootstrap(_varied(0.5, 1.0), _varied(1.0, 1.0))


def test_a_corrupted_oriented_advantage_point_fails_closed(monkeypatch):
    def _shift(out):
        for entry in out.values():
            entry["advantage"] += 1.0

    monkeypatch.setattr(plot, "_advantages_v2", _corrupting(_shift))
    with pytest.raises(plot.NonFiniteValue, match="advantage.*not the raw model"):
        _bootstrap(_varied(0.5, 1.0), _varied(1.0, 1.0))


def test_the_oriented_point_follows_the_frozen_orientation():
    records = _bootstrap(_varied(0.5, 1.0), _varied(1.0, 1.0))
    for record in records:
        ours = record["models"]["ALFNet"]["contrast"]["point"]
        baseline = record["models"]["MoGERNN"]["contrast"]["point"]
        oriented = record["comparison"]["contrast_oriented_advantage"]["point"]
        if record["effect"] == "JoinLift":
            assert oriented == pytest.approx(ours - baseline)
        else:
            assert oriented == pytest.approx(-(ours - baseline))


# --- the single-seed entry point uses the strict loader (B) ------------------
#
# This file used to carry its own permissive parser: a few int/float casts and
# nothing else. It skipped the closed header, the finite/non-negative checks,
# the count and cohort_fraction checks and the track/phase consistency check, so
# a corrupted row reached the statistics unchallenged. There is now one loader.


def _write_csv(directory, rows, header=None, extra_lines=()):
    directory.mkdir(parents=True, exist_ok=True)
    fields = header if header is not None else list(rows[0])
    lines = [",".join(fields)]
    for row in rows:
        lines.append(",".join(str(row[f]) if f in row else "" for f in fields))
    lines.extend(extra_lines)
    (directory / "ples_paired_rows.csv").write_text("\n".join(lines) + "\n",
                                                    encoding="utf-8")
    return directory


def _bundle_with_rows(tmp_path, name, rows, header=None, model="ALFNet",
                      variant="gs", seed=32, schema=3):
    directory = tmp_path / name
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "ples_effects_summary.json").write_text(json.dumps({
        "schema_version": schema, "model": model, "model_variant": variant,
        "seed": seed, "formal_protocol_match": True,
        "protocol_hash": "p" * 8, "training_protocol_hash": "t" * 8,
        "lce_protocol_hash": "l" * 8, "ples_protocol_hash": "e" * 8,
        "dataset": {"name": "METRLA", "fingerprint": "f", "adjacency_hash": "a",
                    "timestamps_hash": "s", "test_values_hash": "v"},
        "bootstrap": {"trajectory_span": 8},
    }), encoding="utf-8")
    (directory / "protocol_manifest.json").write_text(json.dumps({
        "resolved": {"input_len": INPUT_LEN, "main_cohort_ids": [1]},
    }), encoding="utf-8")
    _write_csv(directory, rows, header)
    return directory


def _flat_rows(model="ALFNet"):
    return _rows({tau: 1.0 for tau in range(INPUT_LEN)}, control=1.0, model=model)


def test_the_single_seed_entry_point_rejects_a_duplicate_header(tmp_path):
    from scripts.paired_model_bootstrap import InvalidSchema

    rows = _flat_rows()
    header = list(rows[0]) + ["treated_ae_sum"]
    directory = _bundle_with_rows(tmp_path, "dup", rows, header=header)
    with pytest.raises(InvalidSchema, match="duplicate column"):
        plot._read_result_bundle(directory)


def test_the_single_seed_entry_point_rejects_a_missing_column(tmp_path):
    from scripts.paired_model_bootstrap import InvalidSchema

    rows = _flat_rows()
    header = [f for f in rows[0] if f != "regime"]
    directory = _bundle_with_rows(tmp_path, "miss", rows, header=header)
    with pytest.raises(InvalidSchema, match="missing required column"):
        plot._read_result_bundle(directory)


def test_the_single_seed_entry_point_rejects_an_unexpected_column(tmp_path):
    from scripts.paired_model_bootstrap import InvalidSchema

    rows = _flat_rows()
    for row in rows:
        row["smuggled"] = 1
    directory = _bundle_with_rows(tmp_path, "extra", rows)
    with pytest.raises(InvalidSchema, match="unexpected column"):
        plot._read_result_bundle(directory)


@pytest.mark.parametrize("field,value,match", [
    ("treated_ae_sum", "nan", "not finite"),
    ("treated_ae_sum", "inf", "not finite"),
    ("treated_ae_sum", "-1.0", "negative"),
    ("count", "0", "positive integer"),
    ("node_count", "0", "positive integer"),
    ("cohort_fraction", "1.5", "cohort_fraction"),
    ("dataset", "", "dataset"),
    ("model", "", "model"),
    ("event_timestamp", "", "event_timestamp"),
    ("track", "sideways", "track"),
    ("treated_ae_sum", "oops", "not a valid float"),
])
def test_the_single_seed_entry_point_rejects_a_malformed_row(tmp_path, field, value, match):
    from scripts.paired_model_bootstrap import InvalidRow

    rows = _flat_rows()
    rows[0][field] = value
    directory = _bundle_with_rows(tmp_path, f"bad-{field}-{value}", rows)
    with pytest.raises(InvalidRow, match=match):
        plot._read_result_bundle(directory)


def test_the_single_seed_entry_point_rejects_a_phase_track_mismatch(tmp_path):
    from scripts.paired_model_bootstrap import InvalidRow

    rows = _flat_rows()
    rows[0]["track"] = "join"
    rows[0]["phase"] = "Outage"
    directory = _bundle_with_rows(tmp_path, "phase", rows)
    with pytest.raises(InvalidRow, match="not a join phase"):
        plot._read_result_bundle(directory)


def test_a_retired_schema_is_refused_before_any_row_is_read(tmp_path):
    """Refused for being retired, not for a confusing column error."""
    rows = _flat_rows()
    directory = _bundle_with_rows(tmp_path, "old", rows, schema=2)
    with pytest.raises(ValueError, match="predate the maintained protocol"):
        plot._read_result_bundle(directory)


def test_there_is_no_fallback_to_the_retired_row_filename(tmp_path):
    """A maintained bundle without ples_paired_rows.csv is broken, not old."""
    directory = tmp_path / "nofile"
    directory.mkdir(parents=True)
    (directory / "ples_effects_summary.json").write_text(json.dumps({
        "schema_version": 3, "model": "ALFNet", "seed": 32,
        "dataset": {"name": "METRLA"},
    }), encoding="utf-8")
    (directory / "paired_event_rows.csv").write_text("anything\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="ples_paired_rows.csv"):
        plot._read_result_bundle(directory)


# --- rows must belong to their bundle and describe one design ---------------


def _summary(model="ALFNet", dataset="METRLA"):
    return {"model": model, "dataset": {"name": dataset}}


def test_a_row_model_that_contradicts_the_summary_is_refused():
    ours = _flat_rows(model="SomethingElse")
    baseline = _flat_rows(model="MoGERNN")
    with pytest.raises(ValueError, match="row model"):
        plot.assert_rows_match_bundles(ours, _summary("ALFNet"),
                                       baseline, _summary("MoGERNN"))


def test_a_row_dataset_that_contradicts_the_summary_is_refused():
    ours = _flat_rows()
    for row in ours:
        row["dataset"] = "PEMSBAY"
    baseline = _flat_rows(model="MoGERNN")
    with pytest.raises(ValueError, match="row dataset"):
        plot.assert_rows_match_bundles(ours, _summary("ALFNet"),
                                       baseline, _summary("MoGERNN"))


def test_a_missing_paired_row_is_refused():
    ours = _flat_rows()
    baseline = _flat_rows(model="MoGERNN")[:-1]
    with pytest.raises(ValueError, match="same paired cells"):
        plot.assert_rows_match_bundles(ours, _summary("ALFNet"),
                                       baseline, _summary("MoGERNN"))


@pytest.mark.parametrize("field,value", [
    ("event_timestamp", "2099-01-01 00:00:00"),
    ("regime", "Weekend/Midday"),
    ("cohort_style", "scattered"),
    ("node_count", 99),
    ("count", 77),
])
def test_a_protocol_field_that_differs_between_models_is_refused(field, value):
    ours = _flat_rows()
    baseline = _flat_rows(model="MoGERNN")
    baseline[0][field] = value
    with pytest.raises(ValueError, match=field):
        plot.assert_rows_match_bundles(ours, _summary("ALFNet"),
                                       baseline, _summary("MoGERNN"))


def test_a_duplicate_pair_key_is_refused():
    ours = _flat_rows()
    ours.append(dict(ours[0]))
    baseline = _flat_rows(model="MoGERNN")
    with pytest.raises(ValueError, match="duplicate paired row"):
        plot.assert_rows_match_bundles(ours, _summary("ALFNet"),
                                       baseline, _summary("MoGERNN"))


def test_a_matched_pair_of_row_files_is_accepted():
    plot.assert_rows_match_bundles(_flat_rows(), _summary("ALFNet"),
                                   _flat_rows(model="MoGERNN"), _summary("MoGERNN"))


# --- real sample counts (B2.1b-2) -------------------------------------------
#
# `n_evaluable_cells` was the number of tau curves left after calendar
# aggregation -- not origins, cohorts, raw cells or valid targets. Reporting it
# as an "evaluable" count invited it to be read as a sample size. It is gone,
# replaced by explicitly named counts that state what they are.


def _counts_fixture():
    """Two origins x two cohorts x both tracks x every tau, one weekday regime.

    Exact expected figures, so the assertions below are arithmetic rather than
    whatever the code happens to produce:
      rows        = 2 origins * 2 cohorts * 2 tracks * 4 taus = 32
      episodes    = 2 origins * 2 cohorts                     = 4
      targets     = 32 rows * count 10                        = 320
    """
    rows = []
    for origin in (1, 2):
        for cohort in (1, 2):
            for track in ("join", "fault"):
                for tau in range(INPUT_LEN):
                    rows.append(_row(track, tau, cohort, 2.0, 1.0, origin=origin))
    return rows


def test_top_level_counts_are_exact():
    rows = _counts_fixture()
    main = [r for r in rows if r["cohort_id"] == 1]
    counts = plot.top_level_sample_counts(rows, main, trajectory_span=8)

    assert counts["input"]["raw_row_count_per_model"] == 32
    assert counts["input"]["main_plan_row_count_per_model"] == 16
    assert counts["input"]["excluded_non_main_row_count_per_model"] == 16

    design = counts["paired_design"]
    assert design["paired_row_count"] == 16
    assert design["unique_origin_count"] == 2
    assert design["unique_cohort_count"] == 1
    assert design["unique_origin_cohort_episode_count"] == 2
    assert design["track_count"] == 2
    assert design["tau_value_count"] == INPUT_LEN
    assert design["min_tau"] == 0
    assert design["max_tau"] == INPUT_LEN - 1

    units = counts["bootstrap_units"]
    assert units["unique_origin_count"] == 2
    assert units["overlap_block_count"] >= 1
    assert units["cohort_count"] == 1
    assert units["day_type_strata"] == ["Weekday"]
    assert sum(units["blocks_per_day_type"].values()) == units["overlap_block_count"]

    assert counts["targets"]["paired_valid_target_element_count_sum"] == 16 * 10


def test_endpoint_counts_match_their_own_selection():
    rows = [r for r in _counts_fixture() if r["cohort_id"] == 1]
    curves = plot._curves_for(rows, None)
    counts = plot.endpoint_sample_counts(rows, curves, INPUT_LEN)

    # Join and Fault take every post-event tau; Outage takes one.
    assert counts["JoinLift"]["source_row_count"] == 2 * INPUT_LEN
    assert counts["FaultPenalty"]["source_row_count"] == 2 * INPUT_LEN
    assert counts["OutageFaultPenalty"]["source_row_count"] == 2
    assert counts["OutageFaultPenalty"]["unique_tau_count"] == 1
    assert counts["JoinLift"]["tau_curve_count"] == INPUT_LEN
    assert counts["OutageFaultPenalty"]["tau_curve_count"] == 1
    for effect in ("JoinLift", "FaultPenalty", "OutageFaultPenalty"):
        assert counts[effect]["unique_origin_count"] == 2
        assert counts[effect]["paired_valid_target_element_count_sum"] == \
            counts[effect]["source_row_count"] * 10


def test_endpoint_counts_cross_check_against_the_curves():
    """Two independent paths to the same figure must agree."""
    rows = [r for r in _counts_fixture() if r["cohort_id"] == 1]
    curves = plot._curves_for(rows, None)
    cells = paired_effect_cells(curves, INPUT_LEN)
    counts = plot.endpoint_sample_counts(rows, curves, INPUT_LEN)
    for effect, group in cells.items():
        assert sum(int(c["cell_count"]) for c in group) == \
            counts[effect]["source_row_count"]
        assert sum(int(c["target_count"]) for c in group) == \
            counts[effect]["paired_valid_target_element_count_sum"]


def test_a_curve_cell_count_mismatch_fails_closed(monkeypatch):
    rows = [r for r in _counts_fixture() if r["cohort_id"] == 1]
    curves = plot._curves_for(rows, None)
    for curve in curves:
        curve["cell_count"] = int(curve["cell_count"]) + 1
    with pytest.raises(plot.NonFiniteValue, match="cell_count sum"):
        plot.endpoint_sample_counts(rows, curves, INPUT_LEN)


def test_an_endpoint_with_no_source_rows_fails_closed():
    """Join-only rows leave Fault and Outage empty.

    The frozen cell selector raises first, which is the correct place -- the
    point is that it fails closed rather than reporting counts for an endpoint
    that has no data, so either guard is acceptable.
    """
    rows = [r for r in _counts_fixture()
            if r["cohort_id"] == 1 and r["track"] == "join"]
    curves = plot._curves_for(rows, None)
    with pytest.raises((plot.NonFiniteValue, ValueError)):
        plot.endpoint_sample_counts(rows, curves, INPUT_LEN)


def test_every_record_carries_sample_counts():
    records = _bootstrap(_varied(0.5, 1.0), _varied(1.0, 1.0))
    for record in records:
        counts = record["sample_counts"]
        for field in ("source_row_count", "unique_origin_count",
                      "unique_cohort_count", "unique_origin_cohort_episode_count",
                      "tau_curve_count", "unique_tau_count",
                      "paired_valid_target_element_count_sum"):
            assert isinstance(counts[field], int) and counts[field] >= 0, field
        assert "n_evaluable_cells" not in counts


def test_mae_and_mape_report_identical_counts():
    records = _bootstrap(_varied(0.5, 1.0), _varied(1.0, 1.0))
    for effect in ("JoinLift", "FaultPenalty", "OutageFaultPenalty"):
        mae = _find(records, effect, "MAE")["sample_counts"]
        mape = _find(records, effect, "MAPE")["sample_counts"]
        assert mae == mape


def test_main_writes_counts_and_semantics(tmp_path, monkeypatch):
    payload = _single_seed_main(tmp_path, monkeypatch)
    assert payload["schema_version"] == 7
    assert "sample_counts" in payload
    assert "count_semantics" in payload
    assert "not statistical sample sizes" in payload["count_semantics"]
    assert "overlap-connected origin block" in payload["count_semantics"]
    assert "excludes training randomness" in payload["count_semantics"]

    def _all_ints(node):
        if isinstance(node, dict):
            for value in node.values():
                _all_ints(value)
        elif isinstance(node, list):
            for value in node:
                _all_ints(value)
        elif isinstance(node, bool):
            raise AssertionError("counts must not be booleans")
        elif isinstance(node, (int, float)) and not isinstance(node, str):
            assert node >= 0

    _all_ints(payload["sample_counts"])
    for record in payload["ples_records"]:
        _all_ints(record["sample_counts"])
