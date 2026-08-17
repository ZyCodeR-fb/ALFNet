"""Multiseed PLES: both models, both arms, and a checkable algebra identity.

A PLES contrast is gameable by construction:

    JoinLift           = control(Unseen) - treated(Join)      larger is better
    FaultPenalty       = treated(Fault)  - control(Steady)    smaller is better
    OutageFaultPenalty = the same, read at tau = input_len - 1

so degrading the control arm improves all three without forecasting anything
better. Publishing the contrast without its arms therefore publishes a number a
reader cannot interpret, and publishing it *twice* -- once bare, once with arms
-- lets a consumer pick the bare one. There is now exactly one PLES record and
it always carries both arms and both models.

The important test here is the **end-to-end** one. Earlier tests assembled
point/draw dictionaries by hand, which exercises the classifier but not the
production data flow, so it could not have caught a contrast and its arms being
computed from different draws. The end-to-end test runs real ``RowTensor``s
through ``statistics_and_arms`` -> ``crossed_bootstrap`` -> ``build_ples_records``
and checks the algebra per draw.
"""

import json

import numpy as np
import pytest

from scripts.multiseed_paired_inference import (
    ARM_METRICS,
    EFFECT_DEFINITION,
    MODEL_A,
    MODEL_B,
    PLES_EFFECTS,
    NonFiniteValue,
    _arm_status,
    _interval_status,
    arms_pareto_status,
    build_ples_records,
    build_records,
    combined_status,
    crossed_bootstrap,
    per_seed_values,
    statistics_and_arms,
)
from scripts.paired_model_bootstrap import RowTensor, _pair_keys

SEEDS = [32, 42, 52]
INPUT_LEN = 4
CONTRAST = {"MAE": 0.05, "MAPE": 0.25, "RMSE": None}
ARM = {"MAE": 0.05, "MAPE": 0.25, "RMSE": None}


# --- a minimal but real tensor ------------------------------------------------


def _rows(treated, control, spread=0.0):
    """Two origins x two regimes x two cohorts x both tracks x every tau."""
    rows = []
    for origin in (1, 2):
        for regime in ("Weekday/Midday", "Weekend/Midday"):
            for cohort in (1, 2):
                for track in ("join", "fault"):
                    for tau in range(INPUT_LEN):
                        phase = ({0: "Cold", INPUT_LEN - 1: "Ready"}.get(tau, "Warm")
                                 if track == "join" else
                                 {0: "Onset", INPUT_LEN - 1: "Outage"}.get(tau, "Decay"))
                        bump = spread * tau
                        rows.append({
                            "dataset": "TEST", "model": "M", "origin_id": origin,
                            "calendar_day": f"2026-01-0{origin}", "regime": regime,
                            "event_time_index": origin * 1000,
                            "event_timestamp": "2026-01-01 10:00:00",
                            "cohort_id": cohort, "cohort_fraction": 0.05,
                            "cohort_style": "clustered", "track": track, "tau": tau,
                            "phase": phase, "node_count": 4, "count": 10,
                            "graph_weight": None, "patch_residual_weight": None,
                            "treated_ape_sum": (treated + bump) * 10,
                            "control_ape_sum": control * 10,
                            "treated_ae_sum": (treated + bump) * 10,
                            "control_ae_sum": control * 10,
                            "treated_se_sum": (treated + bump) ** 2 * 10,
                            "control_se_sum": control ** 2 * 10,
                        })
    return rows


def _tensor(rows):
    return RowTensor(rows, _pair_keys(rows), INPUT_LEN, trajectory_span=8)


def _tensors(a_treated, a_control, b_treated, b_control):
    """Same design for every seed, tiny per-seed jitter so seeds are distinct."""
    out = {}
    for index, seed in enumerate(SEEDS):
        jitter = 0.01 * index
        out[seed] = {
            MODEL_A: _tensor(_rows(a_treated + jitter, a_control + jitter)),
            MODEL_B: _tensor(_rows(b_treated + jitter, b_control + jitter)),
        }
    return out


def _run(tensors, samples=60, contrast=CONTRAST, arm=ARM):
    seed_deltas, per_seed_model = per_seed_values(tensors, SEEDS)
    draws, raw = crossed_bootstrap(tensors, SEEDS, samples=samples, seed=11)
    point = {k: float(np.mean([seed_deltas[s][k] for s in SEEDS])) for k in seed_deltas[SEEDS[0]]}
    records = build_ples_records(point, draws, seed_deltas, per_seed_model, raw,
                                 SEEDS, contrast, arm)
    return records, point, draws, raw, seed_deltas, per_seed_model


# --- 1-6: the production data flow -------------------------------------------


def test_end_to_end_produces_every_endpoint_with_both_models_and_arms():
    records, *_ = _run(_tensors(2.0, 1.0, 3.0, 1.0))
    assert len(records) == len(PLES_EFFECTS) * len(ARM_METRICS)
    assert {(r["effect"], r["metric"]) for r in records} == {
        (e, m) for e in PLES_EFFECTS for m in ARM_METRICS}

    for record in records:
        for model in (MODEL_A, MODEL_B):
            for quantity in ("contrast", "treated_absolute_error", "control_absolute_error"):
                entry = record["models"][model][quantity]
                assert set(entry) == {"point", "per_seed", "interval"}
                assert set(entry["per_seed"]) == {str(s) for s in SEEDS}
                assert entry["interval"]["low"] <= entry["interval"]["high"]
        for delta in ("contrast_delta", "treated_delta", "control_delta"):
            assert set(record[delta]) == {"point", "per_seed", "interval"}


def test_the_delta_draw_is_the_difference_of_the_two_raw_draws():
    """Not a separately resampled quantity -- the same replicate, subtracted."""
    _, _, draws, raw, _, _ = _run(_tensors(2.0, 1.0, 3.0, 1.0))
    for effect in PLES_EFFECTS:
        for metric in ARM_METRICS:
            key = ("ples", effect, metric)
            assert np.allclose(draws[key], raw[MODEL_A][key] - raw[MODEL_B][key],
                               atol=0, rtol=0)


def test_all_three_algebra_identities_hold_at_point_per_seed_and_per_draw():
    """Each model's own identity, and the comparison identity.

    The comparison check alone is insufficient: if both models' contrasts were
    broken identically the error cancels in the difference, leaving a comparison
    residual of exactly zero over two wrong effects.
    """
    records, *_ = _run(_tensors(2.0, 1.0, 3.0, 1.5))
    for record in records:
        expected = ("contrast = control - treated"
                    if record["effect"] == "JoinLift" else "contrast = treated - control")
        checks = [record["comparison_algebra_identity"]]
        checks += [record["models"][m]["algebra_identity"] for m in (MODEL_A, MODEL_B)]
        for algebra in checks:
            assert abs(algebra["point_residual"]) <= algebra["tolerance"]
            assert algebra["max_abs_per_seed_residual"] <= algebra["tolerance"]
            assert algebra["max_abs_per_draw_residual"] <= algebra["tolerance"]
            assert algebra["formula"] == expected


def test_outage_uses_the_settled_tau_only():
    """treated rises with tau, so Outage must differ from the Decay mean."""
    tensors = {}
    for index, seed in enumerate(SEEDS):
        tensors[seed] = {
            MODEL_A: _tensor(_rows(2.0 + 0.01 * index, 1.0, spread=1.0)),
            MODEL_B: _tensor(_rows(3.0 + 0.01 * index, 1.0, spread=1.0)),
        }
    records, _, _, _, _, per_seed_model = _run(tensors)
    outage = next(r for r in records
                  if r["effect"] == "OutageFaultPenalty" and r["metric"] == "MAE")
    penalty = next(r for r in records
                   if r["effect"] == "FaultPenalty" and r["metric"] == "MAE")
    assert outage["models"][MODEL_A]["treated_absolute_error"]["point"] != pytest.approx(
        penalty["models"][MODEL_A]["treated_absolute_error"]["point"])


def test_contrast_and_arms_share_one_multiplicity(monkeypatch):
    """One weight vector per replicate reaches every seed and both models."""
    from scripts import multiseed_paired_inference as module

    seen = []
    real = module.statistics_and_arms

    def _spy(tensor, weights):
        seen.append(tuple(np.asarray(weights).tolist()))
        return real(tensor, weights)

    monkeypatch.setattr(module, "statistics_and_arms", _spy)
    tensors = _tensors(2.0, 1.0, 3.0, 1.0)
    seen.clear()
    module.crossed_bootstrap(tensors, SEEDS, samples=1, seed=5)
    assert len(set(seen)) == 1, f"{len(set(seen))} distinct weight vectors in one replicate"


def test_generic_records_contain_no_ples_endpoint():
    """A bare, arms-free PLES verdict must not exist anywhere."""
    tensors = _tensors(2.0, 1.0, 3.0, 1.0)
    seed_deltas, _ = per_seed_values(tensors, SEEDS)
    draws, _ = crossed_bootstrap(tensors, SEEDS, samples=40, seed=3)
    point = {k: float(np.mean([seed_deltas[s][k] for s in SEEDS])) for k in seed_deltas[SEEDS[0]]}
    records = build_records(point, draws, seed_deltas, SEEDS, CONTRAST)
    assert records, "fixture produced no LCE records"
    assert not [r for r in records if r["table"] == "ples"]
    assert not [r for r in records if r["table"] == "arm"]


# --- 7-8: the counterexamples ------------------------------------------------


def test_both_arms_improve_but_contrast_degrades():
    """ALFNet is better on both arms, but the control arm improves more."""
    records, *_ = _run(_tensors(a_treated=2.0, a_control=1.0,
                                b_treated=2.6, b_control=3.0))
    record = next(r for r in records if r["effect"] == "JoinLift" and r["metric"] == "MAE")
    assert record["treated_status"] == "improve"
    assert record["control_status"] == "improve"
    assert record["arms_pareto_status"] == "treated_improves_control_non_degraded"
    assert record["contrast_status"] == "degraded"
    assert record["combined_descriptive_status"] == "contrast_degraded"


def test_control_only_degradation_never_reads_as_favourable():
    """Identical treated arm, worse control arm: JoinLift rises regardless."""
    records, *_ = _run(_tensors(a_treated=2.0, a_control=4.0,
                                b_treated=2.0, b_control=1.0))
    record = next(r for r in records if r["effect"] == "JoinLift" and r["metric"] == "MAE")
    assert record["contrast_status"] == "favourable"
    assert record["control_status"] == "degrade"
    assert record["combined_descriptive_status"] != "all_conditions_met"
    assert record["arms_pareto_status"] == "control_degraded"


# --- 9: non-finite input fails closed ----------------------------------------


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
@pytest.mark.parametrize("target", ["raw_draw", "delta_draw", "point", "per_seed"])
def test_non_finite_values_fail_closed(bad, target):
    records, point, draws, raw, seed_deltas, per_seed_model = _run(
        _tensors(2.0, 1.0, 3.0, 1.0))
    key = ("ples", "JoinLift", "MAE")
    if target == "raw_draw":
        raw[MODEL_A][key] = np.append(raw[MODEL_A][key][:-1], bad)
    elif target == "delta_draw":
        draws[key] = np.append(draws[key][:-1], bad)
    elif target == "point":
        point[key] = bad
    else:
        seed_deltas[42][key] = bad
    with pytest.raises(NonFiniteValue):
        build_ples_records(point, draws, seed_deltas, per_seed_model, raw,
                           SEEDS, CONTRAST, ARM)


def test_breaking_only_one_model_raw_contrast_fails_closed():
    """The case the comparison identity cannot see on its own.

    Corrupt ALFNet's raw contrast *and* the matching MoGERNN raw contrast by the
    same amount: the comparison delta is untouched and its residual stays zero,
    but neither model's reported effect is its own arms' difference any more.
    """
    _, point, draws, raw, seed_deltas, per_seed_model = _run(_tensors(2.0, 1.0, 3.0, 1.0))
    key = ("ples", "JoinLift", "MAE")
    for model in (MODEL_A, MODEL_B):
        raw[model][key] = raw[model][key] + 1.0
        for seed in SEEDS:
            per_seed_model[seed][model][key] += 1.0
    # delta = A - B is preserved, so the schema check and the comparison
    # identity both still pass; only the per-model identity can catch this.
    assert np.allclose(draws[key], raw[MODEL_A][key] - raw[MODEL_B][key])
    with pytest.raises(NonFiniteValue, match="not the signed difference"):
        build_ples_records(point, draws, seed_deltas, per_seed_model, raw,
                           SEEDS, CONTRAST, ARM)


def test_breaking_a_single_model_raw_contrast_fails_closed():
    _, point, draws, raw, seed_deltas, per_seed_model = _run(_tensors(2.0, 1.0, 3.0, 1.0))
    key = ("ples", "JoinLift", "MAE")
    raw[MODEL_A][key] = raw[MODEL_A][key] + 1.0
    for seed in SEEDS:
        per_seed_model[seed][MODEL_A][key] += 1.0
    with pytest.raises(NonFiniteValue):
        build_ples_records(point, draws, seed_deltas, per_seed_model, raw,
                           SEEDS, CONTRAST, ARM)


def test_a_delta_draw_that_is_not_a_minus_b_fails_closed():
    _, point, draws, raw, seed_deltas, per_seed_model = _run(_tensors(2.0, 1.0, 3.0, 1.0))
    key = ("ples", "JoinLift", "MAE")
    draws[key] = draws[key] + 1.0
    with pytest.raises(NonFiniteValue, match="not ALFNet minus MoGERNN"):
        build_ples_records(point, draws, seed_deltas, per_seed_model, raw,
                           SEEDS, CONTRAST, ARM)


# --- key-set and shape faults -------------------------------------------------


def test_an_extra_seed_is_refused():
    _, point, draws, raw, seed_deltas, per_seed_model = _run(_tensors(2.0, 1.0, 3.0, 1.0))
    per_seed_model[99] = per_seed_model[SEEDS[0]]
    with pytest.raises(NonFiniteValue, match="per_seed_model seeds"):
        build_ples_records(point, draws, seed_deltas, per_seed_model, raw,
                           SEEDS, CONTRAST, ARM)


def test_a_missing_model_is_refused():
    _, point, draws, raw, seed_deltas, per_seed_model = _run(_tensors(2.0, 1.0, 3.0, 1.0))
    del raw[MODEL_B]
    with pytest.raises(NonFiniteValue, match="raw_draws models"):
        build_ples_records(point, draws, seed_deltas, per_seed_model, raw,
                           SEEDS, CONTRAST, ARM)


def test_a_missing_ples_endpoint_is_refused():
    _, point, draws, raw, seed_deltas, per_seed_model = _run(_tensors(2.0, 1.0, 3.0, 1.0))
    del raw[MODEL_A][("arm", "JoinLift/control", "MAE")]
    with pytest.raises(NonFiniteValue, match="raw_draws"):
        build_ples_records(point, draws, seed_deltas, per_seed_model, raw,
                           SEEDS, CONTRAST, ARM)


def test_an_extra_ples_endpoint_is_refused():
    _, point, draws, raw, seed_deltas, per_seed_model = _run(_tensors(2.0, 1.0, 3.0, 1.0))
    point[("ples", "Invented", "MAE")] = 0.0
    with pytest.raises(NonFiniteValue, match="point"):
        build_ples_records(point, draws, seed_deltas, per_seed_model, raw,
                           SEEDS, CONTRAST, ARM)


def test_mismatched_draw_lengths_are_refused():
    _, point, draws, raw, seed_deltas, per_seed_model = _run(_tensors(2.0, 1.0, 3.0, 1.0))
    key = ("ples", "JoinLift", "MAE")
    raw[MODEL_A][key] = raw[MODEL_A][key][:-1]
    with pytest.raises(NonFiniteValue, match="lengths differ|not ALFNet minus"):
        build_ples_records(point, draws, seed_deltas, per_seed_model, raw,
                           SEEDS, CONTRAST, ARM)


# --- 10: the two margins move independently ----------------------------------


def test_contrast_and_arm_margins_are_independent():
    tensors = _tensors(a_treated=2.0, a_control=1.0, b_treated=2.02, b_control=1.0)

    wide_contrast = {"MAE": 100.0, "MAPE": 100.0, "RMSE": None}
    records, *_ = _run(tensors, contrast=wide_contrast, arm=ARM)
    record = next(r for r in records if r["effect"] == "JoinLift" and r["metric"] == "MAE")
    assert record["contrast_status"] == "equivalent"

    wide_arm = {"MAE": 100.0, "MAPE": 100.0, "RMSE": None}
    records, *_ = _run(tensors, contrast=CONTRAST, arm=wide_arm)
    record = next(r for r in records if r["effect"] == "JoinLift" and r["metric"] == "MAE")
    assert record["treated_status"] == "equivalent"
    assert record["control_status"] == "equivalent"


# --- status primitives (helper level, retained) ------------------------------


def test_effect_definitions_are_explicit():
    assert EFFECT_DEFINITION["JoinLift"]["orientation"] == "higher_is_better"
    assert EFFECT_DEFINITION["FaultPenalty"]["orientation"] == "lower_is_better"
    assert EFFECT_DEFINITION["OutageFaultPenalty"]["orientation"] == "lower_is_better"


def test_interval_status_respects_orientation():
    assert _interval_status(1.0, 0.5, 1.5, 0.25, True) == "favourable"
    assert _interval_status(-1.0, -1.5, -0.5, 0.25, True) == "degraded"
    assert _interval_status(-1.0, -1.5, -0.5, 0.25, False) == "favourable"
    assert _interval_status(1.0, 0.5, 1.5, 0.25, False) == "degraded"


def test_arm_status_is_always_lower_is_better():
    assert _arm_status(-1.0, -1.5, -0.5, 0.05) == "improve"
    assert _arm_status(1.0, 0.5, 1.5, 0.05) == "degrade"
    assert _arm_status(0.0, -0.01, 0.01, 0.05) == "equivalent"
    assert _arm_status(0.0, -5.0, 5.0, 0.05) == "uncertain"


def test_no_status_uses_confirmatory_wording():
    banned = ("accept", "reject", "pass", "fail", "significant")
    values = set()
    for treated in ("improve", "degrade", "equivalent", "uncertain"):
        for control in ("improve", "degrade", "equivalent", "uncertain"):
            values.add(arms_pareto_status(treated, control))
            for contrast in ("favourable", "degraded", "equivalent", "uncertain"):
                values.add(combined_status(contrast, treated, control))
    for value in values:
        assert not any(word in value.lower() for word in banned), value


# --- main(): console formatting, schema assembly and JSON write ---------------
#
# The gap this closes was a real regression. `seed_tally` was renamed to
# `n_*_seeds`, `main()`'s console formatter kept reading the old unsuffixed
# names, and 695 tests passed because none of them executed `main()`. The real
# CLI would have raised KeyError on the first primary LCE record it printed.


def _main_smoke(tmp_path, monkeypatch, samples=1000):
    """Run the real main() over synthetic tensors, writing into tmp_path.

    `crossed_bootstrap` is stubbed with precomputed draws so the declared
    sample count stays at the production floor without paying for 1000 real
    replicates. Everything after it -- record assembly, console formatting,
    schema assembly and JSON serialisation -- is the production code path.
    """
    import sys

    from scripts import multiseed_paired_inference as module

    tensors = _tensors(2.0, 1.0, 3.0, 1.5)
    provenance = {seed: {"input_len": INPUT_LEN, "row_design_hash": "d" * 16,
                         "row_count": 1, "protocol_hash": "p" * 16,
                         "training_protocol_hash": "t" * 16, "trajectory_span": 8,
                         "main_cohort_ids": [1, 2],
                         MODEL_A: f"alfnet-test-s{seed}",
                         MODEL_B: f"mogernn-test-s{seed}"}
                  for seed in SEEDS}

    monkeypatch.setattr(module, "load_seed_bundles", lambda dataset, root: (SEEDS, {}))
    monkeypatch.setattr(module, "build_tensors", lambda seeds, bundles: (tensors, provenance))

    real_bootstrap = module.crossed_bootstrap
    cached = real_bootstrap(tensors, SEEDS, samples=40, seed=11)
    monkeypatch.setattr(module, "crossed_bootstrap",
                        lambda *a, **k: cached)

    monkeypatch.setattr(sys, "argv", [
        "multiseed_paired_inference.py", "--dataset", "METRLA",
        "--output-dir", str(tmp_path), "--bootstrap-samples", str(samples),
    ])
    module.main()
    return json.loads((tmp_path / "METRLA_multiseed_paired.json").read_text("utf-8"))


def test_main_runs_and_writes_the_current_schema(tmp_path, monkeypatch, capsys):
    payload = _main_smoke(tmp_path, monkeypatch)
    captured = capsys.readouterr().out

    assert payload["schema_version"] == 4
    assert payload["post_hoc_diagnostic_only"] is True
    assert payload["inferential_status"] == "descriptive_not_formal"

    # console shows the full seed tally, not just one number
    assert "ALF/base/tie/eval" in captured

    # generic records are LCE only
    assert payload["records"], "no LCE records written"
    assert not [r for r in payload["records"] if r["table"] in {"ples", "arm"}]

    # every PLES record carries three algebra identities
    assert len(payload["ples_records"]) == len(PLES_EFFECTS) * len(ARM_METRICS)
    for record in payload["ples_records"]:
        assert "comparison_algebra_identity" in record
        for model in (MODEL_A, MODEL_B):
            assert "algebra_identity" in record["models"][model]

    # the two margin families are separate top-level fields
    assert payload["post_hoc_contrast_margins"] != {}
    assert payload["post_hoc_arm_margins"] != {}
    assert "post_hoc_margins" not in payload


def test_main_output_contains_no_legacy_tally_field(tmp_path, monkeypatch):
    payload = _main_smoke(tmp_path, monkeypatch)
    text = json.dumps(payload)
    for legacy in ('"n_favour_alfnet"', '"n_favour_baseline"', '"n_tie"',
                   '"n_evaluable"', '"n_nonfinite"'):
        assert legacy not in text, f"legacy tally field {legacy} still emitted"
    for current in ("n_favour_alfnet_seeds", "n_tie_seeds", "n_evaluable_seeds"):
        assert current in text


def test_main_output_reloads_under_allow_nan_false(tmp_path, monkeypatch):
    _main_smoke(tmp_path, monkeypatch)
    raw = (tmp_path / "METRLA_multiseed_paired.json").read_text("utf-8")
    # json.loads accepts NaN by default; parse_constant makes it an error
    json.loads(raw, parse_constant=_reject_constant)


def _reject_constant(name):
    raise AssertionError(f"non-finite constant {name!r} in written JSON")


def test_main_still_refuses_too_few_samples(tmp_path, monkeypatch):
    """The smoke path must not have weakened the production floor."""
    with pytest.raises(SystemExit):
        _main_smoke(tmp_path, monkeypatch, samples=999)
