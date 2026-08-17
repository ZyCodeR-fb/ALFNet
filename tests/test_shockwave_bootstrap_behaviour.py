"""End-to-end behaviour of the cross-model Shockwave bootstrap.

These tests deliberately do not inspect source text. The previous round's test
asserted that ``overlap_blocks`` appeared in the function body; it passed while
the feature was inert, because the span it needed was never written into
``ples_effects_summary.json`` and the code silently fell back to treating every
origin as an independent draw. Each test here observes the sampling behaviour or
the produced files instead.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from scripts import plot_counterfactual_shockwave as plot


SUM_FIELDS = ("treated_ape_sum", "control_ape_sum", "treated_ae_sum",
              "control_ae_sum", "treated_se_sum", "control_se_sum")


def _rows(gap_by_origin, origin_times, cohorts=(1, 2), input_len=4):
    """Paired rows whose treated-vs-control gap varies per origin.

    ``gap_by_origin`` is the whole point: the advantage between the two models
    must differ across origins, otherwise every bootstrap draw returns the same
    number and any interval-width assertion is vacuous. The chained origins are
    given one gap and the isolated ones another, so an estimator that draws
    origins independently and one that keeps a chain together produce visibly
    different spreads.
    """
    rows = []
    for origin, (time_index, day) in sorted(origin_times.items()):
        gap = gap_by_origin[origin]
        for cohort in cohorts:
            for track in ("join", "fault"):
                for tau in range(input_len):
                    row = {
                        "origin_id": origin, "cohort_id": cohort, "track": track,
                        "tau": tau, "phase": plot_phase(track, tau, input_len),
                        "count": 10, "calendar_day": day,
                        "event_time_index": time_index, "regime": "Weekday/Midday",
                    }
                    for field in SUM_FIELDS:
                        treated = field.startswith("treated")
                        row[field] = (1.0 + gap) if treated else 1.0
                    rows.append(row)
    return rows


def plot_phase(track, tau, input_len):
    join = ("Cold", "Warm", "Ready")
    fault = ("Onset", "Decay", "Outage")
    names = join if track == "join" else fault
    if tau == 0:
        return names[0]
    return names[2] if tau == input_len - 1 else names[1]


# Origins 1-3 are chained within one span; 4 and 5 are far away.
# Origins 1-3 are chained inside one span; 4 and 5 are isolated. The chain
# carries a large advantage and the isolated origins none, so block resampling
# has three effective units where per-origin resampling has five.
CHAINED = {1: (0, "2026-01-01"), 2: (40, "2026-01-01"), 3: (80, "2026-01-02"),
           4: (900, "2026-01-05"), 5: (1800, "2026-01-09")}
OURS_GAP = {1: 4.0, 2: 4.0, 3: 4.0, 4: 0.0, 5: 0.0}
BASE_GAP = {origin: 0.0 for origin in CHAINED}


def test_overlap_blocking_changes_the_interval_it_produces():
    ours = _rows(OURS_GAP, CHAINED)
    baseline = _rows(BASE_GAP, CHAINED)
    blocked = plot.bootstrap_advantages_v2(ours, baseline, 4, samples=1000, seed=7,
                                           trajectory_span=72)
    per_origin = plot.bootstrap_advantages_v2(ours, baseline, 4, samples=1000, seed=7,
                                              trajectory_span=1)
    key = "FaultPenaltyReduction/MAPE"
    assert blocked[key]["advantage"] == pytest.approx(per_origin[key]["advantage"])
    blocked_width = (blocked[key]["advantage_ci"]["high"]
                     - blocked[key]["advantage_ci"]["low"])
    per_origin_width = (per_origin[key]["advantage_ci"]["high"]
                        - per_origin[key]["advantage_ci"]["low"])
    # Guard against a degenerate fixture: if the advantage does not vary across
    # origins both intervals collapse to zero and the comparison below is
    # vacuous, which is how the previous version of this test passed while the
    # feature it claimed to cover was inert.
    assert per_origin_width > 0.1, "fixture produces no bootstrap variance"
    # Chaining correlated origins removes independent draws, so the honest
    # interval is the wider one. If this ever flips, the blocking is not
    # actually reaching the sampler.
    assert blocked_width > per_origin_width * 1.05, (
        f"overlap blocking had no effect: {blocked_width:.4f} vs {per_origin_width:.4f}"
    )


def test_missing_span_raises_instead_of_resampling_origins_independently():
    ours = _rows(OURS_GAP, CHAINED)
    baseline = _rows(BASE_GAP, CHAINED)
    with pytest.raises(ValueError, match="trajectory span"):
        plot.bootstrap_advantages_v2(ours, baseline, 4, samples=1000, seed=7,
                                     trajectory_span=None)


def test_span_is_recovered_from_the_summary_then_the_manifest(tmp_path):
    summary_path = tmp_path / "ples_effects_summary.json"
    summary_path.write_text("{}", encoding="utf-8")

    # 1. declared in the effects summary itself
    assert plot._trajectory_span({"bootstrap": {"trajectory_span": 72}}, summary_path) == 72

    # 2. absent there, present in the manifest written beside it
    (tmp_path / "protocol_manifest.json").write_text(
        json.dumps({"ples": {"bootstrap": {"trajectory_span": 84}}}), encoding="utf-8")
    assert plot._trajectory_span({}, summary_path) == 84

    # 3. reconstructable from the requested steps plus resolved lengths
    (tmp_path / "protocol_manifest.json").write_text(
        json.dumps({"ples": {"bootstrap": {}},
                    "requested": {"pre_steps": 12, "post_steps": 36},
                    "resolved": {"input_len": 12, "pred_len": 12}}), encoding="utf-8")
    assert plot._trajectory_span({}, summary_path) == 72

    # 3b. pred_len missing -> refuse rather than guess it equals input_len
    (tmp_path / "protocol_manifest.json").write_text(
        json.dumps({"ples": {"bootstrap": {}},
                    "requested": {"pre_steps": 12, "post_steps": 36},
                    "resolved": {"input_len": 12}}), encoding="utf-8")
    with pytest.raises(ValueError, match="Refusing to fall back"):
        plot._trajectory_span({}, summary_path)

    # 4. nowhere at all -> refuse, do not degrade
    (tmp_path / "protocol_manifest.json").unlink()
    with pytest.raises(ValueError, match="Refusing to fall back"):
        plot._trajectory_span({}, summary_path)


def test_evaluation_writes_the_span_into_the_effects_summary():
    """The regression that made the previous fix inert on real data.

    Calls the payload builder rather than grepping the source, so renaming a
    local variable cannot make this pass or fail for the wrong reason.
    """
    from experiments.ples_evaluation import _bootstrap_contract

    manifest = {"ples": {"bootstrap": {
        "design": "origins and cohorts are crossed, not nested",
        "outer_unit": "overlap-connected origin block, stratified by day_type",
    }}}
    contract = _bootstrap_contract(manifest, 84)
    assert contract["trajectory_span"] == 84
    assert "overlap-connected" in contract["outer_unit"]
    # and the result must be directly consumable by the reader on the other side
    assert plot._trajectory_span({"bootstrap": contract}, Path("unused")) == 84
    # the source manifest must not be mutated in place
    assert "trajectory_span" not in manifest["ples"]["bootstrap"]


# --- span reconstruction must not assume input_len == pred_len ---------------


def _manifest(pre, post, input_len, pred_len, declared=None):
    return {
        "ples": {"bootstrap": {} if declared is None else {"trajectory_span": declared}},
        "requested": {"pre_steps": pre, "post_steps": post},
        "resolved": {"input_len": input_len, "pred_len": pred_len},
    }


def test_reconstructed_span_uses_pred_len_for_the_horizon():
    from data_provider.ples_protocol import (
        trajectory_span,
        trajectory_span_from_manifest,
    )

    # 12->12 hides the bug: both the right and the wrong formula give 72.
    assert trajectory_span_from_manifest(_manifest(12, 36, 12, 12)) == 72
    # 12->24 exposes it: the old "input_len twice" spelling would give 72.
    assert trajectory_span_from_manifest(_manifest(12, 36, 12, 24)) == 84
    assert trajectory_span(12, 12, 36, 24) == 84
    # 24->6 exposes it in the other direction.
    assert trajectory_span_from_manifest(_manifest(12, 36, 24, 6)) == 78
    # a declared span always wins over reconstruction
    assert trajectory_span_from_manifest(_manifest(12, 36, 12, 24, declared=99)) == 99
    # missing pieces yield nothing rather than a wrong number
    assert trajectory_span_from_manifest({"ples": {"bootstrap": {}}}) is None


def test_plot_script_reconstruction_honours_pred_len(tmp_path):
    summary_path = tmp_path / "ples_effects_summary.json"
    summary_path.write_text("{}", encoding="utf-8")
    (tmp_path / "protocol_manifest.json").write_text(
        json.dumps(_manifest(12, 36, 12, 24)), encoding="utf-8")
    assert plot._trajectory_span({}, summary_path) == 84


def test_paired_script_reconstruction_honours_pred_len(tmp_path):
    from scripts.paired_model_bootstrap import _trajectory_span

    directory = tmp_path / "run"
    directory.mkdir()
    (directory / "protocol_manifest.json").write_text(
        json.dumps(_manifest(12, 36, 12, 24)), encoding="utf-8")
    # the input_len argument is only a fallback; pred_len must come from the file
    assert _trajectory_span(directory, 12) == 84


# --- both sides of a cross-model comparison must declare a matching span -----


def _bundle(tmp_path, name, span):
    directory = tmp_path / name
    directory.mkdir()
    path = directory / "ples_effects_summary.json"
    path.write_text(json.dumps({"bootstrap": {"trajectory_span": span}}), encoding="utf-8")
    return {"bootstrap": {"trajectory_span": span}}, path


def test_cross_model_span_mismatch_is_rejected(tmp_path):
    ours, ours_path = _bundle(tmp_path, "ours", 72)
    base, base_path = _bundle(tmp_path, "base", 84)
    with pytest.raises(ValueError, match="spans differ between the two models"):
        plot._paired_trajectory_span(ours, ours_path, base, base_path)


def test_cross_model_span_agreement_is_accepted(tmp_path):
    ours, ours_path = _bundle(tmp_path, "ours", 72)
    base, base_path = _bundle(tmp_path, "base", 72)
    assert plot._paired_trajectory_span(ours, ours_path, base, base_path) == 72


def test_cross_model_span_resolves_the_baseline_too(tmp_path):
    """Resolving only ours would silently accept an undeclared baseline."""
    ours, ours_path = _bundle(tmp_path, "ours", 72)
    base_dir = tmp_path / "base"
    base_dir.mkdir()
    base_path = base_dir / "ples_effects_summary.json"
    base_path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="Refusing to fall back"):
        plot._paired_trajectory_span(ours, ours_path, {}, base_path)
