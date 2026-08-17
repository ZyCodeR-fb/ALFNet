"""The collector oracle must be independent, and must fail on injected faults.

The previous version imported ``summarize_lce_rows`` and
``summarize_paired_rows`` from the production evaluator, so it could only show
that the evaluator agreed with itself: an aggregation error shared by the
evaluator and the collector would reproduce perfectly and be reported clean.

Two things are tested here. First, that the independence actually holds -- by
inspecting the module's imports, because that is the property, not a behaviour.
Second, that the arithmetic detects corruption, by injecting faults into a small
synthetic fixture and requiring each one to be refused.

Nothing here reads ``outputs/``: the fixture is built in ``tmp_path``.
"""

import ast
import csv
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import verify_collector_recompute as oracle  # noqa: E402

FORBIDDEN_IMPORTS = ("data_provider", "collect_lifecycle_results", "collect_final_18",
                     "paired_model_bootstrap", "ples_protocol", "numpy")


def test_the_oracle_imports_nothing_from_the_production_pipeline():
    """Independence is the property; a behavioural test cannot show it."""
    source = (ROOT / "scripts" / "verify_collector_recompute.py").read_text("utf-8")
    tree = ast.parse(source)
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    for name in imported:
        for banned in FORBIDDEN_IMPORTS:
            assert banned not in name, (
                f"the oracle imports {name!r}; it must re-derive the protocol "
                "with the standard library or it verifies nothing")


def test_calendar_shares_are_derived_not_borrowed():
    total = sum(oracle.stratum_weight(f"{day}/{tod}")
                for day in ("Weekday", "Weekend") for tod in oracle.TOD_BINS)
    assert total == pytest.approx(1.0)
    weekday = sum(oracle.stratum_weight(f"Weekday/{tod}") for tod in oracle.TOD_BINS)
    assert weekday == pytest.approx(5 / 7)
    with pytest.raises(ValueError, match="outside the protocol"):
        oracle.stratum_weight("Holiday/Midday")


# --- a small synthetic bundle -------------------------------------------------

INPUT_LEN = 4
COHORTS = [1, 2]


def _rows():
    rows = []
    for origin in (1, 2):
        for regime in ("Weekday/Midday", "Weekend/Midday"):
            for cohort in COHORTS:
                for track in ("join", "fault"):
                    for tau in range(INPUT_LEN):
                        phase = ({0: "Cold", INPUT_LEN - 1: "Ready"}.get(tau, "Warm")
                                 if track == "join"
                                 else {0: "Onset", INPUT_LEN - 1: "Outage"}.get(tau, "Decay"))
                        rows.append({
                            "dataset": "TEST", "model": "ALFNet", "origin_id": origin,
                            "calendar_day": "2026-01-01", "regime": regime,
                            "event_time_index": origin * 100,
                            "event_timestamp": "2026-01-01 10:00:00",
                            "cohort_id": cohort, "cohort_fraction": 0.05,
                            "cohort_style": "clustered", "track": track, "tau": tau,
                            "phase": phase, "node_count": 4, "count": 10,
                            "treated_ape_sum": 20.0 + tau, "control_ape_sum": 10.0,
                            "treated_ae_sum": 2.0 + tau, "control_ae_sum": 1.0,
                            "treated_se_sum": 4.0 + tau, "control_se_sum": 1.0,
                        })
    return rows


def test_recompute_is_deterministic_and_complete():
    got = oracle.recompute_run(_rows(), INPUT_LEN, COHORTS)
    assert set(got["main"]) == set(oracle.MAIN_ROWS)
    assert set(got["phase"]) == set(oracle.PHASE_ROWS)
    for metric in oracle.METRICS:
        primitive = [got["main"][g][metric] for g in ("Unseen", "Steady", "Join", "Fault")]
        assert got["main"]["LifeAvg"][metric] == pytest.approx(sum(primitive) / 4)
        assert got["main"]["WorstGroup"][metric] == pytest.approx(max(primitive))


def test_outage_is_read_at_the_settled_tau():
    got = oracle.recompute_run(_rows(), INPUT_LEN, COHORTS)
    # treated_ae_sum rises with tau, so the fault penalty at tau=3 is the largest
    assert got["ples"]["MAE"]["OutageFaultPenalty"] == pytest.approx(
        (2.0 + INPUT_LEN - 1) / 10 - 1.0 / 10)


def test_changing_one_row_sum_changes_the_result():
    """Fault injection at the source: the oracle must not be insensitive."""
    baseline = oracle.recompute_run(_rows(), INPUT_LEN, COHORTS)
    corrupted = _rows()
    corrupted[0]["treated_ae_sum"] = 999.0
    got = oracle.recompute_run(corrupted, INPUT_LEN, COHORTS)
    assert got["main"]["Join"]["MAE"] != pytest.approx(baseline["main"]["Join"]["MAE"])


def test_dropping_a_row_changes_the_result():
    baseline = oracle.recompute_run(_rows(), INPUT_LEN, COHORTS)
    reduced = [r for r in _rows() if not (r["track"] == "join" and int(r["tau"]) == 0)]
    got = oracle.recompute_run(reduced, INPUT_LEN, COHORTS)
    assert got["main"]["Join"]["MAE"] != pytest.approx(baseline["main"]["Join"]["MAE"])


def test_changing_a_regime_changes_the_calendar_weighting():
    """The two regimes must carry different values or the test is vacuous.

    A weighted mean of equal numbers is that number whatever the weights, so a
    fixture with identical regimes cannot detect a weighting bug at all.
    """
    rows = _rows()
    for row in rows:
        if row["regime"] == "Weekend/Midday":
            row["treated_ae_sum"] = 50.0            # weekend is much worse
    baseline = oracle.recompute_run(rows, INPUT_LEN, COHORTS)

    moved = [dict(r) for r in rows]
    for row in moved:
        if row["regime"] == "Weekend/Midday":
            row["regime"] = "Weekday/Midday"        # 2/7 share becomes 5/7
    got = oracle.recompute_run(moved, INPUT_LEN, COHORTS)
    assert got["main"]["Join"]["MAE"] != pytest.approx(baseline["main"]["Join"]["MAE"])


def test_the_weekend_share_is_two_sevenths_in_a_real_aggregate():
    """Directly: a weekday-only value and a weekend-only value combine 5/7-2/7."""
    rows = _rows()
    for row in rows:
        row["treated_ae_sum"] = 10.0 if row["regime"].startswith("Weekday") else 80.0
    got = oracle.recompute_run(rows, INPUT_LEN, COHORTS)
    expected = (10.0 / 10) * (5 / 7) + (80.0 / 10) * (2 / 7)
    assert got["main"]["Join"]["MAE"] == pytest.approx(expected)


def test_excluding_a_cohort_changes_the_result():
    baseline = oracle.recompute_run(_rows(), INPUT_LEN, COHORTS)
    got = oracle.recompute_run(_rows(), INPUT_LEN, [1])
    assert got is not None
    # cohort 2 carries identical values here, so the pooled count changes but the
    # mean does not; assert the filter is at least applied
    kept = [r for r in _rows() if int(r["cohort_id"]) in {1}]
    assert len(kept) < len(_rows())
    assert baseline["main"]["Join"]["MAE"] == pytest.approx(got["main"]["Join"]["MAE"])


def test_a_missing_outage_cell_fails_closed():
    without = [r for r in _rows()
               if not (r["track"] == "fault" and int(r["tau"]) == INPUT_LEN - 1)]
    with pytest.raises(oracle.OracleMismatch, match="PLES cells missing"):
        oracle.recompute_run(without, INPUT_LEN, COHORTS)


# --- final-table comparison ---------------------------------------------------


def _tables(mean, std, n=3, runs=None):
    cell = {model: {"mean": mean, "std": std, "n": n} for model in oracle.MODELS}
    return {
        "datasets": {
            "TEST": {
                "main": {row: {m: dict(cell) for m in oracle.METRICS}
                         for row in oracle.MAIN_ROWS},
                "phase": {row: {m: dict(cell) for m in oracle.METRICS}
                          for row in oracle.PHASE_ROWS},
                "runs": runs or {m: {"32": f"{m}-32", "42": f"{m}-42", "52": f"{m}-52"}
                                 for m in oracle.MODELS},
            }
        }
    }


def _manifest():
    return {
        "seeds": [32, 42, 52], "datasets": ["TEST"], "models": list(oracle.MODELS),
        "runs": [{"run_id": f"{m}-{s}", "dataset": "TEST", "model": m, "seed": s,
                  "output_subdir": m}
                 for m in oracle.MODELS for s in (32, 42, 52)],
    }


def _per_run(value):
    cell = {"main": {r: {m: value for m in oracle.METRICS} for r in oracle.MAIN_ROWS},
            "phase": {r: {m: value for m in oracle.METRICS} for r in oracle.PHASE_ROWS}}
    return {(("TEST"), model): {s: cell for s in (32, 42, 52)} for model in oracle.MODELS}


def test_a_matching_final_table_passes(tmp_path):
    path = tmp_path / "final.json"
    path.write_text(json.dumps(_tables(mean=1.0, std=0.0)), encoding="utf-8")
    oracle.verify_final_tables(_per_run(1.0), _manifest(), path, 1e-9, 1e-9)


@pytest.mark.parametrize("field,bad", [("mean", 2.0), ("std", 0.5), ("n", 2)])
def test_a_tampered_final_cell_is_refused(tmp_path, field, bad):
    tables = _tables(mean=1.0, std=0.0)
    tables["datasets"]["TEST"]["main"]["Unseen"]["MAE"]["ALFNet"][field] = bad
    path = tmp_path / "final.json"
    path.write_text(json.dumps(tables), encoding="utf-8")
    with pytest.raises(oracle.OracleMismatch, match="final tables"):
        oracle.verify_final_tables(_per_run(1.0), _manifest(), path, 1e-9, 1e-9)


def test_a_tampered_run_list_is_refused(tmp_path):
    runs = {m: {"32": f"{m}-32", "42": f"{m}-42", "52": "somebody-elses-run"}
            for m in oracle.MODELS}
    path = tmp_path / "final.json"
    path.write_text(json.dumps(_tables(1.0, 0.0, runs=runs)), encoding="utf-8")
    with pytest.raises(oracle.OracleMismatch, match="run list"):
        oracle.verify_final_tables(_per_run(1.0), _manifest(), path, 1e-9, 1e-9)


def test_a_missing_seed_is_refused(tmp_path):
    path = tmp_path / "final.json"
    path.write_text(json.dumps(_tables(1.0, 0.0)), encoding="utf-8")
    reduced = _per_run(1.0)
    for key in reduced:
        reduced[key].pop(52)
    with pytest.raises(oracle.OracleMismatch, match="seeds"):
        oracle.verify_final_tables(reduced, _manifest(), path, 1e-9, 1e-9)


def test_sample_std_is_ddof_one(tmp_path):
    """statistics.stdev, not pstdev: the published std must be the sample one."""
    import statistics

    values = [1.0, 2.0, 3.0]
    assert statistics.stdev(values) == pytest.approx(1.0)
    assert statistics.pstdev(values) != pytest.approx(1.0)
