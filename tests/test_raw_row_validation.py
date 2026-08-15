"""Malformed raw rows must be refused at the CSV door, not absorbed downstream.

The defect this closes was measured, not hypothesised. Injecting a single
``treated_ae_sum = NaN`` into a real METR-LA post-event row produced **114 of
114 finite endpoints**: ``_nanmean`` drops non-finite cells, which is right for
a bootstrap draw that missed a cell and catastrophic for a NaN that was in the
source file. Every later finite check therefore inspected numbers that had
already been silently repaired.

So validation happens where the value enters, and the error carries enough
context to locate the row: file, physical line, dataset/model, and the
origin/cohort/track/tau that identifies the cell.

The CLI guards are exercised through real ``argv`` rather than by asserting on a
constant. A test that only checks ``MIN_BOOTSTRAP_SAMPLES >= 1000`` still passes
after the guard that reads it is deleted.
"""

import csv
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.paired_model_bootstrap import InvalidRow, _read_rows  # noqa: E402

HEADER = ("dataset,model,origin_id,calendar_day,regime,event_time_index,event_timestamp,"
          "cohort_id,cohort_fraction,cohort_style,track,tau,phase,node_count,count,"
          "graph_weight,patch_residual_weight,treated_ape_sum,control_ape_sum,"
          "treated_ae_sum,control_ae_sum,treated_se_sum,control_se_sum")


def _row(**overrides):
    row = {
        "dataset": "METRLA", "model": "ALFNet", "origin_id": 1,
        "calendar_day": "2026-01-01", "regime": "Weekday/Midday",
        "event_time_index": 0, "event_timestamp": "2026-01-01 10:00:00",
        "cohort_id": 1, "cohort_fraction": 0.05, "cohort_style": "clustered",
        "track": "join", "tau": 0, "phase": "Cold", "node_count": 4, "count": 10,
        "graph_weight": 0.5, "patch_residual_weight": 0.5,
        "treated_ape_sum": 1.0, "control_ape_sum": 1.0, "treated_ae_sum": 1.0,
        "control_ae_sum": 1.0, "treated_se_sum": 1.0, "control_se_sum": 1.0,
    }
    row.update(overrides)
    return row


def _write(directory, rows):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "ples_paired_rows.csv"
    fields = HEADER.split(",")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return directory


def test_a_clean_file_is_accepted(tmp_path):
    rows, _ = _read_rows(_write(tmp_path / "ok", [_row(), _row(tau=1, phase="Warm")]))
    assert len(rows) == 2


@pytest.mark.parametrize("field", [
    "treated_ape_sum", "control_ape_sum", "treated_ae_sum",
    "control_ae_sum", "treated_se_sum", "control_se_sum",
])
@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_a_non_finite_sum_is_refused(tmp_path, field, value):
    directory = _write(tmp_path / f"{field}-{value}", [_row(**{field: value})])
    with pytest.raises(InvalidRow, match="not finite"):
        _read_rows(directory)


def test_a_negative_sum_is_refused(tmp_path):
    """These are sums of absolute or squared errors; negative is impossible."""
    with pytest.raises(InvalidRow, match="negative"):
        _read_rows(_write(tmp_path / "neg", [_row(treated_ae_sum=-1.0)]))


@pytest.mark.parametrize("field", ["count", "node_count"])
def test_a_non_positive_count_is_refused(tmp_path, field):
    with pytest.raises(InvalidRow, match="positive integer"):
        _read_rows(_write(tmp_path / f"c-{field}", [_row(**{field: 0})]))


@pytest.mark.parametrize("value", [0.0, -0.1, 1.5, "nan"])
def test_an_out_of_range_cohort_fraction_is_refused(tmp_path, value):
    with pytest.raises(InvalidRow, match="cohort_fraction"):
        _read_rows(_write(tmp_path / f"f{value}", [_row(cohort_fraction=value)]))


def test_a_non_finite_graph_weight_is_refused(tmp_path):
    with pytest.raises(InvalidRow, match="graph_weight"):
        _read_rows(_write(tmp_path / "gw", [_row(graph_weight="nan")]))


def test_an_empty_required_label_is_refused(tmp_path):
    with pytest.raises(InvalidRow, match="regime"):
        _read_rows(_write(tmp_path / "rg", [_row(regime="")]))


def test_the_error_names_the_file_line_and_cell(tmp_path):
    directory = _write(tmp_path / "ctx", [_row(), _row(tau=1, phase="Warm",
                                                       origin_id=7, cohort_id=3,
                                                       treated_se_sum="nan")])
    with pytest.raises(InvalidRow) as caught:
        _read_rows(directory)
    message = str(caught.value)
    assert "ples_paired_rows.csv:3" in message, message   # header is line 1
    assert "origin_id=7" in message
    assert "cohort_id=3" in message
    assert "METRLA" in message and "ALFNet" in message


def test_validation_can_be_disabled_only_explicitly(tmp_path):
    """The escape hatch exists for fixtures, and is never the default."""
    directory = _write(tmp_path / "off", [_row(treated_ae_sum="nan")])
    with pytest.raises(InvalidRow):
        _read_rows(directory)
    rows, _ = _read_rows(directory, validate=False)
    assert len(rows) == 1


# --- CLI guards, through real argv -------------------------------------------


def _run_cli(*args):
    return subprocess.run(
        [sys.executable, "scripts/multiseed_paired_inference.py", *args],
        cwd=ROOT, capture_output=True, text=True,
    )


@pytest.mark.parametrize("samples", ["0", "-1", "999"])
def test_cli_refuses_too_few_bootstrap_samples(samples, tmp_path):
    result = _run_cli("--dataset", "METRLA", "--output-dir", str(tmp_path),
                      "--bootstrap-samples", samples)
    assert result.returncode != 0
    assert "at least 1000" in result.stderr, result.stderr


@pytest.mark.parametrize("margin", ["nan", "inf", "-0.5"])
def test_cli_refuses_an_invalid_margin(margin, tmp_path):
    result = _run_cli("--dataset", "METRLA", "--output-dir", str(tmp_path),
                      "--bootstrap-samples", "1000", "--post-hoc-margin-mae", margin)
    assert result.returncode != 0
    assert "finite and >= 0" in result.stderr, result.stderr


def test_cli_rejects_an_unknown_dataset(tmp_path):
    result = _run_cli("--dataset", "PEMS03", "--output-dir", str(tmp_path))
    assert result.returncode != 0


# --- pure-function guards ----------------------------------------------------


def test_crossed_bootstrap_refuses_non_positive_samples():
    from scripts.multiseed_paired_inference import NonFiniteValue, crossed_bootstrap

    with pytest.raises(NonFiniteValue, match="samples > 0"):
        crossed_bootstrap({}, [32], samples=0, seed=1)


def test_build_records_refuses_a_key_set_mismatch():
    import numpy as np

    from scripts.multiseed_paired_inference import NonFiniteValue, build_records

    key = ("main", "Unseen", "MAE")
    other = ("main", "Steady", "MAE")
    margins = {"MAE": 0.05, "MAPE": 0.25, "RMSE": None}
    with pytest.raises(NonFiniteValue, match="key sets disagree"):
        build_records({key: -1.0}, {other: np.full(10, -1.0)},
                      {32: {key: -1.0}}, [32], margins)


def test_build_records_refuses_empty_draws():
    import numpy as np

    from scripts.multiseed_paired_inference import NonFiniteValue, build_records

    key = ("main", "Unseen", "MAE")
    margins = {"MAE": 0.05, "MAPE": 0.25, "RMSE": None}
    with pytest.raises(NonFiniteValue, match="no bootstrap draws"):
        build_records({key: -1.0}, {key: np.array([])}, {32: {key: -1.0}}, [32], margins)
