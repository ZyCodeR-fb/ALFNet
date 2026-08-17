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


# --- schema: the header is refused before any row is parsed ------------------


def _write_raw(directory, header, lines):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "ples_paired_rows.csv"
    path.write_text("\n".join([header, *lines]) + "\n", encoding="utf-8")
    return directory


def test_a_duplicate_column_is_refused(tmp_path):
    """csv.DictReader keeps only the last, so the file is ambiguous.

    The concrete hazard: a first NaN column and a second finite column of the
    same name would validate cleanly on a value the file does not unambiguously
    contain.
    """
    from scripts.paired_model_bootstrap import InvalidSchema

    header = HEADER + ",treated_ae_sum"
    row = ",".join(str(v) for v in _row().values()) + ",1.0"
    with pytest.raises(InvalidSchema, match="duplicate column"):
        _read_rows(_write_raw(tmp_path / "dup", header, [row]))


def test_a_missing_required_column_is_refused(tmp_path):
    from scripts.paired_model_bootstrap import InvalidSchema

    fields = HEADER.split(",")
    fields.remove("regime")
    values = _row()
    del values["regime"]
    with pytest.raises(InvalidSchema, match="missing required column"):
        _read_rows(_write_raw(tmp_path / "miss", ",".join(fields),
                              [",".join(str(v) for v in values.values())]))


def test_an_unexpected_column_is_refused(tmp_path):
    """A new field is a protocol change, not something to drop silently."""
    from scripts.paired_model_bootstrap import InvalidSchema

    header = HEADER + ",surprise"
    row = ",".join(str(v) for v in _row().values()) + ",7"
    with pytest.raises(InvalidSchema, match="unexpected column"):
        _read_rows(_write_raw(tmp_path / "extra", header, [row]))


def test_a_non_numeric_value_names_the_file_line_and_field(tmp_path):
    with pytest.raises(InvalidRow, match="not a valid float"):
        _read_rows(_write(tmp_path / "oops", [_row(treated_ae_sum="oops")]))


@pytest.mark.parametrize("field", ["model", "calendar_day", "event_timestamp",
                                   "cohort_style", "dataset", "regime"])
def test_an_empty_critical_string_is_refused(tmp_path, field):
    with pytest.raises(InvalidRow, match=field):
        _read_rows(_write(tmp_path / f"s-{field}", [_row(**{field: ""})]))


def test_a_phase_that_does_not_belong_to_its_track_is_refused(tmp_path):
    """`Outage` is a fault phase; on a join row it is a protocol violation."""
    with pytest.raises(InvalidRow, match="not a join phase"):
        _read_rows(_write(tmp_path / "ph", [_row(track="join", phase="Outage")]))


def test_an_unknown_track_or_phase_is_refused(tmp_path):
    with pytest.raises(InvalidRow, match="track"):
        _read_rows(_write(tmp_path / "tk", [_row(track="sideways")]))
    with pytest.raises(InvalidRow, match="phase"):
        _read_rows(_write(tmp_path / "pz", [_row(phase="Lukewarm")]))


def test_there_is_no_production_validation_bypass():
    """`validate=False` was reachable from production; it is gone."""
    import inspect

    from scripts.paired_model_bootstrap import read_rows

    assert "validate" not in inspect.signature(read_rows).parameters


# --- endpoint inventory comes from the protocol, not from the run ------------


def test_the_inventory_is_114_endpoints_and_126_with_arms():
    from scripts.multiseed_paired_inference import expected_endpoint_keys

    assert len(expected_endpoint_keys(12, include_ples_arms=False)) == 114
    assert len(expected_endpoint_keys(12, include_ples_arms=True)) == 126


def test_a_missing_endpoint_is_caught_even_when_every_source_lacks_it():
    """The check used to compare the run against itself, so this passed."""
    from scripts.multiseed_paired_inference import (
        NonFiniteValue,
        assert_endpoint_inventory,
        expected_endpoint_keys,
    )

    full = expected_endpoint_keys(12)
    assert_endpoint_inventory(full, 12, "complete")

    missing = set(full)
    missing.discard(("main", "Unseen", "MAE"))
    with pytest.raises(NonFiniteValue, match="inventory mismatch"):
        assert_endpoint_inventory(missing, 12, "missing one")

    extra = set(full) | {("main", "Invented", "MAE")}
    with pytest.raises(NonFiniteValue, match="inventory mismatch"):
        assert_endpoint_inventory(extra, 12, "extra one")


# --- the legacy single-seed CLI refuses to produce new active reports --------


def test_the_legacy_paired_bootstrap_cli_is_deprecated():
    result = subprocess.run(
        [sys.executable, "scripts/paired_model_bootstrap.py",
         "--model_a", "x", "--model_b", "y"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "DEPRECATED" in result.stderr, result.stderr


def test_the_legacy_cli_also_requires_enough_draws():
    result = subprocess.run(
        [sys.executable, "scripts/paired_model_bootstrap.py",
         "--model_a", "x", "--model_b", "y",
         "--i-understand-this-is-deprecated-legacy-output",
         "--bootstrap_samples", "10"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "at least 1000" in result.stderr, result.stderr
