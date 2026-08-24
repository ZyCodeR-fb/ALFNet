import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from data_provider.v4_metrics import PHASE_ORDER, tau_balanced_metrics
from experiments.v4_validation_collector import (
    CollectorRefused, ValidationCellCollector, aggregate_seeds, cell_count,
)

SCALER = {"mean": 100.0, "std": 10.0}
REGIME = "Weekday/EarlyAM"


def _context(phase="validation"):
    return SimpleNamespace(phase=phase)


def _batch(offset=0.0):
    entries = [
        ("zero_history", -1, 0), ("steady", -1, 0),
        ("join", 0, 1), ("join", 1, 6), ("join", 2, 12),
        ("fault", 3, 1), ("fault", 4, 6), ("fault", 5, 12),
    ]
    size = len(entries)
    target = torch.ones(size, 2, 2, 1)
    prediction = target + 0.5 + offset
    return {
        "target": target,
        "loss_mask": torch.ones(size, 2, 2, dtype=torch.bool),
        "scenario": [entry[0] for entry in entries],
        "phase_id": torch.tensor([entry[1] for entry in entries]),
        "evidence_steps": torch.tensor([entry[2] for entry in entries]),
        "origin": torch.arange(10, 10 + size),
        "calendar_regime": [REGIME] * size,
    }, prediction


def _table(offset=0.0):
    batch, prediction = _batch(offset)
    collector = ValidationCellCollector(_context(), SCALER)
    collector.update(batch, prediction)
    return collector, collector.table()


def test_table_has_exactly_33_cells_and_no_worst_group():
    _, table = _table()
    assert tuple(table["main"]) == ("Unseen", "Steady", "Join", "Fault", "LifeAvg")
    assert tuple(table["phase"]) == PHASE_ORDER
    assert "WorstGroup" not in json.dumps(table)
    assert cell_count(table) == 33
    json.dumps(table, allow_nan=False)


def test_group_arithmetic_and_lifeavg():
    collector, table = _table()
    expected = tau_balanced_metrics(collector.rows["main"]["Unseen"], "treated")
    assert table["main"]["Unseen"]["MAE"] == pytest.approx(expected["MAE"])
    for metric in ("MAE", "RMSE", "MAPE"):
        four = [table["main"][name][metric]
                for name in ("Unseen", "Steady", "Join", "Fault")]
        assert table["main"]["LifeAvg"][metric] == pytest.approx(np.mean(four))


def test_missing_phase_and_non_validation_fail_closed():
    batch, prediction = _batch()
    collector = ValidationCellCollector(_context(), SCALER)
    collector.update({key: (value[:-1] if hasattr(value, "__getitem__") else value)
                      for key, value in batch.items()}, prediction[:-1])
    with pytest.raises(CollectorRefused, match="Outage"):
        collector.table()
    with pytest.raises(CollectorRefused, match="validation"):
        ValidationCellCollector(_context("test"), SCALER)


def test_three_seed_aggregate_uses_sample_std():
    tables = {seed: _table(offset)[1]
              for seed, offset in zip((32, 42, 52), (0.0, 0.1, 0.2))}
    summary = aggregate_seeds(tables)
    values = [tables[seed]["main"]["Unseen"]["MAE"] for seed in (32, 42, 52)]
    cell = summary["main"]["Unseen"]["MAE"]
    assert cell["mean"] == pytest.approx(np.mean(values))
    assert cell["std"] == pytest.approx(np.std(values, ddof=1))
    assert summary["std_ddof"] == 1
