from types import SimpleNamespace

import pytest
import torch

from experiments.v4_low_flow_evaluator import (
    LowFlowCollector,
    LowFlowRefused,
    compare_reports,
)


def _batch():
    target = torch.tensor([[[[2.0], [7.0], [15.0], [30.0], [80.0]],
                            [[4.0], [9.0], [18.0], [40.0], [100.0]]]])
    return {
        "target": target,
        "loss_mask": torch.ones(1, 2, 5, dtype=torch.bool),
        "scenario": ["zero_history"],
        "origin": torch.tensor([12]),
    }


def _report(offset=1.0):
    context = SimpleNamespace(phase="validation", target="q")
    collector = LowFlowCollector(context, {"mean": 0.0, "std": 1.0}, pred_len=2)
    batch = _batch()
    collector.update(batch, batch["target"] + offset)
    return collector.report()


def test_every_target_enters_one_frozen_bin_and_wmape_is_diagnostic():
    report = _report(1.0)
    assert report["overall"]["count"] == 10
    assert sum(cell["count"] for cell in report["bins"].values()) == 10
    assert sum(cell["share"] for cell in report["bins"].values()) == pytest.approx(1.0)
    assert report["overall"]["MAE"] == pytest.approx(1.0)
    assert report["overall"]["signed_bias"] == pytest.approx(1.0)
    assert report["overall"]["overprediction_rate"] == pytest.approx(1.0)
    assert report["overall"]["WMAPE"] == pytest.approx(10 / 305 * 100)
    assert report["bins"]["flow_0_5"]["target_min"] == pytest.approx(2.0)
    assert report["bins"]["flow_0_5"]["target_max"] == pytest.approx(4.0)
    assert set(report["horizons"]) == {"1", "2"}


def test_non_unseen_samples_do_not_enter_the_diagnostic():
    context = SimpleNamespace(phase="validation", target="q")
    collector = LowFlowCollector(context, {"mean": 0.0, "std": 1.0}, pred_len=2)
    batch = _batch()
    batch["scenario"] = ["steady"]
    collector.update(batch, batch["target"])
    with pytest.raises(LowFlowRefused, match="no Unseen"):
        collector.report()


def test_speed_and_training_contexts_are_refused():
    with pytest.raises(LowFlowRefused, match="validation/test"):
        LowFlowCollector(SimpleNamespace(phase="train", target="q"),
                         {"mean": 0.0, "std": 1.0}, 2)
    with pytest.raises(LowFlowRefused, match="flow target"):
        LowFlowCollector(SimpleNamespace(phase="validation", target="v"),
                         {"mean": 0.0, "std": 1.0}, 2)


def test_model_comparison_checks_counts_and_reports_left_minus_right():
    left, right = _report(1.0), _report(2.0)
    compared = compare_reports(left, right)
    assert compared["bins"]["flow_0_5"]["delta_left_minus_right"]["MAE"] == \
        pytest.approx(-1.0)
    right["sample_count"] += 1
    with pytest.raises(LowFlowRefused, match="sample_count"):
        compare_reports(left, right)
