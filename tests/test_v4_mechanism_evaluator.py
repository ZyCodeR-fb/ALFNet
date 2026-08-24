from types import SimpleNamespace

import pytest
import torch

from experiments.v4_mechanism_evaluator import (
    MechanismRefused, V4MechanismCollector, compare_model_reports,
)


def _pair():
    scenarios = ["join", "join", "fault", "fault"]
    treated = {
        "target": torch.ones(4, 1, 1, 1),
        "loss_mask": torch.ones(4, 1, 1, dtype=torch.bool),
        "scenario": scenarios,
        "origin": torch.tensor([10, 11, 12, 13]),
        "sample_index": torch.tensor([0, 1, 2, 3]),
        "evidence_steps": torch.tensor([1, 2, 1, 2]),
        "calendar_regime": ["Weekday/EarlyAM"] * 4,
    }
    control = {key: (value.clone() if torch.is_tensor(value) else list(value))
               for key, value in treated.items()}
    control["scenario"] = ["zero_history", "zero_history", "steady", "steady"]
    return treated, control


def _report(treated_errors, control_errors):
    treated, control = _pair()
    treated_prediction = treated["target"] + torch.tensor(
        treated_errors, dtype=torch.float32).reshape(4, 1, 1, 1)
    control_prediction = control["target"] + torch.tensor(
        control_errors, dtype=torch.float32).reshape(4, 1, 1, 1)
    collector = V4MechanismCollector(
        SimpleNamespace(phase="validation"),
        {"mean": 0.0, "std": 1.0}, input_len=2)
    collector.update(treated, treated_prediction, control, control_prediction)
    return collector.report()


def test_k_study_and_ples_keep_both_absolute_arms():
    report = _report([1, 2, 3, 4], [4, 5, 1, 2])
    assert set(report["k_study"]["join"]) == {"1", "2"}
    join = report["ples"]["JoinLift"]
    assert join["treated_absolute_error"]["MAE"] == pytest.approx(1.5)
    assert join["control_absolute_error"]["MAE"] == pytest.approx(4.5)
    assert join["effect"]["MAE"] == pytest.approx(3.0)
    outage = report["ples"]["OutageFaultPenalty"]
    assert outage["effect"]["MAE"] == pytest.approx(2.0)


def test_model_comparison_checks_the_contrast_arm_identity():
    left = _report([1, 2, 3, 4], [4, 5, 1, 2])
    right = _report([2, 3, 4, 5], [5, 6, 2, 3])
    comparison = compare_model_reports(left, right)
    for effect in comparison["records"].values():
        for metric in effect.values():
            assert metric["algebra_residual"] == pytest.approx(0.0, abs=1e-9)


def test_mismatched_pair_identity_fails_closed():
    treated, control = _pair()
    control["origin"][0] += 1
    collector = V4MechanismCollector(
        SimpleNamespace(phase="validation"),
        {"mean": 0.0, "std": 1.0}, input_len=2)
    with pytest.raises(MechanismRefused, match="origin"):
        collector.update(treated, treated["target"], control, control["target"])
