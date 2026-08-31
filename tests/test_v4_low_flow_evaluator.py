from types import SimpleNamespace

import pytest
import torch

from experiments.v4_low_flow_evaluator import (
    LowFlowCollector,
    LowFlowRefused,
    compare_reports,
)
from scripts.eval_v4_low_flow import expected_low_flow_identity


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


def test_candidate_low_flow_identity_is_bound_to_its_exact_config():
    class _Context:
        def __init__(self, static, exposure):
            self._static, self._exposure = static, exposure

        def static_asset_hash(self):
            return self._static

        def exposure_hash(self):
            return self._exposure

    run = SimpleNamespace(
        name="PEMS03/ALFNet[LOW]/seed32", dataset="PEMS03", model="ALFNet",
        seed=32, entry={"candidate": "LOW"})
    session = SimpleNamespace(
        args=SimpleNamespace(model_variant="LOW"),
        train_context=_Context("static", "train"),
        validation_context=_Context("static", "validation"),
        seed_contract={"run_seed": 32})
    plan = {
        "kind": "v4_alfnet_validation_candidates",
        "configs": {"BASE": {"sha256": "base-config"},
                    "LOW": {"sha256": "low-config"}},
        "data": {"PEMS03": {
            "payload_sha256": "payload", "unseen_sha256": "unseen",
            "manifest_sha256": "manifest"}},
    }
    identity = expected_low_flow_identity(run, session, plan)
    assert identity["config_sha256"] == "low-config"
    assert identity["model_variant"] == "LOW"
    assert identity["validation_exposure_hash"] == "validation"

    run.entry = {"candidate": "MISSING"}
    with pytest.raises(ValueError, match="no config authority"):
        expected_low_flow_identity(run, session, plan)
