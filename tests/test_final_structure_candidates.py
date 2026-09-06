"""Contracts for the final cumulative AR structure experiment."""

from copy import deepcopy
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from experiments.v4_runner import (
    apply_synthetic_unseen_training,
    build_v4_session,
    compute_losses,
)
from model.ALFNet.ALFNet import (
    ForecastCalibrationHead,
    ReliabilityPolynomialAdapter,
)
from scripts import build_masked_node_candidates as build

ROOT = Path(__file__).resolve().parents[1]


def config(name, expert="ALFNet"):
    if name == "AR_BASE":
        return (ROOT / "configs/paper/direct_head_candidates/PEMS03" /
                f"AR_BASE_{expert}.yaml")
    return (ROOT / "configs/paper/masked_node_candidates/PEMS03" /
            f"{name}_{expert}.yaml")


def payload(name, expert="ALFNet"):
    return OmegaConf.to_container(OmegaConf.load(config(name, expert)), resolve=True)


def test_plan_is_three_new_arms_plus_a_frozen_control():
    plan = build.assert_plan_still_holds(build.load_plan())
    assert plan["candidates"] == [
        "OBS_MASK", "OBS_MASK_POLY", "OBS_MASK_POLY_CAL"]
    assert plan["trial_budget"] == 3
    assert plan["reference_control"]["name"] == "AR_BASE"
    assert plan["reference_control"]["best_epoch"] == 58
    assert plan["selection_split"] == "validation"
    assert plan["test_access"] == "forbidden"
    assert plan["training_contract"]["max_epoch"] == 100
    assert plan["training_contract"]["early_stopping_start_epoch"] == 40
    assert plan["training_contract"]["patience"] == 10
    assert plan["training_contract"]["early_termination"] is True


def test_candidate_recipe_axes_are_strictly_cumulative():
    recipes = {name: payload(name)["family"]["recipe"]
               for name in ("AR_BASE",) + build.CANDIDATES}
    mask = set(build.MASK_RECIPE)
    poly = set(build.POLY_RECIPE)
    cal = set(build.CAL_RECIPE)
    assert not (mask | poly | cal) & set(recipes["AR_BASE"])
    assert {key: recipes["OBS_MASK"][key] for key in mask} == build.MASK_RECIPE
    assert not (poly | cal) & set(recipes["OBS_MASK"])
    assert {key: recipes["OBS_MASK_POLY"][key] for key in mask} == \
        build.MASK_RECIPE
    assert {key: recipes["OBS_MASK_POLY"][key] for key in poly} == \
        build.POLY_RECIPE
    assert not cal & set(recipes["OBS_MASK_POLY"])
    assert {key: recipes["OBS_MASK_POLY_CAL"][key] for key in mask} == \
        build.MASK_RECIPE
    assert {key: recipes["OBS_MASK_POLY_CAL"][key] for key in poly} == \
        build.POLY_RECIPE
    assert {key: recipes["OBS_MASK_POLY_CAL"][key] for key in cal} == \
        build.CAL_RECIPE


@pytest.mark.parametrize("expert", ("ALFGraph", "ALFTemporal"))
def test_nested_experts_keep_the_same_architecture(expert):
    reference = payload("OBS_MASK", expert)
    reference["runtime"]["model_variant"] = "same"
    for name in build.CANDIDATES[1:]:
        item = payload(name, expert)
        item["runtime"]["model_variant"] = "same"
        assert item == reference


def _mask_session_and_batch():
    session = build_v4_session(
        config("OBS_MASK"), device="cpu", run_seed=32,
        batch_size=2, num_workers=0)
    return session, next(iter(session.train_loader))


def test_observed_node_mask_is_deterministic_target_blind_and_non_mutating():
    session, batch = _mask_session_and_batch()
    original = deepcopy(batch)
    first = apply_synthetic_unseen_training(session, batch, epoch=7)
    poisoned = deepcopy(batch)
    poisoned["target"] = torch.randn_like(poisoned["target"]) * 1e9
    second = apply_synthetic_unseen_training(session, poisoned, epoch=7)
    torch.testing.assert_close(
        first["synthetic_unseen_node_mask"],
        second["synthetic_unseen_node_mask"], rtol=0, atol=0)
    assert bool(first["synthetic_unseen_node_mask"].any())
    selected = first["synthetic_unseen_node_mask"][:, None, :]
    assert not bool(first["input_available_mask"][selected.expand_as(
        first["input_available_mask"])].any())
    assert not bool(first["decoder_available_mask"][selected.expand_as(
        first["decoder_available_mask"])].any())
    assert bool((first["base_visible_input"][
        selected.unsqueeze(-1).expand_as(first["base_visible_input"])] == 0).all())
    torch.testing.assert_close(batch["base_visible_input"],
                               original["base_visible_input"], rtol=0, atol=0)
    assert not bool((first["synthetic_unseen_node_mask"]
                     & (batch["target_cohort_mask"] > 0.5)).any())


def test_baseline_and_validation_are_not_augmented():
    base = build_v4_session(
        config("AR_BASE"), device="cpu", run_seed=32,
        batch_size=1, num_workers=0)
    batch = next(iter(base.train_loader))
    assert apply_synthetic_unseen_training(base, batch, 0) is batch
    masked, train_batch = _mask_session_and_batch()
    validation = next(iter(masked.validation_loader))
    assert apply_synthetic_unseen_training(
        masked, validation, 0, training=False) is validation


def test_polynomial_support_excludes_unavailable_sender_and_has_identity_fallback():
    adjacency = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    value = torch.tensor([[[[2.0], [100.0]]]])
    reliability = torch.tensor([[[1.0, 0.0]]])
    propagated = ReliabilityPolynomialAdapter._propagate(
        adjacency, value, reliability)
    assert propagated[0, 0, 0, 0] == 2.0  # only neighbour unavailable
    assert propagated[0, 0, 1, 0] == 2.0  # node 0 is a reliable sender
    available = ReliabilityPolynomialAdapter._propagate(
        adjacency, value, torch.ones_like(reliability))
    assert available[0, 0, 0, 0] == 100.0


def test_polynomial_and_calibration_residuals_start_as_identity_and_train():
    adjacency = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    value = torch.randn(2, 3, 2, 1, requires_grad=True)
    poly = ReliabilityPolynomialAdapter(1)
    output = poly(value, adjacency, torch.ones(2, 3, 2))
    torch.testing.assert_close(output, value, rtol=0, atol=0)
    output.square().mean().backward()
    assert poly.output_proj.weight.grad is not None
    assert bool((poly.output_proj.weight.grad != 0).any())

    prediction = torch.randn(2, 12, 2, 1, requires_grad=True)
    roles = torch.nn.functional.one_hot(
        torch.zeros(2, 2, dtype=torch.long), 4).float()
    calibration = ForecastCalibrationHead(1, 12)
    calibrated = calibration(prediction, roles)
    torch.testing.assert_close(calibrated, prediction, rtol=0, atol=0)
    calibrated.square().mean().backward()
    assert calibration.mlp[-1].weight.grad is not None
    assert bool((calibration.mlp[-1].weight.grad != 0).any())


def test_train_scale_bin_loss_is_finite_at_zero_without_inverse_target():
    prediction = torch.tensor([[[[0.1], [0.2]]]], requires_grad=True)
    target = torch.zeros_like(prediction)
    output = {
        "pred": prediction,
        "low_flow_calibration_mode":
            "role_horizon_residual_train_scale_bins",
        "low_flow_bin_loss_weight": 0.10,
    }
    batch = {
        "target": target,
        "loss_mask": torch.ones(1, 1, 2, dtype=torch.bool),
    }
    losses = compute_losses(
        output, batch, scaler={"mean": 100.0, "std": 50.0})
    assert torch.isfinite(losses["low_flow_bin_mix"])
    losses["optimize"].backward()
    assert torch.isfinite(prediction.grad).all()


@pytest.mark.parametrize("name", ("AR_BASE",) + build.CANDIDATES)
def test_each_candidate_builds_with_the_declared_method(name):
    session = build_v4_session(
        config(name), device="cpu", run_seed=32,
        batch_size=1, num_workers=0)
    model = session.model
    assert model.decoder_output_mode == "residual_delta"
    assert (model.synthetic_unseen_training_mode is not None) == \
        (name != "AR_BASE")
    assert (model.reliability_polynomial_adapter is not None) == \
        (name in {"OBS_MASK_POLY", "OBS_MASK_POLY_CAL"})
    assert (model.forecast_calibration_head is not None) == \
        (name == "OBS_MASK_POLY_CAL")
