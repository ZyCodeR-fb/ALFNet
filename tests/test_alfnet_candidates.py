"""Bounded ALFNet candidate contracts; no test split and no baseline changes."""

import itertools
import json
from pathlib import Path

import pytest
import torch

from args import load_args_from_config
from experiments.v4_runner import compute_losses
from layers.gnn import calculate_random_walk_matrix
from layers.lifecycle_moge import (
    MULTISCALE_SUPPORT_NAMES,
    MULTISCALE_SUPPORT_SET,
    build_lifecycle_supports,
)
from lib.runtime_context import build_runtime_context
from model.ALFTemporal.ALFTemporal import _reorder_data
from scripts import build_alfnet_candidates

ROOT = Path(__file__).resolve().parents[1]


def config(name, expert="ALFNet"):
    return (ROOT / "configs/paper/alfnet_candidates/PEMS03" /
            f"{name}_{expert}.yaml")


def test_the_candidate_inventory_and_selection_rule_are_frozen():
    plan = build_alfnet_candidates.assert_plan_still_holds(
        build_alfnet_candidates.load_plan())
    assert plan["candidates"] == [
        "BASE", "UNSEEN", "LOW", "SUP", "PATCH", "LOW_SUP", "ALL"]
    assert plan["trial_budget"] == 7
    assert plan["dataset"] == "PEMS03" and plan["seeds"] == [32]
    assert plan["selection_split"] == "validation"
    assert plan["test_access"] == "forbidden"
    assert plan["selection_rule"]["test_metrics_may_influence_selection"] is False


@pytest.mark.parametrize("name", build_alfnet_candidates.CANDIDATES)
def test_every_candidate_changes_exactly_its_declared_axes(name):
    flags = build_alfnet_candidates._recipe_flags(name)
    outer = load_args_from_config(str(config(name)), {})
    graph = load_args_from_config(str(config(name, "ALFGraph")), {})
    patch = load_args_from_config(str(config(name, "ALFTemporal")), {})
    recipe = outer.model_family_cfg["recipe"]
    assert (recipe["graph_diffusion_role_scale"]["aas"] == 1.0) == flags["unseen"]
    assert (recipe.get("forecast_scale_loss_mode") == "signed_log_huber") == flags["low"]
    assert (graph.model_family_cfg["recipe"]["moge_support_set"] ==
            MULTISCALE_SUPPORT_SET) == flags["support"]
    assert (patch.model_family_cfg["model"]["spa_patchsize"] == 13) == flags["patch"]
    if not flags["patch"]:
        assert patch.model_family_cfg["model"]["spa_patchsize"] == 12


@pytest.mark.parametrize("dataset", ("PEMS03", "PEMS04", "PEMS08", "PEMSBAY"))
def test_multiscale_supports_are_numerically_distinct_on_every_active_graph(dataset):
    adjacency = build_runtime_context(dataset, "train").adjacency.float()
    forward = calculate_random_walk_matrix(adjacency)
    backward = calculate_random_walk_matrix(adjacency.T)
    supports = build_lifecycle_supports(
        adjacency, forward, backward, support_set=MULTISCALE_SUPPORT_SET,
        selected_names=MULTISCALE_SUPPORT_NAMES)
    assert supports.shape[0] == 5
    for left, right in itertools.combinations(range(5), 2):
        assert not torch.allclose(supports[left], supports[right], atol=1e-7,
                                  rtol=1e-6), (dataset, left, right)
    assert torch.isfinite(supports).all()


def test_patch_candidate_adds_exactly_one_boundary_context_slot_per_pems03_part():
    context = build_runtime_context("PEMS03", "train")
    parts = [torch.tensor(part).numpy() for part in context.layout_parts]
    assert max(map(len, parts)) == 12
    original, positions, all_nodes = _reorder_data(
        parts, context.adjacency.numpy(), 13)
    assert len(all_nodes) == 13 * len(parts)
    for index, part in enumerate(parts):
        token = all_nodes[index * 13:(index + 1) * 13]
        assert set(part).issubset(set(token))
        assert len(set(token) - set(part)) == 13 - len(part)


def test_signed_log_huber_is_finite_and_continuous_at_zero_without_target_division():
    prediction = torch.tensor([[[[0.1], [0.2]]]], requires_grad=True)
    target = torch.zeros_like(prediction)
    batch = {
        "target": target,
        "loss_mask": torch.ones(1, 1, 2, dtype=torch.bool),
        "lifecycle_context": torch.zeros(1, 1, 2, 6),
    }
    batch["lifecycle_context"][..., 0] = 1.0
    meta = {"pred": prediction, "forecast_loss_mode": "role_balanced_mix",
            "forecast_role_balanced_beta": 0.1,
            "forecast_scale_loss_mode": "signed_log_huber",
            "forecast_scale_loss_weight": 0.05,
            "forecast_scale_loss_scale_fraction": 0.05}
    first = compute_losses(meta, batch, scaler={"mean": 0.0, "std": 100.0})
    assert torch.isfinite(first["scale_mix"])
    first["optimize"].backward()
    assert torch.isfinite(prediction.grad).all() and bool((prediction.grad != 0).any())
    batch["target"] = torch.full_like(target, 1e-12)
    second = compute_losses(meta, batch, scaler={"mean": 0.0, "std": 100.0})
    assert abs(float((first["scale_mix"] - second["scale_mix"]).detach())) < 1e-9


def test_base_has_no_low_value_auxiliary_and_mogernn_is_not_in_the_plan():
    base = load_args_from_config(str(config("BASE")), {})
    recipe = base.model_family_cfg["recipe"]
    assert "forecast_scale_loss_mode" not in recipe
    plan = build_alfnet_candidates.load_plan()
    assert plan["models"] == ["ALFNet"]
    assert all(run["model"] == "ALFNet" for run in plan["runs"])
