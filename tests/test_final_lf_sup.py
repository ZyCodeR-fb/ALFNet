"""Contracts for the frozen final LF_SUP/SUP model."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from args import load_args_from_config
from data_provider.v4_lifecycle_loader import V4LifecycleDataset
from experiments.v4_runner import compute_losses
from layers.gnn import calculate_random_walk_matrix
from layers.lifecycle_moge import (
    DISTINCT_MULTISCALE_SUPPORT_SET,
    build_lifecycle_supports,
    distinct_support_names,
)
from lib.runtime_context import build_runtime_context

ROOT = Path(__file__).resolve().parents[1]
FLOW = ("PEMS03", "PEMS04", "PEMS08")
SPEED = ("METRLA", "PEMSBAY")


def _loss_batch(target):
    batch, horizon, nodes, _ = target.shape
    lifecycle = torch.zeros(batch, 12, nodes, 6)
    lifecycle[..., 0] = 1.0
    return {
        "target": target,
        "loss_mask": torch.ones(batch, horizon, nodes, dtype=torch.bool),
        "lifecycle_context": lifecycle,
    }


def test_safe_relative_uses_original_units_and_train_fitted_floor():
    prediction = torch.tensor([[[[0.5], [4.5]]]])
    target = torch.tensor([[[[0.0], [4.0]]]])
    losses = compute_losses({
        "pred": prediction,
        "forecast_safe_relative_weight": 0.05,
        "forecast_safe_relative_floor_policy":
            "positive_target_q10_on_train_window_training_known_nodes",
    }, _loss_batch(target), scaler={
        "mean": 10.0, "std": 2.0, "safe_relative_floor": 5.0})
    expected = torch.tensor((1 / 10 + 1 / 18) / 2)
    assert torch.allclose(losses["safe_relative"], expected)
    assert torch.allclose(losses["optimize"], losses["base"] + 0.05 * expected)


def test_safe_relative_floor_ignores_non_training_and_non_known_values():
    signal = np.asarray([[1.0, 1e9], [2.0, 1e9], [3.0, -1e9], [999.0, 999.0]])
    natural = np.ones_like(signal, dtype=bool)
    scaler = V4LifecycleDataset._fit_scaler(
        signal, natural, train_end=3, known_nodes=np.asarray([0]))
    assert scaler["mean"] == 2.0
    assert scaler["safe_relative_floor"] == pytest.approx(1.2)


def test_final_configs_apply_lf_only_to_flow_and_sup_to_every_dataset():
    for dataset in FLOW + SPEED:
        outer = load_args_from_config(
            ROOT / f"configs/paper/v4/{dataset}_ALFNet.yaml")
        graph = load_args_from_config(
            ROOT / f"configs/paper/v4/experts/{dataset}_ALFGraph.yaml")
        expected_weight = 0.05 if dataset in FLOW else 0.0
        assert outer.model_family_cfg["recipe"][
            "forecast_safe_relative_weight"] == expected_weight
        assert graph.model_family_cfg["recipe"][
            "moge_support_set"] == DISTINCT_MULTISCALE_SUPPORT_SET
        expected_variant = "alfnet_lf_sup_v1" if dataset in FLOW else "alfnet_sup_v1"
        assert outer.model_variant == expected_variant


@pytest.mark.parametrize("dataset", FLOW + SPEED)
def test_final_router_width_matches_resolved_support_inventory(dataset):
    from model.ALFGraph.ALFGraph import Model

    context = build_runtime_context(dataset, "train")
    args = load_args_from_config(
        ROOT / f"configs/paper/v4/experts/{dataset}_ALFGraph.yaml")
    model = Model.from_runtime_context(args, context)
    expected = distinct_support_names(context.adjacency)
    assert model.resolved_support_names == expected
    assert {conv.num_experts for conv in model._routed_convs()} == {len(expected)}


def test_resolved_supports_are_pairwise_distinct():
    adjacency = np.asarray([
        [0.0, 0.2, 0.8, 0.0],
        [0.0, 0.0, 0.4, 0.6],
        [0.7, 0.0, 0.0, 0.3],
        [0.1, 0.9, 0.0, 0.0],
    ])
    tensor = torch.as_tensor(adjacency, dtype=torch.float32)
    forward = calculate_random_walk_matrix(tensor)
    backward = calculate_random_walk_matrix(tensor.T)
    names = distinct_support_names(adjacency)
    supports = build_lifecycle_supports(
        tensor, forward, backward, names)
    for left in range(len(names)):
        for right in range(left):
            assert not torch.allclose(
                supports[left], supports[right], atol=1e-7, rtol=1e-6)


def test_patch_capacity_is_exactly_the_largest_resolved_part():
    for path in sorted((ROOT / "configs/paper/v4_data_manifests").glob(
            "*_run_manifest.json")):
        manifest = json.loads(path.read_text("utf-8"))
        context = build_runtime_context(manifest["dataset"], "train")
        assert manifest["layout"]["spa_patchsize"] == max(
            len(part) for part in context.layout_parts)


def test_retired_candidate_tree_and_second_evidence_router_are_absent():
    assert not (ROOT / "configs/paper/v4_candidates").exists()
    assert not (ROOT / "configs/paper/v4_lf_sup_candidate_plan.json").exists()
    assert not (ROOT / "layers/evidence_router.py").exists()
