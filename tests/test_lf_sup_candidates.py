"""Behavioural contracts for the two bounded retraining candidates."""

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
    MULTISCALE_SUPPORT_NAMES,
    build_lifecycle_supports,
    distinct_support_names,
)

ROOT = Path(__file__).resolve().parents[1]


def _loss_batch(target):
    batch, horizon, nodes, _ = target.shape
    lifecycle = torch.zeros(batch, 12, nodes, 6)
    lifecycle[..., 0] = 1.0
    return {
        "target": target,
        "loss_mask": torch.ones(batch, horizon, nodes, dtype=torch.bool),
        "lifecycle_context": lifecycle,
    }


def test_lf_weight_zero_is_exactly_the_existing_objective():
    prediction = torch.tensor([[[[0.2], [-0.3]]]])
    target = torch.tensor([[[[0.0], [0.1]]]])
    batch = _loss_batch(target)
    plain = compute_losses({"pred": prediction}, batch, scaler={"mean": 5, "std": 2})
    switched_off = compute_losses({
        "pred": prediction,
        "forecast_safe_relative_weight": 0.0,
        "forecast_safe_relative_floor_policy":
            "positive_target_q10_on_train_window_training_known_nodes",
    }, batch, scaler={"mean": 5, "std": 2})
    assert torch.equal(plain["optimize"], switched_off["optimize"])
    assert switched_off["safe_relative"] is None


def test_lf_uses_original_units_and_a_train_fitted_floor():
    prediction = torch.tensor([[[[0.5], [4.5]]]])
    target = torch.tensor([[[[0.0], [4.0]]]])
    batch = _loss_batch(target)
    meta = {
        "pred": prediction,
        "forecast_safe_relative_weight": 0.05,
        "forecast_safe_relative_floor_policy":
            "positive_target_q10_on_train_window_training_known_nodes",
    }
    scaler = {"mean": 10.0, "std": 2.0, "safe_relative_floor": 5.0}
    losses = compute_losses(meta, batch, scaler=scaler)
    # errors are both one original unit; denominators are 10 and 18.
    expected = torch.tensor((1 / 10 + 1 / 18) / 2)
    assert torch.allclose(losses["safe_relative"], expected)
    assert torch.allclose(
        losses["optimize"], losses["base"] + 0.05 * expected)


def test_lf_floor_ignores_non_training_and_non_known_values():
    signal = np.asarray([[1.0, 1e9], [2.0, 1e9], [3.0, -1e9], [999.0, 999.0]])
    natural = np.ones_like(signal, dtype=bool)
    scaler = V4LifecycleDataset._fit_scaler(
        signal, natural, train_end=3, known_nodes=np.asarray([0]))
    assert scaler["mean"] == 2.0
    assert scaler["safe_relative_floor"] == pytest.approx(1.2)
    assert scaler["safe_relative_floor_n_values"] == 3


def _walks(adjacency):
    tensor = torch.as_tensor(adjacency, dtype=torch.float32)
    return tensor, calculate_random_walk_matrix(tensor), \
        calculate_random_walk_matrix(tensor.T)


def test_sup_keeps_every_resolved_operator_numerically_distinct():
    adjacency = np.asarray([
        [0.0, 0.2, 0.8, 0.0],
        [0.0, 0.0, 0.4, 0.6],
        [0.7, 0.0, 0.0, 0.3],
        [0.1, 0.9, 0.0, 0.0],
    ])
    names = distinct_support_names(adjacency)
    assert set(names).issubset(MULTISCALE_SUPPORT_NAMES)
    tensor, forward, backward = _walks(adjacency)
    supports = build_lifecycle_supports(
        tensor, forward, backward, DISTINCT_MULTISCALE_SUPPORT_SET, names)
    for left in range(len(names)):
        for right in range(left):
            assert not torch.allclose(
                supports[left], supports[right], atol=1e-7, rtol=1e-6)


def test_sup_collapses_direction_only_when_the_graph_is_symmetric():
    directed = np.asarray([[0.0, 0.2, 0.8], [0.3, 0.0, 0.7], [0.9, 0.1, 0.0]])
    symmetric = np.maximum(directed, directed.T)
    directed_names = distinct_support_names(directed)
    symmetric_names = distinct_support_names(symmetric)
    assert "forward_1" in directed_names and "backward_1" in directed_names
    assert "forward_1" in symmetric_names and "backward_1" not in symmetric_names
    assert "forward_2" in symmetric_names and "backward_2" not in symmetric_names


def test_all_patch_capacities_equal_the_resolved_maximum():
    for path in sorted((ROOT / "configs/paper/v4_data_manifests").glob("*_run_manifest.json")):
        manifest = json.loads(path.read_text("utf-8"))
        from lib.runtime_context import build_runtime_context
        context = build_runtime_context(manifest["dataset"], "train")
        largest = max(len(part) for part in context.layout_parts)
        assert manifest["layout"]["spa_patchsize"] == largest


def test_removed_evidence_router_is_not_a_runtime_source():
    assert not (ROOT / "layers/evidence_router.py").exists()
    plan_source = (ROOT / "scripts/build_v4_five_dataset_plan.py").read_text("utf-8")
    assert "layers/evidence_router.py" not in plan_source


def test_candidate_configs_are_generated_and_independent():
    from scripts.build_v4_lf_sup_candidates import LF_WEIGHT, generated
    for path, body in generated().items():
        assert path.read_text("utf-8") == body
    base = load_args_from_config(
        ROOT / "configs/paper/v4_candidates/PEMS08/BASE_ALFNet.yaml")
    lf = load_args_from_config(
        ROOT / "configs/paper/v4_candidates/PEMS08/LF_ALFNet.yaml")
    sup = load_args_from_config(
        ROOT / "configs/paper/v4_candidates/PEMS08/SUP_ALFNet.yaml")
    assert base.model_family_cfg["recipe"]["forecast_safe_relative_weight"] == 0
    assert lf.model_family_cfg["recipe"]["forecast_safe_relative_weight"] == LF_WEIGHT
    assert base.model_family_cfg["experts"]["graph"]["config_path"] != \
        sup.model_family_cfg["experts"]["graph"]["config_path"]


def test_sup_model_router_width_matches_the_resolved_inventory():
    from lib.runtime_context import build_runtime_context
    from model.ALFGraph.ALFGraph import Model
    context = build_runtime_context("PEMS08", "train")
    args = load_args_from_config(
        ROOT / "configs/paper/v4_candidates/PEMS08/experts/PEMS08_ALFGraph_sup.yaml")
    model = Model.from_runtime_context(args, context)
    expected = distinct_support_names(context.adjacency)
    assert model.resolved_support_names == expected
    assert {conv.num_experts for conv in model._routed_convs()} == {len(expected)}


def test_candidate_plan_is_exactly_five_validation_only_runs():
    from scripts import build_v4_lf_sup_plan as builder
    from scripts.train_v4 import load_plan_authority, resolve_plan
    plan = builder.assert_plan_still_holds(builder.load_plan())
    assert plan["trial_budget"] == len(plan["runs"]) == 5
    assert plan["selection"]["split"] == "validation"
    assert plan["selection"]["test_forbidden"] is True
    authoritative = load_plan_authority(builder.PLAN_PATH)
    for entry in authoritative["configs"].values():
        _, registered = resolve_plan(entry["path"], authoritative)
        assert registered["sha256"] == entry["sha256"]
