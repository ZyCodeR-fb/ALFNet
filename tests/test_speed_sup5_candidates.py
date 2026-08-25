"""Contracts for the final bounded speed SUP5 candidate study."""

from pathlib import Path

import pytest
import torch

from args import load_args_from_config
from experiments.v4_runner import build_v4_session, resolve_context
from layers.gnn import calculate_random_walk_matrix
from layers.lifecycle_moge import (
    DISTINCT_MULTISCALE_SUPPORT_SET,
    EXPERIMENTAL_UNIFIED_MULTISCALE5_SUPPORT_SET,
    build_lifecycle_supports,
    distinct_support_names,
)
from lib.runtime_context import build_runtime_context
from scripts import build_v4_speed_sup5_plan as builder
from scripts.train_v4 import RunnerRefused, load_plan_authority
from scripts.v4_launcher import plan_runs, resolve_plan_and_root

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    ("METRLA", "SUP5", "ALFNet"),
    ("METRLA", "LF_SUP5", "ALFNet"),
    ("PEMSBAY", "BASE", "ALFNet"),
    ("PEMSBAY", "SUP5", "ALFNet"),
    ("PEMSBAY", "LF_SUP5", "ALFNet"),
    ("PEMSBAY", "CONTROL", "MoGERNN"),
}


def test_plan_is_exactly_six_validation_only_runs():
    plan = builder.assert_plan_still_holds(builder.load_plan())
    assert plan["trial_budget"] == 6
    assert plan["seeds"] == [32]
    assert plan["selection"]["split"] == "validation"
    assert plan["selection"]["test_forbidden"] is True
    assert {(row["dataset"], row["candidate"], row["model"])
            for row in plan["runs"]} == EXPECTED


@pytest.mark.parametrize("dataset", ("METRLA", "PEMSBAY"))
def test_sup5_is_exactly_five_ordered_speed_supports(dataset):
    context = build_runtime_context(dataset, "train")
    names = distinct_support_names(
        context.adjacency, EXPERIMENTAL_UNIFIED_MULTISCALE5_SUPPORT_SET)
    assert names == ("identity", "forward_1", "backward_1",
                     "forward_2", "backward_2")


@pytest.mark.parametrize("dataset", ("PEMS03", "PEMS04", "PEMS08"))
def test_sup5_is_numerically_identical_to_current_flow_supports(dataset):
    context = build_runtime_context(dataset, "train")
    adjacency = torch.as_tensor(context.adjacency, dtype=torch.float32)
    forward = calculate_random_walk_matrix(adjacency)
    backward = calculate_random_walk_matrix(adjacency.T)
    old_names = distinct_support_names(adjacency, DISTINCT_MULTISCALE_SUPPORT_SET)
    new_names = distinct_support_names(
        adjacency, EXPERIMENTAL_UNIFIED_MULTISCALE5_SUPPORT_SET)
    old = build_lifecycle_supports(
        adjacency, forward, backward, old_names, DISTINCT_MULTISCALE_SUPPORT_SET)
    new = build_lifecycle_supports(
        adjacency, forward, backward, new_names,
        EXPERIMENTAL_UNIFIED_MULTISCALE5_SUPPORT_SET)
    assert old.shape == new.shape == (5, adjacency.shape[0], adjacency.shape[0])
    assert torch.allclose(old, new, atol=0.0, rtol=0.0)


@pytest.mark.parametrize(
    "dataset,candidate,weight",
    (("METRLA", "SUP5", 0.0), ("METRLA", "LF_SUP5", 0.05),
     ("PEMSBAY", "SUP5", 0.0), ("PEMSBAY", "LF_SUP5", 0.05)),
)
def test_candidate_factorial_is_exact(dataset, candidate, weight):
    path = ROOT / f"configs/paper/v4_speed_sup5_candidates/{dataset}/{candidate}_ALFNet.yaml"
    args = load_args_from_config(path)
    recipe = args.model_family_cfg["recipe"]
    assert recipe["forecast_safe_relative_weight"] == weight
    assert recipe["candidate_validation_only"] is True
    assert recipe["experimental_speed_safe_relative"] is (weight > 0)
    graph = load_args_from_config(
        ROOT / f"configs/paper/v4_speed_sup5_candidates/{dataset}/experts/ALFGraph_SUP5.yaml")
    assert graph.model_family_cfg["recipe"]["moge_support_set"] == \
        EXPERIMENTAL_UNIFIED_MULTISCALE5_SUPPORT_SET


def test_candidate_models_construct_with_five_router_experts():
    for dataset in ("METRLA", "PEMSBAY"):
        session = build_v4_session(
            ROOT / f"configs/paper/v4_speed_sup5_candidates/{dataset}/SUP5_ALFNet.yaml",
            device="cpu", run_seed=32, batch_size=1, num_workers=0)
        graph = session.model.graph_expert
        assert graph.resolved_support_names == (
            "identity", "forward_1", "backward_1", "forward_2", "backward_2")
        assert {conv.num_experts for conv in graph._routed_convs()} == {5}


def test_candidate_plan_is_a_trainer_and_launcher_authority(monkeypatch):
    plan = load_plan_authority(builder.PLAN_PATH)
    assert plan["plan_sha256"] == builder.load_plan()["plan_sha256"]
    resolved, root = resolve_plan_and_root(builder.PLAN_PATH)
    runs = plan_runs(root, resolved)
    assert len(runs) == 6
    assert len({run.name for run in runs}) == 6
    assert {(run.dataset, run.candidate, run.model) for run in runs} == EXPECTED


def test_candidate_training_still_refuses_test_context():
    with pytest.raises(RunnerRefused, match="test.*context"):
        resolve_context("METRLA", "test", 32)
