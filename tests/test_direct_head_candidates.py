"""Strict controls for the validation-only DIRECT_HEAD ablation."""

from copy import deepcopy
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from args import load_args_from_config
from experiments.v4_runner import (
    build_model_teacher_mask,
    build_v4_session,
    decoder_training_metadata,
    decoder_training_policy,
)
from layers.lifecycle_moge import LifecycleRoutedGraphConv
from lib.runtime_context import build_runtime_context
from model.ALFGraph.ALFGraph import DirectMultiHorizonHead, Model as ALFGraph
from scripts import build_direct_head_candidates as build

ROOT = Path(__file__).resolve().parents[1]


def config(name, expert="ALFNet"):
    return ROOT / "configs/paper/direct_head_candidates/PEMS03" / f"{name}_{expert}.yaml"


def payload(name, expert):
    return OmegaConf.to_container(OmegaConf.load(config(name, expert)), resolve=True)


def test_plan_is_two_fixed_budget_validation_only_runs():
    plan = build.assert_plan_still_holds(build.load_plan())
    assert plan["candidates"] == ["AR_BASE", "DIRECT_HEAD"]
    assert plan["trial_budget"] == 2
    assert plan["dataset"] == "PEMS03" and plan["seeds"] == [32]
    assert plan["selection_split"] == "validation"
    assert plan["test_access"] == "forbidden"
    assert plan["training_contract"]["epochs"] == 70
    assert plan["training_contract"]["early_termination"] is False


@pytest.mark.parametrize("expert", ("ALFNet", "ALFGraph", "ALFTemporal"))
def test_candidate_training_data_and_protocol_are_identical(expert):
    left, right = payload("AR_BASE", expert), payload("DIRECT_HEAD", expert)
    assert left["dataset"] == right["dataset"]
    assert left["training"] == right["training"]
    assert left["lifecycle"] == right["lifecycle"]


def test_only_material_architecture_difference_is_the_decoder_head():
    left, right = payload("AR_BASE", "ALFGraph"), payload("DIRECT_HEAD", "ALFGraph")
    assert left["family"]["recipe"].pop("decoder_output_mode") == "residual_delta"
    assert right["family"]["recipe"].pop("decoder_output_mode") == \
        "direct_multi_horizon"
    for item in (left, right):
        item["runtime"]["model_variant"] = "identity"
    assert left == right

    left, right = payload("AR_BASE", "ALFTemporal"), payload("DIRECT_HEAD", "ALFTemporal")
    for item in (left, right):
        item["runtime"]["model_variant"] = "identity"
    assert left == right

    left, right = payload("AR_BASE", "ALFNet"), payload("DIRECT_HEAD", "ALFNet")
    for item in (left, right):
        item["runtime"]["model_variant"] = "identity"
        item["runtime"]["model_recipe"] = "identity"
        item["family"]["experts"]["graph"]["config_path"] = "graph"
        item["family"]["experts"]["patch"]["config_path"] = "patch"
    assert left == right


def test_direct_head_shape_gradient_and_no_recurrence():
    head = DirectMultiHorizonHead(
        output_size=1, hidden_size=8, num_layers=2, out_steps=12, scale=0.3)
    anchor = torch.randn(3, 7, 1)
    hidden = torch.randn(2, 3, 7, 8, requires_grad=True)
    output = head(anchor, hidden)
    assert output.shape == (3, 12, 7, 1)
    output.square().mean().backward()
    assert hidden.grad is not None and bool((hidden.grad != 0).any())
    assert all(parameter.grad is not None for parameter in head.parameters())


def test_graph_encoder_and_support_router_are_preserved_in_both_arms():
    context = build_runtime_context("PEMS03", "train")
    models = {}
    for name in build.CANDIDATES:
        args = load_args_from_config(str(config(name, "ALFGraph")), {})
        models[name] = ALFGraph.from_runtime_context(args, context)
    assert type(models["AR_BASE"].encoder) is type(models["DIRECT_HEAD"].encoder)
    assert len(models["AR_BASE"].encoder.cells) == len(models["DIRECT_HEAD"].encoder.cells)
    assert models["AR_BASE"].resolved_support_names == \
        models["DIRECT_HEAD"].resolved_support_names
    for name in build.CANDIDATES:
        encoder_routers = [module for module in models[name].encoder.modules()
                           if isinstance(module, LifecycleRoutedGraphConv)]
        assert len(encoder_routers) > 0
    assert models["AR_BASE"].uses_external_teacher_forcing is True
    assert models["DIRECT_HEAD"].uses_external_teacher_forcing is False


def test_direct_model_is_target_blind_and_runner_reports_that_contract():
    session = build_v4_session(
        config("DIRECT_HEAD"), device="cpu", run_seed=32,
        batch_size=1, num_workers=0)
    batch = next(iter(session.validation_loader))
    session.model.eval()
    teacher = build_model_teacher_mask(session, batch, epoch=0, training=True)
    assert not bool(teacher.any())
    assert decoder_training_policy("ALFNet", session.model) == \
        "target_blind_direct_multi_horizon"
    metadata = decoder_training_metadata(session, epoch=0, training=True)
    assert metadata["ratio"] == 0.0
    assert metadata["teacher_forcing_policy"] == "none"
    assert session.model.decoder_output_mode == "direct_multi_horizon"

    poisoned = deepcopy(batch)
    poisoned["target"] = torch.randn_like(batch["target"]) * 1e9
    with torch.no_grad():
        first = session.model.forward_v4(batch, session.validation_context)["pred"]
        second = session.model.forward_v4(poisoned, session.validation_context)["pred"]
    torch.testing.assert_close(first, second, rtol=0, atol=0)

    illegal = torch.ones_like(batch["decoder_available_mask"], dtype=torch.bool)
    with pytest.raises(ValueError, match="target-blind"):
        session.model.forward_v4(
            batch, session.validation_context, teacher_forcing_mask=illegal)


def test_ar_base_keeps_the_existing_parameter_count_and_policy():
    session = build_v4_session(
        config("AR_BASE"), device="cpu", run_seed=32,
        batch_size=1, num_workers=0)
    assert sum(parameter.numel() for parameter in session.model.parameters()) == 1_037_529
    assert decoder_training_policy("ALFNet", session.model) == \
        "lifecycle_masked_autoregressive_teacher_forcing"
