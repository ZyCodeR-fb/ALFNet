"""Structure, build and finite-value checks for the slimmed ALFNet variants.

fg1 is the main plan (patch branch conditions the graph encoder via FiLM);
rg2 is the low-risk fallback (both experts still decode, gate retained).
"""

import torch
import pytest

from args import load_args_from_config
from layers.lifecycle_moge import LifecycleRoutedGraphConv
from model.regist_model import load_model_class


VARIANTS = {
    "fg1": "configs/ALFNet/fg1.yaml",
    "rg2": "configs/ALFNet/rg2.yaml",
    "rf_l3s": "configs/ALFNet/rf_l3s.yaml",
}


def build(name):
    args = load_args_from_config(VARIANTS[name], [])
    args.device = "cpu"
    return args, load_model_class(args.model)(args)


def _batch(args, model, batch=2):
    nodes = len(model.known_nodes)
    generator = torch.Generator().manual_seed(7)
    return dict(
        adj=torch.rand(nodes, nodes, generator=generator),
        x=torch.randn(batch, args.input_len, nodes, args.enc_in, generator=generator),
        tm=torch.rand(batch, args.input_len, 3, generator=generator) * 5,
        dec=torch.randn(batch, args.pred_len, nodes, args.c_out, generator=generator),
        dm=torch.rand(batch, args.pred_len, 3, generator=generator) * 5,
    )


@pytest.mark.parametrize("name", sorted(VARIANTS))
def test_variant_builds_and_produces_finite_gradients(name):
    args, model = build(name)
    data = _batch(args, model)
    model.train()
    out = model(data["adj"], data["x"], data["tm"], None, data["dec"], data["dm"], 0)
    pred = out["pred"]
    assert pred.shape == (2, args.pred_len, len(model.known_nodes), args.c_out)
    assert torch.isfinite(pred).all(), f"{name} produced non-finite predictions"
    loss = pred.pow(2).mean()
    for value in (out.get("aux_losses") or {}).values():
        if torch.is_tensor(value):
            loss = loss + value.float()
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert grads, f"{name} produced no gradients"
    assert all(torch.isfinite(p.grad).all() for p in grads)


def test_film_variant_drops_the_two_forecast_machinery():
    _, model = build("fg1")
    assert model.fusion_mode == "film"
    # no second decoder, no blend gate, no residual budget, no route/regret
    assert model.patch_expert.backbone.regression_conv is None
    assert model.availability_gates is None
    assert model.temporal_refiner is None
    assert not hasattr(model, "patch_budget")
    assert model.lambda_route == 0.0 and model.lambda_regret == 0.0
    assert model.film_head is not None
    # and the forward advertises no routing metadata for losses that are gone
    args, _ = build("fg1")
    data = _batch(args, model)
    out = model(data["adj"], data["x"], data["tm"], None, data["dec"], data["dm"], 0)
    assert "route_routing" not in out and "regret_routing" not in out


def test_film_head_starts_as_the_identity():
    """A zero-initialised head must leave the DCRNN state untouched at step 0."""
    _, model = build("fg1")
    features = torch.randn(3, 5, model.patch_expert.feature_dim)
    gamma, beta = model.film_head(features).chunk(2, dim=-1)
    assert torch.allclose(gamma, torch.zeros_like(gamma))
    assert torch.allclose(beta, torch.zeros_like(beta))
    # the forward applies (1 + gamma), so the modulation is exactly identity
    hidden = torch.randn(2, 3, 5, gamma.shape[-1])
    assert torch.allclose(hidden * (1.0 + gamma).unsqueeze(0) + beta.unsqueeze(0), hidden)


def test_fallback_variant_keeps_the_gate_and_opens_vs():
    _, model = build("rg2")
    assert model.fusion_mode == "residual_gate"
    assert model.availability_gates is not None
    assert model.temporal_refiner is not None
    assert model.patch_expert.backbone.regression_conv is not None
    # vs was 0.00 in rf_l3s; the fallback turns the spatial branch on for
    # zero-history nodes, which are the group that only has spatial information.
    assert float(model.patch_budget[2]) == pytest.approx(0.20)


@pytest.mark.parametrize("name,layers,timefilter", [("fg1", 1, False), ("rg2", 2, False), ("rf_l3s", 5, True)])
def test_patch_branch_depth_and_timefilter_are_yaml_switches(name, layers, timefilter):
    _, model = build(name)
    blocks = model.patch_expert.backbone.spa_encoder
    assert len(blocks) == layers
    assert model.patch_expert.use_timefilter is timefilter
    assert all((block.timefilter_patch_mixer is not None) is timefilter for block in blocks)


@pytest.mark.parametrize("name,top_k", [("fg1", 1), ("rg2", 1), ("rf_l3s", 2)])
def test_moge_top_k_is_a_yaml_switch_and_keeps_sparse_routing(name, top_k):
    _, model = build(name)
    convs = [m for m in model.modules() if isinstance(m, LifecycleRoutedGraphConv)]
    assert convs, "MoGE routing must survive in every variant"
    assert {c.router.k for c in convs} == {top_k}
    # top-k=1 is still sparse routing over five experts, not a deleted MoGE
    assert {c.router.num_experts for c in convs} == {5}


def test_parameter_budget_shrinks_without_losing_the_mainline():
    sizes = {}
    for name in VARIANTS:
        _, model = build(name)
        sizes[name] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert model.shared_input_imputer is not None, f"{name} lost the shared GRIN"
        assert hasattr(model.graph_expert, "decoder"), f"{name} lost the Residual-Delta decoder"
    assert sizes["fg1"] < sizes["rg2"] < sizes["rf_l3s"]
    assert sizes["fg1"] < 0.35 * sizes["rf_l3s"]


def test_film_mode_rejects_a_patch_expert_that_still_decodes():
    """Mismatched switches must fail loudly rather than build dead modules."""
    args = load_args_from_config(VARIANTS["fg1"], [])
    args.device = "cpu"
    args.model_family_cfg["experts"]["patch"]["config_path"] = (
        "./configs/ALFTemporal/METRLA_lite2.yaml"
    )
    with pytest.raises(ValueError, match="emit=features"):
        load_model_class(args.model)(args)


def test_residual_gate_rejects_a_feature_only_patch_expert():
    args = load_args_from_config(VARIANTS["rg2"], [])
    args.device = "cpu"
    args.model_family_cfg["experts"]["patch"]["config_path"] = (
        "./configs/ALFTemporal/METRLA_cond1.yaml"
    )
    with pytest.raises(ValueError, match="no forecast to blend"):
        load_model_class(args.model)(args)


@pytest.mark.parametrize("name", sorted(VARIANTS))
def test_state_dict_round_trips(name):
    _, model = build(name)
    state = model.state_dict()
    _, fresh = build(name)
    fresh.load_state_dict(state, strict=True)
    assert set(fresh.state_dict()) == set(state)


# --- unified output root -----------------------------------------------------


def test_output_root_collects_every_artefact_under_one_run_folder():
    import os
    from types import SimpleNamespace

    from args import OUTPUT_SUBDIRS, apply_output_root

    args = SimpleNamespace(output_root="outputs", run_id="fg1-s32",
                           model="ALFNet", exp_id="a" * 176)
    apply_output_root(args)
    base = os.path.join("outputs", "ALFNet", "fg1-s32")
    for name, sub in OUTPUT_SUBDIRS.items():
        assert getattr(args, name) == os.path.join(base, sub)
    assert args.profile_dir == os.path.join(base, "profiles")
    # run_id must not be shadowed by the long exp_id anywhere in the tree
    assert all("a" * 20 not in getattr(args, name) for name in OUTPUT_SUBDIRS)
    assert args.paths_are_run_scoped is True


def test_output_root_defaults_the_run_folder_to_the_experiment_id():
    import os
    from types import SimpleNamespace

    from args import apply_output_root

    args = SimpleNamespace(output_root="outputs", run_id=None,
                           model="MoGERNN", exp_id="MoGERNN-x")
    apply_output_root(args)
    assert args.best_checkpoint_path == os.path.join(
        "outputs", "MoGERNN", "MoGERNN-x", "checkpoints")


def test_legacy_layout_is_untouched_without_output_root():
    from types import SimpleNamespace

    from args import apply_output_root

    args = SimpleNamespace(output_root=None, run_id=None, model="ALFNet",
                           exp_id="x", best_checkpoint_path="./check_points/")
    apply_output_root(args)
    assert args.best_checkpoint_path == "./check_points/"
    assert not getattr(args, "paths_are_run_scoped", False)


def test_test_async_loads_checkpoint_from_the_same_run_scoped_directory(tmp_path):
    """Training and test_async must agree on the checkpoint directory."""
    import os
    from types import SimpleNamespace

    import torch

    from main import load_pretrained_model

    model = torch.nn.Linear(2, 1)
    checkpoint_dir = tmp_path / "outputs" / "ALFNet" / "fg1-s32" / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            # This regression targets path resolution; compatibility is covered
            # independently by the checkpoint tests.
            "config": None,
        },
        checkpoint_dir / "best_val_checkpoint.pth",
    )
    args = SimpleNamespace(
        best_checkpoint_path=str(checkpoint_dir),
        paths_are_run_scoped=True,
        exp_id="a-long-exp-id-that-must-not-be-appended",
        pretrained_model_path=None,
        model="ALFNet",
        model_variant="fg1",
        input_len=12,
        pred_len=12,
        enc_in=7,
        c_out=1,
        d_model=64,
        training_protocol_hash="hash",
    )

    loaded, resolved = load_pretrained_model(model, args)
    assert isinstance(loaded, torch.nn.Linear)
    assert resolved == str(checkpoint_dir) + os.sep


def test_result_directories_are_gitignored():
    import subprocess

    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    for candidate in ("outputs/x", "run_v5/x", "_smoke/x", "_prof/x",
                      "predictions/x", "check_points/x", "a/b.pth"):
        result = subprocess.run(
            ["git", "check-ignore", "-q", "--no-index", candidate],
            cwd=root, capture_output=True,
        )
        assert result.returncode == 0, f"{candidate} is not ignored"
