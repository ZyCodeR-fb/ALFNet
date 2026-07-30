"""Structure, build and finite-value checks for the slimmed ALFNet variants.

fg1 is the FiLM-conditioned plan; fg1r adds an FS-only bounded additive
residual on top of it. rf_l3s is the incumbent two-expert model.
"""

import torch
import pytest

from args import load_args_from_config
from layers.lifecycle_moge import LifecycleRoutedGraphConv
from model.regist_model import load_model_class


VARIANTS = {
    "fg1": "configs/ALFNet/fg1.yaml",
    "fg1r": "configs/ALFNet/fg1r.yaml",
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


@pytest.mark.parametrize("name,layers,timefilter",
                         [("fg1", 1, False), ("fg1r", 1, False), ("rf_l3s", 5, True)])
def test_patch_branch_depth_and_timefilter_are_yaml_switches(name, layers, timefilter):
    _, model = build(name)
    blocks = model.patch_expert.backbone.spa_encoder
    assert len(blocks) == layers
    assert model.patch_expert.use_timefilter is timefilter
    assert all((block.timefilter_patch_mixer is not None) is timefilter for block in blocks)


@pytest.mark.parametrize("name,top_k", [("fg1", 1), ("fg1r", 1), ("rf_l3s", 2)])
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
    assert sizes["fg1"] < sizes["fg1r"] < sizes["rf_l3s"]
    # the residual head is a small addition, not a second decoder
    assert sizes["fg1r"] - sizes["fg1"] < 20_000
    assert sizes["fg1"] < 0.35 * sizes["rf_l3s"]


def test_film_mode_rejects_a_patch_expert_that_still_decodes():
    """Mismatched switches must fail loudly rather than build dead modules."""
    args = load_args_from_config(VARIANTS["fg1"], [])
    args.device = "cpu"
    args.model_family_cfg["experts"]["patch"]["config_path"] = (
        "./configs/ALFTemporal/METRLA_alftemporal_t65.yaml"
    )
    with pytest.raises(ValueError, match="emit=features"):
        load_model_class(args.model)(args)


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
            "training_protocol_hash": "hash",
            # This regression targets path resolution, but the payload still has
            # to carry provenance: a checkpoint without it is refused outright
            # so a result can never be attributed to the wrong model.
            "config": {
                "model": "ALFNet",
                "model_variant": "fg1",
                "router_mode": None,
                "canonical_config_hash": "canonical",
                "training_protocol_hash": "hash",
            },
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
        model_family_cfg={},
        input_len=12,
        pred_len=12,
        enc_in=7,
        raw_enc_in=1,
        c_out=1,
        d_model=64,
        lifecycle_training_strategy="lcbt_v3",
        lifecycle_role_features=True,
        lifecycle_history_age_feature=True,
        lifecycle_observation_mask_feature=True,
        lifecycle_age_mode="state_duration",
        training_protocol_hash="hash",
        canonical_config_hash="canonical",
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


# --- FS-only bounded additive residual (fg1r) --------------------------------


def _role_batch(args, model, roles, batch=2):
    """Batch whose last input step assigns each node an explicit role."""
    data = _batch(args, model, batch)
    nodes = len(model.known_nodes)
    onehot = torch.zeros(batch, args.input_len, nodes, 4)
    for index, role in enumerate(roles):
        onehot[:, :, index, role] = 1.0
    data["x"][..., model.role_start:model.role_end] = onehot
    return data


def test_residual_is_exactly_zero_off_the_fault_role():
    """Unseen, Steady and Join must stay bit-identical to fg1."""
    args, model = build("fg1r")
    model.eval()
    nodes = len(model.known_nodes)
    # every node non-FS: role 0=AAS, 1=NAS, 2=VS
    roles = [index % 3 for index in range(nodes)]
    data = _role_batch(args, model, roles)
    torch.manual_seed(0)
    for layer in model.fault_residual_head:
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.normal_(layer.weight, std=0.5)
            torch.nn.init.normal_(layer.bias, std=0.5)
    with torch.no_grad():
        with_head = model(data["adj"], data["x"], data["tm"], None, None, data["dm"], 0)["pred"]
        model.fault_residual_head = None
        without = model(data["adj"], data["x"], data["tm"], None, None, data["dm"], 0)["pred"]
    assert torch.equal(with_head, without), "residual leaked onto a non-fault role"


def test_residual_moves_only_the_fault_nodes():
    args, model = build("fg1r")
    model.eval()
    nodes = len(model.known_nodes)
    roles = [3 if index % 4 == 0 else 0 for index in range(nodes)]  # every 4th is FS
    fault_nodes = [index for index, role in enumerate(roles) if role == 3]
    data = _role_batch(args, model, roles)
    torch.manual_seed(0)
    for layer in model.fault_residual_head:
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.normal_(layer.weight, std=0.5)
            torch.nn.init.normal_(layer.bias, std=0.5)
    with torch.no_grad():
        with_head = model(data["adj"], data["x"], data["tm"], None, None, data["dm"], 0)["pred"]
        head, model.fault_residual_head = model.fault_residual_head, None
        without = model(data["adj"], data["x"], data["tm"], None, None, data["dm"], 0)["pred"]
    delta = (with_head - without).abs()
    other = [index for index in range(nodes) if index not in fault_nodes]
    assert delta[:, :, other, :].max() == 0.0
    assert delta[:, :, fault_nodes, :].max() > 0.0
    # bounded by scale * tanh(.) <= scale
    assert delta.max() <= model.fault_residual_scale + 1e-6


def test_residual_head_is_zero_initialised_so_fg1r_starts_as_fg1():
    _, model = build("fg1r")
    last = model.fault_residual_head[-1]
    assert torch.allclose(last.weight, torch.zeros_like(last.weight))
    assert torch.allclose(last.bias, torch.zeros_like(last.bias))
    assert model.fault_residual_scale == pytest.approx(0.35)


def test_fg1_has_no_residual_head_and_fg1r_only_adds_that():
    _, plain = build("fg1")
    _, residual = build("fg1r")
    assert plain.fault_residual_head is None
    assert plain.fault_residual_scale == 0.0
    extra = set(residual.state_dict()) - set(plain.state_dict())
    assert all(key.startswith("fault_residual_head.") for key in extra), extra
    # nothing else changed: no second decoder, no gate, no budget came back
    assert residual.patch_expert.backbone.regression_conv is None
    assert residual.availability_gates is None
    assert residual.temporal_refiner is None
    assert not hasattr(residual, "patch_budget")


def test_residual_is_rejected_outside_film_mode():
    args = load_args_from_config(VARIANTS["rf_l3s"], [])
    args.device = "cpu"
    args.model_family_cfg["recipe"]["fault_residual_scale"] = 0.35
    with pytest.raises(ValueError, match="residual_gate mode already blends"):
        load_model_class(args.model)(args)


def test_retired_candidate_files_are_gone():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    for stale in ("configs/ALFNet/rg2.yaml", "configs/ALFTemporal/METRLA_lite2.yaml"):
        assert not (root / stale).exists(), f"{stale} should have been deleted with rg2"
