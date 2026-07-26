import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from args import load_args_from_config
from layers.DCRNNCell import DiffConv
from layers.lifecycle_moge import (
    LIFECYCLE_STATE_CHANNELS,
    SUPPORT_NAMES,
    LifecycleRoutedGraphConv,
    LifecycleRouter,
    build_lifecycle_supports,
)
from layers.source_grin import GRINInputImputer
from layers.source_timefilter import TimeFilterPatchMixer
from model.ALFGraph.ALFGraph import Model as ALFGraph
from model.ALFNet.ALFNet import Model as ALFNet
from model.ALFTemporal.ALFTemporal import Model as ALFTemporal
from model.ALFTemporal.ALFTemporal import _build_spatial_layout
from model.ALFTemporal.patchstg import WindowAttBlock
from utils.model_output import sum_aux_losses, unpack_model_output


ROOT = Path(__file__).resolve().parents[1]
COMPARISON_MODELS = (
    "MoGERNN",
    "PatchSTG",
    "FCLSTM",
    "GWNET",
    "HA",
    "MCPST_FSL",
    "SLPF",
    "STBP",
    "STFiT",
    "STGCN",
    "STGODE",
    "STID",
)


def _read_yaml(relative_path):
    with (ROOT / relative_path).open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def test_streamlined_yaml_surface_has_one_alfnet_mainline():
    assert [path.name for path in (ROOT / "configs/ALFNet").glob("*.yaml")] == [
        "rf_l3s.yaml"
    ]
    config = _read_yaml("configs/ALFNet/rf_l3s.yaml")
    assert config["runtime"]["model"] == "ALFNet"
    assert config["runtime"]["model_variant"] == "rf_l3s"
    assert config["lifecycle"]["lifecycle_training_strategy"] == "lcbt_v3"
    assert config["async_test"]["lce_protocol_version"] == 3
    assert config["async_test"]["ples_protocol_version"] == 3
    assert set(config["family"]["experts"]) == {"graph", "patch"}

    for model in COMPARISON_MODELS:
        config_paths = [
            path.name for path in (ROOT / "configs" / model).glob("*.yaml")
        ]
        assert config_paths == ["METRLA_lcbt_v3.yaml"]
        args = load_args_from_config(
            str(ROOT / "configs" / model / config_paths[0]),
            overrides={"device": "cpu"},
        )
        assert args.model == model
        assert args.lifecycle_training_strategy == "lcbt_v3"
        assert args.stop_based == "val_lifecycle"
        assert args.lce_protocol_version == 3
        assert args.ples_protocol_version == 3


def test_alfnet_contains_one_shared_grin_and_no_graph_local_grin():
    args = load_args_from_config(
        str(ROOT / "configs/ALFNet/rf_l3s.yaml"),
        overrides={"device": "cpu"},
    )
    model = ALFNet(args)
    grin_modules = [
        module
        for module in model.modules()
        if isinstance(module, GRINInputImputer)
    ]

    assert grin_modules == [model.shared_input_imputer]
    assert not any(
        isinstance(module, GRINInputImputer)
        for module in model.graph_expert.modules()
    )
    # 3_288_746 + 4_608 for the widened patch input convolution that carries the
    # availability / state_age / observation-mask channels (6 * 64 * 12), then
    # + 151_824 for the lifecycle-routed graph core replacing DiffConv.
    assert sum(parameter.numel() for parameter in model.parameters()) == 3_445_178
    assert not hasattr(model, "fusion_gate")
    assert all(
        not hasattr(block, "nattn") and not hasattr(block, "sattn")
        for block in model.patch_expert.backbone.spa_encoder
    )
    assert model.graph_expert.decoder.scale == 0.3


def _alftemporal_args(lifecycle_input=None):
    args = load_args_from_config(
        str(ROOT / "configs/ALFTemporal/METRLA_alftemporal_t65.yaml"),
        overrides={"device": "cpu"},
    )
    if lifecycle_input is not None:
        args.model_family_cfg["model"]["lifecycle_input"] = lifecycle_input
    return args


def test_alftemporal_consumes_prediction_time_lifecycle_context():
    expert = ALFTemporal(_alftemporal_args())

    # raw + availability(4) + state_age + observation mask, then two calendar
    # channels appended inside the backbone embedding.
    assert expert.lifecycle_input is True
    assert expert.expert_input_dim == 7
    assert expert.backbone.input_channels == 9
    assert expert.backbone.input_st_fc.in_channels == 9


def test_alftemporal_lifecycle_input_ablation_falls_back_to_raw_only():
    expert = ALFTemporal(_alftemporal_args(lifecycle_input=False))

    assert expert.expert_input_dim == 1
    assert expert.backbone.input_channels == 3


def test_alfnet_routes_lifecycle_channels_to_both_experts():
    args = load_args_from_config(
        str(ROOT / "configs/ALFNet/rf_l3s.yaml"),
        overrides={"device": "cpu"},
    )
    model = ALFNet(args)

    assert model.graph_input_dim == args.enc_in
    assert model.patch_input_dim == args.enc_in
    assert model.patch_expert.expert_input_dim == args.enc_in


def test_family_override_sweeps_one_leaf_without_touching_the_config():
    """Parallel arms must differ in exactly one scalar, provably."""
    import argparse

    from args import _add_arguments, _load_config_payload, finalize_args
    from main import build_exp_id

    config_path = ROOT / "configs/ALFNet/rf_l3s.yaml"
    path = str(config_path)
    before = config_path.read_bytes()

    def _load(*overrides):
        parser = argparse.ArgumentParser()
        _add_arguments(parser)
        defaults, family = _load_config_payload(path, parser)
        parser.set_defaults(**defaults)
        args = parser.parse_args(["--device", "cpu", *overrides])
        args.model_family_cfg = family
        return finalize_args(args)

    baseline = _load().model_family_cfg["recipe"]["patch_residual_role_scale"]
    untouched = {key: baseline[key] for key in ("aas", "vs", "fs")}

    budgets, exp_ids = {}, set()
    for value in ("0.10", "0.20", "0.35"):
        args = _load("--family_override", f"recipe.patch_residual_role_scale.nas={value}")
        budgets[value] = args.model_family_cfg["recipe"]["patch_residual_role_scale"]
        exp_ids.add(build_exp_id(args))

    assert [budgets[v]["nas"] for v in ("0.10", "0.20", "0.35")] == [0.10, 0.20, 0.35]
    # Only nas moves; every other role budget keeps whatever the config holds.
    for scale in budgets.values():
        assert {key: scale[key] for key in ("aas", "vs", "fs")} == untouched
    # Distinct experiment directories, so parallel arms cannot overwrite one another.
    assert len(exp_ids) == 3
    # The override lives in memory: the file on disk is byte-identical afterwards.
    assert config_path.read_bytes() == before

    with pytest.raises(ValueError, match="refusing to invent"):
        _load("--family_override", "recipe.no_such_knob=1")
    with pytest.raises(ValueError, match="dotted.path=value"):
        _load("--family_override", "recipe.patch_residual_role_scale.nas")


def test_join_role_receives_a_non_zero_patch_budget():
    """A zero NAS budget silences the patch expert on the whole Join track."""
    args = load_args_from_config(
        str(ROOT / "configs/ALFNet/rf_l3s.yaml"),
        overrides={"device": "cpu"},
    )
    model = ALFNet(args).eval()
    aas, nas, vs, fs = model.patch_budget.tolist()

    assert nas > 0.0
    assert (aas, fs) == (pytest.approx(0.20), pytest.approx(0.35))
    assert vs == 0.0  # a never-observed node still has no temporal evidence

    nodes, probe = 157, [0, 1, 2]
    beta = {}
    for name, role_index in (("NAS", 1), ("VS", 2)):
        x = torch.zeros(1, args.input_len, nodes, args.enc_in)
        x[..., 0] = torch.randn(1, args.input_len, nodes)
        x[..., 1] = 1.0  # everyone AAS
        x[..., 5] = 1.0  # state_age
        x[..., 6] = 1.0  # observation mask
        x[:, :, probe, 1:5] = 0.0
        x[:, :, probe, 1 + role_index] = 1.0
        t_mark = torch.zeros(1, args.input_len, 3)
        t_mark[..., 1] = 8.0
        with torch.no_grad():
            output = model(torch.eye(nodes), x, t_mark, None, None, t_mark, 0)
        beta[name] = (1.0 - output["route_routing"]["alpha"])[:, :, probe, :]

    assert beta["NAS"].min() > 0.0, "Join nodes must be able to use the patch expert"
    assert beta["NAS"].max() <= nas
    assert torch.equal(beta["VS"], torch.zeros_like(beta["VS"]))


def test_alftemporal_kdtree_layout_covers_every_node():
    positions = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [0.0, 0.1],
            [1.0, 0.0],
            [1.1, 0.0],
            [1.0, 0.1],
            [0.0, 1.0],
            [0.1, 1.0],
        ],
        dtype=float,
    )
    adjacency = np.eye(positions.shape[0], dtype=float)
    original, reordered, padded, part_count = _build_spatial_layout(
        positions,
        adjacency,
        recur_times=1,
        spa_patchsize=4,
    )

    assert part_count == 2
    assert sorted(original) == list(range(positions.shape[0]))
    assert len(reordered) == positions.shape[0]
    assert len(padded) == part_count * 4


def _alfgraph_args(**recipe_overrides):
    args = load_args_from_config(
        str(ROOT / "configs/ALFGraph/METRLA_alfgraph_sd.yaml"),
        overrides={"device": "cpu"},
    )
    args.model_family_cfg = copy.deepcopy(args.model_family_cfg)
    args.model_family_cfg["recipe"].update(recipe_overrides)
    return args


def _alfgraph_batch(args, nodes=5, role_index=0):
    x = torch.zeros(2, args.input_len, nodes, args.enc_in)
    x[..., 0] = torch.randn(2, args.input_len, nodes)
    x[..., 1 + role_index] = 1.0
    x[..., 5] = 1.0  # state_age
    x[..., 6] = 1.0  # observation mask
    return x, torch.randn(2, args.pred_len, nodes, args.c_out)


def test_alfgraph_preserves_final_residual_delta_contract():
    args = _alfgraph_args()
    model = ALFGraph(args)
    nodes = 5
    x, teacher = _alfgraph_batch(args, nodes)

    output = model(torch.eye(nodes), x, x_dec=teacher, epoch=40)

    assert output["pred"].shape == (2, args.pred_len, nodes, args.c_out)
    assert torch.isfinite(output["pred"]).all()


def test_alfgraph_keeps_the_observation_mask_channel():
    """After GRIN the mask is the only measured-vs-imputed signal the router has."""
    args = _alfgraph_args()
    model = ALFGraph(args)
    first_conv = model.encoder.cells[0].forget_gate

    assert model.state_dim == LIFECYCLE_STATE_CHANNELS
    assert first_conv.in_channels == args.enc_in + args.d_model


def test_alfgraph_routes_five_supports_with_top_k_two():
    args = _alfgraph_args()
    model = ALFGraph(args)
    routed = [m for m in model.modules() if isinstance(m, LifecycleRoutedGraphConv)]

    # encoder 2 layers + decoder 2 layers, three gates each.
    assert len(routed) == 12
    assert not [m for m in model.modules() if isinstance(m, DiffConv)]
    assert all(conv.num_experts == 5 and conv.router.k == 2 for conv in routed)

    supports = build_lifecycle_supports(
        torch.eye(4) + torch.ones(4, 4),
        torch.eye(4),
        torch.eye(4),
    )
    assert supports.shape == (len(SUPPORT_NAMES), 4, 4)
    assert torch.equal(supports[0], torch.eye(4))  # identity expert keeps self
    for support in supports:
        assert torch.allclose(support.sum(dim=1), torch.ones(4), atol=1e-5)


def test_lifecycle_router_activates_exactly_top_k_experts():
    router = LifecycleRouter(LIFECYCLE_STATE_CHANNELS, num_experts=5, k=2).eval()
    torch.manual_seed(0)
    router.w_gate.data.normal_(0.0, 1.0)
    state = torch.randn(3, 7, LIFECYCLE_STATE_CHANNELS)

    gates, aux_loss = router(state)

    assert gates.shape == (3, 7, 5)
    assert torch.all((gates > 0).sum(dim=-1) == 2)
    assert torch.allclose(gates.sum(dim=-1), torch.ones(3, 7), atol=1e-6)
    assert torch.isfinite(aux_loss) and aux_loss >= 0


def test_alfgraph_router_loss_reaches_the_alfnet_objective():
    args = _alfgraph_args()
    model = ALFGraph(args)
    nodes = 5
    x, teacher = _alfgraph_batch(args, nodes)

    output = model(torch.eye(nodes), x, x_dec=teacher, epoch=40)
    prediction, meta = unpack_model_output(output)
    total = sum_aux_losses(meta.get("aux_losses"), device=prediction.device)

    assert set(meta["aux_losses"]) == {"graph_router"}
    assert total is not None and torch.isfinite(total)

    total.backward()
    assert model.encoder.cells[0].forget_gate.router.w_gate.grad.abs().sum() > 0


def test_alfgraph_diffusion_control_restores_the_single_operator():
    """The parameter-matched control keeps DiffConv and emits no router loss."""
    args = _alfgraph_args(graph_conv="diffusion")
    model = ALFGraph(args)
    nodes = 5
    x, teacher = _alfgraph_batch(args, nodes)

    output = model(torch.eye(nodes), x, x_dec=teacher, epoch=40)

    assert len([m for m in model.modules() if isinstance(m, DiffConv)]) == 12
    assert not [m for m in model.modules() if isinstance(m, LifecycleRoutedGraphConv)]
    assert "aux_losses" not in output
    assert torch.isfinite(output["pred"]).all()


def test_grin_uses_explicit_mask_without_exogenous_duplication():
    config = SimpleNamespace(
        enc_in=7,
        raw_enc_in=1,
        d_model=8,
        lifecycle_observation_mask_feature=True,
    )
    recipe = {
        "grin_hidden_dim": 8,
        "grin_ff_dim": 8,
        "grin_kernel_size": 1,
        "grin_decoder_order": 1,
        "grin_use_extra_features": True,
        "grin_use_role_mask": True,
    }
    imputer = GRINInputImputer(config, recipe)
    x = torch.randn(2, 4, 5, 7)
    x[..., 1:5] = 0.0
    x[..., 1] = 1.0
    x[..., 6] = 1.0
    x[:, :, 0, 6] = 0.0

    observed = imputer._observed_mask(x[..., :1], x[..., 1:6], x_enc=x)
    output = imputer(torch.eye(5), x)

    assert imputer.bigril.fwd_rnn.u_size == 5
    assert observed[:, :, 0].sum() == 0
    assert observed[:, :, 1:].all()
    assert output.shape == x.shape
    assert torch.isfinite(output).all()

def test_crossformer_timefilter_block_preserves_patch_shape():
    block = WindowAttBlock(
        hidden_size=16,
        num_heads=2,
        num=4,
        size=3,
        crossformer_router_factor=2,
        timefilter_n_heads=2,
        timefilter_layers=1,
        timefilter_d_ff=32,
        timefilter_top_p=0.65,
        timefilter_alpha=0.1,
        timefilter_residual_scale=0.10,
    )
    x = torch.randn(2, 3, 12, 16)

    output = block(x)

    assert output.shape == x.shape
    assert torch.isfinite(output).all()


def test_timefilter_adapter_preserves_tensor_contract():
    mixer = TimeFilterPatchMixer(
        hidden_dim=16,
        patch_num=4,
        patch_size=3,
        n_heads=2,
        n_blocks=1,
        d_ff=32,
        top_p=0.65,
        alpha=0.1,
        residual_scale=0.10,
    )
    x = torch.randn(2, 3, 4, 3, 16)

    output = mixer(x)

    assert output.shape == x.shape
    assert torch.isfinite(output).all()
