from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml

from args import load_args_from_config
from layers.source_grin import GRINInputImputer
from layers.source_timefilter import TimeFilterPatchMixer
from model.ALFGraph.ALFGraph import Model as ALFGraph
from model.ALFNet.ALFNet import Model as ALFNet
from model.ALFTemporal.ALFTemporal import _build_spatial_layout
from model.ALFTemporal.patchstg import WindowAttBlock


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
        "rf_l2s.yaml"
    ]
    config = _read_yaml("configs/ALFNet/rf_l2s.yaml")
    assert config["runtime"]["model"] == "ALFNet"
    assert config["runtime"]["model_variant"] == "rf_l2s"
    assert config["lifecycle"]["lifecycle_training_strategy"] == "lcbt_v2"
    assert config["async_test"]["lce_protocol_version"] == 1
    assert config["async_test"]["ples_protocol_version"] == 2
    assert set(config["family"]["experts"]) == {"graph", "patch"}

    for model in COMPARISON_MODELS:
        assert [
            path.name for path in (ROOT / "configs" / model).glob("*.yaml")
        ] == ["METRLA_lcbt_v2.yaml"]


def test_alfnet_contains_one_shared_grin_and_no_graph_local_grin():
    args = load_args_from_config(
        str(ROOT / "configs/ALFNet/rf_l2s.yaml"),
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
    assert sum(parameter.numel() for parameter in model.parameters()) == 3_288_746
    assert not hasattr(model, "fusion_gate")
    assert all(
        not hasattr(block, "nattn") and not hasattr(block, "sattn")
        for block in model.patch_expert.backbone.spa_encoder
    )
    assert model.graph_expert.decoder.scale == 0.3


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


def test_alfgraph_preserves_final_residual_delta_contract():
    args = load_args_from_config(
        str(ROOT / "configs/ALFGraph/METRLA_alfgraph_sd.yaml"),
        overrides={"device": "cpu"},
    )
    model = ALFGraph(args)
    batch, steps, nodes = 2, args.input_len, 5
    x = torch.randn(batch, steps, nodes, args.enc_in)
    x[..., -1] = 1.0
    teacher = torch.randn(batch, args.pred_len, nodes, args.c_out)

    output = model(torch.eye(nodes), x, x_dec=teacher, epoch=40)

    assert output.shape == (batch, args.pred_len, nodes, args.c_out)
    assert torch.isfinite(output).all()


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
