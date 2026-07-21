import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
import yaml


ROOT = Path(__file__).resolve().parents[1]


class _StubAttention(nn.Module):
    """Parameter-compatible stand-in for timm Attention in local unit tests."""

    def __init__(
        self,
        dim,
        num_heads=1,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        return self.proj(self.qkv(x).chunk(3, dim=-1)[2])


class _StubMlp(nn.Module):

    def __init__(
        self,
        in_features,
        hidden_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, in_features)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


def _install_timm_stub():
    timm = types.ModuleType("timm")
    models = types.ModuleType("timm.models")
    vision_transformer = types.ModuleType("timm.models.vision_transformer")
    vision_transformer.Attention = _StubAttention
    vision_transformer.Mlp = _StubMlp
    timm.models = models
    models.vision_transformer = vision_transformer
    sys.modules["timm"] = timm
    sys.modules["timm.models"] = models
    sys.modules["timm.models.vision_transformer"] = vision_transformer


def _load_module(module_name, relative_path):
    spec = importlib.util.spec_from_file_location(module_name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_stubbed_timm_modules = (
    "timm",
    "timm.models",
    "timm.models.vision_transformer",
)
_saved_timm_modules = {
    name: sys.modules.get(name) for name in _stubbed_timm_modules
}
_install_timm_stub()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_stubbed_model_modules = (
    "model",
    "model.ALFTemporal",
    "model.ALFTemporal.ALFTemporal",
    "model.ALFTemporal.patchstg",
    "model.comparison_model",
    "model.comparison_model.family_utils",
)
_saved_model_modules = {
    name: sys.modules.get(name) for name in _stubbed_model_modules
}
_model_pkg = types.ModuleType("model")
_model_pkg.__path__ = [str(ROOT / "model")]
_alftemporal_pkg = types.ModuleType("model.ALFTemporal")
_alftemporal_pkg.__path__ = [str(ROOT / "model" / "ALFTemporal")]
_comparison_pkg = types.ModuleType("model.comparison_model")
_comparison_pkg.__path__ = [str(ROOT / "model" / "comparison_model")]
_family_utils = types.ModuleType("model.comparison_model.family_utils")
_family_utils.build_patchstg_cfg = lambda configs: {}
sys.modules["model"] = _model_pkg
sys.modules["model.ALFTemporal"] = _alftemporal_pkg
sys.modules["model.comparison_model"] = _comparison_pkg
sys.modules["model.comparison_model.family_utils"] = _family_utils

_alfgraph = _load_module(
    "alfgraph_for_final_contract_test",
    "model/ALFGraph/ALFGraph.py",
)
_alftemporal = _load_module(
    "model.ALFTemporal.ALFTemporal",
    "model/ALFTemporal/ALFTemporal.py",
)
_patchstg = _load_module(
    "alftemporal_patchstg_for_final_contract_test",
    "model/ALFTemporal/patchstg.py",
)
_source_grin = _load_module(
    "source_grin_for_final_contract_test",
    "layers/source_grin.py",
)
_source_timefilter = _load_module(
    "source_timefilter_for_final_contract_test",
    "layers/source_timefilter.py",
)

for _name, _module in _saved_model_modules.items():
    if _module is None:
        sys.modules.pop(_name, None)
    else:
        sys.modules[_name] = _module
for _name, _module in _saved_timm_modules.items():
    if _module is None:
        sys.modules.pop(_name, None)
    else:
        sys.modules[_name] = _module


def _read_yaml(relative_path):
    with (ROOT / relative_path).open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def test_final_yaml_surface_preserves_rf_alf_sd_and_adds_lcbt_ablations():
    assert sorted(
        path.name for path in (ROOT / "configs/ALFNet").glob("*.yaml")
    ) == [
        "rf_alf_sd.yaml",
        "rf_lcb2_g.yaml",
        "rf_lcb2_s.yaml",
        "rf_lcb_g.yaml",
        "rf_lcb_s.yaml",
    ]
    assert [path.name for path in (ROOT / "configs/ALFGraph").glob("*.yaml")] == [
        "METRLA_alfgraph_sd.yaml"
    ]
    assert [path.name for path in (ROOT / "configs/ALFTemporal").glob("*.yaml")] == [
        "METRLA_alftemporal_t65.yaml"
    ]

    alfnet = _read_yaml("configs/ALFNet/rf_alf_sd.yaml")
    graph_lcbt = _read_yaml("configs/ALFNet/rf_lcb_g.yaml")
    shared_lcbt = _read_yaml("configs/ALFNet/rf_lcb_s.yaml")
    graph_lcbt_v2 = _read_yaml("configs/ALFNet/rf_lcb2_g.yaml")
    shared_lcbt_v2 = _read_yaml("configs/ALFNet/rf_lcb2_s.yaml")
    graph = _read_yaml("configs/ALFGraph/METRLA_alfgraph_sd.yaml")
    temporal = _read_yaml("configs/ALFTemporal/METRLA_alftemporal_t65.yaml")

    assert alfnet["runtime"]["model"] == "ALFNet"
    assert alfnet["runtime"]["model_variant"] == "rf_alf_sd"
    assert alfnet["family"]["experts"]["graph"]["model"] == "ALFGraph"
    assert alfnet["family"]["experts"]["patch"]["model"] == "ALFTemporal"
    assert alfnet["family"]["recipe"]["fusion_mode"] == "graph_patch_residual"
    assert graph["family"]["recipe"]["use_grin_input_imputer"] is True
    assert graph["family"]["recipe"]["cell_type"] == "dcrnn"
    assert graph["family"]["recipe"]["decoder_output_mode"] == "residual_delta"
    assert temporal["family"]["model"]["attention_type"] == "crossformer_tsa"
    assert temporal["family"]["model"]["use_timefilter_patch_mixer"] is True
    assert temporal["family"]["model"]["timefilter_top_p"] == 0.65
    assert "lifecycle_observation_mask_feature" not in alfnet["lifecycle"]
    assert graph_lcbt["family"]["recipe"]["input_imputation_scope"] == "graph"
    assert graph_lcbt["family"]["experts"]["graph"]["overrides"] == {
        "lifecycle_observation_mask_feature": True
    }
    assert shared_lcbt["family"]["recipe"]["input_imputation_scope"] == "shared"
    assert shared_lcbt["family"]["experts"]["graph"]["family_overrides"] == {
        "recipe": {"use_grin_input_imputer": False}
    }
    assert graph_lcbt_v2["lifecycle"]["lifecycle_training_strategy"] == "lcbt_v2"
    assert graph_lcbt_v2["family"]["recipe"]["input_imputation_scope"] == "graph"
    assert shared_lcbt_v2["lifecycle"]["lifecycle_training_strategy"] == "lcbt_v2"
    assert shared_lcbt_v2["family"]["recipe"]["input_imputation_scope"] == "shared"


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
    adj = np.eye(positions.shape[0], dtype=float)

    ori_idx, reo_idx, all_idx, part_count = _alftemporal._build_spatial_layout(
        positions,
        adj,
        recur_times=1,
        spa_patchsize=4,
        layout_type="kd_tree",
    )

    assert part_count == 2
    assert sorted(ori_idx) == list(range(positions.shape[0]))
    assert len(reo_idx) == positions.shape[0]
    assert len(all_idx) == part_count * 4


def test_alftemporal_rejects_removed_spatial_layouts():
    with pytest.raises(ValueError, match="only retains the final KDTree"):
        _alftemporal._build_spatial_layout(
            np.zeros((4, 2), dtype=float),
            np.eye(4, dtype=float),
            recur_times=1,
            spa_patchsize=2,
            layout_type="morton",
        )


def test_final_dcrnn_cell_preserves_hidden_shape():
    cell = _alfgraph._build_recurrent_cell(
        "dcrnn",
        input_size=6,
        hidden_size=8,
        k=1,
    )
    x = torch.randn(2, 5, 6)
    h = torch.randn(2, 5, 8)
    adj = torch.eye(5)

    out = cell(x, h, adj, adj)

    assert out.shape == h.shape
    assert torch.isfinite(out).all()


def test_grin_input_imputer_preserves_lifecycle_shape():
    cfg = types.SimpleNamespace(enc_in=6, raw_enc_in=1, d_model=8)
    recipe = {
        "grin_hidden_dim": 8,
        "grin_ff_dim": 8,
        "grin_kernel_size": 1,
        "grin_decoder_order": 1,
        "grin_use_extra_features": True,
        "grin_use_role_mask": True,
    }
    imputer = _source_grin.GRINInputImputer(cfg, recipe)
    x = torch.randn(2, 4, 5, 6)
    x[:, :, 0, 0] = 0.0
    roles = torch.zeros(2, 4, 5, 4)
    roles[..., 1] = 1.0
    x[..., 1:5] = roles

    out = imputer(torch.eye(5), x)

    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_grin_input_imputer_uses_explicit_mask_without_exogenous_duplication():
    cfg = types.SimpleNamespace(
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
    imputer = _source_grin.GRINInputImputer(cfg, recipe)
    x = torch.randn(2, 4, 5, 7)
    x[..., 1:5] = 0.0
    x[..., 1] = 1.0
    x[..., 6] = 1.0
    x[:, :, 0, 6] = 0.0

    explicit_mask = imputer._observed_mask(
        x[..., :1],
        x[..., 1:6],
        x_enc=x,
    )
    out = imputer(torch.eye(5), x)

    assert imputer.bigril.fwd_rnn.u_size == 5
    assert explicit_mask[:, :, 0].sum() == 0
    assert explicit_mask[:, :, 1:].all()
    assert out.shape == x.shape
    assert imputer.without_observation_mask(out).shape[-1] == 6
    assert torch.isfinite(out).all()


def test_alfgraph_can_disable_internal_grin_for_shared_scope():
    cfg = types.SimpleNamespace(
        pred_len=3,
        look_back=0,
        c_out=1,
        enc_in=6,
        raw_enc_in=1,
        d_model=8,
        e_layers=1,
        k_hop=1,
        lifecycle_observation_mask_feature=False,
        model_family_cfg={
            "recipe": {
                "use_grin_input_imputer": False,
                "decoder_output_mode": "residual_delta",
                "decoder_residual_scale": 0.3,
                "cell_type": "dcrnn",
            }
        },
    )

    model = _alfgraph.Model(cfg)

    assert model.data_pre is None
    assert not any(
        isinstance(module, _source_grin.GRINInputImputer)
        for module in model.modules()
    )


def test_crossformer_timefilter_block_preserves_patch_shape():
    block = _patchstg.WindowAttBlock(
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

    out = block(x)

    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_timefilter_adapter_preserves_tensor_contract():
    mixer = _source_timefilter.TimeFilterPatchMixer(
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

    out = mixer(x)

    assert out.shape == x.shape
    assert torch.isfinite(out).all()
