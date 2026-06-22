import importlib.util
import sys
import types
from pathlib import Path

import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]


class _StubAttention(nn.Module):
    def __init__(self, dim, num_heads=1, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        return self.proj(x)


class _StubMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        hidden_features = hidden_features or in_features
        self.net = nn.Sequential(nn.Linear(in_features, hidden_features), act_layer(), nn.Linear(hidden_features, in_features))

    def forward(self, x):
        return self.net(x)


def _install_timm_stub():
    timm = types.ModuleType("timm")
    models = types.ModuleType("timm.models")
    vision_transformer = types.ModuleType("timm.models.vision_transformer")
    vision_transformer.Attention = _StubAttention
    vision_transformer.Mlp = _StubMlp
    timm.models = models
    models.vision_transformer = vision_transformer
    sys.modules.setdefault("timm", timm)
    sys.modules.setdefault("timm.models", models)
    sys.modules.setdefault("timm.models.vision_transformer", vision_transformer)


def _load_module(module_name, relative_path):
    spec = importlib.util.spec_from_file_location(module_name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_install_timm_stub()
_mogernn = _load_module("mogernn_for_core_variant_test", "model/MoGERNN.py")
_patchstg = _load_module("patchstg_for_core_variant_test", "model/PatchSTG/patchstg.py")
_build_recurrent_cell = _mogernn._build_recurrent_cell
WindowAttBlock = _patchstg.WindowAttBlock
PatchRevIN = _patchstg.PatchRevIN


def test_state_decay_dcrnn_cell_preserves_hidden_shape():
    cell = _build_recurrent_cell("state_decay_dcrnn_s010_g030", input_size=6, hidden_size=8, k=1)
    x = torch.randn(2, 5, 6)
    h = torch.randn(2, 5, 8)
    adj = torch.eye(5)

    out = cell(x, h, adj, adj)

    assert out.shape == h.shape
    assert torch.isfinite(out).all()
    assert cell.max_decay_scale == 0.10


def test_patchstg_crossformer_tsa_window_block_preserves_shape():
    block = WindowAttBlock(
        hidden_size=16,
        num_heads=2,
        num=4,
        size=3,
        attention_type="crossformer_tsa",
        crossformer_router_factor=2,
    )
    x = torch.randn(2, 3, 12, 16)

    out = block(x)

    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_patch_revin_round_trip_preserves_values():
    revin = PatchRevIN(num_features=1, affine=True)
    x = torch.randn(3, 12, 5, 1) * 7.0 + 40.0

    out = revin(revin(x, "norm"), "denorm")

    assert out.shape == x.shape
    assert torch.allclose(out, x, atol=1e-4, rtol=1e-4)
