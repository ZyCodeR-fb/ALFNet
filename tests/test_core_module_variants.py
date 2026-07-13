import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
_model_pkg = types.ModuleType("model")
_model_pkg.__path__ = [str(ROOT / "model")]
_alftemporal_pkg = types.ModuleType("model.ALFTemporal")
_alftemporal_pkg.__path__ = [str(ROOT / "model/ALFTemporal")]
_comparison_pkg = types.ModuleType("model.comparison_model")
_family_utils = types.ModuleType("model.comparison_model.family_utils")
_family_utils.build_patchstg_cfg = lambda configs: {}
sys.modules.setdefault("model", _model_pkg)
sys.modules.setdefault("model.ALFTemporal", _alftemporal_pkg)
sys.modules.setdefault("model.comparison_model", _comparison_pkg)
sys.modules.setdefault("model.comparison_model.family_utils", _family_utils)
_mogernn = _load_module("alfgraph_for_core_variant_test", "model/ALFGraph/ALFGraph.py")
_alftemporal = _load_module("model.ALFTemporal.ALFTemporal", "model/ALFTemporal/ALFTemporal.py")
_patchstg = _load_module("alftemporal_for_core_variant_test", "model/ALFTemporal/patchstg.py")
_source_grin = _load_module("source_grin_for_core_variant_test", "layers/source_grin.py")
_source_graph_unet = _load_module("source_graph_unet_for_core_variant_test", "layers/source_graph_unet.py")
_source_token_learner = _load_module("source_token_learner_for_core_variant_test", "layers/source_token_learner.py")
_source_set_transformer = _load_module("source_set_transformer_for_core_variant_test", "layers/source_set_transformer.py")
_source_pattern_bank = _load_module("source_pattern_bank_for_core_variant_test", "layers/source_pattern_bank.py")
_alfnet_recovery = _load_module("alfnet_recovery_for_core_variant_test", "model/ALFNet/recovery.py")
_alfnet_pkg = types.ModuleType("model.ALFNet")
_alfnet_pkg.__path__ = [str(ROOT / "model/ALFNet")]
sys.modules.setdefault("model.ALFNet", _alfnet_pkg)
sys.modules.setdefault("model.ALFNet.recovery", _alfnet_recovery)
_args_stub = types.ModuleType("args")
_args_stub.load_args_from_config = lambda *args, **kwargs: None
sys.modules.setdefault("args", _args_stub)
_stbp_pkg = types.ModuleType("model.comparison_model.STBP")
_stbp_module = types.ModuleType("model.comparison_model.STBP.stbp")
_stbp_module.STBP_Model = nn.Identity
sys.modules.setdefault("model.comparison_model.STBP", _stbp_pkg)
sys.modules.setdefault("model.comparison_model.STBP.stbp", _stbp_module)
_model_output = types.ModuleType("utils.model_output")
_model_output.unpack_model_output = lambda output: (output, {})
sys.modules.setdefault("utils.model_output", _model_output)
_alfnet = _load_module("alfnet_for_shared_input_test", "model/ALFNet/ALFNet.py")
_build_recurrent_cell = _mogernn._build_recurrent_cell
WindowAttBlock = _patchstg.WindowAttBlock
PatchRevIN = _patchstg.PatchRevIN
GRINInputImputer = _source_grin.GRINInputImputer
build_sensor_state_context = _alfnet_recovery.build_sensor_state_context
graph_unet_parts = _source_graph_unet.graph_unet_parts
TokenLearnerPatchMixer = _source_token_learner.TokenLearnerPatchMixer
SetTransformerPatchMixer = _source_set_transformer.SetTransformerPatchMixer
PatternBankPatchMixer = _source_pattern_bank.PatternBankPatchMixer


def test_alftemporal_new_spatial_layouts_cover_nodes_with_patch_capacity():
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
            [0.0, 1.1],
            [1.0, 1.0],
            [1.1, 1.0],
            [1.0, 1.1],
        ],
        dtype=float,
    )
    adj = np.eye(positions.shape[0], dtype=float)
    for start in (0, 3, 6, 9):
        for i in range(start, start + 3):
            for j in range(start, start + 3):
                if i != j:
                    adj[i, j] = 1.0

    for layout in ("morton", "fps_anchor", "heavy_edge"):
        ori_idx, reo_idx, all_idx, part_count = _alftemporal._build_spatial_layout(
            positions,
            adj,
            recur_times=2,
            spa_patchsize=4,
            layout_type=layout,
        )

        assert part_count == 4
        assert sorted(ori_idx) == list(range(positions.shape[0]))
        assert len(all_idx) == part_count * 4
        assert len(reo_idx) == positions.shape[0]
        assert min(all_idx) >= 0
        assert max(all_idx) < positions.shape[0]
        assert max(reo_idx) < len(all_idx)


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
    imputer = GRINInputImputer(cfg, recipe)
    x = torch.randn(2, 4, 5, 6)
    x[:, :, 0, 0] = 0.0
    roles = torch.zeros(2, 4, 5, 4)
    roles[..., 1] = 1.0
    x[..., 1:5] = roles
    adj = torch.eye(5)

    out = imputer(adj, x)

    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_sensor_state_context_soft_split_preserves_probability_mass():
    # Legacy order: AAS, NAS, VS, FS. Sensor-state order:
    # Cold-start, Warm-up, Adapted, Degrading, Outage, Steady.
    last_role = torch.eye(4).view(1, 4, 4)
    visibility = torch.tensor([[[0.75], [0.25], [0.0], [0.60]]])

    state = build_sensor_state_context(last_role, visibility)

    expected = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.75, 0.25, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.60, 0.40, 0.0],
        ]
    ).view(1, 4, 6)
    assert state.shape == (1, 4, 6)
    assert torch.allclose(state, expected)
    assert torch.allclose(state.sum(dim=-1), torch.ones(1, 4))


def test_alfgraph_preprocessed_input_skips_data_preparation():
    class CountingPreprocessor(nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, adj, x_enc, x_t_mark=None, pos_mark=None):
            self.calls += 1
            return x_enc + 1.0

    class IdentitySeq2Seq(nn.Module):
        def forward(self, x_enc, x_dec, A_q, A_h, initial_input, teacher_forcing_ratio):
            return x_enc

    model = _mogernn.Model.__new__(_mogernn.Model)
    nn.Module.__init__(model)
    model.data_pre = CountingPreprocessor()
    model.raw_enc_in = 1
    model.output_dim = 1
    model.temporal_input_mixer = None
    model.adaptive_support = None
    model.current_node_ids = None
    model.seq2seq = IdentitySeq2Seq()
    model.lifecycle_output_adapter = None
    model.role_output_calibrator = None
    adj = torch.eye(3)
    x = torch.randn(2, 4, 3, 6)
    x_dec = torch.zeros(2, 2, 3, 1)

    prepared = model.preprocess_input(adj, x)
    out = model(adj, prepared, None, None, x_dec, None, 0, input_is_preprocessed=True)

    assert model.data_pre.calls == 1
    assert torch.allclose(out, prepared)


class _FakeGraphExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.preprocess_calls = 0
        self.forward_inputs = []
        self.forward_preprocessed = []

    def preprocess_input(self, adj, x_enc, x_t_mark=None, pos_mark=None):
        self.preprocess_calls += 1
        out = x_enc.clone()
        out[..., :1] = out[..., :1] + 5.0
        return out

    def set_current_node_ids(self, node_ids):
        return None

    def forward(self, adj, x_enc, x_t_mark, pos_mark, x_dec, dec_t_mark, epoch, input_is_preprocessed=False):
        self.forward_inputs.append(x_enc.detach().clone())
        self.forward_preprocessed.append(input_is_preprocessed)
        return x_enc[:, :2, :, :1] * self.scale


class _FakePatchExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.forward_inputs = []

    def forward(self, adj, x_enc, x_t_mark, pos_mark, x_dec, dec_t_mark, epoch):
        self.forward_inputs.append(x_enc.detach().clone())
        return x_enc[:, :2, :, :1] * self.scale


def _build_alfnet_for_input_mode(monkeypatch, recipe_overrides=None):
    graph = _FakeGraphExpert()
    patch = _FakePatchExpert()
    expert_args = [
        (graph, types.SimpleNamespace(enc_in=6), None),
        (patch, types.SimpleNamespace(enc_in=1), None),
    ]
    monkeypatch.setattr(_alfnet, "_load_expert_from_spec", lambda *args, **kwargs: expert_args.pop(0))
    monkeypatch.setattr(_alfnet, "load_project_adjacency", lambda *args, **kwargs: np.eye(3, dtype=np.float32))
    monkeypatch.setattr(_alfnet, "resolve_known_unknown_nodes", lambda *args, **kwargs: (list(range(3)), []))
    recipe = {
        "end_to_end": True,
        "expert_train_mode": True,
        "fusion_mode": "graph_patch_residual",
        "patch_residual_mode": "gap",
        "use_role_residual": False,
    }
    recipe.update(recipe_overrides or {})
    configs = types.SimpleNamespace(
        lifecycle_role_features=True,
        lifecycle_history_age_feature=True,
        model_family_cfg={
            "experts": {
                "graph": {"model": "ALFGraph", "config_path": "unused"},
                "patch": {"model": "ALFTemporal", "config_path": "unused"},
            },
            "model": {"state_hidden_dim": 8, "state_dim": 8, "gate_hidden_dim": 8},
            "recipe": recipe,
        },
        model_recipe="testam_role_mmoe",
        device="cpu",
        raw_enc_in=1,
        c_out=1,
        pred_len=2,
        input_len=4,
        d_model=8,
        data_path="unused",
        unknown_nodes_path="",
        unknown_nodes_list=[],
        unknown_nodes_num=0,
        seed=0,
    )
    return _alfnet.Model(configs), graph, patch


def test_alfnet_new_input_modes_are_opt_in(monkeypatch):
    legacy, _, _ = _build_alfnet_for_input_mode(monkeypatch)
    sensor6, _, _ = _build_alfnet_for_input_mode(
        monkeypatch,
        {"lifecycle_state_mode": "sensor_state_6"},
    )

    assert legacy.state_encoder[0].in_features == 6
    assert len(legacy.role_fusion_gates) == 4
    assert sensor6.state_encoder[0].in_features == 8
    assert len(sensor6.role_fusion_gates) == 6


def test_alfnet_shared_input_recovery_is_yaml_controlled(monkeypatch):
    legacy, legacy_graph, legacy_patch = _build_alfnet_for_input_mode(monkeypatch)
    shared, shared_graph, shared_patch = _build_alfnet_for_input_mode(
        monkeypatch,
        {"shared_input_recovery": "graph_preprocessor"},
    )
    adj = torch.eye(3)
    raw = torch.zeros(2, 4, 3, 1)
    roles = torch.zeros(2, 4, 3, 4)
    roles[..., 0] = 1.0
    age = torch.ones(2, 4, 3, 1)
    x_enc = torch.cat([raw, roles, age], dim=-1)
    x_dec = torch.zeros(2, 2, 3, 1)

    legacy._run_experts(adj, x_enc, None, None, x_dec, None, 0)
    shared._run_experts(adj, x_enc, None, None, x_dec, None, 0)

    assert legacy_graph.preprocess_calls == 0
    assert torch.allclose(legacy_patch.forward_inputs[0], raw)
    assert shared_graph.preprocess_calls == 1
    assert shared_graph.forward_preprocessed == [True]
    assert torch.allclose(shared_patch.forward_inputs[0], raw + 5.0)


def test_graph_unet_parts_cover_all_nodes():
    adj = torch.eye(8).numpy()
    adj[0, 1] = adj[1, 0] = 1.0
    adj[2, 3] = adj[3, 2] = 1.0

    parts = graph_unet_parts(adj, recur_times=2)
    covered = sorted(int(node) for part in parts for node in part.tolist())

    assert covered == list(range(8))
    assert len(parts) == 4


def test_patchstg_timefilter_window_block_preserves_shape():
    block = WindowAttBlock(
        hidden_size=16,
        num_heads=2,
        num=4,
        size=3,
        attention_type="crossformer_tsa",
        crossformer_router_factor=2,
        use_timefilter_patch_mixer=True,
        timefilter_n_heads=2,
        timefilter_layers=1,
        timefilter_residual_scale=0.05,
    )
    x = torch.randn(2, 3, 12, 16)

    out = block(x)

    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_patchstg_duet_window_block_preserves_shape():
    block = WindowAttBlock(
        hidden_size=16,
        num_heads=2,
        num=4,
        size=3,
        attention_type="crossformer_tsa",
        crossformer_router_factor=2,
        use_duet_patch_router=True,
        duet_num_experts=2,
        duet_k=1,
        duet_hidden_size=8,
        duet_residual_scale=0.05,
    )
    x = torch.randn(2, 3, 12, 16)

    out = block(x)

    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_patchstg_token_learner_window_block_preserves_shape():
    block = WindowAttBlock(
        hidden_size=16,
        num_heads=2,
        num=4,
        size=3,
        attention_type="crossformer_tsa",
        crossformer_router_factor=2,
        use_token_learner_patch_mixer=True,
        token_learner_num_tokens=4,
        token_learner_bottleneck_dim=8,
        token_learner_residual_scale=0.05,
    )
    x = torch.randn(2, 3, 12, 16)

    out = block(x)

    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_patchstg_set_transformer_window_block_preserves_shape():
    block = WindowAttBlock(
        hidden_size=16,
        num_heads=2,
        num=4,
        size=3,
        attention_type="crossformer_tsa",
        crossformer_router_factor=2,
        use_set_transformer_patch_mixer=True,
        set_transformer_num_inds=4,
        set_transformer_num_heads=2,
        set_transformer_residual_scale=0.05,
    )
    x = torch.randn(2, 3, 12, 16)

    out = block(x)

    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_patchstg_pattern_bank_window_block_preserves_shape():
    block = WindowAttBlock(
        hidden_size=16,
        num_heads=2,
        num=4,
        size=3,
        attention_type="crossformer_tsa",
        crossformer_router_factor=2,
        use_pattern_bank_patch_mixer=True,
        pattern_bank_num_patterns=6,
        pattern_bank_residual_scale=0.05,
    )
    x = torch.randn(2, 3, 12, 16)

    out = block(x)

    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_token_learner_patch_mixer_preserves_shape_and_starts_near_identity():
    mixer = TokenLearnerPatchMixer(
        hidden_dim=16,
        num_tokens=4,
        bottleneck_dim=8,
        residual_scale=0.05,
    )
    x = torch.randn(2, 3, 4, 5, 16)

    out = mixer(x)

    assert out.shape == x.shape
    assert torch.isfinite(out).all()
    assert torch.allclose(out, x, atol=1e-6, rtol=1e-6)


def test_set_transformer_patch_mixer_preserves_shape_and_starts_near_identity():
    mixer = SetTransformerPatchMixer(
        hidden_dim=16,
        num_heads=2,
        num_inds=4,
        residual_scale=0.05,
    )
    x = torch.randn(2, 3, 4, 5, 16)

    out = mixer(x)

    assert out.shape == x.shape
    assert torch.isfinite(out).all()
    assert torch.allclose(out, x, atol=1e-6, rtol=1e-6)


def test_pattern_bank_patch_mixer_preserves_shape_and_starts_near_identity():
    mixer = PatternBankPatchMixer(
        hidden_dim=16,
        num_patterns=6,
        residual_scale=0.05,
    )
    x = torch.randn(2, 3, 4, 5, 16)

    out = mixer(x)

    assert out.shape == x.shape
    assert torch.isfinite(out).all()
    assert torch.allclose(out, x, atol=1e-6, rtol=1e-6)
