import importlib
import glob
import os
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from args import load_args_from_config
from lib.utils import load_project_adjacency, resolve_known_unknown_nodes
from model.RoleFusion.recovery import GraphInputRecoveryAdapter
from model.STBP.stbp import STBP_Model
from utils.model_output import unpack_model_output


ROLE_AAS = 0
ROLE_NAS = 1
ROLE_VS = 2
ROLE_FS = 3
ROLE_COUNT = 4


class linear2d(nn.Module):
    # Copied from references/GCN-M/models/memoryModule.py, with no logic change.
    def __init__(self, c_in, c_out):
        super(linear2d, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)

    def forward(self, x):
        return self.mlp(x)


class linear3d(nn.Module):
    # Copied from references/GCN-M/models/memoryModule.py, with no logic change.
    def __init__(self, c_in, c_out):
        super(linear3d, self).__init__()
        self.mlp = torch.nn.Conv3d(c_in, c_out, kernel_size=(1, 1, 1), padding=(0, 0, 0), stride=(1, 1, 1), bias=True)

    def forward(self, x):
        return self.mlp(x)


class MemoryModule(nn.Module):
    # Copied from references/GCN-M/models/memoryModule.py, with shape comments preserved.
    def __init__(self, c_in, c_out):
        # c_in: local features z, by default: c_in = 1
        # c_out: output channel, by default: c_out=32, same to the "start_conv" in the main model
        super(MemoryModule, self).__init__()
        # embeddings for Query, Input Memory, Output Memory
        self.mlp_q = linear2d(c_in, c_out)
        self.mlp_m = linear3d(c_in, c_out)
        self.mlp_c = linear3d(c_in, c_out)

    def forward(self, x_local, x_hist):
        """
            ccompute the weighted sum from historical samples
        :param x_local: (B, L, D, F), (batch_size, L, nbr_node, nbr_feature), nbr_feature=1 or 3
        :param x_hist: (B, L, n*tau, D, F), (B, L, n*tau, nbr_node, nbr_feature), nbr_feature=1 or 3
        :return: e: (B, L, 1, D) (batch_size, L, nbr_feature, nbr_node), enriched traffic embeddings
        """

        # x_local: (B, L, D, F) -> (B, F, L, D)
        x_local = x_local.permute((0, 3, 1, 2)).contiguous()
        # x_hist: (B, L, n*tau, D, F) -> (B, F, n*tau, L, D)
        x_hist = x_hist.permute((0, 4, 2, 1, 3)).contiguous()
        q = self.mlp_q(x_local) #(B, F, L, D) -> (B, F', L, D)
        q = torch.unsqueeze(q, dim=-1) #(B, F', L, D) -> (B, F', L, D, 1)

        m = self.mlp_m(x_hist) #(B, F, n*tau, L, D) -> (B, F', n*tau, L, D)
        c = self.mlp_c(x_hist) #(B, F, n*tau, L, D) -> (B, F', n*tau, L, D)
        m = m.permute((0, 1, 3, 2, 4)).contiguous()  # (B, F', n*tau, L, D) -> (B, F', L, n*tau, D)
        c = c.permute((0, 1, 3, 2, 4)).contiguous()  # (B, F', n*tau, L, D) -> (B, F', L, n*tau, D)

        mq = torch.matmul(m, q) #(B, F', L, n*tau, 1)
        mq = torch.squeeze(mq, dim=-1) #(B, F', L, n*tau)
        q = torch.squeeze(q, dim=-1)  # (B, F', L, D)
        attention_scores = F.softmax(F.relu(mq), dim=-1) ##(B, F', L, n*tau)

        o = torch.einsum('bflt,bfltd->bfld', (attention_scores, c)).contiguous() #(B, F', L, D)
        e = q.add(o)

        return e #(B, F', L, D)


class GCNMMemoryRecoveryHead(nn.Module):
    def __init__(self, output_dim, hidden_dim, scale):
        super().__init__()
        self.memory_model = MemoryModule(output_dim, hidden_dim)
        self.proj = nn.Conv2d(hidden_dim, output_dim, kernel_size=(1, 1))
        self.scale = float(scale)

    def forward(self, pred, raw_history, last_role):
        x_local = pred
        x_hist = raw_history.unsqueeze(1).expand(-1, pred.size(1), -1, -1, -1)
        e = self.memory_model(x_local, x_hist)
        residual = self.proj(F.relu(e)).permute(0, 2, 3, 1).contiguous()
        fs_mask = last_role[..., ROLE_FS:ROLE_FS + 1].unsqueeze(1)
        residual = fs_mask * torch.tanh(residual)
        return pred + self.scale * residual, residual


def _fc_block(in_dim, units, activations):
    layers = []
    current_dim = in_dim
    for unit, activation in zip(units, activations):
        layers.append(nn.Linear(current_dim, unit))
        if activation == "relu":
            layers.append(nn.ReLU())
        elif activation == "tanh":
            layers.append(nn.Tanh())
        elif activation is not None:
            raise ValueError(f"Unsupported activation in INCREASE-style FC: {activation}")
        current_dim = unit
    return nn.Sequential(*layers)


class INCREASESpatialTransferHead(nn.Module):
    def __init__(self, output_dim, hidden_dim, pred_len, scale):
        super().__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.pred_len = pred_len
        self.scale = float(scale)

        # PyTorch port of references/INCREASE/METR-LA/INCREASE/model.py::model.
        self.x_gp_fw_in = _fc_block(output_dim, [hidden_dim, hidden_dim], ["relu", None])
        self.x_gp_bw_in = _fc_block(output_dim, [hidden_dim, hidden_dim], ["relu", None])
        self.y_gp_fw_fc = _fc_block(hidden_dim, [hidden_dim, hidden_dim], ["relu", None])
        self.y_gp_bw_fc = _fc_block(hidden_dim, [hidden_dim, hidden_dim], ["relu", None])
        self.x_gp_fw_fc = _fc_block(hidden_dim, [hidden_dim, hidden_dim], ["relu", None])
        self.x_gp_bw_fc = _fc_block(hidden_dim, [hidden_dim, hidden_dim], ["relu", None])
        self.x_gp_fw_gate = _fc_block(hidden_dim, [hidden_dim, hidden_dim], ["relu", "tanh"])
        self.x_gp_bw_gate = _fc_block(hidden_dim, [hidden_dim, hidden_dim], ["relu", "tanh"])
        self.y_gp_fw_proj = _fc_block(hidden_dim, [hidden_dim], [None])
        self.y_gp_bw_proj = _fc_block(hidden_dim, [hidden_dim], [None])
        self.y_fc = _fc_block(2 * hidden_dim, [hidden_dim, hidden_dim], ["relu", None])
        self.x_fc = _fc_block(2 * hidden_dim, [hidden_dim, hidden_dim], ["relu", None])
        self.g1 = _fc_block(hidden_dim, [hidden_dim, hidden_dim], ["relu", "relu"])
        self.te = nn.Embedding(pred_len, hidden_dim)
        self.cell = nn.GRUCell(2 * hidden_dim, hidden_dim)
        self.g2_state = nn.Linear(hidden_dim, hidden_dim)
        self.g2_x = nn.Linear(hidden_dim, hidden_dim)
        self.out = _fc_block(hidden_dim, [hidden_dim, hidden_dim, output_dim], ["relu", "relu", None])

    def _support(self, adj):
        adj = adj.to(dtype=torch.float32)
        gp_fw = adj / adj.sum(1, keepdim=True).clamp_min(1e-6)
        gp_bw = adj.T / adj.T.sum(1, keepdim=True).clamp_min(1e-6)
        return gp_fw, gp_bw

    def _aggregate(self, gp, x):
        return torch.einsum("nm,btmd->btnd", gp, x).contiguous()

    def forward(self, pred, adj, last_role):
        gp_fw, gp_bw = self._support(adj)
        x_gp_fw = self.x_gp_fw_in(pred)
        x_gp_bw = self.x_gp_bw_in(pred)

        # spatial aggregation
        y_gp_fw = self._aggregate(gp_fw, x_gp_fw)
        y_gp_bw = self._aggregate(gp_bw, x_gp_bw)
        y_gp_fw = self.y_gp_fw_fc(y_gp_fw)
        y_gp_bw = self.y_gp_bw_fc(y_gp_bw)
        x_gp_fw = self.x_gp_fw_fc(x_gp_fw)
        x_gp_bw = self.x_gp_bw_fc(x_gp_bw)
        x_gp_fw = torch.abs(y_gp_fw - x_gp_fw)
        x_gp_bw = torch.abs(y_gp_bw - x_gp_bw)
        x_gp_fw = self._aggregate(gp_fw, x_gp_fw)
        x_gp_bw = self._aggregate(gp_bw, x_gp_bw)
        x_gp_fw = self.x_gp_fw_gate(x_gp_fw)
        x_gp_bw = self.x_gp_bw_gate(x_gp_bw)
        y_gp_fw = self.y_gp_fw_proj(y_gp_fw)
        y_gp_bw = self.y_gp_bw_proj(y_gp_bw)
        y_gp_fw = x_gp_fw + y_gp_fw
        y_gp_bw = x_gp_bw + y_gp_bw

        # temporal modeling
        y = torch.cat((y_gp_fw, y_gp_bw), dim=-1)
        y = self.y_fc(y)
        x = torch.cat((x_gp_fw, x_gp_bw), dim=-1)
        x = self.x_fc(x)
        g1 = torch.exp(-self.g1(x))
        y = g1 * y
        time_idx = torch.arange(self.pred_len, device=pred.device)
        TE = self.te(time_idx).view(1, self.pred_len, 1, self.hidden_dim).expand(pred.size(0), -1, pred.size(2), -1)
        y = torch.cat((y, TE), dim=-1)

        state = pred.new_zeros(pred.size(0) * pred.size(2), self.hidden_dim)
        pred_steps = []
        for i in range(self.pred_len):
            if i == 0:
                g2 = torch.relu(self.g2_state(state))
            else:
                g2 = torch.relu(self.g2_x(x[:, i - 1].reshape(-1, self.hidden_dim)))
            g2 = torch.exp(-g2)
            state = g2 * state
            state = self.cell(y[:, i].reshape(-1, 2 * self.hidden_dim), state)
            pred_steps.append(state.view(pred.size(0), pred.size(2), self.hidden_dim).unsqueeze(1))
        residual = torch.cat(pred_steps, dim=1)
        residual = self.out(residual)
        lowvis_mask = (last_role[..., ROLE_NAS:ROLE_NAS + 1] + last_role[..., ROLE_VS:ROLE_VS + 1]).unsqueeze(1)
        residual = lowvis_mask * torch.tanh(residual)
        return pred + self.scale * residual, residual


class _SilentLogger:

    def info(self, *args, **kwargs):
        return None


class STBPPatternBankResidual(nn.Module):

    def __init__(self, configs, num_nodes, recipe_cfg):
        super().__init__()
        hidden_channel = int(recipe_cfg.get("pattern_bank_hidden_channel", max(192, configs.d_model * 3)))
        dropout = float(recipe_cfg.get("pattern_bank_dropout", 0.0))
        self.residual_scale = float(recipe_cfg.get("pattern_bank_scale", 0.1))
        role_residual_scale = recipe_cfg.get("pattern_bank_role_scale", {}) or {}
        self.register_buffer(
            "role_residual_scale",
            torch.tensor(
                [
                    float(role_residual_scale.get("aas", 0.0)),
                    float(role_residual_scale.get("nas", self.residual_scale)),
                    float(role_residual_scale.get("vs", self.residual_scale)),
                    float(role_residual_scale.get("fs", self.residual_scale)),
                ],
                dtype=torch.float32,
            ),
            persistent=False,
        )
        self.pred_len = configs.pred_len
        self.model_args = SimpleNamespace(
            dropout=dropout,
            y_len=self.pred_len,
            base_node_size=num_nodes,
            year=0,
            model={
                "in_channel": self.pred_len,
                "out_channel": self.pred_len,
                "hidden_channel": hidden_channel,
            },
            logger=_SilentLogger(),
        )
        self.backbone = STBP_Model(self.model_args)

    def forward(self, pred, adj, current_node_ids=None, last_role=None):
        if pred.size(-1) != 1:
            raise ValueError("STBPPatternBankResidual currently supports single-channel target_features=v")
        N = adj.shape[0]
        if N > self.backbone.num_nodes:
            self.backbone.expand_adaptive_params(N)
        x = pred[..., 0].permute(0, 2, 1).reshape((-1, N, self.backbone.args.model["in_channel"]))
        B, N, T = x.shape
        x = F.relu(self.backbone.fconv1(x))

        if current_node_ids is None:
            current_node_ids = list(range(N))
        adaptive_params = self.backbone.pattern_bank[current_node_ids]

        node_emb = adaptive_params.unsqueeze(0).expand(B, *adaptive_params.shape).permute(0,2,1).unsqueeze(-1)  # [bs, N, feature]
        x = x.permute(0,2,1).unsqueeze(-1)

        x = self.backbone.stmodule(x, node_emb).squeeze(-1)
        x = x.permute(0,2,1).squeeze(-1)
        x = self.backbone.fconv2(x)
        x = x.reshape((-1, self.backbone.args.model["out_channel"]))
        x = x + pred[..., 0].permute(0, 2, 1).reshape(-1, self.pred_len)
        residual = self.backbone.fc(self.backbone.activation(x))
        residual = F.dropout(residual, p=self.backbone.dropout, training=self.backbone.training)
        residual = residual.reshape(pred.shape[0], pred.shape[2], self.pred_len).permute(0, 2, 1).unsqueeze(-1).contiguous()
        if last_role is None:
            return pred + self.residual_scale * residual, residual
        scale = torch.sum(last_role * self.role_residual_scale.view(1, 1, ROLE_COUNT).to(last_role.device), dim=-1, keepdim=True)
        scale = scale.unsqueeze(1).expand_as(residual)
        return pred + scale * residual, residual


class EACNodeAdaptiveInput(nn.Module):
    # Adapted from references/EAC/src/model/model.py::EAC_Model:
    # keep the U/V low-rank adaptive-parameter idea, and apply it only to
    # missing lifecycle input positions before frozen experts run.
    def __init__(self, num_nodes, input_len, input_dim, rank, role_scale):
        super(EACNodeAdaptiveInput, self).__init__()
        self.rank = int(rank)
        self.input_len = int(input_len)
        self.input_dim = int(input_dim)
        self.U = nn.Parameter(torch.empty(num_nodes, self.rank).uniform_(-0.1, 0.1))
        self.V = nn.Parameter(torch.empty(self.rank, self.input_len * self.input_dim).uniform_(-0.1, 0.1))
        self.num_nodes = int(num_nodes)
        self.register_buffer(
            "role_input_scale",
            torch.tensor(
                [
                    float(role_scale.get("aas", 0.0)),
                    float(role_scale.get("nas", 0.0)),
                    float(role_scale.get("vs", 1.0)),
                    float(role_scale.get("fs", 0.5)),
                ],
                dtype=torch.float32,
            ),
            persistent=False,
        )

    def expand_adaptive_params(self, new_num_nodes):
        if new_num_nodes > self.num_nodes:
            new_params = nn.Parameter(torch.empty(new_num_nodes - self.num_nodes, self.rank, dtype=self.U.dtype, device=self.U.device).uniform_(-0.1, 0.1))
            self.U = nn.Parameter(torch.cat([self.U, new_params], dim=0))
            self.num_nodes = new_num_nodes

    def forward(self, raw_x_enc, role_seq, current_node_ids=None):
        B, T, N, C = raw_x_enc.shape
        if T != self.input_len or C != self.input_dim:
            raise ValueError(
                f"EACNodeAdaptiveInput expects input shape [B,{self.input_len},N,{self.input_dim}], got {tuple(raw_x_enc.shape)}"
            )
        if N > self.num_nodes:
            self.expand_adaptive_params(N)
        if current_node_ids is None:
            current_node_ids = list(range(N))
        adaptive_params = torch.mm(self.U[current_node_ids, :], self.V)  # [N, input_len * input_dim]
        adaptive_params = adaptive_params.reshape(N, self.input_len, self.input_dim)
        adaptive_params = adaptive_params.permute(1, 0, 2).unsqueeze(0).expand(B, -1, -1, -1)
        role_factor = torch.sum(role_seq * self.role_input_scale.view(1, 1, 1, ROLE_COUNT).to(role_seq.device), dim=-1, keepdim=True)
        prompt = role_factor * adaptive_params
        return raw_x_enc + prompt, prompt


class TPBPatternInputAdapter(nn.Module):
    # Adapted from references/TPB/model/Meta_Models/meta_patch.py::PatternEncoder_patternkeyv2:
    # keep the pattern-key attention order, and add only a small projection so
    # matched patterns can be used as a lifecycle input prompt for frozen experts.
    def __init__(
        self,
        input_len,
        input_dim,
        pattern_path,
        num_patterns,
        pattern_dim,
        role_scale,
        allow_random_pattern=False,
        use_age_gate=False,
        age_gate_cutoff=0.5,
        age_gate_min=0.0,
    ):
        super(TPBPatternInputAdapter, self).__init__()
        self.input_len = int(input_len)
        self.input_dim = int(input_dim)
        self.input_patch_dim = self.input_len * self.input_dim

        pattern = None
        pattern_source = None
        candidate_paths = []
        if pattern_path:
            pattern_path = os.path.normpath(pattern_path)
            candidate_paths.append(pattern_path)
        candidate_paths.extend(sorted(glob.glob(os.path.join("external_weights", "tpb_patterns", "cosine_10_cl.pt"))))
        candidate_paths.extend(sorted(glob.glob(os.path.join("external_weights", "tpb_patterns", "cosine_10_cl_1.pt"))))
        candidate_paths.extend(sorted(glob.glob(os.path.join("references", "TPB", "pattern", "*", "cosine_10_cl.pt"))))
        candidate_paths.extend(sorted(glob.glob(os.path.join("references", "MTPB", "pattern", "*", "cosine_10_cl_1.pt"))))
        for candidate_path in candidate_paths:
            if not candidate_path or not os.path.exists(candidate_path):
                continue
            pattern = torch.load(candidate_path, map_location="cpu", weights_only=False)
            if not isinstance(pattern, torch.Tensor):
                raise ValueError(f"TPB pattern file must contain a tensor, got {type(pattern)} from {candidate_path}")
            pattern = pattern.float()
            if pattern.dim() != 2:
                raise ValueError(f"TPB pattern tensor must be 2-D [K,D], got {tuple(pattern.shape)} from {candidate_path}")
            pattern_source = candidate_path
            break
        if pattern is None:
            if not allow_random_pattern:
                searched = ", ".join(candidate_paths) if candidate_paths else "<none>"
                raise FileNotFoundError(
                    "TPB pattern file not found. Put cosine_10_cl.pt under "
                    "external_weights/tpb_patterns/ or set allow_random_tpb_pattern=true "
                    f"for a random-pattern ablation. Searched: {searched}"
                )
            pattern = torch.empty(int(num_patterns), int(pattern_dim)).uniform_(-0.1, 0.1)
            pattern_source = "random_trainable"

        self.K, self.D = pattern.shape
        self.temporature = self.D ** 0.5
        self.use_age_gate = bool(use_age_gate)
        self.age_gate_cutoff = float(age_gate_cutoff)
        self.age_gate_min = float(age_gate_min)
        self.Qnet = nn.Linear(self.input_patch_dim, self.D)
        self.qry_mat = nn.Parameter(torch.randn(self.D, self.K))
        self.out_proj = nn.Linear(self.D, self.input_patch_dim)
        self.pattern_source = pattern_source
        if pattern_source == "random_trainable":
            self.pattern = nn.Parameter(pattern)
        else:
            self.register_buffer("pattern", pattern, persistent=True)
        self.register_buffer(
            "role_input_scale",
            torch.tensor(
                [
                    float(role_scale.get("aas", 0.0)),
                    float(role_scale.get("nas", 0.0)),
                    float(role_scale.get("vs", 1.0)),
                    float(role_scale.get("fs", 0.5)),
                ],
                dtype=torch.float32,
            ),
            persistent=False,
        )

    def forward(self, raw_x_enc, role_seq, history_age_seq=None):
        B, T, N, C = raw_x_enc.shape
        if T != self.input_len or C != self.input_dim:
            raise ValueError(
                f"TPBPatternInputAdapter expects input shape [B,{self.input_len},N,{self.input_dim}], got {tuple(raw_x_enc.shape)}"
            )

        X = raw_x_enc.permute(0, 2, 1, 3).reshape(B, N, 1, self.input_patch_dim)
        qry = self.Qnet(X)
        weight = torch.einsum("dk,bnld->bnlk", self.qry_mat, qry) / self.temporature
        weight = F.softmax(weight, dim=3)
        w_X = torch.einsum("bnlk,kd->bnld", weight, self.pattern.to(raw_x_enc.device))
        prompt = self.out_proj(w_X).reshape(B, N, T, C).permute(0, 2, 1, 3).contiguous()

        role_factor = torch.sum(role_seq * self.role_input_scale.view(1, 1, 1, ROLE_COUNT).to(role_seq.device), dim=-1, keepdim=True)
        if self.use_age_gate:
            if history_age_seq is None:
                raise ValueError("TPBPatternInputAdapter use_age_gate=True requires history_age_seq")
            cutoff = max(self.age_gate_cutoff, 1e-6)
            age_gate = ((cutoff - history_age_seq) / cutoff).clamp(min=self.age_gate_min, max=1.0)
            role_factor = role_factor * age_gate
        prompt = role_factor * prompt
        return raw_x_enc + prompt, prompt


def _resolve_model_module_name(model_name):
    if model_name == "GWNET":
        return "GWNet"
    return model_name


def _load_expert_from_spec(spec, device, expert_name):
    model_name = spec["model"]
    config_path = spec["config_path"]
    checkpoint_path = spec["checkpoint_path"]
    if any(token in checkpoint_path for token in ["*", "?", "["]):
        matches = glob.glob(checkpoint_path)
        if not matches:
            raise FileNotFoundError(f"RoleFusion {expert_name} expert checkpoint glob has no matches: {checkpoint_path}")
        checkpoint_path = max(matches, key=os.path.getmtime)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"RoleFusion {expert_name} expert config not found: {config_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"RoleFusion {expert_name} expert checkpoint not found: {checkpoint_path}")
    expert_args = load_args_from_config(config_path, overrides={"device": device})
    if expert_args.model != model_name:
        expert_args.model = model_name
    module_name = _resolve_model_module_name(model_name)
    module = importlib.import_module(f"model.{module_name}")
    expert = module.Model(expert_args).float()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError(f"Expert checkpoint at {checkpoint_path} must contain a 'state_dict'")
    expert.load_state_dict(checkpoint["state_dict"])
    return expert, expert_args, os.path.normpath(checkpoint_path)


class Model(nn.Module):

    def __init__(self, configs):
        super().__init__()
        if not configs.lifecycle_role_features or not configs.lifecycle_history_age_feature:
            raise ValueError("RoleFusion requires lifecycle_role_features=true and lifecycle_history_age_feature=true")
        self.cfg = getattr(configs, "model_family_cfg", {}) or {}
        self.recipe = configs.model_recipe
        if self.recipe not in {"hard_route", "soft_fuse", "soft_fuse_v2", "oracle_gate", "role_mmoe", "testam_role_mmoe"}:
            raise ValueError(f"Unsupported RoleFusion recipe: {self.recipe}")
        experts_cfg = self.cfg.get("experts", {})
        if "graph" not in experts_cfg or "patch" not in experts_cfg:
            raise ValueError("RoleFusion family.experts must define both graph and patch experts")
        self.graph_trainable_patterns = list(experts_cfg["graph"].get("trainable_param_patterns", []) or [])
        self.patch_trainable_patterns = list(experts_cfg["patch"].get("trainable_param_patterns", []) or [])

        self.graph_expert, self.graph_args, self.graph_checkpoint_path = _load_expert_from_spec(experts_cfg["graph"], configs.device, "graph")
        self.patch_expert, self.patch_args, self.patch_checkpoint_path = _load_expert_from_spec(experts_cfg["patch"], configs.device, "patch")

        self.raw_enc_in = configs.raw_enc_in
        self.output_dim = configs.c_out
        self.pred_len = configs.pred_len
        self.role_start = self.raw_enc_in
        self.role_end = self.role_start + ROLE_COUNT
        self.age_idx = self.role_end
        full_adj = load_project_adjacency({"data_path": configs.data_path})
        self.full_num_nodes = int(full_adj.shape[0])
        self.known_nodes, _ = resolve_known_unknown_nodes(
            {
                "data_path": configs.data_path,
                "unknown_nodes_path": configs.unknown_nodes_path,
                "unknown_nodes_list": configs.unknown_nodes_list,
                "unknown_nodes_num": configs.unknown_nodes_num,
                "seed": configs.seed,
            }
        )

        recipe_cfg = self.cfg.get("recipe", {})
        model_cfg = self.cfg.get("model", {})
        state_hidden_dim = int(model_cfg.get("state_hidden_dim", 32))
        state_dim = int(model_cfg.get("state_dim", 32))
        gate_hidden_dim = int(model_cfg.get("gate_hidden_dim", 32))
        self.lambda_prior = float(recipe_cfg.get("lambda_prior", 0.1))
        self.lambda_entropy = float(recipe_cfg.get("lambda_entropy", 0.01))
        self.lambda_residual = float(recipe_cfg.get("lambda_residual", 0.01))
        self.lambda_smooth = float(recipe_cfg.get("lambda_smooth", 0.01))
        self.lambda_decisive = float(recipe_cfg.get("lambda_decisive", 0.02))
        self.lambda_oracle = float(recipe_cfg.get("lambda_oracle", 0.1))
        self.lambda_testam = float(recipe_cfg.get("lambda_testam", 0.0))
        self.testam_warmup_epoch = int(recipe_cfg.get("testam_warmup_epoch", 0))
        self.testam_threshold = float(recipe_cfg.get("testam_threshold", 0.0))
        self.lambda_fs_recovery = float(recipe_cfg.get("lambda_fs_recovery", 0.0))
        self.lambda_spatial_transfer = float(recipe_cfg.get("lambda_spatial_transfer", 0.0))
        self.lambda_pattern_bank = float(recipe_cfg.get("lambda_pattern_bank", 0.0))
        self.lambda_eac_input = float(recipe_cfg.get("lambda_eac_input", 0.0))
        self.lambda_tpb_input = float(recipe_cfg.get("lambda_tpb_input", 0.0))
        self.residual_scale = float(recipe_cfg.get("residual_scale", 0.5))
        self.graph_recovery_apply_to = str(recipe_cfg.get("graph_recovery_apply_to", "graph_input")).lower()
        recovery_expert_role_scale = recipe_cfg.get("graph_recovery_expert_role_scale", {}) or {}
        self.register_buffer(
            "graph_recovery_expert_role_scale",
            torch.tensor(
                [
                    float(recovery_expert_role_scale.get("aas", 0.0)),
                    float(recovery_expert_role_scale.get("nas", 0.0)),
                    float(recovery_expert_role_scale.get("vs", 0.0)),
                    float(recovery_expert_role_scale.get("fs", 1.0)),
                ],
                dtype=torch.float32,
            ),
        )
        role_residual_scale = recipe_cfg.get(
            "role_residual_scale",
            {"aas": self.residual_scale, "nas": self.residual_scale, "vs": self.residual_scale, "fs": self.residual_scale},
        )
        self.register_buffer(
            "role_residual_scale",
            torch.tensor(
                [
                    float(role_residual_scale.get("aas", self.residual_scale)),
                    float(role_residual_scale.get("nas", self.residual_scale)),
                    float(role_residual_scale.get("vs", self.residual_scale)),
                    float(role_residual_scale.get("fs", self.residual_scale)),
                ],
                dtype=torch.float32,
            ),
        )
        self.gate_temperature = float(recipe_cfg.get("gate_temperature", 1.0))
        role_graph_prior = recipe_cfg.get(
            "role_graph_prior",
            {"aas": 0.3, "nas": 0.7, "vs": 0.8, "fs": 0.2},
        )
        self.register_buffer(
            "role_graph_prior",
            torch.tensor(
                [
                    float(role_graph_prior.get("aas", 0.3)),
                    float(role_graph_prior.get("nas", 0.7)),
                    float(role_graph_prior.get("vs", 0.8)),
                    float(role_graph_prior.get("fs", 0.2)),
                ],
                dtype=torch.float32,
            ),
        )
        role_graph_alpha_floor = recipe_cfg.get(
            "role_graph_alpha_floor",
            {"aas": 0.0, "nas": 0.0, "vs": 0.0, "fs": 0.0},
        )
        self.register_buffer(
            "role_graph_alpha_floor",
            torch.tensor(
                [
                    float(role_graph_alpha_floor.get("aas", 0.0)),
                    float(role_graph_alpha_floor.get("nas", 0.0)),
                    float(role_graph_alpha_floor.get("vs", 0.0)),
                    float(role_graph_alpha_floor.get("fs", 0.0)),
                ],
                dtype=torch.float32,
            ),
        )

        if self.recipe in {"soft_fuse", "oracle_gate", "role_mmoe", "testam_role_mmoe"}:
            self.state_encoder = nn.Sequential(
                nn.Linear(ROLE_COUNT + 2, state_hidden_dim),
                nn.ReLU(),
                nn.Linear(state_hidden_dim, state_dim),
                nn.ReLU(),
            )
            self.fusion_gate = nn.Sequential(
                nn.Linear(state_dim + 3, gate_hidden_dim),
                nn.ReLU(),
                nn.Linear(gate_hidden_dim, 1),
            )
            self.graph_confidence_head = nn.Identity()
            self.patch_confidence_head = nn.Identity()
            self.state_bias_head = nn.Identity()
            self.residual_head = nn.Identity()
            if self.recipe in {"oracle_gate", "role_mmoe", "testam_role_mmoe"}:
                gate_input_dim = state_dim + 3 * self.output_dim
                self.role_fusion_gates = nn.ModuleList(
                    [
                        nn.Sequential(
                            nn.Linear(gate_input_dim, gate_hidden_dim),
                            nn.ReLU(),
                            nn.Linear(gate_hidden_dim, 1),
                        )
                        for _ in range(ROLE_COUNT)
                    ]
                )
            else:
                self.role_fusion_gates = nn.ModuleList()
            if self.recipe in {"role_mmoe", "testam_role_mmoe"}:
                residual_input_dim = state_dim + 4 * self.output_dim
                self.role_residual_heads = nn.ModuleList(
                    [
                        nn.Sequential(
                            nn.Linear(residual_input_dim, gate_hidden_dim),
                            nn.ReLU(),
                            nn.Linear(gate_hidden_dim, self.output_dim),
                        )
                        for _ in range(ROLE_COUNT)
                    ]
                )
            else:
                self.role_residual_heads = nn.ModuleList()
        elif self.recipe == "soft_fuse_v2":
            state_input_dim = ROLE_COUNT + 5
            self.state_encoder = nn.Sequential(
                nn.Linear(state_input_dim, state_hidden_dim),
                nn.ReLU(),
                nn.Linear(state_hidden_dim, state_dim),
                nn.ReLU(),
            )
            confidence_input_dim = state_dim + 3
            self.graph_confidence_head = nn.Sequential(
                nn.Linear(confidence_input_dim, gate_hidden_dim),
                nn.ReLU(),
                nn.Linear(gate_hidden_dim, 1),
            )
            self.patch_confidence_head = nn.Sequential(
                nn.Linear(confidence_input_dim, gate_hidden_dim),
                nn.ReLU(),
                nn.Linear(gate_hidden_dim, 1),
            )
            self.state_bias_head = nn.Sequential(
                nn.Linear(state_dim, gate_hidden_dim),
                nn.ReLU(),
                nn.Linear(gate_hidden_dim, 1),
            )
            residual_input_dim = state_dim + 6
            self.residual_head = nn.Sequential(
                nn.Linear(residual_input_dim, gate_hidden_dim),
                nn.ReLU(),
                nn.Linear(gate_hidden_dim, 1),
            )
            self.fusion_gate = nn.Identity()
            self.role_fusion_gates = nn.ModuleList()
            self.role_residual_heads = nn.ModuleList()
        else:
            self.state_encoder = nn.Identity()
            self.fusion_gate = nn.Identity()
            self.graph_confidence_head = nn.Identity()
            self.patch_confidence_head = nn.Identity()
            self.state_bias_head = nn.Identity()
            self.residual_head = nn.Identity()
            self.role_fusion_gates = nn.ModuleList()
            self.role_residual_heads = nn.ModuleList()

        self.fs_recovery_head = None
        if bool(recipe_cfg.get("use_fs_recovery_head", False)):
            self.fs_recovery_head = GCNMMemoryRecoveryHead(
                output_dim=self.output_dim,
                hidden_dim=int(model_cfg.get("fs_recovery_hidden_dim", gate_hidden_dim)),
                scale=float(recipe_cfg.get("fs_recovery_scale", 0.1)),
            )
        self.spatial_transfer_head = None
        if bool(recipe_cfg.get("use_spatial_transfer_head", False)):
            self.spatial_transfer_head = INCREASESpatialTransferHead(
                output_dim=self.output_dim,
                hidden_dim=int(model_cfg.get("spatial_transfer_hidden_dim", gate_hidden_dim)),
                pred_len=self.pred_len,
                scale=float(recipe_cfg.get("spatial_transfer_scale", 0.1)),
            )
        self.lifecycle_pattern_bank = None
        if bool(recipe_cfg.get("use_lifecycle_pattern_bank", False)):
            self.lifecycle_pattern_bank = STBPPatternBankResidual(
                configs=configs,
                num_nodes=int(recipe_cfg.get("pattern_bank_num_nodes", self.full_num_nodes)),
                recipe_cfg=recipe_cfg,
            )
        self.eac_graph_input_adapter = None
        if bool(recipe_cfg.get("use_eac_graph_input_adapter", False)):
            self.eac_graph_input_adapter = EACNodeAdaptiveInput(
                num_nodes=int(recipe_cfg.get("eac_num_nodes", self.full_num_nodes)),
                input_len=configs.input_len,
                input_dim=self.raw_enc_in,
                rank=int(recipe_cfg.get("eac_rank", 8)),
                role_scale=recipe_cfg.get("eac_role_scale", {}),
            )
        self.tpb_graph_input_adapter = None
        if bool(recipe_cfg.get("use_tpb_graph_input_adapter", False)):
            self.tpb_graph_input_adapter = TPBPatternInputAdapter(
                input_len=configs.input_len,
                input_dim=self.raw_enc_in,
                pattern_path=recipe_cfg.get("tpb_pattern_path", None),
                num_patterns=int(recipe_cfg.get("tpb_num_patterns", 10)),
                pattern_dim=int(recipe_cfg.get("tpb_pattern_dim", max(64, configs.d_model * 2))),
                role_scale=recipe_cfg.get("tpb_role_scale", {}),
                allow_random_pattern=bool(recipe_cfg.get("allow_random_tpb_pattern", False)),
                use_age_gate=bool(recipe_cfg.get("tpb_use_age_gate", False)),
                age_gate_cutoff=float(recipe_cfg.get("tpb_age_gate_cutoff", 0.5)),
                age_gate_min=float(recipe_cfg.get("tpb_age_gate_min", 0.0)),
            )
        self.graph_input_recovery = None
        if bool(recipe_cfg.get("use_graph_input_recovery", False)):
            self.graph_input_recovery = GraphInputRecoveryAdapter(
                recipe_cfg=recipe_cfg,
                full_adj=full_adj,
                input_dim=self.raw_enc_in,
                full_num_nodes=self.full_num_nodes,
            )

        self._configure_expert_trainability()

    def _set_expert_trainability(self, expert, patterns):
        for name, param in expert.named_parameters():
            param.requires_grad = False
            if patterns and any(pattern in name for pattern in patterns):
                param.requires_grad = True
        expert.eval()

    def _configure_expert_trainability(self):
        for expert, patterns in (
            (self.graph_expert, self.graph_trainable_patterns),
            (self.patch_expert, self.patch_trainable_patterns),
        ):
            expert.eval()
            self._set_expert_trainability(expert, patterns)

    def train(self, mode=True):
        super().train(mode)
        self._configure_expert_trainability()
        return self

    def _extract_role_and_age(self, x_enc):
        if x_enc.size(-1) <= self.age_idx:
            raise ValueError(
                f"RoleFusion expects raw + role-onehot + history-age channels, got {x_enc.size(-1)} total channels"
            )
        role_seq = x_enc[..., self.role_start:self.role_end]
        history_age_seq = x_enc[..., self.age_idx:self.age_idx + 1]
        last_role = role_seq[:, -1]
        last_age = history_age_seq[:, -1]
        visibility_ratio = (role_seq[..., ROLE_AAS] + role_seq[..., ROLE_NAS]).mean(dim=1, keepdim=False).unsqueeze(-1)
        return role_seq, last_role, last_age, visibility_ratio

    def _resolve_current_node_ids(self, current_num_nodes):
        if current_num_nodes == len(getattr(self, "known_nodes", [])):
            return self.known_nodes
        if current_num_nodes == getattr(self, "full_num_nodes", current_num_nodes):
            return list(range(self.full_num_nodes))
        return list(range(current_num_nodes))

    def _build_gate_features(self, x_enc):
        _, last_role, last_age, visibility_ratio = self._extract_role_and_age(x_enc)
        state_input = torch.cat([last_role, last_age, visibility_ratio], dim=-1)
        return self.state_encoder(state_input), last_role

    def _build_confidence_state(self, x_enc):
        role_seq, last_role, last_age, visibility_ratio = self._extract_role_and_age(x_enc)
        raw_seq = x_enc[..., : self.output_dim]
        visible_mask = (role_seq[..., ROLE_AAS] + role_seq[..., ROLE_NAS]) > 0.5
        visible_mask_f = visible_mask.unsqueeze(-1).to(dtype=raw_seq.dtype)
        visible_count = visible_mask_f.sum(dim=1, keepdim=False).clamp_min(1.0)
        visible_mean = (raw_seq * visible_mask_f).sum(dim=1, keepdim=False) / visible_count
        centered = (raw_seq - visible_mean.unsqueeze(1)) * visible_mask_f
        visible_std = torch.sqrt((centered.pow(2).sum(dim=1, keepdim=False) / visible_count).clamp_min(1e-6))

        last_visible = torch.zeros_like(raw_seq[:, 0])
        prev_visible = torch.zeros_like(raw_seq[:, 0])
        found_last = torch.zeros_like(raw_seq[:, 0], dtype=torch.bool)
        found_prev = torch.zeros_like(raw_seq[:, 0], dtype=torch.bool)
        for t in range(raw_seq.size(1) - 1, -1, -1):
            current_mask = visible_mask[:, t].unsqueeze(-1)
            take_last = (~found_last) & current_mask
            last_visible = torch.where(take_last, raw_seq[:, t], last_visible)
            found_last = found_last | current_mask
            take_prev = found_last & (~found_prev) & current_mask & (~take_last)
            prev_visible = torch.where(take_prev, raw_seq[:, t], prev_visible)
            found_prev = found_prev | take_prev
        recent_trend = torch.where(found_prev, last_visible - prev_visible, torch.zeros_like(last_visible))
        state_input = torch.cat([last_role, last_age, visibility_ratio, last_visible, recent_trend, visible_std], dim=-1)
        return self.state_encoder(state_input), last_role, last_visible

    def _run_experts(self, adj, x_enc, x_t_mark, pos_mark, x_dec, dec_t_mark, epoch):
        raw_x_enc = x_enc[..., :self.raw_enc_in]
        expert_x_dec = None if x_dec is None else x_dec[..., : self.output_dim]
        aux_losses = {}
        graph_x_enc = raw_x_enc
        recovered_graph_x_enc = None
        use_recovery_as_expert = self.graph_input_recovery is not None and self.graph_recovery_apply_to == "role_expert"
        has_graph_input_adapter = (
            (self.graph_input_recovery is not None and not use_recovery_as_expert)
            or self.eac_graph_input_adapter is not None
            or self.tpb_graph_input_adapter is not None
        )
        role_seq = None
        history_age_seq = None
        if self.graph_input_recovery is not None:
            role_seq, last_role, last_age, visibility_ratio = self._extract_role_and_age(x_enc)
            recovery_state = torch.cat([last_role, last_age, visibility_ratio], dim=-1)
            current_node_ids = self._resolve_current_node_ids(raw_x_enc.size(2))
            recovered_graph_x_enc, _, recovery_aux_losses = self.graph_input_recovery(
                graph_x_enc,
                role_seq,
                last_role,
                adj,
                current_node_ids=current_node_ids,
                state_features=recovery_state,
            )
            if not use_recovery_as_expert:
                graph_x_enc = recovered_graph_x_enc
            aux_losses.update(recovery_aux_losses)
        if self.eac_graph_input_adapter is not None:
            if role_seq is None:
                role_seq, _, _, _ = self._extract_role_and_age(x_enc)
            current_node_ids = self._resolve_current_node_ids(raw_x_enc.size(2))
            graph_x_enc, prompt = self.eac_graph_input_adapter(graph_x_enc, role_seq, current_node_ids=current_node_ids)
            aux_losses["rolefusion_eac_input"] = self.lambda_eac_input * prompt.pow(2).mean()
        if self.tpb_graph_input_adapter is not None:
            if role_seq is None:
                role_seq, _, _, _ = self._extract_role_and_age(x_enc)
            history_age_seq = x_enc[..., self.age_idx:self.age_idx + 1]
            graph_x_enc, prompt = self.tpb_graph_input_adapter(graph_x_enc, role_seq, history_age_seq=history_age_seq)
            aux_losses["rolefusion_tpb_input"] = self.lambda_tpb_input * prompt.pow(2).mean()
        if has_graph_input_adapter:
            graph_output = self.graph_expert(adj, graph_x_enc, x_t_mark, pos_mark, expert_x_dec, dec_t_mark, epoch)
        else:
            with torch.no_grad():
                graph_output = self.graph_expert(adj, graph_x_enc, x_t_mark, pos_mark, expert_x_dec, dec_t_mark, epoch)
        recovery_graph_output = None
        if use_recovery_as_expert:
            recovery_graph_output = self.graph_expert(adj, recovered_graph_x_enc, x_t_mark, pos_mark, expert_x_dec, dec_t_mark, epoch)
        with torch.no_grad():
            patch_output = self.patch_expert(adj, raw_x_enc, x_t_mark, pos_mark, expert_x_dec, dec_t_mark, epoch)
        y_graph, _ = unpack_model_output(graph_output)
        y_patch, _ = unpack_model_output(patch_output)
        y_recovered_graph = None
        if recovery_graph_output is not None:
            y_recovered_graph, _ = unpack_model_output(recovery_graph_output)
        return y_graph, y_patch, aux_losses, y_recovered_graph

    def _hard_route(self, y_graph, y_patch, last_role):
        role_ids = torch.argmax(last_role, dim=-1)
        graph_mask = ((role_ids == ROLE_VS) | (role_ids == ROLE_NAS)).unsqueeze(1).unsqueeze(-1)
        return torch.where(graph_mask, y_graph, y_patch)

    def _apply_alpha_floor(self, alpha, last_role):
        floor = torch.sum(last_role * self.role_graph_alpha_floor.view(1, 1, ROLE_COUNT), dim=-1, keepdim=True)
        floor = floor.unsqueeze(1).expand_as(alpha)
        return floor + (1.0 - floor) * alpha

    def _soft_fuse(self, x_enc, y_graph, y_patch):
        node_state, last_role = self._build_gate_features(x_enc)
        state_t = node_state.unsqueeze(1).expand(-1, y_graph.size(1), -1, -1)
        gate_input = torch.cat([state_t, y_graph, y_patch, torch.abs(y_graph - y_patch)], dim=-1)
        alpha = torch.sigmoid(self.fusion_gate(gate_input))
        alpha = self._apply_alpha_floor(alpha, last_role)
        fused = alpha * y_graph + (1.0 - alpha) * y_patch

        prior = torch.sum(last_role * self.role_graph_prior.view(1, 1, ROLE_COUNT), dim=-1, keepdim=True)
        prior = prior.unsqueeze(1).expand_as(alpha)
        prior_loss = torch.mean((alpha - prior) ** 2)
        eps = 1e-6
        entropy = -(alpha * torch.log(alpha + eps) + (1.0 - alpha) * torch.log(1.0 - alpha + eps)).mean()
        return fused, {
            "rolefusion_prior": self.lambda_prior * prior_loss,
            "rolefusion_entropy": self.lambda_entropy * entropy,
        }

    def _apply_role_heads(self, adj, x_enc, pred, last_role, aux_losses):
        if self.fs_recovery_head is not None:
            pred, residual = self.fs_recovery_head(pred, x_enc[..., :self.output_dim], last_role)
            aux_losses["rolefusion_fs_recovery"] = self.lambda_fs_recovery * residual.pow(2).mean()
        if self.spatial_transfer_head is not None:
            pred, residual = self.spatial_transfer_head(pred, adj, last_role)
            aux_losses["rolefusion_spatial_transfer"] = self.lambda_spatial_transfer * residual.pow(2).mean()
        if self.lifecycle_pattern_bank is not None:
            current_node_ids = self._resolve_current_node_ids(pred.size(2))
            pred, residual = self.lifecycle_pattern_bank(pred, adj, current_node_ids=current_node_ids, last_role=last_role)
            aux_losses["rolefusion_pattern_bank"] = self.lambda_pattern_bank * residual.pow(2).mean()
        return pred

    def _role_gate(self, adj, x_enc, y_graph, y_patch, with_residual=False, y_recovered_graph=None):
        node_state, last_role = self._build_gate_features(x_enc)
        state_t = node_state.unsqueeze(1).expand(-1, y_graph.size(1), -1, -1)
        gate_input = torch.cat([state_t, y_graph, y_patch, torch.abs(y_graph - y_patch)], dim=-1)
        role_logits = torch.cat([gate(gate_input) for gate in self.role_fusion_gates], dim=-1)
        role_weights = last_role.unsqueeze(1).expand(-1, y_graph.size(1), -1, -1)
        alpha = torch.sigmoid(torch.sum(role_logits * role_weights, dim=-1, keepdim=True))
        alpha = self._apply_alpha_floor(alpha, last_role)
        fused = alpha * y_graph + (1.0 - alpha) * y_patch

        prior = torch.sum(last_role * self.role_graph_prior.view(1, 1, ROLE_COUNT), dim=-1, keepdim=True)
        prior = prior.unsqueeze(1).expand_as(alpha)
        prior_loss = torch.mean((alpha - prior) ** 2)
        eps = 1e-6
        entropy = -(alpha * torch.log(alpha + eps) + (1.0 - alpha) * torch.log(1.0 - alpha + eps)).mean()
        aux_losses = {
            "rolefusion_prior": self.lambda_prior * prior_loss,
            "rolefusion_entropy": self.lambda_entropy * entropy,
        }
        if with_residual:
            residual_input = torch.cat([state_t, fused, y_graph, y_patch, torch.abs(y_graph - y_patch)], dim=-1)
            role_residuals = torch.stack([head(residual_input) for head in self.role_residual_heads], dim=-2)
            residual = torch.sum(role_residuals * role_weights.unsqueeze(-1), dim=-2)
            residual_scale = torch.sum(last_role * self.role_residual_scale.view(1, 1, ROLE_COUNT), dim=-1, keepdim=True)
            residual_scale = residual_scale.unsqueeze(1).expand_as(residual)
            residual = residual_scale * torch.tanh(residual)
            fused = fused + residual
            aux_losses["rolefusion_role_residual"] = self.lambda_residual * residual.pow(2).mean()
        if y_recovered_graph is not None:
            recovery_scale = torch.sum(
                last_role * self.graph_recovery_expert_role_scale.view(1, 1, ROLE_COUNT),
                dim=-1,
                keepdim=True,
            )
            recovery_scale = recovery_scale.unsqueeze(1).expand_as(fused)
            fused = fused + recovery_scale * (y_recovered_graph - fused)
        fused = self._apply_role_heads(adj, x_enc, fused, last_role, aux_losses)
        routing_meta = {
            "alpha": alpha,
            "y_graph": y_graph.detach(),
            "y_patch": y_patch.detach(),
            "lambda_oracle": self.lambda_oracle,
        }
        if self.lambda_testam > 0 or self.recipe == "testam_role_mmoe":
            routing_meta["testam"] = {
                "gate": torch.cat([alpha, 1.0 - alpha], dim=-1),
                "expert_preds": torch.stack([y_graph.detach(), y_patch.detach()], dim=-1),
                "lambda_testam": self.lambda_testam,
                "warmup_epoch": self.testam_warmup_epoch,
                "threshold": self.testam_threshold,
            }
        return fused, aux_losses, routing_meta

    def _soft_fuse_v2(self, x_enc, y_graph, y_patch):
        node_state, _, last_visible = self._build_confidence_state(x_enc)
        state_t = node_state.unsqueeze(1).expand(-1, y_graph.size(1), -1, -1)
        last_obs = last_visible.unsqueeze(1).expand_as(y_graph)

        graph_prev = torch.cat([last_obs[:, :1], y_graph[:, :-1]], dim=1)
        patch_prev = torch.cat([last_obs[:, :1], y_patch[:, :-1]], dim=1)
        graph_delta_last = torch.abs(y_graph - last_obs)
        patch_delta_last = torch.abs(y_patch - last_obs)
        graph_step_change = torch.abs(y_graph - graph_prev)
        patch_step_change = torch.abs(y_patch - patch_prev)

        graph_conf_input = torch.cat([state_t, y_graph, graph_delta_last, graph_step_change], dim=-1)
        patch_conf_input = torch.cat([state_t, y_patch, patch_delta_last, patch_step_change], dim=-1)
        graph_logit = self.graph_confidence_head(graph_conf_input)
        patch_logit = self.patch_confidence_head(patch_conf_input)
        state_bias = self.state_bias_head(state_t)
        conf_logits = torch.cat([graph_logit + state_bias, patch_logit - state_bias], dim=-1) / max(self.gate_temperature, 1e-6)
        alpha = torch.softmax(conf_logits, dim=-1)[..., :1]

        fused = alpha * y_graph + (1.0 - alpha) * y_patch
        disagreement = torch.abs(y_graph - y_patch)
        residual_input = torch.cat([state_t, alpha, y_graph, y_patch, disagreement, graph_logit, patch_logit], dim=-1)
        residual = self.residual_scale * torch.tanh(self.residual_head(residual_input))
        pred = fused + residual

        eps = 1e-6
        entropy = -(alpha * torch.log(alpha + eps) + (1.0 - alpha) * torch.log(1.0 - alpha + eps))
        disagreement_weight = disagreement.detach() / (disagreement.detach().mean() + eps)
        weighted_entropy = (disagreement_weight * entropy).mean()
        residual_reg = residual.pow(2).mean()
        if alpha.size(1) > 1:
            smooth_reg = torch.abs(alpha[:, 1:] - alpha[:, :-1]).mean()
        else:
            smooth_reg = alpha.new_tensor(0.0)
        decisive_reg = (disagreement_weight * (alpha * (1.0 - alpha))).mean()

        return pred, {
            "rolefusion_v2_entropy": self.lambda_entropy * weighted_entropy,
            "rolefusion_v2_residual": self.lambda_residual * residual_reg,
            "rolefusion_v2_smooth": self.lambda_smooth * smooth_reg,
            "rolefusion_v2_decisive": self.lambda_decisive * decisive_reg,
        }

    def forward(self, adj, x_enc, x_t_mark=None, pos_mark=None, x_dec=None, dec_t_mark=None, epoch=0):
        y_graph, y_patch, expert_aux_losses, y_recovered_graph = self._run_experts(adj, x_enc, x_t_mark, pos_mark, x_dec, dec_t_mark, epoch)
        if self.recipe == "hard_route":
            _, last_role, _, _ = self._extract_role_and_age(x_enc)
            pred = self._hard_route(y_graph, y_patch, last_role)
            return {"pred": pred, "aux_losses": expert_aux_losses}
        if self.recipe == "soft_fuse_v2":
            pred, aux_losses = self._soft_fuse_v2(x_enc, y_graph, y_patch)
        elif self.recipe == "oracle_gate":
            pred, aux_losses, oracle_routing = self._role_gate(adj, x_enc, y_graph, y_patch, with_residual=False, y_recovered_graph=y_recovered_graph)
            aux_losses.update(expert_aux_losses)
            return {"pred": pred, "aux_losses": aux_losses, "oracle_routing": oracle_routing}
        elif self.recipe in {"role_mmoe", "testam_role_mmoe"}:
            pred, aux_losses, routing_meta = self._role_gate(adj, x_enc, y_graph, y_patch, with_residual=True, y_recovered_graph=y_recovered_graph)
            aux_losses.update(expert_aux_losses)
            output = {"pred": pred, "aux_losses": aux_losses, "oracle_routing": routing_meta}
            if "testam" in routing_meta:
                output["testam_routing"] = routing_meta["testam"]
            return output
        else:
            pred, aux_losses = self._soft_fuse(x_enc, y_graph, y_patch)
        aux_losses.update(expert_aux_losses)
        return {"pred": pred, "aux_losses": aux_losses}
