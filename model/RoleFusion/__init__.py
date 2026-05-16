import importlib
import os

import torch
import torch.nn as nn

from args import load_args_from_config
from utils.model_output import unpack_model_output


ROLE_AAS = 0
ROLE_NAS = 1
ROLE_VS = 2
ROLE_FS = 3
ROLE_COUNT = 4


def _resolve_model_module_name(model_name):
    if model_name == "GWNET":
        return "GWNet"
    return model_name


def _load_expert_from_spec(spec, device, expert_name):
    model_name = spec["model"]
    config_path = spec["config_path"]
    checkpoint_path = spec["checkpoint_path"]
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
        if self.recipe not in {"hard_route", "soft_fuse"}:
            raise ValueError(f"Unsupported RoleFusion recipe: {self.recipe}")
        experts_cfg = self.cfg.get("experts", {})
        if "graph" not in experts_cfg or "patch" not in experts_cfg:
            raise ValueError("RoleFusion family.experts must define both graph and patch experts")

        self.graph_expert, self.graph_args, self.graph_checkpoint_path = _load_expert_from_spec(experts_cfg["graph"], configs.device, "graph")
        self.patch_expert, self.patch_args, self.patch_checkpoint_path = _load_expert_from_spec(experts_cfg["patch"], configs.device, "patch")

        self.raw_enc_in = configs.raw_enc_in
        self.output_dim = configs.c_out
        self.pred_len = configs.pred_len
        self.role_start = self.raw_enc_in
        self.role_end = self.role_start + ROLE_COUNT
        self.age_idx = self.role_end

        recipe_cfg = self.cfg.get("recipe", {})
        model_cfg = self.cfg.get("model", {})
        state_hidden_dim = int(model_cfg.get("state_hidden_dim", 32))
        state_dim = int(model_cfg.get("state_dim", 32))
        gate_hidden_dim = int(model_cfg.get("gate_hidden_dim", 32))
        self.lambda_prior = float(recipe_cfg.get("lambda_prior", 0.1))
        self.lambda_entropy = float(recipe_cfg.get("lambda_entropy", 0.01))
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

        if self.recipe == "soft_fuse":
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
        else:
            self.state_encoder = nn.Identity()
            self.fusion_gate = nn.Identity()

        self._freeze_experts()

    def _freeze_experts(self):
        for expert in (self.graph_expert, self.patch_expert):
            expert.eval()
            for param in expert.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        self._freeze_experts()
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

    def _build_gate_features(self, x_enc):
        _, last_role, last_age, visibility_ratio = self._extract_role_and_age(x_enc)
        state_input = torch.cat([last_role, last_age, visibility_ratio], dim=-1)
        return self.state_encoder(state_input), last_role

    def _run_experts(self, adj, x_enc, x_t_mark, pos_mark, x_dec, dec_t_mark, epoch):
        raw_x_enc = x_enc[..., :self.raw_enc_in]
        expert_x_dec = None if x_dec is None else x_dec[..., : self.output_dim]
        with torch.no_grad():
            graph_output = self.graph_expert(adj, raw_x_enc, x_t_mark, pos_mark, expert_x_dec, dec_t_mark, epoch)
            patch_output = self.patch_expert(adj, raw_x_enc, x_t_mark, pos_mark, expert_x_dec, dec_t_mark, epoch)
        y_graph, _ = unpack_model_output(graph_output)
        y_patch, _ = unpack_model_output(patch_output)
        return y_graph, y_patch

    def _hard_route(self, y_graph, y_patch, last_role):
        role_ids = torch.argmax(last_role, dim=-1)
        graph_mask = ((role_ids == ROLE_VS) | (role_ids == ROLE_NAS)).unsqueeze(1).unsqueeze(-1)
        return torch.where(graph_mask, y_graph, y_patch)

    def _soft_fuse(self, x_enc, y_graph, y_patch):
        node_state, last_role = self._build_gate_features(x_enc)
        state_t = node_state.unsqueeze(1).expand(-1, y_graph.size(1), -1, -1)
        gate_input = torch.cat([state_t, y_graph, y_patch, torch.abs(y_graph - y_patch)], dim=-1)
        alpha = torch.sigmoid(self.fusion_gate(gate_input))
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

    def forward(self, adj, x_enc, x_t_mark=None, pos_mark=None, x_dec=None, dec_t_mark=None, epoch=0):
        y_graph, y_patch = self._run_experts(adj, x_enc, x_t_mark, pos_mark, x_dec, dec_t_mark, epoch)
        if self.recipe == "hard_route":
            _, last_role, _, _ = self._extract_role_and_age(x_enc)
            pred = self._hard_route(y_graph, y_patch, last_role)
            return {"pred": pred}
        pred, aux_losses = self._soft_fuse(x_enc, y_graph, y_patch)
        return {"pred": pred, "aux_losses": aux_losses}
