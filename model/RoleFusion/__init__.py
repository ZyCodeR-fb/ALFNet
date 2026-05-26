import importlib
import glob
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
        if self.recipe not in {"hard_route", "soft_fuse", "soft_fuse_v2", "oracle_gate", "role_mmoe"}:
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
        self.residual_scale = float(recipe_cfg.get("residual_scale", 0.5))
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

        if self.recipe in {"soft_fuse", "oracle_gate", "role_mmoe"}:
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
            if self.recipe in {"oracle_gate", "role_mmoe"}:
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
            if self.recipe == "role_mmoe":
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

    def _role_gate(self, x_enc, y_graph, y_patch, with_residual=False):
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
        return fused, aux_losses, {
            "alpha": alpha,
            "y_graph": y_graph.detach(),
            "y_patch": y_patch.detach(),
            "lambda_oracle": self.lambda_oracle,
        }

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
        y_graph, y_patch = self._run_experts(adj, x_enc, x_t_mark, pos_mark, x_dec, dec_t_mark, epoch)
        if self.recipe == "hard_route":
            _, last_role, _, _ = self._extract_role_and_age(x_enc)
            pred = self._hard_route(y_graph, y_patch, last_role)
            return {"pred": pred}
        if self.recipe == "soft_fuse_v2":
            pred, aux_losses = self._soft_fuse_v2(x_enc, y_graph, y_patch)
        elif self.recipe == "oracle_gate":
            pred, aux_losses, oracle_routing = self._role_gate(x_enc, y_graph, y_patch, with_residual=False)
            return {"pred": pred, "aux_losses": aux_losses, "oracle_routing": oracle_routing}
        elif self.recipe == "role_mmoe":
            pred, aux_losses, oracle_routing = self._role_gate(x_enc, y_graph, y_patch, with_residual=True)
            return {"pred": pred, "aux_losses": aux_losses, "oracle_routing": oracle_routing}
        else:
            pred, aux_losses = self._soft_fuse(x_enc, y_graph, y_patch)
        return {"pred": pred, "aux_losses": aux_losses}
