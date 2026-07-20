import glob
import importlib
import os

import torch
import torch.nn as nn

from args import load_args_from_config
from lib.utils import load_project_adjacency, resolve_known_unknown_nodes
from utils.model_output import unpack_model_output


ROLE_AAS = 0
ROLE_NAS = 1
ROLE_VS = 2
ROLE_FS = 3
ROLE_COUNT = 4


def _build_role_scale(role_scale, default_role_scale):
    role_scale = role_scale or {}
    return torch.tensor(
        [
            float(role_scale.get("aas", default_role_scale.get("aas", 1.0))),
            float(role_scale.get("nas", default_role_scale.get("nas", 1.0))),
            float(role_scale.get("vs", default_role_scale.get("vs", 1.0))),
            float(role_scale.get("fs", default_role_scale.get("fs", 1.0))),
        ],
        dtype=torch.float32,
    )


class TemporalConvOutputRefiner(nn.Module):

    def __init__(
        self,
        output_dim,
        hidden_dim,
        kernel_size,
        dilation,
        scale,
        role_scale,
    ):
        super().__init__()
        kernel_size = int(kernel_size)
        dilation = int(dilation)
        if kernel_size % 2 == 0:
            raise ValueError(
                "TemporalConvOutputRefiner requires an odd kernel_size to "
                "preserve prediction length"
            )
        padding = dilation * (kernel_size - 1) // 2
        self.input_proj = nn.Conv1d(
            output_dim,
            hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.output_proj = nn.Conv1d(hidden_dim, output_dim, kernel_size=1)
        self.activation = nn.GELU()
        self.scale = float(scale)
        self.register_buffer(
            "role_output_scale",
            _build_role_scale(
                role_scale,
                {"aas": 1.0, "nas": 1.0, "vs": 1.0, "fs": 1.0},
            ),
            persistent=False,
        )
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, pred, last_role):
        batch, pred_len, num_nodes, output_dim = pred.shape
        x = pred.permute(0, 2, 3, 1).reshape(
            batch * num_nodes,
            output_dim,
            pred_len,
        )
        residual = self.output_proj(self.activation(self.input_proj(x)))
        residual = residual.reshape(
            batch,
            num_nodes,
            output_dim,
            pred_len,
        ).permute(0, 3, 1, 2).contiguous()
        role_scale = torch.sum(
            last_role
            * self.role_output_scale.to(last_role.device).view(
                1, 1, ROLE_COUNT
            ),
            dim=-1,
            keepdim=True,
        )
        residual = (
            role_scale.unsqueeze(1).to(dtype=residual.dtype)
            * torch.tanh(residual)
        )
        return pred + self.scale * residual, residual


class GraphDiffusionOutputRefiner(nn.Module):

    def __init__(
        self,
        output_dim,
        hidden_dim,
        diffusion_steps,
        scale,
        role_scale,
        use_backward=True,
        add_self_loop=True,
    ):
        super().__init__()
        self.diffusion_steps = int(diffusion_steps)
        self.use_backward = bool(use_backward)
        self.add_self_loop = bool(add_self_loop)
        self.scale = float(scale)
        support_count = 1 + self.diffusion_steps * (
            2 if self.use_backward else 1
        )
        self.input_proj = nn.Linear(support_count * output_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.GELU()
        self.register_buffer(
            "role_output_scale",
            _build_role_scale(
                role_scale,
                {"aas": 1.0, "nas": 1.0, "vs": 1.0, "fs": 1.0},
            ),
            persistent=False,
        )
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def _support(self, adj, dtype):
        support = adj.to(dtype=dtype)
        if self.add_self_loop:
            support = support + torch.eye(
                support.size(0),
                device=support.device,
                dtype=dtype,
            )
        forward = support / support.sum(1, keepdim=True).clamp_min(1e-6)
        backward = support.T / support.T.sum(1, keepdim=True).clamp_min(1e-6)
        return forward, backward

    @staticmethod
    def _aggregate(support, x):
        return torch.einsum("nm,btmd->btnd", support, x).contiguous()

    def forward(self, pred, adj, last_role):
        forward, backward = self._support(adj.to(pred.device), pred.dtype)
        features = [pred]
        current_forward = pred
        current_backward = pred
        for _ in range(self.diffusion_steps):
            current_forward = self._aggregate(forward, current_forward)
            features.append(current_forward)
            if self.use_backward:
                current_backward = self._aggregate(backward, current_backward)
                features.append(current_backward)
        residual = self.output_proj(
            self.activation(self.input_proj(torch.cat(features, dim=-1)))
        )
        role_scale = torch.sum(
            last_role
            * self.role_output_scale.to(last_role.device).view(
                1, 1, ROLE_COUNT
            ),
            dim=-1,
            keepdim=True,
        )
        residual = (
            role_scale.unsqueeze(1).to(dtype=residual.dtype)
            * torch.tanh(residual)
        )
        return pred + self.scale * residual, residual


def _resolve_model_module_name(model_name):
    if model_name in {"ALFGraph", "ALFNet", "ALFTemporal"}:
        return f"{model_name}.{model_name}"
    if model_name == "GWNET":
        return "GWNet"
    return model_name


def _load_expert_from_spec(spec, device, expert_name):
    model_name = spec["model"]
    config_path = spec["config_path"]
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"ALFNet {expert_name} expert config not found: {config_path}"
        )
    expert_args = load_args_from_config(
        config_path,
        overrides={"device": device},
    )
    expert_args.model = model_name
    module_name = _resolve_model_module_name(model_name)
    module = importlib.import_module(f"model.{module_name}")
    expert = module.Model(expert_args).float()

    checkpoint_path = spec.get("checkpoint_path", None)
    if not bool(spec.get("load_checkpoint", True)) or checkpoint_path in {
        None,
        "",
    }:
        return expert, expert_args, None
    if any(token in checkpoint_path for token in ["*", "?", "["]):
        matches = glob.glob(checkpoint_path)
        if not matches:
            raise FileNotFoundError(
                f"ALFNet {expert_name} expert checkpoint glob has no matches: "
                f"{checkpoint_path}"
            )
        checkpoint_path = max(matches, key=os.path.getmtime)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"ALFNet {expert_name} expert checkpoint not found: {checkpoint_path}"
        )
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError(
            f"Expert checkpoint at {checkpoint_path} must contain a 'state_dict'"
        )
    expert.load_state_dict(
        checkpoint["state_dict"],
        strict=bool(spec.get("strict_load", True)),
    )
    return expert, expert_args, os.path.normpath(checkpoint_path)


class Model(nn.Module):

    def __init__(self, configs):
        super().__init__()
        if (
            not configs.lifecycle_role_features
            or not configs.lifecycle_history_age_feature
        ):
            raise ValueError(
                "ALFNet requires lifecycle_role_features=true and "
                "lifecycle_history_age_feature=true"
            )
        self.cfg = getattr(configs, "model_family_cfg", {}) or {}
        self.recipe = configs.model_recipe
        if self.recipe != "testam_role_mmoe":
            raise ValueError(
                "ALFNet only retains the final testam_role_mmoe recipe, "
                f"got {self.recipe}."
            )
        experts_cfg = self.cfg.get("experts", {})
        if "graph" not in experts_cfg or "patch" not in experts_cfg:
            raise ValueError(
                "ALFNet family.experts must define both graph and patch experts"
            )
        self.graph_trainable_patterns = list(
            experts_cfg["graph"].get("trainable_param_patterns", []) or []
        )
        self.patch_trainable_patterns = list(
            experts_cfg["patch"].get("trainable_param_patterns", []) or []
        )

        self.graph_expert, self.graph_args, self.graph_checkpoint_path = (
            _load_expert_from_spec(
                experts_cfg["graph"],
                configs.device,
                "graph",
            )
        )
        self.patch_expert, self.patch_args, self.patch_checkpoint_path = (
            _load_expert_from_spec(
                experts_cfg["patch"],
                configs.device,
                "patch",
            )
        )

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
        if str(recipe_cfg.get("fusion_mode", "")).lower() not in {
            "graph_patch_residual",
            "patch_residual",
            "residual",
        }:
            raise ValueError(
                "ALFNet final recipe requires fusion_mode=graph_patch_residual."
            )
        self.patch_residual_mode = str(
            recipe_cfg.get("patch_residual_mode", "gap")
        ).lower()
        if self.patch_residual_mode not in {"gap", "patch_gap", "expert_gap"}:
            raise ValueError(
                "ALFNet final recipe only retains patch_residual_mode=gap."
            )
        if bool(recipe_cfg.get("learnable_patch_residual_budget", False)):
            raise ValueError(
                "ALFNet final recipe uses fixed role budgets, not a learnable budget."
            )
        if not bool(recipe_cfg.get("end_to_end", False)):
            raise ValueError("ALFNet final recipe requires end_to_end=true.")
        if bool(recipe_cfg.get("use_role_residual", False)):
            raise ValueError(
                "ALFNet final recipe no longer includes role residual heads."
            )
        if not bool(recipe_cfg.get("use_graph_expert_diffusion_refiner", False)):
            raise ValueError(
                "ALFNet final recipe requires the graph diffusion output refiner."
            )
        if not bool(recipe_cfg.get("use_patch_expert_temporal_refiner", False)):
            raise ValueError(
                "ALFNet final recipe requires the temporal output refiner."
            )

        state_hidden_dim = int(model_cfg.get("state_hidden_dim", 32))
        state_dim = int(model_cfg.get("state_dim", 32))
        gate_hidden_dim = int(model_cfg.get("gate_hidden_dim", 32))
        self.lambda_oracle = float(recipe_cfg.get("lambda_oracle", 0.1))
        self.lambda_testam = float(recipe_cfg.get("lambda_testam", 0.0))
        self.lambda_role_balanced = float(
            recipe_cfg.get("lambda_role_balanced", 0.0)
        )
        self.lambda_regret = float(recipe_cfg.get("lambda_regret", 0.0))
        self.lambda_route = float(recipe_cfg.get("lambda_route", 0.0))
        self.route_target = str(
            recipe_cfg.get("route_target", "hard_bce")
        ).lower()
        self.route_temperature = float(recipe_cfg.get("route_temperature", 1.0))
        self.route_warmup_epoch = int(recipe_cfg.get("route_warmup_epoch", 0))
        self.route_role_balanced = bool(
            recipe_cfg.get("route_role_balanced", False)
        )
        self.forecast_loss_mode = str(
            recipe_cfg.get("forecast_loss_mode", "standard")
        ).lower()
        self.forecast_role_balanced_beta = float(
            recipe_cfg.get("forecast_role_balanced_beta", 0.5)
        )
        self.forecast_focal_gamma = float(
            recipe_cfg.get("forecast_focal_gamma", 1.0)
        )
        self.forecast_focal_max_weight = float(
            recipe_cfg.get("forecast_focal_max_weight", 5.0)
        )
        self.forecast_focal_beta = float(
            recipe_cfg.get("forecast_focal_beta", 0.5)
        )
        self.lambda_patch_residual = float(
            recipe_cfg.get("lambda_patch_residual", 0.0)
        )
        self.lambda_patch_delta = float(
            recipe_cfg.get("lambda_patch_delta", 0.0)
        )
        self.patch_residual_max_scale = float(
            recipe_cfg.get("patch_residual_max_scale", 0.5)
        )
        self.patch_residual_budget_mode = str(
            recipe_cfg.get("patch_residual_budget_mode", "role")
        ).lower()
        self.end_to_end = True
        self.expert_train_mode = bool(
            recipe_cfg.get("expert_train_mode", True)
        )
        self.graph_expert_lr = float(
            recipe_cfg.get("graph_expert_lr", configs.lr)
        )
        self.patch_expert_lr = float(
            recipe_cfg.get("patch_expert_lr", configs.lr)
        )
        self.gate_lr = float(recipe_cfg.get("gate_lr", configs.lr))
        self.testam_role_balanced = bool(
            recipe_cfg.get("testam_role_balanced", False)
        )
        self.testam_warmup_epoch = int(
            recipe_cfg.get("testam_warmup_epoch", 0)
        )
        self.testam_threshold = float(recipe_cfg.get("testam_threshold", 0.0))

        # Keep legacy buffers so existing rf_alf_sd checkpoints remain strict-loadable.
        recovery_role_scale = recipe_cfg.get(
            "graph_recovery_expert_role_scale", {}
        ) or {}
        self.register_buffer(
            "graph_recovery_expert_role_scale",
            torch.tensor(
                [
                    float(recovery_role_scale.get("aas", 0.0)),
                    float(recovery_role_scale.get("nas", 0.0)),
                    float(recovery_role_scale.get("vs", 0.0)),
                    float(recovery_role_scale.get("fs", 1.0)),
                ],
                dtype=torch.float32,
            ),
        )
        residual_scale = float(recipe_cfg.get("residual_scale", 0.5))
        role_residual_scale = recipe_cfg.get(
            "role_residual_scale",
            {
                "aas": residual_scale,
                "nas": residual_scale,
                "vs": residual_scale,
                "fs": residual_scale,
            },
        )
        self.register_buffer(
            "role_residual_scale",
            torch.tensor(
                [
                    float(role_residual_scale.get("aas", residual_scale)),
                    float(role_residual_scale.get("nas", residual_scale)),
                    float(role_residual_scale.get("vs", residual_scale)),
                    float(role_residual_scale.get("fs", residual_scale)),
                ],
                dtype=torch.float32,
            ),
        )
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
        alpha_floor = recipe_cfg.get(
            "role_graph_alpha_floor",
            {"aas": 0.0, "nas": 0.0, "vs": 0.0, "fs": 0.0},
        )
        self.register_buffer(
            "role_graph_alpha_floor",
            torch.tensor(
                [
                    float(alpha_floor.get("aas", 0.0)),
                    float(alpha_floor.get("nas", 0.0)),
                    float(alpha_floor.get("vs", 0.0)),
                    float(alpha_floor.get("fs", 0.0)),
                ],
                dtype=torch.float32,
            ),
        )
        alpha_ceiling = recipe_cfg.get(
            "role_graph_alpha_ceiling",
            {"aas": 1.0, "nas": 1.0, "vs": 1.0, "fs": 1.0},
        )
        self.register_buffer(
            "role_graph_alpha_ceiling",
            torch.tensor(
                [
                    float(alpha_ceiling.get("aas", 1.0)),
                    float(alpha_ceiling.get("nas", 1.0)),
                    float(alpha_ceiling.get("vs", 1.0)),
                    float(alpha_ceiling.get("fs", 1.0)),
                ],
                dtype=torch.float32,
            ),
        )
        patch_role_scale = recipe_cfg.get(
            "patch_residual_role_scale",
            {"aas": 0.3, "nas": 0.1, "vs": 0.05, "fs": 0.35},
        )
        self.register_buffer(
            "patch_residual_role_scale",
            torch.tensor(
                [
                    float(patch_role_scale.get("aas", 0.3)),
                    float(patch_role_scale.get("nas", 0.1)),
                    float(patch_role_scale.get("vs", 0.05)),
                    float(patch_role_scale.get("fs", 0.35)),
                ],
                dtype=torch.float32,
            ),
        )
        self.register_parameter("patch_residual_budget_logit", None)
        patch_delta_role_weight = recipe_cfg.get(
            "patch_delta_role_weight",
            patch_role_scale,
        )
        self.register_buffer(
            "patch_delta_role_weight",
            torch.tensor(
                [
                    float(
                        patch_delta_role_weight.get(
                            "aas", patch_role_scale.get("aas", 0.3)
                        )
                    ),
                    float(
                        patch_delta_role_weight.get(
                            "nas", patch_role_scale.get("nas", 0.1)
                        )
                    ),
                    float(
                        patch_delta_role_weight.get(
                            "vs", patch_role_scale.get("vs", 0.05)
                        )
                    ),
                    float(
                        patch_delta_role_weight.get(
                            "fs", patch_role_scale.get("fs", 0.35)
                        )
                    ),
                ],
                dtype=torch.float32,
            ),
        )

        self.state_encoder = nn.Sequential(
            nn.Linear(ROLE_COUNT + 2, state_hidden_dim),
            nn.ReLU(),
            nn.Linear(state_hidden_dim, state_dim),
            nn.ReLU(),
        )
        # Unused by the final residual path, retained for checkpoint compatibility.
        self.fusion_gate = nn.Sequential(
            nn.Linear(state_dim + 3, gate_hidden_dim),
            nn.ReLU(),
            nn.Linear(gate_hidden_dim, 1),
        )
        gate_input_dim = state_dim + 3 * self.output_dim
        self.role_fusion_gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(gate_input_dim, gate_hidden_dim),
                nn.ReLU(),
                nn.Linear(gate_hidden_dim, 1),
            )
            for _ in range(ROLE_COUNT)
        ])

        graph_refiner_hidden_dim = int(
            model_cfg.get("graph_refiner_hidden_dim", gate_hidden_dim)
        )
        self.graph_expert_diffusion_refiner = GraphDiffusionOutputRefiner(
            output_dim=self.output_dim,
            hidden_dim=graph_refiner_hidden_dim,
            diffusion_steps=int(recipe_cfg.get("graph_diffusion_steps", 1)),
            scale=float(recipe_cfg.get("graph_diffusion_scale", 0.05)),
            role_scale=recipe_cfg.get("graph_diffusion_role_scale", {}),
            use_backward=bool(
                recipe_cfg.get("graph_diffusion_use_backward", True)
            ),
            add_self_loop=bool(
                recipe_cfg.get("graph_diffusion_add_self_loop", True)
            ),
        )
        temporal_hidden_dim = int(
            model_cfg.get("temporal_refiner_hidden_dim", gate_hidden_dim)
        )
        self.patch_expert_temporal_refiner = TemporalConvOutputRefiner(
            output_dim=self.output_dim,
            hidden_dim=temporal_hidden_dim,
            kernel_size=int(recipe_cfg.get("temporal_refiner_kernel_size", 3)),
            dilation=int(recipe_cfg.get("temporal_refiner_dilation", 1)),
            scale=float(recipe_cfg.get("temporal_refiner_scale", 0.05)),
            role_scale=recipe_cfg.get("temporal_refiner_role_scale", {}),
        )
        self._configure_expert_trainability()

    def _set_expert_trainability(self, expert, patterns):
        for param in expert.parameters():
            param.requires_grad = True
        expert.train(self.training and self.expert_train_mode)

    def _configure_expert_trainability(self):
        for expert, patterns in (
            (self.graph_expert, self.graph_trainable_patterns),
            (self.patch_expert, self.patch_trainable_patterns),
        ):
            self._set_expert_trainability(expert, patterns)

    def train(self, mode=True):
        super().train(mode)
        self._configure_expert_trainability()
        return self

    def get_optimizer_param_groups(self, base_lr):
        base_lr = float(base_lr)
        graph_lr = self.graph_expert_lr or base_lr
        patch_lr = self.patch_expert_lr or base_lr
        gate_lr = self.gate_lr or base_lr
        graph_params = [
            param for param in self.graph_expert.parameters()
            if param.requires_grad
        ]
        patch_params = [
            param for param in self.patch_expert.parameters()
            if param.requires_grad
        ]
        expert_param_ids = {id(param) for param in graph_params}
        expert_param_ids.update(id(param) for param in patch_params)
        routing_params = [
            param
            for param in self.parameters()
            if param.requires_grad and id(param) not in expert_param_ids
        ]
        groups = []
        if graph_params:
            groups.append({"params": graph_params, "lr": graph_lr})
        if patch_params:
            groups.append({"params": patch_params, "lr": patch_lr})
        if routing_params:
            groups.append({"params": routing_params, "lr": gate_lr})
        return groups

    def _extract_role_and_age(self, x_enc):
        if x_enc.size(-1) <= self.age_idx:
            raise ValueError(
                "ALFNet expects raw + role-onehot + history-age channels, "
                f"got {x_enc.size(-1)} total channels"
            )
        role_seq = x_enc[..., self.role_start:self.role_end]
        history_age_seq = x_enc[..., self.age_idx:self.age_idx + 1]
        last_role = role_seq[:, -1]
        last_age = history_age_seq[:, -1]
        visibility_ratio = (
            role_seq[..., ROLE_AAS] + role_seq[..., ROLE_NAS]
        ).mean(dim=1).unsqueeze(-1)
        return role_seq, last_role, last_age, visibility_ratio

    def _resolve_current_node_ids(self, current_num_nodes):
        if current_num_nodes == len(self.known_nodes):
            return self.known_nodes
        if current_num_nodes == self.full_num_nodes:
            return list(range(self.full_num_nodes))
        return list(range(current_num_nodes))

    def _format_graph_expert_input(self, raw_graph_x, x_enc):
        graph_enc_in = int(
            getattr(self.graph_args, "enc_in", self.raw_enc_in)
        )
        if graph_enc_in <= self.raw_enc_in:
            return raw_graph_x
        if x_enc.size(-1) < graph_enc_in:
            raise ValueError(
                f"ALFNet graph expert expects {graph_enc_in} input channels, "
                f"but ALFNet input has {x_enc.size(-1)} channels."
            )
        return torch.cat([
            raw_graph_x,
            x_enc[..., self.raw_enc_in:graph_enc_in],
        ], dim=-1)

    def _run_experts(
        self,
        adj,
        x_enc,
        x_t_mark,
        pos_mark,
        x_dec,
        dec_t_mark,
        epoch,
    ):
        raw_x_enc = x_enc[..., :self.raw_enc_in]
        expert_x_dec = (
            None if x_dec is None else x_dec[..., :self.output_dim]
        )
        current_node_ids = self._resolve_current_node_ids(raw_x_enc.size(2))
        graph_x_enc = self._format_graph_expert_input(raw_x_enc, x_enc)
        if hasattr(self.graph_expert, "set_current_node_ids"):
            self.graph_expert.set_current_node_ids(current_node_ids)
        graph_output = self.graph_expert(
            adj,
            graph_x_enc,
            x_t_mark,
            pos_mark,
            expert_x_dec,
            dec_t_mark,
            epoch,
        )
        patch_output = self.patch_expert(
            adj,
            raw_x_enc,
            x_t_mark,
            pos_mark,
            expert_x_dec,
            dec_t_mark,
            epoch,
        )
        y_graph, _ = unpack_model_output(graph_output)
        y_patch, _ = unpack_model_output(patch_output)
        return y_graph, y_patch

    def _patch_residual_role_budget(self, last_role, visibility_ratio=None):
        if self.patch_residual_budget_mode in {
            "visibility",
            "visibility_ratio",
            "availability",
        }:
            if visibility_ratio is None:
                raise ValueError(
                    "visibility patch residual budget requires visibility_ratio"
                )
            return self.patch_residual_max_scale * visibility_ratio.clamp(
                0.0, 1.0
            )
        role_scale = self.patch_residual_role_scale.to(
            device=last_role.device,
            dtype=last_role.dtype,
        )
        return torch.sum(
            last_role * role_scale.view(1, 1, ROLE_COUNT),
            dim=-1,
            keepdim=True,
        )

    def _graph_patch_residual(self, adj, x_enc, y_graph, y_patch):
        _, last_role, last_age, visibility_ratio = self._extract_role_and_age(
            x_enc
        )
        node_state = self.state_encoder(
            torch.cat([last_role, last_age, visibility_ratio], dim=-1)
        )
        y_graph, _ = self.graph_expert_diffusion_refiner(
            y_graph,
            adj,
            last_role,
        )
        y_patch, _ = self.patch_expert_temporal_refiner(y_patch, last_role)

        state_t = node_state.unsqueeze(1).expand(
            -1, y_graph.size(1), -1, -1
        )
        disagreement = torch.abs(y_graph - y_patch)
        gate_input = torch.cat([
            state_t,
            y_graph,
            y_patch,
            disagreement,
        ], dim=-1)
        role_logits = torch.cat(
            [gate(gate_input) for gate in self.role_fusion_gates],
            dim=-1,
        )
        role_weights = last_role.unsqueeze(1).expand(
            -1, y_graph.size(1), -1, -1
        )
        patch_gate = torch.sigmoid(
            torch.sum(role_logits * role_weights, dim=-1, keepdim=True)
        )
        role_scale = self._patch_residual_role_budget(
            last_role,
            visibility_ratio,
        )
        role_scale_t = role_scale.unsqueeze(1).expand_as(patch_gate)
        beta = patch_gate * role_scale_t
        patch_delta = y_patch - y_graph
        residual = beta * patch_delta
        fused = y_graph + residual
        residual_candidate = y_graph + role_scale_t * patch_delta

        aux_losses = {}
        if self.lambda_patch_residual > 0:
            aux_losses["ALFNet_patch_residual"] = (
                self.lambda_patch_residual * residual.pow(2).mean()
            )
        alpha = 1.0 - beta
        routing_meta = {
            "alpha": alpha,
            "y_graph": y_graph.detach(),
            "y_patch": residual_candidate.detach(),
            "lambda_oracle": self.lambda_oracle,
            "lambda_role_balanced": self.lambda_role_balanced,
            "lambda_regret": self.lambda_regret,
        }
        if self.lambda_patch_delta > 0:
            routing_meta["patch_residual_training"] = {
                "residual": residual,
                "y_graph": y_graph.detach(),
                "lambda_patch_delta": self.lambda_patch_delta,
                "role_weight": self.patch_delta_role_weight,
            }
        routing_meta["testam"] = {
            "gate": torch.cat([alpha, 1.0 - alpha], dim=-1),
            "expert_preds": torch.stack([
                y_graph.detach(),
                residual_candidate.detach(),
            ], dim=-1),
            "lambda_testam": self.lambda_testam,
            "role_balanced": self.testam_role_balanced,
            "warmup_epoch": self.testam_warmup_epoch,
            "threshold": self.testam_threshold,
        }
        if self.lambda_route > 0:
            routing_meta["route"] = {
                "alpha": alpha,
                "y_graph": y_graph.detach(),
                "y_patch": residual_candidate.detach(),
                "lambda_route": self.lambda_route,
                "target": self.route_target,
                "temperature": self.route_temperature,
                "warmup_epoch": self.route_warmup_epoch,
                "role_balanced": self.route_role_balanced,
            }
        return fused, aux_losses, routing_meta

    def forward(
        self,
        adj,
        x_enc,
        x_t_mark=None,
        pos_mark=None,
        x_dec=None,
        dec_t_mark=None,
        epoch=0,
    ):
        y_graph, y_patch = self._run_experts(
            adj,
            x_enc,
            x_t_mark,
            pos_mark,
            x_dec,
            dec_t_mark,
            epoch,
        )
        pred, aux_losses, routing_meta = self._graph_patch_residual(
            adj,
            x_enc,
            y_graph,
            y_patch,
        )
        patch_residual_training = routing_meta.pop(
            "patch_residual_training",
            None,
        )
        output = {
            "pred": pred,
            "aux_losses": aux_losses,
            "oracle_routing": routing_meta,
        }
        if self.forecast_loss_mode != "standard":
            output["forecast_loss_mode"] = self.forecast_loss_mode
            output["forecast_role_balanced_beta"] = (
                self.forecast_role_balanced_beta
            )
            output["forecast_focal_gamma"] = self.forecast_focal_gamma
            output["forecast_focal_max_weight"] = (
                self.forecast_focal_max_weight
            )
            output["forecast_focal_beta"] = self.forecast_focal_beta
        output["testam_routing"] = routing_meta["testam"]
        if "route" in routing_meta:
            output["route_routing"] = routing_meta["route"]
        if patch_residual_training is not None:
            output["patch_residual_training"] = patch_residual_training
        return output
