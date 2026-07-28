import importlib
import os

import torch
import torch.nn as nn

from args import load_args_from_config
from layers.evidence_router import (
    EvidenceRouter,
    ObservedNLinear,
    mask_shape,
    neighbour_evidence,
    routing_diagnostics,
    self_evidence,
)
from layers.source_grin import GRINInputImputer
from lib.utils import load_project_adjacency, resolve_known_unknown_nodes
from utils.model_output import unpack_model_output


AVAILABILITY_COUNT = 4
# Availability one-hot order, fixed by _role_tensor and the loader.
ROLE_FS = 3


def _role_tensor(values, defaults):
    values = values or {}
    return torch.tensor(
        [float(values.get(name, defaults[name])) for name in ("aas", "nas", "vs", "fs")],
        dtype=torch.float32,
    )


class TemporalOutputRefiner(nn.Module):

    def __init__(self, output_dim, hidden_dim, kernel_size, dilation, scale, role_scale):
        super().__init__()
        if int(kernel_size) % 2 == 0:
            raise ValueError("TemporalOutputRefiner requires an odd kernel_size")
        padding = int(dilation) * (int(kernel_size) - 1) // 2
        self.input_proj = nn.Conv1d(
            output_dim,
            hidden_dim,
            kernel_size=int(kernel_size),
            padding=padding,
            dilation=int(dilation),
        )
        self.output_proj = nn.Conv1d(hidden_dim, output_dim, kernel_size=1)
        self.activation = nn.GELU()
        self.scale = float(scale)
        self.register_buffer(
            "role_scale",
            _role_tensor(role_scale, {"aas": 1.0, "nas": 1.0, "vs": 1.0, "fs": 1.0}),
            persistent=False,
        )
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, prediction, last_availability):
        batch, pred_len, num_nodes, output_dim = prediction.shape
        hidden = prediction.permute(0, 2, 3, 1).reshape(
            batch * num_nodes,
            output_dim,
            pred_len,
        )
        residual = self.output_proj(self.activation(self.input_proj(hidden)))
        residual = residual.reshape(batch, num_nodes, output_dim, pred_len)
        residual = residual.permute(0, 3, 1, 2).contiguous()
        scale = torch.sum(
            last_availability * self.role_scale.to(last_availability).view(1, 1, -1),
            dim=-1,
            keepdim=True,
        )
        return prediction + self.scale * scale.unsqueeze(1) * torch.tanh(residual)


class GraphOutputRefiner(nn.Module):

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
        support_count = 1 + self.diffusion_steps * (2 if self.use_backward else 1)
        self.input_proj = nn.Linear(support_count * output_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.GELU()
        self.register_buffer(
            "role_scale",
            _role_tensor(role_scale, {"aas": 1.0, "nas": 1.0, "vs": 1.0, "fs": 1.0}),
            persistent=False,
        )
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def _supports(self, adjacency, dtype):
        support = adjacency.to(dtype=dtype)
        if self.add_self_loop:
            support = support + torch.eye(
                support.size(0),
                device=support.device,
                dtype=dtype,
            )
        forward = support / support.sum(1, keepdim=True).clamp_min(1e-6)
        backward_support = support.T
        backward = backward_support / backward_support.sum(1, keepdim=True).clamp_min(1e-6)
        return forward, backward

    @staticmethod
    def _aggregate(support, value):
        return torch.einsum("nm,btmd->btnd", support, value).contiguous()

    def forward(self, prediction, adjacency, last_availability):
        forward, backward = self._supports(adjacency.to(prediction.device), prediction.dtype)
        features = [prediction]
        current_forward = prediction
        current_backward = prediction
        for _ in range(self.diffusion_steps):
            current_forward = self._aggregate(forward, current_forward)
            features.append(current_forward)
            if self.use_backward:
                current_backward = self._aggregate(backward, current_backward)
                features.append(current_backward)
        residual = self.output_proj(
            self.activation(self.input_proj(torch.cat(features, dim=-1)))
        )
        scale = torch.sum(
            last_availability * self.role_scale.to(last_availability).view(1, 1, -1),
            dim=-1,
            keepdim=True,
        )
        return prediction + self.scale * scale.unsqueeze(1) * torch.tanh(residual)


def _expert_module_name(model_name):
    if model_name in {"ALFGraph", "ALFTemporal"}:
        return f"{model_name}.{model_name}"
    if model_name == "GWNET":
        return "GWNet"
    return model_name


def _build_expert(spec, device, expert_name):
    config_path = spec["config_path"]
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"ALFNet {expert_name} config not found: {config_path}")
    expert_args = load_args_from_config(config_path, overrides={"device": device})
    expert_args.model = spec["model"]
    module = importlib.import_module(f"model.{_expert_module_name(spec['model'])}")
    return module.Model(expert_args).float(), expert_args


class Model(nn.Module):

    def __init__(self, configs):
        super().__init__()
        if not (
            configs.lifecycle_role_features
            and configs.lifecycle_history_age_feature
            and configs.lifecycle_observation_mask_feature
        ):
            raise ValueError("ALFNet requires availability, state-age, and observation-mask inputs")
        if configs.lifecycle_training_strategy != "lcbt_v3":
            raise ValueError("ALFNet streamlined mainline requires lifecycle_training_strategy=lcbt_v3")

        family = getattr(configs, "model_family_cfg", {}) or {}
        experts = family.get("experts", {})
        if set(experts) != {"graph", "patch"}:
            raise ValueError("ALFNet requires exactly graph and patch experts")
        self.graph_expert, self.graph_args = _build_expert(
            experts["graph"], configs.device, "graph"
        )
        self.patch_expert, self.patch_args = _build_expert(
            experts["patch"], configs.device, "patch"
        )
        self.graph_input_dim = int(self.graph_args.enc_in)
        # The patch expert declares how much lifecycle context it consumes, so
        # its ablation switch lives in the ALFTemporal config and needs no
        # duplicated wiring here.
        self.patch_input_dim = int(
            getattr(self.patch_expert, "expert_input_dim", configs.raw_enc_in)
        )
        if self.patch_input_dim > int(configs.enc_in):
            raise ValueError(
                f"patch expert requests {self.patch_input_dim} channels but the "
                f"lifecycle input only provides {int(configs.enc_in)}"
            )

        recipe = family.get("recipe", {})
        model_cfg = family.get("model", {})
        self.raw_enc_in = int(configs.raw_enc_in)
        self.output_dim = int(configs.c_out)
        self.role_start = self.raw_enc_in
        self.role_end = self.role_start + AVAILABILITY_COUNT
        self.age_index = self.role_end
        self.mask_index = self.age_index + 1
        self.forecast_role_balanced_beta = float(
            recipe.get("forecast_role_balanced_beta", 0.10)
        )
        # "residual_gate": both experts decode, an availability gate blends them
        #   (original, kept as the low-risk fallback).
        # "film": the patch branch emits a per-node condition vector that
        #   modulates the ALFGraph encoder state; the prediction is the graph
        #   expert's. The gate, the route BCE and the regret term all describe a
        #   two-forecast blend that no longer exists, so they are switched off
        #   together rather than left dangling at weight zero.
        self.fusion_mode = str(recipe.get("fusion_mode", "residual_gate"))
        if self.fusion_mode not in {"residual_gate", "film"}:
            raise ValueError(f"unsupported ALFNet fusion_mode: {self.fusion_mode}")
        self.film_fusion = self.fusion_mode == "film"
        if self.film_fusion and getattr(self.patch_expert, "emit", "prediction") != "features":
            raise ValueError(
                "fusion_mode=film requires the patch expert to run with "
                "family.model.emit=features; otherwise it still builds a "
                "regression head whose output nothing consumes"
            )
        if not self.film_fusion and getattr(self.patch_expert, "emit", "prediction") == "features":
            raise ValueError(
                "the patch expert emits features but fusion_mode is "
                f"{self.fusion_mode!r}, which has no forecast to blend"
            )
        self.lambda_route = 0.0 if self.film_fusion else float(recipe.get("lambda_route", 0.05))
        self.lambda_regret = 0.0 if self.film_fusion else float(recipe.get("lambda_regret", 0.03))
        self.route_warmup_epoch = int(recipe.get("route_warmup_epoch", 0))
        self.route_role_balanced = bool(recipe.get("route_role_balanced", False))
        self.graph_expert_lr = float(recipe.get("graph_expert_lr", configs.lr))
        self.patch_expert_lr = float(recipe.get("patch_expert_lr", configs.lr))
        self.gate_lr = float(recipe.get("gate_lr", configs.lr))

        graph_recipe = (getattr(self.graph_args, "model_family_cfg", {}) or {}).get(
            "recipe", {}
        )
        self.shared_input_imputer = GRINInputImputer(configs, graph_recipe)

        state_hidden_dim = int(model_cfg.get("state_hidden_dim", 32))
        state_dim = int(model_cfg.get("state_dim", 32))
        gate_hidden_dim = int(model_cfg.get("gate_hidden_dim", 32))
        self.state_encoder = nn.Sequential(
            nn.Linear(AVAILABILITY_COUNT + 2, state_hidden_dim),
            nn.ReLU(),
            nn.Linear(state_hidden_dim, state_dim),
            nn.ReLU(),
        )
        gate_input_dim = state_dim + 3 * self.output_dim
        self.availability_gates = None if self.film_fusion else nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(gate_input_dim, gate_hidden_dim),
                    nn.ReLU(),
                    nn.Linear(gate_hidden_dim, 1),
                )
                for _ in range(AVAILABILITY_COUNT)
            ]
        )
        # The FiLM head is the only module this plan adds: two affine maps from
        # the patch branch's per-node condition vector to (gamma, beta) over the
        # DCRNN hidden state. gamma is centred at 1 and beta at 0 so an
        # untrained head is the identity and cannot destabilise the graph
        # expert at initialisation.
        self.film_head = None
        if self.film_fusion:
            hidden_size = int(self.graph_args.d_model)
            self.film_head = nn.Sequential(
                nn.Linear(int(self.patch_expert.feature_dim), gate_hidden_dim),
                nn.ReLU(),
                nn.Linear(gate_hidden_dim, 2 * hidden_size),
            )
            nn.init.zeros_(self.film_head[-1].weight)
            nn.init.zeros_(self.film_head[-1].bias)

        # FiLM is multiplicative: it can reweight the encoder state but cannot
        # inject a forecast component the graph expert never produced. Measured
        # cost of that on seed 32: Fault MAE 3.8337 (two-expert blend) -> 4.3613
        # (fg1). This head restores an additive correction on failed sensors
        # only, reusing the patch condition features rather than rebuilding the
        # second decoder. tanh bounds it and the zero-initialised output layer
        # makes an untrained model identical to fg1.
        # C: evidence-aware router. Off -> fg1's FiLM path is untouched.
        self.evidence_router_enabled = bool(recipe.get("evidence_router", False))
        self.detach_imputer_for_temporal = bool(
            recipe.get("detach_imputer_for_temporal", True)
        )
        # D: observed-history NLinear residual. Usable with or without C.
        self.observed_nlinear_enabled = bool(recipe.get("observed_nlinear", False))
        self.observed_residual_scale = float(recipe.get("observed_residual_scale", 0.35))
        self.pred_len = int(configs.pred_len)
        self.input_len = int(configs.input_len)
        if (self.evidence_router_enabled or self.observed_nlinear_enabled) and not self.film_fusion:
            raise ValueError(
                "evidence_router and observed_nlinear extend the FiLM-conditioned "
                "path; enable fusion_mode=film first"
            )
        self.observed_nlinear = None
        if self.observed_nlinear_enabled:
            self.observed_nlinear = ObservedNLinear(
                self.input_len, self.pred_len, self.output_dim
            )
        self.patch_projection = None
        self.evidence_router = None
        if self.evidence_router_enabled:
            # C must run without D: a light projection turns the existing patch
            # condition features into Y_patch.
            self.patch_projection = nn.Sequential(
                nn.Linear(int(self.patch_expert.feature_dim), gate_hidden_dim),
                nn.ReLU(),
                nn.Linear(gate_hidden_dim, self.pred_len * self.output_dim),
            )
            nn.init.zeros_(self.patch_projection[-1].weight)
            nn.init.zeros_(self.patch_projection[-1].bias)
            self.evidence_router = EvidenceRouter(
                self.output_dim,
                self.pred_len,
                AVAILABILITY_COUNT,
                hidden_dim=gate_hidden_dim,
                self_evidence_floor=float(recipe.get("self_evidence_floor", 0.02)),
            )
        self.last_routing_diagnostics = None
        self._last_temporal = None
        # film mode zeroes lambda_route because there was no second forecast to
        # route between; the evidence router restores one, so it gets its own.
        self.lambda_route_evidence = (
            float(recipe.get("lambda_route", 0.05)) if self.evidence_router_enabled else 0.0
        )

        self.fault_residual_scale = float(recipe.get("fault_residual_scale", 0.0))
        if self.fault_residual_scale and not self.film_fusion:
            raise ValueError(
                "fault_residual_scale corrects a FiLM-conditioned forecast; "
                "residual_gate mode already blends a second forecast"
            )
        self.fault_residual_head = None
        if self.fault_residual_scale:
            self.fault_residual_head = nn.Sequential(
                nn.Linear(int(self.patch_expert.feature_dim), gate_hidden_dim),
                nn.ReLU(),
                nn.Linear(gate_hidden_dim, int(configs.pred_len) * self.output_dim),
            )
            nn.init.zeros_(self.fault_residual_head[-1].weight)
            nn.init.zeros_(self.fault_residual_head[-1].bias)
        # NAS is the role every Join node carries at the last input step. A zero
        # budget makes beta identically zero there, which switches the patch
        # expert (and the route/regret gradients) off for the whole Join track.
        if not self.film_fusion:
            self.register_buffer(
                "patch_budget",
                _role_tensor(
                    recipe.get("patch_residual_role_scale", {}),
                    {"aas": 0.20, "nas": 0.20, "vs": 0.0, "fs": 0.35},
                ),
                persistent=False,
            )
        elif recipe.get("patch_residual_role_scale"):
            raise ValueError(
                "patch_residual_role_scale gates a residual that fusion_mode=film "
                "does not produce; remove it from the recipe"
            )
        self.graph_refiner = GraphOutputRefiner(
            self.output_dim,
            int(model_cfg.get("graph_refiner_hidden_dim", gate_hidden_dim)),
            int(recipe.get("graph_diffusion_steps", 1)),
            float(recipe.get("graph_diffusion_scale", 0.05)),
            recipe.get("graph_diffusion_role_scale", {}),
            bool(recipe.get("graph_diffusion_use_backward", True)),
            bool(recipe.get("graph_diffusion_add_self_loop", True)),
        )
        self.temporal_refiner = None if self.film_fusion else TemporalOutputRefiner(
            self.output_dim,
            int(model_cfg.get("temporal_refiner_hidden_dim", gate_hidden_dim)),
            int(recipe.get("temporal_refiner_kernel_size", 3)),
            int(recipe.get("temporal_refiner_dilation", 1)),
            float(recipe.get("temporal_refiner_scale", 0.05)),
            recipe.get("temporal_refiner_role_scale", {}),
        )

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

    def get_optimizer_param_groups(self, base_lr):
        graph_params = list(self.graph_expert.parameters()) + list(
            self.shared_input_imputer.parameters()
        )
        patch_params = list(self.patch_expert.parameters())
        expert_ids = {id(parameter) for parameter in graph_params + patch_params}
        fusion_params = [
            parameter
            for parameter in self.parameters()
            if id(parameter) not in expert_ids
        ]
        return [
            {"params": graph_params, "lr": self.graph_expert_lr},
            {"params": patch_params, "lr": self.patch_expert_lr},
            {"params": fusion_params, "lr": self.gate_lr or float(base_lr)},
        ]

    def _state(self, x_enc):
        availability = x_enc[..., self.role_start:self.role_end]
        last_availability = availability[:, -1]
        last_age = x_enc[:, -1, :, self.age_index:self.age_index + 1]
        visibility = x_enc[..., self.mask_index:self.mask_index + 1].mean(dim=1)
        state = self.state_encoder(
            torch.cat([last_availability, last_age, visibility], dim=-1)
        )
        return state, last_availability

    def _evidence_context(self, x_enc, adjacency, last_availability, state):
        """Everything the router reads, all of it known at prediction time."""
        observation_mask = x_enc[..., self.mask_index:self.mask_index + 1]
        e_self = self_evidence(observation_mask)
        e_nbr = neighbour_evidence(adjacency, observation_mask)
        return {
            "availability": last_availability,
            "state_age": x_enc[:, -1, :, self.age_index:self.age_index + 1],
            "visibility": observation_mask.mean(dim=1),
            "e_self": e_self,
            "e_nbr": e_nbr,
            "mask_shape": mask_shape(observation_mask),
        }, observation_mask, e_self

    def _evidence_fusion(self, y_graph, patch_features, x_enc, adjacency,
                         last_availability, state):
        """Yt = Y_patch + g_self * Y_self, then one beta over the graph anchor.

        `Y_self` is gated by the node's own evidence -- with no measurements the
        NLinear has nothing to project. `Y_patch` deliberately is not: Unseen
        and Outage both have e_self = 0, and gating the patch path on e_self
        would switch off exactly the correction that the fault track shows is
        worth up to -2.35 MAE. The router separates the two cases from the
        availability context, state_age and the neighbourhood evidence instead.
        """
        context, observation_mask, e_self = self._evidence_context(
            x_enc, adjacency, last_availability, state
        )
        y_temporal = y_graph
        if self.patch_projection is not None:
            projected = self.patch_projection(patch_features)
            y_temporal = y_graph + projected.reshape(
                y_graph.shape[0], y_graph.shape[2], self.pred_len, self.output_dim
            ).permute(0, 2, 1, 3)
        if self.observed_nlinear is not None:
            raw = x_enc[..., :self.output_dim]
            y_self = self.observed_nlinear(raw, observation_mask)
            gate = (
                self.evidence_router.self_gate(e_self)
                if self.evidence_router is not None
                else (e_self / 0.02).clamp(0.0, 1.0)
            ).unsqueeze(1)
            y_temporal = y_temporal + gate * (y_self - y_graph) * (
                1.0 if self.evidence_router is not None else self.observed_residual_scale
            )
        self._last_temporal = y_temporal
        if self.evidence_router is None:
            # D alone: a bounded observed-history residual on the graph output.
            return y_temporal, None
        prediction, beta = self.evidence_router(y_graph, y_temporal, context)
        if not self.training:
            self.last_routing_diagnostics = {
                "beta_mean": float(beta.mean()),
                "beta_std": float(beta.std()),
            }
        return prediction, beta

    def _current_node_ids(self, node_count):
        if node_count == len(self.known_nodes):
            return self.known_nodes
        if node_count == self.full_num_nodes:
            return list(range(self.full_num_nodes))
        return list(range(node_count))

    def _run_experts(
        self,
        adjacency,
        x_enc,
        x_t_mark,
        pos_mark,
        x_dec,
        dec_t_mark,
        epoch,
    ):
        filled = self.shared_input_imputer(adjacency, x_enc, x_t_mark, pos_mark)
        # filled is [imputed_raw, availability(4), state_age, observation_mask];
        # each expert takes the prefix it declares.
        graph_input = filled[..., :self.graph_input_dim]
        # The shared GRIN is in the graph parameter group. Letting the temporal
        # branch push gradients back into it is the mechanism that cost fg1 its
        # Unseen accuracy relative to the two-expert model; detaching keeps the
        # imputer optimised for the path that depends on it.
        patch_source = filled.detach() if (
            self.evidence_router_enabled and self.detach_imputer_for_temporal
        ) else filled
        patch_input = patch_source[..., :self.patch_input_dim]
        if hasattr(self.graph_expert, "set_current_node_ids"):
            self.graph_expert.set_current_node_ids(self._current_node_ids(filled.size(2)))
        decoder = None if x_dec is None else x_dec[..., :self.output_dim]
        patch_output = self.patch_expert(
            adjacency, patch_input, x_t_mark, pos_mark, decoder, dec_t_mark, epoch
        )
        if self.film_fusion:
            # One forecast, one decoder. The patch branch only conditions it.
            features = patch_output["features"]
            gamma, beta = self.film_head(features).chunk(2, dim=-1)
            graph_output = self.graph_expert(
                adjacency, graph_input, x_t_mark, pos_mark, decoder, dec_t_mark,
                epoch, film=(1.0 + gamma, beta),
            )
            y_graph, graph_meta = unpack_model_output(graph_output)
            return y_graph, features, graph_meta.get("aux_losses")
        graph_output = self.graph_expert(
            adjacency, graph_input, x_t_mark, pos_mark, decoder, dec_t_mark, epoch
        )
        y_graph, graph_meta = unpack_model_output(graph_output)
        y_patch, _ = unpack_model_output(patch_output)
        return y_graph, y_patch, graph_meta.get("aux_losses")

    def forward(
        self,
        adjacency,
        x_enc,
        x_t_mark=None,
        pos_mark=None,
        x_dec=None,
        dec_t_mark=None,
        epoch=0,
    ):
        y_graph, y_patch, expert_aux_losses = self._run_experts(
            adjacency,
            x_enc,
            x_t_mark,
            pos_mark,
            x_dec,
            dec_t_mark,
            epoch,
        )
        state, last_availability = self._state(x_enc)
        y_graph = self.graph_refiner(y_graph, adjacency, last_availability)
        if self.film_fusion:
            prediction = y_graph
            if self.evidence_router_enabled or self.observed_nlinear_enabled:
                prediction, beta = self._evidence_fusion(
                    prediction, y_patch, x_enc, adjacency, last_availability, state
                )
                if self.evidence_router_enabled:
                    return {
                        "pred": prediction,
                        "aux_losses": expert_aux_losses,
                        "forecast_loss_mode": "role_balanced_mix",
                        "forecast_role_balanced_beta": self.forecast_role_balanced_beta,
                        "route_routing": {
                            "alpha": 1.0 - beta,
                            "y_graph": y_graph.detach(),
                            "y_patch": self._last_temporal.detach(),
                            "lambda_route": float(self.lambda_route_evidence),
                            "warmup_epoch": self.route_warmup_epoch,
                            "role_balanced": self.route_role_balanced,
                        },
                    }
                return {
                    "pred": prediction,
                    "aux_losses": expert_aux_losses,
                    "forecast_loss_mode": "role_balanced_mix",
                    "forecast_role_balanced_beta": self.forecast_role_balanced_beta,
                }
            if self.fault_residual_head is not None:
                # One-hot, so the gate is exactly 0 on AAS/NAS/VS nodes and the
                # Unseen, Steady and Join paths stay bit-identical to fg1.
                fault_gate = last_availability[..., ROLE_FS].unsqueeze(1).unsqueeze(-1)
                correction = self.fault_residual_head(y_patch).reshape(
                    y_graph.shape[0], y_graph.shape[2], y_graph.shape[1], self.output_dim
                ).permute(0, 2, 1, 3)
                prediction = y_graph + (
                    self.fault_residual_scale * fault_gate * torch.tanh(correction)
                )
            return {
                "pred": prediction,
                "aux_losses": expert_aux_losses,
                "forecast_loss_mode": "role_balanced_mix",
                "forecast_role_balanced_beta": self.forecast_role_balanced_beta,
            }
        y_patch = self.temporal_refiner(y_patch, last_availability)
        state = state.unsqueeze(1).expand(-1, y_graph.size(1), -1, -1)
        gate_input = torch.cat(
            [state, y_graph, y_patch, torch.abs(y_graph - y_patch)],
            dim=-1,
        )
        logits = torch.cat([gate(gate_input) for gate in self.availability_gates], dim=-1)
        availability = last_availability.unsqueeze(1).expand_as(logits)
        patch_gate = torch.sigmoid(torch.sum(logits * availability, dim=-1, keepdim=True))
        budget = torch.sum(
            last_availability * self.patch_budget.to(last_availability).view(1, 1, -1),
            dim=-1,
            keepdim=True,
        ).unsqueeze(1)
        beta = patch_gate * budget
        patch_candidate = y_graph + budget * (y_patch - y_graph)
        prediction = y_graph + beta * (y_patch - y_graph)
        alpha = 1.0 - beta
        expert_meta = {
            "alpha": alpha,
            "y_graph": y_graph.detach(),
            "y_patch": patch_candidate.detach(),
        }
        return {
            "pred": prediction,
            "aux_losses": expert_aux_losses,
            "forecast_loss_mode": "role_balanced_mix",
            "forecast_role_balanced_beta": self.forecast_role_balanced_beta,
            "route_routing": {
                **expert_meta,
                "lambda_route": self.lambda_route,
                "warmup_epoch": self.route_warmup_epoch,
                "role_balanced": self.route_role_balanced,
            },
            "regret_routing": {
                **expert_meta,
                "lambda_regret": self.lambda_regret,
            },
        }
