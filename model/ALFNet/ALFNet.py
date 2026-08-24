import contextlib
import importlib
import math
import os

import torch
import torch.nn as nn

from args import load_args_from_config
from layers.protected_residual import (
    ProtectedTemporalResidual,
    companion_residual_loss,
    self_evidence,
)
from layers.source_grin import GRINInputImputer
from lib.lcbt_v4 import assert_v4_strategy
from lib.teacher_forcing import masked_teacher_forcing
from lib.v4_batch import read_common, resolve_adjacency
from utils.model_output import unpack_model_output


_RNG_OFFSET_PROTECTED_RESIDUAL = 30_013
# 40_009 was _RNG_OFFSET_TEACHER. The offsets are fixed per module rather than
# enumerated, so retiring one does not move any other module's initialisation;
# leaving the value unused keeps that true if something is ever added here.


@contextlib.contextmanager
def _isolated_rng(seed, offset):
    """Draw this module's initialisation from its own stream.

    Optional modules are built in the middle of __init__, so any random number
    they consume shifts every module constructed after them. That made a
    candidate's shared weights differ from its base under the same seed, which
    turns an ablation into two unrelated runs.
    """
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed) + int(offset))
        yield


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


def _build_expert(spec, device, expert_name, runtime_context=None,
                  run_seed=None, model_init_seed=None):
    config_path = spec["config_path"]
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"ALFNet {expert_name} config not found: {config_path}")
    expert_args = load_args_from_config(config_path, overrides={"device": device})
    expert_args.model = spec["model"]
    if run_seed is not None:
        expert_args.seed = int(run_seed)
    if model_init_seed is not None:
        expert_args.model_init_seed = int(model_init_seed)
    module = importlib.import_module(f"model.{_expert_module_name(spec['model'])}")
    if runtime_context is None:
        raise ValueError("ALFNet requires the active v4 RuntimeDataContext")
    entry = getattr(module.Model, "from_runtime_context", None)
    if entry is None:
        raise ValueError(
            f"ALFNet v4 requires {spec['model']}.Model.from_runtime_context; "
            "refusing a private asset-resolution fallback")
    return entry(expert_args, runtime_context).float(), expert_args


class Model(nn.Module):

    @classmethod
    def from_runtime_context(cls, configs, runtime_context):
        """The v4 entry point. One context, both experts, nothing from a path."""
        if runtime_context is None:
            raise ValueError(
                "ALFNet.from_runtime_context requires a RuntimeDataContext; "
                "v4 fails closed rather than falling back to data_path")
        from lib.context_binding import assert_config_agrees_with_context

        assert_config_agrees_with_context(configs, runtime_context, "ALFNet")
        return cls(configs, runtime_context=runtime_context)

    def __init__(self, configs, runtime_context=None):
        super().__init__()
        if not (
            configs.lifecycle_role_features
            and configs.lifecycle_history_age_feature
            and configs.lifecycle_observation_mask_feature
        ):
            raise ValueError("ALFNet requires availability, state-age, and observation-mask inputs")
        if runtime_context is None:
            raise ValueError("construct ALFNet through from_runtime_context")
        assert_v4_strategy(configs, "ALFNet")

        family = getattr(configs, "model_family_cfg", {}) or {}
        experts = family.get("experts", {})
        if set(experts) != {"graph", "patch"}:
            raise ValueError("ALFNet requires exactly graph and patch experts")
        self.graph_expert, self.graph_args = _build_expert(
            experts["graph"], configs.device, "graph", runtime_context,
            run_seed=configs.seed,
            model_init_seed=getattr(configs, "model_init_seed", configs.seed),
        )
        self.patch_expert, self.patch_args = _build_expert(
            experts["patch"], configs.device, "patch", runtime_context,
            run_seed=configs.seed,
            model_init_seed=getattr(configs, "model_init_seed", configs.seed),
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
        self.forecast_safe_relative_weight = float(
            recipe.get("forecast_safe_relative_weight", 0.0)
        )
        self.forecast_safe_relative_floor_policy = str(
            recipe.get(
                "forecast_safe_relative_floor_policy",
                "positive_target_q10_on_train_window_training_known_nodes",
            )
        )
        if (not math.isfinite(self.forecast_safe_relative_weight)
                or self.forecast_safe_relative_weight < 0):
            raise ValueError(
                "forecast_safe_relative_weight must be finite and >= 0")
        if (self.forecast_safe_relative_weight > 0
                and str(configs.target_features) != "q"):
            raise ValueError(
                "safe-relative loss is a flow-only candidate; set "
                "forecast_safe_relative_weight=0 for non-flow targets")
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
        # The residual is the only thing training in the frozen stage; it
        # needs a real step size, not the gate group's 5e-5.
        self.residual_lr = float(recipe.get("residual_lr", 5e-4))

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
        self.pred_len = int(configs.pred_len)
        self.input_len = int(configs.input_len)
        # R: protected, directly-supervised temporal residual over the graph
        # anchor. Off -> fg1 unchanged.
        self.temporal_residual_kind = str(recipe.get("temporal_residual", "none"))
        if self.temporal_residual_kind not in {"none", "tcn", "dlinear"}:
            raise ValueError(
                f"unsupported temporal_residual: {self.temporal_residual_kind}"
            )
        self.temporal_residual_enabled = self.temporal_residual_kind != "none"
        if self.temporal_residual_enabled and not self.film_fusion:
            raise ValueError("temporal_residual extends the FiLM-conditioned anchor")
        self.companion_loss_weight = float(recipe.get("companion_loss_weight", 1.0))
        self.freeze_anchor = bool(recipe.get("freeze_anchor", False))
        self.temporal_residual = None
        if self.temporal_residual_enabled:
            with _isolated_rng(
                    getattr(configs, "model_init_seed", configs.seed),
                    _RNG_OFFSET_PROTECTED_RESIDUAL):
                self.temporal_residual = ProtectedTemporalResidual(
                    int(configs.input_len),
                    int(configs.pred_len),
                    self.output_dim,
                    AVAILABILITY_COUNT,
                    graph_hidden_dim=int(self.graph_args.d_model),
                    hidden=int(model_cfg.get("residual_hidden_dim", 32)),
                    kind=self.temporal_residual_kind,
                    kernel_size=int(recipe.get("residual_kernel_size", 3)),
                    dilation=int(recipe.get("residual_dilation", 1)),
                    dropout=float(recipe.get("residual_dropout", 0.1)),
                    residual_scale=float(recipe.get("temporal_residual_scale", 1.0)),
                    self_evidence_floor=float(recipe.get("self_evidence_floor", 0.02)),
                )
        self.last_residual_diagnostics = None
        self._last_encoder_state = None
        self._last_router_diagnostics = None
        # Distillation (K) is gone. The deliverable is trained from random
        # initialisation in one stage, and a teacher makes that impossible: it
        # needs a second trained model on disk and a frozen schedule around it.
        # `kt` reached 35/36 that way and is kept only as a recorded upper
        # bound. Rejecting the keys rather than ignoring them means a stale YAML
        # cannot quietly reintroduce a two-stage path under a mainline name.
        retired_distillation = [
            key for key in ("lambda_kd", "teacher_checkpoint", "teacher_config_path")
            if recipe.get(key)
        ]
        if retired_distillation:
            raise ValueError(
                f"distillation was removed; recipe still sets {retired_distillation}. "
                "The deliverable is single-stage and end-to-end -- train the "
                "residual jointly instead."
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

        self.runtime_context = runtime_context
        self.static_asset_hash = runtime_context.static_asset_hash()
        self._v4_adjacency_cache = {}
        self.full_num_nodes = runtime_context.num_nodes
        self.known_nodes = [int(node) for node in runtime_context.known_nodes]
        for name, expert in (("graph", self.graph_expert),
                             ("patch", self.patch_expert)):
            bound = getattr(expert, "static_asset_hash", None)
            if bound != self.static_asset_hash:
                raise ValueError(
                    f"ALFNet {name} expert bound to static assets {bound!r}, "
                    f"outer has {self.static_asset_hash!r}; the two would be "
                    "reading different graphs")
            nodes = getattr(expert, "full_num_nodes", None) or getattr(
                expert, "context_full_num_nodes", None)
            if nodes is not None and int(nodes) != self.full_num_nodes:
                raise ValueError(
                    f"ALFNet {name} expert has {nodes} nodes, outer has "
                    f"{self.full_num_nodes}")

    def get_optimizer_param_groups(self, base_lr):
        graph_params = list(self.graph_expert.parameters()) + list(
            self.shared_input_imputer.parameters()
        )
        patch_params = list(self.patch_expert.parameters())
        residual_params = [
            parameter
            for module in (self.temporal_residual,)
            if module is not None
            for parameter in module.parameters()
        ]
        residual_ids = {id(parameter) for parameter in residual_params}
        expert_ids = {id(parameter) for parameter in graph_params + patch_params}
        fusion_params = [
            parameter
            for parameter in self.parameters()
            if id(parameter) not in expert_ids and id(parameter) not in residual_ids
        ]
        groups = [
            {"params": graph_params, "lr": self.graph_expert_lr},
            {"params": patch_params, "lr": self.patch_expert_lr},
            {"params": fusion_params, "lr": self.gate_lr or float(base_lr)},
        ]
        if residual_params:
            groups.append({"params": residual_params, "lr": self.residual_lr})
        return groups

    def _state(self, x_enc):
        availability = x_enc[..., self.role_start:self.role_end]
        last_availability = availability[:, -1]
        last_age = x_enc[:, -1, :, self.age_index:self.age_index + 1]
        visibility = x_enc[..., self.mask_index:self.mask_index + 1].mean(dim=1)
        state = self.state_encoder(
            torch.cat([last_availability, last_age, visibility], dim=-1)
        )
        return state, last_availability

    def _protected_residual(self, y_graph, x_enc, adjacency, last_availability):
        """Y_hat = Y_g + g(availability, age, visibility) * R.

        The gate is hard zero without own-history, so the zero-history and
        unseen paths return the anchor bit for bit.
        """
        observation_mask = x_enc[..., self.mask_index:self.mask_index + 1]
        recovered = x_enc[..., :self.output_dim]
        age = x_enc[:, -1, :, self.age_index:self.age_index + 1]
        visibility = observation_mask.mean(dim=1)
        e_self = self_evidence(observation_mask)
        graph_state = self._last_encoder_state
        if graph_state is None:
            graph_state = torch.zeros(
                y_graph.shape[0], y_graph.shape[2], int(self.graph_args.d_model),
                device=y_graph.device, dtype=y_graph.dtype,
            )
        residual, gate = self.temporal_residual(
            recovered, observation_mask, last_availability, age, visibility,
            e_self, graph_state.detach(),
        )
        if not self.training:
            self.last_residual_diagnostics = {
                "gate_mean": float(gate.mean()),
                "gate_max": float(gate.max()),
                "residual_abs_mean": float(residual.abs().mean()),
            }
        return y_graph + residual, {"residual": residual, "gate": gate}

    def freeze_anchor_parameters(self):
        """Stage one: train only the residual, leaving the anchor exactly as is.

        With the anchor frozen the protected groups are constant across
        candidates, so ranking by validation LifeAvg is ranking by
        Steady/Join/Fault -- the part of the validation signal whose ordering
        matches the test set (6/6 per group, against 1/6 for the 20-node
        validation-unseen group).
        """
        trainable = set()
        if self.temporal_residual is not None:
            trainable = {id(p) for p in self.temporal_residual.parameters()}
        frozen = 0
        for parameter in self.parameters():
            if id(parameter) not in trainable:
                parameter.requires_grad_(False)
                frozen += 1
        return frozen

    # --- the v4 entry point ---------------------------------------------------
    #
    # One adapter, and it is the only way a v4 batch reaches this model. The
    # validation lives in `lib.v4_batch` because MoGERNN now consumes the same
    # batch and the whole point is that both receive the identical common
    # tensor; two copies of these checks would be two places for it to drift.

    def _v4_adjacency(self, runtime_context, device, dtype):
        return resolve_adjacency(runtime_context, self._v4_adjacency_cache,
                                 device, dtype)

    def _v4_inputs(self, batch, runtime_context, teacher_forcing_mask):
        """Map one named v4 batch onto this model's tensors."""
        common = read_common(batch, runtime_context, self.static_asset_hash,
                             self.raw_enc_in, "ALFNet", self._v4_adjacency_cache,
                             extra_node_fields=("lifecycle_context",),
                             require_target=True)
        base = common["base_visible_input"]
        lifecycle = batch["lifecycle_context"]
        if lifecycle.shape[-1] != AVAILABILITY_COUNT + 2:
            raise ValueError(
                f"lifecycle_context has {lifecycle.shape[-1]} channels, expected "
                f"{AVAILABILITY_COUNT + 2} (roles, state age, observation mask)")

        # The common dynamic signal first, ALFNet's method features after it, in
        # the channel order the imputer and both experts already index. MoGERNN
        # receives the identical base_visible_input; only ALFNet appends this.
        x_enc = torch.cat([base, lifecycle.to(base.dtype)], dim=-1)
        if x_enc.shape[-1] != self.mask_index + 1:
            raise ValueError(
                f"assembled input has {x_enc.shape[-1]} channels, the lifecycle "
                f"layout ends at {self.mask_index}")

        teacher_input, teacher_mask = masked_teacher_forcing(
            batch["target"][..., :self.output_dim],
            batch["decoder_available_mask"], teacher_forcing_mask,
            self.training, "ALFNet")
        return (common["adjacency"], x_enc, common["encoder_time_marks"],
                common["decoder_time_marks"], teacher_input, teacher_mask)

    def forward_v4(self, batch, runtime_context, epoch=0, teacher_forcing_mask=None):
        """Run one v4 named batch. Full N everywhere, and no unmasked future.

        The target reaches the graph decoder only through
        `where(teacher_forcing_mask, target, 0)`, and only at positions the
        batch's own `decoder_available_mask` allows. With no mask -- the default,
        and the only thing eval accepts -- the tensor is all zeros and every
        decoder step feeds on its own output.
        """
        (adjacency, x_enc, marks, horizon, teacher_input,
         teacher_mask) = self._v4_inputs(batch, runtime_context, teacher_forcing_mask)
        return self(adjacency, x_enc, marks, None, teacher_input, horizon, epoch,
                    full_node_axis=True, teacher_mask=teacher_mask)

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
        full_node_axis=False,
        teacher_mask=None,
    ):
        # One imputer, one call, on the full node axis in v4. Both branches then
        # slice their declared channel prefix off this single tensor, so the
        # common raw/imputed channels they receive are the same numbers by
        # construction rather than by two paths agreeing.
        filled = self.shared_input_imputer(adjacency, x_enc, x_t_mark, pos_mark)
        if full_node_axis and filled.size(2) != self.full_num_nodes:
            raise ValueError(
                f"the shared imputer returned {filled.size(2)} nodes, the context "
                f"has {self.full_num_nodes}")
        # filled is [imputed_raw, availability(4), state_age, observation_mask];
        # each expert takes the prefix it declares.
        graph_input = filled[..., :self.graph_input_dim]
        patch_input = filled[..., :self.patch_input_dim]
        if hasattr(self.graph_expert, "set_current_node_ids"):
            self.graph_expert.set_current_node_ids(self._current_node_ids(filled.size(2)))
        decoder = None if x_dec is None else x_dec[..., :self.output_dim]
        patch_output = (
            self.patch_expert.forward_v4(patch_input, x_t_mark) if full_node_axis
            else self.patch_expert(
                adjacency, patch_input, x_t_mark, pos_mark, decoder, dec_t_mark, epoch)
        )
        if self.film_fusion:
            # One forecast, one decoder. The patch branch only conditions it.
            features = patch_output["features"]
            gamma, beta = self.film_head(features).chunk(2, dim=-1)
            graph_output = self.graph_expert(
                adjacency, graph_input, x_t_mark, pos_mark, decoder, dec_t_mark,
                epoch, film=(1.0 + gamma, beta), teacher_mask=teacher_mask,
            )
            y_graph, graph_meta = unpack_model_output(graph_output)
            self._last_encoder_state = graph_meta.get("encoder_state")
            self._last_router_diagnostics = graph_meta.get("router_diagnostics")
            return y_graph, features, graph_meta.get("aux_losses")
        graph_output = self.graph_expert(
            adjacency, graph_input, x_t_mark, pos_mark, decoder, dec_t_mark, epoch,
            teacher_mask=teacher_mask,
        )
        y_graph, graph_meta = unpack_model_output(graph_output)
        y_patch, _ = unpack_model_output(patch_output)
        return y_graph, y_patch, graph_meta.get("aux_losses")

    def _forecast_metadata(self, expert_aux_losses):
        """Training-objective declaration consumed by the shared v4 runner."""
        return {
            "aux_losses": expert_aux_losses,
            "forecast_loss_mode": "role_balanced_mix",
            "forecast_role_balanced_beta": self.forecast_role_balanced_beta,
            "forecast_safe_relative_weight": self.forecast_safe_relative_weight,
            "forecast_safe_relative_floor_policy": (
                self.forecast_safe_relative_floor_policy
            ),
        }

    def forward(
        self,
        adjacency,
        x_enc,
        x_t_mark=None,
        pos_mark=None,
        x_dec=None,
        dec_t_mark=None,
        epoch=0,
        full_node_axis=False,
        teacher_mask=None,
    ):
        y_graph, y_patch, expert_aux_losses = self._run_experts(
            adjacency,
            x_enc,
            x_t_mark,
            pos_mark,
            x_dec,
            dec_t_mark,
            epoch,
            full_node_axis,
            teacher_mask,
        )
        state, last_availability = self._state(x_enc)
        y_graph = self.graph_refiner(y_graph, adjacency, last_availability)
        if self.film_fusion:
            prediction = y_graph
            if self.temporal_residual is not None:
                prediction, extras = self._protected_residual(
                    y_graph, x_enc, adjacency, last_availability
                )
                return {
                    "pred": prediction,
                    **self._forecast_metadata(expert_aux_losses),
                    # The residual is supervised directly against the anchor's
                    # error rather than only through the fused loss.
                    "companion_residual": {
                        "residual": extras["residual"],
                        "anchor": y_graph,
                        "weight": self.companion_loss_weight,
                    },
                    "residual_gate": extras["gate"],
                    "router_diagnostics": self._last_router_diagnostics,
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
                **self._forecast_metadata(expert_aux_losses),
                # fg1 and the router candidates take this path.
                "router_diagnostics": self._last_router_diagnostics,
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
            **self._forecast_metadata(expert_aux_losses),
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
