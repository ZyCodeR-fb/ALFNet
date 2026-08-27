import importlib
import os

import torch
import torch.nn as nn

from args import load_args_from_config
from layers.source_grin import GRINInputImputer
from lib.lcbt_v4 import assert_v4_strategy
from lib.teacher_forcing import masked_teacher_forcing
from lib.v4_batch import read_common, resolve_adjacency
from utils.model_output import unpack_model_output


AVAILABILITY_COUNT = 4


def _role_tensor(values, defaults):
    values = values or {}
    return torch.tensor(
        [float(values.get(name, defaults[name])) for name in ("aas", "nas", "vs", "fs")],
        dtype=torch.float32,
    )


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
    raise ValueError(f"the final ALFNet has no {model_name!r} expert")


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
        self.fusion_mode = str(recipe.get("fusion_mode", "film"))
        if self.fusion_mode != "film":
            raise ValueError("the final ALFNet supports FiLM fusion only")
        self.film_fusion = True
        if getattr(self.patch_expert, "emit", "prediction") != "features":
            raise ValueError(
                "fusion_mode=film requires the patch expert to run with "
                "family.model.emit=features; otherwise it still builds a "
                "regression head whose output nothing consumes"
            )
        retired = [key for key in (
            "lambda_route", "lambda_regret", "route_warmup_epoch",
            "route_role_balanced", "temporal_residual", "fault_residual_scale",
            "patch_residual_role_scale", "lambda_kd", "teacher_checkpoint",
            "teacher_config_path") if recipe.get(key)]
        if retired:
            raise ValueError(f"retired ALFNet recipe keys are not supported: {retired}")
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
        # The FiLM head is the only module this plan adds: two affine maps from
        # the patch branch's per-node condition vector to (gamma, beta) over the
        # DCRNN hidden state. gamma is centred at 1 and beta at 0 so an
        # untrained head is the identity and cannot destabilise the graph
        # expert at initialisation.
        hidden_size = int(self.graph_args.d_model)
        self.film_head = nn.Sequential(
            nn.Linear(int(self.patch_expert.feature_dim), gate_hidden_dim),
            nn.ReLU(),
            nn.Linear(gate_hidden_dim, 2 * hidden_size),
        )
        nn.init.zeros_(self.film_head[-1].weight)
        nn.init.zeros_(self.film_head[-1].bias)
        self._last_router_diagnostics = None
        self.graph_refiner = GraphOutputRefiner(
            self.output_dim,
            int(model_cfg.get("graph_refiner_hidden_dim", gate_hidden_dim)),
            int(recipe.get("graph_diffusion_steps", 1)),
            float(recipe.get("graph_diffusion_scale", 0.05)),
            recipe.get("graph_diffusion_role_scale", {}),
            bool(recipe.get("graph_diffusion_use_backward", True)),
            bool(recipe.get("graph_diffusion_add_self_loop", True)),
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
        expert_ids = {id(parameter) for parameter in graph_params + patch_params}
        fusion_params = [
            parameter
            for parameter in self.parameters()
            if id(parameter) not in expert_ids
        ]
        groups = [
            {"params": graph_params, "lr": self.graph_expert_lr},
            {"params": patch_params, "lr": self.patch_expert_lr},
            {"params": fusion_params, "lr": self.gate_lr or float(base_lr)},
        ]
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
        # One forecast, one decoder. The patch branch only conditions it.
        features = patch_output["features"]
        gamma, beta = self.film_head(features).chunk(2, dim=-1)
        graph_output = self.graph_expert(
            adjacency, graph_input, x_t_mark, pos_mark, decoder, dec_t_mark,
            epoch, film=(1.0 + gamma, beta), teacher_mask=teacher_mask,
        )
        y_graph, graph_meta = unpack_model_output(graph_output)
        self._last_router_diagnostics = graph_meta.get("router_diagnostics")
        return y_graph, features, graph_meta.get("aux_losses")

    def _forecast_metadata(self, expert_aux_losses):
        """Training-objective declaration consumed by the shared v4 runner."""
        return {
            "aux_losses": expert_aux_losses,
            "forecast_loss_mode": "role_balanced_mix",
            "forecast_role_balanced_beta": self.forecast_role_balanced_beta,
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
        _, last_availability = self._state(x_enc)
        y_graph = self.graph_refiner(y_graph, adjacency, last_availability)
        return {
            "pred": y_graph,
            **self._forecast_metadata(expert_aux_losses),
            "router_diagnostics": self._last_router_diagnostics,
        }
