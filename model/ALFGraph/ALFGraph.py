import torch
from torch import nn

from layers.DCRNNCell import DCRNNCell
from layers.gnn import calculate_random_walk_matrix
from layers.lifecycle_moge import (
    LIFECYCLE_STATE_CHANNELS,
    SUPPORT_NAMES,
    LifecycleRoutedGraphConv,
    build_lifecycle_supports,
)


class Encoder(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, k, conv_factory=None):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.cells = nn.ModuleList(
            [
                DCRNNCell(
                    input_size if layer == 0 else hidden_size,
                    hidden_size,
                    k,
                    conv_factory,
                )
                for layer in range(num_layers)
            ]
        )

    def forward(self, sequence, forward_support, backward_support, supports, state_sequence):
        batch, steps, nodes, _ = sequence.shape
        hidden = torch.zeros(
            self.num_layers,
            batch,
            nodes,
            self.hidden_size,
            device=sequence.device,
            dtype=sequence.dtype,
        )
        for step in range(steps):
            layer_input = sequence[:, step]
            # The lifecycle state is per step: a Join node is VS before its
            # cutoff and NAS after it, so routing follows the event as it lands.
            step_state = None if state_sequence is None else state_sequence[:, step]
            next_states = []
            for layer, cell in enumerate(self.cells):
                layer_input = cell(
                    layer_input,
                    hidden[layer],
                    forward_support,
                    backward_support,
                    supports=supports,
                    state=step_state,
                )
                next_states.append(layer_input)
            hidden = torch.stack(next_states, dim=0)
        return hidden


class ResidualDeltaDecoder(nn.Module):

    def __init__(self, output_size, hidden_size, num_layers, out_steps, k, scale, conv_factory=None):
        super().__init__()
        self.output_size = int(output_size)
        self.num_layers = int(num_layers)
        self.out_steps = int(out_steps)
        self.scale = float(scale)
        self.cells = nn.ModuleList(
            [
                DCRNNCell(
                    output_size if layer == 0 else hidden_size,
                    hidden_size,
                    k,
                    conv_factory,
                )
                for layer in range(num_layers)
            ]
        )
        self.projection = nn.Linear(hidden_size, output_size)

    def forward(
        self,
        initial_input,
        hidden,
        forward_support,
        backward_support,
        teacher_input,
        teacher_forcing_ratio,
        supports=None,
        state=None,
    ):
        current = initial_input
        outputs = []
        for step in range(self.out_steps):
            anchor = current
            next_states = []
            for layer, cell in enumerate(self.cells):
                current = cell(
                    current,
                    hidden[layer],
                    forward_support,
                    backward_support,
                    supports=supports,
                    state=state,
                )
                next_states.append(current)
            hidden = torch.stack(next_states, dim=0)
            output = anchor[..., :self.output_size] + self.scale * self.projection(current)
            outputs.append(output)
            # `teacher_input is not None` short-circuits before the draw, so a
            # call that supplies one consumes exactly the RNG it always did.
            # Without the guard a decoder-input-free training call indexes None:
            # v4 withholds the future entirely this stage, and a masked decoder
            # input is the next stage's job, not a crash here.
            use_teacher = (
                self.training
                and teacher_input is not None
                and torch.rand(1).item() < teacher_forcing_ratio
            )
            current = teacher_input[:, step] if use_teacher else output
        return torch.stack(outputs, dim=1)


class Model(nn.Module):

    @classmethod
    def from_runtime_context(cls, configs, runtime_context):
        """The v4 entry point.

        ALFGraph reads no graph from disk and must not start: its adjacency
        arrives per forward call. So this binds only what it can be held to --
        the static asset hash, the node count and the node order -- and verifies
        the config does not contradict them.
        """
        if runtime_context is None:
            raise ValueError(
                "ALFGraph.from_runtime_context requires a RuntimeDataContext; "
                "v4 fails closed rather than falling back to data_path")
        from lib.context_binding import assert_config_agrees_with_context

        assert_config_agrees_with_context(configs, runtime_context, "ALFGraph")
        model = cls(configs)
        identity = runtime_context.static_asset_identity()
        model.runtime_context = runtime_context
        model.static_asset_hash = runtime_context.static_asset_hash()
        model.context_full_num_nodes = identity["full_num_nodes"]
        model.context_node_ids_sha256 = identity["node_ids_sha256"]
        return model

    def __init__(self, configs):
        super().__init__()
        self.runtime_context = None
        self.static_asset_hash = None
        if not configs.lifecycle_observation_mask_feature:
            raise ValueError("ALFGraph requires the explicit observation-mask channel")
        recipe = (getattr(configs, "model_family_cfg", {}) or {}).get("recipe", {})
        if bool(recipe.get("use_grin_input_imputer", False)):
            raise ValueError("ALFNet uses one shared GRIN, not an ALFGraph-local GRIN")
        output_steps = configs.pred_len if configs.pred_len > 0 else configs.look_back
        # The observation mask stays in: after GRIN it is the only channel that
        # says whether a value was measured or imputed, which is exactly what the
        # router needs.
        backbone_input = int(configs.enc_in)
        self.raw_enc_in = int(configs.raw_enc_in)
        self.output_dim = int(configs.c_out)
        self.graph_conv = str(recipe.get("graph_conv", "lifecycle_moge"))
        if self.graph_conv not in {"lifecycle_moge", "diffusion"}:
            raise ValueError(f"unsupported ALFGraph graph_conv: {self.graph_conv}")
        self.routed = self.graph_conv == "lifecycle_moge"
        self.state_dim = LIFECYCLE_STATE_CHANNELS
        if self.raw_enc_in + self.state_dim != backbone_input:
            raise ValueError(
                f"ALFGraph expects raw({self.raw_enc_in}) + lifecycle({self.state_dim}) "
                f"channels, but enc_in is {backbone_input}"
            )

        conv_factory = None
        if self.routed:
            num_experts = int(recipe.get("moge_num_experts", len(SUPPORT_NAMES)))
            top_k = int(recipe.get("moge_top_k", 2))
            noisy_gating = bool(recipe.get("moge_noisy_gating", True))
            loss_coef = float(recipe.get("moge_load_balance_coef", 1e-2))
            # Default keeps the historical noisy top-k path so existing runs
            # reproduce; `soft` and `st_top1` make the router reachable from the
            # forecast loss, which at k=1 it provably is not.
            router_mode = str(recipe.get("moge_router_mode", "noisy_topk"))
            self.router_mode = router_mode

            def conv_factory(in_size, out_size):
                return LifecycleRoutedGraphConv(
                    in_size,
                    out_size,
                    num_experts=num_experts,
                    state_dim=self.state_dim,
                    k=top_k,
                    noisy_gating=noisy_gating,
                    loss_coef=loss_coef,
                    router_mode=router_mode,
                )

        self.encoder = Encoder(
            backbone_input,
            configs.d_model,
            configs.e_layers,
            configs.k_hop,
            conv_factory,
        )
        self.decoder = ResidualDeltaDecoder(
            configs.c_out,
            configs.d_model,
            configs.e_layers,
            output_steps,
            configs.k_hop,
            recipe.get("decoder_residual_scale", 0.3),
            conv_factory,
        )

    def set_current_node_ids(self, node_ids):
        return None

    def _routed_convs(self):
        return [
            module
            for module in self.modules()
            if isinstance(module, LifecycleRoutedGraphConv)
        ]

    def _named_routed_convs(self):
        """Named so usage and entropy can be read per layer, not just pooled.

        A router that collapses in the decoder while staying busy in the encoder
        looks healthy in an aggregate and is not.
        """
        return [
            (name, module)
            for name, module in self.named_modules()
            if isinstance(module, LifecycleRoutedGraphConv)
        ]

    def set_router_diagnostics(self, enabled):
        for _, conv in self._named_routed_convs():
            conv.set_collect_diagnostics(enabled)

    def pop_router_diagnostics(self):
        report = {}
        for name, conv in self._named_routed_convs():
            stats = conv.pop_router_diagnostics()
            if stats is not None:
                report[name] = stats
        return report

    def forward(
        self,
        adjacency,
        x_enc,
        x_t_mark=None,
        pos_mark=None,
        x_dec=None,
        dec_t_mark=None,
        epoch=0,
        film=None,
    ):
        forward_support = calculate_random_walk_matrix(adjacency)
        backward_support = calculate_random_walk_matrix(adjacency.T)
        supports, state_sequence = None, None
        if self.routed:
            supports = build_lifecycle_supports(
                adjacency,
                forward_support,
                backward_support,
            )
            state_sequence = x_enc[..., self.raw_enc_in:self.raw_enc_in + self.state_dim]
            for conv in self._routed_convs():
                conv.reset_router_stats()
            # Collect only at evaluation. The statistics are cheap next to the
            # einsums, but this runs 24 recurrent steps per forward across 12
            # routers and training does not read them.
            self.set_router_diagnostics(not self.training)

        hidden = self.encoder(
            x_enc,
            forward_support,
            backward_support,
            supports,
            state_sequence,
        )
        if film is not None:
            # Feature-wise linear modulation of the encoder state handed to the
            # decoder (Perez et al., AAAI 2018; GNN-FiLM, ICML 2020). Lifecycle
            # events change which sensors are observable, which is an encoding-
            # stage condition, so the modulation lands here rather than in the
            # decoder. gamma/beta are [batch, nodes, hidden] and broadcast over
            # the encoder layers.
            gamma, beta = film
            hidden = hidden * gamma.unsqueeze(0) + beta.unsqueeze(0)
        initial_input = x_enc[:, -1, :, :self.output_dim]
        teacher_forcing_ratio = 1.0 - float(epoch) / 40.0
        prediction = self.decoder(
            initial_input,
            hidden,
            forward_support,
            backward_support,
            x_dec,
            teacher_forcing_ratio,
            supports,
            # No lifecycle state exists beyond the input window, so the decoder
            # holds the last observed one rather than inventing a future.
            None if state_sequence is None else state_sequence[:, -1],
        )
        # The last encoder layer's state, detached: the residual branch reads
        # the anchor's summary without pushing gradients back into it.
        output = {"pred": prediction, "encoder_state": hidden[-1].detach()}
        if self.routed:
            terms = [
                loss
                for loss in (conv.pop_router_loss() for conv in self._routed_convs())
                if loss is not None
            ]
            if terms:
                output["aux_losses"] = {"graph_router": torch.stack(terms).mean()}
            diagnostics = self.pop_router_diagnostics()
            if diagnostics:
                output["router_diagnostics"] = diagnostics
        return output
