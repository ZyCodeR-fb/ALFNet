import torch
import torch.nn as nn
import torch.nn.functional as F


ROLE_AAS = 0
ROLE_NAS = 1
ROLE_VS = 2
ROLE_FS = 3
ROLE_COUNT = 4
EPSILON = 1e-8


def reverse_tensor(tensor=None, axis=-1):
    # Copied from references/grin/lib/nn/utils/ops.py::reverse_tensor.
    if tensor is None:
        return None
    if tensor.dim() <= 1:
        return tensor
    indices = range(tensor.size()[axis])[::-1]
    indices = torch.LongTensor(indices).to(tensor.device)
    return tensor.index_select(axis, indices)


class SpatialConvOrderK(nn.Module):
    # Copied from references/grin/lib/nn/layers/spatial_conv.py with epsilon local
    # and no package-level imports.
    def __init__(self, c_in, c_out, support_len=3, order=2, include_self=True):
        super(SpatialConvOrderK, self).__init__()
        self.include_self = include_self
        c_in = (order * support_len + (1 if include_self else 0)) * c_in
        self.mlp = nn.Conv2d(c_in, c_out, kernel_size=1)
        self.order = order

    @staticmethod
    def compute_support(adj, device=None):
        if device is not None:
            adj = adj.to(device)
        adj_bwd = adj.T
        adj_fwd = adj / (adj.sum(1, keepdims=True) + EPSILON)
        adj_bwd = adj_bwd / (adj_bwd.sum(1, keepdims=True) + EPSILON)
        support = [adj_fwd, adj_bwd]
        return support

    @staticmethod
    def compute_support_orderK(adj, k, include_self=False, device=None):
        if isinstance(adj, (list, tuple)):
            support = adj
        else:
            support = SpatialConvOrderK.compute_support(adj, device)
        supp_k = []
        for a in support:
            ak = a
            for _ in range(k - 1):
                ak = torch.matmul(ak, a.T)
                if not include_self:
                    ak.fill_diagonal_(0.0)
                supp_k.append(ak)
        return support + supp_k

    def forward(self, x, support):
        # [batch, features, nodes, steps]
        if x.dim() < 4:
            squeeze = True
            x = torch.unsqueeze(x, -1)
        else:
            squeeze = False
        out = [x] if self.include_self else []
        if type(support) is not list:
            support = [support]
        for a in support:
            x1 = torch.einsum("ncvl,wv->ncwl", (x, a)).contiguous()
            out.append(x1)
            for _ in range(2, self.order + 1):
                x2 = torch.einsum("ncvl,wv->ncwl", (x1, a)).contiguous()
                out.append(x2)
                x1 = x2

        out = torch.cat(out, dim=1)
        out = self.mlp(out)
        if squeeze:
            out = out.squeeze(-1)
        return out


class GCGRUCell(nn.Module):
    # Copied from references/grin/lib/nn/layers/gcrnn.py.
    def __init__(self, d_in, num_units, support_len, order, activation="tanh"):
        super(GCGRUCell, self).__init__()
        self.activation_fn = getattr(torch, activation)
        self.forget_gate = SpatialConvOrderK(c_in=d_in + num_units, c_out=num_units, support_len=support_len, order=order)
        self.update_gate = SpatialConvOrderK(c_in=d_in + num_units, c_out=num_units, support_len=support_len, order=order)
        self.c_gate = SpatialConvOrderK(c_in=d_in + num_units, c_out=num_units, support_len=support_len, order=order)

    def forward(self, x, h, adj):
        x_gates = torch.cat([x, h], dim=1)
        r = torch.sigmoid(self.forget_gate(x_gates, adj))
        u = torch.sigmoid(self.update_gate(x_gates, adj))
        x_c = torch.cat([x, r * h], dim=1)
        c = self.c_gate(x_c, adj)
        c = self.activation_fn(c)
        return u * h + (1.0 - u) * c


class SpatialDecoder(nn.Module):
    # Copied from references/grin/lib/nn/layers/gril.py, with global attention
    # branch intentionally omitted because it is disabled in the original LA config.
    def __init__(self, d_in, d_model, d_out, support_len, order=1):
        super(SpatialDecoder, self).__init__()
        self.order = order
        self.lin_in = nn.Conv1d(d_in, d_model, kernel_size=1)
        self.graph_conv = SpatialConvOrderK(c_in=d_model, c_out=d_model, support_len=support_len * order, order=1, include_self=False)
        self.lin_out = nn.Conv1d(2 * d_model, d_model, kernel_size=1)
        self.read_out = nn.Conv1d(2 * d_model, d_out, kernel_size=1)
        self.activation = nn.PReLU()
        self.adj = None

    def forward(self, x, m, h, u, adj, cached_support=False):
        x_in = [x, m, h] if u is None else [x, m, u, h]
        x_in = torch.cat(x_in, 1)
        if self.order > 1:
            if cached_support and (self.adj is not None):
                adj = self.adj
            else:
                adj = SpatialConvOrderK.compute_support_orderK(adj, self.order, include_self=False, device=x_in.device)
                self.adj = adj if cached_support else None

        x_in = self.lin_in(x_in)
        out = self.graph_conv(x_in, adj)
        out = torch.cat([out, h], 1)
        out = self.activation(self.lin_out(out))
        out = torch.cat([out, h], 1)
        return self.read_out(out), out


class GRIL(nn.Module):
    # Copied from references/grin/lib/nn/layers/gril.py with current_node_ids
    # support so one module works with train subgraphs and full test graphs.
    def __init__(
        self,
        input_size,
        hidden_size,
        u_size=None,
        n_layers=1,
        dropout=0.0,
        kernel_size=2,
        decoder_order=1,
        support_len=2,
        n_nodes=None,
        layer_norm=False,
    ):
        super(GRIL, self).__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.u_size = int(u_size) if u_size is not None else 0
        self.n_layers = int(n_layers)
        rnn_input_size = 2 * self.input_size + self.u_size

        self.cells = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(self.n_layers):
            self.cells.append(
                GCGRUCell(
                    d_in=rnn_input_size if i == 0 else self.hidden_size,
                    num_units=self.hidden_size,
                    support_len=support_len,
                    order=kernel_size,
                )
            )
            self.norms.append(nn.GroupNorm(num_groups=1, num_channels=self.hidden_size) if layer_norm else nn.Identity())
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None
        self.first_stage = nn.Conv1d(in_channels=self.hidden_size, out_channels=self.input_size, kernel_size=1)
        self.spatial_decoder = SpatialDecoder(
            d_in=rnn_input_size + self.hidden_size,
            d_model=self.hidden_size,
            d_out=self.input_size,
            support_len=2,
            order=decoder_order,
        )
        if n_nodes is not None:
            self.h0 = self.init_hidden_states(n_nodes)
        else:
            self.register_parameter("h0", None)

    def init_hidden_states(self, n_nodes):
        h0 = []
        for _ in range(self.n_layers):
            std = 1.0 / torch.sqrt(torch.tensor(self.hidden_size, dtype=torch.float))
            vals = torch.distributions.Normal(0, std).sample((self.hidden_size, n_nodes))
            h0.append(nn.Parameter(vals))
        return nn.ParameterList(h0)

    def get_h0(self, x, current_node_ids=None):
        if self.h0 is not None:
            states = []
            for h in self.h0:
                h_s = h
                if current_node_ids is not None:
                    h_s = h_s[:, current_node_ids]
                states.append(h_s.expand(x.shape[0], -1, -1))
            return states
        return [torch.zeros(size=(x.shape[0], self.hidden_size, x.shape[2])).to(x.device)] * self.n_layers

    def update_state(self, x, h, adj):
        rnn_in = x
        for layer, (cell, norm) in enumerate(zip(self.cells, self.norms)):
            rnn_in = h[layer] = norm(cell(rnn_in, h[layer], adj))
            if self.dropout is not None and layer < (self.n_layers - 1):
                rnn_in = self.dropout(rnn_in)
        return h

    def forward(self, x, adj, mask=None, u=None, h=None, cached_support=False, current_node_ids=None):
        *_, steps = x.size()
        if mask is None:
            mask = torch.ones_like(x, dtype=torch.bool)
        if h is None:
            h = self.get_h0(x, current_node_ids=current_node_ids)
        elif not isinstance(h, list):
            h = [*h]

        predictions, imputations, states = [], [], []
        representations = []
        for step in range(steps):
            x_s = x[..., step]
            m_s = mask[..., step].to(dtype=x.dtype)
            h_s = h[-1]
            u_s = u[..., step] if u is not None else None
            xs_hat_1 = self.first_stage(h_s)
            x_s = torch.where(m_s.bool(), x_s, xs_hat_1)
            xs_hat_2, repr_s = self.spatial_decoder(x=x_s, m=m_s, h=h_s, u=u_s, adj=adj, cached_support=cached_support)
            x_s = torch.where(m_s.bool(), x_s, xs_hat_2)
            inputs = [x_s, m_s]
            if u_s is not None:
                inputs.append(u_s)
            inputs = torch.cat(inputs, dim=1)
            h = self.update_state(inputs, h, adj)
            imputations.append(xs_hat_2)
            predictions.append(xs_hat_1)
            states.append(torch.stack(h, dim=0))
            representations.append(repr_s)

        imputations = torch.stack(imputations, dim=-1)
        predictions = torch.stack(predictions, dim=-1)
        states = torch.stack(states, dim=-1)
        representations = torch.stack(representations, dim=-1)
        return imputations, predictions, representations, states


class BiGRIL(nn.Module):
    # Copied from references/grin/lib/nn/layers/gril.py with current_node_ids
    # support for node embeddings.
    def __init__(
        self,
        input_size,
        hidden_size,
        ff_size,
        ff_dropout,
        n_layers=1,
        dropout=0.0,
        n_nodes=None,
        support_len=2,
        kernel_size=2,
        decoder_order=1,
        u_size=0,
        embedding_size=0,
        layer_norm=False,
        merge="mlp",
    ):
        super(BiGRIL, self).__init__()
        self.fwd_rnn = GRIL(
            input_size=input_size,
            hidden_size=hidden_size,
            n_layers=n_layers,
            dropout=dropout,
            n_nodes=n_nodes,
            support_len=support_len,
            kernel_size=kernel_size,
            decoder_order=decoder_order,
            u_size=u_size,
            layer_norm=layer_norm,
        )
        self.bwd_rnn = GRIL(
            input_size=input_size,
            hidden_size=hidden_size,
            n_layers=n_layers,
            dropout=dropout,
            n_nodes=n_nodes,
            support_len=support_len,
            kernel_size=kernel_size,
            decoder_order=decoder_order,
            u_size=u_size,
            layer_norm=layer_norm,
        )

        if n_nodes is None:
            embedding_size = 0
        if embedding_size > 0:
            self.emb = nn.Parameter(torch.empty(embedding_size, n_nodes))
            nn.init.kaiming_normal_(self.emb, nonlinearity="relu")
        else:
            self.register_parameter("emb", None)

        if merge == "mlp":
            self._impute_from_states = True
            self.out = nn.Sequential(
                nn.Conv2d(in_channels=4 * hidden_size + input_size + embedding_size, out_channels=ff_size, kernel_size=1),
                nn.ReLU(),
                nn.Dropout(ff_dropout),
                nn.Conv2d(in_channels=ff_size, out_channels=input_size, kernel_size=1),
            )
        elif merge in ["mean", "sum", "min", "max"]:
            self._impute_from_states = False
            self.out = getattr(torch, merge)
        else:
            raise ValueError("Merge option %s not allowed." % merge)
        self.supp = None

    def forward(self, x, adj, mask=None, u=None, cached_support=False, current_node_ids=None):
        if cached_support and (self.supp is not None):
            supp = self.supp
        else:
            supp = SpatialConvOrderK.compute_support(adj, x.device)
            self.supp = supp if cached_support else None
        fwd_out, fwd_pred, fwd_repr, _ = self.fwd_rnn(
            x, supp, mask=mask, u=u, cached_support=cached_support, current_node_ids=current_node_ids
        )
        rev_x, rev_mask, rev_u = [reverse_tensor(tens) for tens in (x, mask, u)]
        *bwd_res, _ = self.bwd_rnn(
            rev_x, supp, mask=rev_mask, u=rev_u, cached_support=cached_support, current_node_ids=current_node_ids
        )
        bwd_out, bwd_pred, bwd_repr = [reverse_tensor(res) for res in bwd_res]

        if self._impute_from_states:
            inputs = [fwd_repr, bwd_repr, mask.to(dtype=x.dtype)]
            if self.emb is not None:
                b, *_, s = fwd_repr.shape
                emb = self.emb
                if current_node_ids is not None:
                    emb = emb[:, current_node_ids]
                inputs += [emb.view(1, *emb.shape, 1).expand(b, -1, -1, s)]
            imputation = torch.cat(inputs, dim=1)
            imputation = self.out(imputation)
        else:
            imputation = torch.stack([fwd_out, bwd_out], dim=1)
            imputation = self.out(imputation, dim=1)

        predictions = torch.stack([fwd_out, bwd_out, fwd_pred, bwd_pred], dim=0)
        return imputation, predictions


class GRINet(nn.Module):
    # Copied from references/grin/lib/nn/models/grin.py with current_node_ids
    # propagation and no einops dependency.
    def __init__(
        self,
        adj,
        d_in,
        d_hidden,
        d_ff,
        ff_dropout,
        n_layers=1,
        kernel_size=2,
        decoder_order=1,
        d_u=0,
        d_emb=0,
        layer_norm=False,
        merge="mlp",
        impute_only_holes=True,
    ):
        super(GRINet, self).__init__()
        self.d_in = d_in
        self.d_hidden = d_hidden
        self.d_u = int(d_u) if d_u is not None else 0
        self.d_emb = int(d_emb) if d_emb is not None else 0
        self.register_buffer("adj", torch.as_tensor(adj).float())
        self.impute_only_holes = impute_only_holes
        self.bigrill = BiGRIL(
            input_size=self.d_in,
            ff_size=d_ff,
            ff_dropout=ff_dropout,
            hidden_size=self.d_hidden,
            embedding_size=self.d_emb,
            n_nodes=self.adj.shape[0],
            n_layers=n_layers,
            kernel_size=kernel_size,
            decoder_order=decoder_order,
            u_size=self.d_u,
            layer_norm=layer_norm,
            merge=merge,
        )

    def forward(self, x, mask=None, u=None, adj=None, current_node_ids=None):
        x = x.permute(0, 3, 2, 1).contiguous()
        if mask is not None:
            mask = mask.permute(0, 3, 2, 1).contiguous().bool()
        if u is not None:
            u = u.permute(0, 3, 2, 1).contiguous()
        if adj is None:
            adj = self.adj
        imputation, prediction = self.bigrill(x, adj, mask=mask, u=u, cached_support=self.training, current_node_ids=current_node_ids)
        if self.impute_only_holes and not self.training:
            imputation = torch.where(mask, x, imputation)
        imputation = imputation.permute(0, 3, 2, 1).contiguous()
        prediction = prediction.permute(0, 1, 4, 3, 2).contiguous()
        if self.training:
            return imputation, prediction
        return imputation


class DenseSPINLayer(nn.Module):
    # Adapted from references/SPIN/spin/layers/temporal_graph_additive_attention.py:
    # same temporal + spatial message-passing idea, implemented with dense
    # adjacency to avoid torch_geometric/tsl runtime dependencies.
    def __init__(self, hidden_size, temporal_self_attention=True, dropout=0.0):
        super().__init__()
        self.temporal_self_attention = temporal_self_attention
        self.spatial_src = nn.Linear(hidden_size, hidden_size)
        self.spatial_tgt = nn.Linear(hidden_size, hidden_size, bias=False)
        self.msg_nn = nn.Sequential(nn.PReLU(init=0.2), nn.Linear(hidden_size, hidden_size), nn.PReLU(init=0.2), nn.Linear(hidden_size, hidden_size))
        self.msg_gate = nn.Linear(hidden_size, 1, bias=False)
        self.lin_skip = nn.Linear(hidden_size, hidden_size, bias=False)
        if temporal_self_attention:
            self.temporal_attn = nn.MultiheadAttention(hidden_size, num_heads=2, dropout=dropout, batch_first=True)
        else:
            self.temporal_attn = None
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = float(dropout)

    def _spatial_attention(self, h, adj, mask):
        b, t, n, c = h.shape
        adj = adj.to(device=h.device, dtype=h.dtype)
        adj = adj / adj.sum(1, keepdim=True).clamp_min(EPSILON)
        msg_src = self.spatial_src(h)
        msg_tgt = self.spatial_tgt(h)
        msg = msg_src.unsqueeze(2) + msg_tgt.unsqueeze(3)  # [B,T,N_target,N_source,C]
        msg = self.msg_nn(msg)
        gate = self.msg_gate(msg).squeeze(-1)
        spatial_mask = adj.unsqueeze(0).unsqueeze(0) > 0
        if mask is not None:
            spatial_mask = spatial_mask & mask.squeeze(-1).unsqueeze(2).bool()
        gate = gate.masked_fill(~spatial_mask, -1e9)
        alpha = torch.softmax(gate, dim=-1)
        alpha = torch.where(spatial_mask.any(dim=-1, keepdim=True), alpha, torch.zeros_like(alpha))
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return torch.einsum("btnm,btnmc->btnc", alpha, msg)

    def forward(self, h, adj, mask=None):
        out = self._spatial_attention(h, adj, mask)
        if self.temporal_attn is not None:
            b, t, n, c = h.shape
            temp_in = h.permute(0, 2, 1, 3).reshape(b * n, t, c)
            temp = self.temporal_attn(temp_in, temp_in, temp_in, need_weights=False)[0]
            temp = temp.reshape(b, n, t, c).permute(0, 2, 1, 3).contiguous()
            out = out + temp
        out = out + self.lin_skip(h)
        return self.norm(out)


class SPINRecoveryNet(nn.Module):
    # Adapted from references/SPIN/spin/models/spin.py::SPINModel.
    def __init__(
        self,
        input_size,
        hidden_size,
        n_nodes,
        output_size=None,
        temporal_self_attention=True,
        n_layers=4,
        eta=3,
        dropout=0.0,
    ):
        super().__init__()
        output_size = output_size or input_size
        self.n_nodes = int(n_nodes)
        self.n_layers = int(n_layers)
        self.eta = int(eta)
        self.node_emb = nn.Parameter(torch.empty(self.n_nodes, hidden_size))
        nn.init.kaiming_normal_(self.node_emb, nonlinearity="relu")
        self.u_enc = nn.Sequential(nn.Linear(1, hidden_size), nn.LeakyReLU(), nn.Linear(hidden_size, hidden_size), nn.ReLU())
        self.h_enc = nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, hidden_size))
        self.h_norm = nn.LayerNorm(hidden_size)
        self.valid_emb = nn.Parameter(torch.empty(self.n_nodes, hidden_size))
        self.mask_emb = nn.Parameter(torch.empty(self.n_nodes, hidden_size))
        nn.init.kaiming_normal_(self.valid_emb, nonlinearity="relu")
        nn.init.kaiming_normal_(self.mask_emb, nonlinearity="relu")
        self.x_skip = nn.ModuleList()
        self.encoder = nn.ModuleList()
        self.readout = nn.ModuleList()
        for _ in range(n_layers):
            self.x_skip.append(nn.Linear(input_size, hidden_size))
            self.encoder.append(DenseSPINLayer(hidden_size, temporal_self_attention=temporal_self_attention, dropout=dropout))
            self.readout.append(nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, output_size)))

    def _select_node_param(self, param, current_node_ids, n):
        if current_node_ids is None:
            return param[:n]
        return param[current_node_ids]

    def forward(self, x, mask, adj, current_node_ids=None):
        b, t, n, _ = x.shape
        x = x * mask.to(dtype=x.dtype)
        time_pos = torch.linspace(0.0, 1.0, t, device=x.device, dtype=x.dtype).view(1, t, 1)
        node_emb = self._select_node_param(self.node_emb, current_node_ids, n).view(1, 1, n, -1)
        q = self.u_enc(time_pos).unsqueeze(2) + node_emb
        h = self.h_enc(x) + q
        h = torch.where(mask.bool(), h, q)
        h = self.h_norm(h)
        imputations = []
        valid_emb = self._select_node_param(self.valid_emb, current_node_ids, n).view(1, 1, n, -1)
        mask_emb = self._select_node_param(self.mask_emb, current_node_ids, n).view(1, 1, n, -1)
        for layer_idx in range(self.n_layers):
            if layer_idx == self.eta:
                h = torch.where(mask.bool(), h + valid_emb, h + mask_emb)
            h = h + self.x_skip[layer_idx](x) * mask.to(dtype=x.dtype)
            h = self.encoder[layer_idx](h, adj, mask=mask)
            imputations.append(self.readout[layer_idx](h))
        return imputations[-1]


class GraphInputRecoveryAdapter(nn.Module):
    def __init__(self, recipe_cfg, full_adj, input_dim, full_num_nodes):
        super().__init__()
        self.recovery_name = str(recipe_cfg.get("graph_recovery_type", "grin")).lower()
        self.lambda_observed = float(recipe_cfg.get("lambda_graph_recovery_observed", 0.0))
        self.replace_observed = bool(recipe_cfg.get("graph_recovery_replace_observed", False))
        self.gate_floor = float(recipe_cfg.get("graph_recovery_gate_floor", 0.0))
        role_scale = recipe_cfg.get("graph_recovery_role_scale", {}) or {}
        self.register_buffer(
            "role_recovery_scale",
            torch.tensor(
                [
                    float(role_scale.get("aas", 0.0)),
                    float(role_scale.get("nas", 1.0)),
                    float(role_scale.get("vs", 1.0)),
                    float(role_scale.get("fs", 1.0)),
                ],
                dtype=torch.float32,
            ),
            persistent=False,
        )
        self.state_gate = None
        if bool(recipe_cfg.get("graph_recovery_state_gate", False)):
            gate_hidden = int(recipe_cfg.get("graph_recovery_gate_hidden_dim", 16))
            self.state_gate = nn.Sequential(
                nn.Linear(ROLE_COUNT + 2, gate_hidden),
                nn.ReLU(),
                nn.Linear(gate_hidden, 1),
                nn.Sigmoid(),
            )
        if self.recovery_name == "grin":
            self.recovery = GRINet(
                adj=full_adj,
                d_in=input_dim,
                d_hidden=int(recipe_cfg.get("grin_hidden_dim", 32)),
                d_ff=int(recipe_cfg.get("grin_ff_dim", 32)),
                ff_dropout=float(recipe_cfg.get("grin_ff_dropout", 0.0)),
                n_layers=int(recipe_cfg.get("grin_layers", 1)),
                kernel_size=int(recipe_cfg.get("grin_kernel_size", 2)),
                decoder_order=int(recipe_cfg.get("grin_decoder_order", 1)),
                d_emb=int(recipe_cfg.get("grin_embedding_dim", 8)),
                layer_norm=bool(recipe_cfg.get("grin_layer_norm", False)),
                merge=str(recipe_cfg.get("grin_merge", "mlp")),
                impute_only_holes=False,
            )
        elif self.recovery_name == "spin":
            self.recovery = SPINRecoveryNet(
                input_size=input_dim,
                hidden_size=int(recipe_cfg.get("spin_hidden_dim", 32)),
                n_nodes=full_num_nodes,
                output_size=input_dim,
                temporal_self_attention=bool(recipe_cfg.get("spin_temporal_self_attention", True)),
                n_layers=int(recipe_cfg.get("spin_layers", 4)),
                eta=int(recipe_cfg.get("spin_eta", 3)),
                dropout=float(recipe_cfg.get("spin_dropout", 0.0)),
            )
        else:
            raise ValueError(f"Unsupported graph_recovery_type: {self.recovery_name}")

    def _visible_mask(self, role_seq):
        return ((role_seq[..., ROLE_AAS] + role_seq[..., ROLE_NAS]) > 0.5).unsqueeze(-1)

    def forward(self, raw_x_enc, role_seq, last_role, adj, current_node_ids=None, state_features=None):
        observed_mask = self._visible_mask(role_seq)
        if self.recovery_name == "grin":
            recovery_output = self.recovery(raw_x_enc, mask=observed_mask, adj=adj, current_node_ids=current_node_ids)
            imputed = recovery_output[0] if isinstance(recovery_output, tuple) else recovery_output
        else:
            imputed = self.recovery(raw_x_enc, observed_mask, adj, current_node_ids=current_node_ids)

        role_factor = torch.sum(last_role * self.role_recovery_scale.view(1, 1, ROLE_COUNT).to(last_role.device), dim=-1, keepdim=True)
        if self.state_gate is not None:
            if state_features is None:
                state_features = torch.cat(
                    [
                        last_role,
                        torch.zeros_like(last_role[..., :1]),
                        torch.zeros_like(last_role[..., :1]),
                    ],
                    dim=-1,
                )
            gate = self.state_gate(state_features.to(dtype=raw_x_enc.dtype))
            gate = self.gate_floor + (1.0 - self.gate_floor) * gate
            role_factor = role_factor * gate
        role_factor = role_factor.unsqueeze(1).to(dtype=raw_x_enc.dtype)
        missing_mask = (~observed_mask).to(dtype=raw_x_enc.dtype)
        if self.replace_observed:
            graph_x = raw_x_enc + role_factor * (imputed - raw_x_enc)
        else:
            graph_x = raw_x_enc + missing_mask * role_factor * (imputed - raw_x_enc)

        aux_losses = {}
        if self.lambda_observed > 0:
            observed_float = observed_mask.to(dtype=raw_x_enc.dtype)
            denom = observed_float.sum().clamp_min(1.0)
            aux_losses["rolefusion_graph_recovery_observed"] = self.lambda_observed * ((imputed - raw_x_enc).pow(2) * observed_float).sum() / denom
        return graph_x, imputed, aux_losses
