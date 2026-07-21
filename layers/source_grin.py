import torch
from torch import nn


epsilon = 1e-8


def reverse_tensor(x):
    if x is None:
        return None
    return torch.flip(x, dims=(-1,))


class SpatialConvOrderK(nn.Module):
    """Copied from references/grin/lib/nn/layers/spatial_conv.py."""

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
        adj_fwd = adj / (adj.sum(1, keepdims=True) + epsilon)
        adj_bwd = adj_bwd / (adj_bwd.sum(1, keepdims=True) + epsilon)
        return [adj_fwd, adj_bwd]

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
        if x.dim() < 4:
            squeeze = True
            x = torch.unsqueeze(x, -1)
        else:
            squeeze = False
        out = [x] if self.include_self else []
        if not isinstance(support, list):
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
    """Copied from references/grin/lib/nn/layers/gcrnn.py."""

    def __init__(self, d_in, num_units, support_len, order, activation="tanh"):
        super(GCGRUCell, self).__init__()
        self.activation_fn = getattr(torch, activation)
        self.forget_gate = SpatialConvOrderK(d_in + num_units, num_units, support_len=support_len, order=order)
        self.update_gate = SpatialConvOrderK(d_in + num_units, num_units, support_len=support_len, order=order)
        self.c_gate = SpatialConvOrderK(d_in + num_units, num_units, support_len=support_len, order=order)

    def forward(self, x, h, adj):
        x_gates = torch.cat([x, h], dim=1)
        r = torch.sigmoid(self.forget_gate(x_gates, adj))
        u = torch.sigmoid(self.update_gate(x_gates, adj))
        x_c = torch.cat([x, r * h], dim=1)
        c = self.activation_fn(self.c_gate(x_c, adj))
        return u * h + (1.0 - u) * c


class SpatialDecoder(nn.Module):
    """Copied from references/grin/lib/nn/layers/gril.py without the optional attention block."""

    def __init__(self, d_in, d_model, d_out, support_len, order=1):
        super(SpatialDecoder, self).__init__()
        self.order = order
        self.lin_in = nn.Conv1d(d_in, d_model, kernel_size=1)
        self.graph_conv = SpatialConvOrderK(
            c_in=d_model,
            c_out=d_model,
            support_len=support_len * order,
            order=1,
            include_self=False,
        )
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
    """Copied from references/grin/lib/nn/layers/gril.py."""

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

    def get_h0(self, x):
        if self.h0 is not None:
            return [h.expand(x.shape[0], -1, -1) for h in self.h0]
        return [torch.zeros(size=(x.shape[0], self.hidden_size, x.shape[2]), device=x.device, dtype=x.dtype) for _ in range(self.n_layers)]

    def update_state(self, x, h, adj):
        rnn_in = x
        for layer, (cell, norm) in enumerate(zip(self.cells, self.norms)):
            rnn_in = h[layer] = norm(cell(rnn_in, h[layer], adj))
            if self.dropout is not None and layer < (self.n_layers - 1):
                rnn_in = self.dropout(rnn_in)
        return h

    def forward(self, x, adj, mask=None, u=None, h=None, cached_support=False):
        *_, steps = x.size()
        if mask is None:
            mask = torch.ones_like(x, dtype=torch.bool)
        if h is None:
            h = self.get_h0(x)
        elif not isinstance(h, list):
            h = [*h]

        predictions, imputations, states = [], [], []
        representations = []
        for step in range(steps):
            x_s = x[..., step]
            m_s = mask[..., step]
            h_s = h[-1]
            u_s = u[..., step] if u is not None else None
            xs_hat_1 = self.first_stage(h_s)
            x_s = torch.where(m_s, x_s, xs_hat_1)
            xs_hat_2, repr_s = self.spatial_decoder(
                x=x_s,
                m=m_s.to(dtype=x_s.dtype),
                h=h_s,
                u=u_s,
                adj=adj,
                cached_support=cached_support,
            )
            x_s = torch.where(m_s, x_s, xs_hat_2)
            inputs = [x_s, m_s.to(dtype=x_s.dtype)]
            if u_s is not None:
                inputs.append(u_s)
            h = self.update_state(torch.cat(inputs, dim=1), h, adj)
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
    """Copied from references/grin/lib/nn/layers/gril.py."""

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
                nn.Conv2d(4 * hidden_size + input_size + embedding_size, ff_size, kernel_size=1),
                nn.ReLU(),
                nn.Dropout(ff_dropout),
                nn.Conv2d(ff_size, input_size, kernel_size=1),
            )
        elif merge in ["mean", "sum", "min", "max"]:
            self._impute_from_states = False
            self.out = getattr(torch, merge)
        else:
            raise ValueError("Merge option %s not allowed." % merge)
        self.supp = None

    def forward(self, x, adj, mask=None, u=None, cached_support=False):
        if cached_support and (self.supp is not None):
            supp = self.supp
        else:
            supp = SpatialConvOrderK.compute_support(adj, x.device)
            self.supp = supp if cached_support else None

        fwd_out, fwd_pred, fwd_repr, _ = self.fwd_rnn(x, supp, mask=mask, u=u, cached_support=cached_support)
        rev_x, rev_mask, rev_u = [reverse_tensor(tens) for tens in (x, mask, u)]
        *bwd_res, _ = self.bwd_rnn(rev_x, supp, mask=rev_mask, u=rev_u, cached_support=cached_support)
        bwd_out, bwd_pred, bwd_repr = [reverse_tensor(res) for res in bwd_res]

        if self._impute_from_states:
            inputs = [fwd_repr, bwd_repr, mask.to(dtype=x.dtype)]
            if self.emb is not None:
                b, *_, s = fwd_repr.shape
                inputs += [self.emb.view(1, *self.emb.shape, 1).expand(b, -1, -1, s)]
            imputation = self.out(torch.cat(inputs, dim=1))
        else:
            imputation = torch.stack([fwd_out, bwd_out], dim=1)
            imputation = self.out(imputation, dim=1)

        predictions = torch.stack([fwd_out, bwd_out, fwd_pred, bwd_pred], dim=0)
        return imputation, predictions


class GRINInputImputer(nn.Module):
    """Thin adapter that uses copied GRIN BiGRIL to fill lifecycle-missing inputs."""

    def __init__(self, configs, recipe_cfg):
        super(GRINInputImputer, self).__init__()
        self.raw_enc_in = getattr(configs, "raw_enc_in", configs.enc_in)
        self.total_enc_in = configs.enc_in
        self.role_start = self.raw_enc_in
        self.role_end = self.role_start + 4
        self.age_idx = self.role_end
        self.use_observation_mask = bool(
            getattr(configs, "lifecycle_observation_mask_feature", False)
        )
        self.observation_mask_idx = (
            self.age_idx + 1 if self.use_observation_mask else None
        )
        if (
            self.observation_mask_idx is not None
            and self.total_enc_in <= self.observation_mask_idx
        ):
            raise ValueError(
                "Explicit lifecycle observation mask is enabled, but the "
                f"configured input has only {self.total_enc_in} channels."
            )
        self.use_role_mask = bool(recipe_cfg.get("grin_use_role_mask", True))
        self.use_extra_features = bool(recipe_cfg.get("grin_use_extra_features", True))
        self.cached_support = bool(recipe_cfg.get("grin_cached_support", False))
        self.extra_feature_indices = [
            index
            for index in range(self.raw_enc_in, self.total_enc_in)
            if index != self.observation_mask_idx
        ]
        extra_dim = len(self.extra_feature_indices) if self.use_extra_features else 0
        hidden_dim = int(recipe_cfg.get("grin_hidden_dim", configs.d_model))
        ff_dim = int(recipe_cfg.get("grin_ff_dim", hidden_dim))
        self.bigril = BiGRIL(
            input_size=self.raw_enc_in,
            hidden_size=hidden_dim,
            ff_size=ff_dim,
            ff_dropout=float(recipe_cfg.get("grin_ff_dropout", 0.0)),
            n_layers=int(recipe_cfg.get("grin_layers", 1)),
            dropout=float(recipe_cfg.get("grin_dropout", 0.0)),
            n_nodes=None,
            support_len=2,
            kernel_size=int(recipe_cfg.get("grin_kernel_size", 2)),
            decoder_order=int(recipe_cfg.get("grin_decoder_order", 1)),
            u_size=extra_dim,
            embedding_size=0,
            layer_norm=bool(recipe_cfg.get("grin_layer_norm", False)),
            merge=str(recipe_cfg.get("grin_merge", "mlp")),
        )

    def _observed_mask(self, raw_x, extra_x, x_enc=None):
        if self.observation_mask_idx is not None:
            if x_enc is None:
                raise ValueError(
                    "Explicit observation-mask mode requires the original input tensor."
                )
            return x_enc[
                ...,
                self.observation_mask_idx:self.observation_mask_idx + 1,
            ] > 0.5
        if self.use_role_mask and extra_x is not None and self.total_enc_in >= self.role_end:
            role_seq = extra_x[..., :4]
            return ((role_seq[..., 0:1] + role_seq[..., 1:2]) > 0.5)
        return torch.abs(raw_x).sum(dim=-1, keepdim=True) > 0

    def without_observation_mask(self, x_enc):
        if self.observation_mask_idx is None:
            return x_enc
        return torch.cat(
            [
                x_enc[..., :self.observation_mask_idx],
                x_enc[..., self.observation_mask_idx + 1:],
            ],
            dim=-1,
        )

    def forward(self, adj, x_enc, x_t_mark=None, pos_mark=None):
        if x_enc.size(-1) != self.total_enc_in:
            raise ValueError(
                f"Lifecycle input channel mismatch: model expects {self.total_enc_in} channels "
                f"(raw={self.raw_enc_in}, extra={self.total_enc_in - self.raw_enc_in}), "
                f"but dataloader produced {x_enc.size(-1)}."
            )
        raw_x = x_enc[..., :self.raw_enc_in]
        extra_x = None
        if self.extra_feature_indices:
            extra_x = x_enc[..., self.extra_feature_indices]
        observed = self._observed_mask(raw_x, extra_x, x_enc=x_enc).expand_as(raw_x)
        x = raw_x.permute(0, 3, 2, 1).contiguous()
        mask = observed.permute(0, 3, 2, 1).contiguous()
        u = None
        if self.use_extra_features and extra_x is not None:
            u = extra_x.permute(0, 3, 2, 1).contiguous()
        imputed, _ = self.bigril(
            x,
            adj.to(device=x.device, dtype=x.dtype),
            mask=mask,
            u=u,
            cached_support=self.cached_support,
        )
        imputed = imputed.permute(0, 3, 2, 1).contiguous()
        filled_raw = torch.where(observed, raw_x, imputed)
        if self.total_enc_in <= self.raw_enc_in:
            return filled_raw
        return torch.cat([filled_raw, x_enc[..., self.raw_enc_in:]], dim=-1)
