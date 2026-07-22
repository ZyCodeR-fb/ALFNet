import torch
import torch.nn as nn
from einops import rearrange, repeat

from layers.source_timefilter import TimeFilterPatchMixer


class CrossformerFullAttention(nn.Module):
    """Crossformer FullAttention with only local naming adapted."""

    def __init__(self, scale=None, attention_dropout=0.1):
        super(CrossformerFullAttention, self).__init__()
        self.scale = scale
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values):
        _, _, _, embed_dim = queries.shape
        scale = self.scale or embed_dim ** -0.5
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        attn = self.dropout(torch.softmax(scale * scores, dim=-1))
        return torch.einsum("bhls,bshd->blhd", attn, values).contiguous()


class CrossformerAttentionLayer(nn.Module):
    """Crossformer AttentionLayer with only local naming adapted."""

    def __init__(self, d_model, n_heads, d_keys=None, d_values=None, dropout=0.1):
        super(CrossformerAttentionLayer, self).__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.inner_attention = CrossformerFullAttention(
            scale=None,
            attention_dropout=dropout,
        )
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values):
        batch_size, query_len, _ = queries.shape
        _, key_len, _ = keys.shape
        queries = self.query_projection(queries).view(
            batch_size, query_len, self.n_heads, -1
        )
        keys = self.key_projection(keys).view(
            batch_size, key_len, self.n_heads, -1
        )
        values = self.value_projection(values).view(
            batch_size, key_len, self.n_heads, -1
        )
        output = self.inner_attention(queries, keys, values)
        return self.out_projection(output.view(batch_size, query_len, -1))


class CrossformerTwoStageAttentionLayer(nn.Module):
    """Crossformer TwoStageAttentionLayer adapted to PatchSTG patch tensors."""

    def __init__(self, seg_num, factor, d_model, n_heads, d_ff=None, dropout=0.1):
        super(CrossformerTwoStageAttentionLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.time_attention = CrossformerAttentionLayer(
            d_model, n_heads, dropout=dropout
        )
        self.dim_sender = CrossformerAttentionLayer(
            d_model, n_heads, dropout=dropout
        )
        self.dim_receiver = CrossformerAttentionLayer(
            d_model, n_heads, dropout=dropout
        )
        self.router = nn.Parameter(torch.randn(seg_num, factor, d_model))
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.MLP1 = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.MLP2 = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        batch = x.shape[0]
        time_in = rearrange(
            x,
            "b ts_d seg_num d_model -> (b ts_d) seg_num d_model",
        )
        time_enc = self.time_attention(time_in, time_in, time_in)
        dim_in = self.norm1(time_in + self.dropout(time_enc))
        dim_in = self.norm2(dim_in + self.dropout(self.MLP1(dim_in)))
        dim_send = rearrange(
            dim_in,
            "(b ts_d) seg_num d_model -> (b seg_num) ts_d d_model",
            b=batch,
        )
        batch_router = repeat(
            self.router,
            "seg_num factor d_model -> (repeat seg_num) factor d_model",
            repeat=batch,
        )
        dim_buffer = self.dim_sender(batch_router, dim_send, dim_send)
        dim_receive = self.dim_receiver(dim_send, dim_buffer, dim_buffer)
        dim_enc = self.norm3(dim_send + self.dropout(dim_receive))
        dim_enc = self.norm4(dim_enc + self.dropout(self.MLP2(dim_enc)))
        return rearrange(
            dim_enc,
            "(b seg_num) ts_d d_model -> b ts_d seg_num d_model",
            b=batch,
        )


class WindowAttBlock(nn.Module):

    def __init__(
        self,
        hidden_size,
        num_heads,
        num,
        size,
        mlp_ratio=4.0,
        crossformer_router_factor=4,
        crossformer_dropout=0.1,
        timefilter_n_heads=4,
        timefilter_layers=1,
        timefilter_d_ff=0,
        timefilter_top_p=0.5,
        timefilter_alpha=0.1,
        timefilter_dropout=0.1,
        timefilter_residual_scale=0.10,
    ):
        super().__init__()
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.num, self.size = num, size
        self.tsa = CrossformerTwoStageAttentionLayer(
            seg_num=num,
            factor=int(crossformer_router_factor),
            d_model=hidden_size,
            n_heads=num_heads,
            d_ff=mlp_hidden_dim,
            dropout=float(crossformer_dropout),
        )
        self.timefilter_patch_mixer = TimeFilterPatchMixer(
            hidden_dim=hidden_size,
            patch_num=num,
            patch_size=size,
            n_heads=timefilter_n_heads,
            n_blocks=timefilter_layers,
            d_ff=None if int(timefilter_d_ff) <= 0 else int(timefilter_d_ff),
            top_p=timefilter_top_p,
            alpha=timefilter_alpha,
            dropout=timefilter_dropout,
            residual_scale=timefilter_residual_scale,
        )

    def forward(self, x):
        batch, time_steps, token_count, hidden_dim = x.shape
        if self.num * self.size != token_count:
            raise ValueError(
                f"Patch token mismatch: expected {self.num * self.size}, "
                f"got {token_count}."
            )
        x = x.reshape(
            batch,
            time_steps,
            self.num,
            self.size,
            hidden_dim,
        )
        tsa_in = x.permute(0, 1, 3, 2, 4).reshape(
            batch * time_steps,
            self.size,
            self.num,
            hidden_dim,
        )
        x = self.tsa(tsa_in).reshape(
            batch,
            time_steps,
            self.size,
            self.num,
            hidden_dim,
        ).permute(0, 1, 3, 2, 4).contiguous()
        x = self.timefilter_patch_mixer(x)
        return x.reshape(batch, time_steps, token_count, hidden_dim)


class PatchSTG(nn.Module):

    def __init__(
        self,
        tem_patchsize,
        tem_patchnum,
        node_num,
        spa_patchsize,
        spa_patchnum,
        tod,
        dow,
        layers,
        factors,
        input_dims,
        node_dims,
        tod_dims,
        dow_dims,
        ori_parts_idx,
        reo_parts_idx,
        reo_all_idx,
        crossformer_router_factor=4,
        crossformer_dropout=0.1,
        timefilter_n_heads=4,
        timefilter_layers=1,
        timefilter_d_ff=0,
        timefilter_top_p=0.5,
        timefilter_alpha=0.1,
        timefilter_dropout=0.1,
        timefilter_residual_scale=0.10,
    ):
        super(PatchSTG, self).__init__()
        self.node_num = node_num
        self.ori_parts_idx, self.reo_parts_idx = ori_parts_idx, reo_parts_idx
        self.reo_all_idx = reo_all_idx
        self.tod, self.dow = tod, dow
        dims = input_dims + tod_dims + dow_dims + node_dims

        self.input_st_fc = nn.Conv2d(
            in_channels=3,
            out_channels=input_dims,
            kernel_size=(1, tem_patchsize),
            stride=(1, tem_patchsize),
            bias=True,
        )
        self.node_emb = nn.Parameter(torch.empty(node_num, node_dims))
        nn.init.xavier_uniform_(self.node_emb)
        self.time_in_day_emb = nn.Parameter(torch.empty(tod, tod_dims))
        nn.init.xavier_uniform_(self.time_in_day_emb)
        self.day_in_week_emb = nn.Parameter(torch.empty(dow, dow_dims))
        nn.init.xavier_uniform_(self.day_in_week_emb)

        self.spa_encoder = nn.ModuleList([
            WindowAttBlock(
                dims,
                1,
                spa_patchnum // factors,
                spa_patchsize * factors,
                mlp_ratio=1,
                crossformer_router_factor=crossformer_router_factor,
                crossformer_dropout=crossformer_dropout,
                timefilter_n_heads=timefilter_n_heads,
                timefilter_layers=timefilter_layers,
                timefilter_d_ff=timefilter_d_ff,
                timefilter_top_p=timefilter_top_p,
                timefilter_alpha=timefilter_alpha,
                timefilter_dropout=timefilter_dropout,
                timefilter_residual_scale=timefilter_residual_scale,
            )
            for _ in range(layers)
        ])
        self.regression_conv = nn.Conv2d(
            in_channels=tem_patchnum * dims,
            out_channels=tem_patchsize * tem_patchnum,
            kernel_size=(1, 1),
            bias=True,
        )

    def forward(self, x, te):
        embedded_x = self.embedding(x, te)
        reordered = embedded_x[:, :, self.reo_all_idx, :]
        for block in self.spa_encoder:
            reordered = block(reordered)

        original = torch.zeros(
            reordered.shape[0],
            reordered.shape[1],
            self.node_num,
            reordered.shape[-1],
            device=x.device,
            dtype=reordered.dtype,
        )
        original[:, :, self.ori_parts_idx, :] = (
            reordered[:, :, self.reo_parts_idx, :]
        )
        return self.regression_conv(
            original.transpose(2, 3).reshape(
                original.shape[0],
                -1,
                original.shape[-2],
                1,
            )
        )

    def embedding(self, x, te):
        batch, _, _, _ = x.shape
        input_signal = torch.cat([
            x,
            te[..., 0:1] / self.tod,
            te[..., 1:2] / self.dow,
        ], dim=-1).float()
        input_data = self.input_st_fc(input_signal.transpose(1, 3)).transpose(1, 3)
        tod_index = te[:, -input_data.shape[1]:, :, 0].long().clamp(
            min=0,
            max=self.tod - 1,
        ).to(self.time_in_day_emb.device)
        input_data = torch.cat([
            input_data,
            self.time_in_day_emb[tod_index],
        ], dim=-1)
        dow_index = te[:, -input_data.shape[1]:, :, 1].long().clamp(
            min=0,
            max=self.dow - 1,
        ).to(self.day_in_week_emb.device)
        input_data = torch.cat([
            input_data,
            self.day_in_week_emb[dow_index],
        ], dim=-1)
        node_emb = self.node_emb.unsqueeze(0).unsqueeze(1).expand(
            batch,
            input_data.shape[1],
            -1,
            -1,
        )
        return torch.cat([input_data, node_emb], dim=-1)
