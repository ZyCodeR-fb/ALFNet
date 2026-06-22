import torch
import torch.nn as nn
from einops import rearrange, repeat
from timm.models.vision_transformer import Attention, Mlp


class PatchRevIN(nn.Module):
    # Adapted from RevIN for PatchSTG tensors shaped as [batch, time, node, channel].
    # The original module treats variables as features; here each node-channel pair keeps
    # its own time-wise statistics to avoid mixing sensor scales.
    def __init__(self, num_features, eps=1e-5, affine=True):
        super().__init__()
        self.num_features = int(num_features)
        self.eps = float(eps)
        self.affine = bool(affine)
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(1, 1, 1, self.num_features))
            self.affine_bias = nn.Parameter(torch.zeros(1, 1, 1, self.num_features))

    def forward(self, x, mode):
        if mode == "norm":
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == "denorm":
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _get_statistics(self, x):
        self.mean = torch.mean(x, dim=1, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = (x - self.mean) / self.stdev
        if self.affine:
            x = x * self.affine_weight + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = (x - self.affine_bias) / (self.affine_weight + self.eps * self.eps)
        return x * self.stdev + self.mean


class STAEAttentionLayer(nn.Module):
    # Copied from references/STAEformer/model/STAEformer.py::AttentionLayer,
    # with the class name changed to avoid colliding with timm Attention.
    def __init__(self, model_dim, num_heads=8, mask=False):
        super().__init__()

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.mask = mask

        self.head_dim = model_dim // num_heads

        self.FC_Q = nn.Linear(model_dim, model_dim)
        self.FC_K = nn.Linear(model_dim, model_dim)
        self.FC_V = nn.Linear(model_dim, model_dim)

        self.out_proj = nn.Linear(model_dim, model_dim)

    def forward(self, query, key, value):
        batch_size = query.shape[0]
        tgt_length = query.shape[-2]
        src_length = key.shape[-2]

        query = self.FC_Q(query)
        key = self.FC_K(key)
        value = self.FC_V(value)

        query = torch.cat(torch.split(query, self.head_dim, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.head_dim, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.head_dim, dim=-1), dim=0)

        key = key.transpose(-1, -2)

        attn_score = (query @ key) / self.head_dim**0.5

        if self.mask:
            mask = torch.ones(tgt_length, src_length, dtype=torch.bool, device=query.device).tril()
            attn_score.masked_fill_(~mask, -torch.inf)

        attn_score = torch.softmax(attn_score, dim=-1)
        out = attn_score @ value
        out = torch.cat(torch.split(out, batch_size, dim=0), dim=-1)

        out = self.out_proj(out)

        return out


class STAESelfAttentionLayer(nn.Module):
    # Copied from references/STAEformer/model/STAEformer.py::SelfAttentionLayer,
    # with only the attention class name adjusted.
    def __init__(
        self, model_dim, feed_forward_dim=2048, num_heads=8, dropout=0, mask=False
    ):
        super().__init__()

        self.attn = STAEAttentionLayer(model_dim, num_heads, mask)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim),
        )
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, dim=-2):
        x = x.transpose(dim, -2)
        residual = x
        out = self.attn(x, x, x)
        out = self.dropout1(out)
        out = self.ln1(residual + out)

        residual = out
        out = self.feed_forward(out)
        out = self.dropout2(out)
        out = self.ln2(residual + out)

        out = out.transpose(dim, -2)
        return out


class BiFormerAttentionLePE(nn.Module):
    # Copied from references/BiFormer/models/_common.py::AttentionLePE,
    # with the class name changed for local PatchSTG use.
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., side_dwconv=5):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.lepe = nn.Conv2d(dim, dim, kernel_size=side_dwconv, stride=1, padding=side_dwconv//2, groups=dim) if side_dwconv > 0 else \
                    lambda x: torch.zeros_like(x)

    def forward(self, x):
        _, H, W, _ = x.size()
        x = rearrange(x, 'n h w c -> n (h w) c')

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        lepe = self.lepe(rearrange(x, 'n (h w) c -> n c h w', h=H, w=W))
        lepe = rearrange(lepe, 'n c h w -> n (h w) c')

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = x + lepe

        x = self.proj(x)
        x = self.proj_drop(x)

        x = rearrange(x, 'n (h w) c -> n h w c', h=H, w=W)
        return x


class CrossformerFullAttention(nn.Module):
    # Copied from references/Crossformer/cross_models/attn.py::FullAttention,
    # with only the class name changed for local PatchSTG use.
    def __init__(self, scale=None, attention_dropout=0.1):
        super(CrossformerFullAttention, self).__init__()
        self.scale = scale
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values):
        _, _, _, E = queries.shape
        scale = self.scale or E ** -0.5
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        attn = self.dropout(torch.softmax(scale * scores, dim=-1))
        out = torch.einsum("bhls,bshd->blhd", attn, values)
        return out.contiguous()


class CrossformerAttentionLayer(nn.Module):
    # Copied from references/Crossformer/cross_models/attn.py::AttentionLayer,
    # with only the class name changed for local PatchSTG use.
    def __init__(self, d_model, n_heads, d_keys=None, d_values=None, dropout=0.1):
        super(CrossformerAttentionLayer, self).__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.inner_attention = CrossformerFullAttention(scale=None, attention_dropout=dropout)
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)
        out = self.inner_attention(queries, keys, values)
        out = out.view(B, L, -1)
        return self.out_projection(out)


class CrossformerTwoStageAttentionLayer(nn.Module):
    # Copied from references/Crossformer/cross_models/attn.py::TwoStageAttentionLayer,
    # with the class name changed; PatchSTG wraps input as [batch*time, nodes, patches, dim].
    def __init__(self, seg_num, factor, d_model, n_heads, d_ff=None, dropout=0.1):
        super(CrossformerTwoStageAttentionLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.time_attention = CrossformerAttentionLayer(d_model, n_heads, dropout=dropout)
        self.dim_sender = CrossformerAttentionLayer(d_model, n_heads, dropout=dropout)
        self.dim_receiver = CrossformerAttentionLayer(d_model, n_heads, dropout=dropout)
        self.router = nn.Parameter(torch.randn(seg_num, factor, d_model))

        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.MLP1 = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
        self.MLP2 = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))

    def forward(self, x):
        batch = x.shape[0]
        time_in = rearrange(x, "b ts_d seg_num d_model -> (b ts_d) seg_num d_model")
        time_enc = self.time_attention(time_in, time_in, time_in)
        dim_in = time_in + self.dropout(time_enc)
        dim_in = self.norm1(dim_in)
        dim_in = dim_in + self.dropout(self.MLP1(dim_in))
        dim_in = self.norm2(dim_in)

        dim_send = rearrange(dim_in, "(b ts_d) seg_num d_model -> (b seg_num) ts_d d_model", b=batch)
        batch_router = repeat(self.router, "seg_num factor d_model -> (repeat seg_num) factor d_model", repeat=batch)
        dim_buffer = self.dim_sender(batch_router, dim_send, dim_send)
        dim_receive = self.dim_receiver(dim_send, dim_buffer, dim_buffer)
        dim_enc = dim_send + self.dropout(dim_receive)
        dim_enc = self.norm3(dim_enc)
        dim_enc = dim_enc + self.dropout(self.MLP2(dim_enc))
        dim_enc = self.norm4(dim_enc)
        return rearrange(dim_enc, "(b seg_num) ts_d d_model -> b ts_d seg_num d_model", b=batch)


class WindowAttBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, num, size, mlp_ratio=4.0, attention_order="depth_breadth", shift_size=0, attention_type="timm", lepe_kernel_size=5, crossformer_router_factor=4, crossformer_dropout=0.1):
        super().__init__()
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.num, self.size = num, size
        self.attention_order = str(attention_order).lower()
        self.shift_size = int(shift_size)
        self.attention_type = str(attention_type).lower()
        self.use_crossformer_tsa = self.attention_type in {"crossformer_tsa", "crossformer", "tsa"}
        self.use_depth_lepe = self.attention_type in {"biformer_lepe", "biformer_lepe_depth", "lepe_depth"}
        self.use_breadth_lepe = self.attention_type in {"biformer_lepe", "biformer_lepe_breadth", "lepe_breadth"}
        if self.use_crossformer_tsa:
            self.tsa = CrossformerTwoStageAttentionLayer(
                seg_num=num,
                factor=int(crossformer_router_factor),
                d_model=hidden_size,
                n_heads=num_heads,
                d_ff=mlp_hidden_dim,
                dropout=float(crossformer_dropout),
            )

        self.nnorm1 = nn.LayerNorm(hidden_size)
        if self.use_breadth_lepe:
            self.nattn = BiFormerAttentionLePE(hidden_size, num_heads=num_heads, qkv_bias=True, attn_drop=0.1, proj_drop=0.1, side_dwconv=lepe_kernel_size)
        else:
            self.nattn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, attn_drop=0.1, proj_drop=0.1)
        self.nnorm2 = nn.LayerNorm(hidden_size)
        self.nmlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=0.1)

        self.snorm1 = nn.LayerNorm(hidden_size)
        if self.use_depth_lepe:
            self.sattn = BiFormerAttentionLePE(hidden_size, num_heads=num_heads, qkv_bias=True, attn_drop=0.1, proj_drop=0.1, side_dwconv=lepe_kernel_size)
        else:
            self.sattn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, attn_drop=0.1, proj_drop=0.1)
        self.snorm2 = nn.LayerNorm(hidden_size)
        self.smlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=0.1)

    def _depth_attention(self, x, B, T, P, N, D):
        qkv = self.snorm1(x.reshape(B*T*P,N,D))
        if self.use_depth_lepe:
            attn_out = self.sattn(qkv.reshape(B*T*P, N, 1, D)).reshape(B,T,P,N,D)
        else:
            attn_out = self.sattn(qkv).reshape(B,T,P,N,D)
        x = x + attn_out
        x = x + self.smlp(self.snorm2(x))
        return x

    def _breadth_attention(self, x, B, T, P, N, D):
        qkv = self.nnorm1(x.transpose(2,3).reshape(B*T*N,P,D))
        if self.use_breadth_lepe:
            attn_out = self.nattn(qkv.reshape(B*T*N, P, 1, D)).reshape(B,T,N,P,D).transpose(2,3)
        else:
            attn_out = self.nattn(qkv).reshape(B,T,N,P,D).transpose(2,3)
        x = x + attn_out
        x = x + self.nmlp(self.nnorm2(x))
        return x

    def _crossformer_tsa_attention(self, x, B, T, P, N, D):
        tsa_in = x.permute(0, 1, 3, 2, 4).reshape(B * T, N, P, D)
        tsa_out = self.tsa(tsa_in)
        return tsa_out.reshape(B, T, N, P, D).permute(0, 1, 3, 2, 4).contiguous()

    def forward(self, x):
        B,T,_,D = x.shape
        # P: ptach num and N: patch size
        P, N = self.num, self.size
        assert self.num * self.size == _
        shift_size = self.shift_size % _ if self.shift_size > 0 else 0
        if shift_size:
            x = torch.roll(x, shifts=-shift_size, dims=2)
        x = x.reshape(B, T, P, N, D)

        if self.use_crossformer_tsa:
            x = self._crossformer_tsa_attention(x, B, T, P, N, D)
        elif self.attention_order in {"depth_breadth", "spatial_patch", "default"}:
            x = self._depth_attention(x, B, T, P, N, D)
            x = self._breadth_attention(x, B, T, P, N, D)
        elif self.attention_order in {"breadth_depth", "cross_local"}:
            x = self._breadth_attention(x, B, T, P, N, D)
            x = self._depth_attention(x, B, T, P, N, D)
        else:
            raise ValueError(f"Unsupported PatchSTG attention_order: {self.attention_order}")

        x = x.reshape(B,T,-1,D)
        if shift_size:
            x = torch.roll(x, shifts=shift_size, dims=2)
        return x

class PatchSTG(nn.Module):
    def __init__(self, tem_patchsize, tem_patchnum,
                        node_num, spa_patchsize, spa_patchnum,
                        tod, dow,
                        layers, factors,
                        input_dims, node_dims, tod_dims, dow_dims,
                        ori_parts_idx, reo_parts_idx, reo_all_idx,
                        attention_order="depth_breadth",
                        shift_size=0,
                        attention_type="timm",
                        lepe_kernel_size=5,
                        crossformer_router_factor=4,
                        crossformer_dropout=0.1,
                        use_staeformer_adapter=False,
                        staeformer_adapter_mode="temporal",
                        staeformer_adapter_layers=1,
                        staeformer_num_heads=4,
                        staeformer_feed_forward_dim=256,
                        staeformer_dropout=0.1
                ):
        super(PatchSTG, self).__init__()
        self.node_num = node_num
        self.ori_parts_idx, self.reo_parts_idx = ori_parts_idx, reo_parts_idx
        self.reo_all_idx = reo_all_idx
        self.tod, self.dow = tod, dow
        self.use_staeformer_adapter = bool(use_staeformer_adapter)
        self.staeformer_adapter_mode = str(staeformer_adapter_mode).lower()

        # model_dims = input_emb + spa_emb + tem_emb
        dims = input_dims + tod_dims + dow_dims + node_dims

        # spatio-temporal embedding -> section 4.1 in paper
        # input_emb
        self.input_st_fc = nn.Conv2d(in_channels=3, out_channels=input_dims, kernel_size=(1, tem_patchsize), stride=(1, tem_patchsize), bias=True)
        # spa_emb
        self.node_emb = nn.Parameter(
                torch.empty(node_num, node_dims))
        nn.init.xavier_uniform_(self.node_emb)
        # tem_emb
        self.time_in_day_emb = nn.Parameter(
                torch.empty(tod, tod_dims))
        nn.init.xavier_uniform_(self.time_in_day_emb)
        self.day_in_week_emb = nn.Parameter(
                torch.empty(dow, dow_dims))
        nn.init.xavier_uniform_(self.day_in_week_emb)

        # dual attention encoder -> section 4.3 in paper, factors for merging the leaf nodes of KDTree
        self.spa_encoder = nn.ModuleList([
            WindowAttBlock(
                dims,
                1,
                spa_patchnum//factors,
                spa_patchsize*factors,
                mlp_ratio=1,
                attention_order=attention_order,
                shift_size=(int(shift_size) if i % 2 == 1 else 0),
                attention_type=attention_type,
                lepe_kernel_size=lepe_kernel_size,
                crossformer_router_factor=crossformer_router_factor,
                crossformer_dropout=crossformer_dropout,
            )
            for i in range(layers)
        ])
        self.staeformer_adapter_t = nn.ModuleList()
        self.staeformer_adapter_s = nn.ModuleList()
        if self.use_staeformer_adapter:
            if self.staeformer_adapter_mode in {"temporal", "spatiotemporal", "both"}:
                self.staeformer_adapter_t = nn.ModuleList([
                    STAESelfAttentionLayer(dims, staeformer_feed_forward_dim, staeformer_num_heads, staeformer_dropout)
                    for _ in range(staeformer_adapter_layers)
                ])
            if self.staeformer_adapter_mode in {"spatial", "spatiotemporal", "both"}:
                self.staeformer_adapter_s = nn.ModuleList([
                    STAESelfAttentionLayer(dims, staeformer_feed_forward_dim, staeformer_num_heads, staeformer_dropout)
                    for _ in range(staeformer_adapter_layers)
                ])

        # projection decoder -> section 4.4 in paper
        self.regression_conv = nn.Conv2d(in_channels=tem_patchnum*dims, out_channels=tem_patchsize*tem_patchnum, kernel_size=(1, 1), bias=True)

    def forward(self, x, te):
        # x: [B,T,N,1] input traffic
        # te: [B,T,N,2] time information

        # spatio-temporal embedding -> section 4.1 in paper
        embeded_x = self.embedding(x, te)
        rex = embeded_x[:,:,self.reo_all_idx,:] # select patched points

        # dual attention encoder -> section 4.3 in paper
        for block in self.spa_encoder:
            rex = block(rex)

        orginal = torch.zeros(rex.shape[0],rex.shape[1],self.node_num,rex.shape[-1]).to(x.device)
        orginal[:,:,self.ori_parts_idx,:] = rex[:,:,self.reo_parts_idx,:] # back to the original indices
        for attn in self.staeformer_adapter_t:
            orginal = attn(orginal, dim=1)
        for attn in self.staeformer_adapter_s:
            orginal = attn(orginal, dim=2)

        # projection decoder -> section 4.4 in paper
        pred_y = self.regression_conv(orginal.transpose(2,3).reshape(orginal.shape[0],-1,orginal.shape[-2],1))

        return pred_y # [B,T,N,1]

    def embedding(self, x, te):
        b,t,n,_ = x.shape

        # input traffic + time of day + day of week as the input signal
        x1 = torch.cat([x,(te[...,0:1]/self.tod),(te[...,1:2]/self.dow)], -1).float()
        input_data = self.input_st_fc(x1.transpose(1,3)).transpose(1,3)
        t, d = input_data.shape[1], input_data.shape[-1]        

        # cat time of day embedding
        t_i_d_data = te[:, -input_data.shape[1]:, :, 0].long().clamp(min=0, max=self.tod - 1).to(self.time_in_day_emb.device)
        input_data = torch.cat([input_data, self.time_in_day_emb[t_i_d_data]], -1)

        # cat day of week embedding
        d_i_w_data = te[:, -input_data.shape[1]:, :, 1].long().clamp(min=0, max=self.dow - 1).to(self.day_in_week_emb.device)
        input_data = torch.cat([input_data, self.day_in_week_emb[d_i_w_data]], -1)

        # cat spatial embedding
        node_emb = self.node_emb.unsqueeze(0).unsqueeze(1).expand(b, t, -1, -1)
        input_data = torch.cat([input_data, node_emb], -1)

        return input_data
