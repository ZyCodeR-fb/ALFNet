import torch
import torch.nn as nn
from einops import rearrange
from timm.models.vision_transformer import Attention, Mlp


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


class WindowAttBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, num, size, mlp_ratio=4.0, attention_order="depth_breadth", shift_size=0, attention_type="timm", lepe_kernel_size=5):
        super().__init__()
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.num, self.size = num, size
        self.attention_order = str(attention_order).lower()
        self.shift_size = int(shift_size)
        self.attention_type = str(attention_type).lower()
        self.use_depth_lepe = self.attention_type in {"biformer_lepe", "biformer_lepe_depth", "lepe_depth"}
        self.use_breadth_lepe = self.attention_type in {"biformer_lepe", "biformer_lepe_breadth", "lepe_breadth"}

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

    def forward(self, x):
        B,T,_,D = x.shape
        # P: ptach num and N: patch size
        P, N = self.num, self.size
        assert self.num * self.size == _
        shift_size = self.shift_size % _ if self.shift_size > 0 else 0
        if shift_size:
            x = torch.roll(x, shifts=-shift_size, dims=2)
        x = x.reshape(B, T, P, N, D)

        if self.attention_order in {"depth_breadth", "spatial_patch", "default"}:
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
