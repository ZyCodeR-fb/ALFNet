import torch
from torch import nn
import torch.nn.functional as F


class GCN(nn.Module):
    """Copied from references/TimeFilter/layers/TimeFilter_layers.py."""

    def __init__(self, dim, n_heads):
        super().__init__()
        self.proj = nn.Linear(dim, dim)
        self.n_heads = n_heads

    def forward(self, adj, x):
        B, L, D = x.shape
        x = self.proj(x).view(B, L, self.n_heads, -1)
        adj = F.normalize(adj, p=1, dim=-1)
        x = torch.einsum("bhij,bjhd->bihd", adj, x).contiguous()
        return x.view(B, L, -1)


class MaskMoE(nn.Module):
    """Copied from references/TimeFilter/layers/TimeFilter_layers.py::mask_moe."""

    def __init__(self, n_vars, top_p=0.5, num_experts=3, in_dim=96):
        super().__init__()
        self.num_experts = num_experts
        self.n_vars = n_vars
        self.in_dim = in_dim
        self.gate = nn.Linear(self.in_dim, num_experts, bias=False)
        self.noise = nn.Linear(self.in_dim, num_experts, bias=False)
        self.noisy_gating = True
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(2)
        self.top_p = top_p

    def cv_squared(self, x):
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def cross_entropy(self, x):
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return -torch.mul(x, torch.log(x + eps)).sum(dim=1).mean()

    def noisy_top_k_gating(self, x, is_training, noise_epsilon=1e-2):
        clean_logits = self.gate(x)
        if self.noisy_gating and is_training:
            raw_noise = self.noise(x)
            noise_stddev = self.softplus(raw_noise) + noise_epsilon
            logits = clean_logits + torch.randn_like(clean_logits) * noise_stddev
        else:
            logits = clean_logits

        logits = self.softmax(logits)
        loss_dynamic = self.cross_entropy(logits)
        sorted_probs, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        mask = cumulative_probs > self.top_p
        threshold_indices = mask.long().argmax(dim=-1)
        threshold_mask = torch.nn.functional.one_hot(threshold_indices, num_classes=sorted_indices.size(-1)).bool()
        mask = mask & ~threshold_mask

        top_p_mask = torch.zeros_like(mask)
        zero_indices = (mask == 0).nonzero(as_tuple=True)
        top_p_mask[zero_indices[0], zero_indices[1], sorted_indices[zero_indices[0], zero_indices[1], zero_indices[2]]] = 1

        sorted_probs = torch.where(mask, 0.0, sorted_probs)
        loss_importance = self.cv_squared(sorted_probs.sum(0))
        return top_p_mask, loss_importance + 0.1 * loss_dynamic

    def _build_masks(self, L, device, dtype):
        N = max(1, L // self.n_vars)
        masks = []
        base_index = torch.arange(L, device=device)
        for k in range(L):
            S = ((base_index % N == k % N) & (base_index != k)).to(dtype)
            T = ((base_index >= k // N * N) & (base_index < k // N * N + N)).to(dtype)
            ST = torch.ones(L, dtype=dtype, device=device) - S - T
            masks.append(torch.stack([S, T, ST], dim=0))
        return torch.stack(masks, dim=0)

    def forward(self, x, masks=None, is_training=None):
        B, H, L, _ = x.shape
        device = x.device
        dtype = torch.float32
        mask_base = torch.eye(L, device=device, dtype=dtype).unsqueeze(0).unsqueeze(0)
        if self.top_p == 0.0:
            return mask_base, x.new_tensor(0.0)

        gates, loss = self.noisy_top_k_gating(x.reshape(B * H, L, L), is_training)
        gates = gates.reshape(B, H, L, -1).float()
        if masks is None:
            masks = self._build_masks(L, device, dtype)
        mask = torch.einsum("bhli,lid->bhld", gates, masks) + mask_base
        return mask, loss


def mask_topk(x, alpha=0.5, largest=False):
    k = int(alpha * x.shape[-1])
    k = max(1, min(k, x.shape[-1]))
    _, topk_indices = torch.topk(x, k, dim=-1, largest=largest)
    mask = torch.ones_like(x, dtype=torch.float32)
    mask.scatter_(-1, topk_indices, 0)
    return mask


class GraphLearner(nn.Module):
    """Copied from references/TimeFilter/layers/TimeFilter_layers.py."""

    def __init__(self, dim, n_vars, top_p=0.5, in_dim=96):
        super().__init__()
        self.dim = dim
        self.proj_1 = nn.Linear(dim, dim)
        self.proj_2 = nn.Linear(dim, dim)
        self.n_vars = n_vars
        self.mask_moe = MaskMoE(n_vars, top_p=top_p, in_dim=in_dim)

    def forward(self, x, masks=None, alpha=0.5, is_training=False):
        adj = F.gelu(torch.einsum("bhid,bhjd->bhij", self.proj_1(x), self.proj_2(x)))
        adj = adj * mask_topk(adj, alpha)
        mask, loss = self.mask_moe(adj, masks, is_training)
        return adj * mask, loss


class GraphFilter(nn.Module):
    """Copied from references/TimeFilter/layers/TimeFilter_layers.py."""

    def __init__(self, dim, n_vars, n_heads=4, scale=None, top_p=0.5, dropout=0.0, in_dim=96):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.scale = dim ** (-0.5) if scale is None else scale
        self.dropout = nn.Dropout(dropout)
        self.graph_learner = GraphLearner(self.dim // self.n_heads, n_vars, top_p, in_dim=in_dim)
        self.graph_conv = GCN(self.dim, self.n_heads)

    def forward(self, x, masks=None, alpha=0.5, is_training=False):
        B, L, _ = x.shape
        adj, loss = self.graph_learner(
            x.reshape(B, L, self.n_heads, -1).permute(0, 2, 1, 3),
            masks,
            alpha,
            is_training,
        )
        adj = torch.softmax(adj, dim=-1)
        adj = self.dropout(adj)
        return self.graph_conv(adj, x), loss


class GraphBlock(nn.Module):
    """Copied from references/TimeFilter/layers/TimeFilter_layers.py."""

    def __init__(self, dim, n_vars, d_ff=None, n_heads=4, top_p=0.5, dropout=0.0, in_dim=96):
        super().__init__()
        self.dim = dim
        self.d_ff = dim * 4 if d_ff is None else d_ff
        self.gnn = GraphFilter(self.dim, n_vars, n_heads, top_p=top_p, dropout=dropout, in_dim=in_dim)
        self.norm1 = nn.LayerNorm(self.dim)
        self.ffn = nn.Sequential(
            nn.Linear(self.dim, self.d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_ff, self.dim),
        )
        self.norm2 = nn.LayerNorm(self.dim)

    def forward(self, x, masks=None, alpha=0.5, is_training=False):
        out, loss = self.gnn(self.norm1(x), masks, alpha, is_training)
        x = x + out
        x = x + self.ffn(self.norm2(x))
        return x, loss


class TimeFilterBackbone(nn.Module):
    """Copied from references/TimeFilter/layers/TimeFilter_layers.py::TimeFilter_Backbone."""

    def __init__(self, hidden_dim, n_vars, d_ff=None, n_heads=4, n_blocks=3, top_p=0.5, dropout=0.0, in_dim=96):
        super().__init__()
        self.blocks = nn.ModuleList([
            GraphBlock(hidden_dim, n_vars, d_ff or hidden_dim * 2, n_heads, top_p, dropout, in_dim)
            for _ in range(n_blocks)
        ])
        self.n_blocks = n_blocks

    def forward(self, x, masks=None, alpha=0.5, is_training=False):
        moe_loss = x.new_tensor(0.0)
        for block in self.blocks:
            x, loss = block(x, masks, alpha, is_training)
            moe_loss = moe_loss + loss
        return x, moe_loss / max(1, self.n_blocks)


class TimeFilterPatchMixer(nn.Module):
    """PatchSTG adapter around copied TimeFilter graph filtration blocks."""

    def __init__(
        self,
        hidden_dim,
        patch_num,
        patch_size,
        n_heads=4,
        n_blocks=1,
        d_ff=None,
        top_p=0.5,
        alpha=0.1,
        dropout=0.1,
        residual_scale=0.10,
    ):
        super().__init__()
        self.patch_num = int(patch_num)
        self.patch_size = int(patch_size)
        self.alpha = float(alpha)
        self.residual_scale = float(residual_scale)
        self.backbone = TimeFilterBackbone(
            hidden_dim=hidden_dim,
            n_vars=self.patch_size,
            d_ff=d_ff,
            n_heads=int(n_heads),
            n_blocks=int(n_blocks),
            top_p=float(top_p),
            dropout=float(dropout),
            in_dim=self.patch_num * self.patch_size,
        )
        self.last_moe_loss = None

    def forward(self, x):
        B, T, P, N, D = x.shape
        tokens = x.permute(0, 1, 3, 2, 4).reshape(B * T, N * P, D)
        out, loss = self.backbone(tokens, masks=None, alpha=self.alpha, is_training=self.training)
        self.last_moe_loss = loss
        out = out.reshape(B, T, N, P, D).permute(0, 1, 3, 2, 4).contiguous()
        return x + self.residual_scale * (out - x)
