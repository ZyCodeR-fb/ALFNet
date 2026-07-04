import math

import torch
import torch.nn.functional as F
from torch import nn


class PatternBankPatchMixer(nn.Module):
    """Patch-token adapter based on TPB PatternEncoder.

    The retrieval step mirrors references/TPB/model/Meta_Models/meta_patch.py::
    PatternEncoder: project query and pattern keys, compute scaled dot-product
    weights over a compact pattern bank, then reconstruct a pattern context.
    """

    def __init__(
        self,
        hidden_dim,
        num_patterns=16,
        residual_scale=0.05,
    ):
        super().__init__()
        hidden_dim = int(hidden_dim)
        self.num_patterns = int(num_patterns)
        self.hidden_dim = hidden_dim
        self.temperature = math.sqrt(hidden_dim)
        self.residual_scale = float(residual_scale)
        self.pattern = nn.Parameter(torch.empty(self.num_patterns, hidden_dim))
        nn.init.xavier_uniform_(self.pattern)
        self.Qnet = nn.Linear(hidden_dim, hidden_dim)
        self.Knet = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(3 * hidden_dim, hidden_dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x):
        batch_size, steps, patch_num, patch_size, hidden_dim = x.shape
        flat = x.reshape(batch_size * steps, patch_num * patch_size, hidden_dim)
        pattern = self.Knet(self.pattern)
        query = self.Qnet(flat)
        weight = torch.einsum("kd,bld->blk", pattern, query) / self.temperature
        weight = F.softmax(weight, dim=-1)
        context = torch.einsum("blk,kd->bld", weight, self.pattern)
        residual_input = torch.cat([flat, context, flat - context], dim=-1)
        residual = self.out_proj(residual_input).reshape(batch_size, steps, patch_num, patch_size, hidden_dim)
        return x + self.residual_scale * residual
