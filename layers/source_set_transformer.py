import math

import torch
import torch.nn.functional as F
from torch import nn


class MAB(nn.Module):
    """Copied from references/set_transformer/modules.py::MAB."""

    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = torch.softmax(Q_.bmm(K_.transpose(1, 2)) / math.sqrt(self.dim_V), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, "ln0", None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, "ln1", None) is None else self.ln1(O)
        return O


class ISAB(nn.Module):
    """Copied from references/set_transformer/modules.py::ISAB."""

    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        return self.mab1(X, H)


class SetTransformerPatchMixer(nn.Module):
    """PatchSTG adapter around Set Transformer inducing-point attention."""

    def __init__(
        self,
        hidden_dim,
        num_heads=2,
        num_inds=8,
        ln=True,
        residual_scale=0.05,
    ):
        super().__init__()
        hidden_dim = int(hidden_dim)
        self.residual_scale = float(residual_scale)
        self.encoder = ISAB(hidden_dim, hidden_dim, int(num_heads), int(num_inds), ln=bool(ln))
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x):
        batch_size, steps, patch_num, patch_size, hidden_dim = x.shape
        flat = x.reshape(batch_size * steps, patch_num * patch_size, hidden_dim)
        mixed = self.encoder(flat)
        residual = self.out_proj(mixed).reshape(batch_size, steps, patch_num, patch_size, hidden_dim)
        return x + self.residual_scale * residual
