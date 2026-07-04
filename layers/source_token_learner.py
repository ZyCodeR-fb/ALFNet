import torch
from torch import nn


class TokenLearnerPatchMixer(nn.Module):
    """PyTorch adapter of TokenLearnerV1.1 for PatchSTG patch tokens.

    The token-selection rule follows references/token_learner/model.py::
    TokenLearnerModuleV11: LayerNorm -> MLP -> softmax masks -> weighted
    token pooling. A small token-to-token attention fuses the learned tokens
    back to the original patch-token grid.
    """

    def __init__(
        self,
        hidden_dim,
        num_tokens=8,
        bottleneck_dim=64,
        dropout=0.0,
        residual_scale=0.05,
    ):
        super().__init__()
        hidden_dim = int(hidden_dim)
        self.num_tokens = int(num_tokens)
        self.residual_scale = float(residual_scale)
        self.norm = nn.LayerNorm(hidden_dim)
        self.token_masking = nn.Sequential(
            nn.Linear(hidden_dim, int(bottleneck_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(bottleneck_dim), self.num_tokens),
        )
        self.query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x):
        batch_size, steps, patch_num, patch_size, hidden_dim = x.shape
        flat = x.reshape(batch_size * steps, patch_num * patch_size, hidden_dim)
        selected = self.token_masking(self.norm(flat))
        selected = torch.softmax(selected.transpose(1, 2), dim=-1)
        learned_tokens = torch.einsum("bsl,bld->bsd", selected, flat)

        scale = hidden_dim ** -0.5
        attn = torch.softmax(
            torch.matmul(self.query(flat), self.key(learned_tokens).transpose(-1, -2)) * scale,
            dim=-1,
        )
        fused = torch.matmul(attn, self.value(learned_tokens))
        residual = self.out_proj(fused).reshape(batch_size, steps, patch_num, patch_size, hidden_dim)
        return x + self.residual_scale * residual
