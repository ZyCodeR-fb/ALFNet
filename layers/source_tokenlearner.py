"""PyTorch port of TokenLearner v1.1 / TokenFuser.

Ported from references/token_learner/model.py (Google Research, JAX/Flax):
``TokenLearnerModuleV11`` and ``TokenFuser``. The formulas are kept as written
there -- LayerNorm, an MLP with GELU producing the token-masking logits, softmax
over the spatial axis for learning, sigmoid for fusing, and a zero-initialised
Dense on the token axis inside the fuser. The only adaptation is the spatial
axis: the reference flattens ``h*w`` image patches, we pass the node axis
directly, so nothing else has to change.

Paper: TokenLearner: Adaptive Space-Time Tokenization for Videos (NeurIPS 2021).
"""

import torch
from torch import nn


class MlpBlock(nn.Module):
    """attention_layers.MlpBlock as used by the reference token modules."""

    def __init__(self, in_dim, mlp_dim, out_dim, dropout_rate=0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, mlp_dim)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(mlp_dim, out_dim)
        self.drop2 = nn.Dropout(dropout_rate)

    def forward(self, x):
        return self.drop2(self.fc2(self.drop1(self.act(self.fc1(x)))))


class TokenLearnerV11(nn.Module):
    """`[B, S, C] -> [B, num_tokens, C]` by learned spatial attention."""

    def __init__(self, in_dim, num_tokens, bottleneck_dim=64, dropout_rate=0.0):
        super().__init__()
        self.num_tokens = int(num_tokens)
        self.norm = nn.LayerNorm(in_dim)
        self.token_masking = MlpBlock(in_dim, bottleneck_dim, self.num_tokens, dropout_rate)

    def forward(self, inputs):
        selected = self.token_masking(self.norm(inputs))      # [B, S, n_token]
        selected = selected.transpose(1, 2)                   # [B, n_token, S]
        selected = torch.softmax(selected, dim=-1)
        # 'bsi,bid->bsd' in the reference's einsum notation
        return torch.einsum("bts,bsd->btd", selected, inputs)


class TokenFuser(nn.Module):
    """`([B, num_tokens, C], [B, S, C]) -> [B, S, C]`.

    The zero-initialised token-axis Dense is the reference's, and it is what
    makes an untrained fuser emit zeros rather than noise.
    """

    def __init__(self, in_dim, num_tokens, bottleneck_dim=64, dropout_rate=0.0,
                 use_normalization=True):
        super().__init__()
        self.use_normalization = bool(use_normalization)
        self.num_tokens = int(num_tokens)
        self.mix_norm1 = nn.LayerNorm(in_dim)
        self.token_mix = nn.Linear(self.num_tokens, self.num_tokens)
        nn.init.zeros_(self.token_mix.weight)
        nn.init.zeros_(self.token_mix.bias)
        self.mix_norm2 = nn.LayerNorm(in_dim)
        self.original_norm = nn.LayerNorm(in_dim)
        self.token_masking = MlpBlock(in_dim, bottleneck_dim, self.num_tokens, dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, inputs, original):
        if self.use_normalization:
            inputs = self.mix_norm1(inputs)
        inputs = self.token_mix(inputs.transpose(1, 2)).transpose(1, 2)
        if self.use_normalization:
            inputs = self.mix_norm2(inputs)
        mix = torch.sigmoid(self.token_masking(self.original_norm(original)))
        return self.dropout(torch.einsum("bsc,bhs->bhc", inputs, mix))


class NodeTokenizer(nn.Module):
    """Adaptive replacement for the KD-tree reorder, over the node axis.

    The KD-tree groups nodes by fixed coordinates, so the grouping cannot react
    to traffic state, to which sensors are currently missing, or to a lifecycle
    event. TokenLearner derives the groups from the features themselves and
    TokenFuser maps them back to every node, which node-level forecasting needs
    (pooling methods that drop nodes cannot).
    """

    def __init__(self, in_dim, num_tokens, bottleneck_dim=64, dropout_rate=0.0):
        super().__init__()
        self.num_tokens = int(num_tokens)
        self.learner = TokenLearnerV11(in_dim, num_tokens, bottleneck_dim, dropout_rate)
        self.fuser = TokenFuser(in_dim, num_tokens, bottleneck_dim, dropout_rate)

    def tokenize(self, x):
        """[B, T, N, D] -> [B, T, num_tokens, D]"""
        batch, steps, nodes, dim = x.shape
        flat = x.reshape(batch * steps, nodes, dim)
        return self.learner(flat).reshape(batch, steps, self.num_tokens, dim)

    def fuse(self, tokens, original):
        """([B, T, num_tokens, D], [B, T, N, D]) -> [B, T, N, D]"""
        batch, steps, _, dim = tokens.shape
        nodes = original.shape[2]
        fused = self.fuser(
            tokens.reshape(batch * steps, self.num_tokens, dim),
            original.reshape(batch * steps, nodes, dim),
        )
        return fused.reshape(batch, steps, nodes, dim)
