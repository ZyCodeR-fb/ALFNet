"""PyTorch port of the TSMixer residual block.

Ported from references/tsmixer/tsmixer_basic/models/tsmixer.py (Google Research,
TensorFlow) ``res_block``: pre-norm, a Dense on the transposed time axis with
activation, dropout and a residual add; then a pre-norm two-layer feature MLP
with dropout and a residual add. Both halves are kept as written.

The reference block mixes two axes, time and channel, because its input is
[Batch, Length, Channel]. Our tensor is [Batch, Time, Token, Dim]: the token
axis carries what "channel" carries there (the set of series being mixed), and
Dim is the embedding the reference does not have. Applying the same res_block
shape to all three axes is the direct extension, and it is one module rather
than a stack of different ones -- the alternative would be to flatten Token*Dim
into a single Dense, which is 10240-wide here.

Paper: TSMixer: An All-MLP Architecture for Time Series Forecasting (TMLR 2023).
"""

import torch
from torch import nn


class _MixLayer(nn.Module):
    """One res_block half: pre-norm, Dense over `axis`, dropout, residual."""

    def __init__(self, norm_dim, mix_dim, hidden_dim=None, dropout=0.0, activation="relu"):
        super().__init__()
        self.norm = nn.LayerNorm(norm_dim)
        self.act = nn.ReLU() if activation == "relu" else nn.GELU()
        self.dropout = nn.Dropout(dropout)
        if hidden_dim is None:                       # temporal / token linear
            self.fc1 = nn.Linear(mix_dim, mix_dim)
            self.fc2 = None
        else:                                        # feature linear
            self.fc1 = nn.Linear(mix_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, mix_dim)

    def forward(self, x, axis):
        """`x` is [..., mix_dim, norm_dim] once `axis` is moved to -2."""
        residual = x
        z = self.norm(x)
        z = z.transpose(axis, -1)
        z = self.act(self.fc1(z))
        if self.fc2 is not None:
            z = self.fc2(self.dropout(z))
        z = z.transpose(axis, -1)
        return residual + self.dropout(z)


class TSMixerBlock(nn.Module):
    """Time / token / feature mixing on a [B, T, K, D] tensor.

    Replaces the Crossformer two-stage attention layer. With the patch branch
    restored to three time segments, the time-mixing Dense finally acts on a
    real time axis: under the previous configuration a single stride-12
    convolution collapsed T to 1, so nothing temporal was being mixed at all.
    """

    def __init__(self, time_steps, num_tokens, hidden_dim, ff_dim=None,
                 dropout=0.1, activation="relu"):
        super().__init__()
        if int(time_steps) < 2:
            raise ValueError(
                "TSMixer needs at least two time segments; with tem_patchnum=1 "
                "the time axis is length 1 and time mixing is a no-op"
            )
        self.time_steps = int(time_steps)
        self.num_tokens = int(num_tokens)
        ff_dim = int(ff_dim or hidden_dim)
        self.time_mix = _MixLayer(hidden_dim, self.time_steps, None, dropout, activation)
        self.token_mix = _MixLayer(hidden_dim, self.num_tokens, None, dropout, activation)
        self.feature_mix = _MixLayer(hidden_dim, hidden_dim, ff_dim, dropout, activation)

    def forward(self, x):
        if x.shape[1] != self.time_steps or x.shape[2] != self.num_tokens:
            raise ValueError(
                f"TSMixerBlock expects [B, {self.time_steps}, {self.num_tokens}, D], "
                f"got {tuple(x.shape)}"
            )
        x = self.time_mix(x, axis=1)     # mixes across the 3 time segments
        x = self.token_mix(x, axis=2)    # mixes across tokens (spatial)
        x = self.feature_mix(x, axis=3)  # mixes across the embedding
        return x
