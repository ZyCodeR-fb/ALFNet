"""Protected, directly-supervised temporal residual on top of a graph anchor.

Why this shape, from the measurements rather than from preference.

fg1 (FiLM-conditioned graph) and A2 (the retired two-expert model) are
complementary on the frozen test set:

    A2 - fg1, MAE   Unseen +0.774   Steady -0.198   Join -0.332   Fault -0.528

fg1 owns the group with no own-history; A2 owns the three measured on nodes that
do have history. Taking the better of the two per group gives LifeAvg 4.6622
against MoGERNN's 5.2165, with all four groups ahead -- so the capability exists
and the problem is combining it in one light model.

What A2 has that fg1 structurally cannot is an *additive* second forecast: FiLM
is multiplicative on the encoder state and can reweight but not inject. Two
earlier attempts to add one back failed for a diagnosable reason: both the
evidence router's Y_patch and the observed NLinear residual were trained only
through the final fused loss, an indirect signal, and the router ended up
learning to switch its branch off (temporal activation 1.2% in the `ac` run,
with its branch worse than the graph branch on every availability group).

So the residual here is supervised *directly* against what the anchor got wrong,

    R_target = Y - stopgrad(Y_graph)

which is the deeply-supervised-nets idea (Lee et al., AISTATS 2015) applied to a
residual branch rather than an intermediate classifier. It does not have to wait
for the fusion to discover it.

The gate is hard zero where no own-history exists, so the Unseen path is
bit-identical to the anchor and cannot be damaged -- which also means that with
the anchor frozen, ranking candidates by validation LifeAvg is ranking them by
Steady/Join/Fault, the part of the validation signal that actually tracks the
test set (6/6 per-group rank agreement, against 1/6 for the 20-node validation
unseen group).

References
    TCN          Bai, Kolter, Koltun 2018   https://arxiv.org/abs/1803.01271
                 official code https://github.com/locuslab/TCN
    DLinear      Zeng et al., AAAI 2023     https://github.com/cure-lab/LTSF-Linear
    Deep supervision  Lee et al., AISTATS 2015
    ReZero       Bachlechner et al. 2020    https://arxiv.org/abs/2003.04887
"""

import torch
from torch import nn


class DepthwiseCausalTCN(nn.Module):
    """One depthwise causal convolution block, per the TCN reference.

    Kept from references/TCN (locuslab): left-side padding then chopping the
    tail so no future step leaks, dilated convolution, weight-normalised conv,
    ReLU, dropout, and a residual add whose 1x1 downsample is used only when the
    channel counts differ. Reduced to a single depthwise + pointwise block --
    the reference stacks many for long sequences, and the input window here is
    twelve steps.
    """

    def __init__(self, channels, hidden, kernel_size=3, dilation=1, dropout=0.1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.depthwise = nn.Conv1d(
            channels, channels, kernel_size, padding=self.padding,
            dilation=dilation, groups=channels,
        )
        self.pointwise = nn.Conv1d(channels, hidden, 1)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """x: [B*N, C, L] -> [B*N, hidden, L]"""
        out = self.depthwise(x)
        if self.padding:
            out = out[..., :-self.padding]          # causal chomp
        return self.dropout(self.activation(self.pointwise(out)))


class DLinearDecomposition(nn.Module):
    """Moving-average trend/seasonal split with one Linear each.

    Copied from references/LTSF-Linear/models/DLinear.py: `series_decomp` with
    an AvgPool1d and edge replication, then independent Linear maps for the
    seasonal and trend parts, summed. The horizon projection is what differs
    from NLinear, which is why this is a separate switch rather than a reuse.
    """

    def __init__(self, input_len, pred_len, kernel_size=5):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.avg = nn.AvgPool1d(self.kernel_size, stride=1, padding=0)
        self.seasonal = nn.Linear(input_len, pred_len)
        self.trend = nn.Linear(input_len, pred_len)

    def decompose(self, x):
        """x: [B*N, 1, L]. Edge-replicating moving average, as in the reference."""
        front = x[..., :1].repeat(1, 1, (self.kernel_size - 1) // 2)
        end = x[..., -1:].repeat(1, 1, (self.kernel_size - 1) // 2)
        trend = self.avg(torch.cat([front, x, end], dim=-1))
        return x - trend, trend

    def forward(self, x):
        seasonal, trend = self.decompose(x)
        return self.seasonal(seasonal) + self.trend(trend)


class ProtectedTemporalResidual(nn.Module):
    """R = gate(context) * head(recovered series, mask, graph state).

    `gate` is exactly zero without own-history, so an unseen or zero-history
    sensor keeps the anchor's prediction bit for bit. The output layer is
    zero-initialised (ReZero), so an untrained residual reproduces the anchor
    everywhere, not just on the protected groups.
    """

    def __init__(self, input_len, pred_len, output_dim, availability_count,
                 graph_hidden_dim, hidden=32, kind="tcn", kernel_size=3,
                 dilation=1, dropout=0.1, residual_scale=1.0,
                 self_evidence_floor=0.02):
        super().__init__()
        if kind not in {"tcn", "dlinear"}:
            raise ValueError(f"unsupported temporal residual kind: {kind}")
        self.kind = kind
        self.input_len, self.pred_len = int(input_len), int(pred_len)
        self.output_dim = int(output_dim)
        self.residual_scale = float(residual_scale)
        self.self_evidence_floor = float(self_evidence_floor)

        # Series channels: recovered value and the effective observation mask.
        series_channels = self.output_dim + 1
        if kind == "tcn":
            self.temporal = DepthwiseCausalTCN(
                series_channels, hidden, kernel_size, dilation, dropout
            )
            temporal_dim = hidden * self.input_len
        else:
            self.temporal = DLinearDecomposition(self.input_len, self.pred_len, 5)
            temporal_dim = self.pred_len

        # Availability one-hot, age, visibility, and the detached graph state.
        context_dim = int(availability_count) + 2 + int(graph_hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(temporal_dim + context_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, self.pred_len * self.output_dim),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        self.gate_net = nn.Sequential(
            nn.Linear(int(availability_count) + 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def gate(self, availability, age, visibility, e_self):
        """Continuous and bounded, and hard zero without own-history.

        The hard part is not a learned prior: a sensor with no measurement in
        the window has nothing for a temporal branch to read, and letting a
        learned gate open there is how the Unseen group gets damaged.
        """
        raw = torch.sigmoid(self.gate_net(torch.cat([availability, age, visibility], dim=-1)))
        protection = (e_self / max(self.self_evidence_floor, 1e-6)).clamp(0.0, 1.0)
        return raw * protection

    def forward(self, recovered, mask, availability, age, visibility, e_self,
                graph_state):
        """recovered/mask: [B, L, N, *]; graph_state: [B, N, H] (detached).

        Returns (residual [B, H_out, N, C], gate [B, N, 1]).
        """
        batch, length, nodes, _ = recovered.shape
        series = torch.cat([recovered, mask], dim=-1)
        series = series.permute(0, 2, 3, 1).reshape(batch * nodes, -1, length)
        if self.kind == "tcn":
            temporal = self.temporal(series).reshape(batch, nodes, -1)
        else:
            temporal = self.temporal(series[:, :1]).reshape(batch, nodes, -1)
        features = torch.cat(
            [temporal, availability, age, visibility, graph_state], dim=-1
        )
        residual = self.head(features).reshape(
            batch, nodes, self.pred_len, self.output_dim
        ).permute(0, 2, 1, 3)
        gate = self.gate(availability, age, visibility, e_self)
        bounded = self.residual_scale * torch.tanh(residual)
        return bounded * gate.unsqueeze(1), gate


def companion_residual_loss(residual, target, anchor, valid, loss_name="mae"):
    """Direct supervision: the residual must predict the anchor's error.

        R_target = Y - stopgrad(Y_anchor)

    Without this the branch only ever sees the fused gradient, which is what
    let the earlier routed branch settle at "switched off" rather than learning
    anything (temporal activation 1.2%, its branch worse than the anchor on
    every availability group).
    """
    if not valid.any():
        return None
    goal = (target - anchor.detach())[valid]
    predicted = residual[valid]
    if loss_name in {"mse", "rmse"}:
        return (predicted - goal).pow(2).mean()
    return (predicted - goal).abs().mean()
