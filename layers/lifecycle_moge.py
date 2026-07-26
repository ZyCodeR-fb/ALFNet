"""Lifecycle-routed heterogeneous graph aggregation for ALFGraph.

This replaces the single fixed diffusion operator inside the DCRNN cell with a
sparsely routed mixture of heterogeneous propagation supports. The router reads
the node's lifecycle state (availability one-hot, state_age, observation mask)
rather than its raw series: after the shared GRIN imputation a Cold node and a
Steady node carry similar-looking values, so the raw series can no longer tell
them apart, while the lifecycle state still can. Every routing input is known at
prediction time.

Provenance
----------
Sparse noisy top-k gating and the load-balancing loss are copied from GMoE
(Wang et al., NeurIPS 2023), ``references/GMoE/graphproppred/moe.py``, which in
turn adapts https://github.com/davidmrau/mixture-of-experts (Shazeer et al.,
2017). Adapted only to (a) accept ``[batch, node, feature]`` tensors and (b)
route on lifecycle state.

The aggregator family follows MoGE, ``references/MoGERNN/layers/agg.py``. Two
deliberate deviations from that file:

* MoGE's max/min pooling aggregators materialise a ``[batch, channels, node,
  node]`` tensor. That is affordable once per window over ``input_len`` features
  but not inside a recurrent cell over ``hidden_size`` features at every step,
  so only the matmul-form aggregators are kept here.
* MoGE derives its mean and weighted-mean aggregators from the same weighted
  adjacency, differing only in whether the row denominator counts neighbours or
  sums weights. Expressed as propagation supports those two differ by a per-row
  scalar alone, which defeats the purpose of having both. Here `mean` uses the
  binarised adjacency and `weighted_mean` the weighted one, so the two experts
  are genuinely different operators.
"""

import math

import torch
from torch import nn


# Availability one-hot (4) + state_age (1) + observation mask (1).
LIFECYCLE_STATE_CHANNELS = 6
SQRT_HALF = 0.7071067811865476
SUPPORT_NAMES = ("identity", "mean", "weighted_mean", "forward_diffusion", "backward_diffusion")


def cv_squared(x):
    """The squared coefficient of variation of a sample.

    Copied from ``references/GMoE/graphproppred/moe.py``.
    """
    eps = 1e-10
    if x.shape[0] == 1:
        return torch.tensor([0], device=x.device, dtype=x.dtype)
    return x.float().var() / (x.float().mean() ** 2 + eps)


def build_lifecycle_supports(adjacency, forward_support, backward_support):
    """Stack the five propagation supports into one ``[num_experts, node, node]``.

    Stacking lets the whole mixture run as two batched einsums per call. The
    graph core is a 24-step recursion over 12 convolutions, so collapsing 5
    separate kernel launches per call into 1 matters more than the identity
    expert's redundant multiply by ``eye``.
    """
    weights = torch.nan_to_num(adjacency.float(), nan=0.0)
    binary = (weights > 0).to(weights.dtype)
    identity = torch.eye(weights.shape[0], device=weights.device, dtype=weights.dtype)
    mean_support = binary / binary.sum(dim=1, keepdim=True).clamp_min(1.0)
    weighted_support = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
    return torch.stack(
        [
            identity,
            mean_support,
            weighted_support,
            forward_support.to(weights.dtype),
            backward_support.to(weights.dtype),
        ],
        dim=0,
    )


class LifecycleRouter(nn.Module):
    """Noisy top-k gating over lifecycle state.

    ``noisy_top_k_gating``, ``_gates_to_load`` and ``_prob_in_top_k`` are copied
    from GMoE; only the flattening of the node dimension is new.
    """

    def __init__(self, state_dim, num_experts, k=2, noisy_gating=True, loss_coef=1e-2):
        super().__init__()
        if not 1 <= int(k) <= int(num_experts):
            raise ValueError("router top-k must satisfy 1 <= k <= num_experts")
        self.num_experts = int(num_experts)
        self.k = int(k)
        self.noisy_gating = bool(noisy_gating)
        self.loss_coef = float(loss_coef)
        self.w_gate = nn.Parameter(torch.zeros(int(state_dim), self.num_experts))
        self.w_noise = nn.Parameter(torch.zeros(int(state_dim), self.num_experts))
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))

    def _gates_to_load(self, gates):
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        prob_if_in = self._normal_cdf((clean_values - threshold_if_in) / noise_stddev)
        prob_if_out = self._normal_cdf((clean_values - threshold_if_out) / noise_stddev)
        return torch.where(is_in, prob_if_in, prob_if_out)

    def _normal_cdf(self, value):
        """Closed-form standard-normal CDF.

        ``Normal(mean, std).cdf(x)`` evaluates exactly this, but builds a
        distribution object with argument validation and broadcasting on every
        call. Inside a 24-step recurrence that dispatch cost dominates: measured
        0.273 ms against 0.075 ms for the formula, with bit-identical output.
        """
        return 0.5 * (1.0 + torch.erf((value - self.mean) * SQRT_HALF / self.std))

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        clean_logits = x @ self.w_gate
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = self.softplus(raw_noise_stddev) + noise_epsilon
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)
        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        top_k_gates = self.softmax(top_k_logits)

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        if self.noisy_gating and self.k < self.num_experts and train:
            load = self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load

    def forward(self, state):
        """state: [batch, node, state_dim] -> gates [batch, node, num_experts]."""
        batch, nodes, _ = state.shape
        flat = state.reshape(batch * nodes, -1)
        gates, load = self.noisy_top_k_gating(flat, self.training)
        importance = gates.sum(0)
        aux_loss = self.loss_coef * (cv_squared(importance) + cv_squared(load))
        return gates.reshape(batch, nodes, self.num_experts), aux_loss.squeeze()


class LifecycleRoutedGraphConv(nn.Module):
    """Drop-in replacement for ``DiffConv`` with per-node per-step expert routing.

    ``DiffConv`` concatenates ``[x, A_q x, A_h x]`` and mixes them with a single
    weight matrix shared by every node and every lifecycle state. Here each
    support owns a weight matrix and the router picks ``k`` of them per node.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        num_experts=len(SUPPORT_NAMES),
        state_dim=LIFECYCLE_STATE_CHANNELS,
        k=2,
        noisy_gating=True,
        loss_coef=1e-2,
    ):
        super().__init__()
        if int(num_experts) != len(SUPPORT_NAMES):
            raise ValueError(
                f"lifecycle routing is defined over {len(SUPPORT_NAMES)} supports: {SUPPORT_NAMES}"
            )
        self.num_experts = int(num_experts)
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.Theta = nn.Parameter(
            torch.empty(self.num_experts, self.in_channels, self.out_channels)
        )
        self.bias = nn.Parameter(torch.empty(self.out_channels))
        self.router = LifecycleRouter(
            state_dim,
            self.num_experts,
            k=k,
            noisy_gating=noisy_gating,
            loss_coef=loss_coef,
        )
        self._aux_terms = []
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.out_channels)
        self.Theta.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def reset_router_stats(self):
        self._aux_terms = []

    def pop_router_loss(self):
        if not self._aux_terms:
            return None
        total = torch.stack(self._aux_terms).mean()
        self._aux_terms = []
        return total

    def forward(self, x, A_q, A_h, supports=None, state=None, **kwargs):
        if supports is None or state is None:
            raise ValueError(
                "LifecycleRoutedGraphConv needs propagation supports and lifecycle state"
            )
        if supports.shape[0] != self.num_experts:
            raise ValueError(
                f"expected {self.num_experts} supports, got {supports.shape[0]}"
            )
        gates, aux_loss = self.router(state)
        self._aux_terms.append(aux_loss)

        # einsum beats a broadcast matmul here: the broadcast form materialises
        # [batch, experts, node, node] intermediates, and measured 58% slower for
        # the propagation and 47% slower for the contraction.
        propagated = torch.einsum("enm,bmd->bend", supports, x)
        expert_outputs = torch.einsum("bend,edo->beno", propagated, self.Theta)
        # GMoE averages over the expert axis; summing instead keeps the routed
        # output on the same scale as DiffConv, which matters because this feeds
        # the GRU's sigmoid/tanh gates directly. The explicit broadcast is 20%
        # faster than the equivalent einsum for this small contraction.
        return (expert_outputs * gates.transpose(1, 2).unsqueeze(-1)).sum(dim=1) + self.bias
