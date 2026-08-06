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


ROUTER_MODES = ("noisy_topk", "soft", "st_top1", "static")

# Softmax temperature for the straight-through gradient. Fixed at 1.0 rather
# than exposed as a knob: this round tests whether a differentiable router helps
# at all, and a temperature sweep would confound that with tuning.
STRAIGHT_THROUGH_TEMPERATURE = 1.0


class LifecycleRouter(nn.Module):
    """Route each node to propagation supports from its lifecycle state.

    Three modes, selected by ``mode``:

    ``noisy_topk``
        The historical path, copied from GMoE (``noisy_top_k_gating``,
        ``_gates_to_load``, ``_prob_in_top_k``); only the flattening of the node
        dimension is new. **At k=1 this router is not trainable by the forecast
        loss.** ``top_k_logits`` is then a single column, ``Softmax(1)`` of one
        element is identically 1.0 with derivative 0, and the top-k indices are
        not differentiable, so the only gradient reaching ``w_gate`` comes from
        the load-balance term -- a uniformity prior that knows nothing about
        prediction error. Measured on the shipped ``fg1-s32`` checkpoint: the
        injected noise is ~586x the learned logits, and training-time routing
        agrees with eval-time routing 20.0% of the time, which is exactly chance
        for five experts. Kept as the default so historical runs reproduce.

    ``soft``
        Softmax over all experts, no noise and no top-k. Sparsity buys nothing
        here -- ``LifecycleRoutedGraphConv`` evaluates all five supports before
        applying the gates either way (measured 2.165 ms at k=1 against 2.222 ms
        at k=5) -- so the sparse router costs differentiability for no saving.

    ``st_top1``
        Hard top-1 forward, so the output scale stays exactly 1.0 as the GRU
        gates expect, with the gradient taken through the soft distribution
        (straight-through estimator; Bengio et al. 2013, and the hard variant of
        Jang et al. ICLR 2017). Deterministic -- see
        ``_straight_through_top1_gating`` for why the Gumbel sample is left out.

    ``static``
        One learned convex combination shared by every node, with no lifecycle
        input at all: five logits per convolution instead of a 6x5 projection.
        This is the control arm for "does the *dynamic* part earn its keep?".
        Measured on the trained `gs` router, conditioning changes the forecast
        by 0.74% on average (max 6.6e-02 against a mean prediction magnitude of
        0.295) and moves a gate by at most 0.19 away from 1/5 -- real, but the
        same order as the differences between candidate models. That is a
        question for a controlled run, not for a judgement call, and this mode
        is what makes the run controlled. If it matches `soft`, the honest
        description of the mechanism is a heterogeneous operator mixture rather
        than lifecycle-aware routing.

    The auxiliary load-balance formula is deliberately unchanged across modes:
    under ``soft`` every gate is positive, so ``_gates_to_load`` is constant and
    ``cv_squared`` of it is zero, leaving the importance term alone. That keeps
    the objective identical rather than introducing a second difference between
    candidates.
    """

    def __init__(self, state_dim, num_experts, k=2, noisy_gating=True, loss_coef=1e-2,
                 mode="noisy_topk"):
        super().__init__()
        if not 1 <= int(k) <= int(num_experts):
            raise ValueError("router top-k must satisfy 1 <= k <= num_experts")
        if str(mode) not in ROUTER_MODES:
            raise ValueError(f"unsupported router mode {mode!r}; expected one of {ROUTER_MODES}")
        self.mode = str(mode)
        self.num_experts = int(num_experts)
        self.k = int(k)
        self.noisy_gating = bool(noisy_gating)
        self.loss_coef = float(loss_coef)
        # Allocated per mode, not unconditionally: giving every mode every
        # tensor would change the parameter count of the existing variants and
        # break their checkpoints, and would leave dead weights in the graph.
        # `static` genuinely is the smaller model -- five logits instead of two
        # 6x5 projections -- and the count should say so.
        if self.mode == "static":
            # Zero init, so it starts at exactly 1/E: the same place `soft`
            # starts, leaving conditioning as the only difference between them.
            self.w_static = nn.Parameter(torch.zeros(self.num_experts))
        else:
            self.w_gate = nn.Parameter(torch.zeros(int(state_dim), self.num_experts))
            self.w_noise = nn.Parameter(torch.zeros(int(state_dim), self.num_experts))
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        # Diagnostics are off by default: the graph core runs 24 recurrent steps
        # per forward, and accumulating a tensor per step per router costs more
        # than the statistics are worth outside an evaluation pass.
        self._collect = False
        self._usage_total = torch.zeros(self.num_experts)
        self._entropy_total = 0.0
        self._rows = 0

    def _routing_weight(self):
        """Whichever tensor this mode routes with; `static` has no `w_gate`."""
        return self.w_static if self.mode == "static" else self.w_gate

    def _reset_diagnostics(self):
        reference = self._routing_weight()
        self._usage_total = torch.zeros(
            self.num_experts, device=reference.device, dtype=reference.dtype
        )
        self._entropy_total = 0.0
        self._rows = 0

    def set_collect_diagnostics(self, enabled):
        self._collect = bool(enabled)
        self._reset_diagnostics()

    def pop_diagnostics(self):
        """Expert usage plus the per-node and population entropies.

        Every quantity is a true total divided by the routed-node count, so a
        short final batch contributes exactly its own rows and no more.
        """
        if not self._rows:
            return None
        rows = int(self._rows)
        usage = self._usage_total / rows
        gate_entropy = self._entropy_total / rows
        share = usage / usage.sum().clamp_min(1e-12)
        usage_entropy = -(share.clamp_min(1e-12).log() * share).sum()
        uniform = math.log(self.num_experts)
        self._reset_diagnostics()
        return {
            "routed_nodes": rows,
            "usage": usage.tolist(),
            # per-node: 1.0 = every node mixes uniformly. Always 0 for a hard
            # router, so do not read it as decisiveness there.
            "entropy_ratio": float(gate_entropy / uniform),
            # population: 1.0 = the five operators are used equally often, which
            # for `soft` means a uniform ensemble rather than routing.
            "usage_entropy_ratio": float(usage_entropy / uniform),
        }

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

    def _soft_gating(self, x):
        """Dense softmax over every expert. Fully differentiable by construction."""
        gates = self.softmax(x @ self.w_gate)
        return gates, self._gates_to_load(gates)

    def _static_gating(self, x):
        """One learned mixture for every node; the lifecycle state is ignored.

        `x` is accepted and discarded on purpose: the call site is shared, and
        taking the argument makes it obvious at a glance that this mode *could*
        read the state and deliberately does not.
        """
        gates = torch.softmax(self.w_static, dim=0).expand(x.shape[0], -1)
        return gates, self._gates_to_load(gates)

    def _straight_through_top1_gating(self, x, train):
        """Hard one-hot forward, soft gradient backward. Deterministic.

        `hard - soft.detach() + soft` is numerically `hard`, so the forward pass
        selects exactly one support at weight 1.0 and the routed output keeps
        DiffConv's scale. The backward pass sees `d(soft)`, which is what makes
        `w_gate` reachable from the forecast loss at all.

        No Gumbel noise, deliberately. The defect this mode exists to fix is a
        train/eval mismatch -- with `noisy_topk` the injected noise was ~586x the
        learned logits and the two passes agreed at chance. Sampling the routing
        during training reproduces exactly that failure in milder form: measured
        here, ST-Gumbel at temperature 1.0 agreed with its own eval-time routing
        33% of the time, against 100% for the deterministic form. Five experts
        and a six-dimensional router do not present the exploration problem that
        motivates Gumbel in large sparse MoE, and the load-balance term already
        supplies pressure against collapse -- now alongside a real forecast
        gradient rather than instead of one.
        """
        soft = self.softmax((x @ self.w_gate) / STRAIGHT_THROUGH_TEMPERATURE)
        hard = torch.zeros_like(soft).scatter_(
            1, soft.argmax(dim=1, keepdim=True), 1.0
        )
        gates = hard - soft.detach() + soft
        return gates, self._gates_to_load(gates)

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
        if self.mode == "soft":
            gates, load = self._soft_gating(flat)
        elif self.mode == "static":
            gates, load = self._static_gating(flat)
        elif self.mode == "st_top1":
            gates, load = self._straight_through_top1_gating(flat, self.training)
        else:
            gates, load = self.noisy_top_k_gating(flat, self.training)
        importance = gates.sum(0)
        aux_loss = self.loss_coef * (cv_squared(importance) + cv_squared(load))
        self._record(gates)
        return gates.reshape(batch, nodes, self.num_experts), aux_loss.squeeze()

    def _record(self, gates):
        """Usage mass and two entropies, because one of them is degenerate.

        `gate_entropy` is the per-node entropy of the gate vector. It separates
        "this node mixes every operator" from "this node commits to one" -- but
        for a hard router the gate is one-hot by construction, so it is
        identically zero and says nothing. Reading only that number would have
        made `st_top1` look maximally decisive and `soft` look useless.

        `usage_entropy` is the entropy of the *population* usage, i.e. how the
        mass is spread across experts once averaged over nodes. That is the one
        that distinguishes real specialisation from a uniform ensemble, and it
        is meaningful for both hard and soft routers.
        """
        if not self._collect:
            return
        with torch.no_grad():
            mass = gates.detach()
            probs = mass / mass.sum(dim=1, keepdim=True).clamp_min(1e-12)
            # Accumulate totals and a row count rather than per-call means: the
            # last batch of an epoch is short, and averaging means-of-means
            # weights its rows more heavily than the rest.
            self._usage_total = self._usage_total + mass.sum(dim=0)
            self._entropy_total = self._entropy_total + float(
                -(probs.clamp_min(1e-12).log() * probs).sum()
            )
            self._rows += int(mass.shape[0])


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
        router_mode="noisy_topk",
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
            mode=router_mode,
        )
        self._aux_terms = []
        self.reset_parameters()

    @property
    def router_mode(self):
        return self.router.mode

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.out_channels)
        self.Theta.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def reset_router_stats(self):
        self._aux_terms = []

    def set_collect_diagnostics(self, enabled):
        self.router.set_collect_diagnostics(enabled)

    def pop_router_diagnostics(self):
        return self.router.pop_diagnostics()

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
