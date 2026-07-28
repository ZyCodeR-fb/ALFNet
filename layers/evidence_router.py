"""Evidence-aware graph/temporal collaboration (C) and observed NLinear (D).

The design follows a measured result rather than an assumption. Isolating the
temporal path (A2 minus D1, same protocol and seed) on the frozen test set:

  Join track, MAE   k=1 -0.042 (n.s.)   k=2 -0.221*   k=12 -0.272*
  Fault track, MAE  k=1 -0.698*         k=6 -1.201*   k=12 -2.350*

so the temporal path is useless when the node has no history of its own (Cold),
but its advantage *grows* as the node's own history degrades along the fault
track -- the opposite of "less evidence, trust the graph". Splitting Outage by
how compact the failed cohort is explains why:

  clustered cohorts (1.1-2.4 graph hops)  temporal benefit -3.165
  scattered cohorts (4.2-7.0 graph hops)  temporal benefit -1.619

When neighbours fail together, graph propagation pulls from neighbours that are
themselves dark and collapses (graph-only MAE 10.19 at 1.10 hops), while the
patch path still has spatial context and a learned periodic profile.

Two independent quantities therefore drive the split, and neither substitutes
for the other:

  e_self  own observable history  -> whether the *self* temporal path can say
                                     anything at all
  e_nbr   neighbourhood observability -> whether the *graph* path is reliable

Unseen and Outage both have e_self = 0, so e_self cannot separate them. What
does is the availability context and state_age already present in the input
channels: a never-observed sensor and a sensor that went dark after a long
observed run are different rows there. That is why only `Y_self` is gated by
e_self; `Y_patch` is not, and the router -- not a hand-written prior -- decides
how much of it to use.
"""

import torch
from torch import nn
import torch.nn.functional as F


class ObservedNLinear(nn.Module):
    """D: NLinear's de-trending projection, adapted into a bounded residual.

    What is kept from references/LTSF-Linear/models/NLinear.py: the last
    observed value is subtracted, a single Linear maps the de-trended sequence
    axis to the horizon. What is *not* kept: the reference then adds the anchor
    back and returns a complete forecast. Returning that here made an untrained
    head emit naive persistence, and blending 35% of it into the graph forecast
    moved the prediction by up to 0.93 before any training -- the `d` run never
    recovered from it. So this returns the projection and the anchor separately
    and the caller uses the projection as a bounded residual, which is zero at
    initialisation. Calling it "the original NLinear output" would be wrong.

    Two further adaptations for lifecycle data: "last value" means last
    *effective* value, and unobserved steps are held at that value rather than
    feeding zeros into the projection. The input is the raw measurement window,
    never GRIN's imputed one -- the point is a residual grounded in
    measurements that actually happened.

    Paper: Are Transformers Effective for Time Series Forecasting? (AAAI 2023).
    """

    def __init__(self, input_len, pred_len, output_dim=1):
        super().__init__()
        self.input_len, self.pred_len, self.output_dim = int(input_len), int(pred_len), int(output_dim)
        self.projection = nn.Linear(self.input_len, self.pred_len)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, observed, mask):
        """observed/mask: [B, L, N, 1]. Returns (residual, persistence anchor).

        `residual` is exactly zero for an untrained head, so enabling D cannot
        move the prediction before it has learned anything.
        """
        observed = torch.nan_to_num(observed, nan=0.0, posinf=0.0, neginf=0.0)
        valid = mask > 0.5
        # index of the last observed step per node, 0 when never observed
        steps = torch.arange(self.input_len, device=observed.device).view(1, -1, 1, 1)
        last_index = (steps * valid).amax(dim=1, keepdim=True)
        last_value = torch.gather(observed, 1, last_index.expand(-1, 1, -1, -1))
        last_value = last_value * valid.any(dim=1, keepdim=True)
        series = torch.where(valid, observed, last_value.expand_as(observed)) - last_value
        projected = self.projection(series.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        # The reference returns `projected + seq_last`, i.e. a full forecast whose
        # untrained value is persistence. Blending 35% of persistence into the
        # graph forecast from epoch 0 is a large perturbation, not a residual --
        # measured as mean |delta| 0.081 / max 0.931 against the graph-only
        # prediction before any training, and it cost the d run its Unseen
        # accuracy. Returning the anchor separately lets the caller add a
        # residual that really is zero at initialisation.
        return projected, last_value


def self_evidence(mask, half_life=4.0):
    """Recency-weighted share of the node's own observed steps. [B, N, 1]"""
    length = mask.shape[1]
    steps = torch.arange(length, device=mask.device, dtype=mask.dtype)
    weight = torch.pow(0.5, (length - 1 - steps) / float(half_life)).view(1, -1, 1, 1)
    return (mask * weight).sum(dim=1) / weight.sum()


def neighbour_evidence(adjacency, mask):
    """Share of observed graph neighbours at the last input step. [B, N, 1]

    Low means the graph path is drawing on neighbours that are themselves dark,
    which is the clustered-outage case where graph propagation collapses.
    """
    weights = torch.nan_to_num(adjacency.abs(), nan=0.0)
    weights = weights - torch.diag_embed(torch.diagonal(weights))
    row_sum = weights.sum(dim=-1, keepdim=True).clamp(min=1e-6)
    normalized = weights / row_sum
    return torch.einsum("nm,bmc->bnc", normalized, mask[:, -1])


def mask_shape(mask):
    """Prefix/suffix missing runs and the plain observed fraction. [B, N, 3]"""
    valid = (mask > 0.5).float()
    length = valid.shape[1]
    # leading run of missing steps: how long before the sensor first appears
    prefix = (torch.cumprod(1.0 - valid, dim=1).sum(dim=1)) / length
    # trailing run of missing steps: how long it has been dark at window end
    suffix = (torch.cumprod(1.0 - valid.flip(1), dim=1).sum(dim=1)) / length
    return torch.cat([prefix, suffix, valid.mean(dim=1)], dim=-1)


class EvidenceRouter(nn.Module):
    """C: node-level beta over an explicit graph anchor.

        Yhat = Yg + beta * (Yt - Yg)

    The graph output is the anchor, so beta = 0 recovers the graph path exactly
    and an untrained router is safe. beta is produced by a learned sigmoid, not
    by a hand-written prior: the one prior I tried to write by hand would have
    forced beta -> 0 at Outage, which the measurements above show is precisely
    backwards. The router is instead *given* e_nbr so it can learn that a dark
    neighbourhood means the graph is the unreliable one.

    Routing supervision reuses the existing detached expert-error target
    (`_route_loss`, hard_bce), the TESTAM idea of labelling which expert was
    right per node. No test labels and no phase labels enter the model.
    """

    def __init__(self, output_dim, pred_len, availability_count, hidden_dim=32,
                 self_evidence_floor=0.02, initial_beta_logit=-2.0,
                 router_weight_std=1e-2):
        super().__init__()
        self.output_dim = int(output_dim)
        self.self_evidence_floor = float(self_evidence_floor)
        # availability one-hot + state_age + visibility + e_self + e_nbr
        # + prefix/suffix/observed-fraction + Yg + Yt + |Yg - Yt|
        context_dim = int(availability_count) + 2 + 2 + 3
        series_dim = 3 * int(pred_len) * self.output_dim
        self.router = nn.Sequential(
            nn.Linear(context_dim + series_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        # Zero-initialising both weight and bias puts beta at exactly sigmoid(0)
        # = 0.5 for every node -- a symmetric fixed point, and the c run sat on
        # it (beta_std 0.0017 after 11 epochs, route accuracy 0.488, below
        # chance). The bias starts negative instead: beta ~ 0.12, so the model
        # begins graph-dominant and safe, but not at a symmetry point.
        # A zero weight makes beta a constant regardless of the inputs -- the
        # bias alone decides it, so beta_std is exactly 0 and the router is not
        # a router. A small weight keeps the starting point near the bias while
        # letting the evidence features actually move beta per node.
        nn.init.normal_(self.router[-1].weight, std=float(router_weight_std))
        nn.init.constant_(self.router[-1].bias, float(initial_beta_logit))

    def self_gate(self, e_self):
        """Gates `Y_self` only. `Y_patch` is deliberately not gated by this."""
        return (e_self / max(self.self_evidence_floor, 1e-6)).clamp(0.0, 1.0)

    def forward(self, y_graph, y_temporal, context):
        """y_*: [B, H, N, C]; context: dict of [B, N, *] evidence features."""
        batch, horizon, nodes, _ = y_graph.shape
        flat_g = y_graph.permute(0, 2, 1, 3).reshape(batch, nodes, -1)
        flat_t = y_temporal.permute(0, 2, 1, 3).reshape(batch, nodes, -1)
        features = torch.cat(
            [
                context["availability"], context["state_age"], context["visibility"],
                context["e_self"], context["e_nbr"], context["mask_shape"],
                flat_g, flat_t, (flat_g - flat_t).abs(),
            ],
            dim=-1,
        )
        beta = torch.sigmoid(self.router(features)).unsqueeze(1)   # [B, 1, N, 1]
        return y_graph + beta * (y_temporal - y_graph), beta


def routing_diagnostics(beta, y_graph, y_temporal, target, valid, availability):
    """Per-availability-group beta, branch errors and route accuracy.

    Reported so router collapse is visible rather than inferred: a router that
    always picks one branch shows up as a flat mean beta across groups and a
    route accuracy stuck at the base rate.
    """
    with torch.no_grad():
        graph_error = (y_graph - target).abs()
        temporal_error = (y_temporal - target).abs()
        better = (temporal_error < graph_error).float()
        expanded = beta.expand_as(graph_error)
        if not valid.any():
            return {}
        chosen = (expanded > 0.5).float()
        # Raw accuracy is only interpretable against the base rate: if the
        # temporal branch is better on 80% of cells, always picking it scores
        # 0.80 and always picking the graph scores 0.20, and neither is
        # informative. Balanced accuracy is the number to compare with 0.5.
        base_rate = float(better[valid].mean())
        positive = valid & (better > 0.5)
        negative = valid & (better <= 0.5)
        recall_temporal = float(chosen[positive].mean()) if positive.any() else float("nan")
        recall_graph = float((1.0 - chosen)[negative].mean()) if negative.any() else float("nan")
        balanced = [value for value in (recall_temporal, recall_graph) if value == value]
        report = {
            "beta_mean": float(expanded[valid].mean()),
            "beta_std": float(expanded[valid].std()),
            "temporal_activation_ratio": float(chosen[valid].mean()),
            "route_accuracy": float((chosen == better)[valid].float().mean()),
            "route_target_base_rate": base_rate,
            "route_balanced_accuracy": sum(balanced) / len(balanced) if balanced else float("nan"),
        }
        for index in range(availability.shape[-1]):
            role = availability[..., index].unsqueeze(1).unsqueeze(-1) > 0.5
            selected = valid & role.expand_as(valid)
            if selected.any():
                report[f"beta_role{index}"] = float(expanded[selected].mean())
                report[f"graph_mae_role{index}"] = float(graph_error[selected].mean())
                report[f"temporal_mae_role{index}"] = float(temporal_error[selected].mean())
        return report
