"""The router must be reachable from the forecast loss.

The default `noisy_topk` path at k=1 is not, and that is not a subtle claim: a
single-column `Softmax(1)` is identically 1.0 with derivative 0, and the top-k
indices carry no gradient. Only the load-balance term reaches `w_gate`, and that
term is a uniformity prior with no knowledge of prediction error -- measured on
the shipped `fg1-s32` checkpoint, training-time routing agreed with eval-time
routing 20.0% of the time, exactly chance for five experts.

Every test here isolates the *forecast* gradient: the auxiliary loss is drained
and discarded before `backward`, so a candidate cannot pass by leaning on it.
"""

import math

import pytest
import torch

from layers.lifecycle_moge import (
    ROUTER_MODES,
    SUPPORT_NAMES,
    LifecycleRoutedGraphConv,
    LifecycleRouter,
    build_lifecycle_supports,
)


NODES, BATCH, CIN, COUT = 16, 3, 8, 12
DIFFERENTIABLE_MODES = ["soft", "st_top1"]


def _conv(mode, k=1, seed=0):
    torch.manual_seed(seed)
    return LifecycleRoutedGraphConv(
        CIN, COUT, num_experts=len(SUPPORT_NAMES), k=k, router_mode=mode
    )


def _inputs(seed=7):
    generator = torch.Generator().manual_seed(seed)
    adjacency = (torch.rand(NODES, NODES, generator=generator) < 0.25).float()
    adjacency = adjacency * torch.rand(NODES, NODES, generator=generator)
    forward = adjacency / adjacency.sum(1, keepdim=True).clamp_min(1e-6)
    backward = adjacency.T / adjacency.T.sum(1, keepdim=True).clamp_min(1e-6)
    supports = build_lifecycle_supports(adjacency, forward, backward)
    x = torch.randn(BATCH, NODES, CIN, generator=generator)
    state = torch.zeros(BATCH, NODES, 6)
    state.scatter_(2, torch.randint(0, 4, (BATCH, NODES, 1), generator=generator), 1.0)
    state[..., 4] = torch.rand(BATCH, NODES, generator=generator)
    state[..., 5] = (torch.rand(BATCH, NODES, generator=generator) > 0.3).float()
    return supports, x, state


def _forecast_only_gate_grad(conv, supports, x, state):
    """Gradient of a prediction-shaped loss w.r.t. w_gate, auxiliary term removed."""
    conv.train()
    conv.zero_grad(set_to_none=True)
    conv.reset_router_stats()
    output = conv(x, None, None, supports=supports, state=state)
    conv.reset_router_stats()          # drop the auxiliary term before backward
    output.pow(2).mean().backward()
    assert conv.router.w_gate.grad is not None
    return float(conv.router.w_gate.grad.abs().max())


# --- the defect this round exists to fix -------------------------------------


def test_default_topk1_router_is_unreachable_from_the_forecast_loss():
    """Pin the defect so a future edit cannot quietly reintroduce it elsewhere."""
    supports, x, state = _inputs()
    assert _forecast_only_gate_grad(_conv("noisy_topk", k=1), supports, x, state) == 0.0


def test_topk_above_one_is_reachable():
    """k>=2 works, which is why this is specifically a top-1 defect."""
    supports, x, state = _inputs()
    assert _forecast_only_gate_grad(_conv("noisy_topk", k=2), supports, x, state) > 0.0


def test_single_element_softmax_is_the_mechanism():
    """The root cause, isolated: softmax over one element has zero derivative."""
    logits = torch.zeros(4, 1, requires_grad=True)
    torch.nn.Softmax(1)(logits).sum().backward()
    assert float(logits.grad.abs().max()) == 0.0


# --- the two candidates ------------------------------------------------------


@pytest.mark.parametrize("mode", DIFFERENTIABLE_MODES)
def test_candidate_router_receives_forecast_gradient(mode):
    supports, x, state = _inputs()
    grad = _forecast_only_gate_grad(_conv(mode, k=1), supports, x, state)
    assert grad > 0.0, f"{mode} router is still unreachable from the forecast loss"


@pytest.mark.parametrize("mode", DIFFERENTIABLE_MODES)
def test_candidate_gradient_is_not_the_auxiliary_loss_in_disguise(mode):
    """Belt and braces: build the graph, throw the auxiliary term away entirely.

    `reset_router_stats` clears the accumulator, so `pop_router_loss` returns
    None and no auxiliary tensor can participate in the backward pass.
    """
    supports, x, state = _inputs()
    conv = _conv(mode, k=1)
    conv.train()
    conv.zero_grad(set_to_none=True)
    conv.reset_router_stats()
    output = conv(x, None, None, supports=supports, state=state)
    conv.reset_router_stats()
    assert conv.pop_router_loss() is None, "an auxiliary term survived the reset"
    output.pow(2).mean().backward()
    assert float(conv.router.w_gate.grad.abs().max()) > 0.0


def test_straight_through_keeps_a_hard_one_hot_forward():
    """Scale matters: the routed output feeds the GRU's sigmoid/tanh directly."""
    torch.manual_seed(0)
    router = LifecycleRouter(6, len(SUPPORT_NAMES), k=1, mode="st_top1")
    router.train()
    _, _, state = _inputs()
    gates, _ = router(state)
    assert torch.allclose(gates.sum(dim=-1), torch.ones_like(gates.sum(dim=-1)), atol=1e-5)
    nonzero = (gates > 1e-6).sum(dim=-1)
    assert torch.all(nonzero == 1), "straight-through top-1 selected more than one support"
    assert torch.allclose(gates.max(dim=-1).values, torch.ones(gates.shape[:-1]), atol=1e-5)


def test_soft_router_mixes_every_expert_and_sums_to_one():
    torch.manual_seed(0)
    router = LifecycleRouter(6, len(SUPPORT_NAMES), k=1, mode="soft")
    router.train()
    _, _, state = _inputs()
    gates, _ = router(state)
    assert torch.allclose(gates.sum(dim=-1), torch.ones_like(gates.sum(dim=-1)), atol=1e-5)
    assert torch.all(gates > 0), "a dense router must give every expert some mass"


@pytest.mark.parametrize("mode", DIFFERENTIABLE_MODES)
def test_candidate_train_and_eval_routing_agree_far_above_chance(mode):
    """The failure that made this round necessary was a train/eval mismatch.

    With `noisy_topk` the injected noise dominated the learned logits (~586x on
    the shipped checkpoint), so the routing used in training and the routing
    used at inference agreed at chance. Both candidates are deterministic, so
    they must agree exactly; an ST-Gumbel variant was measured at 33% and
    rejected for reproducing the very defect being fixed.
    """
    torch.manual_seed(0)
    router = LifecycleRouter(6, len(SUPPORT_NAMES), k=1, mode=mode)
    # a router that has actually learned something, rather than the zero init
    with torch.no_grad():
        router.w_gate.normal_(0.0, 1.0)
    _, _, state = _inputs()

    router.eval()
    reference = router(state)[0].argmax(dim=-1)
    router.train()
    agreements = [
        float((router(state)[0].argmax(dim=-1) == reference).float().mean())
        for _ in range(10)
    ]
    mean_agreement = sum(agreements) / len(agreements)
    chance = 1.0 / len(SUPPORT_NAMES)
    assert mean_agreement == 1.0, (
        f"{mode}: train/eval routing agreed {mean_agreement:.1%}, chance is {chance:.1%}; "
        "a deterministic router must ship the model it trained"
    )


# --- diagnostics -------------------------------------------------------------


@pytest.mark.parametrize("mode", ROUTER_MODES)
def test_diagnostics_report_usage_and_entropy(mode):
    supports, x, state = _inputs()
    conv = _conv(mode, k=1)
    conv.eval()
    conv.set_collect_diagnostics(True)
    with torch.no_grad():
        conv(x, None, None, supports=supports, state=state)
    report = conv.pop_router_diagnostics()
    assert report is not None
    assert len(report["usage"]) == len(SUPPORT_NAMES)
    assert report["usage"] == pytest.approx(report["usage"])
    assert sum(report["usage"]) == pytest.approx(1.0, abs=1e-4)
    # entropy_ratio is normalised by log(num_experts), so 1.0 means "uninformative"
    assert 0.0 <= report["entropy_ratio"] <= 1.0 + 1e-6
    assert report["entropy"] == pytest.approx(
        report["entropy_ratio"] * math.log(len(SUPPORT_NAMES)), rel=1e-5
    )
    assert conv.pop_router_diagnostics() is None, "diagnostics must drain on read"


def test_diagnostics_are_off_by_default():
    """The graph core runs 24 recurrent steps per forward; this is not free."""
    supports, x, state = _inputs()
    conv = _conv("soft", k=1)
    conv.eval()
    with torch.no_grad():
        conv(x, None, None, supports=supports, state=state)
    assert conv.pop_router_diagnostics() is None


def test_unknown_router_mode_is_rejected():
    with pytest.raises(ValueError, match="unsupported router mode"):
        LifecycleRouter(6, len(SUPPORT_NAMES), k=1, mode="argmax")


# --- the candidates' ablation boundary ---------------------------------------


import yaml                                                        # noqa: E402
from pathlib import Path                                           # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = {"gs": "soft", "g1": "st_top1"}


def _flat(node, prefix=""):
    out = {}
    for key, value in (node or {}).items():
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(_flat(value, path + "."))
        else:
            out[path] = value
    return out


def _yaml(relative):
    return yaml.safe_load((ROOT / relative).read_text(encoding="utf-8"))


@pytest.mark.parametrize("variant,mode", sorted(CANDIDATES.items()))
def test_candidate_yaml_differs_from_fg1_only_through_the_graph_expert(variant, mode):
    base = _flat(_yaml("configs/ALFNet/fg1.yaml"))
    other = _flat(_yaml(f"configs/ALFNet/{variant}.yaml"))
    drift = {key for key in set(base) | set(other) if base.get(key) != other.get(key)}
    assert drift == {"runtime.model_variant", "family.experts.graph.config_path"}, sorted(drift)


@pytest.mark.parametrize("variant,mode", sorted(CANDIDATES.items()))
def test_candidate_graph_config_differs_from_k1_only_in_the_router_mode(variant, mode):
    """The whole point of the round: one key, nothing else."""
    graph_path = _yaml(f"configs/ALFNet/{variant}.yaml")["family"]["experts"]["graph"]["config_path"]
    base = _flat(_yaml("configs/ALFGraph/METRLA_alfgraph_k1.yaml"))
    other = _flat(_yaml(graph_path.lstrip("./")))
    drift = {key for key in set(base) | set(other) if base.get(key) != other.get(key)}
    assert drift == {"runtime.model_variant", "family.recipe.moge_router_mode"}, sorted(drift)
    assert other["family.recipe.moge_router_mode"] == mode
    # the candidates exist because top-1 is the broken case; keep them on it
    assert other["family.recipe.moge_top_k"] == 1


def test_router_report_phase_states_match_the_loader():
    """The report reconstructs phase-by-k routing from `w_gate` alone, which is
    only valid while its state table matches what the loader actually builds.

    Encoding, from `data_provider/graph_data_loader.py`:
      join  k -> role NAS, age k/L, mask 1     fault k -> role FS, age k/L, mask 0
      steady  -> role AAS, age 1.0, mask 1     unseen  -> role VS, age 0.0, mask 0
    """
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        from router_report import protocol_states
    finally:
        sys.path.pop(0)

    states = protocol_states(input_len=12)
    by_label = {(phase, k): (role, age, mask) for phase, k, role, age, mask in states}

    assert by_label[("Unseen", None)] == (2, 0.0, 0.0)      # ROLE_VS
    assert by_label[("Steady", None)] == (0, 1.0, 1.0)      # ROLE_AAS
    # join track: NAS, observed, age ramps with evidence
    assert by_label[("Cold", 1)] == (1, 1 / 12, 1.0)
    assert by_label[("Ready", 12)] == (1, 1.0, 1.0)
    assert by_label[("Warm", 6)] == (1, 6 / 12, 1.0)
    # fault track: FS, unobserved
    assert by_label[("Onset", 1)] == (3, 1 / 12, 0.0)
    assert by_label[("Outage", 12)] == (3, 1.0, 0.0)
    assert by_label[("Decay", 6)] == (3, 6 / 12, 0.0)
    # k1 / k12 are their own phases, k2..11 are the middle phase
    assert sum(1 for phase, k, *_ in states if phase == "Warm") == 10
    assert sum(1 for phase, k, *_ in states if phase == "Decay") == 10
    assert len(states) == 26


def test_router_report_flags_a_uniform_router():
    """A router whose softmax is flat carries no information, however varied its
    argmax looks. Entropy has to say so."""
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        from router_report import analyse
    finally:
        sys.path.pop(0)

    # near-zero weights: argmax varies by phase, but every probability is ~1/5
    torch.manual_seed(0)
    state_dict = {"g.router.w_gate": torch.randn(6, len(SUPPORT_NAMES)) * 1e-3}
    report = analyse(state_dict, "noisy_topk")["g"]
    assert report["mean_entropy_ratio"] > 0.999, "a flat router must read as uninformative"
    assert report["mean_top1_margin"] < 1e-2
    assert not report["collapsed"], "varied argmax is exactly the misleading case"

    decisive = {"g.router.w_gate": torch.zeros(6, len(SUPPORT_NAMES))}
    decisive["g.router.w_gate"][:, 2] = 10.0        # everything to one expert
    strong = analyse(decisive, "soft")["g"]
    assert strong["collapsed"] is True
    assert strong["mean_entropy_ratio"] < 0.1


def test_fg1_still_uses_the_historical_router():
    """fg1 is the reference line; changing its router silently would invalidate
    every comparison already on disk."""
    graph = _flat(_yaml("configs/ALFGraph/METRLA_alfgraph_k1.yaml"))
    assert graph.get("family.recipe.moge_router_mode") is None
    assert graph["family.recipe.moge_top_k"] == 1
