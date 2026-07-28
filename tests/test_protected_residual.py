"""Contracts for the protected, directly-supervised temporal residual (R).

The three properties that make this candidate safe to train, each measured:

  * an untrained residual reproduces fg1 exactly, so enabling R cannot regress;
  * the gate is hard zero without own-history, so the zero-history and unseen
    paths stay bit-identical while the anchor is frozen;
  * the residual receives non-zero gradient from its own companion loss, which
    is what the previous routed branch never had.
"""

import torch
import pytest
import yaml

from pathlib import Path

from args import load_args_from_config
from layers.protected_residual import (
    DLinearDecomposition,
    DepthwiseCausalTCN,
    ProtectedTemporalResidual,
    companion_residual_loss,
)
from model.regist_model import load_model_class


ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ["rt", "rd", "rtg"]
ROLE_AAS, ROLE_NAS, ROLE_VS, ROLE_FS = 0, 1, 2, 3


def build(name, seed=32):
    """Seeded on every call: without it two builds continue one RNG stream and
    every comparison below would be against a differently-initialised model."""
    import pytorch_lightning as pl

    pl.seed_everything(seed, workers=True)
    args = load_args_from_config(f"configs/ALFNet/{name}.yaml", [])
    args.device = "cpu"
    return args, load_model_class(args.model)(args)


def _batch(args, model, batch=2, seed=5):
    nodes = len(model.known_nodes)
    generator = torch.Generator().manual_seed(seed)
    return dict(
        adj=torch.rand(nodes, nodes, generator=generator),
        x=torch.randn(batch, args.input_len, nodes, args.enc_in, generator=generator),
        tm=torch.rand(batch, args.input_len, 3, generator=generator) * 5,
        dec=torch.randn(batch, args.pred_len, nodes, args.c_out, generator=generator),
        dm=torch.rand(batch, args.pred_len, 3, generator=generator) * 5,
    )


def _amplify(module, std=0.6):
    for layer in module.modules():
        if isinstance(layer, (torch.nn.Linear, torch.nn.Conv1d)):
            torch.nn.init.normal_(layer.weight, std=std)
            if layer.bias is not None:
                torch.nn.init.normal_(layer.bias, std=std)


# --- the causal convolution must not read the future -------------------------


def test_depthwise_tcn_is_causal():
    torch.manual_seed(0)
    block = DepthwiseCausalTCN(channels=2, hidden=4, kernel_size=3, dilation=1, dropout=0.0)
    block.eval()
    x = torch.randn(3, 2, 12)
    reference = block(x)
    assert reference.shape == (3, 4, 12)
    perturbed = x.clone()
    perturbed[..., -1] += 10.0                    # change only the last step
    out = block(perturbed)
    # every earlier position must be untouched: information cannot flow back
    assert torch.allclose(out[..., :-1], reference[..., :-1], atol=1e-6), (
        "the convolution leaks future information into earlier steps"
    )
    assert not torch.allclose(out[..., -1], reference[..., -1])


def test_dlinear_decomposition_splits_trend_and_seasonal():
    torch.manual_seed(0)
    module = DLinearDecomposition(input_len=12, pred_len=12, kernel_size=5)
    x = torch.arange(12, dtype=torch.float32).view(1, 1, 12).repeat(4, 1, 1)
    seasonal, trend = module.decompose(x)
    assert seasonal.shape == trend.shape == x.shape
    assert torch.allclose(seasonal + trend, x, atol=1e-5), "the split must be exact"
    # a pure ramp is all trend
    assert float(seasonal.abs().max()) < 1.0
    assert module(x).shape == (4, 1, 12)


# --- an untrained residual reproduces the anchor exactly ---------------------


@pytest.mark.parametrize("name", VARIANTS)
def test_untrained_residual_reproduces_fg1(name):
    args, model = build(name)
    _, anchor = build("fg1")
    anchor.load_state_dict(
        {k: v for k, v in model.state_dict().items() if k in anchor.state_dict()},
        strict=True,
    )
    model.eval()
    anchor.eval()
    data = _batch(args, model)
    with torch.no_grad():
        with_residual = model(data["adj"], data["x"], data["tm"], None, None, data["dm"], 0)["pred"]
        plain = anchor(data["adj"], data["x"], data["tm"], None, None, data["dm"], 0)["pred"]
    assert torch.allclose(with_residual, plain, atol=1e-6), (
        "a zero-initialised residual already changed the forecast"
    )


@pytest.mark.parametrize("name", VARIANTS)
def test_residual_adds_only_its_own_parameters(name):
    _, plain = build("fg1")
    _, model = build(name)
    extra = set(model.state_dict()) - set(plain.state_dict())
    assert extra and all(key.startswith("temporal_residual.") for key in extra), extra
    common = set(plain.state_dict()) & set(model.state_dict())
    drifted = [
        key for key in sorted(common)
        if not torch.equal(plain.state_dict()[key], model.state_dict()[key])
    ]
    # the residual is built inside a forked RNG, so the anchor is untouched
    assert not drifted, f"enabling R perturbed the anchor at {drifted[:5]}"


# --- the gate is hard zero without own-history -------------------------------


def test_gate_is_exactly_zero_without_own_history():
    torch.manual_seed(0)
    residual = ProtectedTemporalResidual(
        input_len=12, pred_len=12, output_dim=1, availability_count=4,
        graph_hidden_dim=8, hidden=8, kind="tcn",
    )
    _amplify(residual)
    availability = torch.zeros(2, 5, 4)
    availability[..., ROLE_VS] = 1.0
    age = torch.rand(2, 5, 1)
    visibility = torch.zeros(2, 5, 1)
    e_self = torch.zeros(2, 5, 1)
    gate = residual.gate(availability, age, visibility, e_self)
    assert float(gate.abs().max()) == 0.0
    # with history the gate is free to open, and stays in [0, 1]
    open_gate = residual.gate(availability, age, torch.ones(2, 5, 1), torch.ones(2, 5, 1))
    assert float(open_gate.min()) >= 0.0 and float(open_gate.max()) <= 1.0
    assert float(open_gate.max()) > 0.0


@pytest.mark.parametrize("name", ["rt", "rd"])
def test_zero_history_nodes_keep_the_anchor_bit_for_bit(name):
    args, model = build(name)
    _amplify(model.temporal_residual)
    model.eval()
    nodes = len(model.known_nodes)
    data = _batch(args, model)
    onehot = torch.zeros(2, args.input_len, nodes, 4)
    onehot[..., ROLE_VS] = 1.0
    data["x"][..., model.role_start:model.role_end] = onehot
    data["x"][..., model.mask_index] = 0.0            # nothing ever observed
    with torch.no_grad():
        with_residual = model(data["adj"], data["x"], data["tm"], None, None, data["dm"], 0)["pred"]
        head, model.temporal_residual = model.temporal_residual, None
        plain = model(data["adj"], data["x"], data["tm"], None, None, data["dm"], 0)["pred"]
        model.temporal_residual = head
    assert torch.equal(with_residual, plain), (
        "the residual reached a node with no own-history"
    )


@pytest.mark.parametrize("name", ["rt", "rd"])
def test_nodes_with_history_do_receive_a_correction(name):
    args, model = build(name)
    _amplify(model.temporal_residual)
    model.eval()
    nodes = len(model.known_nodes)
    data = _batch(args, model)
    onehot = torch.zeros(2, args.input_len, nodes, 4)
    onehot[..., ROLE_AAS] = 1.0
    data["x"][..., model.role_start:model.role_end] = onehot
    data["x"][..., model.mask_index] = 1.0
    with torch.no_grad():
        with_residual = model(data["adj"], data["x"], data["tm"], None, None, data["dm"], 0)["pred"]
        head, model.temporal_residual = model.temporal_residual, None
        plain = model(data["adj"], data["x"], data["tm"], None, None, data["dm"], 0)["pred"]
        model.temporal_residual = head
    assert not torch.allclose(with_residual, plain), "the residual never fires"
    assert float((with_residual - plain).abs().max()) <= model.temporal_residual.residual_scale + 1e-5


# --- direct supervision actually reaches the residual ------------------------


def test_companion_loss_targets_the_anchor_error():
    target = torch.full((2, 3, 4, 1), 5.0)
    anchor = torch.full_like(target, 4.0)
    valid = torch.ones_like(target, dtype=torch.bool)
    perfect = torch.full_like(target, 1.0)          # exactly Y - Y_anchor
    assert float(companion_residual_loss(perfect, target, anchor, valid)) == pytest.approx(0.0)
    wrong = torch.zeros_like(target)
    assert float(companion_residual_loss(wrong, target, anchor, valid)) == pytest.approx(1.0)
    # the anchor must be detached: no gradient may flow back into it
    anchor = anchor.clone().requires_grad_(True)
    residual = torch.zeros_like(target).requires_grad_(True)
    companion_residual_loss(residual, target, anchor, valid).backward()
    assert residual.grad is not None and float(residual.grad.abs().max()) > 0
    assert anchor.grad is None or float(anchor.grad.abs().max()) == 0.0


@pytest.mark.parametrize("name", ["rt", "rd"])
def test_residual_receives_non_zero_companion_gradient(name):
    args, model = build(name)
    model.train()
    data = _batch(args, model)
    out = model(data["adj"], data["x"], data["tm"], None, data["dec"], data["dm"], 0)
    companion = out["companion_residual"]
    loss = companion_residual_loss(
        companion["residual"], data["dec"], companion["anchor"],
        torch.ones_like(data["dec"], dtype=torch.bool), "mae",
    )
    assert loss is not None and torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in model.temporal_residual.parameters() if p.grad is not None]
    assert grads, "the residual received no gradient from its own supervision"
    assert any(float(g.abs().max()) > 0 for g in grads)
    assert all(torch.isfinite(g).all() for g in grads)
    # and nothing flowed into the frozen anchor through the companion term
    anchor_grads = [
        p.grad for n, p in model.named_parameters()
        if not n.startswith("temporal_residual.") and p.grad is not None
    ]
    assert all(float(g.abs().max()) == 0.0 for g in anchor_grads), (
        "the companion loss leaked gradient into the anchor"
    )


# --- freezing --------------------------------------------------------------


@pytest.mark.parametrize("name", VARIANTS)
def test_freeze_leaves_only_the_residual_trainable(name):
    _, model = build(name)
    assert model.freeze_anchor is True
    model.freeze_anchor_parameters()
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable, "nothing is trainable"
    assert all(n.startswith("temporal_residual.") for n in trainable), sorted(trainable)[:5]
    residual_count = sum(p.numel() for p in model.temporal_residual.parameters())
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == residual_count


@pytest.mark.parametrize("name", ["rt", "rd"])
def test_frozen_anchor_does_not_move_under_optimisation(name):
    args, model = build(name)
    model.freeze_anchor_parameters()
    before = {n: p.detach().clone() for n, p in model.named_parameters()
              if not n.startswith("temporal_residual.")}
    residual_before = {n: p.detach().clone() for n, p in model.named_parameters()
                       if n.startswith("temporal_residual.")}
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-2)
    data = _batch(args, model)
    model.train()
    for _ in range(5):
        opt.zero_grad(set_to_none=True)
        out = model(data["adj"], data["x"], data["tm"], None, data["dec"], data["dm"], 0)
        out["pred"].pow(2).mean().backward()
        opt.step()
    current = dict(model.named_parameters())
    for key, value in before.items():
        assert torch.equal(value, current[key]), f"{key} moved while frozen"
    moved = max(
        float((current[key].detach() - value).abs().max())
        for key, value in residual_before.items()
    )
    assert moved > 0, "the residual did not train"


@pytest.mark.parametrize("name", VARIANTS)
def test_variant_forwards_and_backwards(name):
    args, model = build(name)
    data = _batch(args, model)
    model.train()
    out = model(data["adj"], data["x"], data["tm"], None, data["dec"], data["dm"], 0)
    assert torch.isfinite(out["pred"]).all()
    assert out["pred"].shape == (2, args.pred_len, len(model.known_nodes), args.c_out)
    loss = out["pred"].pow(2).mean()
    for value in (out.get("aux_losses") or {}).values():
        if torch.is_tensor(value):
            loss = loss + value.float()
    loss.backward()
    grads = [p for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(p.grad).all() for p in grads)


# --- the worst-group training switch ----------------------------------------


def test_worst_group_loss_is_off_by_default_and_penalises_the_worst_cell():
    import argparse

    from args import _add_arguments
    from experiments.inductive_epoch import _group_worst_loss

    parser = argparse.ArgumentParser()
    _add_arguments(parser)
    assert float(parser.get_default("lcbt_worst_group_weight")) == 0.0

    y = torch.zeros(4, 3, 2, 1)
    y_hat = torch.zeros_like(y)
    y_hat[2:] = 4.0                       # one group is much worse
    valid = torch.ones_like(y, dtype=torch.bool)
    groups = torch.tensor([0, 0, 1, 1])
    steps = torch.tensor([1, 1, 1, 1])
    assert _group_worst_loss(y_hat, y, valid, groups, steps, "mae", 0.0) is None
    soft = _group_worst_loss(y_hat, y, valid, groups, steps, "mae", 1.0, temperature=0.05)
    mean = (y_hat - y).abs()[valid].mean()
    assert float(soft) > float(mean), "the soft max must exceed the plain average"
    assert float(soft) == pytest.approx(4.0, abs=0.05), "it should approach the worst cell"


# --- YAML hygiene -----------------------------------------------------------


def _flat(node, prefix=""):
    out = {}
    for key, value in (node or {}).items():
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(_flat(value, path + "."))
        else:
            out[path] = value
    return out


ALLOWED = {
    "runtime.model_variant",
    "family.recipe.temporal_residual", "family.recipe.temporal_residual_scale",
    "family.recipe.companion_loss_weight", "family.recipe.residual_kernel_size",
    "family.recipe.residual_dilation", "family.recipe.residual_dropout",
    "family.recipe.freeze_anchor", "family.recipe.residual_lr",
}


@pytest.mark.parametrize("name", VARIANTS)
def test_variant_yaml_differs_from_fg1_only_in_the_residual_switch(name):
    base = _flat(yaml.safe_load((ROOT / "configs/ALFNet/fg1.yaml").read_text(encoding="utf-8")))
    other = _flat(yaml.safe_load((ROOT / f"configs/ALFNet/{name}.yaml").read_text(encoding="utf-8")))
    drift = {key for key in set(base) | set(other) if base.get(key) != other.get(key)}
    assert drift <= ALLOWED, f"{name}.yaml drifted on {sorted(drift - ALLOWED)}"


def test_rtg_differs_from_rt_only_in_its_name():
    """G is a CLI training switch, so rtg's YAML must equal rt's."""
    left = _flat(yaml.safe_load((ROOT / "configs/ALFNet/rt.yaml").read_text(encoding="utf-8")))
    right = _flat(yaml.safe_load((ROOT / "configs/ALFNet/rtg.yaml").read_text(encoding="utf-8")))
    drift = {key for key in set(left) | set(right) if left.get(key) != right.get(key)}
    assert drift == {"runtime.model_variant"}, sorted(drift)
