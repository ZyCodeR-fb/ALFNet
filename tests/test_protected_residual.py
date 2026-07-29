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
# The stage-two family: anchor frozen, residual supervised directly.
VARIANTS = ["rt", "rd", "rtg"]
# JR is the same module wired for the deliverable: joint, single stage, no
# companion term. The structural contracts below hold for both families; the
# freezing contracts hold only for VARIANTS.
END_TO_END = ["jr"]
ALL_RESIDUAL = VARIANTS + END_TO_END
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


@pytest.mark.parametrize("name", ALL_RESIDUAL)
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


@pytest.mark.parametrize("name", ALL_RESIDUAL)
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


@pytest.mark.parametrize("name", ["rt", "rd", "jr"])
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


@pytest.mark.parametrize("name", ["rt", "rd", "jr"])
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


@pytest.mark.parametrize("name", ALL_RESIDUAL)
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


# --- JR: the same residual, wired for the deliverable ------------------------
#
# The deliverable must be trainable from random initialisation in one stage.
# rt/rd/rtg and kt/kd all fail that: they freeze the anchor and need either a
# companion target or a teacher checkpoint. JR removes both devices, so these
# tests check the *absence* of every stage-two mechanism, not just that the
# forward pass runs.


def test_jr_carries_no_stage_two_mechanism():
    _, model = build("jr")
    assert model.temporal_residual is not None
    assert model.freeze_anchor is False, "JR must not freeze the anchor"
    assert model.companion_loss_weight == 0.0, "JR must not use direct supervision"
    assert model.lambda_kd == 0.0 and model._teacher is None, "JR must not use a teacher"


def test_jr_yaml_loads_no_pretrained_expert():
    """The experts must be built from scratch, not restored from fg1/A2."""
    config = yaml.safe_load((ROOT / "configs/ALFNet/jr.yaml").read_text(encoding="utf-8"))
    recipe = config["family"]["recipe"]
    assert "teacher_checkpoint" not in recipe and "teacher_config_path" not in recipe
    for expert in config["family"]["experts"].values():
        assert expert["load_checkpoint"] is False
        assert not expert["checkpoint_path"]


def test_jr_differs_from_rt_only_in_the_two_stage_switches():
    """Pin the claim in jr.yaml: nothing else was retuned along the way.

    If a later edit moves residual_lr or the kernel size as well, the run stops
    being an isolated test of "joint instead of frozen" and this fails.
    """
    left = _flat(yaml.safe_load((ROOT / "configs/ALFNet/rt.yaml").read_text(encoding="utf-8")))
    right = _flat(yaml.safe_load((ROOT / "configs/ALFNet/jr.yaml").read_text(encoding="utf-8")))
    drift = {key for key in set(left) | set(right) if left.get(key) != right.get(key)}
    assert drift == {
        "runtime.model_variant",
        "family.recipe.companion_loss_weight",
        "family.recipe.freeze_anchor",
    }, sorted(drift)


def test_jr_emits_a_zero_weight_companion_term():
    """weight 0 is what makes `inductive_epoch` skip the companion loss."""
    args, model = build("jr")
    model.train()
    data = _batch(args, model)
    out = model(data["adj"], data["x"], data["tm"], None, data["dec"], data["dm"], 0)
    assert float(out["companion_residual"]["weight"]) == 0.0
    assert out["distillation"] is None


def test_jr_trains_the_anchor_and_the_residual_together():
    """The joint counterpart of the frozen test: both sides must move.

    Under the plain forecast loss alone -- no companion term, no teacher -- the
    residual has to pick up gradient through the sum, and the anchor must keep
    training. If either side is static the run is not the end-to-end candidate.
    """
    args, model = build("jr")
    assert all(p.requires_grad for p in model.parameters())
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    data = _batch(args, model)
    model.train()
    for _ in range(5):
        opt.zero_grad(set_to_none=True)
        out = model(data["adj"], data["x"], data["tm"], None, data["dec"], data["dm"], 0)
        (out["pred"] - data["dec"]).pow(2).mean().backward()
        opt.step()
    current = dict(model.named_parameters())
    moved = {
        name: float((current[name].detach() - value).abs().max())
        for name, value in before.items()
    }
    residual_moved = max(v for n, v in moved.items() if n.startswith("temporal_residual."))
    anchor_moved = max(v for n, v in moved.items() if not n.startswith("temporal_residual."))
    assert residual_moved > 0, "the residual received no gradient from the forecast loss"
    assert anchor_moved > 0, "the anchor did not train -- this is not joint training"


@pytest.mark.parametrize("name", ALL_RESIDUAL)
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


# --- K: distillation from a frozen teacher ----------------------------------


DISTILL = ["kt", "kd"]


def test_teacher_stays_out_of_the_shipped_model():
    """The teacher trains nothing, ships nothing, and is never checkpointed."""
    _, model = build("kt")
    assert model._teacher is not None
    assert not any(p.requires_grad for p in model._teacher.parameters())
    assert not any("teacher" in key for key in model.state_dict()), (
        "the teacher leaked into the checkpoint"
    )
    optimised = {id(p) for g in model.get_optimizer_param_groups(1e-3) for p in g["params"]}
    assert not any(id(p) in optimised for p in model._teacher.parameters())
    # and it stays in eval even when the parent is put in train mode
    model.train()
    assert model._teacher.training is False


def test_teacher_protocol_hash_mismatch_is_refused():
    from layers.teacher_gap import FrozenTeacher

    stub = torch.nn.Linear(1, 1)
    provenance = {"sha256": "x", "training_protocol_hash": "aaa"}
    FrozenTeacher(stub, provenance, required_protocol_hash="aaa")
    with pytest.raises(ValueError, match="different LCBT protocol"):
        FrozenTeacher(stub, provenance, required_protocol_hash="bbb")


def test_distillation_supervises_the_applied_residual():
    from layers.teacher_gap import distillation_loss

    anchor = torch.zeros(2, 3, 4, 1)
    teacher = torch.full_like(anchor, 2.0)
    valid = torch.ones_like(anchor, dtype=torch.bool)
    gate = torch.ones(2, 4, 1)
    exact = torch.full_like(anchor, 2.0)          # residual equals the gap
    loss, report = distillation_loss(exact, teacher, anchor, valid, gate)
    assert float(loss) == pytest.approx(0.0)
    assert report["teacher_gap_abs_mean"] == pytest.approx(2.0)
    assert report["teacher_gap_fit_mae"] == pytest.approx(0.0)
    wrong = torch.zeros_like(anchor)
    loss, report = distillation_loss(wrong, teacher, anchor, valid, gate)
    assert float(loss) == pytest.approx(2.0)
    # closed gates are excluded rather than pushed at an unreachable target
    closed = torch.zeros(2, 4, 1)
    assert distillation_loss(exact, teacher, anchor, valid, closed) == (None, {})


def test_distillation_target_is_detached_from_the_teacher():
    from layers.teacher_gap import distillation_loss

    anchor = torch.zeros(2, 3, 4, 1, requires_grad=True)
    teacher = torch.full_like(anchor, 2.0).requires_grad_(True)
    residual = torch.zeros_like(anchor).requires_grad_(True)
    loss, _ = distillation_loss(residual, teacher, anchor, torch.ones_like(anchor, dtype=torch.bool),
                               torch.ones(2, 4, 1))
    loss.backward()
    assert residual.grad is not None and float(residual.grad.abs().max()) > 0
    for tensor in (teacher, anchor):
        assert tensor.grad is None or float(tensor.grad.abs().max()) == 0.0


@pytest.mark.parametrize("name", DISTILL)
def test_distillation_variant_forwards_and_backwards(name):
    args, model = build(name)
    data = _batch(args, model)
    model.train()
    out = model(data["adj"], data["x"], data["tm"], None, data["dec"], data["dm"], 0)
    assert torch.isfinite(out["pred"]).all()
    distill = out["distillation"]
    assert distill is not None and torch.isfinite(distill["teacher"]).all()
    assert not distill["teacher"].requires_grad, "the teacher output carries grad"
    from layers.teacher_gap import distillation_loss

    loss, report = distillation_loss(
        distill["residual"], distill["teacher"], distill["anchor"],
        torch.ones_like(data["dec"], dtype=torch.bool), distill["gate"], "mae",
    )
    assert loss is not None and torch.isfinite(loss)
    assert set(report) >= {"teacher_gap_abs_mean", "teacher_gap_fit_mae",
                           "teacher_gap_corr", "teacher_gap_cells"}
    (out["pred"].pow(2).mean() + loss).backward()
    grads = [p.grad for p in model.temporal_residual.parameters() if p.grad is not None]
    assert grads and any(float(g.abs().max()) > 0 for g in grads)
    assert all(float(p.grad.abs().max()) == 0.0
               for p in model._teacher.parameters() if p.grad is not None)


@pytest.mark.parametrize("name", DISTILL)
def test_distillation_keeps_the_unseen_path_bit_identical(name):
    args, model = build(name)
    _amplify(model.temporal_residual)
    model.eval()
    nodes = len(model.known_nodes)
    data = _batch(args, model)
    onehot = torch.zeros(2, args.input_len, nodes, 4)
    onehot[..., ROLE_VS] = 1.0
    data["x"][..., model.role_start:model.role_end] = onehot
    data["x"][..., model.mask_index] = 0.0
    with torch.no_grad():
        with_kd = model(data["adj"], data["x"], data["tm"], None, None, data["dm"], 0)["pred"]
        head, model.temporal_residual = model.temporal_residual, None
        plain = model(data["adj"], data["x"], data["tm"], None, None, data["dm"], 0)["pred"]
        model.temporal_residual = head
    assert torch.equal(with_kd, plain)


def test_teacher_requires_a_residual_to_distil_into():
    args = load_args_from_config("configs/ALFNet/fg1.yaml", [])
    args.device = "cpu"
    args.model_family_cfg["recipe"]["teacher_checkpoint"] = "x.pth"
    args.model_family_cfg["recipe"]["teacher_config_path"] = "configs/ALFNet/rf_l3s.yaml"
    with pytest.raises(ValueError, match="nothing to distil into"):
        load_model_class(args.model)(args)


@pytest.mark.parametrize("name", DISTILL)
def test_distillation_yaml_differs_from_rt_only_in_the_teacher_keys(name):
    allowed = ALLOWED | {
        "family.recipe.lambda_kd", "family.recipe.teacher_checkpoint",
        "family.recipe.teacher_config_path",
    }
    base = _flat(yaml.safe_load((ROOT / "configs/ALFNet/rt.yaml").read_text(encoding="utf-8")))
    other = _flat(yaml.safe_load((ROOT / f"configs/ALFNet/{name}.yaml").read_text(encoding="utf-8")))
    drift = {key for key in set(base) | set(other) if base.get(key) != other.get(key)}
    assert drift <= allowed, f"{name}.yaml drifted on {sorted(drift - allowed)}"


def test_teacher_follows_the_model_onto_its_device():
    """Unregistered modules are skipped by .to(); the teacher must not be."""
    _, model = build("kt")
    assert next(model._teacher.parameters()).dtype == torch.float32
    model = model.to(torch.float64)
    assert next(model.parameters()).dtype == torch.float64
    assert next(model._teacher.parameters()).dtype == torch.float64, (
        "the teacher stayed behind when the model was converted"
    )
    if torch.cuda.is_available():
        model = model.float().cuda()
        assert next(model._teacher.parameters()).is_cuda


def test_teacher_is_not_used_at_inference():
    """The shipped model is the anchor plus the residual, nothing more."""
    args, model = build("kt")
    data = _batch(args, model)
    model.eval()
    with torch.no_grad():
        out = model(data["adj"], data["x"], data["tm"], None, None, data["dm"], 0)
    assert out["distillation"] is None, "the teacher ran during inference"
    model.train()
    out = model(data["adj"], data["x"], data["tm"], None, data["dec"], data["dm"], 0)
    assert out["distillation"] is not None


def test_async_test_mode_does_not_need_the_teacher_checkpoint():
    args = load_args_from_config("configs/ALFNet/kt.yaml", [])
    args.device = "cpu"
    args.test_for_async_lifecycle = True
    args.model_family_cfg["recipe"]["teacher_checkpoint"] = "does/not/exist.pth"
    model = load_model_class(args.model)(args)
    assert model._teacher is None and model.teacher_enabled is False
