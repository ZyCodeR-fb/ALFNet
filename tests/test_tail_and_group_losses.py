"""Contracts for the two training-only switches chosen from the diagnostics.

T (`--lcbt_tail_weight`): average top-q squared error inside each lifecycle
group. Chosen because fg1's one failing test cell is Ready RMSE, and the Ready
error distribution is a tail: the worst 5% of cells carry 67.9% of the squared
error, MAE 0.192 against RMSE 0.339.

W (`--lcbt_worst_group_weight`): smooth worst cell over group x evidence length.
Chosen because fg1's other three failing cells (Join MAE, Warm MAE, Ready MAE)
are individual phases invisible in the mean.

P (PCGrad/CAGrad) is deliberately absent: measured over 24 real batches, every
pairwise gradient cosine between the four lifecycle groups is positive
(+0.138 to +0.357). Gradient surgery has nothing to correct.
"""

import argparse
import math

import pytest
import torch
import yaml

from pathlib import Path

from args import _add_arguments
from experiments.inductive_epoch import _group_tail_loss, _group_worst_loss


ROOT = Path(__file__).resolve().parents[1]


def _cells(values, groups):
    """values: [B, H, N, 1] errors already added to a zero prediction."""
    y = torch.zeros_like(values)
    valid = torch.ones_like(values, dtype=torch.bool)
    return values, y, valid, torch.tensor(groups)


def test_both_switches_are_off_by_default():
    parser = argparse.ArgumentParser()
    _add_arguments(parser)
    assert float(parser.get_default("lcbt_tail_weight")) == 0.0
    assert float(parser.get_default("lcbt_worst_group_weight")) == 0.0
    assert float(parser.get_default("lcbt_tail_quantile")) == 0.1


def test_tail_loss_targets_the_tail_not_the_mean():
    # one group, 100 cells, 10 of them large
    values = torch.zeros(1, 100, 1, 1)
    values[0, :90] = 1.0
    values[0, 90:] = 5.0
    y_hat, y, valid, groups = _cells(values, [0])
    mean_squared = (y_hat - y).pow(2).mean()
    tail = _group_tail_loss(y_hat, y, valid, groups, "mse", 1.0, quantile=0.1)
    assert float(tail) == pytest.approx(25.0, rel=1e-5), "top 10% is the ten 5.0 cells"
    assert float(tail) > float(mean_squared) * 2, "the tail term must exceed the mean"
    # a smaller quantile concentrates harder
    assert float(_group_tail_loss(y_hat, y, valid, groups, "mse", 1.0, 0.01)) == pytest.approx(25.0)


def test_tail_loss_is_squared_regardless_of_the_main_loss():
    values = torch.full((1, 10, 1, 1), 3.0)
    y_hat, y, valid, groups = _cells(values, [0])
    for name in ("mae", "mse", "huber"):
        assert float(_group_tail_loss(y_hat, y, valid, groups, name, 1.0, 1.0)) == pytest.approx(9.0)


def test_tail_loss_is_computed_inside_each_group():
    """A large group must not drown out a small one."""
    values = torch.zeros(4, 10, 1, 1)
    values[:2] = 1.0          # group 0, small errors
    values[2:] = 4.0          # group 1, large errors
    y_hat, y, valid, groups = _cells(values, [0, 0, 1, 1])
    per_group = _group_tail_loss(y_hat, y, valid, groups, "mse", 1.0, 1.0)
    # mean over group means: (1 + 16) / 2
    assert float(per_group) == pytest.approx(8.5, rel=1e-5)
    # pooling everything instead would give (2*1 + 2*16) / 4 = 8.5 here only by
    # coincidence of equal sizes; make the sizes unequal to separate them
    values = torch.zeros(4, 10, 1, 1)
    values[:3] = 1.0
    values[3:] = 4.0
    y_hat, y, valid, groups = _cells(values, [0, 0, 0, 1])
    assert float(_group_tail_loss(y_hat, y, valid, groups, "mse", 1.0, 1.0)) == pytest.approx(8.5)


def test_tail_loss_off_and_gradient_flows():
    values = torch.full((2, 8, 3, 1), 2.0, requires_grad=True)
    y = torch.zeros_like(values)
    valid = torch.ones_like(values, dtype=torch.bool)
    groups = torch.tensor([0, 1])
    assert _group_tail_loss(values, y, valid, groups, "mse", 0.0) is None
    loss = _group_tail_loss(values, y, valid, groups, "mse", 1.0, 0.25)
    loss.backward()
    assert values.grad is not None and float(values.grad.abs().max()) > 0
    assert torch.isfinite(values.grad).all()


def test_tail_loss_survives_a_group_with_no_valid_cells():
    values = torch.full((2, 4, 2, 1), 1.0)
    y = torch.zeros_like(values)
    valid = torch.ones_like(values, dtype=torch.bool)
    valid[1] = False                      # group 1 contributes nothing
    groups = torch.tensor([0, 1])
    loss = _group_tail_loss(values, y, valid, groups, "mse", 1.0, 0.5)
    assert loss is not None and torch.isfinite(loss)
    valid[:] = False
    assert _group_tail_loss(values, y, valid, groups, "mse", 1.0, 0.5) is None


def test_worst_group_and_tail_compose_without_interfering():
    values = torch.zeros(4, 12, 2, 1, requires_grad=True)
    with torch.no_grad():
        values[2:] = 3.0
    y = torch.zeros_like(values)
    valid = torch.ones_like(values, dtype=torch.bool)
    groups = torch.tensor([0, 0, 1, 1])
    steps = torch.tensor([1, 2, 1, 2])
    tail = _group_tail_loss(values, y, valid, groups, "mae", 1.0, 0.5)
    worst = _group_worst_loss(values, y, valid, groups, steps, "mae", 1.0, 0.1)
    assert tail is not None and worst is not None
    total = tail + worst
    total.backward()
    assert torch.isfinite(values.grad).all() and float(values.grad.abs().max()) > 0


# --- the three arms differ only in the training switch ----------------------


def _flat(node, prefix=""):
    out = {}
    for key, value in (node or {}).items():
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(_flat(value, path + "."))
        else:
            out[path] = value
    return out


@pytest.mark.parametrize("name", ["tq", "wg", "tw"])
def test_arm_yaml_equals_fg1_except_for_its_name(name):
    """T and W are CLI switches, so the model configuration must be identical."""
    base = _flat(yaml.safe_load((ROOT / "configs/ALFNet/fg1.yaml").read_text(encoding="utf-8")))
    other = _flat(yaml.safe_load((ROOT / f"configs/ALFNet/{name}.yaml").read_text(encoding="utf-8")))
    drift = {key for key in set(base) | set(other) if base.get(key) != other.get(key)}
    assert drift == {"runtime.model_variant"}, sorted(drift)


@pytest.mark.parametrize("knob,value", [
    ("lcbt_tail_weight", 0.5),
    ("lcbt_tail_quantile", 0.05),
    ("lcbt_worst_group_weight", 0.5),
    ("lcbt_worst_group_temperature", 0.2),
])
def test_training_switch_changes_the_experiment_signature(knob, value):
    """Otherwise two arms would share a checkpoint directory."""
    from types import SimpleNamespace

    from main import TRAINING_OBJECTIVE_KNOBS, _build_train_signature

    base = dict(
        model_variant="v", model_recipe="r", model_config_path="", model_family_cfg={},
        d_model=64, k_hop=1, e_layers=2, d_layers=1, activation="gelu", loss="mse",
        optimizer="adamw", scheduler="None", scheduler_patience=1, lr=5e-5,
        weight_decay=1e-4, batch_size=64, max_epoch=100, patience=10,
        stop_based="val_lifecycle", num_used_experts=3, mean_expert=True,
        weight_expert=True, max_expert=True, min_expert=True, diffusion_expert=True,
        train_init_checkpoint_path=None,
    )
    default = _build_train_signature(SimpleNamespace(**base, **TRAINING_OBJECTIVE_KNOBS))
    changed = dict(TRAINING_OBJECTIVE_KNOBS)
    changed[knob] = value
    assert _build_train_signature(SimpleNamespace(**base, **changed)) != default
