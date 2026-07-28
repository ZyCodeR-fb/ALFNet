"""Frozen teacher supplying a model-difference target for the residual (K).

Why the target changed. The directly-supervised residual (R) learned

    R_target = Y - stopgrad(Y_anchor)

on the training split, where the anchor had already converged. That target is
dominated by label noise rather than by anything the inputs determine, and the
runs show it: the DLinear variant's best validation epoch was 1 of 11, i.e. the
untrained residual was the best one, and all three variants ended 0.001-0.055
*worse* than the anchor on Steady/Join/Fault.

    Y - Y_anchor(x)      is not a function of x  -> partly unlearnable
    Y_teacher(x) - Y_anchor(x)  is a function of x -> learnable in principle

so this supplies the second target. The teacher is A2, the retired two-expert
model, whose advantage over fg1 is visible on the validation split alone
(S+J+F mean 3.6717 against 4.0659) and therefore needs no reference to the test
set to justify.

The teacher runs forward-only under no_grad, never trains, and never appears at
inference: the shipped model is the anchor plus a ~15k-parameter residual.
"""

import hashlib
import json
import os

import torch
from torch import nn


def checkpoint_fingerprint(path):
    """Bytes hash of the checkpoint plus the protocol hash it was trained under."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    summary_path = os.path.join(os.path.dirname(path), "training_summary.json")
    protocol_hash = None
    if os.path.exists(summary_path):
        with open(summary_path, "r", encoding="utf-8") as handle:
            protocol_hash = json.load(handle).get("training_protocol_hash")
    return {
        "path": os.path.abspath(path),
        "sha256": digest.hexdigest(),
        "training_protocol_hash": protocol_hash,
    }


class FrozenTeacher(nn.Module):
    """A2 held frozen in eval mode, used only to produce Y_teacher(x).

    Refuses to load if the teacher was trained under a different protocol hash
    than the run using it: a teacher fitted to another split or another scaler
    would be supplying targets from a different problem.
    """

    def __init__(self, model, provenance, required_protocol_hash=None):
        super().__init__()
        teacher_hash = provenance.get("training_protocol_hash")
        if required_protocol_hash and teacher_hash != required_protocol_hash:
            raise ValueError(
                "Teacher checkpoint was trained under a different LCBT protocol:\n"
                f"  teacher {teacher_hash}\n  current {required_protocol_hash}\n"
                "Refusing to distil across protocols."
            )
        self.teacher = model
        for parameter in self.teacher.parameters():
            parameter.requires_grad_(False)
        self.provenance = dict(provenance)
        # nn.Module.__init__ leaves `training` True; the override below only
        # fires when train() is called, so put it in eval explicitly.
        self.eval()

    def train(self, mode=True):
        # The teacher never leaves eval: dropout and scheduled sampling would
        # make its target depend on the training-time random state.
        return super().train(False)

    @torch.no_grad()
    def forward(self, adjacency, x_enc, x_t_mark, pos_mark, dec_t_mark, epoch=0):
        output = self.teacher(adjacency, x_enc, x_t_mark, pos_mark, None, dec_t_mark, epoch)
        prediction = output["pred"] if isinstance(output, dict) else output
        return prediction.detach()


def distillation_loss(applied_residual, teacher, anchor, valid, gate,
                      loss_name="mae", gate_threshold=1e-6):
    """Match the *applied* residual to the teacher's correction of the anchor.

        target = stopgrad(Y_teacher - Y_anchor)

    Supervising the head's raw output instead would train something the model
    does not actually add: the value that reaches the forecast is the gated,
    tanh-bounded one, so that is what is supervised.

    Cells whose gate is zero -- the zero-history and unseen nodes -- are excluded
    rather than pushed towards a target they can never move, which keeps the
    protected path exactly the anchor's.
    """
    open_gate = (gate > gate_threshold).unsqueeze(1).expand_as(valid)
    selected = valid & open_gate
    if not selected.any():
        return None, {}
    target = (teacher - anchor).detach()
    predicted = applied_residual
    difference = (predicted - target)[selected]
    loss = difference.pow(2).mean() if loss_name in {"mse", "rmse"} else difference.abs().mean()
    with torch.no_grad():
        goal = target[selected]
        estimate = predicted[selected]
        centred_goal = goal - goal.mean()
        centred_estimate = estimate - estimate.mean()
        denominator = centred_goal.norm() * centred_estimate.norm()
        report = {
            "teacher_gap_abs_mean": float(goal.abs().mean()),
            "teacher_gap_fit_mae": float((estimate - goal).abs().mean()),
            "teacher_gap_corr": float(
                (centred_goal * centred_estimate).sum() / denominator
            ) if float(denominator) > 0 else 0.0,
            "teacher_gap_cells": float(selected.float().mean()),
        }
    return loss, report
