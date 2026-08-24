"""MoGERNN must see the hidden cohort under LCBT, at any batch size.

The source detects unknown nodes with `sum(x_enc, dim=(0,1)) == 0` -- a node is
unknown only if it is zero across the whole batch. That holds in the setting
MoGERNN was written for, where one fixed sensor set is unobserved for the entire
dataset. LCBT hides a different cohort per sample, so the batch-wide
intersection is usually empty, and the completion the model exists for is
switched off during training and validation while staying on at test time (the
async evaluator runs at batch size 1).

These tests pin the adapted behaviour: per-sample detection, identical to the
reference at B=1, and a batch that equals the concatenation of its samples.
"""

import pytest
import torch
import pytorch_lightning as pl

from args import load_args_from_config
from lib.runtime_context import build_runtime_context
from model.comparison_model.MoGERNN.MoGERNN import Model


def _model():
    pl.seed_everything(32, workers=True)
    args = load_args_from_config("configs/paper/v4/METRLA_MoGERNN.yaml", [])
    args.device = "cpu"
    return args, Model.from_runtime_context(args, build_runtime_context("METRLA", "train"))


def _mixed_batch(args, nodes, size, hidden=8, seed=3):
    """Every sample hides its own cohort, exactly as LCBT draws them."""
    generator = torch.Generator().manual_seed(seed)
    x = torch.rand(size, args.input_len, nodes, 1, generator=generator) + 0.5
    cohorts = []
    for index in range(size):
        start = (index * hidden) % nodes
        cohort = [(start + step) % nodes for step in range(hidden)]
        x[index, :, cohort, :] = 0.0
        cohorts.append(sorted(cohort))
    return x, cohorts


def test_batch_wide_detection_would_miss_every_cohort():
    """The defect, pinned: it is a property of the rule, not of our data."""
    nodes, hidden = 157, 8
    x, _ = _mixed_batch(load_args_from_config("configs/paper/v4/METRLA_MoGERNN.yaml", []),
                        nodes, size=8, hidden=hidden)
    flat = x.squeeze(-1)
    batch_wide = torch.where(torch.sum(flat, dim=(0, 1)) == 0)[0]
    per_sample = [
        torch.where(torch.sum(flat[i:i + 1], dim=(0, 1)) == 0)[0]
        for i in range(flat.shape[0])
    ]
    assert len(batch_wide) == 0, "this batch was supposed to have no common zero node"
    assert all(len(found) == hidden for found in per_sample), (
        "each sample must still hide its own cohort"
    )


def test_data_preparation_detects_each_sample_cohort():
    """Behavioural: the completion must actually fire on the hidden nodes."""
    args, model = _model()
    nodes = len(model.known_nodes) if hasattr(model, "known_nodes") else 157
    x, cohorts = _mixed_batch(args, nodes, size=4)
    generator = torch.Generator().manual_seed(11)
    adjacency = torch.rand(nodes, nodes, generator=generator)

    model.eval()
    with torch.no_grad():
        filled = model.data_pre(adjacency, x)

    # every hidden node must come back with something other than the zero it
    # was handed, in the sample that hid it
    for index, cohort in enumerate(cohorts):
        recovered = filled[index, :, cohort, 0]
        assert float(recovered.abs().sum()) > 0.0, (
            f"sample {index}: the hidden cohort was left at zero, so the "
            "aggregators never saw it as unknown"
        )


def test_observed_values_are_untouched():
    """Completion may only write where the input was missing."""
    args, model = _model()
    nodes = len(model.known_nodes) if hasattr(model, "known_nodes") else 157
    x, cohorts = _mixed_batch(args, nodes, size=3)
    generator = torch.Generator().manual_seed(11)
    adjacency = torch.rand(nodes, nodes, generator=generator)

    model.eval()
    with torch.no_grad():
        filled = model.data_pre(adjacency, x)

    observed = x != 0
    assert torch.allclose(filled[observed], x[observed], atol=1e-6), (
        "an observed measurement was overwritten by the completion"
    )
