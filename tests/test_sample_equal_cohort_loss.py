"""Every sample must carry the same weight, whatever its cohort size.

LCBT draws cohorts at 2.5%, 5% and 10% of the graph. Pooling all valid cells and
taking one mean therefore weighted a 5% sample twice as heavily as a 2.5% one
and a 10% sample four times, while LCE scores each cell-first row equally. The
training objective and the evaluation protocol disagreed about what matters.
"""

import pytest
import torch

from experiments.inductive_epoch import _element_loss, _sample_equal_cohort_loss


def _batch(cohort_sizes, errors, nodes=40, horizon=3):
    """One sample per cohort size, each with a constant per-cell error."""
    size = len(cohort_sizes)
    y = torch.zeros(size, horizon, nodes, 1)
    y_hat = torch.zeros_like(y)
    mask = torch.zeros_like(y, dtype=torch.bool)
    for index, (count, error) in enumerate(zip(cohort_sizes, errors)):
        mask[index, :, :count, :] = True
        y_hat[index, :, :count, :] = error
    return y_hat, y, mask


def test_samples_are_weighted_equally_regardless_of_cohort_size():
    # sample 0 covers 4 nodes with error 1, sample 1 covers 16 nodes with error 3
    y_hat, y, mask = _batch([4, 16], [1.0, 3.0])

    pooled = _element_loss(y_hat, y, "mae")[mask].mean()
    per_sample = _sample_equal_cohort_loss(y_hat, y, mask, "mae")

    # pooling weights by cell count: (4*1 + 16*3) / 20 = 2.6
    assert float(pooled) == pytest.approx(2.6)
    # equal per sample: (1 + 3) / 2 = 2.0
    assert float(per_sample) == pytest.approx(2.0)


def test_a_four_times_larger_cohort_does_not_dominate():
    """The 2.5% / 10% extreme, which is the ratio the protocol actually draws."""
    small, large = 4, 16
    y_hat, y, mask = _batch([small, large], [10.0, 0.0])
    per_sample = _sample_equal_cohort_loss(y_hat, y, mask, "mae")
    # the small cohort's error must survive at full half-weight
    assert float(per_sample) == 5.0


def test_identical_errors_give_the_same_value_as_pooling():
    """A sanity control: the two only differ when the samples differ."""
    y_hat, y, mask = _batch([4, 16], [2.0, 2.0])
    pooled = _element_loss(y_hat, y, "mae")[mask].mean()
    assert float(_sample_equal_cohort_loss(y_hat, y, mask, "mae")) == float(pooled)


def test_empty_cohorts_contribute_nothing_rather_than_nan():
    y_hat, y, mask = _batch([0, 8], [0.0, 4.0])
    value = _sample_equal_cohort_loss(y_hat, y, mask, "mae")
    assert value is not None and torch.isfinite(value)
    assert float(value) == 4.0


def test_all_empty_returns_none():
    y_hat, y, mask = _batch([0, 0], [0.0, 0.0])
    assert _sample_equal_cohort_loss(y_hat, y, mask, "mae") is None


def test_gradient_flows_to_every_sample():
    y_hat, y, mask = _batch([4, 16], [1.0, 3.0])
    y_hat = y_hat.clone().requires_grad_(True)
    _sample_equal_cohort_loss(y_hat, y, mask, "mse").backward()
    per_sample_grad = y_hat.grad.abs().sum(dim=(1, 2, 3))
    assert torch.all(per_sample_grad > 0), "a sample received no gradient"


def test_mse_and_mae_agree_with_their_definitions():
    y_hat, y, mask = _batch([5], [2.0])
    assert float(_sample_equal_cohort_loss(y_hat, y, mask, "mae")) == 2.0
    assert float(_sample_equal_cohort_loss(y_hat, y, mask, "mse")) == 4.0
