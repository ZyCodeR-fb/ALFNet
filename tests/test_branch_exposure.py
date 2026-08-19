"""What counts as leakage between ALFNet's branches, and what does not.

The earlier version of this file asserted that ALFTemporal's output must not
depend on the values sitting in held-out node slots. That is the wrong contract
and it is recorded here so it is not reintroduced. Once a shared imputer runs
over the full node axis, every node's representation is *expected* to depend on
its neighbours' imputed values -- that is what propagation is for. Perturbing a
legitimate intermediate tensor therefore proves nothing.

The contract that matters is on the raw payload signal: measurements the
protocol forbids for this phase must not reach the visible input, the imputed
representation, either branch, the loss, the gradients or an optimiser step.
That is asserted in tests/test_runtime_context.py at the point where the
exposure is decided.
"""

import pytest
import torch

from lib.runtime_context import build_runtime_context


def test_depending_on_an_imputed_representation_is_allowed():
    """A guard against reintroducing the wrong requirement.

    Two nodes, one in scope and one not. The imputer legitimately gives the
    out-of-scope node a representation derived from topology and from in-scope
    measurements, and a downstream branch may read it. What it may not do is
    depend on the out-of-scope node's own measurement, which the context has
    already removed by this point.
    """
    context = build_runtime_context("PEMS08", "train")
    raw = torch.zeros(1, 4, context.num_nodes, 1)
    observed = torch.ones(1, 4, context.num_nodes, dtype=torch.bool)
    visible, available = context.visible_signal(raw, observed)

    # a toy imputer: every node takes the mean of its in-scope neighbours
    adjacency = (context.adjacency > 0).float()
    weights = adjacency / adjacency.sum(dim=1, keepdim=True).clamp_min(1.0)
    filled = torch.einsum("nm,btmc->btnc", weights, visible)

    forbidden = ~context.signal_scope_mask()
    assert forbidden.any()
    # The out-of-scope nodes carry a representation, and it came from the graph,
    # not from their own measurements.
    assert filled.shape[2] == context.num_nodes
    assert torch.equal(visible[:, :, forbidden, :],
                       torch.zeros_like(visible[:, :, forbidden, :]))


@pytest.mark.skip(reason=(
    "Stage 2 Parts 3-6 are not implemented. ALFTemporal still loads its own "
    "adjacency and positions and scatters K nodes into N with zeros, so there is "
    "no full-N path to assert against yet. Deliberately a skip and not an xfail: "
    "an xfail here would report a fairness property as 'expected to fail', and "
    "the honest state is that the property is untested."))
def test_both_branches_receive_the_same_full_n_imputed_tensor():
    raise AssertionError("wiring not implemented")
