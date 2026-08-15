"""Two of the five propagation "operators" are the same matrix.

``SUPPORT_NAMES`` lists five supports, but

    weighted_mean     = A / rowsum.clamp_min(1e-6)        (lifecycle_moge.py)
    forward_diffusion = diag(1/rowsum) @ A                (gnn.py)

are the same operator wherever the row sum clears the clamp. Measured on the
real graphs the largest disagreement is 5.96e-08 on METR-LA and PEMS-BAY -- a
float path difference, not a different operator -- and on PEMS03/PEMS04 it is
confined to 2 of 358 and 8 of 307 rows, every one of them a row whose sum falls
under the 1e-6 clamp.

This matters for what may be claimed. A router's usage mass is not identifiable
between two labels that name one matrix, so a per-expert usage split reported
across ``weighted_mean`` and ``forward_diffusion`` carries no information, and
"five heterogeneous propagation operators" overstates the operator set on every
dataset.

The tests are algebraic and run on synthetic graphs, so they need no file from
``data/`` (which is gitignored) and hold for any future dataset. Removing or
redefining the duplicate is a model change requiring a v4 retrain -- these tests
are expected to be *updated* then, deliberately, not silently.
"""

import numpy as np
import pytest
import torch

from layers.gnn import calculate_random_walk_matrix
from layers.lifecycle_moge import SUPPORT_NAMES, build_lifecycle_supports

WEIGHTED = SUPPORT_NAMES.index("weighted_mean")
FORWARD = SUPPORT_NAMES.index("forward_diffusion")
BACKWARD = SUPPORT_NAMES.index("backward_diffusion")


def _supports(adjacency):
    tensor = torch.as_tensor(adjacency, dtype=torch.float32)
    return build_lifecycle_supports(
        tensor,
        calculate_random_walk_matrix(tensor),
        calculate_random_walk_matrix(tensor.T),
    )


def _random_directed(n, seed, scale=1.0):
    rng = np.random.default_rng(seed)
    adjacency = rng.random((n, n)) * scale
    adjacency[rng.random((n, n)) < 0.7] = 0.0
    np.fill_diagonal(adjacency, 0.0)
    # guarantee every row clears the clamp so the identity is exact
    for row in range(n):
        if adjacency[row].sum() < 1e-3:
            adjacency[row, (row + 1) % n] = 1.0
    return adjacency


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_weighted_mean_is_the_forward_random_walk(seed):
    """The duplication, on directed graphs with well-conditioned rows."""
    supports = _supports(_random_directed(24, seed))
    difference = float((supports[WEIGHTED] - supports[FORWARD]).abs().max())
    assert difference < 1e-6, (
        f"weighted_mean and forward_diffusion differ by {difference:.3e}; if the "
        "operator set was deliberately redefined, update this test and retrain"
    )


def test_the_five_labels_carry_at_most_four_distinct_matrices():
    supports = _supports(_random_directed(24, 11))
    distinct = []
    for index in range(len(SUPPORT_NAMES)):
        if not any(float((supports[index] - supports[other]).abs().max()) < 1e-6
                   for other in distinct):
            distinct.append(index)
    assert len(distinct) <= 4, "five labels should not yield five distinct matrices"
    assert len(distinct) == 4, (
        "expected exactly four distinct operators on a directed graph: "
        "identity, binary mean, weighted/forward, backward"
    )


def test_a_symmetric_graph_collapses_to_three_operators():
    """Undirected input makes forward and backward identical as well.

    The PEMS converters symmetrise, so this is the case those datasets are in --
    PEMS08 ships one connected symmetric graph and reaches three.
    """
    adjacency = _random_directed(24, 7)
    adjacency = np.maximum(adjacency, adjacency.T)
    supports = _supports(adjacency)
    assert float((supports[FORWARD] - supports[BACKWARD]).abs().max()) == 0.0
    distinct = []
    for index in range(len(SUPPORT_NAMES)):
        if not any(float((supports[index] - supports[other]).abs().max()) < 1e-6
                   for other in distinct):
            distinct.append(index)
    assert len(distinct) == 3


def test_the_only_disagreement_is_the_clamp_on_vanishing_rows():
    """Where the two do differ, it is the 1e-6 clamp, not a design difference.

    A row whose weights sum below the clamp is under-normalised by
    ``weighted_mean`` and correctly normalised by the random walk. That is an
    implementation artifact and must not be presented as a fourth operator.
    """
    adjacency = _random_directed(12, 5)
    adjacency[3] = 0.0
    adjacency[3, 4] = 1e-9          # sums to far below the clamp
    supports = _supports(adjacency)

    per_row = (supports[WEIGHTED] - supports[FORWARD]).abs().max(dim=1).values
    offenders = torch.nonzero(per_row > 1e-6).flatten().tolist()
    assert offenders == [3], f"expected only the vanishing row to differ, got {offenders}"

    row_sums = torch.as_tensor(adjacency, dtype=torch.float32).sum(1)
    assert 0 < float(row_sums[3]) < 1e-6
    # the random walk normalises it to 1.0; the clamped version does not
    assert float(supports[FORWARD][3, 4]) == pytest.approx(1.0)
    assert float(supports[WEIGHTED][3, 4]) < 1e-2


def test_router_usage_is_unidentifiable_between_duplicated_labels():
    """Why the duplication is a reporting problem and not only a compute one.

    With two labels naming one matrix, any split of gate mass between them
    leaves the aggregated output unchanged, so the observed split is not a
    measurement of anything.
    """
    supports = _supports(_random_directed(16, 3))
    features = torch.randn(2, 16, 5)
    propagated = torch.einsum("enm,bmd->bend", supports, features)

    total = 0.6
    outputs = []
    for weighted_share in (0.0, 0.3, 0.6):
        gates = torch.zeros(len(SUPPORT_NAMES))
        gates[WEIGHTED] = weighted_share
        gates[FORWARD] = total - weighted_share
        outputs.append(torch.einsum("bend,e->bnd", propagated, gates))
    for other in outputs[1:]:
        assert torch.allclose(outputs[0], other, atol=1e-5), (
            "moving gate mass between the duplicated labels changed the output, "
            "which would mean they are not duplicates after all"
        )
