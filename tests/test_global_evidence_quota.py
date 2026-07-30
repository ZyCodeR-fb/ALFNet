"""Join/Fault evidence coverage must be uniform globally, not per cell.

v3 allocated k=1..L inside each (scenario, calendar regime) independently.
`_allocate_counts` breaks remainder ties by index, so every cell handed its
leftovers to the smallest k and the bias accumulated: measured on METR-LA the
Join counts ran 361 at k=1 down to 352 at k=12. The protocol claims uniform
coverage of evidence length, so that tilt is a real deviation.
"""

import numpy as np
import pytest

from data_provider.lcbt_protocol import (
    assign_global_evidence,
    global_evidence_quota,
)


@pytest.mark.parametrize("total", [0, 1, 11, 12, 13, 100, 4321])
@pytest.mark.parametrize("input_len", [12])
def test_quota_counts_differ_by_at_most_one(total, input_len):
    quota = global_evidence_quota(total, input_len)
    assert len(quota) == total
    if total == 0:
        return
    counts = np.bincount(quota, minlength=input_len + 1)[1:]
    assert counts.sum() == total
    assert int(counts.max() - counts.min()) <= 1


def test_quota_covers_every_k_once_the_total_allows_it():
    quota = global_evidence_quota(12, 12)
    assert sorted(quota.tolist()) == list(range(1, 13))


def test_rotation_moves_the_remainder_off_the_small_k():
    """The v3 failure mode, pinned: the leftovers must not always be k=1..r."""
    remainders = []
    for rotation in range(12):
        quota = global_evidence_quota(12 * 5 + 3, 12, rotation=rotation)
        counts = np.bincount(quota, minlength=13)[1:]
        remainders.append(tuple(np.flatnonzero(counts == counts.max()) + 1))
    # every rotation puts the three extras somewhere else
    assert len(set(remainders)) == 12
    assert remainders[0] == (1, 2, 3)
    assert remainders[5] == (6, 7, 8)


def test_assignment_preserves_the_global_counts():
    rng = np.random.default_rng(0)
    indices_by_regime = {
        "Weekday/AMPeak": np.arange(0, 40),
        "Weekday/Midday": np.arange(40, 111),
        "Weekend/Evening": np.arange(111, 124),
    }
    order, quota = assign_global_evidence(indices_by_regime, 12, rng)
    total = sum(len(v) for v in indices_by_regime.values())
    assert len(order) == total and len(quota) == total
    assert sorted(order.tolist()) == sorted(
        int(i) for bucket in indices_by_regime.values() for i in bucket
    )
    counts = np.bincount(quota, minlength=13)[1:]
    assert int(counts.max() - counts.min()) <= 1


def test_each_regime_gets_a_spread_of_evidence_lengths():
    """Interleaving, not block dealing: no regime may get only the short k."""
    rng = np.random.default_rng(1)
    indices_by_regime = {
        "A": np.arange(0, 120),
        "B": np.arange(120, 240),
        "C": np.arange(240, 360),
    }
    order, quota = assign_global_evidence(indices_by_regime, 12, rng)
    assigned = dict(zip(order.tolist(), quota.tolist()))
    for regime, bucket in indices_by_regime.items():
        values = [assigned[int(index)] for index in bucket]
        assert len(set(values)) == 12, f"{regime} did not see every evidence length"
        counts = np.bincount(values, minlength=13)[1:]
        # the two-dimensional quota makes this exact, not merely close
        assert int(counts.max() - counts.min()) <= 1, counts


def test_both_margins_are_exact_with_uneven_regimes():
    """The hard case: regime sizes that are not multiples of L.

    Each regime's own counts must differ by at most one, *and* the global counts
    must too. A single shuffled global quota gives only the second (measured
    range 8 across 12 lengths at 120 samples per regime); a per-regime quota
    without a cursor carried between regimes gives only the first.
    """
    rng = np.random.default_rng(5)
    indices_by_regime = {
        "A": np.arange(0, 37),
        "B": np.arange(37, 90),
        "C": np.arange(90, 121),
        "D": np.arange(121, 128),
    }
    order, quota = assign_global_evidence(indices_by_regime, 12, rng)
    assigned = dict(zip(order.tolist(), quota.tolist()))

    global_counts = np.bincount(quota, minlength=13)[1:]
    assert int(global_counts.max() - global_counts.min()) <= 1, global_counts

    for regime, bucket in indices_by_regime.items():
        values = [assigned[int(index)] for index in bucket]
        counts = np.bincount(values, minlength=13)[1:]
        assert int(counts.max() - counts.min()) <= 1, (regime, counts)


def test_empty_input_len_is_rejected():
    with pytest.raises(ValueError, match="input_len must be positive"):
        global_evidence_quota(10, 0)
