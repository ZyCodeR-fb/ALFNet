import pytest
from scripts.collect_patch16_update import rank_values


def test_rank_includes_candidate_and_shares_exact_ties():
    assert rank_values([3., 1., 1., 2.]) == [3, 1, 1, 2]


def test_rank_uses_unrounded_values():
    assert rank_values([1.00004, 1.00003]) == [2, 1]


def test_rank_rejects_nonfinite():
    with pytest.raises(AssertionError):
        rank_values([1., float('nan')])
