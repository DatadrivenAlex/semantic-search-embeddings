import math

import pytest

from semsearch.metrics import mean, ndcg_at_k, recall_at_k, reciprocal_rank


def test_recall_partial_and_perfect() -> None:
    retrieved = ["a", "b", "c", "d"]
    relevant = {"b", "d", "e"}
    assert recall_at_k(retrieved, relevant, 3) == pytest.approx(1 / 3)
    assert recall_at_k(retrieved, relevant, 4) == pytest.approx(2 / 3)
    assert recall_at_k(["b", "d", "e"], relevant, 3) == pytest.approx(1.0)


def test_mrr_rank_positions() -> None:
    assert reciprocal_rank(["a", "b", "c"], {"b", "c"}) == pytest.approx(0.5)
    assert reciprocal_rank(["b"], {"b"}) == pytest.approx(1.0)
    assert reciprocal_rank(["a", "c"], {"b"}) == 0.0


def test_ndcg_hand_computed_and_perfect() -> None:
    grades = {"docA": 2, "docB": 1, "docC": 0}
    # DCG of [1, 2] over IDCG of [2, 1], cutoff 2.
    # 1/log2(2) + 3/log2(3)  divided by  3/log2(2) + 1/log2(3)
    log2_3 = math.log2(3)
    expected = (1.0 + 3.0 / log2_3) / (3.0 + 1.0 / log2_3)
    assert ndcg_at_k(["docB", "docA"], grades, 2) == pytest.approx(expected)
    assert ndcg_at_k(["docB", "docA"], grades, 2) == pytest.approx(0.7967075810)
    assert ndcg_at_k(["docA", "docB", "docC"], grades, 3) == pytest.approx(1.0)


def test_metrics_reject_bad_inputs() -> None:
    with pytest.raises(ValueError, match="empty"):
        recall_at_k(["a"], set(), 1)
    with pytest.raises(ValueError, match="empty"):
        reciprocal_rank(["a"], set())
    with pytest.raises(ValueError, match="positive"):
        ndcg_at_k(["a"], {"a": 0}, 1)
    with pytest.raises(ValueError, match="positive"):
        recall_at_k(["a"], {"a"}, 0)
    with pytest.raises(ValueError, match="empty"):
        mean([])
