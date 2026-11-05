import pytest

from semsearch.fusion import reciprocal_rank_fusion


def test_rrf_known_scores_and_order() -> None:
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "c", "a"]], rrf_k=60)
    assert [doc_id for doc_id, _score in fused] == ["b", "a", "c"]
    assert fused[0][1] == pytest.approx(1 / 62 + 1 / 61)
    assert fused[1][1] == pytest.approx(1 / 61 + 1 / 63)
    assert fused[2][1] == pytest.approx(1 / 63 + 1 / 62)


def test_rrf_tie_breaks_on_doc_id() -> None:
    fused = reciprocal_rank_fusion([["a"], ["b"]], rrf_k=60)
    assert [doc_id for doc_id, _score in fused] == ["a", "b"]
    assert fused[0][1] == pytest.approx(fused[1][1])


def test_rrf_keeps_first_position_of_a_duplicate() -> None:
    fused = reciprocal_rank_fusion([["a", "a", "b"]], rrf_k=60)
    scores = dict(fused)
    assert scores["a"] == pytest.approx(1 / 61)
    assert scores["b"] == pytest.approx(1 / 63)


def test_rrf_empty() -> None:
    assert reciprocal_rank_fusion([[], []]) == []


def test_rrf_rejects_non_positive_k() -> None:
    with pytest.raises(ValueError, match="rrf_k"):
        reciprocal_rank_fusion([["a"]], rrf_k=0)
