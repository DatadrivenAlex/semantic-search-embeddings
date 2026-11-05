import pytest

from semsearch.dense_index import DenseIndex
from semsearch.embedders import HashingEmbedder
from semsearch.types import Document


def test_dense_ranks_overlapping_document_first(tiny_documents) -> None:
    ranked = DenseIndex(tiny_documents, HashingEmbedder(dim=64)).rank("reset SSO password")
    assert ranked[0][0] == "d-password"
    assert ranked[0][1] > ranked[1][1]


def test_dense_scores_are_sorted_and_bounded(tiny_documents) -> None:
    ranked = DenseIndex(tiny_documents, HashingEmbedder(dim=64)).rank("gpu training quota")
    scores = [score for _doc_id, score in ranked]
    assert scores == sorted(scores, reverse=True)
    assert all(-1.0 - 1e-5 <= score <= 1.0 + 1e-5 for score in scores)


def test_dense_tie_break_is_doc_id() -> None:
    documents = [
        Document("b-doc", "Same", "alpha beta gamma"),
        Document("a-doc", "Same", "alpha beta gamma"),
    ]
    ranked = DenseIndex(documents, HashingEmbedder(dim=32)).rank("alpha beta gamma")
    assert [doc_id for doc_id, _score in ranked] == ["a-doc", "b-doc"]
    assert ranked[0][1] == pytest.approx(ranked[1][1])


def test_dense_rejects_empty_corpus() -> None:
    with pytest.raises(ValueError, match="at least one"):
        DenseIndex([], HashingEmbedder(dim=8))
