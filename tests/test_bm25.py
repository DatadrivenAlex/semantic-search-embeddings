import pytest

from semsearch.bm25_index import BM25Index
from semsearch.types import Document


def test_bm25_ranks_keyword_document_first(tiny_documents) -> None:
    ranked = BM25Index(tiny_documents).rank("reset SSO password")
    assert ranked[0][0] == "d-password"
    assert ranked[0][1] > ranked[1][1]


def test_bm25_unseen_terms_do_not_crash(tiny_documents) -> None:
    ranked = BM25Index(tiny_documents).rank("xylophone quarantine")
    assert len(ranked) == 3
    assert all(score == 0.0 for _doc_id, score in ranked)


def test_bm25_k_is_a_full_ranking(tiny_documents) -> None:
    ranked = BM25Index(tiny_documents).rank("vpn")
    assert [doc_id for doc_id, _score in ranked][0] == "d-vpn"
    assert len(ranked) == len(tiny_documents)


def test_bm25_tie_break_is_doc_id() -> None:
    documents = [
        Document("b-doc", "Alpha", "alpha beta"),
        Document("a-doc", "Alpha", "alpha beta"),
    ]
    ranked = BM25Index(documents).rank("alpha")
    assert [doc_id for doc_id, _score in ranked] == ["a-doc", "b-doc"]
    assert ranked[0][1] == pytest.approx(ranked[1][1])


def test_bm25_rejects_empty_corpus() -> None:
    with pytest.raises(ValueError, match="at least one"):
        BM25Index([])
