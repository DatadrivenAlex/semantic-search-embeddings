import math

import pytest

from semsearch.fusion import reciprocal_rank_fusion
from semsearch.service import SearchService


def test_search_methods_return_password_doc_first(tiny_service: SearchService) -> None:
    for method in ("bm25", "dense", "hybrid"):
        hits = tiny_service.search("reset SSO password", method=method, k=3)
        assert [hit.doc_id for hit in hits][0] == "d-password"
        assert [hit.rank for hit in hits] == [1, 2, 3]
        assert len(hits) == 3


def test_unknown_method(tiny_service: SearchService) -> None:
    with pytest.raises(ValueError, match="Unknown method"):
        tiny_service.search("password", method="splade")


def test_hybrid_matches_manual_rrf(tiny_service: SearchService) -> None:
    query = "password vpn gpu"
    lexical = [doc_id for doc_id, _score in tiny_service.bm25.rank(query)]
    dense = [doc_id for doc_id, _score in tiny_service.dense.rank(query)]
    expected = reciprocal_rank_fusion([lexical, dense], rrf_k=tiny_service.rrf_k)
    hits = tiny_service.search(query, method="hybrid", k=3)
    assert [hit.doc_id for hit in hits] == [doc_id for doc_id, _score in expected[:3]]
    assert hits[0].score == pytest.approx(expected[0][1])


def test_k_must_be_positive(tiny_service: SearchService) -> None:
    with pytest.raises(ValueError, match="k must be positive"):
        tiny_service.search("password", k=0)


def test_results_only_contain_known_ids(tiny_service: SearchService) -> None:
    known = set(tiny_service.by_id)
    for method in ("bm25", "dense", "hybrid"):
        hits = tiny_service.search("quota", method=method, k=5)
        assert {hit.doc_id for hit in hits} <= known
        assert len(hits) == 3


def test_empty_query_still_returns_k(tiny_service: SearchService) -> None:
    hits = tiny_service.search("   ", method="hybrid", k=2)
    assert len(hits) == 2
    assert all(math.isfinite(hit.score) for hit in hits)


def test_duplicate_document_ids_rejected(tiny_documents) -> None:
    from semsearch.embedders import HashingEmbedder

    duplicated = [tiny_documents[0], tiny_documents[0]]
    with pytest.raises(ValueError, match="unique"):
        SearchService(duplicated, HashingEmbedder(dim=16))
