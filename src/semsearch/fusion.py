"""Reciprocal rank fusion.

Cormack, Clarke, and Buettcher, SIGIR 2009. A document at 1-based rank r
contributes 1 / (k + r). k = 60 is the usual constant. Raw BM25 scores and
cosine scores are never compared directly.
"""

from __future__ import annotations


def reciprocal_rank_fusion(
    rankings: list[list[str]],
    *,
    rrf_k: int = 60,
) -> list[tuple[str, float]]:
    """Fuse ranked id lists. Duplicate ids inside one list keep their first rank."""

    if rrf_k <= 0:
        raise ValueError("rrf_k must be positive")
    scores: dict[str, float] = {}
    for ranking in rankings:
        seen: set[str] = set()
        for rank, doc_id in enumerate(ranking, start=1):
            if doc_id in seen:
                continue
            seen.add(doc_id)
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))
