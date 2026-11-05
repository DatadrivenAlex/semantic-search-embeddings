"""Lexical baseline using Okapi BM25 from rank_bm25."""

from __future__ import annotations

import numpy as np
from rank_bm25 import BM25Okapi

from semsearch.tokenize import tokenize
from semsearch.types import Document


class BM25Index:
    """BM25 over `Document.indexed_text`. Ties break on document id."""

    def __init__(self, documents: list[Document], *, k1: float = 1.5, b: float = 0.75) -> None:
        if not documents:
            raise ValueError("BM25 needs at least one document")
        self.documents = documents
        tokenized = [tokenize(document.indexed_text) or ["__empty__"] for document in documents]
        self._bm25 = BM25Okapi(tokenized, k1=k1, b=b)

    def rank(self, query: str) -> list[tuple[str, float]]:
        scores = np.asarray(self._bm25.get_scores(tokenize(query)), dtype=np.float64)
        order = sorted(
            range(len(self.documents)),
            key=lambda index: (-float(scores[index]), self.documents[index].id),
        )
        return [(self.documents[index].id, float(scores[index])) for index in order]
