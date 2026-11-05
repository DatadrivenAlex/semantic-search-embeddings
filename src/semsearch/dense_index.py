"""Exact dense retrieval with a FAISS inner-product index."""

from __future__ import annotations

import faiss
import numpy as np

from semsearch.embedders import Embedder
from semsearch.types import Document

# Exact search on a 40-document FAQ does not need a thread pool, and a single
# thread keeps score ties from depending on reduction order.
faiss.omp_set_num_threads(1)


class DenseIndex:
    """IndexFlatIP over L2-normalized embeddings. Inner product is cosine."""

    def __init__(self, documents: list[Document], embedder: Embedder) -> None:
        if not documents:
            raise ValueError("Dense index needs at least one document")
        self.documents = documents
        self.embedder = embedder
        matrix = np.ascontiguousarray(
            embedder.encode([document.indexed_text for document in documents]),
            dtype=np.float32,
        )
        if matrix.shape != (len(documents), embedder.dim):
            raise ValueError(
                f"Embedder returned shape {matrix.shape}, expected {(len(documents), embedder.dim)}"
            )
        self.index = faiss.IndexFlatIP(embedder.dim)
        self.index.add(matrix)

    def rank(self, query: str) -> list[tuple[str, float]]:
        query_matrix = np.ascontiguousarray(self.embedder.encode([query]), dtype=np.float32)
        scores, indices = self.index.search(query_matrix, len(self.documents))
        hits: list[tuple[str, float]] = []
        for score, index in zip(scores[0].tolist(), indices[0].tolist(), strict=True):
            if index < 0:
                continue
            hits.append((self.documents[index].id, float(score)))
        hits.sort(key=lambda item: (-item[1], item[0]))
        return hits
