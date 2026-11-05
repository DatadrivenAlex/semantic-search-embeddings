"""Search over one corpus with BM25, dense retrieval, or both."""

from __future__ import annotations

import os

from semsearch.bm25_index import BM25Index
from semsearch.data import load_documents
from semsearch.dense_index import DenseIndex
from semsearch.embedders import Embedder, build_embedder
from semsearch.fusion import reciprocal_rank_fusion
from semsearch.paths import default_data_dir
from semsearch.types import Document, Hit

METHODS = ("bm25", "dense", "hybrid")


class SearchService:
    def __init__(
        self,
        documents: list[Document],
        embedder: Embedder,
        *,
        rrf_k: int = 60,
    ) -> None:
        if not documents:
            raise ValueError("SearchService needs at least one document")
        if len({document.id for document in documents}) != len(documents):
            raise ValueError("Document ids must be unique")
        self.documents = list(documents)
        self.embedder = embedder
        self.rrf_k = rrf_k
        self.by_id = {document.id: document for document in self.documents}
        self.bm25 = BM25Index(self.documents)
        self.dense = DenseIndex(self.documents, embedder)

    def rank_documents(self, query: str, method: str) -> list[tuple[str, float]]:
        key = method.strip().lower()
        if key == "bm25":
            return self.bm25.rank(query)
        if key == "dense":
            return self.dense.rank(query)
        if key == "hybrid":
            lexical = [doc_id for doc_id, _score in self.bm25.rank(query)]
            dense = [doc_id for doc_id, _score in self.dense.rank(query)]
            return reciprocal_rank_fusion([lexical, dense], rrf_k=self.rrf_k)
        raise ValueError(f"Unknown method {method!r}. Choose one of {', '.join(METHODS)}.")

    def search(self, query: str, method: str = "hybrid", k: int = 5) -> list[Hit]:
        if k <= 0:
            raise ValueError("k must be positive")
        ranked = self.rank_documents(query, method)[:k]
        hits: list[Hit] = []
        for rank, (doc_id, score) in enumerate(ranked, start=1):
            document = self.by_id[doc_id]
            hits.append(
                Hit(
                    doc_id=document.id,
                    title=document.title,
                    text=document.text,
                    score=float(score),
                    rank=rank,
                )
            )
        return hits


def build_default_service(
    *,
    embedder_name: str | None = None,
    model_name: str | None = None,
) -> SearchService:
    """Index the bundled FAQ. The embedder defaults to the offline hasher."""

    name = (
        embedder_name
        if embedder_name is not None
        else os.environ.get("SEMSEARCH_EMBEDDER", "hashing")
    )
    documents = load_documents(default_data_dir() / "corpus.jsonl")
    embedder = build_embedder(name, model_name=model_name)
    return SearchService(documents, embedder)
