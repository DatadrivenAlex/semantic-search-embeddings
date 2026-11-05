"""Shared records for documents, queries, and hits."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Document:
    """One FAQ entry. `indexed_text` is what BM25 and the embedder see."""

    id: str
    title: str
    text: str

    @property
    def indexed_text(self) -> str:
        return f"{self.title}\n{self.text}"


@dataclass(frozen=True)
class Query:
    id: str
    text: str


@dataclass(frozen=True)
class Hit:
    doc_id: str
    title: str
    text: str
    score: float
    rank: int
