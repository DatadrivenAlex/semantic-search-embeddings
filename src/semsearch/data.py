"""Load the FAQ corpus, queries, and relevance judgments."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from semsearch.paths import default_data_dir
from semsearch.types import Document, Query


@dataclass(frozen=True)
class Dataset:
    documents: list[Document]
    queries: list[Query]
    qrels: dict[str, dict[str, int]]


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} is not valid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} must be a JSON object")
            rows.append(row)
    return rows


def _require_str(row: dict, key: str, path: Path) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path} row {row!r} needs a non-empty string field {key!r}")
    return value.strip()


def load_documents(path: Path) -> list[Document]:
    documents: list[Document] = []
    seen: set[str] = set()
    for row in _read_jsonl(path):
        doc_id = _require_str(row, "id", path)
        if doc_id in seen:
            raise ValueError(f"Duplicate document id {doc_id}")
        seen.add(doc_id)
        documents.append(
            Document(
                id=doc_id,
                title=_require_str(row, "title", path),
                text=_require_str(row, "text", path),
            )
        )
    if not documents:
        raise ValueError(f"{path} does not contain any documents")
    return documents


def load_queries(path: Path) -> list[Query]:
    queries: list[Query] = []
    seen: set[str] = set()
    for row in _read_jsonl(path):
        query_id = _require_str(row, "id", path)
        if query_id in seen:
            raise ValueError(f"Duplicate query id {query_id}")
        seen.add(query_id)
        queries.append(Query(id=query_id, text=_require_str(row, "text", path)))
    if not queries:
        raise ValueError(f"{path} does not contain any queries")
    return queries


def load_qrels(path: Path) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}
    for row in _read_jsonl(path):
        query_id = _require_str(row, "query_id", path)
        doc_id = _require_str(row, "doc_id", path)
        relevance = row.get("relevance")
        if isinstance(relevance, bool) or not isinstance(relevance, int) or relevance < 0:
            raise ValueError(f"Relevance for {query_id}/{doc_id} must be an integer >= 0")
        bucket = qrels.setdefault(query_id, {})
        if doc_id in bucket:
            raise ValueError(f"Duplicate qrel for {query_id}/{doc_id}")
        bucket[doc_id] = relevance
    if not qrels:
        raise ValueError(f"{path} does not contain any qrels")
    return qrels


def _validate_alignment(dataset: Dataset) -> None:
    doc_ids = {document.id for document in dataset.documents}
    query_ids = {query.id for query in dataset.queries}
    for query_id, grades in dataset.qrels.items():
        if query_id not in query_ids:
            raise ValueError(f"Qrel refers to unknown query {query_id}")
        for doc_id in grades:
            if doc_id not in doc_ids:
                raise ValueError(f"Qrel refers to unknown document {doc_id}")
    for query in dataset.queries:
        grades = dataset.qrels.get(query.id, {})
        if not any(grade > 0 for grade in grades.values()):
            raise ValueError(f"Query {query.id} has no positive relevance judgment")


def load_dataset(data_dir: Path | None = None) -> Dataset:
    directory = data_dir or default_data_dir()
    dataset = Dataset(
        documents=load_documents(directory / "corpus.jsonl"),
        queries=load_queries(directory / "queries.jsonl"),
        qrels=load_qrels(directory / "qrels.jsonl"),
    )
    _validate_alignment(dataset)
    return dataset
