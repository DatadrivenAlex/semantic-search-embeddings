import json
from pathlib import Path

import pytest

from semsearch.data import load_dataset


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _write_toy(
    directory: Path,
    *,
    qrels: list[dict] | None = None,
    queries: list[dict] | None = None,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    _write_jsonl(
        directory / "corpus.jsonl",
        [
            {"id": "d1", "title": "One", "text": "alpha"},
            {"id": "d2", "title": "Two", "text": "beta"},
        ],
    )
    _write_jsonl(
        directory / "queries.jsonl",
        queries
        if queries is not None
        else [
            {"id": "q1", "text": "alpha"},
            {"id": "q2", "text": "beta"},
        ],
    )
    _write_jsonl(
        directory / "qrels.jsonl",
        qrels
        if qrels is not None
        else [
            {"query_id": "q1", "doc_id": "d1", "relevance": 2},
            {"query_id": "q2", "doc_id": "d2", "relevance": 1},
        ],
    )


def test_bundled_dataset_is_aligned() -> None:
    dataset = load_dataset()
    assert len(dataset.documents) == 40
    assert len(dataset.queries) == 20
    doc_ids = {document.id for document in dataset.documents}
    query_ids = {query.id for query in dataset.queries}
    assert len(doc_ids) == 40
    assert len(query_ids) == 20
    labeled = 0
    for query in dataset.queries:
        grades = dataset.qrels[query.id]
        assert any(grade > 0 for grade in grades.values())
        assert set(grades) <= doc_ids
        labeled += len(grades)
    assert labeled == 41


def test_duplicate_doc_id_rejected(tmp_path: Path) -> None:
    _write_toy(tmp_path)
    with (tmp_path / "corpus.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"id": "d1", "title": "Again", "text": "alpha"}) + "\n")
    with pytest.raises(ValueError, match="Duplicate document id"):
        load_dataset(tmp_path)


def test_dangling_qrel_rejected(tmp_path: Path) -> None:
    _write_toy(
        tmp_path,
        qrels=[{"query_id": "q1", "doc_id": "missing", "relevance": 1}],
    )
    with pytest.raises(ValueError, match="unknown document"):
        load_dataset(tmp_path)


def test_query_without_positive_rejected(tmp_path: Path) -> None:
    _write_toy(
        tmp_path,
        qrels=[
            {"query_id": "q1", "doc_id": "d1", "relevance": 2},
            {"query_id": "q2", "doc_id": "d2", "relevance": 0},
        ],
    )
    with pytest.raises(ValueError, match="no positive"):
        load_dataset(tmp_path)


def test_negative_relevance_rejected(tmp_path: Path) -> None:
    _write_toy(
        tmp_path,
        qrels=[{"query_id": "q1", "doc_id": "d1", "relevance": -1}],
    )
    with pytest.raises(ValueError, match="Relevance"):
        load_dataset(tmp_path)
