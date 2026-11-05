"""Compare BM25, dense retrieval, and hybrid fusion on labeled queries."""

from __future__ import annotations

from semsearch.metrics import mean, ndcg_at_k, recall_at_k, reciprocal_rank
from semsearch.service import METHODS, SearchService
from semsearch.types import Query

METRIC_DECIMALS = 6


def _round_metric(value: float) -> float:
    """Six decimal places, via text so the checked-in report stays stable."""

    return float(f"{value:.6f}")


def evaluate_method(
    service: SearchService,
    queries: list[Query],
    qrels: dict[str, dict[str, int]],
    method: str,
    *,
    k: int,
) -> dict[str, float]:
    recalls: list[float] = []
    reciprocal_ranks: list[float] = []
    ndcgs: list[float] = []
    for query in queries:
        grades = qrels.get(query.id, {})
        relevant = {doc_id for doc_id, grade in grades.items() if grade > 0}
        if not relevant:
            continue
        retrieved = [doc_id for doc_id, _score in service.rank_documents(query.text, method)]
        recalls.append(recall_at_k(retrieved, relevant, k))
        reciprocal_ranks.append(reciprocal_rank(retrieved, relevant))
        ndcgs.append(ndcg_at_k(retrieved, grades, k))
    if not recalls:
        raise ValueError("No queries with positive relevance labels")
    return {
        f"recall@{k}": _round_metric(mean(recalls)),
        "mrr": _round_metric(mean(reciprocal_ranks)),
        f"ndcg@{k}": _round_metric(mean(ndcgs)),
    }


def build_report(
    service: SearchService,
    queries: list[Query],
    qrels: dict[str, dict[str, int]],
    *,
    k: int = 10,
    methods: tuple[str, ...] = METHODS,
    command: str | None = None,
) -> dict:
    if k <= 0:
        raise ValueError("k must be positive")
    embedder_flag = "hashing" if service.embedder.name.startswith("hashing") else "minilm"
    metrics = {method: evaluate_method(service, queries, qrels, method, k=k) for method in methods}
    return {
        "embedder": service.embedder.name,
        "embedder_dim": service.embedder.dim,
        "embedder_note": service.embedder.description,
        "command": command or f"python -m semsearch eval --k {k} --embedder {embedder_flag}",
        "k": k,
        "rrf_k": service.rrf_k,
        "num_documents": len(service.documents),
        "num_queries": len(queries),
        "relevant_grade_min": 1,
        "metrics": metrics,
    }


def format_markdown_table(report: dict) -> str:
    """Markdown table. Four decimal places, derived from the report values."""

    k = report["k"]
    recall_key = f"recall@{k}"
    ndcg_key = f"ndcg@{k}"
    lines = [
        f"| Method | Recall@{k} | MRR | nDCG@{k} |",
        "| --- | ---: | ---: | ---: |",
    ]
    for method, scores in report["metrics"].items():
        recall = f"{scores[recall_key]:.4f}"
        reciprocal = f"{scores['mrr']:.4f}"
        ndcg = f"{scores[ndcg_key]:.4f}"
        lines.append(f"| {method} | {recall} | {reciprocal} | {ndcg} |")
    return "\n".join(lines)


def report_caption(report: dict) -> str:
    return (
        f"Embedder that produced this table: `{report['embedder']}` "
        f"(dimension {report['embedder_dim']}). "
        f"Command: `{report['command']}`."
    )
