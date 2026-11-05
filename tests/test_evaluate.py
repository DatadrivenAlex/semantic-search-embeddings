import json
from pathlib import Path

from semsearch.data import load_dataset
from semsearch.evaluate import (
    build_report,
    evaluate_method,
    format_markdown_table,
    report_caption,
)
from semsearch.service import build_default_service
from semsearch.types import Query

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = REPO_ROOT / "evals" / "hashing_k10.json"


class _FixedRanker:
    """Returns the same ranking for every query and method."""

    def rank_documents(self, query: str, method: str) -> list[tuple[str, float]]:
        del query, method
        return [("a", 1.0), ("b", 0.5), ("c", 0.1)]


def test_evaluate_macro_averages_labeled_queries() -> None:
    queries = [Query("q1", "one"), Query("q2", "two")]
    qrels = {"q1": {"a": 2}, "q2": {"c": 1}}
    # q1's relevant doc is rank 1. q2's relevant doc is rank 3, so recall@2 misses it.
    # MRR still walks the full ranking, so q2 contributes 1/3.
    metrics = evaluate_method(_FixedRanker(), queries, qrels, "bm25", k=2)  # type: ignore[arg-type]
    assert metrics["recall@2"] == 0.5
    assert metrics["mrr"] == float(f"{(1.0 + 1.0 / 3.0) / 2.0:.6f}")


def test_markdown_table_uses_four_decimals() -> None:
    report = {
        "k": 10,
        "metrics": {
            "bm25": {"recall@10": 0.5, "mrr": 1.0, "ndcg@10": 0.796708},
        },
    }
    table = format_markdown_table(report)
    assert "| bm25 | 0.5000 | 1.0000 | 0.7967 |" in table


def test_checked_in_hashing_report_matches_fresh_run() -> None:
    saved = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    dataset = load_dataset()
    service = build_default_service(embedder_name="hashing")
    fresh = build_report(
        service,
        dataset.queries,
        dataset.qrels,
        k=saved["k"],
        command=saved["command"],
    )
    assert saved["embedder"] == "hashing-md5-v1"
    assert saved["embedder_dim"] == 384
    assert "not a neural semantic model" in saved["embedder_note"]
    assert fresh == saved


def test_readme_quotes_the_checked_in_run() -> None:
    saved = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert format_markdown_table(saved) in readme
    assert report_caption(saved) in readme
    assert saved["embedder_note"] in readme
