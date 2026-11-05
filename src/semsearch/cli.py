"""Command line search and evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from semsearch.data import load_dataset
from semsearch.evaluate import build_report, format_markdown_table, report_caption
from semsearch.service import METHODS, build_default_service


def _add_embedder_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--embedder",
        default="hashing",
        choices=("hashing", "minilm"),
        help="hashing is offline. minilm downloads all-MiniLM-L6-v2 on first use.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Sentence-transformers model name or local path. Only with --embedder minilm.",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="semsearch",
        description="Search a small technical FAQ with BM25, dense retrieval, or hybrid fusion.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    search = subparsers.add_parser("search", help="Rank FAQ entries for one query")
    search.add_argument("query", help="Natural language query")
    search.add_argument("--method", default="hybrid", choices=METHODS)
    search.add_argument("--k", type=int, default=5)
    _add_embedder_args(search)
    search.add_argument("--json", action="store_true", dest="as_json")
    search.set_defaults(func=_cmd_search)

    evaluate = subparsers.add_parser("eval", help="Compare methods on the labeled queries")
    evaluate.add_argument("--k", type=int, default=10)
    _add_embedder_args(evaluate)
    evaluate.add_argument("--json", action="store_true", dest="as_json")
    evaluate.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write the JSON report to this path",
    )
    evaluate.set_defaults(func=_cmd_eval)
    return parser


def _cmd_search(args: argparse.Namespace) -> int:
    if not args.query.strip():
        print("error: query is empty", file=sys.stderr)
        return 2
    if args.k <= 0:
        print("error: k must be positive", file=sys.stderr)
        return 2
    service = build_default_service(embedder_name=args.embedder, model_name=args.model)
    hits = service.search(args.query, method=args.method, k=args.k)
    payload = {
        "query": args.query,
        "method": args.method,
        "embedder": service.embedder.name,
        "k": args.k,
        "hits": [
            {
                "rank": hit.rank,
                "doc_id": hit.doc_id,
                "title": hit.title,
                "score": hit.score,
                "text": hit.text,
            }
            for hit in hits
        ],
    }
    if args.as_json:
        print(json.dumps(payload, indent=2))
        return 0
    print(f"embedder: {service.embedder.name}")
    print(f"method: {args.method}")
    if not hits:
        print("No hits.")
        return 0
    for hit in hits:
        print(f"{hit.rank}. {hit.doc_id}  score={hit.score:.4f}")
        print(f"   {hit.title}")
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    if args.k <= 0:
        print("error: k must be positive", file=sys.stderr)
        return 2
    dataset = load_dataset()
    service = build_default_service(embedder_name=args.embedder, model_name=args.model)
    command = f"python -m semsearch eval --k {args.k} --embedder {args.embedder}"
    if args.model:
        command += f" --model {args.model}"
    report = build_report(
        service,
        dataset.queries,
        dataset.qrels,
        k=args.k,
        command=command,
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.as_json:
        print(json.dumps(report, indent=2))
    else:
        print(report_caption(report))
        print(report["embedder_note"])
        print(
            f"Documents: {report['num_documents']}. "
            f"Queries: {report['num_queries']}. "
            f"Cutoff k={report['k']}. RRF k={report['rrf_k']}."
        )
        print(format_markdown_table(report))
        if args.output is not None:
            print(f"Wrote {args.output}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
