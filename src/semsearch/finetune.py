"""Build query/document pairs and, optionally, fine-tune a bi-encoder.

MultipleNegativesRankingLoss uses other positives in the batch as negatives.
The pair builder is offline. Training imports sentence-transformers only when
called, and CI does not call it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from semsearch.data import Dataset, load_dataset
from semsearch.embedders import DEFAULT_NEURAL_MODEL

_ML_INSTALL_HINT = "Fine-tuning needs the optional ml extra: pip install -e '.[ml]'"


def build_training_pairs(
    dataset: Dataset,
    *,
    min_relevance: int = 1,
) -> list[dict[str, str]]:
    """Positive pairs only. In-batch negatives come from the other rows."""

    if min_relevance < 1:
        raise ValueError("min_relevance must be at least 1")
    documents = {document.id: document for document in dataset.documents}
    queries = {query.id: query for query in dataset.queries}
    pairs: list[dict[str, str]] = []
    for query_id in sorted(dataset.qrels):
        for doc_id in sorted(dataset.qrels[query_id]):
            grade = dataset.qrels[query_id][doc_id]
            if grade < min_relevance:
                continue
            pairs.append(
                {
                    "query_id": query_id,
                    "doc_id": doc_id,
                    "query": queries[query_id].text,
                    "positive": documents[doc_id].indexed_text,
                }
            )
    return pairs


def resolve_batch_size(num_pairs: int, requested: int) -> int:
    if num_pairs < 2:
        raise ValueError(
            "MultipleNegativesRankingLoss needs at least 2 pairs so the batch has a negative."
        )
    if requested < 1:
        raise ValueError("batch size must be positive")
    return min(requested, num_pairs)


def write_pairs(pairs: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for pair in pairs:
            handle.write(json.dumps(pair, ensure_ascii=True) + "\n")


def train_biencoder(
    pairs: list[dict[str, str]],
    output_dir: Path,
    *,
    model_name: str = DEFAULT_NEURAL_MODEL,
    epochs: int = 1,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
) -> Path:
    """Fine-tune with MultipleNegativesRankingLoss and save the bi-encoder."""

    if epochs < 1:
        raise ValueError("epochs must be positive")
    batch_size = resolve_batch_size(len(pairs), batch_size)
    try:
        from datasets import Dataset as HFDataset
        from sentence_transformers import (
            SentenceTransformer,
            SentenceTransformerTrainer,
            SentenceTransformerTrainingArguments,
        )
        from sentence_transformers.losses import MultipleNegativesRankingLoss
    except ImportError as exc:
        raise RuntimeError(_ML_INSTALL_HINT) from exc

    model = SentenceTransformer(model_name)
    train_dataset = HFDataset.from_dict(
        {
            "anchor": [pair["query"] for pair in pairs],
            "positive": [pair["positive"] for pair in pairs],
        }
    )
    loss = MultipleNegativesRankingLoss(model)
    output_dir.mkdir(parents=True, exist_ok=True)
    args = SentenceTransformerTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_ratio=0.1,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        seed=7,
    )
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        loss=loss,
    )
    trainer.train()
    model.save(str(output_dir))
    return output_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="finetune",
        description=(
            "Write query/positive pairs from the FAQ labels, or fine-tune "
            "all-MiniLM-L6-v2 with MultipleNegativesRankingLoss."
        ),
    )
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--pairs-output", type=Path, default=Path("outputs/train_pairs.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/finetune"))
    parser.add_argument("--model", default=DEFAULT_NEURAL_MODEL)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--min-relevance", type=int, default=1)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write pairs and exit. Does not import sentence-transformers.",
    )
    args = parser.parse_args(argv)
    try:
        dataset = load_dataset(args.data_dir)
        pairs = build_training_pairs(dataset, min_relevance=args.min_relevance)
        write_pairs(pairs, args.pairs_output)
        print(f"Wrote {len(pairs)} pairs to {args.pairs_output}")
        if args.dry_run:
            return 0
        saved = train_biencoder(
            pairs,
            args.output_dir,
            model_name=args.model,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Saved bi-encoder to {saved}")
    return 0
