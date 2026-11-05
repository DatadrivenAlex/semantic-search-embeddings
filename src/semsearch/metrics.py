"""Recall@k, MRR, and nDCG@k.

A document is relevant for recall and MRR when its grade is greater than 0.
nDCG uses the raw grades. The ideal gain is the graded documents sorted
best-first, so a relevant document missing from the top k lowers nDCG.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


def recall_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    if not relevant:
        raise ValueError("relevant set is empty")
    found = set(retrieved[:k]) & relevant
    return len(found) / len(relevant)


def reciprocal_rank(retrieved: Sequence[str], relevant: set[str]) -> float:
    """1 / rank of the first relevant document in the full list, else 0."""

    if not relevant:
        raise ValueError("relevant set is empty")
    for index, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1.0 / index
    return 0.0


def _dcg(gains: Sequence[float], k: int) -> float:
    total = 0.0
    for index, gain in enumerate(gains[:k], start=1):
        total += (2.0**gain - 1.0) / math.log2(index + 1)
    return total


def ndcg_at_k(retrieved: Sequence[str], grades: Mapping[str, float], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    ideal = sorted((float(grade) for grade in grades.values() if grade > 0), reverse=True)
    ideal_dcg = _dcg(ideal, k)
    if ideal_dcg == 0.0:
        raise ValueError("grades have no positive relevance")
    gains = [float(grades.get(doc_id, 0.0)) for doc_id in retrieved[:k]]
    return _dcg(gains, k) / ideal_dcg


def mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty sequence")
    return sum(values) / len(values)
