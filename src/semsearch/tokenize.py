"""Tokenizer shared by BM25 and the hashing embedder.

There is no stemmer. A short stopword list is removed so words such as
"the" and "how" do not dominate the feature hash. BM25 also sees this
tokenization, which keeps the two methods comparable.
"""

from __future__ import annotations

import re

_TOKEN = re.compile(r"[a-z0-9]+")

# Closed class words plus the question words that start almost every FAQ query.
STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "can",
        "do",
        "for",
        "from",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "my",
        "of",
        "on",
        "or",
        "our",
        "should",
        "so",
        "that",
        "the",
        "this",
        "to",
        "was",
        "we",
        "what",
        "when",
        "where",
        "which",
        "who",
        "with",
        "you",
        "your",
    }
)


def tokenize(text: str, *, drop_stopwords: bool = True) -> list[str]:
    """Lowercase alphanumeric tokens. Stopwords are dropped by default."""

    tokens = _TOKEN.findall(text.lower())
    if not drop_stopwords:
        return tokens
    return [token for token in tokens if token not in STOPWORDS]
