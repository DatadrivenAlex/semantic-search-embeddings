"""Embedder interface: a deterministic hasher, and an optional neural model.

Tests and CI use `HashingEmbedder`. It never downloads weights. The
sentence-transformers model is imported only when someone asks for it.
"""

from __future__ import annotations

import hashlib
from typing import Protocol

import numpy as np

from semsearch.tokenize import tokenize

# Short name accepted by sentence-transformers. 384 dimensions, L2-normalized.
DEFAULT_NEURAL_MODEL = "all-MiniLM-L6-v2"

_ML_INSTALL_HINT = (
    "The neural embedder needs the optional ml extra: pip install -e '.[ml]'. "
    "Tests and the default service use the hashing embedder and do not download a model."
)


class Embedder(Protocol):
    name: str
    dim: int
    description: str

    def encode(self, texts: list[str]) -> np.ndarray:
        """Return one L2-normalized float32 row per text. Shape is (n, dim)."""


class HashingEmbedder:
    """Signed feature hashing over word unigrams and adjacent bigrams.

    The hash is MD5 because `hash()` is randomized per process. This is a
    stable id, not a security control. Overlap in tokens moves vectors
    together. A paraphrase that shares no tokens will not.
    """

    name = "hashing-md5-v1"
    description = (
        "Deterministic signed MD5 feature hashing over word unigrams and bigrams, "
        "L2-normalized. This is not a neural semantic model and it does not download weights."
    )

    def __init__(self, dim: int = 384) -> None:
        if dim < 1:
            raise ValueError("dim must be positive")
        self.dim = dim

    def encode(self, texts: list[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            tokens = tokenize(text)
            features = list(tokens)
            features.extend(
                f"{left} {right}" for left, right in zip(tokens, tokens[1:], strict=False)
            )
            for feature in features:
                digest = hashlib.md5(feature.encode("utf-8")).digest()
                index = int.from_bytes(digest[:4], "little") % self.dim
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                matrix[row, index] += sign
            norm = float(np.linalg.norm(matrix[row]))
            if norm > 0.0:
                matrix[row] /= norm
        return matrix


class SentenceTransformerEmbedder:
    """Dense encoder backed by sentence-transformers. Optional dependency."""

    def __init__(self, model_name: str = DEFAULT_NEURAL_MODEL) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(_ML_INSTALL_HINT) from exc
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)
        dimension = self._model.get_sentence_embedding_dimension()
        if not isinstance(dimension, int) or dimension < 1:
            raise RuntimeError(f"Model {model_name} did not report an embedding dimension")
        self.dim = dimension
        self.name = f"sentence-transformers:{model_name}"
        self.description = (
            f"Neural sentence embeddings from {model_name}, L2-normalized by the encoder."
        )

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vectors = self._model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        array = np.asarray(vectors, dtype=np.float32)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        return array


def build_embedder(
    name: str,
    *,
    model_name: str | None = None,
    dim: int = 384,
) -> Embedder:
    """Build a named embedder. `hashing` is the offline default. `minilm` is optional."""

    key = name.strip().lower().replace("_", "-")
    if key in {"hashing", "hash", "hashing-md5-v1"}:
        if model_name is not None:
            raise ValueError("The hashing embedder does not take a model name.")
        return HashingEmbedder(dim=dim)
    if key in {"minilm", "all-minilm-l6-v2", "sentence-transformers"}:
        return SentenceTransformerEmbedder(model_name or DEFAULT_NEURAL_MODEL)
    raise ValueError(f"Unknown embedder {name!r}. Choose 'hashing' or 'minilm'.")
