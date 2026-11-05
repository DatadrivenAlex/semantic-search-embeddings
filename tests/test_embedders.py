import numpy as np
import pytest

from semsearch.embedders import (
    DEFAULT_NEURAL_MODEL,
    HashingEmbedder,
    SentenceTransformerEmbedder,
    build_embedder,
)


def test_default_neural_model_is_minilm() -> None:
    assert DEFAULT_NEURAL_MODEL == "all-MiniLM-L6-v2"


def test_hashing_shape_dtype_and_dim() -> None:
    embedder = HashingEmbedder(dim=32)
    vectors = embedder.encode(["reset the password", "vpn client"])
    assert vectors.shape == (2, 32)
    assert vectors.dtype == np.float32
    assert embedder.name == "hashing-md5-v1"


def test_hashing_is_deterministic_across_instances() -> None:
    text = ["How do I reset my SSO password?"]
    first = HashingEmbedder(dim=64).encode(text)
    second = HashingEmbedder(dim=64).encode(text)
    np.testing.assert_array_equal(first, second)


def test_hashing_rows_are_l2_normalized() -> None:
    vectors = HashingEmbedder(dim=64).encode(["gpu quota for training", "gpu"])
    norms = np.linalg.norm(vectors, axis=1)
    np.testing.assert_allclose(norms, np.ones(2), atol=1e-6)


def test_hashing_empty_text_is_zero() -> None:
    vector = HashingEmbedder(dim=16).encode([""])
    assert vector.shape == (1, 16)
    assert float(np.linalg.norm(vector)) == 0.0


def test_hashing_empty_batch() -> None:
    vectors = HashingEmbedder(dim=16).encode([])
    assert vectors.shape == (0, 16)


def test_hashing_batch_matches_rows() -> None:
    embedder = HashingEmbedder(dim=32)
    batch = embedder.encode(["alpha beta", "gamma"])
    alone = embedder.encode(["alpha beta"])
    np.testing.assert_array_equal(batch[0], alone[0])


def test_hashing_overlap_beats_unrelated() -> None:
    vectors = HashingEmbedder(dim=128).encode(
        [
            "reset the sso password",
            "reset your sso password please",
            "kubernetes deployment rollback image",
        ]
    )
    near = float(vectors[0] @ vectors[1])
    far = float(vectors[0] @ vectors[2])
    assert near > far
    assert near > 0.2


def test_build_embedder_hashing_aliases() -> None:
    assert build_embedder("hashing").name == "hashing-md5-v1"
    assert build_embedder("hashing-md5-v1", dim=16).dim == 16


def test_build_embedder_unknown() -> None:
    with pytest.raises(ValueError, match="Unknown embedder"):
        build_embedder("tfidf")


def test_hashing_rejects_model_name() -> None:
    with pytest.raises(ValueError, match="does not take a model name"):
        build_embedder("hashing", model_name="all-MiniLM-L6-v2")


def test_hashing_rejects_non_positive_dim() -> None:
    with pytest.raises(ValueError, match="dim"):
        HashingEmbedder(dim=0)


def test_minilm_constructor_reports_missing_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    real_import = builtins.__import__

    def guarded(name, globals=None, locals=None, fromlist=(), level=0):  # noqa: A002
        if name == "sentence_transformers" or name.startswith("sentence_transformers."):
            raise ImportError("blocked for test")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded)
    with pytest.raises(RuntimeError, match="ml extra"):
        SentenceTransformerEmbedder("all-MiniLM-L6-v2")
