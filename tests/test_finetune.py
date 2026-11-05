import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from semsearch.data import load_dataset
from semsearch.finetune import build_training_pairs, resolve_batch_size, train_biencoder

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_pairs_cover_positive_qrels_and_are_sorted() -> None:
    dataset = load_dataset()
    pairs = build_training_pairs(dataset)
    assert len(pairs) == 41
    assert pairs == build_training_pairs(dataset)
    keys = [(pair["query_id"], pair["doc_id"]) for pair in pairs]
    assert keys == sorted(keys)
    assert pairs[0]["query"]
    assert pairs[0]["positive"]
    labeled = {
        (query_id, doc_id)
        for query_id, grades in dataset.qrels.items()
        for doc_id, grade in grades.items()
        if grade >= 1
    }
    assert set(keys) == labeled


def test_resolve_batch_size() -> None:
    assert resolve_batch_size(10, 16) == 10
    assert resolve_batch_size(10, 4) == 4
    with pytest.raises(ValueError, match="at least 2"):
        resolve_batch_size(1, 16)
    with pytest.raises(ValueError, match="positive"):
        resolve_batch_size(4, 0)


def test_dry_run_script_writes_pairs_without_a_model(tmp_path: Path) -> None:
    destination = tmp_path / "pairs.jsonl"
    env = os.environ.copy()
    env["PYTHONPATH"] = "src" + os.pathsep + env.get("PYTHONPATH", "")
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/finetune.py",
            "--dry-run",
            "--pairs-output",
            str(destination),
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    lines = destination.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 41
    row = json.loads(lines[0])
    assert {"query_id", "doc_id", "query", "positive"} <= set(row)


def test_train_reports_missing_extra(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import builtins

    real_import = builtins.__import__

    def guarded(name, globals=None, locals=None, fromlist=(), level=0):  # noqa: A002
        blocked = name in {"datasets", "sentence_transformers"}
        blocked = blocked or name.startswith("sentence_transformers.")
        if blocked:
            raise ImportError("blocked for test")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded)
    pairs = [
        {"query": "how do I reset a password", "positive": "Open the reset page."},
        {"query": "where are secrets stored", "positive": "Production secrets live in Vault."},
    ]
    with pytest.raises(RuntimeError, match="ml extra"):
        train_biencoder(pairs, tmp_path)
