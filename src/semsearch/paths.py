"""Locate the bundled JSONL corpus."""

from __future__ import annotations

import os
from pathlib import Path


def default_data_dir() -> Path:
    """Return the directory that contains `corpus.jsonl`.

    `SEMSEARCH_DATA_DIR` wins. Otherwise look next to the current working
    directory and next to the repository that contains this source file.
    """

    override = os.environ.get("SEMSEARCH_DATA_DIR")
    if override:
        path = Path(override).expanduser()
        if not (path / "corpus.jsonl").is_file():
            raise FileNotFoundError(f"SEMSEARCH_DATA_DIR={path} has no corpus.jsonl")
        return path

    candidates = [
        Path.cwd() / "data",
        Path(__file__).resolve().parents[2] / "data",
    ]
    for candidate in candidates:
        if (candidate / "corpus.jsonl").is_file():
            return candidate
    raise FileNotFoundError(
        "Could not find data/corpus.jsonl. Run from the repository root or set SEMSEARCH_DATA_DIR."
    )
