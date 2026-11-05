import json
from pathlib import Path

import pytest

from semsearch.cli import main


def test_search_cli_prints_top_doc(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["search", "How do I reset my SSO password?", "--method", "bm25", "--k", "1"])
    assert code == 0
    output = capsys.readouterr().out
    assert "faq-sso-reset" in output
    assert "hashing-md5-v1" in output


def test_search_json(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["search", "Vault secrets", "--method", "dense", "--k", "2", "--json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["embedder"] == "hashing-md5-v1"
    assert payload["method"] == "dense"
    assert len(payload["hits"]) == 2


def test_eval_json(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["eval", "--k", "5", "--json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert set(payload["metrics"]) == {"bm25", "dense", "hybrid"}
    assert payload["embedder"] == "hashing-md5-v1"
    assert payload["num_documents"] == 40
    assert payload["num_queries"] == 20
    assert payload["k"] == 5
    for scores in payload["metrics"].values():
        assert set(scores) == {"recall@5", "mrr", "ndcg@5"}
        assert all(0.0 <= value <= 1.0 for value in scores.values())


def test_eval_writes_output(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    destination = tmp_path / "nested" / "report.json"
    code = main(["eval", "--k", "10", "--output", str(destination), "--json"])
    assert code == 0
    saved = json.loads(destination.read_text(encoding="utf-8"))
    printed = json.loads(capsys.readouterr().out)
    assert saved == printed
    assert saved["k"] == 10


def test_unknown_command() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["nope"])
    assert exc.value.code != 0


def test_cli_rejects_empty_query() -> None:
    assert main(["search", "   "]) == 2
    assert main(["search", "password", "--k", "0"]) == 2
    assert main(["eval", "--k", "0"]) == 2
