PYTHON ?= python3

.DEFAULT_GOAL := help

.PHONY: help install install-ml lint format test eval serve finetune-dry-run docker-build docker-run

help:
	@echo "targets: install install-ml lint format test eval serve finetune-dry-run docker-build docker-run"

install:
	$(PYTHON) -m pip install -e ".[dev]"

install-ml:
	$(PYTHON) -m pip install -e ".[dev,ml]"

lint:
	$(PYTHON) -m ruff check src tests scripts
	$(PYTHON) -m ruff format --check src tests scripts

format:
	$(PYTHON) -m ruff check --fix src tests scripts
	$(PYTHON) -m ruff format src tests scripts

test: lint
	$(PYTHON) -m pytest

eval:
	$(PYTHON) -m semsearch eval --k 10 --embedder hashing

serve:
	$(PYTHON) -m uvicorn semsearch.api:app --host 0.0.0.0 --port 8000

finetune-dry-run:
	$(PYTHON) scripts/finetune.py --dry-run --pairs-output outputs/train_pairs.jsonl

docker-build:
	docker build -t semsearch:local .

docker-run:
	docker run --rm -p 8000:8000 semsearch:local
