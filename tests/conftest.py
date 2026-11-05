"""Shared fixtures. The tiny corpus never touches the neural model."""

from __future__ import annotations

from pathlib import Path

import pytest

from semsearch.embedders import HashingEmbedder
from semsearch.service import SearchService
from semsearch.types import Document

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def tiny_documents() -> list[Document]:
    return [
        Document(
            "d-password",
            "SSO password reset",
            "Reset your SSO password from the Okta page.",
        ),
        Document(
            "d-vpn",
            "VPN setup",
            "Install the corporate VPN client for the home office.",
        ),
        Document(
            "d-gpu",
            "GPU quota",
            "Request a GPU for model training in the ML platform form.",
        ),
    ]


@pytest.fixture
def tiny_service(tiny_documents: list[Document]) -> SearchService:
    return SearchService(tiny_documents, HashingEmbedder(dim=64))
