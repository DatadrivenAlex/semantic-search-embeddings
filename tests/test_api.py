from fastapi.testclient import TestClient

from semsearch.api import app, create_app
from semsearch.service import SearchService


def test_health(tiny_service: SearchService) -> None:
    with TestClient(create_app(tiny_service)) as client:
        response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["documents"] == 3
    assert body["embedder"] == "hashing-md5-v1"
    assert body["methods"] == ["bm25", "dense", "hybrid"]


def test_get_search(tiny_service: SearchService) -> None:
    with TestClient(create_app(tiny_service)) as client:
        response = client.get(
            "/search",
            params={"q": "reset SSO password", "method": "bm25", "k": 2},
        )
    assert response.status_code == 200
    body = response.json()
    assert body["method"] == "bm25"
    assert body["embedder"] == "hashing-md5-v1"
    assert body["hits"][0]["doc_id"] == "d-password"
    assert len(body["hits"]) == 2
    assert isinstance(body["hits"][0]["score"], float)


def test_post_search_matches_get(tiny_service: SearchService) -> None:
    with TestClient(create_app(tiny_service)) as client:
        params = {"q": "gpu quota", "method": "hybrid", "k": 2}
        by_get = client.get("/search", params=params).json()
        by_post = client.post(
            "/search",
            json={"query": "gpu quota", "method": "hybrid", "k": 2},
        ).json()
    assert by_post["hits"] == by_get["hits"]


def test_rejects_bad_method_empty_query_and_k(tiny_service: SearchService) -> None:
    with TestClient(create_app(tiny_service)) as client:
        assert client.get("/search", params={"q": "x", "method": "nope"}).status_code == 422
        assert client.get("/search", params={"q": ""}).status_code == 422
        assert client.get("/search", params={"q": "x", "k": 0}).status_code == 422
        assert client.get("/search", params={"q": "x", "k": 51}).status_code == 422
        assert client.post("/search", json={"query": "x", "k": 0}).status_code == 422


def test_root_and_openapi(tiny_service: SearchService) -> None:
    with TestClient(create_app(tiny_service)) as client:
        root = client.get("/")
        schema = client.get("/openapi.json")
    assert root.status_code == 200
    assert root.json()["search"] == "/search"
    paths = schema.json()["paths"]
    assert "/search" in paths
    assert "/health" in paths


def test_default_app_serves_bundled_corpus() -> None:
    with TestClient(app) as client:
        health = client.get("/health")
        found = client.get(
            "/search",
            params={"q": "How do I reset my SSO password?", "method": "bm25", "k": 1},
        )
    assert health.status_code == 200
    assert health.json()["documents"] == 40
    assert health.json()["embedder"] == "hashing-md5-v1"
    assert found.status_code == 200
    assert found.json()["hits"][0]["doc_id"] == "faq-sso-reset"
