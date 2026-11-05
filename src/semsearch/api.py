"""FastAPI search service.

The default process indexes the bundled FAQ with the hashing embedder.
Set SEMSEARCH_EMBEDDER=minilm only when the ml extra is installed.
"""

from __future__ import annotations

from typing import Literal

from fastapi import Depends, FastAPI, Query, Request
from pydantic import BaseModel, Field

from semsearch import __version__
from semsearch.service import METHODS, SearchService, build_default_service

MethodName = Literal["bm25", "dense", "hybrid"]


class HitOut(BaseModel):
    rank: int
    doc_id: str
    title: str
    text: str
    score: float


class SearchResponse(BaseModel):
    query: str
    method: str
    embedder: str
    k: int
    hits: list[HitOut]


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    method: MethodName = "hybrid"
    k: int = Field(default=5, ge=1, le=50)


class HealthResponse(BaseModel):
    status: str
    documents: int
    embedder: str
    methods: list[str]


def get_service(request: Request) -> SearchService:
    service = getattr(request.app.state, "service", None)
    if service is None:
        service = build_default_service()
        request.app.state.service = service
    return service


def _response(service: SearchService, query: str, method: str, k: int) -> SearchResponse:
    hits = service.search(query, method=method, k=k)
    return SearchResponse(
        query=query,
        method=method,
        embedder=service.embedder.name,
        k=k,
        hits=[
            HitOut(
                rank=hit.rank,
                doc_id=hit.doc_id,
                title=hit.title,
                text=hit.text,
                score=hit.score,
            )
            for hit in hits
        ],
    )


def create_app(service: SearchService | None = None) -> FastAPI:
    app = FastAPI(
        title="Lumen FAQ semantic search",
        version=__version__,
        summary="BM25, dense embeddings, and reciprocal rank fusion over a small technical FAQ.",
    )
    app.state.service = service

    @app.get("/", include_in_schema=False)
    def root() -> dict[str, str]:
        return {
            "service": "semsearch",
            "health": "/health",
            "search": "/search",
            "docs": "/docs",
        }

    @app.get("/health", response_model=HealthResponse)
    def health(search: SearchService = Depends(get_service)) -> HealthResponse:
        return HealthResponse(
            status="ok",
            documents=len(search.documents),
            embedder=search.embedder.name,
            methods=list(METHODS),
        )

    @app.get("/search", response_model=SearchResponse)
    def search_get(
        q: str = Query(min_length=1, max_length=2000),
        method: MethodName = Query(default="hybrid"),
        k: int = Query(default=5, ge=1, le=50),
        search: SearchService = Depends(get_service),
    ) -> SearchResponse:
        return _response(search, q, method, k)

    @app.post("/search", response_model=SearchResponse)
    def search_post(
        body: SearchRequest,
        search: SearchService = Depends(get_service),
    ) -> SearchResponse:
        return _response(search, body.query, body.method, body.k)

    return app


app = create_app()
