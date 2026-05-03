"""Pydantic models for FinDoc Agent API."""

from pydantic import BaseModel


class QueryRequest(BaseModel):
    query: str
    doc_filter: list[str] | None = None


class CitationOut(BaseModel):
    doc_id: str
    page_num: int


class PageHitOut(BaseModel):
    doc_id: str
    page_num: int
    score: float
    image_path: str | None = None


class DocInfo(BaseModel):
    doc_id: str
    page_count: int


class HealthResponse(BaseModel):
    status: str          # "ok" | "degraded"
    docs_count: int
    backend: str         # "in_memory" | "qdrant"
