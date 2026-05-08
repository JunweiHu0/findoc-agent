"""Pydantic models for FinDoc Agent API."""

from typing import Optional

from pydantic import BaseModel


class QueryRequest(BaseModel):
    query: str
    doc_filter: list[str] | None = None
    conv_id: str | None = None


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


# ---------------------------------------------------------------------------
# Conversation history API models / 对话历史 API 模型
# ---------------------------------------------------------------------------

class ConversationCreate(BaseModel):
    title: str = ""


class ConversationUpdate(BaseModel):
    title: str


class ConversationOut(BaseModel):
    id: str
    title: str
    created_at: float
    updated_at: float


class MessageOut(BaseModel):
    id: str
    role: str
    content: str
    citations: list[dict] = []
    pages: list[dict] = []
    created_at: float


class ConversationDetail(BaseModel):
    id: str
    title: str
    created_at: float
    updated_at: float
    messages: list[MessageOut] = []


# ---------------------------------------------------------------------------
# Upload and document API models / 上传和文档 API 模型
# ---------------------------------------------------------------------------

class UploadResponse(BaseModel):
    upload_id: str
    doc_id: str
    status: str  # "queued"


class UploadStatusOut(BaseModel):
    """Shape of each SSE frame emitted by GET /api/v1/upload/{id}/status.

    `status` is the current pipeline stage emitted by run_upload_pipeline:
    queued | save | pages | encode | encoding | index | qdrant | register | done | failed.
    """
    doc_id: str
    status: str
    message: str = ""
    pct: float = 0.0


class DocumentOut(BaseModel):
    doc_id: str
    source_filename: str
    page_count: int
    status: str
    created_at: float
    # Inline base64 thumbnail (data URL). None when the doc has no rendered
    # page image — the frontend renders a plain "no image" placeholder.
    thumbnail: Optional[str] = None
