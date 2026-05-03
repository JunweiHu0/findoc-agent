"""Pydantic models for FinDoc Agent API."""

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
# P13: Conversations
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
# P14: Upload
# ---------------------------------------------------------------------------

class UploadResponse(BaseModel):
    upload_id: str
    doc_id: str
    status: str  # "queued"


class UploadStatusOut(BaseModel):
    upload_id: str
    doc_id: str
    status: str  # "queued" | "encoding" | "ready" | "failed"
    page_count: int = 0
    message: str = ""


class DocumentOut(BaseModel):
    doc_id: str
    source_filename: str
    page_count: int
    status: str
    created_at: float
