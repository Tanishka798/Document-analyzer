"""
schemas.py
----------
Why this module exists:
    Centralizes Pydantic request/response models so the API's shape is
    defined in exactly one place, and so routers, FastAPI's auto-generated
    docs, and (later) the Streamlit frontend all agree on field names and
    types. This file will grow substantially in later phases (chat,
    summaries, comparisons, voice); Phase 1 only needs the health schema.
"""

from pydantic import BaseModel, Field


class DependencyStatus(BaseModel):
    """Status of one external dependency the app relies on."""

    name: str
    available: bool
    detail: str = Field(
        default="",
        description="Human-readable detail, e.g. an error message or a "
        "reachable URL. Never include stack traces or local file paths here.",
    )


class HealthResponse(BaseModel):
    """
    Response for GET /health.

    Why we check real dependencies here instead of just returning {"ok": true}:
    the whole point of this app is "it either works locally with the right
    things installed, or it tells you clearly what's missing" (Section 8/14
    of the project spec: never pretend something works). The sidebar in
    Streamlit will poll this endpoint to show real Ollama/model status.
    """

    status: str = Field(description='"ok" if the backend process itself is healthy')
    dependencies: list[DependencyStatus]


# --- Documents (Phase 2/3: ingestion) ---


class DocumentInfo(BaseModel):
    """One row of the documents table, as returned to callers."""

    id: str
    filename: str
    file_size_bytes: int
    status: str = Field(description='"processing", "ready", or "failed"')
    num_chunks: int
    error: str | None = None
    uploaded_at: str


class DocumentListResponse(BaseModel):
    documents: list[DocumentInfo]


# --- Chat / RAG (Phase 5) ---


class ChatRequest(BaseModel):
    query: str = Field(min_length=1)
    document_ids: list[str] | None = Field(
        default=None,
        description="Restrict retrieval to these document ids. Omit to search all documents.",
    )


class SourceChunk(BaseModel):
    document_id: str
    filename: str
    chunk_index: int
    text: str
    distance: float = Field(description="Vector distance; lower means more similar.")


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]


# --- Summarize / Compare (Phase 6) ---


class SummarizeResponse(BaseModel):
    document_id: str
    filename: str
    summary: str


class CompareRequest(BaseModel):
    document_ids: list[str] = Field(min_length=2, max_length=5)


class CompareResponse(BaseModel):
    document_ids: list[str]
    comparison: str


# --- Voice (Phase 8) ---


class TranscribeResponse(BaseModel):
    text: str


class SpeakRequest(BaseModel):
    text: str = Field(min_length=1)
