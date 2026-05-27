"""
API request and response schemas.

These are the contract between the client and the service.
Validation lives here so the routes stay thin.
"""
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator

from app.core.config import get_settings


class IngestDocumentRequest(BaseModel):
    """Payload for POST /documents."""

    text: str = Field(..., min_length=1, description="Raw document text to ingest")
    metadata: dict[str, Any] | None = Field(
        default=None, description="Optional client-supplied metadata"
    )

    @field_validator("text")
    @classmethod
    def enforce_size_limit(cls, v: str) -> str:
        limit = get_settings().max_document_chars
        if len(v) > limit:
            raise ValueError(
                f"Document exceeds maximum size of {limit:,} characters "
                f"(got {len(v):,})."
            )
        return v


class IngestDocumentResponse(BaseModel):
    """Response from POST /documents."""

    doc_id: str
    chunk_count: int
    char_count: int
    created_at: datetime


class AskQuestionRequest(BaseModel):
    """Payload for POST /questions/{doc_id}."""

    question: str = Field(..., min_length=1, max_length=2000)


class SourceChunk(BaseModel):
    """A retrieved chunk surfaced to the caller for transparency."""

    content: str
    score: float | None = None
    chunk_index: int | None = None


class QuestionResponseMetadata(BaseModel):
    """Per-request observability metadata returned to the caller."""

    doc_id: str
    latency_ms: int
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    estimated_cost_usd: float | None = None
    tool_calls: int = 0
    short_circuited: bool = False


class AskQuestionResponse(BaseModel):
    """Response from POST /questions/{doc_id}."""

    answer: str
    source_chunks: list[SourceChunk]
    metadata: QuestionResponseMetadata


class ErrorResponse(BaseModel):
    """Uniform error envelope."""

    error: str
    detail: str | None = None
