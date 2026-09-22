"""Pydantic request and response schemas for the API."""

from __future__ import annotations

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Query / Chat
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    """Schema for a retrieval and generation question request."""

    question: str = Field(..., min_length=1, description="User question to answer using the indexed documents.")
    session_id: str | None = Field(
        default=None,
        description="Chat session ID. Omit to create a new session automatically.",
    )
    use_reranking: bool = Field(default=True, description="Whether to apply cross-encoder reranking.")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of relevant chunks to retrieve.")


class SourceResult(BaseModel):
    """A source chunk referenced in an assistant answer."""

    source_file: str
    chunk_id: int
    chunk_text: str = ""
    # Cosine distance from vector search; None for keyword-only hits.
    distance: float | None = None
    # Populated only when cross-encoder reranking was applied.
    rerank_score: float | None = None


class QueryResponse(BaseModel):
    """Schema for the answer returned to the client."""

    answer: str = Field(..., description="Generated answer based on retrieved context.")
    sources: list[SourceResult] = Field(default_factory=list, description="Retrieved chunks used as evidence.")
    session_id: str | None = Field(default=None, description="Chat session ID for continuing the conversation.")


# ---------------------------------------------------------------------------
# Chat sessions
# ---------------------------------------------------------------------------


class CreateSessionRequest(BaseModel):
    """Optional request body for creating a session with an initial title."""

    title: str | None = None


class SessionInfo(BaseModel):
    """Summary of a chat session."""

    id: str
    title: str | None = None
    created_at: str | None = None
    updated_at: str | None = None


class ChatMessageInfo(BaseModel):
    """A single chat message in the conversation history."""

    id: int
    role: str
    content: str
    # Sources snapshot from chat_messages.sources JSONB; None for user messages.
    sources: list[SourceResult] | None = None
    created_at: str | None = None
