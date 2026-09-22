"""API endpoints for ingestion, querying, and chat session management."""

from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.generation.rag_pipeline import answer_question
from app.ingestion import store
from app.ingestion.background_worker import cleanup_temp_upload
from app.ingestion.chunker import chunk_text
from app.ingestion.ingest import ingest_file
from app.ingestion.loader import load_filing
from app.models import (
    ChatMessageInfo,
    CreateSessionRequest,
    QueryRequest,
    QueryResponse,
    SessionInfo,
    SourceResult,
)

DocumentChunk = store.DocumentChunk

router = APIRouter(prefix="/api", tags=["rag"])

SUPPORTED_EXTENSIONS = {".pdf", ".html", ".htm", ".txt", ".md", ".rtf"}


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------


@router.post("/ingest", status_code=status.HTTP_200_OK)
def ingest_documents(file: UploadFile = File(...)) -> dict:
    """Write the upload to a temporary file, ingest it, then delete the temp file."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="A file upload is required.")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type: {file.filename}. "
                f"Supported types: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            ),
        )

    # Basename only — strips any client-supplied path components before joining.
    safe_name = Path(file.filename).name
    destination = Path(tempfile.mkdtemp(prefix="ingest_")) / safe_name
    try:
        contents = file.file.read()
        destination.write_bytes(contents)
    except Exception as exc:
        cleanup_temp_upload(str(destination))
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {exc}") from exc
    finally:
        file.file.close()

    try:
        store.create_table()
        ingest_file(str(destination))
        chunk_count = len(chunk_text(load_filing(str(destination))))
        # Response shape: {filename, chunks_created, status}.
        return {
            "filename": file.filename,
            "chunks_created": chunk_count,
            "status": "success",
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}") from exc
    finally:
        # Always remove the temp upload, success or failure.
        cleanup_temp_upload(str(destination))


# ---------------------------------------------------------------------------
# Query (conversation persisted; history not sent to the LLM)
# ---------------------------------------------------------------------------


@router.post("/query", status_code=status.HTTP_200_OK)
def query_documents(payload: QueryRequest) -> QueryResponse:
    """Answer a question using hybrid search across all documents.

    Prior turns are stored in Postgres for session display, but they are NOT
    included in the prompt: each question is answered independently.
    """
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    # Resolve session
    session_id = payload.session_id
    if session_id is None:
        chat_session = store.create_session()
        session_id = str(chat_session.id)

    # Prior messages are loaded only for auto-titling the session on its
    # first question; they are intentionally not passed to the LLM.
    prior_messages = store.get_messages(session_id)

    # Retrieve and generate (searches ALL documents — no source_file filter).
    # chat_history stays None for now: history is persisted for display but
    # not sent to the LLM. The parameter remains available for re-enabling.
    result = answer_question(
        query=payload.question,
        top_k=payload.top_k,
        source_file=None,
        use_reranking=payload.use_reranking,
        chat_history=None,
    )

    # Persist user message and assistant response
    sources_for_storage = [
        {
            "source_file": s["source_file"],
            "chunk_id": s["chunk_id"],
            "chunk_text": s.get("chunk_text", ""),
        }
        for s in result["sources"]
    ]
    store.add_message(session_id, "user", payload.question)
    store.add_message(session_id, "assistant", result["answer"], sources=sources_for_storage)

    # Auto-title: use the first user question as the session title if untitled
    if len(prior_messages) == 0:
        title = payload.question[:80]
        store.update_session_title(session_id, title)

    # Response: QueryResponse {answer, sources, session_id}.
    return QueryResponse(
        answer=result["answer"],
        sources=[SourceResult(**s) for s in result["sources"]],
        session_id=session_id,
    )


# ---------------------------------------------------------------------------
# Chat sessions
# ---------------------------------------------------------------------------


@router.post("/sessions", status_code=status.HTTP_201_CREATED)
def create_session(payload: CreateSessionRequest | None = None) -> SessionInfo:
    """Create a new chat session."""
    title = payload.title if payload else None
    chat_session = store.create_session(title=title)
    return SessionInfo(
        id=str(chat_session.id),
        title=chat_session.title,
        created_at=chat_session.created_at.isoformat() if chat_session.created_at else None,
        updated_at=chat_session.updated_at.isoformat() if chat_session.updated_at else None,
    )


@router.get("/sessions", status_code=status.HTTP_200_OK)
def list_sessions() -> list[SessionInfo]:
    """Return all chat sessions ordered by most recently updated."""
    sessions = store.list_sessions()
    return [SessionInfo(**s) for s in sessions]


@router.get("/sessions/{session_id}/messages", status_code=status.HTTP_200_OK)
def get_session_messages(session_id: str) -> list[ChatMessageInfo]:
    """Return all messages for a given chat session."""
    # sources come from the stored chat_messages.sources JSONB snapshot.
    messages = store.get_messages(session_id)
    return [
        ChatMessageInfo(
            id=msg["id"],
            role=msg["role"],
            content=msg["content"],
            sources=[SourceResult(**s) for s in msg["sources"]] if msg.get("sources") else None,
            created_at=msg.get("created_at"),
        )
        for msg in messages
    ]


@router.delete(
    "/sessions/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_model=None,
)
def delete_session(session_id: str) -> None:
    """Delete a chat session and all its messages."""
    store.delete_session(session_id)
    return None


# ---------------------------------------------------------------------------
# Documents listing
# ---------------------------------------------------------------------------


@router.get("/documents", status_code=status.HTTP_200_OK)
def list_documents() -> list[str]:
    """Return the distinct source_file values currently stored in the vector database."""
    with store.SessionLocal() as session:
        rows = (
            session.query(DocumentChunk.source_file)
            .distinct()
            .order_by(DocumentChunk.source_file)
            .all()
        )
    return [row[0] for row in rows]
