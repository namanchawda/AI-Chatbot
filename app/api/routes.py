"""API endpoints for ingestion, querying, and chat session management."""

from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile, status

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


def _require_user(user_id: str) -> str:
    """Validate an explicit user_id and return it, or raise 400.

    The API has no login session, so callers pass the owning account in the
    request instead of it being inferred. Checking it here keeps a typo'd id
    from becoming a 500 from the foreign key.
    """
    if not user_id or store.get_user_by_id(user_id) is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unknown user_id. Sign up through the app first, then pass that account's id.",
        )
    return user_id


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------


@router.post("/ingest", status_code=status.HTTP_200_OK)
def ingest_documents(
    file: UploadFile = File(...),
    user_id: str = Form(..., description="Account that will own the ingested chunks."),
) -> dict:
    """Write the upload to a temporary file, ingest it, then delete the temp file.

    user_id is a required form field (this endpoint is multipart, so it cannot
    carry a Pydantic body) and tags every stored chunk with its owner.
    """
    _require_user(user_id)

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
        ingest_file(str(destination), user_id=user_id)
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
    """Answer a question using hybrid search over the caller's documents.

    Prior turns are stored in Postgres for session display, but they are NOT
    included in the prompt: each question is answered independently.

    Retrieval, session creation and message persistence are all scoped to
    payload.user_id — the API carries no login session, so the owning account
    is supplied explicitly in the body.
    """
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    user_id = _require_user(payload.user_id)

    # Resolve session
    session_id = payload.session_id
    if session_id is None:
        chat_session = store.create_session(user_id=user_id)
        session_id = str(chat_session.id)
    elif session_id not in {s["id"] for s in store.list_sessions(user_id=user_id)}:
        raise HTTPException(status_code=404, detail="Chat session not found for this user_id.")

    # Prior messages are loaded only for auto-titling the session on its
    # first question; they are intentionally not passed to the LLM.
    prior_messages = store.get_messages(session_id, user_id=user_id)

    # Retrieve and generate (searches this account's documents only).
    # chat_history stays None for now: history is persisted for display but
    # not sent to the LLM. The parameter remains available for re-enabling.
    result = answer_question(
        query=payload.question,
        top_k=payload.top_k,
        source_file=None,
        use_reranking=payload.use_reranking,
        chat_history=None,
        user_id=user_id,
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
    store.add_message(session_id, "user", payload.question, user_id=user_id)
    store.add_message(
        session_id, "assistant", result["answer"], sources=sources_for_storage, user_id=user_id
    )

    # Auto-title: use the first user question as the session title if untitled
    if len(prior_messages) == 0:
        title = payload.question[:80]
        store.update_session_title(session_id, title, user_id=user_id)

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
def create_session(payload: CreateSessionRequest) -> SessionInfo:
    """Create a new chat session owned by payload.user_id."""
    user_id = _require_user(payload.user_id)
    chat_session = store.create_session(user_id=user_id, title=payload.title)
    return SessionInfo(
        id=str(chat_session.id),
        title=chat_session.title,
        created_at=chat_session.created_at.isoformat() if chat_session.created_at else None,
        updated_at=chat_session.updated_at.isoformat() if chat_session.updated_at else None,
    )


@router.get("/sessions", status_code=status.HTTP_200_OK)
def list_sessions(
    user_id: str = Query(..., description="Account whose sessions are returned."),
) -> list[SessionInfo]:
    """Return chat sessions ordered by most recently updated, for one account.

    user_id is required. Omitting it is a 422 validation error rather than an
    unscoped list of every account's sessions.
    """
    sessions = store.list_sessions(user_id=_require_user(user_id))
    return [SessionInfo(**s) for s in sessions]


@router.get("/sessions/{session_id}/messages", status_code=status.HTTP_200_OK)
def get_session_messages(
    session_id: str,
    user_id: str = Query(..., description="Account that must own the session."),
) -> list[ChatMessageInfo]:
    """Return all messages for a given chat session owned by user_id.

    user_id is required: another account's messages are never returned, and a
    request without user_id is a 422 instead of an unscoped read.
    """
    # sources come from the stored chat_messages.sources JSONB snapshot.
    messages = store.get_messages(session_id, user_id=_require_user(user_id))
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
def delete_session(
    session_id: str,
    user_id: str = Query(..., description="Account that must own the session."),
) -> None:
    """Delete a chat session and all its messages, only when user_id owns it.

    user_id is required: it is a 422 without one, never a delete scoped to
    nobody (which would match any owner).
    """
    store.delete_session(session_id, user_id=_require_user(user_id))
    return None


# ---------------------------------------------------------------------------
# Documents listing
# ---------------------------------------------------------------------------


@router.get("/documents", status_code=status.HTTP_200_OK)
def list_documents(
    user_id: str = Query(..., description="Account whose documents are returned."),
) -> list[str]:
    """Return distinct source_file values for one account's documents.

    user_id is required: omitting it is a 422, not a listing of every
    account's files.
    """
    owner = _require_user(user_id)
    with store.SessionLocal() as session:
        rows = (
            session.query(DocumentChunk.source_file)
            .distinct()
            .filter(DocumentChunk.user_id == owner)
            .order_by(DocumentChunk.source_file)
            .all()
        )
    return [row[0] for row in rows]
