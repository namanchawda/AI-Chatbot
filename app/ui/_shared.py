"""Shared helpers and constants used across all Streamlit pages."""

from __future__ import annotations

from sqlalchemy import func

from app.ingestion import store
from app.ingestion.job_status import mark_stale_if_needed

DocumentChunk = store.DocumentChunk

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHUNKING_STRATEGY_LABELS = {
    "fixed": "Fixed-size",
    "sentence_aware": "Sentence-aware",
    "paragraph_based": "Paragraph-based",
    "recursive": "Recursive (paragraph → sentence → fixed)",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def format_strategy_label(strategy: str) -> str:
    """Return a human-friendly label for a chunking strategy key."""
    return CHUNKING_STRATEGY_LABELS.get(strategy, strategy)


def get_available_documents(user_id: str) -> list[dict]:
    """Return the (source_file, chunking_strategy) pairs ingested by one account.

    Scoped by user_id in SQL so a freshly created account always sees an empty
    knowledge base, regardless of what anyone else has uploaded.
    """
    if store.SessionLocal is None:
        return []
    with store.SessionLocal() as session:
        rows = (
            session.query(
                DocumentChunk.source_file,
                DocumentChunk.chunking_strategy,
            )
            .filter(DocumentChunk.user_id == user_id)
            .distinct()
            .order_by(DocumentChunk.source_file, DocumentChunk.chunking_strategy)
            .all()
        )
    return [
        {
            "source_file": source_file,
            "chunking_strategy": chunking_strategy or "fixed",
            "label": f"{source_file} \u2014 {format_strategy_label(chunking_strategy or 'fixed')}",
        }
        for source_file, chunking_strategy in rows
    ]


def get_chunk_counts(user_id: str) -> dict[tuple[str, str], int]:
    """Return {(source_file, chunking_strategy): chunk_count} for one account."""
    if store.SessionLocal is None:
        return {}

    with store.SessionLocal() as session:
        rows = (
            session.query(
                DocumentChunk.source_file,
                DocumentChunk.chunking_strategy,
                func.count(DocumentChunk.id).label("chunk_count"),
            )
            .filter(DocumentChunk.user_id == user_id)
            .group_by(DocumentChunk.source_file, DocumentChunk.chunking_strategy)
            .all()
        )
    return {
        (row.source_file, row.chunking_strategy or "fixed"): row.chunk_count
        for row in rows
    }


def is_ingestion_in_progress(user_id: str | None = None) -> bool:
    """Return True if a background ingestion job is currently running.

    Pass user_id to count only that account's job, so one user's upload never
    blocks (or is visible to) another user's pages.
    """
    status = mark_stale_if_needed()
    if user_id is not None and status.get("user_id") != user_id:
        return False
    return bool(status.get("in_progress"))
