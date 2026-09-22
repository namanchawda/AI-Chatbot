"""Shared helpers and constants used across all Streamlit pages."""

from __future__ import annotations

import json
from pathlib import Path

from app.config import settings
from app.ingestion import store
from app.ingestion.job_status import mark_stale_if_needed

DocumentChunk = store.DocumentChunk

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Repo-root JSON cache of the active DB/Groq connection (cleared on disconnect).
CONNECTION_FILE = Path(__file__).resolve().parents[2] / ".streamlit_connection.json"
AVAILABLE_GROQ_MODELS = ["openai/gpt-oss-120b"]
CHUNKING_STRATEGY_LABELS = {
    "fixed": "Fixed-size",
    "sentence_aware": "Sentence-aware",
    "paragraph_based": "Paragraph-based",
    "recursive": "Recursive (paragraph → sentence → fixed)",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def save_connection_details(connection_string: str, groq_api_key: str, groq_model: str) -> None:
    """Persist the active DB and Groq settings to a local JSON file."""
    payload = {
        "connection_string": connection_string,
        "groq_api_key": groq_api_key,
        "groq_model": groq_model,
    }
    CONNECTION_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def clear_connection_details() -> None:
    """Remove any persisted DB/Groq connection cache."""
    if CONNECTION_FILE.exists():
        CONNECTION_FILE.unlink()


def format_strategy_label(strategy: str) -> str:
    """Return a human-friendly label for a chunking strategy key."""
    return CHUNKING_STRATEGY_LABELS.get(strategy, strategy)


def get_available_documents() -> list[dict]:
    """Return distinct (source_file, chunking_strategy) combinations currently ingested in the DB."""
    if store.SessionLocal is None:
        return []
    with store.SessionLocal() as session:
        rows = (
            session.query(
                DocumentChunk.source_file,
                DocumentChunk.chunking_strategy,
            )
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


def is_ingestion_in_progress() -> bool:
    """Return True if a background ingestion job is currently running."""
    status = mark_stale_if_needed()
    return bool(status.get("in_progress"))
