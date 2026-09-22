"""Durable, process-safe status tracking for the Streamlit ingestion worker."""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
STATUS_FILE = ROOT / ".ingestion_status.json"
# A worker that dies without writing a final status is declared failed after this window.
STALE_AFTER_SECONDS = 30 * 60
# Serializes concurrent readers/writers within this process; cross-process safety
# comes from the atomic mkstemp + os.replace write in write_status().
_STATUS_LOCK = threading.RLock()


def default_status() -> dict:
    """Return the complete status shape used by the ingestion worker."""
    return {
        "in_progress": False,
        "filename": "",
        "stage": "idle",
        "current": 0,
        "total": 0,
        "chunk_count": 0,
        "process_pid": None,
        "started_at": None,
        "error": None,
    }


def read_status() -> dict:
    """Read status safely and normalize missing fields from older status files."""
    with _STATUS_LOCK:
        if not STATUS_FILE.exists():
            return default_status()
        try:
            stored = json.loads(STATUS_FILE.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return default_status()

        status = default_status()
        status.update(stored)
        return status


def write_status(**updates) -> dict:
    """Atomically write status while serializing concurrent readers and writers."""
    with _STATUS_LOCK:
        status = read_status()
        # Partial updates merge into the current status so fields not mentioned
        # here (e.g. stage vs. current/total) are never clobbered.
        status.update(updates)
        STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
        # Write to a temp file, fsync, then atomically rename — readers in other
        # processes never observe a half-written JSON file.
        fd, temporary_name = tempfile.mkstemp(prefix=".ingestion_status_", dir=STATUS_FILE.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as temporary_file:
                json.dump(status, temporary_file, indent=2)
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
            os.replace(temporary_name, STATUS_FILE)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)
        return status


def mark_stale_if_needed(status: dict | None = None) -> dict:
    """Mark an abandoned job failed after the configured staleness window."""
    status = status or read_status()
    started_at = status.get("started_at")
    if status.get("in_progress") and isinstance(started_at, (int, float)):
        if time.time() - started_at > STALE_AFTER_SECONDS:
            return write_status(
                in_progress=False,
                stage="stale",
                error="Ingestion was marked failed after exceeding the 30-minute timeout.",
            )
    return status