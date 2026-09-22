"""Standalone multiprocessing target for durable document ingestion.

Runs in a child process with no Streamlit import: rebuilds the DB engine
from the URL passed by the parent, reports progress via the status file,
and always cleans up the temp upload in its finally block.
"""

from __future__ import annotations

import re
import os
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

from app.ingestion import store
from app.ingestion.ingest import ingest_file
from app.ingestion.job_status import read_status, write_status

LOG_DIR = Path(__file__).resolve().parents[2] / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
ERROR_LOG = LOG_DIR / "ingestion_errors.log"


def cleanup_temp_upload(filepath: str) -> None:
    """Best-effort removal of a temporary upload file and its parent temp directory.

    Only paths inside the OS temp directory are touched, so this is safe to call
    with any filepath. Failures are swallowed so cleanup can never mask the
    original ingestion result.
    """
    try:
        path = Path(filepath)
        temp_root = Path(tempfile.gettempdir()).resolve()
        resolved = path.resolve()
        try:
            resolved.relative_to(temp_root)
        except ValueError:
            return
        if resolved.is_file():
            os.remove(resolved)
        parent = resolved.parent
        if parent != temp_root and parent.is_dir():
            shutil.rmtree(parent, ignore_errors=True)
    except Exception as exc:
        print(
            f"[ingestion-worker] Temp cleanup failed for {filepath}: {exc}",
            file=sys.stderr,
            flush=True,
        )


def _log_error(filepath: str, exc: BaseException) -> None:
    """Append the full traceback to the error log file and print to stderr."""
    tb = traceback.format_exception(type(exc), exc, exc.__traceback__)
    header = f"\n{'='*70}\n[{os.getpid()}] {type(exc).__name__}: {exc}\nFile: {filepath}\n"
    entry = header + "".join(tb) + "="*70 + "\n"
    with open(ERROR_LOG, "a", encoding="utf-8") as f:
        f.write(entry)
    print(entry, file=sys.stderr, flush=True)


def run_ingestion_job(
    filepath: str,
    filename: str,
    chunking_strategy: str,
    database_url: str,
) -> None:
    """Run ingestion in a child process without importing the Streamlit app."""
    print(f"[ingestion-worker] Process started (pid={os.getpid()})", flush=True)
    try:
        # Child process starts with no engine — rebuild it from the URL the parent passed.
        store.init_engine(database_url)
        # Record the child pid in the status file for diagnostics.
        write_status(process_pid=os.getpid())

        def update_progress(message: str, progress: float) -> None:
            # Stage name = message before ':' with any trailing '(n%)' stripped
            # (e.g. "Embedding: 3/10 chunks (30%)" → "Embedding").
            stage = re.sub(r"\s*\(\d+%\)\s*$", "", message.split(":", 1)[0]).strip().rstrip(".")
            current = 0
            total = 0
            # Only the embedding stage carries current/total counts; other
            # stages report 0/0 and the UI shows an indeterminate bar.
            if stage == "Embedding" and "/" in message:
                counts = message.split(":", 1)[1].split("(", 1)[0].strip().split("/")
                current = int(counts[0])
                total = int(counts[1].split()[0])
            write_status(stage=stage.lower(), current=current, total=total)

        chunk_count = ingest_file(
            filepath,
            chunking_strategy=chunking_strategy,
            progress_callback=update_progress,
        )
        write_status(
            in_progress=False,
            filename=filename,
            stage="complete",
            current=chunk_count,
            total=chunk_count,
            chunk_count=chunk_count,
            error=None,
        )
    except Exception as exc:
        _log_error(filepath, exc)
        # Persist failure state for the UI monitor, then re-raise so the
        # child process exits non-zero.
        write_status(
            in_progress=False,
            filename=filename,
            stage="failed",
            error=f"{type(exc).__name__}: {exc}",
        )
        raise
    finally:
        # Always remove the temp upload — success or failure — so nothing
        # lingers in the OS temp directory.
        cleanup_temp_upload(filepath)
