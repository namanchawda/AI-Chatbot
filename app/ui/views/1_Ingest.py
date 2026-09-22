"""Ingest page — upload documents, monitor ingestion progress, view ingested files."""

from __future__ import annotations

import multiprocessing
import tempfile
import time
from pathlib import Path

import streamlit as st

from app.config import settings
from app.ingestion import store
from app.ingestion.background_worker import cleanup_temp_upload, run_ingestion_job
from app.ingestion.job_status import mark_stale_if_needed, write_status
from app.ui._shared import (
    CHUNKING_STRATEGY_LABELS,
    clear_connection_details,
    format_strategy_label,
    get_available_documents,
)

DocumentChunk = store.DocumentChunk


# ---------------------------------------------------------------------------
# Helpers local to the ingest page
# ---------------------------------------------------------------------------


def launch_ingestion_worker(
    filepath: str,
    filename: str,
    chunking_strategy: str,
    database_url: str,
) -> str | None:
    """Launch a standalone daemon process after the uploaded file is on disk.

    Returns None on success, or an error message if the worker could not start.
    """
    try:
        # daemon=True: the child is tied to the parent's lifetime and is
        # terminated if Streamlit exits mid-ingestion.
        worker = multiprocessing.Process(
            target=run_ingestion_job,
            args=(filepath, filename, chunking_strategy, database_url),
            daemon=True,
            name="ingestion-worker",
        )
        worker.start()
        print(f"[ingestion-worker] Process spawned, pid={worker.pid}", flush=True)
        return None
    except Exception as exc:
        error_msg = f"Failed to start ingestion process: {exc}"
        print(f"[ingestion-worker] {error_msg}", flush=True)
        write_status(in_progress=False, stage="failed", error=error_msg)
        return error_msg


def render_ingestion_stage_ui(status: dict) -> None:
    """Render stage-based ingestion progress, using real percentage only for embeddings."""
    raw_stage = str(status.get("stage", "starting")).lower().strip()
    stage_key = raw_stage.replace("-", "_").replace(" ", "_")
    current = status.get("current", 0)
    total = status.get("total", 0)

    # Normalize free-form worker stage strings (from progress messages and
    # status writes) onto the fixed pipeline keys below; unknown stages are
    # kept as-is and inserted dynamically into the stage list.
    stage_aliases = {
        "starting": "starting",
        "start": "starting",
        "reading": "reading",
        "read": "reading",
        "extracting": "extracting",
        "extract_text": "extracting",
        "extracting_text": "extracting",
        "parsing": "extracting",
        "chunking": "chunking",
        "chunk": "chunking",
        "creating_chunks": "chunking",
        "embedding": "embedding",
        "embeddings": "embedding",
        "storing": "storing",
        "saving": "storing",
        "database": "storing",
        "complete": "complete",
    }
    current_stage = stage_aliases.get(stage_key, stage_key)

    stages = [
        ("starting", "Preparing document"),
        ("reading", "Reading document"),
        ("extracting", "Extracting text"),
        ("chunking", "Creating chunks"),
        ("embedding", "Generating embeddings"),
        ("storing", "Saving to vector database"),
    ]

    # Unknown stages (not in the fixed pipeline) are inserted just before the
    # final step so they still render in the progress list.
    known_keys = {key for key, _ in stages}
    if current_stage not in known_keys and current_stage != "complete":
        stages.insert(-1, (current_stage, current_stage.replace("_", " ").title()))

    current_index = next(
        (index for index, (key, _) in enumerate(stages) if key == current_stage),
        0,
    )

    filename = status.get("filename", "document")
    stage_label = dict(stages).get(
        current_stage,
        current_stage.replace("_", " ").title(),
    )

    st.info(f"Processing **{filename}**")

    pipeline_html = ["<div style='margin: 0.4rem 0 0.8rem 0;'>"]
    for index, (key, label) in enumerate(stages):
        if index < current_index:
            icon = "\u2713"
            weight = "normal"
        elif index == current_index:
            icon = "\u25cf"
            weight = "600"
        else:
            icon = "\u25cb"
            weight = "normal"

        pipeline_html.append(
            f"<div style='line-height: 1.8; font-weight: {weight};'>"
            f"<span style='display:inline-block; width:24px;'>{icon}</span>{label}"
            f"</div>"
        )
    pipeline_html.append("</div>")
    st.markdown("".join(pipeline_html), unsafe_allow_html=True)

    if current_stage == "embedding" and total > 0:
        percent = round(current / total * 100)
        st.progress(
            min(max(percent / 100, 0.0), 1.0),
            text=f"Generating embeddings: {current}/{total} chunks ({percent}%)",
        )
    elif current_stage not in {"complete", "failed"}:
        st.markdown(
            """
            <div style="
                width: 100%;
                height: 8px;
                border-radius: 999px;
                overflow: hidden;
                background: rgba(128,128,128,0.20);
                margin: 0.4rem 0 0.7rem 0;
            ">
                <div style="
                    width: 35%;
                    height: 100%;
                    border-radius: 999px;
                    background: rgba(128,128,128,0.65);
                    animation: ingestion-slide 1.4s ease-in-out infinite;
                "></div>
            </div>
            <style>
            @keyframes ingestion-slide {
                0%   { transform: translateX(-120%); }
                50%  { transform: translateX(180%); }
                100% { transform: translateX(300%); }
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

    if current_stage == "embedding" and total > 0:
        st.caption(f"Current stage: {stage_label}")
    elif current_stage not in {"complete", "failed"}:
        st.caption(f"{stage_label}... Please wait.")


def render_ingestion_monitor() -> None:
    """Poll durable worker status so progress survives Streamlit reruns and refreshes."""

    # Each fragment re-runs on its own 2-second timer without a full page
    # rerun, so the pipeline view stays live while the worker writes the
    # status file from its separate process.

    @st.fragment(run_every="2s")
    def poll_main_status() -> None:
        # Refresh the in-progress pipeline view (or surface a failure) each tick.
        status = mark_stale_if_needed()
        if status.get("in_progress"):
            render_ingestion_stage_ui(status)
        elif status.get("stage") == "failed":
            st.error(f"Ingestion failed: {status.get('error', 'Unknown error')}")

    @st.fragment(run_every="2s")
    def poll_completion() -> None:
        # On the first tick after completion, stash a one-shot success banner
        # in session_state and rerun so the main page body can display it.
        status = mark_stale_if_needed()
        if status.get("stage") == "complete" and not st.session_state.get("ingestion_success_pending"):
            st.session_state["ingestion_success_pending"] = {
                "filename": status.get("filename", "document"),
                "chunk_count": status.get("chunk_count", status.get("total", 0)),
            }
            st.rerun()

    status = mark_stale_if_needed()
    if status.get("in_progress"):
        poll_main_status()
        poll_completion()


def render_documents_table(documents: list[dict]) -> None:
    """Display ingested documents as a clean table."""
    if not documents:
        st.info("No documents ingested yet.")
        return

    if store.SessionLocal is not None:
        with store.SessionLocal() as session:
            from sqlalchemy import func

            count_rows = (
                session.query(
                    DocumentChunk.source_file,
                    DocumentChunk.chunking_strategy,
                    func.count(DocumentChunk.id).label("chunk_count"),
                )
                .group_by(DocumentChunk.source_file, DocumentChunk.chunking_strategy)
                .all()
            )
        count_map = {
            (row.source_file, row.chunking_strategy or "fixed"): row.chunk_count
            for row in count_rows
        }
    else:
        count_map = {}

    rows_data = []
    for doc in documents:
        key = (doc["source_file"], doc["chunking_strategy"])
        rows_data.append({
            "Filename": doc["source_file"],
            "Strategy": format_strategy_label(doc["chunking_strategy"]),
            "Chunks": count_map.get(key, "?"),
        })

    st.dataframe(rows_data, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Page body
# ---------------------------------------------------------------------------

st.header("\U0001f4e5 Ingest Documents")

# Always register polling fragments so they detect both progress and failure.
render_ingestion_monitor()

ingestion_status = mark_stale_if_needed()
ingestion_in_progress = bool(ingestion_status.get("in_progress"))
ingestion_failed = (
    not ingestion_in_progress
    and ingestion_status.get("stage") == "failed"
    and ingestion_status.get("error")
)

success_pending = st.session_state.pop("ingestion_success_pending", None)
if success_pending:
    st.success(
        f"**{success_pending['filename']}** successfully ingested "
        f"({success_pending['chunk_count']} chunks)."
    )
    st.page_link("views/2_Chat.py", label="Go to Chat \u2192", icon="\U0001f4ac")

if ingestion_failed:
    error_msg = ingestion_status["error"]
    st.error(f"Ingestion failed: {error_msg}")
    st.caption("The file may have been removed or the database connection lost.")
    if st.button("Dismiss error", key="dismiss_ingestion_error"):
        write_status(in_progress=False, stage="idle", error=None)
        st.rerun()

# --- Sidebar (ingestion controls) ---
with st.sidebar:
    if st.button("Disconnect", disabled=ingestion_in_progress):
        st.session_state["db_connected"] = False
        clear_connection_details()
        st.rerun()

    st.header("Upload a document")
    uploaded_file = st.file_uploader(
        "PDF, HTML, or text file",
        type=["pdf", "html", "txt"],
        accept_multiple_files=False,
        disabled=ingestion_in_progress,
    )

    if not ingestion_in_progress and uploaded_file is not None and uploaded_file.size > 20 * 1024 * 1024:
        st.warning("Large files may take several minutes to process on this hosted environment.")

    chunking_strategy = st.selectbox(
        "Chunking strategy",
        options=list(CHUNKING_STRATEGY_LABELS.keys()),
        format_func=lambda strategy: CHUNKING_STRATEGY_LABELS[strategy],
        index=0,
        disabled=ingestion_in_progress,
    )

    if st.button("Ingest", use_container_width=True, disabled=ingestion_in_progress):
        if uploaded_file is None:
            st.warning("Please upload a file before ingesting.")
        elif Path(uploaded_file.name).suffix.lower() not in {".pdf", ".html", ".htm", ".txt"}:
            st.error("Unsupported file type.")
        else:
            # The upload lives only in the OS temp directory while the worker
            # processes it; run_ingestion_job() deletes it afterwards.
            safe_name = Path(uploaded_file.name).name
            temp_dir = tempfile.mkdtemp(prefix="ingest_")
            destination = Path(temp_dir) / safe_name
            ingestion_error: str | None = None
            try:
                # Seed the status file before spawning so the monitor has
                # something to show from the first tick (and staleness timing
                # starts now).
                write_status(
                    in_progress=True,
                    filename=uploaded_file.name,
                    stage="starting",
                    current=0,
                    total=0,
                    started_at=time.time(),
                    error=None,
                )
                destination.write_bytes(uploaded_file.getvalue())
                database_url = store.engine.url.render_as_string(hide_password=False)
                ingestion_error = launch_ingestion_worker(
                    str(destination),
                    uploaded_file.name,
                    chunking_strategy,
                    database_url,
                )
            except Exception as exc:
                ingestion_error = str(exc)
                write_status(in_progress=False, stage="failed", error=ingestion_error)

            if ingestion_error is None:
                st.success(f"Ingestion started for **{uploaded_file.name}**.")
                st.rerun()
            else:
                # Worker never started — clean up here instead of in its finally block.
                cleanup_temp_upload(str(destination))
                st.error(f"Ingestion failed: {ingestion_error}")

    st.caption(f"Groq model: {settings.GROQ_MODEL}")

# --- Main area: ingested documents table ---
st.subheader("Ingested Documents")
documents = get_available_documents()
render_documents_table(documents)

if documents:
    st.caption(f"{len(documents)} document/strategy combination(s) in the knowledge base.")
