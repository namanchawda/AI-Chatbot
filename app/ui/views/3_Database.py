"""Database page — browse the signed-in account's own stored chunks.

A read-only, console-style view over document_chunks: readable metadata and
text only (never the embedding vector), scoped to the logged-in user, with a
source-file filter, a case-insensitive text search, and pagination.
"""

from __future__ import annotations

from sqlalchemy import func, select

import streamlit as st

from app.config import settings
from app.ingestion import store
from app.ui._shared import format_strategy_label

DocumentChunk = store.DocumentChunk

ROWS_PER_PAGE = 25
SNIPPET_LENGTH = 180
# Only present if a migration has added it; the model has no created_at today.
HAS_CREATED_AT = "created_at" in DocumentChunk.__table__.columns


def _reset_page() -> None:
    """Return to the first page and drop the row selection when a filter changes."""
    st.session_state["db_page"] = 0
    st.session_state.pop("db_chunk_table", None)


def _go_to_page(new_page: int) -> None:
    """Move pages and clear the selection so it can't point at another row."""
    st.session_state["db_page"] = new_page
    st.session_state.pop("db_chunk_table", None)


def _escape_like(term: str) -> str:
    """Escape LIKE wildcards so a search for '100%' doesn't match everything."""
    return (
        term.replace("\\", "\\\\")
        .replace("%", "\\%")
        .replace("_", "\\_")
    )


st.header("\U0001f5c3 Database")

user_id = st.session_state.get("user_id")
if not user_id:
    st.info("You need to sign in to browse the database.")
    st.stop()

if store.SessionLocal is None:
    st.info("The database isn't ready yet — reload the page.")
    st.stop()

st.caption(f"Table `{settings.VECTOR_TABLE}` \u2014 your chunks only. Click a row to read its full text.")

# --- Filters ---------------------------------------------------------------

with store.SessionLocal() as session:
    source_files = [
        row[0]
        for row in (
            session.query(DocumentChunk.source_file)
            .filter(DocumentChunk.user_id == user_id)
            .distinct()
            .order_by(DocumentChunk.source_file)
            .all()
        )
    ]

filter_col, search_col = st.columns([1, 2])
with filter_col:
    selected_file = st.selectbox(
        "Source file",
        options=["All files", *source_files],
        key="db_file_filter",
        on_change=_reset_page,
    )
with search_col:
    search_term = st.text_input(
        "Search chunk text",
        key="db_search",
        placeholder="case-insensitive contains\u2026",
        on_change=_reset_page,
    )

filters = [DocumentChunk.user_id == user_id]
if selected_file != "All files":
    filters.append(DocumentChunk.source_file == selected_file)
if search_term.strip():
    filters.append(
        DocumentChunk.chunk_text.ilike(f"%{_escape_like(search_term.strip())}%", escape="\\")
    )

# --- Page state ------------------------------------------------------------

if "db_page" not in st.session_state:
    st.session_state["db_page"] = 0

with store.SessionLocal() as session:
    total = session.query(func.count()).select_from(DocumentChunk).filter(*filters).scalar() or 0

total_pages = max(1, -(-total // ROWS_PER_PAGE))
if st.session_state["db_page"] > total_pages - 1:
    st.session_state["db_page"] = total_pages - 1
page = st.session_state["db_page"]

prev_col, info_col, next_col = st.columns([1, 2, 1])
with prev_col:
    if st.button("\u2190 Previous", use_container_width=True, disabled=page == 0):
        _go_to_page(page - 1)
        st.rerun()
with info_col:
    st.caption(f"Page {page + 1} of {total_pages} \u00b7 {total} chunk(s)")
with next_col:
    if st.button("Next \u2192", use_container_width=True, disabled=page >= total_pages - 1):
        _go_to_page(page + 1)
        st.rerun()

# --- Rows ------------------------------------------------------------------

# `id` is selected only to give each row a collision-free widget key; it is
# never rendered (chunk_id, source_file and strategy can repeat across
# strategies, so they are not unique within a page).
selected_columns = [
    DocumentChunk.id,
    DocumentChunk.chunk_id,
    DocumentChunk.source_file,
    DocumentChunk.chunking_strategy,
    DocumentChunk.chunk_text,
    DocumentChunk.token_count,
]
if HAS_CREATED_AT:
    selected_columns.append(DocumentChunk.created_at)

with store.SessionLocal() as session:
    stmt = (
        select(*selected_columns)
        .where(*filters)
        .order_by(DocumentChunk.source_file.asc(), DocumentChunk.chunk_id.asc())
        .offset(page * ROWS_PER_PAGE)
        .limit(ROWS_PER_PAGE)
    )
    rows = session.execute(stmt).mappings().all()

if not rows:
    st.info("No chunks match the current filters.")
    st.stop()

table_rows = []
for row in rows:
    chunk_text = row["chunk_text"] or ""
    table_rows.append(
        {
            "chunk_id": row["chunk_id"],
            "source_file": row["source_file"],
            "chunking_strategy": format_strategy_label(row["chunking_strategy"] or "fixed"),
            "chunk_text": (
                chunk_text
                if len(chunk_text) <= SNIPPET_LENGTH
                else chunk_text[:SNIPPET_LENGTH].rstrip() + "\u2026"
            ),
            "token_count": row["token_count"],
            **({"created_at": row["created_at"]} if HAS_CREATED_AT else {}),
        }
    )

selection = st.dataframe(
    table_rows,
    use_container_width=True,
    hide_index=True,
    key="db_chunk_table",
    on_select="rerun",
    selection_mode="single-row",
    column_config={
        "chunk_text": st.column_config.TextColumn("chunk_text", width="large"),
    },
)

# --- Full text for the selected row ---------------------------------------

selected_rows = selection.selection.rows if selection else []
if selected_rows:
    chosen = rows[selected_rows[0]]
    with st.expander(
        f"Chunk {chosen['chunk_id']} from {chosen['source_file']} \u2014 full text",
        expanded=True,
    ):
        st.caption(
            f"{format_strategy_label(chosen['chunking_strategy'] or 'fixed')} \u00b7 "
            f"{chosen['token_count']} tokens"
        )
        st.text_area(
            "chunk_text",
            value=chosen["chunk_text"],
            height=320,
            disabled=True,
            key=f"db_chunk_text_{chosen['id']}",
        )
