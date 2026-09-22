"""Chat page — multi-document chat interface, gated on ingestion status."""

from __future__ import annotations

import streamlit as st

from app.generation.rag_pipeline import answer_question
from app.ingestion import store
from app.ingestion.job_status import mark_stale_if_needed
from app.ui._shared import (
    get_available_documents,
    is_ingestion_in_progress,
)

DocumentChunk = store.DocumentChunk


# ---------------------------------------------------------------------------
# Gate: no documents ingested yet
# ---------------------------------------------------------------------------

ingestion_status = mark_stale_if_needed()
documents = get_available_documents()
ingestion_running = is_ingestion_in_progress()

if not documents and not ingestion_running:
    st.header("\U0001f4ac Chat")
    st.info(
        "No documents ingested yet. Head to the **Ingest Documents** page to upload your "
        "first file, then come back here to start chatting."
    )
    st.page_link("views/1_Ingest.py", label="Go to Ingest \u2192", icon="\U0001f4e5")
    st.stop()

# ---------------------------------------------------------------------------
# Gate: ingestion currently running
# ---------------------------------------------------------------------------

if ingestion_running:
    st.header("\U0001f4ac Chat")
    st.info("Ingestion is in progress \u2014 please wait before starting a chat.")
    st.page_link("views/1_Ingest.py", label="View progress on Ingest page \u2192", icon="\U0001f4e5")
    st.stop()


# ---------------------------------------------------------------------------
# Chat page body — all gates passed
# ---------------------------------------------------------------------------

st.header("\U0001f4ac Chat")

# Show which documents are in scope
doc_names = sorted({d["source_file"] for d in documents})
st.caption(f"Chatting across: {', '.join(doc_names)}")

# --- Sidebar: session management ---
with st.sidebar:
    if st.button("New chat", use_container_width=True):
        chat_session = store.create_session()
        st.session_state["active_session_id"] = str(chat_session.id)
        st.session_state["chat_history"] = []
        st.rerun()

    sessions = store.list_sessions()
    for sess in sessions:
        session_id = sess["id"]
        title = sess.get("title") or "Untitled chat"
        is_active = session_id == st.session_state.get("active_session_id")
        label = f"{'>' if is_active else ' '} {title[:40]}"
        col_label, col_del = st.columns([5, 1])
        with col_label:
            if st.button(label, key=f"sess_{session_id}", use_container_width=True):
                st.session_state["active_session_id"] = session_id
                messages = store.get_messages(session_id)
                st.session_state["chat_history"] = [
                    {"role": msg["role"], "content": msg["content"], "sources": msg.get("sources")}
                    for msg in messages
                ]
                st.rerun()
        with col_del:
            if st.button("x", key=f"del_{session_id}"):
                store.delete_session(session_id)
                if st.session_state.get("active_session_id") == session_id:
                    st.session_state["active_session_id"] = None
                    st.session_state["chat_history"] = []
                st.rerun()


# ---------------------------------------------------------------------------
# Chat area
# ---------------------------------------------------------------------------

chat_history = st.session_state["chat_history"]

# Render existing messages from session_state (display only — LLM calls
# below always pass chat_history=None).
for msg in chat_history:
    role = msg.get("role", "user")
    content = msg.get("content", "")

    with st.chat_message(role):
        st.markdown(content)

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state["chat_history"].append({"role": "user", "content": prompt})

    session_id = st.session_state.get("active_session_id")

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and generating..."):
            try:
                # Conversation history stays in the DB/UI only; it is
                # intentionally not sent to the LLM for now.
                result = answer_question(
                    query=prompt,
                    top_k=5,
                    source_file=None,
                    use_reranking=True,
                    chat_history=None,
                )
                answer_text = result["answer"]
                sources = result["sources"]
            except Exception as exc:
                answer_text = f"Error: {exc}"
                sources = []

        st.markdown(answer_text)

    # Persist to DB
    sources_for_storage = [
        {
            "source_file": s["source_file"],
            "chunk_id": s["chunk_id"],
            "chunk_text": s.get("chunk_text", ""),
        }
        for s in sources
    ]

    if session_id is None:
        chat_session = store.create_session()
        session_id = str(chat_session.id)
        st.session_state["active_session_id"] = session_id

    store.add_message(session_id, "user", prompt)
    store.add_message(session_id, "assistant", answer_text, sources=sources_for_storage)

    # First exchange only: use the opening question as the session title.
    if len(st.session_state["chat_history"]) == 1:
        store.update_session_title(session_id, prompt[:80])

    st.session_state["chat_history"].append({
        "role": "assistant",
        "content": answer_text,
        "sources": sources_for_storage,
    })
