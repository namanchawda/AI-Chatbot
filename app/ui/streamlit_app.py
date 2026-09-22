"""Streamlit entry point — connection gate, then hands off to page navigation."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Disable Streamlit's file watcher — it otherwise crawls the whole app tree
# and slows startup without helping here.
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

# Ensure the repo root is importable when launched as `streamlit run app/ui/streamlit_app.py`.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st

from app.config import settings
from app.ingestion import store
from app.ui._shared import (
    AVAILABLE_GROQ_MODELS,
    CONNECTION_FILE,
    clear_connection_details,
    save_connection_details,
)

# ---------------------------------------------------------------------------
# Everything below runs ONLY in the Streamlit runtime, never in a spawned
# multiprocessing child (where __name__ == "__mp_main__").
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Page config and shared session state
    st.set_page_config(page_title="RAG Chat", page_icon="\U0001f4ac", layout="wide")

    if "db_connected" not in st.session_state:
        st.session_state["db_connected"] = False
    if "active_session_id" not in st.session_state:
        st.session_state["active_session_id"] = None
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    # ------------------------------------------------------------------
    # Connection setup — must succeed before any page is reachable
    # ------------------------------------------------------------------

    if not st.session_state.get("db_connected") and CONNECTION_FILE.exists():
        try:
            payload = json.loads(CONNECTION_FILE.read_text(encoding="utf-8"))
            connection_string = payload.get("connection_string", "").strip()
            groq_api_key = payload.get("groq_api_key", "").strip()
            groq_model = payload.get("groq_model", AVAILABLE_GROQ_MODELS[0]).strip()

            if connection_string and groq_api_key:
                # SQLAlchemy needs the psycopg driver prefix; Neon/Postgres URLs
                # usually arrive as bare postgresql://.
                normalized_url = connection_string
                if normalized_url.startswith("postgresql://"):
                    normalized_url = "postgresql+psycopg://" + normalized_url[len("postgresql://"):]
                else:
                    normalized_url = normalized_url.replace("postgresql://", "postgresql+psycopg://", 1)

                store.init_engine(normalized_url)
                store.create_table()
                settings.GROQ_API_KEY = groq_api_key
                settings.GROQ_MODEL = groq_model
                st.session_state["db_connected"] = True
        except Exception as exc:
            clear_connection_details()
            st.error(f"Saved connection failed: {exc}")

    if not st.session_state.get("db_connected"):
        st.title("Connect to your database")
        connection_string = st.text_input(
            "NeonDB or Postgres connection string",
            type="password",
            placeholder="postgresql://user:pass@host/dbname?sslmode=require&channel_binding=require",
        )
        groq_api_key = st.text_input("GROQ_API_KEY", type="password", placeholder="Enter your Groq API key")
        groq_model = st.selectbox("GROQ_MODEL", options=AVAILABLE_GROQ_MODELS, index=0)

        if st.button("Connect"):
            missing = []
            if not connection_string.strip():
                missing.append("Postgres connection string")
            if not groq_api_key.strip():
                missing.append("GROQ_API_KEY")

            if missing:
                st.error(f"Missing required value(s): {', '.join(missing)}")
            else:
                try:
                    # Same driver-prefix normalization as the saved-connection path above.
                    normalized_url = connection_string.strip()
                    if normalized_url.startswith("postgresql://"):
                        normalized_url = "postgresql+psycopg://" + normalized_url[len("postgresql://"):]
                    else:
                        normalized_url = normalized_url.replace("postgresql://", "postgresql+psycopg://", 1)

                    store.init_engine(normalized_url)
                    store.create_table()
                    settings.GROQ_API_KEY = groq_api_key.strip()
                    settings.GROQ_MODEL = groq_model
                    save_connection_details(normalized_url, settings.GROQ_API_KEY, settings.GROQ_MODEL)
                    st.session_state["db_connected"] = True
                    st.rerun()
                except Exception as exc:
                    message = str(exc)
                    if 'type "vector" does not exist' in message:
                        st.error(
                            'Your database doesn\'t have the pgvector extension enabled yet. Go to your '
                            'Neon dashboard\'s SQL Editor and run: CREATE EXTENSION IF NOT EXISTS vector; '
                            '\u2014 then try connecting again.'
                        )
                    else:
                        st.error(f"Connection failed: {message}")

        with st.expander("New here? Setup instructions", expanded=False):
            st.markdown(
                """
                ## Setting up NeonDB (free Postgres + pgvector)
                1. Go to https://neon.tech and sign up (no credit card required)
                2. Click "Create a project", give it any name, choose a nearby region
                3. Open the SQL Editor tab and run: `CREATE EXTENSION IF NOT EXISTS vector;`
                4. Go to Connection Details, select "Pooled connection", and copy the full connection string (starts with `postgresql://`)

                ## Getting a Groq API key (free)
                1. Go to https://console.groq.com and sign up (no credit card required)
                2. Go to "API Keys" in the sidebar, click "Create API Key"
                3. Copy the key immediately — it's only shown once
                """
            )
        st.stop()

    # ------------------------------------------------------------------
    # Connected — define pages and hand off to navigation
    # ------------------------------------------------------------------

    ingest_page = st.Page(
        "views/1_Ingest.py",
        title="Ingest Documents",
        icon="\U0001f4e5",
    )
    chat_page = st.Page(
        "views/2_Chat.py",
        title="Chat",
        icon="\U0001f4ac",
    )

    pg = st.navigation([ingest_page, chat_page])
    pg.run()
