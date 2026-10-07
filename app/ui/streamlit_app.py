"""Streamlit entry point — login gate, then hands off to page navigation.

The database and Groq credentials always come from settings (.env locally,
st.secrets on Streamlit Cloud); there is no connection-string or credential
field to paste. Access to every page is gated on a logged-in account.
"""

from __future__ import annotations

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
from app.ingestion.store import User

# Session keys that describe the logged-in account (plus per-account chat state).
_AUTH_KEYS = ("user_id", "user_email", "active_session_id", "chat_history")


def _establish_session(user: User) -> None:
    """Record the logged-in account and reset any per-account chat state."""
    st.session_state["user_id"] = user.id
    st.session_state["user_email"] = user.email
    # A previous account may have been signed in on this tab — drop its chat.
    st.session_state["active_session_id"] = None
    st.session_state["chat_history"] = []


def _clear_session() -> None:
    """Forget the logged-in account and everything scoped to it."""
    for key in (*_AUTH_KEYS, "auth_error"):
        st.session_state.pop(key, None)


def _handle_auth(mode: str, email: str, password: str) -> None:
    """Run the login/signup attempt, setting session state or an error message."""
    st.session_state.pop("auth_error", None)
    if mode == "Sign up":
        try:
            user = store.create_user(email, password)
        except ValueError as exc:
            st.session_state["auth_error"] = str(exc)
            return
    else:
        user = store.verify_user(email, password)
        if user is None:
            # Deliberately identical for unknown email and wrong password.
            st.session_state["auth_error"] = "Incorrect email or password."
            return
    _establish_session(user)


def _render_login_screen() -> None:
    """Render the login/sign-up form. Never returns to the caller's page setup."""
    st.title("Sign in to RAG Chat")

    if not settings.GROQ_API_KEY:
        st.error(
            "GROQ_API_KEY is missing or empty. Set it in your `.env` file "
            "(locally) or in Streamlit secrets (on Streamlit Cloud) before "
            "starting the app — it can't be entered here."
        )

    auth_error = st.session_state.pop("auth_error", None)
    if auth_error:
        st.error(auth_error)

    mode = st.radio("I want to", ["Log in", "Sign up"], horizontal=True, key="auth_mode")

    with st.form("auth_form"):
        email = st.text_input("Email", key="auth_email", autocomplete="email")
        password = st.text_input("Password", type="password", key="auth_password")
        submitted = st.form_submit_button(mode)

    if submitted:
        _handle_auth(mode, email, password)
        # Reruns whether we succeeded (to enter the app) or failed (to show
        # the error stored in session state above the form).
        st.rerun()

    with st.expander("New here? Setup instructions", expanded=False):
        st.markdown(
            """
            ## One-time app setup
            1. Copy `.env.example` to `.env`
            2. Fill in `POSTGRES_HOST/PORT/DB/USER/PASSWORD` (your Neon or local Postgres)
            3. Run `CREATE EXTENSION IF NOT EXISTS vector;` in your Postgres SQL editor
            4. Set `GROQ_API_KEY` (free key from https://console.groq.com)

            On Streamlit Cloud, put the same keys in **App settings → Secrets** instead.

            ## Accounts
            Pick **Sign up** to create an account with your email and a password
            (8+ characters). Documents and chats are private to your account.
            """
        )
    st.stop()


# ---------------------------------------------------------------------------
# Everything below runs ONLY in the Streamlit runtime, never in a spawned
# multiprocessing child (where __name__ == "__mp_main__").
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Page config and shared session state
    st.set_page_config(page_title="RAG Chat", page_icon="\U0001f4ac", layout="wide")

    for key, default in (
        ("user_id", None),
        ("user_email", None),
        ("active_session_id", None),
        ("chat_history", []),
        ("db_ready", False),
    ):
        if key not in st.session_state:
            st.session_state[key] = default

    # ------------------------------------------------------------------
    # Database bootstrap — engine always comes from settings.POSTGRES_*
    # ------------------------------------------------------------------

    if not st.session_state.get("db_ready"):
        try:
            store._ensure_initialized()
            store.create_table()
            st.session_state["db_ready"] = True
        except Exception as exc:
            message = str(exc)
            if 'type "vector" does not exist' in message:
                st.error(
                    "Your database doesn't have the pgvector extension enabled yet. Go to your "
                    "Neon dashboard's SQL Editor and run: CREATE EXTENSION IF NOT EXISTS vector; "
                    "\u2014 then reload this page."
                )
            else:
                st.error(f"Could not connect to the database: {message}")
            st.caption(
                "Database settings come from `.env` locally or Streamlit secrets on "
                "Streamlit Cloud — check POSTGRES_HOST/PORT/DB/USER/PASSWORD."
            )
            st.stop()

    # ------------------------------------------------------------------
    # Auth gate — nothing else renders until an account is signed in
    # ------------------------------------------------------------------

    if not st.session_state.get("user_id"):
        _render_login_screen()

    with st.sidebar:
        st.caption(f"Signed in as **{st.session_state['user_email']}**")
        if not settings.GROQ_API_KEY:
            st.warning("GROQ_API_KEY is not set — chat answers will fail.")
        if st.button("Log out", use_container_width=True, key="logout_button"):
            _clear_session()
            st.rerun()

    # ------------------------------------------------------------------
    # Authenticated — define pages and hand off to navigation
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
    database_page = st.Page(
        "views/3_Database.py",
        title="Database",
        icon="\U0001f5c3",
    )

    pg = st.navigation([ingest_page, chat_page, database_page])
    pg.run()
