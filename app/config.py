"""Application configuration and environment variable loading.

Reads environment variables (optionally from a .env file) into a single
Settings instance that every module imports as ``settings``.

Streamlit also exposes root-level keys from ``.streamlit/secrets.toml`` as
environment variables when it boots, and ``_secret`` covers keys that live in
a TOML section — so on Streamlit Cloud the same settings resolve from App
settings → Secrets without any code change.
"""

import os

from dotenv import load_dotenv

load_dotenv()


def _secret(name: str) -> str | None:
    """Read a key from Streamlit secrets as a fallback when the env var is absent.

    Only consulted inside a running Streamlit process — outside one there is no
    secrets context, and touching st.secrets would only print warnings.
    """
    try:
        from streamlit.runtime import exists as runtime_exists

        if not runtime_exists():
            return None

        import streamlit as st

        value = st.secrets.get(name)
    except Exception:
        return None
    return str(value) if value is not None else None


def _setting(name: str, default: str | None = None) -> str | None:
    """Resolve a setting: process env / .env first, then Streamlit secrets, then default."""
    return os.getenv(name) or _secret(name) or default


class Settings:
    """Centralized settings for local development and deployment."""

    EMBEDDING_MODEL: str = _setting("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")  # type: ignore[assignment]
    EMBEDDING_DIMENSION: int = int(_setting("EMBEDDING_DIMENSION", "384"))  # type: ignore[arg-type]

    POSTGRES_HOST: str = _setting("POSTGRES_HOST", "localhost")  # type: ignore[assignment]
    POSTGRES_PORT: int = int(_setting("POSTGRES_PORT", "5432"))  # type: ignore[arg-type]
    POSTGRES_DB: str = _setting("POSTGRES_DB", "sec_rag")  # type: ignore[assignment]
    POSTGRES_USER: str = _setting("POSTGRES_USER", "postgres")  # type: ignore[assignment]
    POSTGRES_PASSWORD: str = _setting("POSTGRES_PASSWORD", "postgres")  # type: ignore[assignment]
    POSTGRES_SSLMODE: str = _setting("POSTGRES_SSLMODE", "prefer")  # type: ignore[assignment]

    # Groq credentials — always read from the environment (.env locally,
    # st.secrets on Streamlit Cloud). No runtime/UI override exists.
    GROQ_API_KEY: str | None = _setting("GROQ_API_KEY")
    GROQ_MODEL: str = _setting("GROQ_MODEL", "llama-3.3-70b-versatile")  # type: ignore[assignment]

    VECTOR_TABLE: str = _setting("VECTOR_TABLE", "document_chunks")  # type: ignore[assignment]
    QUERY_TOP_K: int = int(_setting("QUERY_TOP_K", "5"))  # type: ignore[arg-type]


settings = Settings()


def get_settings() -> Settings:
    """Return the active configuration."""
    return settings
