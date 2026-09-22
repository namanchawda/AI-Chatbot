"""Application configuration and environment variable loading.

Reads environment variables (optionally from a .env file) into a single
Settings instance that every module imports as ``settings``.
"""

import os

from dotenv import load_dotenv

load_dotenv()


class Settings:
    """Centralized settings for local development and deployment."""

    EMBEDDING_MODEL: str = os.getenv(
        "EMBEDDING_MODEL",
        "BAAI/bge-small-en-v1.5",
    )
    EMBEDDING_DIMENSION: int = int(os.getenv("EMBEDDING_DIMENSION", "384"))

    POSTGRES_HOST: str = os.getenv("POSTGRES_HOST", "localhost")
    POSTGRES_PORT: int = int(os.getenv("POSTGRES_PORT", "5432"))
    POSTGRES_DB: str = os.getenv("POSTGRES_DB", "sec_rag")
    POSTGRES_USER: str = os.getenv("POSTGRES_USER", "postgres")
    POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD", "postgres")
    POSTGRES_SSLMODE: str = os.getenv("POSTGRES_SSLMODE", "prefer")

    # Streamlit's connection gate may override these at runtime
    # (see app/ui/streamlit_app.py).
    GROQ_API_KEY: str | None = os.getenv("GROQ_API_KEY")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    VECTOR_TABLE: str = os.getenv("VECTOR_TABLE", "document_chunks")
    QUERY_TOP_K: int = int(os.getenv("QUERY_TOP_K", "5"))


settings = Settings()


def get_settings() -> Settings:
    """Return the active configuration."""
    return settings
