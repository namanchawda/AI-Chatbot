"""Storage layer for persisting embedded document chunks to pgvector.

This module creates the vector-retrieval and chat tables and inserts chunk
rows in batches to keep ingestion fast and efficient.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import Computed, ForeignKey, Index, Integer, Text, create_engine
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker

from app.config import settings


class Base(DeclarativeBase):
    """Base class for SQLAlchemy models."""


class DocumentChunk(Base):
    """A single document chunk stored alongside its embedding vector.

    One row per embedded passage, keyed by (source_file, chunk_id).
    """

    __tablename__ = settings.VECTOR_TABLE
    __table_args__ = (
        # GIN index backing keyword search over the generated tsvector column.
        Index(
            f"ix_{settings.VECTOR_TABLE}_search_vector",
            "search_vector",
            postgresql_using="gin",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    # Original filename only (not a path) — groups rows in the UI and filters retrieval.
    source_file: Mapped[str] = mapped_column(Text, nullable=False)
    # Zero-based index of the chunk within its source document.
    chunk_id: Mapped[int] = mapped_column(Integer, nullable=False)
    # Which strategy produced this row ('fixed', 'sentence_aware', ...) — lets
    # multiple strategies coexist for the same file.
    chunking_strategy: Mapped[str | None] = mapped_column(Text, nullable=True, server_default="fixed")
    chunk_text: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)
    # Fixed-dimension pgvector column; dimension comes from settings.EMBEDDING_DIMENSION.
    embedding: Mapped[list[float]] = mapped_column(Vector(settings.EMBEDDING_DIMENSION), nullable=False)
    # Generated (STORED) tsvector maintained by Postgres — powers the keyword
    # half of hybrid search; never written by application code.
    search_vector: Mapped[str | None] = mapped_column(
        TSVECTOR,
        Computed("to_tsvector('english', chunk_text)", persisted=True),
        nullable=True,
    )


class ChatSession(Base):
    """A chat conversation session (title + timestamps; messages live in ChatMessage)."""

    __tablename__ = "chat_sessions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=False), primary_key=True, default=lambda: uuid.uuid4())
    title: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(default=lambda: datetime.now(timezone.utc))
    updated_at: Mapped[datetime] = mapped_column(default=lambda: datetime.now(timezone.utc))

    messages: Mapped[list["ChatMessage"]] = relationship(back_populates="session", cascade="all, delete-orphan")


class ChatMessage(Base):
    """A single message within a chat session."""

    __tablename__ = "chat_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    # DB-level cascade: deleting a session removes its messages.
    session_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=False), ForeignKey("chat_sessions.id", ondelete="CASCADE"))
    role: Mapped[str] = mapped_column(Text, nullable=False)  # "user" or "assistant"
    content: Mapped[str] = mapped_column(Text, nullable=False)
    # JSONB snapshot of {source_file, chunk_id, chunk_text} for assistant turns;
    # None for user messages.
    sources: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(default=lambda: datetime.now(timezone.utc))

    # ORM-side counterpart to the DB cascade above.
    session: Mapped["ChatSession"] = relationship(back_populates="messages")


engine: Any = None
SessionLocal: Any = None


def _default_database_url() -> str:
    """Build the default Postgres URL from the environment settings."""
    return (
        f"postgresql+psycopg://{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}"
        f"@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}"
    )


def init_engine(database_url: str) -> None:
    """Initialize the module-level SQLAlchemy engine and session factory.

    The caller passes the full database URL directly (for example a NeonDB URL),
    which makes the app able to switch DB connection at runtime without reloading
    a module or editing environment values.
    """
    global engine, SessionLocal
    engine = create_engine(database_url, pool_pre_ping=True)
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def _ensure_initialized() -> None:
    """Initialize the default engine lazily when the module is used without UI setup."""
    global engine, SessionLocal
    if engine is None or SessionLocal is None:
        init_engine(_default_database_url())


def ensure_search_vector_column() -> None:
    """Ensure the generated search_vector column and GIN index exist.

    Existing databases may already have the table without the new full-text column.
    In that case we add the migration explicitly instead of silently leaving the
    table in an inconsistent state.
    """
    _ensure_initialized()
    with engine.begin() as conn:
        result = conn.exec_driver_sql(
            """
            SELECT EXISTS (
                SELECT 1
                FROM information_schema.columns
                WHERE table_name = %s AND column_name = 'search_vector'
            );
            """,
            (settings.VECTOR_TABLE,),
        )
        exists = result.scalar()

    if exists:
        return

    print(
        f"Warning: table '{settings.VECTOR_TABLE}' is missing the generated search_vector column. "
        "Existing rows may need a migration before keyword search can be used."
    )

    with engine.begin() as conn:
        conn.exec_driver_sql(
            f"""
            ALTER TABLE {settings.VECTOR_TABLE}
            ADD COLUMN search_vector tsvector
            GENERATED ALWAYS AS (to_tsvector('english', chunk_text)) STORED;
            """
        )
        conn.exec_driver_sql(
            f"""
            CREATE INDEX IF NOT EXISTS ix_{settings.VECTOR_TABLE}_search_vector
            ON {settings.VECTOR_TABLE} USING GIN (search_vector);
            """
        )

    print(f"Migration complete: added search_vector and GIN index for '{settings.VECTOR_TABLE}'.")


def ensure_chunking_strategy_column() -> None:
    """Ensure the chunking strategy metadata column exists and backfills old rows."""
    _ensure_initialized()

    with engine.begin() as conn:
        result = conn.exec_driver_sql(
            """
            SELECT EXISTS (
                SELECT 1
                FROM information_schema.columns
                WHERE table_name = %s AND column_name = 'chunking_strategy'
            );
            """,
            (settings.VECTOR_TABLE,),
        )
        exists = result.scalar()

    if not exists:
        with engine.begin() as conn:
            conn.exec_driver_sql(
                f"ALTER TABLE {settings.VECTOR_TABLE} ADD COLUMN chunking_strategy TEXT DEFAULT 'fixed';"
            )

    with engine.begin() as conn:
        conn.exec_driver_sql(
            f"UPDATE {settings.VECTOR_TABLE} SET chunking_strategy = 'fixed' WHERE chunking_strategy IS NULL;"
        )

    print(f"Migration complete: ensured chunking_strategy column for '{settings.VECTOR_TABLE}'.")


def create_table() -> None:
    """Create the vector table and the search_vector migration if needed."""
    _ensure_initialized()
    Base.metadata.create_all(bind=engine)
    ensure_search_vector_column()
    ensure_chunking_strategy_column()


def store_chunks(
    source_file: str,
    chunks: list[dict],
    embeddings: list[list[float]],
    chunking_strategy: str = "fixed",
) -> None:
    """Insert chunk texts and embeddings into the vector table in a single batch."""
    _ensure_initialized()
    if len(chunks) != len(embeddings):
        raise ValueError("chunks and embeddings must be the same length")

    if not chunks:
        return

    with SessionLocal() as session:
        rows = [
            DocumentChunk(
                source_file=source_file,
                chunk_id=chunk["chunk_id"],
                chunking_strategy=chunking_strategy,
                chunk_text=chunk["text"],
                token_count=chunk["token_count"],
                embedding=embedding,
            )
            for chunk, embedding in zip(chunks, embeddings)
        ]
        session.add_all(rows)
        session.commit()


# ---------------------------------------------------------------------------
# Chat session CRUD
# ---------------------------------------------------------------------------


def create_session(title: str | None = None) -> ChatSession:
    """Create and return a new chat session."""
    _ensure_initialized()
    with SessionLocal() as session:
        chat_session = ChatSession(title=title)
        session.add(chat_session)
        session.commit()
        session.refresh(chat_session)
        return chat_session


def list_sessions() -> list[dict]:
    """Return all chat sessions ordered by most recently updated."""
    _ensure_initialized()
    with SessionLocal() as session:
        rows = (
            session.query(ChatSession)
            .order_by(ChatSession.updated_at.desc())
            .all()
        )
        return [
            {
                "id": str(row.id),
                "title": row.title,
                "created_at": row.created_at.isoformat() if row.created_at else None,
                "updated_at": row.updated_at.isoformat() if row.updated_at else None,
            }
            for row in rows
        ]


def get_messages(session_id: str) -> list[dict]:
    """Return all messages for a session, ordered chronologically."""
    _ensure_initialized()
    with SessionLocal() as session:
        rows = (
            session.query(ChatMessage)
            .filter(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.created_at.asc())
            .all()
        )
        return [
            {
                "id": row.id,
                "role": row.role,
                "content": row.content,
                "sources": row.sources,
                "created_at": row.created_at.isoformat() if row.created_at else None,
            }
            for row in rows
        ]


def add_message(
    session_id: str,
    role: str,
    content: str,
    sources: list[dict] | None = None,
) -> ChatMessage:
    """Append a message to a session and touch the session's updated_at."""
    _ensure_initialized()
    with SessionLocal() as session:
        msg = ChatMessage(
            session_id=session_id,
            role=role,
            content=content,
            sources=sources,
        )
        session.add(msg)
        chat_session = session.get(ChatSession, session_id)
        if chat_session is not None:
            chat_session.updated_at = datetime.now(timezone.utc)
        session.commit()
        session.refresh(msg)
        return msg


def delete_session(session_id: str) -> None:
    """Delete a chat session and all its messages (via cascade)."""
    _ensure_initialized()
    with SessionLocal() as session:
        chat_session = session.get(ChatSession, session_id)
        if chat_session is not None:
            session.delete(chat_session)
            session.commit()


def update_session_title(session_id: str, title: str) -> None:
    """Update the title of a chat session."""
    _ensure_initialized()
    with SessionLocal() as session:
        chat_session = session.get(ChatSession, session_id)
        if chat_session is not None:
            chat_session.title = title
            session.commit()


if __name__ == "__main__":
    _ensure_initialized()
    create_table()
    print(f"Ensured table '{settings.VECTOR_TABLE}' exists with vector dimension {settings.EMBEDDING_DIMENSION}.")
