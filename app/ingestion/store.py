"""Storage layer for persisting embedded document chunks to pgvector.

This module creates the vector-retrieval, account, and chat tables and inserts
chunk rows in batches to keep ingestion fast and efficient.

Every chunk and chat session is owned by exactly one account (``users``), so
retrieval can be scoped to the logged-in user at the SQL level.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

import bcrypt
from pgvector.sqlalchemy import Vector
from sqlalchemy import Computed, ForeignKey, Index, Integer, Text, create_engine
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR, UUID
from sqlalchemy.engine import URL
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker

from app.config import settings

# Sentinel account for rows that carry no user_id — i.e. command-line
# ingestions, which have no logged-in owner. Its password hash is of a random
# value, so the account can never be logged into. Pre-auth rows are deleted by
# the migration instead of being attributed here.
LEGACY_USER_ID = "00000000-0000-0000-0000-000000000000"
LEGACY_USER_EMAIL = "legacy@local.invalid"


class Base(DeclarativeBase):
    """Base class for SQLAlchemy models."""


class User(Base):
    """A registered account. Documents and chat sessions belong to exactly one."""

    __tablename__ = "users"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    email: Mapped[str] = mapped_column(Text, nullable=False, unique=True, index=True)
    # bcrypt digest — plaintext passwords are never stored.
    password_hash: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(default=lambda: datetime.now(timezone.utc))


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
        # Explicitly named so the migration below and create_all() agree.
        Index(f"ix_{settings.VECTOR_TABLE}_user_id", "user_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    # Owning account — chunks are private to one user and filtered in SQL.
    user_id: Mapped[str] = mapped_column(UUID(as_uuid=False), ForeignKey("users.id"), nullable=False)
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
    __table_args__ = (Index("ix_chat_sessions_user_id", "user_id"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=False), primary_key=True, default=lambda: uuid.uuid4())
    # Owning account — the session list is filtered to the logged-in user.
    user_id: Mapped[str] = mapped_column(UUID(as_uuid=False), ForeignKey("users.id"), nullable=False)
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
    """Build the Postgres URL from settings — the single source of truth.

    settings.POSTGRES_* comes from .env locally and from st.secrets on
    Streamlit Cloud. sslmode is appended so hosted providers (Neon) keep the
    TLS requirement that the old pasted connection string carried.
    """
    return URL.create(
        "postgresql+psycopg",
        username=settings.POSTGRES_USER,
        password=settings.POSTGRES_PASSWORD,
        host=settings.POSTGRES_HOST,
        port=settings.POSTGRES_PORT,
        database=settings.POSTGRES_DB,
        query={"sslmode": settings.POSTGRES_SSLMODE or "prefer"},
    ).render_as_string(hide_password=False)


def init_engine() -> None:
    """Initialize the module-level SQLAlchemy engine and session factory.

    There is no runtime override: the engine is always built from
    settings.POSTGRES_* (.env locally, st.secrets on Streamlit Cloud).
    """
    global engine, SessionLocal
    engine = create_engine(_default_database_url(), pool_pre_ping=True)
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def _ensure_initialized() -> None:
    """Initialize the default engine lazily when the module is used without UI setup."""
    global engine, SessionLocal
    if engine is None or SessionLocal is None:
        init_engine()


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


# ---------------------------------------------------------------------------
# Account ownership migration (user_id on existing tables)
# ---------------------------------------------------------------------------


def _column_exists(conn: Any, table_name: str, column_name: str) -> bool:
    """Return True when the given column already exists in the given table."""
    result = conn.exec_driver_sql(
        """
        SELECT EXISTS (
            SELECT 1
            FROM information_schema.columns
            WHERE table_name = %s AND column_name = %s
        );
        """,
        (table_name, column_name),
    )
    return bool(result.scalar())


def _foreign_key_on_column_exists(conn: Any, table_name: str, column_name: str) -> bool:
    """Return True when any foreign key already covers the given column.

    SQLAlchemy names fresh-constraint FKs itself (e.g. ``document_chunks_user_id_fkey``),
    so we detect *any* FK on the column rather than a fixed name to avoid
    stacking a duplicate constraint on a newly created table.
    """
    result = conn.exec_driver_sql(
        """
        SELECT EXISTS (
            SELECT 1
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage kcu
              ON kcu.constraint_name = tc.constraint_name
             AND kcu.table_schema = tc.table_schema
            WHERE tc.table_name = %s
              AND tc.constraint_type = 'FOREIGN KEY'
              AND kcu.column_name = %s
        );
        """,
        (table_name, column_name),
    )
    return bool(result.scalar())


def ensure_legacy_user() -> str:
    """Return the sentinel account's id, creating the account if missing.

    Used when a row is written without a user_id (the CLI has no session), so
    the NOT NULL + FOREIGN KEY constraints are still satisfied. The hash is of
    a random value and the email domain is reserved, so the account cannot be
    logged into.
    """
    _ensure_initialized()
    with SessionLocal() as session:
        if session.get(User, LEGACY_USER_ID) is None:
            session.add(
                User(
                    id=LEGACY_USER_ID,
                    email=LEGACY_USER_EMAIL,
                    password_hash=_hash_password(uuid.uuid4().hex),
                )
            )
            try:
                session.commit()
            except IntegrityError:
                # Another process created it first; the row is there now.
                session.rollback()
    return LEGACY_USER_ID


def _migrate_user_id(table_name: str) -> None:
    """Add the user_id column on one table (see ensure_user_id_columns)."""
    with engine.begin() as conn:
        if not _column_exists(conn, table_name, "user_id"):
            # Nullable first so pre-auth rows can be identified, then removed.
            conn.exec_driver_sql(f"ALTER TABLE {table_name} ADD COLUMN user_id UUID;")

    with engine.begin() as conn:
        # Rows written before per-user ownership have no owner to attribute
        # them to, so they are deleted. chat_messages normally cascade with
        # their sessions; the explicit delete keeps that true if an existing
        # FK was created without ON DELETE CASCADE.
        if table_name == "chat_sessions":
            conn.exec_driver_sql(
                "DELETE FROM chat_messages WHERE session_id IN "
                "(SELECT id FROM chat_sessions WHERE user_id IS NULL);"
            )
        deleted = conn.exec_driver_sql(
            f"DELETE FROM {table_name} WHERE user_id IS NULL;"
        ).rowcount
        conn.exec_driver_sql(
            f"ALTER TABLE {table_name} ALTER COLUMN user_id SET NOT NULL;"
        )
        if not _foreign_key_on_column_exists(conn, table_name, "user_id"):
            conn.exec_driver_sql(
                f"""
                ALTER TABLE {table_name}
                ADD CONSTRAINT fk_{table_name}_user_id
                FOREIGN KEY (user_id) REFERENCES users (id);
                """
            )
        conn.exec_driver_sql(
            f"CREATE INDEX IF NOT EXISTS ix_{table_name}_user_id ON {table_name} (user_id);"
        )

    print(
        f"Migration complete: ensured user_id column for '{table_name}' "
        f"({deleted} pre-auth row(s) removed)."
    )


def ensure_user_id_columns() -> None:
    """Ensure document_chunks and chat_sessions carry an owned user_id column.

    Follows the same detect-then-ALTER pattern as ensure_chunking_strategy_column():
    information_schema tells us whether the column is already there, and only
    then do we migrate. Rows that pre-date per-user ownership are deleted (the
    agreed clean-slate policy), after which NOT NULL and the FOREIGN KEY to
    users are applied, so an already-populated database is upgraded in place
    rather than assumed fresh. Re-running is a no-op.
    """
    _ensure_initialized()
    for table_name in (settings.VECTOR_TABLE, "chat_sessions"):
        _migrate_user_id(table_name)


def create_table() -> None:
    """Create the vector table and the search_vector / user_id migrations if needed."""
    _ensure_initialized()
    Base.metadata.create_all(bind=engine)
    ensure_search_vector_column()
    ensure_chunking_strategy_column()
    ensure_user_id_columns()


def store_chunks(
    source_file: str,
    chunks: list[dict],
    embeddings: list[list[float]],
    chunking_strategy: str = "fixed",
    user_id: str | None = None,
) -> None:
    """Insert chunk texts and embeddings into the vector table in a single batch.

    user_id tags every row with its owning account. When omitted (the CLI has
    no signed-in user) the rows go to the sentinel account, which satisfies the
    NOT NULL/FK constraints and is invisible to every signed-in account.
    """
    _ensure_initialized()
    if len(chunks) != len(embeddings):
        raise ValueError("chunks and embeddings must be the same length")

    if not chunks:
        return

    owner_id = user_id or ensure_legacy_user()

    with SessionLocal() as session:
        rows = [
            DocumentChunk(
                user_id=owner_id,
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
# Accounts (signup / login)
# ---------------------------------------------------------------------------

# Deliberately conservative: length only. Complexity rules, email verification
# and login rate limiting are intentionally not implemented yet.
MIN_PASSWORD_LENGTH = 8


def _hash_password(password: str) -> str:
    """Return a bcrypt digest of the password (salted, never reversible)."""
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def _normalize_email(email: str) -> str:
    """Trim and lowercase an email so lookups are case-insensitive."""
    return email.strip().lower()


def create_user(email: str, password: str) -> User:
    """Create an account and return it.

    Raises ValueError with a UI-friendly message when the email is already
    registered or the password is too short. The plaintext password never
    leaves this function.
    """
    normalized_email = _normalize_email(email)
    if not normalized_email:
        raise ValueError("Email is required.")
    if len(password) < MIN_PASSWORD_LENGTH:
        raise ValueError(f"Password must be at least {MIN_PASSWORD_LENGTH} characters long.")

    _ensure_initialized()
    with SessionLocal() as session:
        if session.query(User).filter(User.email == normalized_email).first() is not None:
            raise ValueError("An account with that email already exists.")

        user = User(
            id=str(uuid.uuid4()),
            email=normalized_email,
            password_hash=_hash_password(password),
        )
        session.add(user)
        try:
            session.commit()
        except IntegrityError as exc:
            # UNIQUE(email) lost a race with a concurrent signup.
            session.rollback()
            raise ValueError("An account with that email already exists.") from exc
        session.refresh(user)
        return user


def verify_user(email: str, password: str) -> User | None:
    """Return the account when the email/password combination matches, else None.

    Unknown email and wrong password both return None so callers cannot tell
    which one failed (no account enumeration through the login form).
    """
    normalized_email = _normalize_email(email)
    if not normalized_email or not password:
        return None

    _ensure_initialized()
    with SessionLocal() as session:
        user = session.query(User).filter(User.email == normalized_email).first()
        if user is None:
            return None
        try:
            matches = bcrypt.checkpw(password.encode("utf-8"), user.password_hash.encode("utf-8"))
        except ValueError:
            # Stored value is not a valid bcrypt digest (e.g. corrupted row).
            return None
        return user if matches else None


def _safe_uuid(value: Any) -> str | None:
    """Return value when it is a syntactically valid UUID, else None.

    Postgres rejects a non-UUID string with a DataError, which would surface
    as an HTTP 500 for a merely absent record; callers treat None as "not
    found" instead.
    """
    try:
        uuid.UUID(str(value))
    except (ValueError, TypeError, AttributeError):
        return None
    return str(value)


def get_user_by_id(user_id: str) -> User | None:
    """Return the account with the given id, or None."""
    if _safe_uuid(user_id) is None:
        return None
    _ensure_initialized()
    with SessionLocal() as session:
        return session.get(User, user_id)


def get_user_by_email(email: str) -> User | None:
    """Return the account with the given email, or None."""
    normalized_email = _normalize_email(email)
    if not normalized_email:
        return None
    _ensure_initialized()
    with SessionLocal() as session:
        return session.query(User).filter(User.email == normalized_email).first()


# ---------------------------------------------------------------------------
# Chat session CRUD
# ---------------------------------------------------------------------------


def create_session(user_id: str, title: str | None = None) -> ChatSession:
    """Create and return a new chat session owned by the given account."""
    _ensure_initialized()
    with SessionLocal() as session:
        chat_session = ChatSession(user_id=user_id, title=title)
        session.add(chat_session)
        session.commit()
        session.refresh(chat_session)
        return chat_session


def list_sessions(user_id: str) -> list[dict]:
    """Return one account's chat sessions ordered by most recently updated.

    user_id is required — there is no "every session" mode, so a caller that
    omits it raises TypeError and one that passes None/"" raises ValueError.
    """
    if not user_id:
        raise ValueError("user_id is required: chat sessions must be scoped to a single account.")
    _ensure_initialized()
    if _safe_uuid(user_id) is None:
        return []
    with SessionLocal() as session:
        rows = (
            session.query(ChatSession)
            .filter(ChatSession.user_id == user_id)
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


def get_messages(session_id: str, user_id: str) -> list[dict]:
    """Return all messages for a session owned by user_id, ordered chronologically.

    user_id is required: messages of sessions owned by another account come
    back empty rather than leaked, and omitting the argument raises TypeError.
    """
    if not user_id:
        raise ValueError("user_id is required: messages must be scoped to a single account.")
    if _safe_uuid(session_id) is None or _safe_uuid(user_id) is None:
        return []
    _ensure_initialized()
    with SessionLocal() as session:
        rows = (
            session.query(ChatMessage)
            .join(ChatSession, ChatMessage.session_id == ChatSession.id)
            .filter(ChatMessage.session_id == session_id)
            .filter(ChatSession.user_id == user_id)
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
    *,
    user_id: str,
) -> ChatMessage:
    """Append a message to a session owned by user_id and touch updated_at.

    user_id is required: a missing argument raises TypeError, and a session
    owned by any other account (or by nobody) raises ValueError.
    """
    if not user_id:
        raise ValueError("user_id is required: messages must be scoped to a single account.")
    if _safe_uuid(session_id) is None:
        raise ValueError("Chat session not found.")
    _ensure_initialized()
    with SessionLocal() as session:
        chat_session = session.get(ChatSession, session_id)
        if chat_session is None or chat_session.user_id != user_id:
            raise ValueError("Chat session not found.")
        msg = ChatMessage(
            session_id=session_id,
            role=role,
            content=content,
            sources=sources,
        )
        session.add(msg)
        chat_session.updated_at = datetime.now(timezone.utc)
        session.commit()
        session.refresh(msg)
        return msg


def delete_session(session_id: str, user_id: str) -> None:
    """Delete a chat session and all its messages (via cascade).

    user_id is required: a missing argument raises TypeError, and a session
    owned by any other account is left untouched.
    """
    if not user_id:
        raise ValueError("user_id is required: sessions must be scoped to a single account.")
    if _safe_uuid(session_id) is None or _safe_uuid(user_id) is None:
        return
    _ensure_initialized()
    with SessionLocal() as session:
        chat_session = session.get(ChatSession, session_id)
        if chat_session is None:
            return
        if chat_session.user_id != user_id:
            return
        session.delete(chat_session)
        session.commit()


def update_session_title(session_id: str, title: str, *, user_id: str) -> None:
    """Update a session's title, only when user_id owns that session.

    user_id is required: a missing argument raises TypeError, and a session
    owned by any other account is left untouched.
    """
    if not user_id:
        raise ValueError("user_id is required: sessions must be scoped to a single account.")
    if _safe_uuid(session_id) is None or _safe_uuid(user_id) is None:
        return
    _ensure_initialized()
    with SessionLocal() as session:
        chat_session = session.get(ChatSession, session_id)
        if chat_session is None:
            return
        if chat_session.user_id != user_id:
            return
        chat_session.title = title
        session.commit()


if __name__ == "__main__":
    _ensure_initialized()
    create_table()
    print(f"Ensured table '{settings.VECTOR_TABLE}' exists with vector dimension {settings.EMBEDDING_DIMENSION}.")
