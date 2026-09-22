"""Ingestion orchestration: load a document, chunk, embed, and store in Postgres.

End-to-end flow used by the background worker, the API, and the CLI:
load text → chunk → embed in batch → store rows in the pgvector-backed table.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

from app.ingestion import store
from app.ingestion.chunker import chunk_text
from app.ingestion.embedder import embed_texts
from app.ingestion.loader import load_filing

DocumentChunk = store.DocumentChunk


def ingest_file(
    filepath: str,
    chunking_strategy: str = "fixed",
    progress_callback: Callable[[str, float], None] | None = None,
    extracted_text: str | None = None,
) -> int:
    """Load a filing, create chunks, embed them, and store the result in Postgres.

    Step 1: read the document text from a local file via load_filing().
    Step 2: split the text into token-based chunks with chunk_text().
    Step 3: convert the chunk list into plain text strings and batch-embed them.
    Step 4: store the chunk metadata and embeddings into the vector table.
    """
    # Progress percentages are coarse stage markers (load 5% → chunk 20% →
    # embed 20-85% → store 95% → done 100%) mapped by the UI pipeline display.
    def report(message: str, progress: float) -> None:
        if progress_callback is not None:
            progress_callback(message, progress)

    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Filing not found: {filepath}")

    report("Loading document... (5%)", 0.05)
    if extracted_text is not None:
        raw_text = extracted_text
    else:
        try:
            raw_text = load_filing(str(path))
        except Exception as exc:
            raise RuntimeError(f"Failed to extract text from {path.name}: {exc}") from exc
    report("Chunking... (20%)", 0.2)
    chunks = chunk_text(raw_text, strategy=chunking_strategy)
    print(f"Loaded {filepath}: produced {len(chunks)} chunks")

    # Zero-length input produced no chunks — nothing to embed or store.
    if not chunks:
        print(f"No chunks generated for {filepath}; skipping storage.")
        return 0

    texts = [chunk["text"] for chunk in chunks]
    def report_embedding_progress(current: int, total: int) -> None:
        percentage = round(current / total * 100) if total else 100
        if current == 0:
            report(f"Embedding: 0/{total} chunks (0%)", 0.2)
        else:
            report(
                f"Embedding: {current}/{total} chunks ({percentage}%)",
                0.2 + 0.65 * current / total,
            )

    embeddings = embed_texts(texts, progress_callback=report_embedding_progress)

    if len(embeddings) != len(chunks):
        raise ValueError(
            f"Embedding count mismatch for {filepath}: {len(embeddings)} embeddings for {len(chunks)} chunks"
        )

    # Store only the filename (not the temp path) — this becomes
    # DocumentChunk.source_file shown in the UI and used for retrieval filters.
    source_file = path.name
    report("Storing... (95%)", 0.95)
    store.store_chunks(source_file, chunks, embeddings, chunking_strategy=chunking_strategy)
    report("Complete (100%)", 1.0)
    print(f"Stored {len(chunks)} chunks for {source_file} in the vector database.")
    return len(chunks)


def file_already_ingested(source_file: str, chunking_strategy: str | None = None) -> bool:
    """Return True if the given source_file and chunking strategy are already present."""
    with store.SessionLocal() as session:
        query = session.query(DocumentChunk).filter(DocumentChunk.source_file == source_file)
        if chunking_strategy is not None:
            query = query.filter(DocumentChunk.chunking_strategy == chunking_strategy)
        return query.first() is not None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingest a local filing file into the pgvector-backed RAG database."
    )
    parser.add_argument("filepath", nargs="?", help="Path to the local PDF/HTML/TXT file to ingest.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip the duplicate-ingestion confirmation prompt.",
    )
    parser.add_argument(
        "--chunking-strategy",
        choices=["fixed", "sentence_aware", "paragraph_based", "recursive"],
        default="fixed",
        help="Chunking strategy to use for this ingestion run.",
    )
    args = parser.parse_args()

    if not args.filepath:
        parser.print_usage()
        print("Usage: python -m app.ingestion.ingest <file_path> [--force]")
        raise SystemExit(1)

    file_path = Path(args.filepath)
    if not file_path.exists():
        print(f"File not found: {args.filepath}")
        raise SystemExit(1)

    try:
        store.create_table()
        print("Database table ready.")
    except Exception as exc:  # pragma: no cover - CLI script convenience
        print(f"Failed to create table: {exc}")
        raise SystemExit(1)

    # Interactive re-ingest guard for the CLI path only; UI/API callers skip this.
    source_name = file_path.name
    if file_already_ingested(source_name, chunking_strategy=args.chunking_strategy) and not args.force:
        response = input(
            f"Warning: {source_name} already exists with chunking strategy '{args.chunking_strategy}' in the vector database. Re-ingest anyway? [y/N]: "
        ).strip().lower()
        if response not in {"y", "yes"}:
            print("Skipping re-ingestion.")
            raise SystemExit(0)

    try:
        ingest_file(str(file_path), chunking_strategy=args.chunking_strategy)
    except Exception as exc:
        print(f"Failed to ingest {args.filepath}: {exc}")
        raise SystemExit(1)
