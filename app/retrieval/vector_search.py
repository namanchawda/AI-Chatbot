"""Vector search logic for retrieving the most relevant document chunks.

Embeds the query once, then runs a pgvector cosine-distance query against
the stored embeddings. Reranking and hybrid fusion live in sibling modules
(reranker.py, hybrid_search.py).
"""

from __future__ import annotations

from sqlalchemy import select
from pgvector.sqlalchemy import Vector

from app.config import settings
from app.ingestion import store
from app.ingestion.embedder import embed_texts

DocumentChunk = store.DocumentChunk


def search(
    query: str,
    top_k: int | None = None,
    source_file: str | list[str] | None = None,
    chunking_strategy: str | None = None,
) -> list[dict]:
    """Return the closest matching chunk records for a query string.

    The query text is embedded once, then compared against the stored pgvector
    embeddings using cosine distance. Results are ordered from closest to farthest.

    source_file can be a single filename, a list of filenames, or None (search all).
    """
    if top_k is None:
        top_k = settings.QUERY_TOP_K

    query_embedding = embed_texts([query])[0]

    # cosine_distance returns 0.0 for identical vectors and 2.0 for opposite
    # ones; ascending order puts the closest matches first.
    stmt = (
        select(
            DocumentChunk.chunk_text.label("chunk_text"),
            DocumentChunk.source_file.label("source_file"),
            DocumentChunk.chunk_id.label("chunk_id"),
            (DocumentChunk.embedding.cosine_distance(query_embedding)).label("distance"),
        )
        .where(DocumentChunk.embedding.isnot(None))
        .order_by(DocumentChunk.embedding.cosine_distance(query_embedding).asc())
        .limit(top_k)
    )

    # Optional SQL filters: single filename (==), list of filenames (IN),
    # and/or a specific chunking strategy — None means "search all".
    if source_file is not None:
        if isinstance(source_file, list):
            stmt = stmt.where(DocumentChunk.source_file.in_(source_file))
        else:
            stmt = stmt.where(DocumentChunk.source_file == source_file)
    if chunking_strategy is not None:
        stmt = stmt.where(DocumentChunk.chunking_strategy == chunking_strategy)

    with store.SessionLocal() as session:
        rows = session.execute(stmt).mappings().all()

    return [
        {
            "chunk_text": row["chunk_text"],
            "source_file": row["source_file"],
            "chunk_id": row["chunk_id"],
            "distance": float(row["distance"]),
        }
        for row in rows
    ]


if __name__ == "__main__":
    queries = [
        "What are Apple's main risk factors?",
        "What is JPMorgan's approach to interest rate risk?",
    ]

    for question in queries:
        matches = search(question, top_k=3)
        print(f"\nQuery: {question}")
        if not matches:
            print("No matching chunks found.")
            continue

        top = matches[0]
        print(f"source_file: {top['source_file']}")
        print(f"chunk_id: {top['chunk_id']}")
        print(f"distance: {top['distance']:.6f}")
        print(f"chunk_text: {top['chunk_text'][:200]}")
