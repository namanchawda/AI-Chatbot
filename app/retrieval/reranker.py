"""Cross-encoder reranking for final retrieval narrowing in the RAG pipeline."""

from __future__ import annotations

from sentence_transformers import CrossEncoder

from app.config import settings


# Loaded once at import time — model download/load is expensive and the
# reranker is reused across every query in the process.
cross_encoder = CrossEncoder("BAAI/bge-reranker-base", max_length=512)


def rerank(query: str, candidates: list[dict], top_k: int = None) -> list[dict]:
    """Re-score retrieval candidates using a cross-encoder and return them sorted by relevance.

    Each candidate dict must include at least: "chunk_text". The function appends a
    new "rerank_score" key to each result and then re-sorts descending by that score.
    """
    if top_k is None:
        top_k = settings.QUERY_TOP_K

    if not candidates:
        return []

    # Score each (query, chunk_text) pair with the cross-encoder.
    pairs = [(query, candidate.get("chunk_text", "")) for candidate in candidates]
    scores = cross_encoder.predict(pairs, show_progress_bar=False, convert_to_numpy=True)

    reranked = []
    for candidate, score in zip(candidates, scores):
        # Shallow-copy before mutating so the caller's candidate dicts are untouched.
        item = dict(candidate)
        item["rerank_score"] = float(score)
        reranked.append(item)

    # Re-sort descending by the new score and trim to top_k.
    reranked.sort(key=lambda row: row["rerank_score"], reverse=True)
    return reranked[:top_k]


if __name__ == "__main__":
    query = "What are the key risks in a global company facing supply-chain and regulatory pressure?"
    candidates = [
        {
            "chunk_text": "The company is exposed to rising commodity prices, supply disruptions, and global logistics constraints.",
            "source_file": "doc_a.txt",
            "chunk_id": 1,
        },
        {
            "chunk_text": "This section discusses the company's cash management policies and how it funds operations.",
            "source_file": "doc_b.txt",
            "chunk_id": 2,
        },
        {
            "chunk_text": "The firm faces geopolitical and regulatory challenges, including export controls and compliance risk.",
            "source_file": "doc_c.txt",
            "chunk_id": 3,
        },
        {
            "chunk_text": "This draft explains the company's office renovation plan and employee commute benefits.",
            "source_file": "doc_d.txt",
            "chunk_id": 4,
        },
    ]

    print("Before rerank:")
    for index, candidate in enumerate(candidates, start=1):
        print(f"{index}. {candidate['chunk_text']}")

    reranked = rerank(query, candidates, top_k=3)

    print("\nAfter rerank:")
    for index, candidate in enumerate(reranked, start=1):
        print(f"{index}. score={candidate['rerank_score']:.4f} | {candidate['chunk_text']}")
