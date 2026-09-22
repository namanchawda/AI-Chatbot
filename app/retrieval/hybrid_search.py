"""Hybrid retrieval combining dense vector search and keyword search via RRF."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from app.retrieval.keyword_search import keyword_search
from app.retrieval.vector_search import search


def reciprocal_rank_fusion(vector_results: list[dict], keyword_results: list[dict], k: int = 60) -> list[dict]:
    """Merge ranked results from two retrieval methods using Reciprocal Rank Fusion.

    Each result contributes 1/(k + rank) to its chunk's fused score; a chunk
    appearing in both lists has both contributions summed. k=60 is the standard
    RRF constant that dampens the weight of top-ranked items. Results are keyed
    by (source_file, chunk_id) and returned sorted by fused score descending.
    """
    fused: dict[tuple[str, int], float] = {}
    seen: set[tuple[str, int]] = set()

    # Score each list independently, then sum contributions for duplicates.
    for rank, row in enumerate(vector_results, start=1):
        key = (row["source_file"], row["chunk_id"])
        seen.add(key)
        fused[key] = fused.get(key, 0.0) + 1.0 / (k + rank)

    for rank, row in enumerate(keyword_results, start=1):
        key = (row["source_file"], row["chunk_id"])
        seen.add(key)
        fused[key] = fused.get(key, 0.0) + 1.0 / (k + rank)

    # Rebuild full rows for every seen key, keeping vector distance and
    # keyword rank_score side by side (either may be None if the chunk
    # appeared in only one of the two lists).
    merged = []
    for key in seen:
        source_file, chunk_id = key
        vector_row = next((row for row in vector_results if row["source_file"] == source_file and row["chunk_id"] == chunk_id), None)
        keyword_row = next((row for row in keyword_results if row["source_file"] == source_file and row["chunk_id"] == chunk_id), None)

        merged.append(
            {
                "chunk_text": vector_row["chunk_text"] if vector_row is not None else keyword_row["chunk_text"],
                "source_file": source_file,
                "chunk_id": chunk_id,
                "rrf_score": fused[key],
                "distance": vector_row["distance"] if vector_row is not None else None,
                "rank_score": keyword_row["rank_score"] if keyword_row is not None else None,
            }
        )

    merged.sort(key=lambda row: row["rrf_score"], reverse=True)
    return merged


def hybrid_search(
    query: str,
    top_k: int = None,
    source_file: str | list[str] | None = None,
    chunking_strategy: str | None = None,
    wide_candidate_k: int = 10,
) -> list[dict]:
    """Combine dense vector retrieval and keyword retrieval using RRF.

    Dense and keyword retrieval are allowed to each retrieve wider candidate sets,
    then their ranks are merged to produce the final top_k list.

    source_file can be a single filename, a list of filenames, or None (search all).
    """
    if top_k is None:
        top_k = 5
    if wide_candidate_k <= 0:
        raise ValueError("wide_candidate_k must be greater than zero")

    # Fan out to both retrievers concurrently (one worker each), then join.
    with ThreadPoolExecutor(max_workers=2) as executor:
        vector_future = executor.submit(
            search,
            query=query,
            top_k=wide_candidate_k,
            source_file=source_file,
            chunking_strategy=chunking_strategy,
        )
        keyword_future = executor.submit(
            keyword_search,
            query=query,
            top_k=wide_candidate_k,
            source_file=source_file,
            chunking_strategy=chunking_strategy,
        )
        vector_candidates = vector_future.result()
        keyword_candidates = keyword_future.result()

    fused = reciprocal_rank_fusion(vector_candidates, keyword_candidates, k=60)
    # Truncate to at least top_k rows, but never fewer than the wide candidate
    # count — so callers requesting a small top_k still receive the full fused
    # set when wide_candidate_k is larger.
    final = fused[: max(top_k, wide_candidate_k)]

    return [
        {
            "chunk_text": row["chunk_text"],
            "source_file": row["source_file"],
            "chunk_id": row["chunk_id"],
            "distance": row["distance"],
            "rank_score": row["rank_score"],
            "rrf_score": row["rrf_score"],
        }
        for row in final
    ]


if __name__ == "__main__":
    query = "What are Apple's main risk factors?"
    results = hybrid_search(query, top_k=5)
    print(f"Query: {query}\n")
    for idx, row in enumerate(results, start=1):
        print(f"{idx}. {row['source_file']} | chunk_id={row['chunk_id']} | rrf={row['rrf_score']:.6f}")
        print(row['chunk_text'][:220])
        print("---")
