"""Command-line query utility for the multi-document RAG pipeline."""

from __future__ import annotations

import argparse

from app.generation.rag_pipeline import answer_question
from app.ingestion import store


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the query utility."""
    parser = argparse.ArgumentParser(
        description="Ask a question against the indexed SEC filing database."
    )
    parser.add_argument("question", help="The question to answer using the RAG pipeline.")
    parser.add_argument(
        "--source",
        dest="source_file",
        help="Optional source filing filename to filter retrieval, e.g. jpm_10k_2025.html",
    )
    return parser.parse_args()


def main() -> None:
    """Run the RAG pipeline on the parsed question and print the answer with sources."""
    args = parse_args()
    result = answer_question(args.question, source_file=args.source_file)

    print("\nQuestion:")
    print(args.question)
    print("\nAnswer:")
    print(result["answer"])
    print("\nSources:")
    if result.get("sources"):
        for idx, source in enumerate(result["sources"], start=1):
            print(f"{idx}. {source['source_file']} | chunk_id={source['chunk_id']} | distance={source['distance']}")
    else:
        print("No sources retrieved.")


if __name__ == "__main__":
    store._ensure_initialized()
    main()
