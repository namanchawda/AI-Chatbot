"""Chunking utilities for splitting document text into retrieval-sized passages.

Strategies split on token windows (cl100k_base), sentence boundaries, or
paragraphs; every strategy returns the shared chunk dict shape
{"text", "chunk_id", "token_count"} consumed by embedding and storage.
"""

from __future__ import annotations

import re

import tiktoken


def _token_count(text: str, encoding=None) -> int:
    """Return the token count for a string using the OpenAI cl100k_base encoding."""
    if not text or not text.strip():
        return 0
    encoding = encoding or tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


def _make_chunk(text: str, chunk_id: int, token_count: int | None = None, encoding=None) -> dict:
    """Normalize a chunk payload to the shared record shape."""
    cleaned = text.strip()
    return {
        "text": cleaned,
        "chunk_id": chunk_id,
        "token_count": token_count if token_count is not None else _token_count(cleaned, encoding),
    }


def chunk_fixed(text: str, chunk_size: int = 512, chunk_overlap: int = 50) -> list[dict]:
    """Split text into fixed-size token windows with overlap."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be non-negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    if not text or not text.strip():
        return []

    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)

    # Windows advance by the non-overlapping portion so consecutive chunks
    # share the overlap region (sliding window).
    step_size = chunk_size - chunk_overlap
    chunks: list[dict] = []
    start = 0
    chunk_id = 0

    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text_value = encoding.decode(chunk_tokens)
        chunks.append(
            {
                "text": chunk_text_value.strip(),
                "chunk_id": chunk_id,
                "token_count": len(chunk_tokens),
            }
        )
        if end == len(tokens):
            break
        start += step_size
        chunk_id += 1

    return chunks


def _split_sentences(text: str) -> list[str]:
    """Split text into sentence-sized units while preserving the original wording."""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    # Split after [.!?] followed by whitespace, or on blank lines; the
    # lookbehind keeps terminal punctuation attached to each sentence.
    parts = re.split(r"(?<=[.!?])\s+|\n\s*\n+", normalized)
    return [part.strip() for part in parts if part and part.strip()]


def chunk_sentence_aware(text: str, max_tokens: int = 512) -> list[dict]:
    """Group whole sentences into chunks without splitting any sentence across chunks."""
    if max_tokens <= 0:
        raise ValueError("max_tokens must be greater than zero")

    if not text or not text.strip():
        return []

    sentences = _split_sentences(text)
    if not sentences:
        return []

    encoding = tiktoken.get_encoding("cl100k_base")
    chunks: list[dict] = []
    current_sentences: list[str] = []
    current_token_count = 0
    chunk_id = 0

    for sentence in sentences:
        sentence_tokens = _token_count(sentence, encoding)

        if sentence_tokens > max_tokens:
            if current_sentences:
                chunks.append(_make_chunk(" ".join(current_sentences), chunk_id, current_token_count, encoding))
                chunk_id += 1
                current_sentences = []
                current_token_count = 0

            # A single sentence larger than the token budget: flush the pending
            # group first, then hard-split it with the fixed strategy so no
            # chunk exceeds max_tokens.
            long_sentence_chunks = chunk_fixed(sentence, chunk_size=max_tokens, chunk_overlap=0)
            for long_chunk in long_sentence_chunks:
                chunks.append(
                    {
                        "text": long_chunk["text"].strip(),
                        "chunk_id": chunk_id,
                        "token_count": long_chunk["token_count"],
                    }
                )
                chunk_id += 1
            continue

        if current_sentences and current_token_count + sentence_tokens > max_tokens:
            chunks.append(_make_chunk(" ".join(current_sentences), chunk_id, current_token_count, encoding))
            chunk_id += 1
            current_sentences = [sentence]
            current_token_count = sentence_tokens
        else:
            current_sentences.append(sentence)
            current_token_count += sentence_tokens

    if current_sentences:
        chunks.append(_make_chunk(" ".join(current_sentences), chunk_id, current_token_count, encoding))

    return chunks


def chunk_paragraph_based(text: str, max_tokens: int = 512) -> list[dict]:
    """Split text by paragraph first, falling back to sentence-aware chunking for oversized paragraphs."""
    if max_tokens <= 0:
        raise ValueError("max_tokens must be greater than zero")

    if not text or not text.strip():
        return []

    encoding = tiktoken.get_encoding("cl100k_base")
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n+", text.strip()) if part and part.strip()]
    if not paragraphs:
        return []

    chunks: list[dict] = []
    chunk_id = 0

    for paragraph in paragraphs:
        paragraph_tokens = _token_count(paragraph, encoding)
        if paragraph_tokens <= max_tokens:
            chunks.append(_make_chunk(paragraph, chunk_id, paragraph_tokens, encoding))
            chunk_id += 1
            continue

        for sentence_chunk in chunk_sentence_aware(paragraph, max_tokens=max_tokens):
            chunks.append(
                {
                    "text": sentence_chunk["text"],
                    "chunk_id": chunk_id,
                    "token_count": sentence_chunk["token_count"],
                }
            )
            chunk_id += 1

    return chunks


def chunk_recursive(text: str, max_tokens: int = 512) -> list[dict]:
    """Try paragraph, then sentence, then fixed-size chunking in order of priority."""
    if max_tokens <= 0:
        raise ValueError("max_tokens must be greater than zero")

    if not text or not text.strip():
        return []

    encoding = tiktoken.get_encoding("cl100k_base")
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n+", text.strip()) if part and part.strip()]
    if not paragraphs:
        return []

    chunks: list[dict] = []
    chunk_id = 0

    for paragraph in paragraphs:
        paragraph_tokens = _token_count(paragraph, encoding)
        if paragraph_tokens <= max_tokens:
            chunks.append(_make_chunk(paragraph, chunk_id, paragraph_tokens, encoding))
            chunk_id += 1
            continue

        # Accept the sentence-level split only when it fully respects the
        # token budget; otherwise fall back to fixed splitting.
        sentence_chunks = chunk_sentence_aware(paragraph, max_tokens=max_tokens)
        if sentence_chunks and all(candidate["token_count"] <= max_tokens for candidate in sentence_chunks):
            for candidate in sentence_chunks:
                chunks.append(
                    {
                        "text": candidate["text"],
                        "chunk_id": chunk_id,
                        "token_count": candidate["token_count"],
                    }
                )
                chunk_id += 1
            continue

        for fixed_chunk in chunk_fixed(paragraph, chunk_size=max_tokens, chunk_overlap=0):
            chunks.append(
                {
                    "text": fixed_chunk["text"],
                    "chunk_id": chunk_id,
                    "token_count": fixed_chunk["token_count"],
                }
            )
            chunk_id += 1

    return chunks


def chunk_text(text: str, strategy: str = "fixed", **kwargs) -> list[dict]:
    """Dispatch to the selected chunking strategy and return consistent chunk dictionaries.

    ``**kwargs`` are forwarded to the selected strategy (e.g. ``chunk_size``,
    ``chunk_overlap``, ``max_tokens``).
    """
    strategy_map = {
        "fixed": chunk_fixed,
        "sentence_aware": chunk_sentence_aware,
        "paragraph_based": chunk_paragraph_based,
        "recursive": chunk_recursive,
    }

    if strategy not in strategy_map:
        valid = ", ".join(sorted(strategy_map))
        raise ValueError(f"Unknown chunking strategy '{strategy}'. Valid strategies: {valid}")

    return strategy_map[strategy](text, **kwargs)


if __name__ == "__main__":
    sample_text = (
        "Management discussed revenue growth and operating leverage across the business. "
        "The company increased investment in product development, enterprise software, and "
        "international expansion. Risk factors include inflation, supply chain constraints, "
        "cybersecurity exposure, and competition. The company also continued to focus on "
        "cash generation and disciplined capital allocation. "
        * 8
    )

    chunks = chunk_text(sample_text, strategy="fixed", chunk_size=128, chunk_overlap=20)
    print(f"Total chunks: {len(chunks)}")
    for chunk in chunks[:3]:
        print(f"Chunk {chunk['chunk_id']} | Tokens: {chunk['token_count']}")
        print(chunk["text"][:200].replace("\n", " "))
        print("-" * 60)
