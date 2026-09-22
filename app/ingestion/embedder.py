"""Embedding utilities for generating vector representations with sentence-transformers."""

from __future__ import annotations

from collections.abc import Callable

from sentence_transformers import SentenceTransformer

from app.config import settings

# Process-wide singleton: loading a sentence-transformers model is expensive,
# so the first embed_texts() call loads it and later calls reuse it.
MODEL = None


def load_embedding_model(model_name: str = settings.EMBEDDING_MODEL):
    """Return the singleton sentence-transformers model used for document embeddings."""
    global MODEL
    # Reload only when nothing is cached or a different model was requested;
    # _model_card_vars is where sentence-transformers exposes the loaded name.
    if MODEL is None or getattr(MODEL, "_model_card_vars", {}).get("model_name") != model_name:
        MODEL = SentenceTransformer(model_name)
    return MODEL


def embed_texts(
    texts: list[str],
    progress_callback: Callable[[int, int], None] | None = None,
    batch_size: int = 16,
) -> list[list[float]]:
    """Embed texts in small batches and report progress after each completed batch."""
    if not texts:
        return []
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")

    model = load_embedding_model(settings.EMBEDDING_MODEL)
    total = len(texts)
    if progress_callback is not None:
        progress_callback(0, total)

    embeddings = []
    # encode() is called per batch so the progress callback reports accurate
    # counts between batches instead of only at the end.
    for start in range(0, total, batch_size):
        batch_embeddings = model.encode(
            texts[start : start + batch_size],
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=False,
            normalize_embeddings=False,
        )
        embeddings.extend(batch_embeddings)
        if progress_callback is not None:
            progress_callback(min(start + batch_size, total), total)

    return [[float(value) for value in vector] for vector in embeddings]


if __name__ == "__main__":
    sample_texts = [
        "Apple reported strong revenue growth in the Services business.",
        "JPMorgan is focused on risk management and capital allocation.",
    ]

    vectors = embed_texts(sample_texts)
    print(f"Embedded {len(vectors)} texts.")
    print(f"Vector length: {len(vectors[0])}")
    print(f"Expected dimension: {settings.EMBEDDING_DIMENSION}")
    print(f"First vector sample: {vectors[0][:5]}")
