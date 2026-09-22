"""Prompt construction for grounded answering with multi-document context and chat history."""


def build_prompt(
    query: str,
    retrieved_chunks: list[dict],
    chat_history: list[dict] | None = None,
) -> str:
    """Construct a grounded prompt using retrieved context and prior conversation.

    The model is instructed to answer only from the supplied context and to refuse
    to guess when the answer is not present. Each chunk is labeled with its source
    file and chunk id for grounding only; the model must not reproduce those labels
    or any other citations in its visible answer, which is plain prose.

    chat_history, when provided, is a list of {"role": "user"|"assistant", "content": str}
    dicts representing prior turns in the conversation (oldest first).
    """
    context_sections: list[str] = []
    for idx, chunk in enumerate(retrieved_chunks, start=1):
        source_file = chunk.get("source_file", "unknown")
        chunk_id = chunk.get("chunk_id", idx)
        text = chunk.get("chunk_text", "")
        # Label each chunk with its origin for grounding only — the prompt
        # forbids the model from citing these labels back in its answer.
        context_sections.append(
            f"Context {idx} (source_file={source_file}, chunk_id={chunk_id})\n{text.strip()}"
        )
    context_block = "\n\n---\n\n".join(context_sections)

    history_block = ""
    # Optional multi-turn context; skipped entirely when chat_history is
    # None/empty (history is currently disabled at all call sites).
    if chat_history:
        turns: list[str] = []
        for msg in chat_history:
            role_label = "User" if msg.get("role") == "user" else "Assistant"
            turns.append(f"{role_label}: {msg.get('content', '')}")
        history_block = "\n\nConversation so far:\n" + "\n".join(turns) + "\n"

    return (
        "You are a helpful research assistant. Answer the user's question ONLY using the "
        "provided context below. If the answer is not present in the provided context, say "
        "exactly: \"I don't have enough information in the provided context to answer this.\" "
        "Do not use outside knowledge or guess.\n"
        "Answer naturally in plain prose. The source_file and chunk_id labels in the context "
        "are for your reference only: do not insert source citations, footnotes, or bracketed "
        "citation markers of any kind into your answer.\n"
        f"{history_block}\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {query}"
    )
