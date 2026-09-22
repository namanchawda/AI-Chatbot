# Multi-Document RAG Chatbot

## Overview

A multi-document Retrieval-Augmented Generation (RAG) chatbot: upload PDF, HTML, or text documents, then chat with an AI that answers questions grounded in those documents. Ingestion runs as a background job with live progress, retrieval uses hybrid search (dense vectors + PostgreSQL full-text) with optional cross-encoder reranking, and generation is constrained to the retrieved context so the model refuses rather than guesses. Chat sessions and messages persist across restarts and are browsable in the UI. The stack is FastAPI + Streamlit on the front, Postgres/pgvector for storage, local sentence-transformers for embeddings and reranking, and Groq for answer generation.

## Setup

### Requirements

- Python **3.11** (see `runtime.txt`; newer 3.x should work — this repo was also verified on 3.13)
- A Postgres database with the **pgvector** extension (e.g. [Neon](https://neon.tech) free tier)
- A [Groq](https://console.groq.com) API key

### Install

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

### Environment variables

Copy the example file and fill in real values:

```bash
cp .env.example .env
```

| Variable | Purpose |
|---|---|
| `POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_DB`, `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_SSLMODE` | Fallback DB settings for the API/CLI when no UI connection is active |
| `GROQ_API_KEY` | Groq API key (required for generation) |
| `GROQ_MODEL` | Groq model id (`.env.example` sets `openai/gpt-oss-120b`) |
| `EMBEDDING_MODEL` | Local embedding model (default `BAAI/bge-small-en-v1.5`) |
| `EMBEDDING_DIMENSION` | Embedding vector size (default `384`) |
| `VECTOR_TABLE` | Chunk table name (default `document_chunks`) |
| `QUERY_TOP_K` | Default retrieval depth (default `5`) |

Notes:

- The **Streamlit UI has its own connection screen** (Postgres connection string + Groq key/model). Values entered there override `.env` at runtime and are cached in `.streamlit_connection.json`. `.env` still matters for the FastAPI backend and any scripts run outside the UI.
- The database must have pgvector enabled: `CREATE EXTENSION IF NOT EXISTS vector;`

### Run

Start the FastAPI backend (serves `/api/*` and `/health`; tables are auto-created on startup — no manual migration step):

```bash
uvicorn app.main:app --reload --port 8000
```

Start the Streamlit UI (separate terminal, same venv):

```bash
streamlit run app/ui/streamlit_app.py
```

On first UI load, paste your Postgres connection string and Groq API key, click **Connect**, then use the **Ingest Documents** page to upload files and the **Chat** page to ask questions. All DB tables (`document_chunks`, `chat_sessions`, `chat_messages`) are created automatically on startup and on connect.

## Tech Stack

| Layer | Technology |
|---|---|
| Backend API | FastAPI `0.115.0`, Uvicorn `0.30.6`, Pydantic `2.9.2` |
| UI | Streamlit `1.39.0` |
| Vector + relational DB | PostgreSQL + pgvector `0.2.5` (Neon or self-hosted) |
| ORM / driver | SQLAlchemy `2.0.35`, psycopg `3.x` (`psycopg[binary]>=3.2.2`) |
| Embeddings | `BAAI/bge-small-en-v1.5` via sentence-transformers `3.2.1` (local, 384-dim) |
| Reranking | `BAAI/bge-reranker-base` CrossEncoder via sentence-transformers (local) |
| LLM | Groq API (`groq==0.13.0`) — model configured as `GROQ_MODEL`; `.env.example` / UI use `openai/gpt-oss-120b` |
| Document parsing | PyMuPDF (PDF), BeautifulSoup4 `4.12.3` (HTML) |
| Tokenization / chunking | tiktoken `0.8.0` (cl100k_base) |
| Misc | python-dotenv `1.0.1`, python-multipart `0.0.20`, torch `>=2.6.0`, transformers `4.45.2` |

Exact pins are in `requirements.txt`.

## Architecture

### (a) Ingestion

```
upload → temp file (OS temp dir) → text extraction → chunking → embedding → Postgres
                                                                              ↓
                                              temp file deleted (success or failure)
```

1. The file is written to a fresh `tempfile.mkdtemp()` directory (never persisted into the repo or a long-lived data dir).
2. Text is extracted by type: PDF via PyMuPDF, HTML via BeautifulSoup (scripts/styles/iXBRL tags stripped, encoding fallback chain), plain text with encoding fallback.
3. Text is split with one of four strategies (selected per upload):
   - **`fixed`** — token windows, 512 tokens / 50-token overlap
   - **`sentence_aware`** — groups whole sentences up to a token budget; never splits mid-sentence
   - **`paragraph_based`** — splits on paragraph breaks, falls back to sentence-aware for oversized paragraphs
   - **`recursive`** — paragraph → sentence → fixed, in priority order
4. Chunks are embedded in batches with the local sentence-transformers model.
5. Rows are inserted into `document_chunks` (text, metadata, embedding, generated `tsvector`).
6. The temp file and its temp directory are removed in a `finally` block — in the UI path this runs in the background worker process; in the API path it runs at the end of the request.

UI uploads are processed by a **daemon background process** with status persisted to `.ingestion_status.json` (the page polls it every 2s). The API `POST /api/ingest` endpoint runs ingestion synchronously in the request.

### (b) Retrieval

Hybrid search across **all ingested documents by default** (no per-document filter unless one is passed explicitly on the API):

1. **Vector search** — embed the query, cosine distance against stored pgvector embeddings.
2. **Keyword search** — PostgreSQL full-text (`plainto_tsquery` / `ts_rank` over the generated `tsvector`).
3. The two ranked lists are merged with **Reciprocal Rank Fusion** (RRF, `k = 60`), running both retrievers concurrently.
4. **Optional reranking** — a `BAAI/bge-reranker-base` cross-encoder re-scores the fused candidates and narrows to `top_k`. Toggleable per request (`use_reranking`); the Chat UI always enables it.

### (c) Generation

Retrieved chunks are formatted into a grounded prompt (each chunk labeled with source file + chunk id for the model's reference only) and sent to Groq with instructions to answer **only** from that context and to refuse with a fixed phrase when the answer is absent. Low temperature (`0.2`) and a capped completion length keep answers close to the context.

**Important current behavior:** chat history is **not** sent to the LLM. Every request calls the pipeline with `chat_history=None`, so each message is answered as single-turn Q&A. Prior turns are still written to the database for display (see below) — the plumbing to feed them back exists but is intentionally disabled at all call sites.

### (d) Chat sessions

Two tables, created automatically:

- **`chat_sessions`** — id, optional title (auto-set from the first question, truncated to 80 chars), created/updated timestamps.
- **`chat_messages`** — id, session FK (cascade delete), role (`user` / `assistant`), content, optional `sources` JSONB snapshot (the chunks cited for that assistant turn), timestamp.

Sessions are listed in the sidebar, messages are reloaded into the UI when switched, and history survives restarts — but as noted, it is **saved and browsable, not fed back into the prompt**.

### Flow (end to end)

```
                 ┌────────────── Ingestion ──────────────┐
 upload ──► temp file ──► extract ──► chunk ──► embed ──► Postgres
                 │                                         │ delete temp
                 └─────────────────────────────────────────┘

                 ┌────────────── Query / Chat ───────────┐
 question ──► hybrid search (vector ⊕ keyword, RRF) ──► optional rerank
        ──► build prompt (context + question, no history) ──► Groq ──► answer
        ──► persist user + assistant messages to chat_messages
                 └─────────────────────────────────────────┘
```

## AI Usage

Two distinct roles for AI in this project:

1. **AI used to build this project** — developed with AI-assisted tooling (Codex / OpenCode) for implementation, debugging, and refactoring, guided by the project author's architecture decisions, requirements, and manual verification (running the stack, inspecting retrieved chunks, timing extraction, etc.). AI wrote and iterated on code; humans specified behavior and validated results.
2. **AI used inside the running app** — at runtime, user questions are answered by a Groq-hosted LLM (`GROQ_MODEL`, e.g. `openai/gpt-oss-120b`) conditioned on retrieved document context. Embeddings and reranking run locally via sentence-transformers and do not call an external LLM.

## Assumptions

Practical constraints to know before using or deploying this:

- **No authentication or per-user isolation.** All uploaded documents land in one shared knowledge base; any client that can reach the API/UI sees and queries everything.
- **Chat history is disabled for the LLM.** Each message is answered independently (single-turn per message). History is persisted and shown in the UI but not included in the prompt — by design in the current code, not a bug.
- **Supported file types.** UI uploader: PDF, HTML, HTM, TXT. API `POST /api/ingest` additionally accepts `.md` and `.rtf`. The UI shows a warning for files over **5 MB** (soft warning only — there is no hard server-side size cap beyond available memory/disk).
- **Single Postgres/Neon instance.** No multi-region, sharding, or connection-pool tuning for high concurrency; one engine per process with `pool_pre_ping`.
- **Reranking is optional but on by default** (API default `use_reranking=true`; Chat UI always on). The cross-encoder is loaded once at import and adds per-query latency proportional to candidate count.
- **Embeddings and reranking are local and CPU-bound** unless a GPU-capable torch build is installed. First model load downloads weights from Hugging Face; large ingests can take minutes on CPU (progress is shown).
- **No rate limiting, quota, or abuse protection** on the API. Generation cost/latency is whatever Groq + retrieval incur per request.
- **API ingestion is synchronous** — `POST /api/ingest` blocks until the file is fully ingested. Long documents will hold the HTTP connection open. (The Streamlit path uses a background worker instead.)
- **Background ingestion status is file-based** (`.ingestion_status.json`, stale after 30 minutes). It assumes a single worker at a time; concurrent ingests from multiple clients are not coordinated.
- **Connection secrets for the UI are cached on disk** in `.streamlit_connection.json` (repo root, gitignored). Treat any shared deployment accordingly.
