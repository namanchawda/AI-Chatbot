# RAG System — Document Q&A with Hybrid Search & Reranking

A general-purpose Retrieval-Augmented Generation (RAG) system: upload any 
PDF or HTML document, and ask questions about it. Built to understand RAG 
systems end-to-end — including real failure modes, how to diagnose them, 
and how to fix them with evidence, not guesswork.

**Live demo:** https://hybrid-rag-engine.streamlit.app

## What this project demonstrates

- Full RAG pipeline: ingestion → chunking → embedding → vector storage → 
  hybrid retrieval → reranking → grounded generation
- A real, reproducible bug found, diagnosed, and fixed with evidence (see below)
- Multiple selectable chunking strategies, compared head-to-head on the same failure case
- A user-facing UI with zero hardcoded credentials or files — bring your own database and API key

## Architecture
Upload any PDF/HTML document
→ Extract & clean text (strip HTML noise, fix encoding)
→ Chunk (4 selectable strategies — see below)
→ Embed (BAAI/bge-small-en-v1.5, local, 384-dim)
→ Store (Postgres + pgvector, hosted on Neon)

Question
→ Embed query
→ Hybrid retrieval (dense vector search + PostgreSQL full-text search using
`tsvector`/`ts_rank`, merged via Reciprocal Rank Fusion)
→ Rerank (cross-encoder, BAAI/bge-reranker-base) — toggleable
→ Grounded generation (Groq/Llama, instructed to answer only from retrieved
context, refuse if not present), streamed token-by-token
→ Answer + cited sources

## Tech stack

- **Backend:** Python, SQLAlchemy, psycopg3
- **Vector DB:** PostgreSQL + pgvector (via [Neon](https://neon.tech), free tier)
- **Embeddings:** `BAAI/bge-small-en-v1.5` (local, sentence-transformers)
- **Reranker:** `BAAI/bge-reranker-base` (local cross-encoder)
- **PDF parsing:** PyMuPDF (previously pypdf — replaced for performance on complex/image-heavy PDFs, see below)
- **LLM:** Groq API (`openai/gpt-oss-120b`)
- **UI:** Streamlit
- **Full-text search:** PostgreSQL native `tsvector`/`ts_rank`

## A real bug, found and fixed: chunk-boundary fact splitting

While testing against a real document, a direct factual question consistently 
failed with "I don't have enough information," even though the fact was 
clearly present and the correct chunk was being retrieved with high confidence 
(rerank score 0.85+).

**Diagnosis:** inspecting the retrieved chunk directly showed it began mid-sentence 
— the actual fact had been split across two chunks by the naive fixed-size 
chunker (512 tokens, 50-token overlap). The overlap window was smaller than the 
distance from the sentence's start to the cut point, so overlap alone couldn't 
guarantee the fact stayed intact — a structural limitation, not something more 
overlap alone fully solves.

**Fix:** implemented sentence-aware chunking (never splits a sentence across 
chunk boundaries) alongside the original fixed-size approach, selectable per 
document from the UI.

**Result**, tested on the identical question against the same document:

| Chunking strategy | Answer |
|---|---|
| Fixed-size | "I don't have enough information in the provided context to answer this." |
| Sentence-aware | Correct, complete answer with the specific fact cited |

Confirmed that hybrid search and reranking alone could not fix this — both 
were already active on the failing run. Only a structural change to chunking 
resolved it.

## A production bug: slow PDF extraction on complex documents

While testing the deployed app with a large (41MB), image/graphics-heavy PDF,
ingestion appeared to hang with no visible progress.

**Diagnosis:** added timing instrumentation around the PDF text extraction step
specifically (separate from chunking/embedding). Found that `pypdf` took over
16 seconds to extract just 12,845 characters from a single complex PDF — an
extreme mismatch between processing time and actual text content, caused by
`pypdf`'s parser working through the file's embedded graphics/font structures
even though only the text was needed.

**Fix:** switched PDF extraction to PyMuPDF (`fitz`), a faster C-based PDF
parser. Also added real-time, granular ingestion progress (showing live chunk
counts during embedding) and transparent extracted-character-count messaging,
since file size in MB is a poor predictor of processing time — a small,
graphics-heavy PDF can take longer to parse than a much larger, text-only one.

**Result**, same file, before and after:

| Library | Extraction time | Characters extracted |
|---|---|---|
| pypdf | 16.3 seconds | 12,845 |
| PyMuPDF | 0.34 seconds | 14,198 |

A ~48x speedup, while also extracting more complete text — PyMuPDF's parser
handled the file's malformed internal structure (a dictionary key redefinition
error) more gracefully than pypdf did.

## Chunking strategies available

- **Fixed-size** — 512 tokens, 50-token overlap (baseline)
- **Sentence-aware** — groups whole sentences up to a token budget, never splits mid-sentence
- **Paragraph-based** — splits on paragraph breaks, falling back to sentence-aware for oversized paragraphs
- **Recursive** — tries paragraph → sentence → fixed-size, in priority order

## Diagnosing retrieval vs. generation failures

One failure case initially failed under one LLM provider but succeeded under 
another using the identical retrieved chunks — confirming the failure was 
generation-side model caution, not a retrieval problem. Useful diagnostic 
technique: hold retrieved context constant and swap the LLM to isolate which 
pipeline stage is actually failing.

## Hybrid search: fixing a real retrieval miss

Naive vector-only search failed on a specific question, retrieving a noisy 
data fragment instead of the relevant prose — and correctly refused to 
hallucinate an answer rather than guess. Adding hybrid search (dense + PostgreSQL 
full-text search, merged via Reciprocal Rank Fusion) surfaced entirely new, correct 
chunks that pure vector search had missed, fixing the answer without any change 
to generation.

## Getting started

This app requires a Postgres database (with pgvector) and a Groq API key. 
Both are entered directly in the app UI — no `.env` file needed to run it.

### 1. Set up NeonDB (free Postgres + pgvector)
1. Go to https://neon.tech and sign up (no credit card required)
2. Create a project, open the SQL Editor, and run:
```sql
   CREATE EXTENSION IF NOT EXISTS vector;
```
3. Go to Connection Details, select "Pooled connection," and copy the full connection string

### 2. Get a Groq API key (free)
1. Go to https://console.groq.com and sign up
2. Go to API Keys → Create API Key, copy it immediately

### 3. Run the app
```bash
pip install -r requirements.txt
streamlit run app/ui/streamlit_app.py
```
Note: large documents (500,000+ extracted characters) may take several
minutes to embed on free-tier hosting — the app shows live progress during
ingestion.
Paste your Neon connection string and Groq API key on the first screen, click 
Connect. Upload a document, choose a chunking strategy, click Ingest, then 
ask questions.

## Known limitations / deliberately not built

- **Query rewriting** — considered, not implemented. Would add another LLM 
  round-trip before every retrieval; testing showed marginal benefit on 
  well-formed questions relative to the added latency.
- **Separate tables per chunking strategy** — considered, rejected in favor of 
  a single table with a `chunking_strategy` column, filtered jointly with 
  `source_file` at query time.

## Project structure
app/
├── ingestion/ # load, chunk, embed, store documents
├── retrieval/ # vector search, keyword search, hybrid search, reranking
├── generation/ # prompt building, LLM calls, pipeline orchestration
└── ui/ # Streamlit app
