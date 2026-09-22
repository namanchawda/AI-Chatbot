"""FastAPI application entrypoint for the SEC filing RAG service.

Creates all tables on startup via the lifespan hook and mounts the /api router.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router as api_router
from app.ingestion import store


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Ensure all tables (document_chunks + chat sessions) exist on startup."""
    store._ensure_initialized()
    store.create_table()
    yield


app = FastAPI(
    title="AI-Chatbot",
    description="Multi-document RAG Chatbot.",
    version="0.2.0",
    lifespan=lifespan,
)

app.include_router(api_router)


@app.get("/health")
def health_check() -> dict[str, str]:
    """Return a simple health check response for the service."""
    return {"status": "ok"}
