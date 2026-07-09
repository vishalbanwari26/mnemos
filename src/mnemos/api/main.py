from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from mnemos.api.routers import admin, chat, insights, memories, procedural, reflection
from mnemos.config import get_settings
from mnemos.embeddings.factory import get_embedding_client
from mnemos.llm.factory import get_llm_client
from mnemos.storage.factory import get_storage_backend, reset_storage_backend_cache


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    app.state.settings = settings
    # Load the embedding model, construct LLM clients, and open the storage
    # backend once at startup — not per-request. Postgres builds a cheap
    # per-call session wrapper regardless; Qdrant/Neo4j are long-lived
    # connections that must not be reopened per request.
    app.state.embedding_client = get_embedding_client(settings)
    app.state.llm_client = get_llm_client(settings)
    app.state.extraction_llm_client = get_llm_client(settings, for_extraction=True)
    app.state.storage_backend = get_storage_backend(settings)
    yield
    await reset_storage_backend_cache()


def create_app() -> FastAPI:
    app = FastAPI(title="mnemos", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(chat.router)
    app.include_router(memories.router)
    app.include_router(admin.router)
    app.include_router(procedural.router)
    app.include_router(reflection.router)
    app.include_router(insights.router)
    return app


app = create_app()
