"""Proves the public `from mnemos import Memory` entry point actually works
end to end, with the zero-setup path (embedded Qdrant + mock embeddings, no
Docker/Postgres/API key needed) — this is the first thing a new user runs,
so it's tested independently of the storage-backend contract suite.
"""

import uuid

import pytest

from mnemos import Memory
from mnemos.config import Settings


@pytest.mark.asyncio
async def test_memory_factory_remember_and_recall(tmp_path):
    settings = Settings(
        storage_backend="qdrant",
        qdrant_local_path=str(tmp_path / "qdrant"),
        embedding_provider="mock",
    )
    memory = Memory(settings)
    user_id = f"test-user-{uuid.uuid4().hex[:8]}"
    session_id = uuid.uuid4()

    try:
        await memory.remember_episode(
            user_id=user_id,
            session_id=session_id,
            role="user",
            content="I have a dog named Piper",
        )

        result = await memory.recall(user_id=user_id, query="what's my dog's name")

        assert any("Piper" in e.episode.content for e in result.episodes)
    finally:
        await memory.aclose()


@pytest.mark.asyncio
async def test_memory_factory_context_manager(tmp_path):
    settings = Settings(
        storage_backend="qdrant",
        qdrant_local_path=str(tmp_path / "qdrant"),
        embedding_provider="mock",
    )

    async with Memory(settings) as memory:
        fact = await memory.remember_fact(user_id="ctx-user", fact="prefers dark mode")
        assert fact.fact == "prefers dark mode"
