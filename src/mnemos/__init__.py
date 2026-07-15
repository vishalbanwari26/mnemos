"""mnemos: a persistent memory framework for LLM agents.

    from mnemos import Memory

    memory = Memory()  # reads Settings from env/.env — defaults to local
                        # Postgres+pgvector; set STORAGE_BACKEND=qdrant for a
                        # zero-setup embedded store, no server required
    await memory.remember_episode(user_id="u1", session_id=session_id,
                                   role="user", content="I have a dog named Piper")
    result = await memory.recall(user_id="u1", query="what's my dog's name")
    await memory.aclose()

`Memory()` is a thin factory: it builds the configured storage backend and
embedding client from `Settings` and wires them into `MemoryEngine`, so
callers never construct those directly. `MemoryEngine`'s own methods
(`remember_episode`, `remember_fact`, `recall`, `list_episodes`,
`list_facts`, `reset_user`) are the actual public surface — this module just
removes the wiring step.
"""

from mnemos.config import Settings, get_settings
from mnemos.embeddings.factory import get_embedding_client
from mnemos.memory.engine import MemoryEngine
from mnemos.storage.factory import get_storage_backend

__all__ = ["Memory", "MemoryEngine", "Settings"]


def Memory(settings: Settings | None = None) -> MemoryEngine:
    """Build a ready-to-use MemoryEngine from Settings (env/.env by default).

    Pass an explicit `Settings` to override the backend/embedding provider
    for a single instance without touching the environment, e.g. for a
    quick local trial:

        Memory(Settings(storage_backend="qdrant", embedding_provider="local"))
    """
    settings = settings or get_settings()
    storage = get_storage_backend(settings)
    embeddings = get_embedding_client(settings)
    return MemoryEngine(storage, embeddings, settings)
