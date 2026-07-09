import asyncio
import uuid
from datetime import datetime
from typing import Literal, cast

from mnemos.config import Settings, get_settings
from mnemos.embeddings.base import EmbeddingClient
from mnemos.memory.retrieval import MemoryRetriever
from mnemos.memory.schemas import (
    EpisodeCreate,
    EpisodeRead,
    MemoryQueryResult,
    SemanticFactCreate,
    SemanticFactRead,
)
from mnemos.storage.base import StorageBackend


class MemoryEngine:
    """The single entry point the agent and API talk to. Callers never touch
    a StorageBackend or MemoryRetriever directly — that's what lets new
    memory types or a different storage backend get added later by extending
    this facade instead of rewiring every caller.
    """

    def __init__(
        self,
        storage: StorageBackend,
        embedding_client: EmbeddingClient,
        settings: Settings | None = None,
    ):
        self.storage = storage
        self.embeddings = embedding_client
        self.settings = settings or get_settings()
        self.retriever = MemoryRetriever(storage, self.settings)

    async def remember_episode(
        self,
        user_id: str,
        session_id: uuid.UUID,
        role: str,
        content: str,
        *,
        occurred_at: datetime | None = None,
        metadata: dict | None = None,
    ) -> EpisodeRead:
        embedding = self.embeddings.embed_one(content)
        return await self.storage.write_episode(
            EpisodeCreate(
                user_id=user_id,
                session_id=session_id,
                # role is widened to str here (callers include seed data from
                # JSON/API input); EpisodeCreate validates it at runtime.
                role=cast(Literal["user", "assistant"], role),
                content=content,
                occurred_at=occurred_at,
                metadata=metadata or {},
            ),
            embedding=embedding,
        )

    async def remember_fact(
        self,
        user_id: str,
        fact: str,
        *,
        source_episode_ids: list[str] | None = None,
        confidence: float = 1.0,
    ) -> SemanticFactRead:
        embedding = self.embeddings.embed_one(fact)
        return await self.storage.write_fact(
            SemanticFactCreate(
                user_id=user_id,
                fact=fact,
                source_episode_ids=source_episode_ids or [],
                confidence=confidence,
            ),
            embedding=embedding,
        )

    async def recall(
        self,
        user_id: str,
        query: str,
        *,
        now: datetime | None = None,
        similarity_weight: float | None = None,
        recency_weight: float | None = None,
    ) -> MemoryQueryResult:
        query_embedding = self.embeddings.embed_one(query)
        result = await self.retriever.retrieve(
            user_id,
            query_embedding,
            now=now,
            similarity_weight=similarity_weight,
            recency_weight=recency_weight,
        )
        if result.facts:
            # "Facts you keep needing survive; facts nobody asks about fade" —
            # reinforcement is the inverse of reflection's decay pass. Uses
            # real wall-clock time regardless of a simulated `now` above,
            # since this tracks when the fact was actually used, not the
            # (possibly simulated) point in time being scored against.
            await asyncio.gather(
                *(self.storage.reinforce_fact(f.fact.id) for f in result.facts)
            )
        return result

    async def list_episodes(self, user_id: str, *, limit: int = 100) -> list[EpisodeRead]:
        return await self.storage.get_episodes(user_id, limit=limit)

    async def list_facts(
        self, user_id: str, *, status: str = "active", limit: int = 200
    ) -> list[SemanticFactRead]:
        return await self.storage.get_facts(user_id, status=status, limit=limit)

    async def reset_user(self, user_id: str) -> None:
        await self.storage.delete_episodes_for_user(user_id)
        await self.storage.delete_facts_for_user(user_id)
