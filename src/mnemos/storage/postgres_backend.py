import uuid
from datetime import UTC, datetime

from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from mnemos.db.models import EpisodicMemory, SemanticMemory, User
from mnemos.db.session import get_sessionmaker
from mnemos.memory.schemas import EpisodeCreate, EpisodeRead, SemanticFactCreate, SemanticFactRead
from mnemos.storage.base import EpisodeMatch, FactMatch, StorageBackend


class PostgresBackend(StorageBackend):
    """The default backend: Postgres + pgvector, HNSW cosine similarity
    search. Each method opens and commits its own session — no session or
    transaction boundary leaks to callers.
    """

    def __init__(self, sessionmaker: async_sessionmaker[AsyncSession] | None = None):
        self._sessionmaker = sessionmaker or get_sessionmaker()

    async def _ensure_user(self, session: AsyncSession, user_id: str) -> None:
        if await session.get(User, user_id) is None:
            session.add(User(id=user_id))
            await session.flush()

    async def write_episode(self, episode: EpisodeCreate, embedding: list[float]) -> EpisodeRead:
        async with self._sessionmaker() as session:
            await self._ensure_user(session, episode.user_id)
            kwargs = {"occurred_at": episode.occurred_at} if episode.occurred_at else {}
            row = EpisodicMemory(
                user_id=episode.user_id,
                session_id=episode.session_id,
                role=episode.role,
                content=episode.content,
                embedding=embedding,
                memory_metadata=episode.metadata,
                **kwargs,
            )
            session.add(row)
            await session.commit()
            await session.refresh(row)
            return EpisodeRead.model_validate(row)

    async def get_episodes(
        self, user_id: str, *, session_id: uuid.UUID | None = None, limit: int = 100
    ) -> list[EpisodeRead]:
        async with self._sessionmaker() as session:
            stmt = (
                select(EpisodicMemory)
                .where(EpisodicMemory.user_id == user_id)
                .order_by(EpisodicMemory.occurred_at.desc())
                .limit(limit)
            )
            if session_id is not None:
                stmt = stmt.where(EpisodicMemory.session_id == session_id)
            result = await session.execute(stmt)
            return [EpisodeRead.model_validate(row) for row in result.scalars().all()]

    async def search_episodes(
        self, user_id: str, query_embedding: list[float], top_k: int
    ) -> list[EpisodeMatch]:
        async with self._sessionmaker() as session:
            distance = EpisodicMemory.embedding.cosine_distance(query_embedding)
            stmt = (
                select(EpisodicMemory, distance.label("distance"))
                .where(EpisodicMemory.user_id == user_id)
                .order_by(distance)
                .limit(top_k)
            )
            result = await session.execute(stmt)
            return [
                (EpisodeRead.model_validate(row), float(dist)) for row, dist in result.all()
            ]

    async def delete_episodes_for_user(self, user_id: str) -> None:
        async with self._sessionmaker() as session:
            await session.execute(delete(EpisodicMemory).where(EpisodicMemory.user_id == user_id))
            await session.commit()

    async def write_fact(
        self, fact: SemanticFactCreate, embedding: list[float]
    ) -> SemanticFactRead:
        async with self._sessionmaker() as session:
            await self._ensure_user(session, fact.user_id)
            row = SemanticMemory(
                user_id=fact.user_id,
                fact=fact.fact,
                embedding=embedding,
                source_episode_ids=fact.source_episode_ids,
                confidence=fact.confidence,
                memory_metadata=fact.metadata,
            )
            session.add(row)
            await session.commit()
            await session.refresh(row)
            return SemanticFactRead.model_validate(row)

    async def get_facts(
        self, user_id: str, *, status: str = "active", limit: int = 200
    ) -> list[SemanticFactRead]:
        async with self._sessionmaker() as session:
            stmt = (
                select(SemanticMemory)
                .where(SemanticMemory.user_id == user_id, SemanticMemory.status == status)
                .order_by(SemanticMemory.created_at.desc())
                .limit(limit)
            )
            result = await session.execute(stmt)
            return [SemanticFactRead.model_validate(row) for row in result.scalars().all()]

    async def search_facts(
        self, user_id: str, query_embedding: list[float], top_k: int, *, status: str = "active"
    ) -> list[FactMatch]:
        async with self._sessionmaker() as session:
            distance = SemanticMemory.embedding.cosine_distance(query_embedding)
            stmt = (
                select(SemanticMemory, distance.label("distance"))
                .where(SemanticMemory.user_id == user_id, SemanticMemory.status == status)
                .order_by(distance)
                .limit(top_k)
            )
            result = await session.execute(stmt)
            return [
                (SemanticFactRead.model_validate(row), float(dist)) for row, dist in result.all()
            ]

    async def update_fact_status(
        self, fact_id: uuid.UUID, status: str, *, confidence: float | None = None
    ) -> None:
        async with self._sessionmaker() as session:
            values: dict = {"status": status}
            if confidence is not None:
                values["confidence"] = confidence
            await session.execute(
                update(SemanticMemory).where(SemanticMemory.id == fact_id).values(**values)
            )
            await session.commit()

    async def reinforce_fact(self, fact_id: uuid.UUID, *, at: datetime | None = None) -> None:
        at = at or datetime.now(UTC)
        async with self._sessionmaker() as session:
            row = await session.get(SemanticMemory, fact_id)
            if row is None:
                return
            row.last_reinforced_at = at
            row.confidence = min(row.confidence * 1.05, 1.0)
            await session.commit()

    async def delete_facts_for_user(self, user_id: str) -> None:
        async with self._sessionmaker() as session:
            await session.execute(delete(SemanticMemory).where(SemanticMemory.user_id == user_id))
            await session.commit()
