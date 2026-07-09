import asyncio
import uuid
from datetime import UTC, datetime
from functools import partial

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PointStruct,
    VectorParams,
)

from mnemos.memory.schemas import EpisodeCreate, EpisodeRead, SemanticFactCreate, SemanticFactRead
from mnemos.storage.base import EpisodeMatch, FactMatch, StorageBackend

EPISODES_COLLECTION = "episodes"
FACTS_COLLECTION = "facts"


def _user_filter(user_id: str, extra: list[FieldCondition] | None = None) -> Filter:
    must: list = [FieldCondition(key="user_id", match=MatchValue(value=user_id))]
    must.extend(extra or [])
    return Filter(must=must)


class QdrantBackend(StorageBackend):
    """Embedded (local-mode) Qdrant — no server process, single-directory-
    locked client created once and reused for the whole app/CLI/test-session
    lifetime. Qdrant natively combines payload filtering with ANN search in
    one query (unlike Neo4j's vector index, which post-filters) — the one
    concrete advantage this backend has over the graph backend, worth calling
    out in the storage comparison writeup.

    The sync QdrantClient is wrapped with asyncio.to_thread throughout to fit
    the async StorageBackend interface.
    """

    def __init__(self, path: str, dimension: int = 384):
        self._client = QdrantClient(path=path)
        self._dimension = dimension
        self._ensure_collections()

    def _ensure_collections(self) -> None:
        existing = {c.name for c in self._client.get_collections().collections}
        for name in (EPISODES_COLLECTION, FACTS_COLLECTION):
            if name not in existing:
                self._client.create_collection(
                    name,
                    vectors_config=VectorParams(size=self._dimension, distance=Distance.COSINE),
                )

    async def _run(self, fn, /, *args, **kwargs):
        return await asyncio.to_thread(partial(fn, *args, **kwargs))

    # -- episodes ----------------------------------------------------------

    async def write_episode(self, episode: EpisodeCreate, embedding: list[float]) -> EpisodeRead:
        now = datetime.now(UTC)
        point_id = str(uuid.uuid4())
        occurred_at = episode.occurred_at or now
        payload = {
            "user_id": episode.user_id,
            "session_id": str(episode.session_id),
            "role": episode.role,
            "content": episode.content,
            "memory_metadata": episode.metadata,
            "created_at": now.isoformat(),
            "occurred_at": occurred_at.isoformat(),
        }
        await self._run(
            self._client.upsert,
            EPISODES_COLLECTION,
            points=[PointStruct(id=point_id, vector=embedding, payload=payload)],
        )
        return self._episode_from_payload(point_id, payload)

    def _episode_from_payload(self, point_id: str, payload: dict) -> EpisodeRead:
        return EpisodeRead(
            id=uuid.UUID(point_id),
            user_id=payload["user_id"],
            session_id=uuid.UUID(payload["session_id"]),
            role=payload["role"],
            content=payload["content"],
            created_at=datetime.fromisoformat(payload["created_at"]),
            occurred_at=datetime.fromisoformat(payload["occurred_at"]),
        )

    async def get_episodes(
        self, user_id: str, *, session_id: uuid.UUID | None = None, limit: int = 100
    ) -> list[EpisodeRead]:
        extra = (
            [FieldCondition(key="session_id", match=MatchValue(value=str(session_id)))]
            if session_id
            else None
        )
        points, _ = await self._run(
            self._client.scroll,
            EPISODES_COLLECTION,
            scroll_filter=_user_filter(user_id, extra),
            limit=10_000,
            with_payload=True,
            with_vectors=False,
        )
        episodes = [self._episode_from_payload(str(p.id), p.payload) for p in points]
        episodes.sort(key=lambda e: e.occurred_at, reverse=True)
        return episodes[:limit]

    async def search_episodes(
        self, user_id: str, query_embedding: list[float], top_k: int
    ) -> list[EpisodeMatch]:
        result = await self._run(
            self._client.query_points,
            EPISODES_COLLECTION,
            query=query_embedding,
            query_filter=_user_filter(user_id),
            limit=top_k,
            with_payload=True,
        )
        return [
            (self._episode_from_payload(str(p.id), p.payload), 1.0 - p.score) for p in result.points
        ]

    async def delete_episodes_for_user(self, user_id: str) -> None:
        await self._run(
            self._client.delete,
            EPISODES_COLLECTION,
            points_selector=FilterSelector(filter=_user_filter(user_id)),
        )

    # -- facts ---------------------------------------------------------

    async def write_fact(
        self, fact: SemanticFactCreate, embedding: list[float]
    ) -> SemanticFactRead:
        now = datetime.now(UTC)
        point_id = str(uuid.uuid4())
        payload = {
            "user_id": fact.user_id,
            "fact": fact.fact,
            "source_episode_ids": fact.source_episode_ids,
            "confidence": fact.confidence,
            "status": "active",
            "memory_metadata": fact.metadata,
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
            "last_reinforced_at": now.isoformat(),
        }
        await self._run(
            self._client.upsert,
            FACTS_COLLECTION,
            points=[PointStruct(id=point_id, vector=embedding, payload=payload)],
        )
        return self._fact_from_payload(point_id, payload)

    def _fact_from_payload(self, point_id: str, payload: dict) -> SemanticFactRead:
        return SemanticFactRead(
            id=uuid.UUID(point_id),
            user_id=payload["user_id"],
            fact=payload["fact"],
            confidence=payload["confidence"],
            status=payload["status"],
            source_episode_ids=payload["source_episode_ids"],
            created_at=datetime.fromisoformat(payload["created_at"]),
            updated_at=datetime.fromisoformat(payload["updated_at"]),
            last_reinforced_at=datetime.fromisoformat(payload["last_reinforced_at"]),
        )

    async def get_facts(
        self, user_id: str, *, status: str = "active", limit: int = 200
    ) -> list[SemanticFactRead]:
        extra = [FieldCondition(key="status", match=MatchValue(value=status))]
        points, _ = await self._run(
            self._client.scroll,
            FACTS_COLLECTION,
            scroll_filter=_user_filter(user_id, extra),
            limit=10_000,
            with_payload=True,
            with_vectors=False,
        )
        facts = [self._fact_from_payload(str(p.id), p.payload) for p in points]
        facts.sort(key=lambda f: f.created_at, reverse=True)
        return facts[:limit]

    async def search_facts(
        self, user_id: str, query_embedding: list[float], top_k: int, *, status: str = "active"
    ) -> list[FactMatch]:
        extra = [FieldCondition(key="status", match=MatchValue(value=status))]
        result = await self._run(
            self._client.query_points,
            FACTS_COLLECTION,
            query=query_embedding,
            query_filter=_user_filter(user_id, extra),
            limit=top_k,
            with_payload=True,
        )
        return [
            (self._fact_from_payload(str(p.id), p.payload), 1.0 - p.score) for p in result.points
        ]

    async def update_fact_status(
        self, fact_id: uuid.UUID, status: str, *, confidence: float | None = None
    ) -> None:
        payload: dict = {"status": status}
        if confidence is not None:
            payload["confidence"] = confidence
        await self._run(
            self._client.set_payload, FACTS_COLLECTION, payload=payload, points=[str(fact_id)]
        )

    async def reinforce_fact(self, fact_id: uuid.UUID, *, at: datetime | None = None) -> None:
        at = at or datetime.now(UTC)
        points = await self._run(
            self._client.retrieve, FACTS_COLLECTION, ids=[str(fact_id)], with_payload=True
        )
        if not points:
            return
        current_confidence = points[0].payload["confidence"]
        await self._run(
            self._client.set_payload,
            FACTS_COLLECTION,
            payload={
                "last_reinforced_at": at.isoformat(),
                "confidence": min(current_confidence * 1.05, 1.0),
            },
            points=[str(fact_id)],
        )

    async def delete_facts_for_user(self, user_id: str) -> None:
        await self._run(
            self._client.delete,
            FACTS_COLLECTION,
            points_selector=FilterSelector(filter=_user_filter(user_id)),
        )

    async def close(self) -> None:
        await self._run(self._client.close)
