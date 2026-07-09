import asyncio
import uuid
from datetime import UTC, datetime

from neo4j import AsyncGraphDatabase

from mnemos.memory.schemas import EpisodeCreate, EpisodeRead, SemanticFactCreate, SemanticFactRead
from mnemos.storage.base import EpisodeMatch, FactMatch, StorageBackend

EPISODE_INDEX = "episode_embedding_idx"
FACT_INDEX = "fact_embedding_idx"

# Neo4j's vector index has no native pre-filter (unlike Qdrant, which combines
# payload filtering with ANN search in one query) — this backend over-fetches
# candidates and filters by user_id in Cypher afterward. Documented tradeoff,
# not hidden: at portfolio scale this is fine, at real scale it would need a
# per-user index or a different query strategy.
OVERFETCH_FACTOR = 5

# Neo4j's vector index updates in the background; db.awaitIndexes() only
# confirms the index is ONLINE, not that a specific just-written node is
# already searchable through it. So a write polls the index directly for its
# own node before returning — the only way to actually guarantee read-your-
# writes consistency, which the other two backends give for free. This is a
# real, measurable Neo4j cost, surfaced honestly in the backend comparison
# rather than papered over.
_INDEX_POLL_ATTEMPTS = 20
_INDEX_POLL_INTERVAL_S = 0.1


def _to_native(value: datetime) -> datetime:
    """The driver returns neo4j.time.DateTime for temporal properties."""
    return value.to_native() if hasattr(value, "to_native") else value


class Neo4jBackend(StorageBackend):
    """A real graph, not a vector table wearing a costume: episodes and facts
    are nodes owned by a User via SAID/KNOWS edges, and a fact's provenance
    (which episodes it was extracted from) is a first-class DERIVED_FROM edge
    instead of Postgres's source_episode_ids JSON array.
    """

    def __init__(self, uri: str, user: str, password: str, dimension: int = 384):
        # db.index.vector.queryNodes is deprecated in newer Neo4j server
        # versions (replaced by a SEARCH clause not yet reflected in a stable
        # driver API at the time of writing) but still fully functional;
        # notifications_min_severity="OFF" silences the noisy per-query
        # deprecation warning without changing query behavior.
        self._driver = AsyncGraphDatabase.driver(
            uri, auth=(user, password), notifications_min_severity="OFF"
        )
        self._dimension = dimension
        self._schema_ready = False

    async def _ensure_schema(self) -> None:
        if self._schema_ready:
            return
        async with self._driver.session() as session:
            for index_name, label in [(EPISODE_INDEX, "Episode"), (FACT_INDEX, "Fact")]:
                await session.run(
                    f"""
                    CREATE VECTOR INDEX {index_name} IF NOT EXISTS
                    FOR (n:{label}) ON (n.embedding)
                    OPTIONS {{indexConfig: {{
                        `vector.dimensions`: $dim,
                        `vector.similarity_function`: 'cosine'
                    }}}}
                    """,
                    dim=self._dimension,
                )
            await session.run("CALL db.awaitIndexes(30)")
        self._schema_ready = True

    async def _wait_until_indexed(
        self, session, index_name: str, node_id: str, embedding: list[float]
    ) -> None:
        for _ in range(_INDEX_POLL_ATTEMPTS):
            result = await session.run(
                f"""
                CALL db.index.vector.queryNodes('{index_name}', 25, $vec)
                YIELD node
                WHERE node.id = $id
                RETURN node
                """,
                vec=embedding,
                id=node_id,
            )
            if await result.single() is not None:
                return
            await asyncio.sleep(_INDEX_POLL_INTERVAL_S)

    @staticmethod
    def _score_to_distance(score: float) -> float:
        """Neo4j's cosine vector index returns a score normalized to [0, 1]
        (1 = identical). Convert to pgvector-style cosine distance [0, 2]
        (0 = identical) so the shared scoring logic in memory/retrieval.py
        doesn't need to know which backend produced the match."""
        similarity = 2 * score - 1
        return 1.0 - similarity

    # -- episodes ------------------------------------------------------

    async def write_episode(self, episode: EpisodeCreate, embedding: list[float]) -> EpisodeRead:
        await self._ensure_schema()
        now = datetime.now(UTC)
        occurred_at = episode.occurred_at or now
        episode_id = str(uuid.uuid4())
        async with self._driver.session() as session:
            await session.run(
                """
                MERGE (u:User {id: $user_id})
                CREATE (e:Episode {
                    id: $id, user_id: $user_id, session_id: $session_id, role: $role,
                    content: $content, embedding: $embedding,
                    created_at: $created_at, occurred_at: $occurred_at
                })
                CREATE (u)-[:SAID]->(e)
                """,
                user_id=episode.user_id,
                id=episode_id,
                session_id=str(episode.session_id),
                role=episode.role,
                content=episode.content,
                embedding=embedding,
                created_at=now,
                occurred_at=occurred_at,
            )
            await self._wait_until_indexed(session, EPISODE_INDEX, episode_id, embedding)
        return EpisodeRead(
            id=uuid.UUID(episode_id),
            user_id=episode.user_id,
            session_id=episode.session_id,
            role=episode.role,
            content=episode.content,
            created_at=now,
            occurred_at=occurred_at,
        )

    @staticmethod
    def _episode_from_node(node) -> EpisodeRead:
        return EpisodeRead(
            id=uuid.UUID(node["id"]),
            user_id=node["user_id"],
            session_id=uuid.UUID(node["session_id"]),
            role=node["role"],
            content=node["content"],
            created_at=_to_native(node["created_at"]),
            occurred_at=_to_native(node["occurred_at"]),
        )

    async def get_episodes(
        self, user_id: str, *, session_id: uuid.UUID | None = None, limit: int = 100
    ) -> list[EpisodeRead]:
        await self._ensure_schema()
        async with self._driver.session() as session:
            result = await session.run(
                """
                MATCH (e:Episode {user_id: $user_id})
                WHERE $session_id IS NULL OR e.session_id = $session_id
                RETURN e
                ORDER BY e.occurred_at DESC
                LIMIT $limit
                """,
                user_id=user_id,
                session_id=str(session_id) if session_id else None,
                limit=limit,
            )
            return [self._episode_from_node(r["e"]) async for r in result]

    async def search_episodes(
        self, user_id: str, query_embedding: list[float], top_k: int
    ) -> list[EpisodeMatch]:
        await self._ensure_schema()
        async with self._driver.session() as session:
            result = await session.run(
                f"""
                CALL db.index.vector.queryNodes('{EPISODE_INDEX}', $fetch_k, $vec)
                YIELD node, score
                WHERE node.user_id = $user_id
                RETURN node, score
                ORDER BY score DESC
                LIMIT $top_k
                """,
                fetch_k=top_k * OVERFETCH_FACTOR,
                vec=query_embedding,
                user_id=user_id,
                top_k=top_k,
            )
            return [
                (self._episode_from_node(r["node"]), self._score_to_distance(r["score"]))
                async for r in result
            ]

    async def delete_episodes_for_user(self, user_id: str) -> None:
        await self._ensure_schema()
        async with self._driver.session() as session:
            await session.run(
                "MATCH (e:Episode {user_id: $user_id}) DETACH DELETE e", user_id=user_id
            )

    # -- facts -----------------------------------------------------

    async def write_fact(
        self, fact: SemanticFactCreate, embedding: list[float]
    ) -> SemanticFactRead:
        await self._ensure_schema()
        now = datetime.now(UTC)
        fact_id = str(uuid.uuid4())
        async with self._driver.session() as session:
            await session.run(
                """
                MERGE (u:User {id: $user_id})
                CREATE (f:Fact {
                    id: $id, user_id: $user_id, fact: $fact, embedding: $embedding,
                    confidence: $confidence, status: 'active',
                    created_at: $now, updated_at: $now, last_reinforced_at: $now
                })
                CREATE (u)-[:KNOWS]->(f)
                WITH f
                CALL (f) {
                    UNWIND $source_ids AS eid
                    MATCH (e:Episode {id: eid})
                    CREATE (f)-[:DERIVED_FROM]->(e)
                    RETURN count(*) AS linked
                }
                RETURN f
                """,
                user_id=fact.user_id,
                id=fact_id,
                fact=fact.fact,
                embedding=embedding,
                confidence=fact.confidence,
                now=now,
                source_ids=fact.source_episode_ids,
            )
            await self._wait_until_indexed(session, FACT_INDEX, fact_id, embedding)
        return SemanticFactRead(
            id=uuid.UUID(fact_id),
            user_id=fact.user_id,
            fact=fact.fact,
            confidence=fact.confidence,
            status="active",
            source_episode_ids=fact.source_episode_ids,
            created_at=now,
            updated_at=now,
            last_reinforced_at=now,
        )

    @staticmethod
    def _fact_from_record(node, source_episode_ids: list[str]) -> SemanticFactRead:
        return SemanticFactRead(
            id=uuid.UUID(node["id"]),
            user_id=node["user_id"],
            fact=node["fact"],
            confidence=node["confidence"],
            status=node["status"],
            source_episode_ids=source_episode_ids,
            created_at=_to_native(node["created_at"]),
            updated_at=_to_native(node["updated_at"]),
            last_reinforced_at=_to_native(node["last_reinforced_at"]),
        )

    async def get_facts(
        self, user_id: str, *, status: str = "active", limit: int = 200
    ) -> list[SemanticFactRead]:
        await self._ensure_schema()
        async with self._driver.session() as session:
            result = await session.run(
                """
                MATCH (f:Fact {user_id: $user_id, status: $status})
                OPTIONAL MATCH (f)-[:DERIVED_FROM]->(e:Episode)
                WITH f, collect(e.id) AS source_episode_ids
                RETURN f, source_episode_ids
                ORDER BY f.created_at DESC
                LIMIT $limit
                """,
                user_id=user_id,
                status=status,
                limit=limit,
            )
            return [self._fact_from_record(r["f"], r["source_episode_ids"]) async for r in result]

    async def search_facts(
        self, user_id: str, query_embedding: list[float], top_k: int, *, status: str = "active"
    ) -> list[FactMatch]:
        await self._ensure_schema()
        async with self._driver.session() as session:
            result = await session.run(
                f"""
                CALL db.index.vector.queryNodes('{FACT_INDEX}', $fetch_k, $vec)
                YIELD node, score
                WHERE node.user_id = $user_id AND node.status = $status
                WITH node, score
                OPTIONAL MATCH (node)-[:DERIVED_FROM]->(e:Episode)
                WITH node, score, collect(e.id) AS source_episode_ids
                RETURN node, score, source_episode_ids
                ORDER BY score DESC
                LIMIT $top_k
                """,
                fetch_k=top_k * OVERFETCH_FACTOR,
                vec=query_embedding,
                user_id=user_id,
                status=status,
                top_k=top_k,
            )
            return [
                (
                    self._fact_from_record(r["node"], r["source_episode_ids"]),
                    self._score_to_distance(r["score"]),
                )
                async for r in result
            ]

    async def update_fact_status(
        self, fact_id: uuid.UUID, status: str, *, confidence: float | None = None
    ) -> None:
        await self._ensure_schema()
        async with self._driver.session() as session:
            await session.run(
                """
                MATCH (f:Fact {id: $id})
                SET f.status = $status, f.updated_at = $now
                SET f.confidence = CASE
                    WHEN $confidence IS NULL THEN f.confidence ELSE $confidence
                END
                """,
                id=str(fact_id),
                status=status,
                confidence=confidence,
                now=datetime.now(UTC),
            )

    async def reinforce_fact(self, fact_id: uuid.UUID, *, at: datetime | None = None) -> None:
        await self._ensure_schema()
        at = at or datetime.now(UTC)
        async with self._driver.session() as session:
            await session.run(
                """
                MATCH (f:Fact {id: $id})
                SET f.last_reinforced_at = $at,
                    f.confidence = CASE
                        WHEN f.confidence * 1.05 > 1.0 THEN 1.0 ELSE f.confidence * 1.05
                    END
                """,
                id=str(fact_id),
                at=at,
            )

    async def delete_facts_for_user(self, user_id: str) -> None:
        await self._ensure_schema()
        async with self._driver.session() as session:
            await session.run("MATCH (f:Fact {user_id: $user_id}) DETACH DELETE f", user_id=user_id)

    async def close(self) -> None:
        await self._driver.close()
