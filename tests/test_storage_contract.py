"""Backend-contract tests: the same behavioral suite runs against every
StorageBackend implementation, so "swap the backend" is proven, not just
asserted. Parametrized over Postgres, Qdrant, and Neo4j (Neo4j tests skip
gracefully if no local instance is reachable).

Uses MockEmbeddingClient (deterministic, hash-seeded) rather than real
embeddings — this suite is about storage *semantics* (write/get/search/
status/reinforce), not semantic search quality. Real-embedding ranking
correctness is covered separately in tests/unit/test_retrieval.py.
"""

import uuid
from datetime import UTC, datetime, timedelta

import pytest

from mnemos.embeddings.mock_client import MockEmbeddingClient
from mnemos.memory.schemas import EpisodeCreate, SemanticFactCreate

pytestmark = pytest.mark.asyncio

BACKEND_FIXTURES = ["postgres_backend", "qdrant_backend", "neo4j_backend"]

_embedder = MockEmbeddingClient()


@pytest.fixture(params=BACKEND_FIXTURES)
def backend(request):
    return request.getfixturevalue(request.param)


async def test_write_and_get_episode(backend, new_user_id: str):
    session_id = uuid.uuid4()
    written = await backend.write_episode(
        EpisodeCreate(
            user_id=new_user_id, session_id=session_id, role="user", content="I use FastAPI."
        ),
        embedding=_embedder.embed_one("I use FastAPI."),
    )
    assert written.id is not None
    assert written.occurred_at is not None

    episodes = await backend.get_episodes(new_user_id)
    assert len(episodes) == 1
    assert episodes[0].content == "I use FastAPI."


async def test_get_episodes_filters_by_session(backend, new_user_id: str):
    session_a, session_b = uuid.uuid4(), uuid.uuid4()
    for sid, content in [(session_a, "in session a"), (session_b, "in session b")]:
        await backend.write_episode(
            EpisodeCreate(user_id=new_user_id, session_id=sid, role="user", content=content),
            embedding=_embedder.embed_one(content),
        )

    episodes = await backend.get_episodes(new_user_id, session_id=session_a)
    assert len(episodes) == 1
    assert episodes[0].content == "in session a"


async def test_occurred_at_can_be_set_for_simulated_time(backend, new_user_id: str):
    thirty_days_ago = datetime.now(UTC) - timedelta(days=30)
    written = await backend.write_episode(
        EpisodeCreate(
            user_id=new_user_id,
            session_id=uuid.uuid4(),
            role="user",
            content="from the past",
            occurred_at=thirty_days_ago,
        ),
        embedding=_embedder.embed_one("from the past"),
    )
    assert written.occurred_at == thirty_days_ago


async def test_search_episodes_ranks_exact_text_match_first(backend, new_user_id: str):
    session_id = uuid.uuid4()
    target, other = "prefers FastAPI over Flask", "the weather is nice today"
    for content in [target, other]:
        await backend.write_episode(
            EpisodeCreate(user_id=new_user_id, session_id=session_id, role="user", content=content),
            embedding=_embedder.embed_one(content),
        )

    matches = await backend.search_episodes(new_user_id, _embedder.embed_one(target), top_k=2)
    assert len(matches) == 2
    top_episode, top_distance = matches[0]
    assert top_episode.content == target
    assert top_distance < 1e-6  # identical text -> identical mock embedding -> distance ~0


async def test_delete_episodes_for_user(backend, new_user_id: str):
    await backend.write_episode(
        EpisodeCreate(user_id=new_user_id, session_id=uuid.uuid4(), role="user", content="x"),
        embedding=_embedder.embed_one("x"),
    )
    await backend.delete_episodes_for_user(new_user_id)
    assert await backend.get_episodes(new_user_id) == []


async def test_write_and_get_fact(backend, new_user_id: str):
    written = await backend.write_fact(
        SemanticFactCreate(user_id=new_user_id, fact="prefers FastAPI over Flask"),
        embedding=_embedder.embed_one("prefers FastAPI over Flask"),
    )
    assert written.status == "active"
    assert written.confidence == 1.0

    facts = await backend.get_facts(new_user_id)
    assert len(facts) == 1
    assert facts[0].fact == "prefers FastAPI over Flask"


async def test_get_facts_filters_by_status(backend, new_user_id: str):
    await backend.write_fact(
        SemanticFactCreate(user_id=new_user_id, fact="active fact"),
        embedding=_embedder.embed_one("active fact"),
    )
    assert len(await backend.get_facts(new_user_id, status="active")) == 1
    assert len(await backend.get_facts(new_user_id, status="forgotten")) == 0


async def test_search_facts_ranks_exact_text_match_first(backend, new_user_id: str):
    target, other = "prefers FastAPI over Flask", "has a dog named Miso"
    for fact in [target, other]:
        await backend.write_fact(
            SemanticFactCreate(user_id=new_user_id, fact=fact), embedding=_embedder.embed_one(fact)
        )

    matches = await backend.search_facts(new_user_id, _embedder.embed_one(target), top_k=2)
    assert matches[0][0].fact == target
    assert matches[0][1] < 1e-6


async def test_update_fact_status_and_confidence(backend, new_user_id: str):
    fact = await backend.write_fact(
        SemanticFactCreate(user_id=new_user_id, fact="will be merged"),
        embedding=_embedder.embed_one("will be merged"),
    )
    await backend.update_fact_status(fact.id, "merged", confidence=0.5)

    active = await backend.get_facts(new_user_id, status="active")
    merged = await backend.get_facts(new_user_id, status="merged")
    assert active == []
    assert len(merged) == 1
    assert merged[0].confidence == 0.5


async def test_reinforce_fact_bumps_confidence_and_timestamp(backend, new_user_id: str):
    fact = await backend.write_fact(
        SemanticFactCreate(user_id=new_user_id, fact="gets reinforced"),
        embedding=_embedder.embed_one("gets reinforced"),
    )
    original_confidence = fact.confidence
    later = datetime.now(UTC) + timedelta(days=1)

    await backend.reinforce_fact(fact.id, at=later)

    [refreshed] = await backend.get_facts(new_user_id)
    assert refreshed.confidence >= original_confidence
    assert refreshed.last_reinforced_at == later


async def test_delete_facts_for_user(backend, new_user_id: str):
    await backend.write_fact(
        SemanticFactCreate(user_id=new_user_id, fact="to be deleted"),
        embedding=_embedder.embed_one("to be deleted"),
    )
    await backend.delete_facts_for_user(new_user_id)
    assert await backend.get_facts(new_user_id) == []
