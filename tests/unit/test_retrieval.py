"""Retrieval tests run against real local embeddings (not the mock client) —
the whole point of this module is to prove semantic similarity actually
ranks correctly, which a hash-based mock embedding can't demonstrate.
"""

import uuid
from datetime import UTC, datetime, timedelta

import pytest

from mnemos.config import Settings
from mnemos.embeddings.local_client import SentenceTransformerEmbeddingClient
from mnemos.memory.retrieval import MemoryRetriever
from mnemos.memory.schemas import EpisodeCreate
from mnemos.storage.postgres_backend import PostgresBackend

pytestmark = pytest.mark.asyncio


@pytest.fixture(scope="module")
def embedder() -> SentenceTransformerEmbeddingClient:
    return SentenceTransformerEmbeddingClient()


async def test_semantically_similar_episode_ranks_above_unrelated(
    postgres_backend: PostgresBackend,
    new_user_id: str,
    embedder: SentenceTransformerEmbeddingClient,
):
    session_id = uuid.uuid4()

    about_backend = "I've switched to FastAPI over Flask for my backend work."
    unrelated = "My favorite hiking trail is closed for the season."

    for content in [about_backend, unrelated]:
        await postgres_backend.write_episode(
            EpisodeCreate(user_id=new_user_id, session_id=session_id, role="user", content=content),
            embedding=embedder.embed_one(content),
        )

    retriever = MemoryRetriever(
        postgres_backend, Settings(retrieval_top_k_episodic=2, retrieval_top_n=2)
    )
    query = "What backend framework do I prefer?"
    result = await retriever.retrieve(new_user_id, embedder.embed_one(query))

    assert len(result.episodes) == 2
    top = result.episodes[0]
    assert top.episode.content == about_backend
    assert top.similarity > result.episodes[1].similarity


async def test_recency_boosts_older_but_similar_episode_less(
    postgres_backend: PostgresBackend,
    new_user_id: str,
    embedder: SentenceTransformerEmbeddingClient,
):
    session_id = uuid.uuid4()
    now = datetime.now(UTC)

    content = "I use FastAPI for backend development."
    recent = await postgres_backend.write_episode(
        EpisodeCreate(
            user_id=new_user_id,
            session_id=session_id,
            role="user",
            content=content,
            occurred_at=now - timedelta(days=1),
        ),
        embedding=embedder.embed_one(content),
    )
    old = await postgres_backend.write_episode(
        EpisodeCreate(
            user_id=new_user_id,
            session_id=session_id,
            role="user",
            content=content,
            occurred_at=now - timedelta(days=90),
        ),
        embedding=embedder.embed_one(content),
    )

    retriever = MemoryRetriever(
        postgres_backend, Settings(retrieval_top_k_episodic=2, retrieval_top_n=2)
    )
    result = await retriever.retrieve(new_user_id, embedder.embed_one(content), now=now)

    by_id = {e.episode.id: e for e in result.episodes}
    assert by_id[recent.id].score > by_id[old.id].score
    assert by_id[recent.id].recency_factor > by_id[old.id].recency_factor
