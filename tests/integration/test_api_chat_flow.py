import os
from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncEngine

from mnemos.db.models import Base

pytestmark = pytest.mark.asyncio


@pytest_asyncio.fixture
async def api_client(engine: AsyncEngine) -> AsyncIterator[AsyncClient]:
    """Spins up the real FastAPI app (mock LLM + mock embeddings, for speed)
    against the same Postgres+pgvector used by the other integration/unit
    tests, and exercises it purely over HTTP via ASGI transport.
    """
    os.environ["DATABASE_URL"] = engine.url.render_as_string(hide_password=False)
    os.environ["LLM_PROVIDER"] = "mock"
    os.environ["EMBEDDING_PROVIDER"] = "mock"

    import mnemos.config as config_module
    import mnemos.db.session as session_module
    import mnemos.embeddings.factory as embedding_factory

    config_module.get_settings.cache_clear()
    session_module.get_engine.cache_clear()
    embedding_factory._cached_client = None

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    from mnemos.api.main import create_app

    app = create_app()
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            yield client

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

    config_module.get_settings.cache_clear()
    session_module.get_engine.cache_clear()


async def test_chat_flow_stores_and_recalls_memory(api_client: AsyncClient, new_user_id: str):
    await api_client.post(f"/users/{new_user_id}/reset")

    r1 = await api_client.post(
        f"/users/{new_user_id}/messages", json={"message": "I use FastAPI for backend work."}
    )
    assert r1.status_code == 200
    body1 = r1.json()
    assert body1["memory_used"] == 0

    r2 = await api_client.post(
        f"/users/{new_user_id}/messages", json={"message": "I use FastAPI for backend work."}
    )
    assert r2.status_code == 200
    assert r2.json()["memory_used"] > 0

    episodic = await api_client.get(f"/users/{new_user_id}/memories/episodic")
    assert episodic.status_code == 200
    assert len(episodic.json()["episodes"]) == 4


async def test_reset_clears_memory(api_client: AsyncClient, new_user_id: str):
    await api_client.post(f"/users/{new_user_id}/messages", json={"message": "hello"})
    await api_client.post(f"/users/{new_user_id}/reset")

    episodic = await api_client.get(f"/users/{new_user_id}/memories/episodic")
    assert episodic.json()["episodes"] == []


async def test_seed_endpoint_writes_turns_and_extracts_facts(
    api_client: AsyncClient, new_user_id: str
):
    r = await api_client.post(
        f"/users/{new_user_id}/seed",
        json={
            "turns": [
                {"role": "user", "content": "I use FastAPI for backend work."},
                {"role": "assistant", "content": "Nice choice."},
            ]
        },
    )
    assert r.status_code == 200

    episodic = await api_client.get(f"/users/{new_user_id}/memories/episodic")
    assert len(episodic.json()["episodes"]) == 2


async def test_retrieval_trace_returns_scored_result_without_calling_llm(
    api_client: AsyncClient, new_user_id: str
):
    await api_client.post(
        f"/users/{new_user_id}/seed",
        json={"turns": [{"role": "user", "content": "I use FastAPI for backend work."}]},
    )

    r = await api_client.post(
        f"/users/{new_user_id}/retrieval-trace", json={"query": "backend framework"}
    )
    assert r.status_code == 200
    body = r.json()
    assert "episodes" in body and "facts" in body
    assert len(body["episodes"]) == 1
    assert 0 <= body["episodes"][0]["score"] <= 2


async def test_retrieval_trace_accepts_naive_as_of_from_datetime_local_input(
    api_client: AsyncClient, new_user_id: str
):
    """<input type="datetime-local"> sends a naive ISO string with no
    timezone offset — the backend must not crash comparing it against
    tz-aware occurred_at timestamps (see insights._parse_as_of)."""
    await api_client.post(
        f"/users/{new_user_id}/seed",
        json={"turns": [{"role": "user", "content": "I use FastAPI for backend work."}]},
    )

    r = await api_client.post(
        f"/users/{new_user_id}/retrieval-trace",
        json={"query": "backend framework", "as_of": "2026-07-09T14:30"},
    )
    assert r.status_code == 200


async def test_stats_endpoint_reports_counts(api_client: AsyncClient, new_user_id: str):
    await api_client.post(f"/users/{new_user_id}/messages", json={"message": "hello"})

    r = await api_client.get(f"/users/{new_user_id}/stats")
    assert r.status_code == 200
    body = r.json()
    assert body["episodic_total"] == 2
    assert len(body["episodic_by_day"]) == 1


async def test_procedural_and_reflection_endpoints_are_reachable(
    api_client: AsyncClient, new_user_id: str
):
    r = await api_client.get(f"/users/{new_user_id}/procedural")
    assert r.status_code == 200
    assert r.json() == {"strategies": []}

    r = await api_client.post(f"/users/{new_user_id}/reflect")
    assert r.status_code == 200
    assert r.json() == {"facts_merged_into": 0, "facts_decayed": 0, "facts_forgotten": 0}

    r = await api_client.get(f"/users/{new_user_id}/reflection-log")
    assert r.status_code == 200
    assert r.json() == {"entries": []}
