import os
import uuid
from collections.abc import AsyncIterator, Iterator
from typing import Any

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from mnemos.db.models import Base

# CI (and anyone with Docker) gets an ephemeral, isolated Postgres via
# testcontainers. Locally, without Docker, set MNEMOS_TEST_DATABASE_URL to point
# at a scratch Postgres+pgvector database instead (never point this at a real
# dev/prod database — table contents are dropped between tests).
_LOCAL_TEST_DB_URL = os.environ.get("MNEMOS_TEST_DATABASE_URL")


@pytest.fixture(scope="session")
def postgres_container() -> Iterator[Any | None]:
    if _LOCAL_TEST_DB_URL:
        yield None
        return
    from testcontainers.postgres import PostgresContainer

    with PostgresContainer("pgvector/pgvector:pg16", driver="asyncpg") as container:
        yield container


@pytest_asyncio.fixture(scope="session")
async def engine(postgres_container) -> AsyncIterator[AsyncEngine]:
    url = _LOCAL_TEST_DB_URL or postgres_container.get_connection_url()
    eng = create_async_engine(url)
    async with eng.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    yield eng
    await eng.dispose()


@pytest_asyncio.fixture
async def clean_tables(engine: AsyncEngine) -> AsyncIterator[AsyncEngine]:
    """Creates all tables before the test, drops them after. Yields the
    engine itself so callers can build either a raw session or a
    PostgresBackend pointed at the same, freshly-schema'd database."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest_asyncio.fixture
async def db_session(clean_tables: AsyncEngine) -> AsyncIterator[AsyncSession]:
    sessionmaker = async_sessionmaker(clean_tables, expire_on_commit=False)
    async with sessionmaker() as session:
        yield session


@pytest_asyncio.fixture
async def postgres_backend(clean_tables: AsyncEngine):
    from mnemos.storage.postgres_backend import PostgresBackend

    sessionmaker = async_sessionmaker(clean_tables, expire_on_commit=False)
    return PostgresBackend(sessionmaker=sessionmaker)


@pytest.fixture(scope="session")
def qdrant_backend(tmp_path_factory: pytest.TempPathFactory):
    """Session-scoped: Qdrant's local-mode storage directory is locked to a
    single client, so it's opened once and shared across tests, relying on
    each test's unique new_user_id for isolation (same pattern the app uses
    at request/CLI scope)."""
    from mnemos.storage.qdrant_backend import QdrantBackend

    path = str(tmp_path_factory.mktemp("qdrant"))
    return QdrantBackend(path=path)


@pytest_asyncio.fixture(scope="session")
async def neo4j_backend():
    """Session-scoped, like qdrant_backend — Neo4j is a shared server, not
    directory-locked, but reusing one driver across tests is still the
    correct lifecycle (matches how the app opens it once at startup).
    Skipped if no local Neo4j is reachable (e.g. not installed, or CI without
    the service configured) rather than failing the whole suite.
    """
    from neo4j.exceptions import ServiceUnavailable

    from mnemos.storage.neo4j_backend import Neo4jBackend

    uri = os.environ.get("MNEMOS_TEST_NEO4J_URI", "bolt://localhost:7687")
    user = os.environ.get("MNEMOS_TEST_NEO4J_USER", "neo4j")
    password = os.environ.get("MNEMOS_TEST_NEO4J_PASSWORD", "mnemos-neo4j")

    backend = Neo4jBackend(uri=uri, user=user, password=password)
    try:
        await backend._driver.verify_connectivity()
        # Neo4j has no native pre-filter on its vector index (unlike Qdrant),
        # so search over-fetches candidates and filters in Cypher. Leftover
        # nodes from prior test runs — this is a real, shared local server,
        # not an ephemeral testcontainer/tmpdir — can crowd a small user_id
        # out of that over-fetch window, especially with the deterministic
        # MockEmbeddingClient producing identical vectors for repeated test
        # strings across many users. Clear the slate once per test session.
        async with backend._driver.session() as session:
            await session.run("MATCH (n) DETACH DELETE n")
    except ServiceUnavailable:
        pytest.skip(f"No Neo4j reachable at {uri} — skipping Neo4j backend-contract tests")
    yield backend
    await backend.close()


@pytest_asyncio.fixture
async def procedural_memory(clean_tables: AsyncEngine):
    """ProceduralMemory always uses Postgres (see memory/procedural.py) —
    pointed at the same freshly-schema'd test engine as postgres_backend,
    never the real dev-DB sessionmaker its default constructor would use."""
    from mnemos.memory.procedural import ProceduralMemory

    sessionmaker = async_sessionmaker(clean_tables, expire_on_commit=False)
    return ProceduralMemory(sessionmaker=sessionmaker)


@pytest.fixture
def new_user_id() -> str:
    return f"test-user-{uuid.uuid4().hex[:8]}"
