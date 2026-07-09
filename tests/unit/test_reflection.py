from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from mnemos.config import Settings
from mnemos.embeddings.mock_client import MockEmbeddingClient
from mnemos.llm.base import LLMResponse, ToolCall
from mnemos.llm.mock_client import MockLLMClient
from mnemos.memory.engine import MemoryEngine
from mnemos.memory.reflection import ReflectionEngine
from mnemos.storage.postgres_backend import PostgresBackend

pytestmark = pytest.mark.asyncio


def _consolidate_response(statement: str) -> LLMResponse:
    return LLMResponse(
        content="",
        tool_calls=[
            ToolCall(name="consolidate_facts", input={"consolidated_statement": statement})
        ],
        model="mock",
    )


@pytest.fixture
def reflection_engine(postgres_backend: PostgresBackend, clean_tables: AsyncEngine):
    memory = MemoryEngine(postgres_backend, MockEmbeddingClient())
    llm = MockLLMClient()
    sessionmaker = async_sessionmaker(clean_tables, expire_on_commit=False)
    return ReflectionEngine(memory, llm, sessionmaker=sessionmaker), memory


async def test_merge_consolidates_near_duplicate_facts(reflection_engine, new_user_id: str):
    reflection, memory = reflection_engine
    reflection.llm = MockLLMClient(
        responses=[_consolidate_response("prefers FastAPI over Flask for backend work")]
    )

    # Identical text -> identical mock embedding -> guaranteed to cluster,
    # standing in for "near-duplicate wording" without needing a real model.
    await memory.remember_fact(new_user_id, "prefers FastAPI over Flask")
    await memory.remember_fact(new_user_id, "prefers FastAPI over Flask")

    merged_count = await reflection.merge_duplicate_facts(new_user_id)
    assert merged_count == 1

    active = await memory.list_facts(new_user_id, status="active")
    merged = await memory.list_facts(new_user_id, status="merged")
    assert len(active) == 1
    assert active[0].fact == "prefers FastAPI over Flask for backend work"
    assert len(merged) == 2

    log = await reflection.get_log(new_user_id)
    assert len(log) == 1
    assert log[0].action == "merge"


async def test_decay_forgets_fact_below_confidence_threshold(reflection_engine, new_user_id: str):
    reflection, memory = reflection_engine
    reflection.settings = Settings(
        reflection_decay_days=0,
        reflection_decay_factor=0.1,
        reflection_forget_confidence_threshold=0.5,
    )

    await memory.remember_fact(new_user_id, "a fact nobody has needed in a while")

    decayed, forgotten = await reflection.decay_stale_facts(new_user_id)
    assert decayed == 0
    assert forgotten == 1

    active = await memory.list_facts(new_user_id, status="active")
    forgotten_facts = await memory.list_facts(new_user_id, status="forgotten")
    assert active == []
    assert len(forgotten_facts) == 1
    assert forgotten_facts[0].confidence == pytest.approx(0.1)

    log = await reflection.get_log(new_user_id)
    assert len(log) == 1
    assert log[0].action == "forget"


async def test_decay_reduces_confidence_without_forgetting(reflection_engine, new_user_id: str):
    reflection, memory = reflection_engine
    reflection.settings = Settings(
        reflection_decay_days=0,
        reflection_decay_factor=0.9,
        reflection_forget_confidence_threshold=0.1,
    )

    await memory.remember_fact(new_user_id, "a fact that fades slowly")

    decayed, forgotten = await reflection.decay_stale_facts(new_user_id)
    assert decayed == 1
    assert forgotten == 0

    active = await memory.list_facts(new_user_id, status="active")
    assert active[0].confidence == pytest.approx(0.9)


async def test_explicit_forget_marks_fact_forgotten_and_logs_reason(
    reflection_engine, new_user_id: str
):
    reflection, memory = reflection_engine
    fact = await memory.remember_fact(new_user_id, "outdated preference")

    await reflection.forget_fact(new_user_id, fact.id, reason="no longer accurate")

    assert await memory.list_facts(new_user_id, status="active") == []
    forgotten = await memory.list_facts(new_user_id, status="forgotten")
    assert len(forgotten) == 1

    log = await reflection.get_log(new_user_id)
    assert log[0].action == "forget"
    assert "no longer accurate" in log[0].detail


async def test_recall_reinforces_retrieved_facts(reflection_engine, new_user_id: str):
    reflection, memory = reflection_engine
    fact = await memory.remember_fact(new_user_id, "loves climbing")
    original_confidence = fact.confidence
    before = datetime.now(UTC) - timedelta(seconds=1)

    await memory.recall(new_user_id, "loves climbing")

    [refreshed] = await memory.list_facts(new_user_id)
    assert refreshed.confidence >= original_confidence
    assert refreshed.last_reinforced_at > before
