import pytest

from mnemos.config import Settings
from mnemos.embeddings.mock_client import MockEmbeddingClient
from mnemos.llm.base import LLMResponse, ToolCall
from mnemos.llm.mock_client import MockLLMClient
from mnemos.memory.engine import MemoryEngine
from mnemos.memory.extraction import extract_semantic_facts
from mnemos.storage.postgres_backend import PostgresBackend

pytestmark = pytest.mark.asyncio


def _record_facts_response(facts: list[dict]) -> LLMResponse:
    return LLMResponse(
        content="",
        tool_calls=[ToolCall(name="record_facts", input={"facts": facts})],
        model="mock",
    )


async def test_extraction_writes_facts_from_tool_call(
    postgres_backend: PostgresBackend, new_user_id: str
):
    memory = MemoryEngine(postgres_backend, MockEmbeddingClient())
    llm = MockLLMClient(
        responses=[
            _record_facts_response([{"statement": "prefers FastAPI over Flask", "confidence": 0.9}])
        ]
    )

    written = await extract_semantic_facts(memory, llm, new_user_id, "user: I use FastAPI now.")

    assert len(written) == 1
    assert written[0].fact == "prefers FastAPI over Flask"

    facts = await memory.list_facts(new_user_id)
    assert len(facts) == 1


async def test_extraction_skips_empty_facts_list(
    postgres_backend: PostgresBackend, new_user_id: str
):
    memory = MemoryEngine(postgres_backend, MockEmbeddingClient())
    llm = MockLLMClient(responses=[_record_facts_response([])])

    written = await extract_semantic_facts(memory, llm, new_user_id, "user: hey")

    assert written == []
    assert await memory.list_facts(new_user_id) == []


async def test_extraction_dedups_near_identical_fact(
    postgres_backend: PostgresBackend, new_user_id: str
):
    memory = MemoryEngine(postgres_backend, MockEmbeddingClient())
    settings = Settings(semantic_dedup_threshold=0.92)

    fact = {"statement": "prefers FastAPI over Flask", "confidence": 1.0}
    llm = MockLLMClient(responses=[_record_facts_response([fact]), _record_facts_response([fact])])

    first = await extract_semantic_facts(memory, llm, new_user_id, "turn 1", settings=settings)
    second = await extract_semantic_facts(memory, llm, new_user_id, "turn 2", settings=settings)

    assert len(first) == 1
    assert len(second) == 0  # exact-duplicate statement -> same mock embedding -> deduped
    assert len(await memory.list_facts(new_user_id)) == 1
