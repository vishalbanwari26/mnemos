import uuid

import pytest

from mnemos.agent.conversation import ConversationManager
from mnemos.embeddings.mock_client import MockEmbeddingClient
from mnemos.llm.base import LLMResponse
from mnemos.llm.mock_client import MockLLMClient
from mnemos.memory.engine import MemoryEngine
from mnemos.memory.procedural import ProceduralMemory
from mnemos.storage.postgres_backend import PostgresBackend

pytestmark = pytest.mark.asyncio


async def test_handle_message_stores_both_turns_and_returns_reply(
    postgres_backend: PostgresBackend, procedural_memory: ProceduralMemory, new_user_id: str
):
    memory = MemoryEngine(postgres_backend, MockEmbeddingClient())
    llm = MockLLMClient(responses=[LLMResponse(content="Nice to meet you.", model="mock")])
    manager = ConversationManager(memory, llm, procedural=procedural_memory)

    result = await manager.handle_message(new_user_id, uuid.uuid4(), "Hi, I'm Vishal.")

    assert result.reply == "Nice to meet you."
    episodes = await memory.list_episodes(new_user_id)
    assert len(episodes) == 2
    roles = {e.role for e in episodes}
    assert roles == {"user", "assistant"}


async def test_second_turn_retrieves_memory_from_first(
    postgres_backend: PostgresBackend, procedural_memory: ProceduralMemory, new_user_id: str
):
    memory = MemoryEngine(postgres_backend, MockEmbeddingClient())
    llm = MockLLMClient(
        responses=[
            LLMResponse(content="Got it.", model="mock"),
            LLMResponse(content="Sure.", model="mock"),
        ]
    )
    manager = ConversationManager(memory, llm, procedural=procedural_memory)
    session_id = uuid.uuid4()

    await manager.handle_message(new_user_id, session_id, "I use FastAPI.")
    result = await manager.handle_message(new_user_id, session_id, "I use FastAPI.")

    assert result.memory_used > 0
