from typing import Annotated

from fastapi import Depends, Request

from mnemos.agent.conversation import ConversationManager
from mnemos.config import Settings
from mnemos.memory.engine import MemoryEngine
from mnemos.memory.procedural import ProceduralMemory
from mnemos.memory.reflection import ReflectionEngine


def get_memory_engine(request: Request) -> MemoryEngine:
    settings: Settings = request.app.state.settings
    return MemoryEngine(
        request.app.state.storage_backend, request.app.state.embedding_client, settings
    )


MemoryEngineDep = Annotated[MemoryEngine, Depends(get_memory_engine)]


def get_procedural_memory(request: Request) -> ProceduralMemory:
    # Always Postgres, regardless of STORAGE_BACKEND — see memory/procedural.py.
    return ProceduralMemory(request.app.state.settings)


ProceduralMemoryDep = Annotated[ProceduralMemory, Depends(get_procedural_memory)]


def get_conversation_manager(
    request: Request, memory: MemoryEngineDep, procedural: ProceduralMemoryDep
) -> ConversationManager:
    settings: Settings = request.app.state.settings
    return ConversationManager(
        memory,
        request.app.state.llm_client,
        request.app.state.extraction_llm_client,
        settings,
        procedural,
    )


ConversationManagerDep = Annotated[ConversationManager, Depends(get_conversation_manager)]


def get_reflection_engine(request: Request, memory: MemoryEngineDep) -> ReflectionEngine:
    settings: Settings = request.app.state.settings
    # Consolidation reuses the extraction model — same "small structured LLM
    # task" role as fact extraction, not the (possibly pricier) chat model.
    return ReflectionEngine(memory, request.app.state.extraction_llm_client, settings)


ReflectionEngineDep = Annotated[ReflectionEngine, Depends(get_reflection_engine)]
