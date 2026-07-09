import uuid

from fastapi import APIRouter

from mnemos.api.deps import MemoryEngineDep, ReflectionEngineDep
from mnemos.api.schemas import EpisodicMemoriesResponse, SemanticMemoriesResponse

router = APIRouter(tags=["memories"])


@router.get("/users/{user_id}/memories/episodic", response_model=EpisodicMemoriesResponse)
async def list_episodic_memories(user_id: str, memory: MemoryEngineDep) -> EpisodicMemoriesResponse:
    episodes = await memory.list_episodes(user_id)
    return EpisodicMemoriesResponse(episodes=episodes)


@router.get("/users/{user_id}/memories/semantic", response_model=SemanticMemoriesResponse)
async def list_semantic_memories(
    user_id: str, memory: MemoryEngineDep, status: str = "active"
) -> SemanticMemoriesResponse:
    facts = await memory.list_facts(user_id, status=status)
    return SemanticMemoriesResponse(facts=facts)


@router.delete("/users/{user_id}/memories/semantic/{fact_id}", status_code=204)
async def forget_semantic_memory(
    user_id: str,
    fact_id: uuid.UUID,
    reflection: ReflectionEngineDep,
    reason: str = "user_requested",
) -> None:
    await reflection.forget_fact(user_id, fact_id, reason=reason)
