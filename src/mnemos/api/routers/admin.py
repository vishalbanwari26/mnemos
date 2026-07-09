import uuid
from datetime import datetime

from fastapi import APIRouter

from mnemos.api.deps import ConversationManagerDep, MemoryEngineDep
from mnemos.api.schemas import SeedRequest

router = APIRouter(tags=["admin"])


@router.post("/users/{user_id}/reset", status_code=204)
async def reset_user(user_id: str, memory: MemoryEngineDep) -> None:
    await memory.reset_user(user_id)


@router.post("/users/{user_id}/seed", response_model=list[str])
async def seed_user(user_id: str, body: SeedRequest, manager: ConversationManagerDep) -> list[str]:
    session_id = body.session_id or uuid.uuid4()
    occurred_at = datetime.fromisoformat(body.occurred_at) if body.occurred_at else None
    turns = [(t.role, t.content) for t in body.turns]
    facts = await manager.seed_session(user_id, session_id, turns, occurred_at=occurred_at)
    return [f.fact for f in facts]
