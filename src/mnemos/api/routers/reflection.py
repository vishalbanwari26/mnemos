from fastapi import APIRouter

from mnemos.api.deps import ReflectionEngineDep
from mnemos.api.schemas import ReflectionLogResponse, ReflectionRunResponse

router = APIRouter(tags=["reflection"])


@router.post("/users/{user_id}/reflect", response_model=ReflectionRunResponse)
async def run_reflection(user_id: str, reflection: ReflectionEngineDep) -> ReflectionRunResponse:
    summary = await reflection.run(user_id)
    return ReflectionRunResponse(
        facts_merged_into=summary.facts_merged_into,
        facts_decayed=summary.facts_decayed,
        facts_forgotten=summary.facts_forgotten,
    )


@router.get("/users/{user_id}/reflection-log", response_model=ReflectionLogResponse)
async def get_reflection_log(
    user_id: str, reflection: ReflectionEngineDep
) -> ReflectionLogResponse:
    entries = await reflection.get_log(user_id)
    return ReflectionLogResponse(entries=entries)
