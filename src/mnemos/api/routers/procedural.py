from fastapi import APIRouter

from mnemos.api.deps import ProceduralMemoryDep
from mnemos.api.schemas import ProceduralStatsResponse, ProceduralStrategyStats

router = APIRouter(tags=["procedural"])


@router.get("/users/{user_id}/procedural", response_model=ProceduralStatsResponse)
async def get_procedural_stats(
    user_id: str, procedural: ProceduralMemoryDep
) -> ProceduralStatsResponse:
    stats = await procedural.get_stats(user_id)
    return ProceduralStatsResponse(
        strategies=[
            ProceduralStrategyStats(
                strategy_name=row.strategy_name,
                uses=row.uses,
                successes=row.successes,
                success_rate=(row.successes / row.uses) if row.uses else 0.0,
            )
            for row in stats
        ]
    )
