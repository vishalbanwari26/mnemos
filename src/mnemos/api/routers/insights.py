from collections import Counter
from datetime import UTC, datetime

from fastapi import APIRouter

from mnemos.api.deps import MemoryEngineDep
from mnemos.api.schemas import DailyCount, MemoryStatsResponse, RetrievalTraceRequest
from mnemos.memory.schemas import MemoryQueryResult

router = APIRouter(tags=["insights"])


def _parse_as_of(value: str | None) -> datetime | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(value)
    # <input type="datetime-local"> sends a naive string with no offset;
    # retrieval compares this against tz-aware occurred_at timestamps, so a
    # naive value here would raise. Assume the browser meant UTC.
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


@router.post("/users/{user_id}/retrieval-trace", response_model=MemoryQueryResult)
async def retrieval_trace(
    user_id: str, body: RetrievalTraceRequest, memory: MemoryEngineDep
) -> MemoryQueryResult:
    """Runs recall() without calling the LLM — what would be retrieved, with
    per-item similarity/recency/score. Powers the dashboard's retrieval
    trace viewer and, via `as_of`, "time travel" (reusing the same
    `now=` parameter the benchmark harness uses to simulate elapsed time).
    """
    as_of = _parse_as_of(body.as_of)
    return await memory.recall(user_id, body.query, now=as_of)


def _daily_counts(dates: list[datetime]) -> list[DailyCount]:
    counts = Counter(d.date().isoformat() for d in dates)
    return [DailyCount(date=date, count=count) for date, count in sorted(counts.items())]


@router.get("/users/{user_id}/stats", response_model=MemoryStatsResponse)
async def get_stats(user_id: str, memory: MemoryEngineDep) -> MemoryStatsResponse:
    episodes = await memory.list_episodes(user_id, limit=10_000)
    active = await memory.list_facts(user_id, status="active", limit=10_000)
    merged = await memory.list_facts(user_id, status="merged", limit=10_000)
    forgotten = await memory.list_facts(user_id, status="forgotten", limit=10_000)

    return MemoryStatsResponse(
        episodic_total=len(episodes),
        semantic_active=len(active),
        semantic_merged=len(merged),
        semantic_forgotten=len(forgotten),
        episodic_by_day=_daily_counts([e.occurred_at for e in episodes]),
        semantic_by_day=_daily_counts([f.created_at for f in active + merged + forgotten]),
    )
