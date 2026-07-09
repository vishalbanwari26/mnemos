import uuid

from pydantic import BaseModel

from mnemos.memory.schemas import EpisodeRead, ReflectionLogRead, SemanticFactRead


class ChatRequest(BaseModel):
    message: str
    session_id: uuid.UUID | None = None


class ChatResponse(BaseModel):
    reply: str
    session_id: uuid.UUID
    memory_used: int
    facts_learned: list[str]
    strategy_used: str


class ProceduralStrategyStats(BaseModel):
    strategy_name: str
    uses: int
    successes: int
    success_rate: float


class ProceduralStatsResponse(BaseModel):
    strategies: list[ProceduralStrategyStats]


class EpisodicMemoriesResponse(BaseModel):
    episodes: list[EpisodeRead]


class SemanticMemoriesResponse(BaseModel):
    facts: list[SemanticFactRead]


class SeedTurn(BaseModel):
    role: str
    content: str


class SeedRequest(BaseModel):
    session_id: uuid.UUID | None = None
    turns: list[SeedTurn]
    # ISO 8601; lets callers (e.g. the benchmark) backdate a seed session.
    occurred_at: str | None = None


class ReflectionRunResponse(BaseModel):
    facts_merged_into: int
    facts_decayed: int
    facts_forgotten: int


class ReflectionLogResponse(BaseModel):
    entries: list[ReflectionLogRead]


class RetrievalTraceRequest(BaseModel):
    query: str
    # ISO 8601. "Time travel": scores recency as of this moment instead of
    # now, reusing MemoryEngine.recall(now=...) from the benchmark harness.
    as_of: str | None = None


class DailyCount(BaseModel):
    date: str  # YYYY-MM-DD
    count: int


class MemoryStatsResponse(BaseModel):
    episodic_total: int
    semantic_active: int
    semantic_merged: int
    semantic_forgotten: int
    episodic_by_day: list[DailyCount]
    semantic_by_day: list[DailyCount]
