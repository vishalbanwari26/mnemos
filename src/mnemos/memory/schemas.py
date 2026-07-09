import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict


class EpisodeCreate(BaseModel):
    user_id: str
    session_id: uuid.UUID
    role: Literal["user", "assistant"]
    content: str
    occurred_at: datetime | None = None
    metadata: dict = {}


class EpisodeRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    user_id: str
    session_id: uuid.UUID
    role: str
    content: str
    created_at: datetime
    occurred_at: datetime


class SemanticFactCreate(BaseModel):
    user_id: str
    fact: str
    source_episode_ids: list[str] = []
    confidence: float = 1.0
    metadata: dict = {}


class SemanticFactRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    user_id: str
    fact: str
    confidence: float
    status: str
    source_episode_ids: list[str]
    created_at: datetime
    updated_at: datetime
    last_reinforced_at: datetime


class ScoredEpisode(BaseModel):
    episode: EpisodeRead
    similarity: float
    recency_factor: float
    score: float


class ScoredFact(BaseModel):
    fact: SemanticFactRead
    similarity: float
    score: float


class MemoryQueryResult(BaseModel):
    episodes: list[ScoredEpisode]
    facts: list[ScoredFact]


class ReflectionLogRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    user_id: str
    action: str
    fact_ids: list[str]
    detail: str
    created_at: datetime
