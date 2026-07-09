import uuid
from abc import ABC, abstractmethod
from datetime import datetime

from mnemos.memory.schemas import EpisodeCreate, EpisodeRead, SemanticFactCreate, SemanticFactRead

# search_* returns (record, distance) pairs. distance is cosine distance in
# [0, 2] (0 = identical), matching pgvector's `<=>` convention — every
# backend normalizes to this so the scoring logic in memory/retrieval.py is
# written once and shared by all of them.
EpisodeMatch = tuple[EpisodeRead, float]
FactMatch = tuple[SemanticFactRead, float]


class StorageBackend(ABC):
    """Storage for episodic + semantic memory content. Each method is a
    self-contained unit of work (opens/commits internally) — callers never
    manage a session or transaction boundary, which is what lets Postgres,
    Neo4j, and Qdrant sit behind the same interface despite very different
    connection models.

    Procedural memory and the reflection log are deliberately NOT part of
    this interface — they're small operational metadata that always lives in
    Postgres regardless of which backend is selected here (see
    memory/procedural.py, memory/reflection.py).
    """

    @abstractmethod
    async def write_episode(
        self, episode: EpisodeCreate, embedding: list[float]
    ) -> EpisodeRead: ...

    @abstractmethod
    async def get_episodes(
        self, user_id: str, *, session_id: uuid.UUID | None = None, limit: int = 100
    ) -> list[EpisodeRead]: ...

    @abstractmethod
    async def search_episodes(
        self, user_id: str, query_embedding: list[float], top_k: int
    ) -> list[EpisodeMatch]: ...

    @abstractmethod
    async def delete_episodes_for_user(self, user_id: str) -> None: ...

    @abstractmethod
    async def write_fact(
        self, fact: SemanticFactCreate, embedding: list[float]
    ) -> SemanticFactRead: ...

    @abstractmethod
    async def get_facts(
        self, user_id: str, *, status: str = "active", limit: int = 200
    ) -> list[SemanticFactRead]: ...

    @abstractmethod
    async def search_facts(
        self, user_id: str, query_embedding: list[float], top_k: int, *, status: str = "active"
    ) -> list[FactMatch]: ...

    @abstractmethod
    async def update_fact_status(
        self, fact_id: uuid.UUID, status: str, *, confidence: float | None = None
    ) -> None: ...

    @abstractmethod
    async def reinforce_fact(self, fact_id: uuid.UUID, *, at: datetime | None = None) -> None:
        """Bump last_reinforced_at and nudge confidence up slightly (capped at
        1.0) — called whenever a fact actually makes it into a retrieval
        result used to answer. The reflection pass's decay logic is the
        inverse of this: facts nobody asks about fade."""
        ...

    @abstractmethod
    async def delete_facts_for_user(self, user_id: str) -> None: ...

    async def close(self) -> None:
        """Release any held connections/handles. No-op by default."""
        return None
