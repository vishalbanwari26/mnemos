import math
from datetime import UTC, datetime

from mnemos.config import Settings, get_settings
from mnemos.memory.schemas import MemoryQueryResult, ScoredEpisode, ScoredFact
from mnemos.storage.base import EpisodeMatch, FactMatch, StorageBackend

# Scoring is deliberately simple, not a research contribution: semantic facts
# are treated as durable knowledge and ranked mostly on similarity; episodic
# turns get an exponential recency boost so recent context still surfaces
# even when it's a slightly weaker semantic match.
#
# These are pure functions over (record, distance) pairs — any StorageBackend
# implementation feeds them the same way, so the scoring logic is written
# once and shared by Postgres, Neo4j, and Qdrant alike.


def score_episodes(
    matches: list[EpisodeMatch], now: datetime, settings: Settings
) -> list[ScoredEpisode]:
    scored = []
    for episode, distance in matches:
        similarity = 1.0 - distance
        age_days = max((now - episode.occurred_at).total_seconds() / 86400.0, 0.0)
        recency_factor = math.exp(-age_days / settings.recency_half_life_days)
        score = (
            similarity * settings.retrieval_similarity_weight
            + recency_factor * settings.retrieval_recency_weight
        )
        scored.append(
            ScoredEpisode(
                episode=episode, similarity=similarity, recency_factor=recency_factor, score=score
            )
        )
    return scored


def score_facts(matches: list[FactMatch]) -> list[ScoredFact]:
    return [
        ScoredFact(fact=fact, similarity=1.0 - distance, score=1.0 - distance)
        for fact, distance in matches
    ]


def merge_and_rank(
    episodes: list[ScoredEpisode], facts: list[ScoredFact], top_n: int
) -> MemoryQueryResult:
    all_scored: list[ScoredEpisode | ScoredFact] = [*episodes, *facts]
    combined = sorted(all_scored, key=lambda x: x.score, reverse=True)[:top_n]
    return MemoryQueryResult(
        episodes=[e for e in combined if isinstance(e, ScoredEpisode)],
        facts=[f for f in combined if isinstance(f, ScoredFact)],
    )


class MemoryRetriever:
    """Thin orchestration over a StorageBackend: search both tables, apply
    the shared scoring formula above, merge and rank.
    """

    def __init__(self, backend: StorageBackend, settings: Settings | None = None):
        self.backend = backend
        self.settings = settings or get_settings()

    async def retrieve(
        self,
        user_id: str,
        query_embedding: list[float],
        *,
        now: datetime | None = None,
        similarity_weight: float | None = None,
        recency_weight: float | None = None,
    ) -> MemoryQueryResult:
        """`similarity_weight`/`recency_weight` let a caller (procedural
        memory's chosen strategy) override the configured default scoring
        weights for this one call, without mutating shared settings."""
        now = now or datetime.now(UTC)
        s = self.settings
        if similarity_weight is not None or recency_weight is not None:
            s = s.model_copy(
                update={
                    "retrieval_similarity_weight": similarity_weight
                    if similarity_weight is not None
                    else s.retrieval_similarity_weight,
                    "retrieval_recency_weight": recency_weight
                    if recency_weight is not None
                    else s.retrieval_recency_weight,
                }
            )

        episode_matches = await self.backend.search_episodes(
            user_id, query_embedding, s.retrieval_top_k_episodic
        )
        fact_matches = await self.backend.search_facts(
            user_id, query_embedding, s.retrieval_top_k_semantic
        )
        episodes = score_episodes(episode_matches, now, s)
        facts = score_facts(fact_matches)
        return merge_and_rank(episodes, facts, s.retrieval_top_n)
