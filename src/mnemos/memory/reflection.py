import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from mnemos.config import Settings, get_settings
from mnemos.db.models import ReflectionLogEntry
from mnemos.db.session import get_sessionmaker
from mnemos.llm.base import LLMClient, Message
from mnemos.memory.engine import MemoryEngine
from mnemos.memory.schemas import ReflectionLogRead, SemanticFactRead

CONSOLIDATE_TOOL = {
    "name": "consolidate_facts",
    "description": (
        "Merge several related or overlapping facts about a user into one "
        "concise statement that preserves every distinct piece of information."
    ),
    "input_schema": {
        "type": "object",
        "properties": {"consolidated_statement": {"type": "string"}},
        "required": ["consolidated_statement"],
    },
}

CONSOLIDATE_SYSTEM_PROMPT = (
    "You merge several related or overlapping facts about a user into a single, "
    "concise statement. Don't drop any distinct detail from the originals."
)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


def _cluster_by_similarity(embeddings: list[list[float]], threshold: float) -> list[list[int]]:
    """Greedy clustering: O(n^2), fine at the per-user fact counts this
    operates on. Not a research contribution — a documented, simple default,
    same posture as the retrieval scoring formula."""
    n = len(embeddings)
    assigned = [False] * n
    clusters: list[list[int]] = []
    for i in range(n):
        if assigned[i]:
            continue
        cluster = [i]
        assigned[i] = True
        for j in range(i + 1, n):
            if not assigned[j] and _cosine_similarity(embeddings[i], embeddings[j]) >= threshold:
                cluster.append(j)
                assigned[j] = True
        clusters.append(cluster)
    return clusters


@dataclass
class ReflectionSummary:
    facts_merged_into: int  # number of new consolidated facts created
    facts_decayed: int
    facts_forgotten: int


class ReflectionEngine:
    """A consolidation pass over one user's semantic memory: merge near-
    duplicate facts, decay confidence on facts nobody has needed in a while,
    and archive (never hard-delete) anything that decays past the forget
    threshold. Triggered on demand (CLI command, API `POST .../reflect`),
    not on a schedule — there's no task queue in this project, and "on
    demand" is honest about what's actually implemented.

    This single pass covers what the original brainstorm called "memory
    compression" (the merge step) and "dreaming" (the whole pass, run
    offline/on demand) — deliberately not built as separate subsystems.
    """

    def __init__(
        self,
        memory: MemoryEngine,
        llm: LLMClient,
        settings: Settings | None = None,
        sessionmaker: async_sessionmaker[AsyncSession] | None = None,
    ):
        self.memory = memory
        self.llm = llm
        self.settings = settings or get_settings()
        self._sessionmaker = sessionmaker or get_sessionmaker()

    async def run(self, user_id: str) -> ReflectionSummary:
        merged = await self.merge_duplicate_facts(user_id)
        decayed, forgotten = await self.decay_stale_facts(user_id)
        return ReflectionSummary(
            facts_merged_into=merged, facts_decayed=decayed, facts_forgotten=forgotten
        )

    async def merge_duplicate_facts(self, user_id: str) -> int:
        facts = await self.memory.list_facts(user_id, status="active")
        if len(facts) < 2:
            return 0

        embeddings = [self.memory.embeddings.embed_one(f.fact) for f in facts]
        clusters = _cluster_by_similarity(embeddings, self.settings.reflection_merge_threshold)

        merged_count = 0
        for cluster_idxs in clusters:
            if len(cluster_idxs) < 2:
                continue
            cluster_facts = [facts[i] for i in cluster_idxs]
            consolidated_text = await self._consolidate(cluster_facts)
            source_ids = sorted({eid for f in cluster_facts for eid in f.source_episode_ids})

            new_fact = await self.memory.remember_fact(
                user_id,
                consolidated_text,
                source_episode_ids=source_ids,
                confidence=max(f.confidence for f in cluster_facts),
            )
            for f in cluster_facts:
                await self.memory.storage.update_fact_status(f.id, "merged")

            await self._log(
                user_id,
                "merge",
                [str(new_fact.id), *(str(f.id) for f in cluster_facts)],
                f"Merged {len(cluster_facts)} facts into: {consolidated_text!r}",
            )
            merged_count += 1
        return merged_count

    async def _consolidate(self, cluster_facts: list[SemanticFactRead]) -> str:
        facts_text = "\n".join(f"- {f.fact}" for f in cluster_facts)
        response = await self.llm.complete(
            messages=[Message(role="user", content=f"Facts to merge:\n{facts_text}")],
            system=CONSOLIDATE_SYSTEM_PROMPT,
            tools=[CONSOLIDATE_TOOL],
            tool_choice={"type": "tool", "name": "consolidate_facts"},
            max_tokens=256,
            temperature=0,
        )
        if response.tool_calls:
            statement = response.tool_calls[0].input.get("consolidated_statement", "").strip()
            if statement:
                return statement
        return "; ".join(f.fact for f in cluster_facts)  # fallback: never lose the facts

    async def decay_stale_facts(self, user_id: str) -> tuple[int, int]:
        facts = await self.memory.list_facts(user_id, status="active")
        now = datetime.now(UTC)
        decayed = 0
        forgotten = 0

        for f in facts:
            age_days = (now - f.last_reinforced_at).total_seconds() / 86400.0
            if age_days < self.settings.reflection_decay_days:
                continue

            new_confidence = f.confidence * self.settings.reflection_decay_factor
            if new_confidence < self.settings.reflection_forget_confidence_threshold:
                await self.memory.storage.update_fact_status(
                    f.id, "forgotten", confidence=new_confidence
                )
                await self._log(
                    user_id,
                    "forget",
                    [str(f.id)],
                    f"Confidence decayed to {new_confidence:.3f} (below threshold "
                    f"{self.settings.reflection_forget_confidence_threshold}) after "
                    f"{age_days:.1f} days unused: {f.fact!r}",
                )
                forgotten += 1
            else:
                await self.memory.storage.update_fact_status(
                    f.id, "active", confidence=new_confidence
                )
                await self._log(
                    user_id,
                    "decay",
                    [str(f.id)],
                    f"Confidence decayed {f.confidence:.3f} -> {new_confidence:.3f} "
                    f"after {age_days:.1f} days unused: {f.fact!r}",
                )
                decayed += 1
        return decayed, forgotten

    async def forget_fact(self, user_id: str, fact_id: uuid.UUID, *, reason: str) -> None:
        """Explicit, user-requested deletion — still a soft status transition
        (never a hard delete), so the audit trail stays complete."""
        await self.memory.storage.update_fact_status(fact_id, "forgotten")
        await self._log(user_id, "forget", [str(fact_id)], f"User requested deletion: {reason}")

    async def get_log(self, user_id: str, *, limit: int = 100) -> list[ReflectionLogRead]:
        async with self._sessionmaker() as session:
            result = await session.execute(
                select(ReflectionLogEntry)
                .where(ReflectionLogEntry.user_id == user_id)
                .order_by(ReflectionLogEntry.created_at.desc())
                .limit(limit)
            )
            return [ReflectionLogRead.model_validate(row) for row in result.scalars().all()]

    async def _log(self, user_id: str, action: str, fact_ids: list[str], detail: str) -> None:
        async with self._sessionmaker() as session:
            session.add(
                ReflectionLogEntry(user_id=user_id, action=action, fact_ids=fact_ids, detail=detail)
            )
            await session.commit()
