import random
import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from mnemos.config import Settings, get_settings
from mnemos.db.models import ProceduralStrategy, ProceduralTurn, User
from mnemos.db.session import get_sessionmaker

# Three retrieval strategies = different (similarity_weight, recency_weight)
# pairs. "balanced" is the fixed default from the v1 retriever; the other two
# let procedural memory discover, per user, whether that default is actually
# the best fit for how they use the assistant.
STRATEGIES: dict[str, tuple[float, float]] = {
    "semantic_heavy": (0.9, 0.1),
    "balanced": (0.7, 0.3),
    "recency_heavy": (0.4, 0.6),
}

# Heuristic, not a research contribution: if the user's next message looks
# like a correction, the previous turn's retrieval strategy is scored as a
# failure. Crude but real and inspectable — not a hidden fake metric.
CORRECTION_CUES = [
    "no,",
    "no i",
    "that's wrong",
    "that is wrong",
    "actually,",
    "actually i",
    "i didn't say that",
    "i never said",
    "that's not right",
    "that's incorrect",
    "not what i said",
    "not what i meant",
]


def contains_correction_cue(text: str) -> bool:
    lowered = text.lower()
    return any(cue in lowered for cue in CORRECTION_CUES)


class ProceduralMemory:
    """Epsilon-greedy selection over empirical per-user strategy success
    rates. Always backed by Postgres regardless of STORAGE_BACKEND — this is
    small operational metadata, not the memory content the backend
    comparison is about (see storage/base.py).
    """

    def __init__(
        self,
        settings: Settings | None = None,
        sessionmaker: async_sessionmaker[AsyncSession] | None = None,
    ):
        self.settings = settings or get_settings()
        self._sessionmaker = sessionmaker or get_sessionmaker()

    async def choose_strategy(self, user_id: str) -> tuple[str, float, float]:
        """Returns (strategy_name, similarity_weight, recency_weight)."""
        async with self._sessionmaker() as session:
            result = await session.execute(
                select(ProceduralStrategy).where(ProceduralStrategy.user_id == user_id)
            )
            stats = {row.strategy_name: row for row in result.scalars().all()}

        if random.random() < self.settings.procedural_epsilon:
            name = random.choice(list(STRATEGIES.keys()))
        else:

            def success_rate(strategy_name: str) -> float:
                row = stats.get(strategy_name)
                if row is None or row.uses == 0:
                    return 0.5  # unseen strategies get a neutral prior, not last place
                return row.successes / row.uses

            name = max(STRATEGIES.keys(), key=success_rate)

        sim_weight, recency_weight = STRATEGIES[name]
        return name, sim_weight, recency_weight

    async def record_pending_turn(
        self, session_id: uuid.UUID, user_id: str, strategy_name: str
    ) -> None:
        """Marks the strategy used for this turn's reply as awaiting an
        outcome assessment on the next turn in the same session."""
        async with self._sessionmaker() as session:
            if await session.get(User, user_id) is None:
                session.add(User(id=user_id))
                await session.flush()

            existing = await session.get(ProceduralTurn, session_id)
            if existing is None:
                session.add(
                    ProceduralTurn(
                        session_id=session_id, user_id=user_id, strategy_name=strategy_name
                    )
                )
            else:
                existing.user_id = user_id
                existing.strategy_name = strategy_name
            await session.commit()

    async def resolve_pending_turn(self, session_id: uuid.UUID, next_user_message: str) -> None:
        """Scores the previous turn's strategy using the user's follow-up
        message, then clears the pending marker. No-op if there's nothing
        pending (e.g. the first turn in a session)."""
        async with self._sessionmaker() as session:
            pending = await session.get(ProceduralTurn, session_id)
            if pending is None:
                return

            success = not contains_correction_cue(next_user_message)
            row = await session.get(ProceduralStrategy, (pending.user_id, pending.strategy_name))
            if row is None:
                row = ProceduralStrategy(
                    user_id=pending.user_id,
                    strategy_name=pending.strategy_name,
                    uses=0,
                    successes=0,
                )
                session.add(row)
            row.uses += 1
            if success:
                row.successes += 1

            await session.delete(pending)
            await session.commit()

    async def get_stats(self, user_id: str) -> list[ProceduralStrategy]:
        async with self._sessionmaker() as session:
            result = await session.execute(
                select(ProceduralStrategy).where(ProceduralStrategy.user_id == user_id)
            )
            return list(result.scalars().all())
