import uuid

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from mnemos.config import Settings
from mnemos.db.models import ProceduralStrategy
from mnemos.memory.procedural import ProceduralMemory, contains_correction_cue


def test_contains_correction_cue_detects_common_phrasings():
    assert contains_correction_cue("No, that's not what I use.")
    assert contains_correction_cue("Actually, I switched to Postgres.")
    assert contains_correction_cue("That's wrong, I never said that.")
    assert not contains_correction_cue("Yes, that's exactly right.")
    assert not contains_correction_cue("Tell me more about FastAPI.")


@pytest.fixture
def procedural(clean_tables: AsyncEngine) -> ProceduralMemory:
    sessionmaker = async_sessionmaker(clean_tables, expire_on_commit=False)
    return ProceduralMemory(sessionmaker=sessionmaker)


async def test_choose_strategy_defaults_to_a_known_strategy(
    procedural: ProceduralMemory, new_user_id: str
):
    name, sim_w, rec_w = await procedural.choose_strategy(new_user_id)
    assert name in {"semantic_heavy", "balanced", "recency_heavy"}
    assert 0 <= sim_w <= 1
    assert 0 <= rec_w <= 1


async def test_pending_turn_round_trip_records_success(
    procedural: ProceduralMemory, new_user_id: str
):
    session_id = uuid.uuid4()
    await procedural.record_pending_turn(session_id, new_user_id, "balanced")
    await procedural.resolve_pending_turn(session_id, "Yes, exactly right.")

    stats = await procedural.get_stats(new_user_id)
    assert len(stats) == 1
    assert stats[0].strategy_name == "balanced"
    assert stats[0].uses == 1
    assert stats[0].successes == 1


async def test_pending_turn_round_trip_records_failure_on_correction(
    procedural: ProceduralMemory, new_user_id: str
):
    session_id = uuid.uuid4()
    await procedural.record_pending_turn(session_id, new_user_id, "recency_heavy")
    await procedural.resolve_pending_turn(session_id, "No, that's not right at all.")

    stats = await procedural.get_stats(new_user_id)
    assert stats[0].uses == 1
    assert stats[0].successes == 0


async def test_resolve_with_no_pending_turn_is_a_noop(
    procedural: ProceduralMemory, new_user_id: str
):
    await procedural.resolve_pending_turn(uuid.uuid4(), "anything")
    assert await procedural.get_stats(new_user_id) == []


async def test_epsilon_greedy_picks_empirically_best_strategy_when_not_exploring(
    clean_tables: AsyncEngine, new_user_id: str
):
    sessionmaker = async_sessionmaker(clean_tables, expire_on_commit=False)
    # epsilon=0 -> never explore, always exploit the best known strategy.
    procedural = ProceduralMemory(Settings(procedural_epsilon=0.0), sessionmaker=sessionmaker)

    async with sessionmaker() as session:
        from mnemos.db.models import User

        session.add(User(id=new_user_id))
        await session.flush()
        session.add(
            ProceduralStrategy(user_id=new_user_id, strategy_name="balanced", uses=10, successes=9)
        )
        session.add(
            ProceduralStrategy(
                user_id=new_user_id, strategy_name="recency_heavy", uses=10, successes=2
            )
        )
        await session.commit()

    for _ in range(5):
        name, _, _ = await procedural.choose_strategy(new_user_id)
        assert name == "balanced"  # 90% success rate beats 20% and an unseen 50% prior
