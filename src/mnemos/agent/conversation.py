import uuid
from dataclasses import dataclass, field
from datetime import datetime

from mnemos.agent.prompt_builder import build_system_prompt
from mnemos.config import Settings, get_settings
from mnemos.llm.base import LLMClient, LLMResponse, Message
from mnemos.memory.engine import MemoryEngine
from mnemos.memory.extraction import extract_semantic_facts
from mnemos.memory.procedural import ProceduralMemory
from mnemos.memory.schemas import SemanticFactRead


@dataclass
class TurnResult:
    reply: str
    memory_used: int
    llm_response: LLMResponse
    facts_learned: list[SemanticFactRead] = field(default_factory=list)
    strategy_used: str = ""


class ConversationManager:
    """Retrieve -> build context -> call the LLM -> store the new episode ->
    (periodically) extract semantic facts from the turn. The single per-turn
    entry point for both the CLI and the API.
    """

    def __init__(
        self,
        memory: MemoryEngine,
        llm: LLMClient,
        extraction_llm: LLMClient | None = None,
        settings: Settings | None = None,
        procedural: ProceduralMemory | None = None,
    ):
        self.memory = memory
        self.llm = llm
        self.extraction_llm = extraction_llm or llm
        self.settings = settings or get_settings()
        self.procedural = procedural or ProceduralMemory(self.settings)

    async def handle_message(
        self, user_id: str, session_id: uuid.UUID, user_message: str
    ) -> TurnResult:
        # Score the *previous* turn's strategy now that the user's follow-up
        # (this message) is available, before choosing this turn's strategy.
        await self.procedural.resolve_pending_turn(session_id, user_message)
        strategy_name, sim_weight, recency_weight = await self.procedural.choose_strategy(user_id)

        recalled = await self.memory.recall(
            user_id, user_message, similarity_weight=sim_weight, recency_weight=recency_weight
        )
        system_prompt = build_system_prompt(recalled)

        response = await self.llm.complete(
            messages=[Message(role="user", content=user_message)],
            system=system_prompt,
        )

        user_episode = await self.memory.remember_episode(user_id, session_id, "user", user_message)
        assistant_episode = await self.memory.remember_episode(
            user_id, session_id, "assistant", response.content
        )
        await self.procedural.record_pending_turn(session_id, user_id, strategy_name)

        facts_learned = await self._maybe_extract(
            user_id,
            session_id,
            user_message,
            response.content,
            user_episode.id,
            assistant_episode.id,
        )

        return TurnResult(
            reply=response.content,
            memory_used=len(recalled.episodes) + len(recalled.facts),
            llm_response=response,
            facts_learned=facts_learned,
            strategy_used=strategy_name,
        )

    async def seed_session(
        self,
        user_id: str,
        session_id: uuid.UUID,
        turns: list[tuple[str, str]],
        *,
        occurred_at: datetime | None = None,
    ) -> list[SemanticFactRead]:
        """Write a pre-scripted (role, content) conversation directly, without
        calling the chat LLM, then run one extraction pass over the whole
        session. Used by the admin `/seed` endpoint and the benchmark harness
        to seed synthetic history at a specific point in simulated time.
        """
        episode_ids: list[str] = []
        for role, content in turns:
            episode = await self.memory.remember_episode(
                user_id, session_id, role, content, occurred_at=occurred_at
            )
            episode_ids.append(str(episode.id))

        turn_text = "\n".join(f"{role}: {content}" for role, content in turns)
        return await extract_semantic_facts(
            self.memory,
            self.extraction_llm,
            user_id,
            turn_text,
            source_episode_ids=episode_ids,
            settings=self.settings,
        )

    async def _maybe_extract(
        self,
        user_id: str,
        session_id: uuid.UUID,
        user_message: str,
        assistant_message: str,
        user_episode_id: uuid.UUID,
        assistant_episode_id: uuid.UUID,
    ) -> list[SemanticFactRead]:
        n = max(self.settings.extraction_every_n_turns, 1)
        if n > 1:
            turns_so_far = await self.memory.storage.get_episodes(user_id, session_id=session_id)
            assistant_turns = sum(1 for e in turns_so_far if e.role == "assistant")
            if assistant_turns % n != 0:
                return []

        turn_text = f"user: {user_message}\nassistant: {assistant_message}"
        return await extract_semantic_facts(
            self.memory,
            self.extraction_llm,
            user_id,
            turn_text,
            source_episode_ids=[str(user_episode_id), str(assistant_episode_id)],
            settings=self.settings,
        )
