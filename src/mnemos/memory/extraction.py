from mnemos.config import Settings, get_settings
from mnemos.llm.base import LLMClient, Message
from mnemos.memory.engine import MemoryEngine
from mnemos.memory.schemas import SemanticFactRead

RECORD_FACTS_TOOL = {
    "name": "record_facts",
    "description": (
        "Record durable facts learned about the user from this conversation turn "
        "(preferences, tools, projects, decisions). Skip anything trivial, "
        "transient, or not actually stated by the user."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "facts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "statement": {
                            "type": "string",
                            "description": "A short, self-contained statement about the user.",
                        },
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    },
                    "required": ["statement"],
                },
            }
        },
        "required": ["facts"],
    },
}

EXTRACTION_SYSTEM_PROMPT = (
    "You extract durable facts about a user from a single conversation turn. "
    "Only record facts the user actually stated or clearly implied. If there is "
    "nothing worth remembering, call the tool with an empty facts list."
)


async def extract_semantic_facts(
    memory: MemoryEngine,
    llm: LLMClient,
    user_id: str,
    turn_text: str,
    *,
    source_episode_ids: list[str] | None = None,
    settings: Settings | None = None,
) -> list[SemanticFactRead]:
    """Distill a conversation turn into semantic facts and store the ones that
    aren't near-duplicates of what's already known about this user.
    """
    settings = settings or get_settings()

    response = await llm.complete(
        messages=[Message(role="user", content=turn_text)],
        system=EXTRACTION_SYSTEM_PROMPT,
        tools=[RECORD_FACTS_TOOL],
        tool_choice={"type": "tool", "name": "record_facts"},
        max_tokens=512,
        temperature=0,
    )

    if not response.tool_calls:
        return []

    candidates = response.tool_calls[0].input.get("facts", [])
    written: list[SemanticFactRead] = []

    for candidate in candidates:
        statement = candidate.get("statement", "").strip()
        if not statement:
            continue
        confidence = float(candidate.get("confidence", 1.0))

        if await _is_duplicate(memory, user_id, statement, settings):
            continue

        fact = await memory.remember_fact(
            user_id,
            statement,
            source_episode_ids=source_episode_ids or [],
            confidence=confidence,
        )
        written.append(fact)

    return written


async def _is_duplicate(
    memory: MemoryEngine, user_id: str, statement: str, settings: Settings
) -> bool:
    embedding = memory.embeddings.embed_one(statement)
    nearest = await memory.storage.search_facts(user_id, embedding, top_k=1)
    if not nearest:
        return False
    _fact, distance = nearest[0]
    similarity = 1.0 - distance
    return similarity > settings.semantic_dedup_threshold
