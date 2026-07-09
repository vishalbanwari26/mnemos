from mnemos.memory.schemas import MemoryQueryResult

BASE_SYSTEM_PROMPT = (
    "You are a helpful assistant with persistent memory of this user across "
    "sessions. Use the memory context below when it's relevant; don't mention "
    "that you have a memory system unless the user asks about it directly."
)


def build_system_prompt(memory: MemoryQueryResult) -> str:
    parts = [BASE_SYSTEM_PROMPT]

    if memory.facts:
        facts_block = "\n".join(f"- {f.fact.fact}" for f in memory.facts)
        parts.append(f"\nWhat you know about this user:\n{facts_block}")

    if memory.episodes:
        episodes_block = "\n".join(
            f"- ({e.episode.occurred_at:%Y-%m-%d}) {e.episode.role}: {e.episode.content}"
            for e in memory.episodes
        )
        parts.append(f"\nRelevant past conversation:\n{episodes_block}")

    return "\n".join(parts)
