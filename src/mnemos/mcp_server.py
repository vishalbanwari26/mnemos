"""Exposes mnemos as an MCP server, so it can act as persistent memory for a
coding assistant (Claude Code) across sessions, instead of (or alongside) the
assistant's own built-in memory mechanism.

Run directly:      uv run python -m mnemos.mcp_server
Register in Claude Code:
    claude mcp add mnemos -- /path/to/mnemos/.venv/bin/python3 -m mnemos.mcp_server

All memories are stored under a single fixed user_id (MNEMOS_MCP_USER_ID env
var, default "claude-code") — this is a single-user personal memory store,
not a multi-tenant one. Respects STORAGE_BACKEND / .env exactly like the CLI
and API do, so `mnemos_recall` sees the same data the dashboard shows.

Memories written here go straight to semantic memory via remember_fact() —
no episodic layer, no LLM-based extraction. The calling assistant (not
mnemos) is the one deciding what's worth remembering, so re-running
extraction over its own summary would be redundant.
"""

import os
import uuid

from mcp.server.mcpserver import MCPServer

from mnemos.config import Settings, get_settings
from mnemos.embeddings.factory import get_embedding_client
from mnemos.memory.engine import MemoryEngine

USER_ID = os.environ.get("MNEMOS_MCP_USER_ID", "claude-code")

_settings: Settings = get_settings()
_engine: MemoryEngine | None = None


def _get_engine() -> MemoryEngine:
    """Built lazily on first tool call, not at import time — loading the
    local embedding model takes a few seconds, and we don't want that on the
    MCP handshake path before the server can even respond to list_tools."""
    global _engine
    if _engine is None:
        from mnemos.storage.factory import get_storage_backend

        storage = get_storage_backend(_settings)
        embedder = get_embedding_client(_settings)
        _engine = MemoryEngine(storage, embedder, _settings)
    return _engine


server = MCPServer(
    name="mnemos",
    instructions=(
        "Persistent memory for this user across coding sessions, backed by mnemos "
        "(semantic memory with adaptive retrieval and reflection-based forgetting — "
        "see https://github.com/vishalbanwari26/mnemos). Call mnemos_recall near the "
        "start of a task to check for relevant prior context (preferences, past "
        "decisions, project conventions). Call mnemos_remember when you learn something "
        "durable that would help in a *future* session — not routine task details, and "
        "not anything already recorded in the repo's own CLAUDE.md or docs."
    ),
)


@server.tool()
async def mnemos_remember(fact: str) -> str:
    """Store a durable fact worth recalling in a future session — a stated
    preference, a decision and its reasoning, a recurring convention. Not
    routine task details, not anything already in the codebase/CLAUDE.md.
    """
    engine = _get_engine()
    written = await engine.remember_fact(USER_ID, fact)
    return f"Remembered [{written.id}]: {written.fact}"


@server.tool()
async def mnemos_recall(query: str, top_k: int = 5) -> str:
    """Retrieve memories relevant to a query — call this near the start of a
    task that might benefit from prior context about the user or their
    preferences/projects."""
    engine = _get_engine()
    result = await engine.recall(USER_ID, query)
    facts = result.facts[:top_k]
    if not facts:
        return "No relevant memories found."
    return "\n".join(f"- {f.fact.fact} (similarity {f.similarity:.2f})" for f in facts)


@server.tool()
async def mnemos_list_memories(limit: int = 50) -> str:
    """List all currently active memories with their IDs, for browsing or as
    input to mnemos_forget."""
    engine = _get_engine()
    facts = await engine.list_facts(USER_ID, limit=limit)
    if not facts:
        return "No memories stored yet."
    return "\n".join(f"- [{f.id}] {f.fact} (confidence {f.confidence:.2f})" for f in facts)


@server.tool()
async def mnemos_forget(fact_id: str, reason: str = "user_requested") -> str:
    """Forget (archive) a specific memory by its ID, as shown by
    mnemos_list_memories. Soft-delete: the fact is archived, not erased, and
    stays visible in mnemos's audit log — never call this speculatively,
    only when the user explicitly asks to forget something."""
    from mnemos.llm.mock_client import MockLLMClient
    from mnemos.memory.reflection import ReflectionEngine

    engine = _get_engine()
    # forget_fact() never calls the LLM (only merge_duplicate_facts does),
    # so a mock client here avoids requiring a real API key for this tool.
    reflection = ReflectionEngine(engine, MockLLMClient(), _settings)
    await reflection.forget_fact(USER_ID, uuid.UUID(fact_id), reason=reason)
    return f"Forgot memory {fact_id} (reason: {reason})"


@server.tool()
async def mnemos_reflect() -> str:
    """Run a consolidation pass: merge near-duplicate memories, decay stale
    ones, archive anything that decays past the forget threshold. Requires a
    real LLM provider configured (ANTHROPIC_API_KEY or GROQ_API_KEY) for the
    merge step. Call this occasionally to keep memory tidy, not on every
    turn."""
    from mnemos.llm.factory import get_llm_client
    from mnemos.memory.reflection import ReflectionEngine

    if _settings.llm_provider == "mock":
        return (
            "LLM_PROVIDER is set to mock — set ANTHROPIC_API_KEY or GROQ_API_KEY "
            "to run reflection."
        )

    engine = _get_engine()
    llm = get_llm_client(_settings, for_extraction=True)
    reflection = ReflectionEngine(engine, llm, _settings)
    summary = await reflection.run(USER_ID)
    return (
        f"Merged into {summary.facts_merged_into} fact(s), "
        f"decayed {summary.facts_decayed}, forgot {summary.facts_forgotten}."
    )


def main() -> None:
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
