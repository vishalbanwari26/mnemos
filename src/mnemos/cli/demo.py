"""Terminal REPL against ConversationManager directly, no server needed.

    uv run python -m mnemos.cli.demo [--user USER_ID] [--llm anthropic|mock]
"""

import argparse
import asyncio
import uuid

from mnemos.agent.conversation import ConversationManager
from mnemos.config import get_settings
from mnemos.embeddings.factory import get_embedding_client
from mnemos.llm.factory import get_llm_client
from mnemos.memory.engine import MemoryEngine
from mnemos.storage.factory import get_storage_backend, reset_storage_backend_cache


async def run(user_id: str, llm_provider: str | None) -> None:
    settings = get_settings()
    if llm_provider:
        settings = settings.model_copy(update={"llm_provider": llm_provider})

    embedder = get_embedding_client(settings)
    llm = get_llm_client(settings)
    extraction_llm = get_llm_client(settings, for_extraction=True)
    storage = get_storage_backend(settings)
    memory = MemoryEngine(storage, embedder, settings)
    manager = ConversationManager(memory, llm, extraction_llm, settings)
    session_id = uuid.uuid4()

    print(
        f"mnemos demo — user={user_id!r} session={session_id} "
        f"provider={settings.llm_provider} storage={settings.storage_backend}"
    )
    print("Type a message, or 'exit' to quit. Restart this command to test cross-session recall.\n")

    try:
        while True:
            try:
                user_message = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not user_message:
                continue
            if user_message.lower() in {"exit", "quit"}:
                break

            result = await manager.handle_message(user_id, session_id, user_message)

            print(f"assistant> {result.reply}")
            print(
                f"  (used {result.memory_used} retrieved memories, "
                f"strategy={result.strategy_used})"
            )
            for fact in result.facts_learned:
                print(f"  (learned: {fact.fact})")
            print()
    finally:
        # Releases Qdrant's local-mode directory lock / Neo4j driver, if used.
        await reset_storage_backend_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description="mnemos CLI demo")
    parser.add_argument("--user", default="demo-user", help="user id (default: demo-user)")
    parser.add_argument(
        "--llm", choices=["anthropic", "groq", "mock"], default=None, help="override LLM_PROVIDER"
    )
    args = parser.parse_args()
    asyncio.run(run(args.user, args.llm))


if __name__ == "__main__":
    main()
