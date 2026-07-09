"""Manually trigger a reflection pass for a user (merge near-duplicate
facts, decay/forget stale ones).

    uv run python -m mnemos.cli.reflect --user demo-user
"""

import argparse
import asyncio

from mnemos.config import get_settings
from mnemos.embeddings.factory import get_embedding_client
from mnemos.llm.factory import get_llm_client
from mnemos.memory.engine import MemoryEngine
from mnemos.memory.reflection import ReflectionEngine
from mnemos.storage.factory import get_storage_backend, reset_storage_backend_cache


async def run(user_id: str) -> None:
    settings = get_settings()
    embedder = get_embedding_client(settings)
    extraction_llm = get_llm_client(settings, for_extraction=True)
    storage = get_storage_backend(settings)
    memory = MemoryEngine(storage, embedder, settings)
    reflection = ReflectionEngine(memory, extraction_llm, settings)

    print(f"Running reflection for {user_id!r}...")
    summary = await reflection.run(user_id)
    print(f"  merged into {summary.facts_merged_into} consolidated fact(s)")
    print(f"  decayed {summary.facts_decayed} fact(s)")
    print(f"  forgot {summary.facts_forgotten} fact(s)")

    await reset_storage_backend_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description="mnemos reflection pass")
    parser.add_argument("--user", required=True, help="user id to run reflection for")
    args = parser.parse_args()
    asyncio.run(run(args.user))


if __name__ == "__main__":
    main()
