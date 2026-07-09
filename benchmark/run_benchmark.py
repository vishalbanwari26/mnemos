"""The credibility benchmark: seeds synthetic conversations across simulated
days, then asks probe questions about them later, comparing a with-memory
condition against a no-memory (stateless) baseline.

    uv run python -m benchmark.run_benchmark --llm anthropic   # reported number
    uv run python -m benchmark.run_benchmark --llm mock        # fast dev smoke test only

`--llm mock` proves the harness plumbing works end to end (seeding, simulated
time, scoring, report generation); the mock LLM ignores its context, so the
accuracy numbers from that run are meaningless and must never be reported as
the benchmark result.
"""

import argparse
import asyncio
import json
import time
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

from benchmark.report import write_report
from benchmark.scoring import BenchmarkReport, ProbeRecord, keyword_hit
from mnemos.agent.conversation import ConversationManager
from mnemos.agent.prompt_builder import BASE_SYSTEM_PROMPT, build_system_prompt
from mnemos.config import get_settings
from mnemos.embeddings.factory import get_embedding_client
from mnemos.llm.base import Message
from mnemos.llm.factory import get_llm_client
from mnemos.memory.engine import MemoryEngine
from mnemos.storage.factory import get_storage_backend, reset_storage_backend_cache

DATA_DIR = Path(__file__).parent / "data"


def _session_uuid(session_key: str) -> uuid.UUID:
    return uuid.uuid5(uuid.NAMESPACE_DNS, f"mnemos-benchmark-{session_key}")


def _load_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


async def run(llm_provider: str | None, storage_backend: str | None) -> None:
    settings = get_settings()
    overrides = {}
    if llm_provider:
        overrides["llm_provider"] = llm_provider
    if storage_backend:
        overrides["storage_backend"] = storage_backend
    if overrides:
        settings = settings.model_copy(update=overrides)

    if settings.llm_provider == "mock":
        print(
            "WARNING: --llm mock is a fast dev smoke test only. The mock LLM "
            "ignores its context, so this run's accuracy numbers are not the "
            "reported benchmark result. Use --llm anthropic or --llm groq for that.\n"
        )

    embedder = get_embedding_client(settings)
    llm = get_llm_client(settings)
    extraction_llm = get_llm_client(settings, for_extraction=True)
    storage = get_storage_backend(settings)
    memory = MemoryEngine(storage, embedder, settings)
    manager = ConversationManager(memory, llm, extraction_llm, settings)

    conversations = _load_jsonl(DATA_DIR / "conversations_v1.jsonl")
    questions = _load_jsonl(DATA_DIR / "questions_v1.jsonl")
    user_id = conversations[0]["user_id"]

    # All simulated days are anchored far enough in the past that even a
    # day-30 probe question is still "asked" before the real current time.
    bench_epoch = datetime.now(UTC) - timedelta(days=35)

    await memory.reset_user(user_id)

    print(f"Seeding {len(conversations)} sessions across simulated days...")
    for conv in sorted(conversations, key=lambda c: c["day"]):
        occurred_at = bench_epoch + timedelta(days=conv["day"])
        turns = [(t["role"], t["content"]) for t in conv["turns"]]
        await manager.seed_session(
            user_id, _session_uuid(conv["session_id"]), turns, occurred_at=occurred_at
        )

    report = BenchmarkReport()
    print(f"Asking {len(questions)} probe questions (with-memory + no-memory each)...")
    for q in questions:
        asked_at = bench_epoch + timedelta(days=q["day"])

        for condition in ("with_memory", "no_memory"):
            if condition == "with_memory":
                recalled = await memory.recall(user_id, q["question"], now=asked_at)
                system_prompt = build_system_prompt(recalled)
                retrieved_text = " ".join(
                    [f.fact.fact for f in recalled.facts]
                    + [e.episode.content for e in recalled.episodes]
                )
                retrieval_hit = keyword_hit(retrieved_text, q["match_keywords"])
            else:
                system_prompt = BASE_SYSTEM_PROMPT
                retrieval_hit = False

            start = time.perf_counter()
            response = await llm.complete(
                messages=[Message(role="user", content=q["question"])],
                system=system_prompt,
                max_tokens=200,
                temperature=0,
            )
            latency_ms = (time.perf_counter() - start) * 1000

            report.add(
                ProbeRecord(
                    question_id=q["id"],
                    gap_days=q["gap_days"],
                    condition=condition,
                    retrieval_hit=retrieval_hit,
                    answer_hit=keyword_hit(response.content, q["match_keywords"]),
                    latency_ms=latency_ms,
                    input_tokens=response.input_tokens,
                    output_tokens=response.output_tokens,
                )
            )

    write_report(report.summary(), llm_provider=settings.llm_provider)
    await reset_storage_backend_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description="mnemos benchmark")
    parser.add_argument(
        "--llm", choices=["anthropic", "groq", "mock"], default=None, help="override LLM_PROVIDER"
    )
    parser.add_argument(
        "--storage",
        choices=["postgres", "qdrant", "neo4j"],
        default=None,
        help="override STORAGE_BACKEND",
    )
    args = parser.parse_args()
    asyncio.run(run(args.llm, args.storage))


if __name__ == "__main__":
    main()
