"""Compares Postgres+pgvector, Qdrant, and Neo4j on the same seeded dataset:
write latency, search latency, and recall parity (same queries, same
ground-truth keywords, same embeddings). Storage-only — no LLM calls, so it
runs without ANTHROPIC_API_KEY and isolates the backend from extraction/
answer-generation quality, which benchmark/run_benchmark.py covers instead.

    uv run python -m benchmark.compare_backends
    uv run python -m benchmark.compare_backends --backends postgres qdrant
"""

import argparse
import asyncio
import json
import time
import uuid
from pathlib import Path

from benchmark.report import RESULTS_DIR
from benchmark.scoring import keyword_hit
from mnemos.config import get_settings
from mnemos.embeddings.base import EmbeddingClient
from mnemos.embeddings.factory import get_embedding_client
from mnemos.memory.engine import MemoryEngine
from mnemos.storage.factory import get_storage_backend, reset_storage_backend_cache

DATA_DIR = Path(__file__).parent / "data"
ALL_BACKENDS = ["postgres", "qdrant", "neo4j"]


def _load_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    return s[min(int(len(s) * p), len(s) - 1)]


async def run_backend(
    backend_name: str, embedder: EmbeddingClient, conversations: list[dict], questions: list[dict]
) -> dict:
    settings = get_settings().model_copy(update={"storage_backend": backend_name})
    storage = get_storage_backend(settings)
    memory = MemoryEngine(storage, embedder, settings)
    user_id = f"backend-compare-{backend_name}"

    await memory.reset_user(user_id)

    write_latencies_ms: list[float] = []
    for conv in conversations:
        session_id = uuid.uuid5(uuid.NAMESPACE_DNS, f"compare-{backend_name}-{conv['session_id']}")
        for turn in conv["turns"]:
            start = time.perf_counter()
            await memory.remember_episode(user_id, session_id, turn["role"], turn["content"])
            write_latencies_ms.append((time.perf_counter() - start) * 1000)
        # Ground-truth facts written directly (bypassing LLM extraction) —
        # this script measures the storage layer, not extraction quality.
        for fact in conv["facts_embedded"]:
            start = time.perf_counter()
            await memory.remember_fact(user_id, fact)
            write_latencies_ms.append((time.perf_counter() - start) * 1000)

    search_latencies_ms: list[float] = []
    hits = 0
    for q in questions:
        start = time.perf_counter()
        result = await memory.recall(user_id, q["question"])
        search_latencies_ms.append((time.perf_counter() - start) * 1000)
        retrieved_text = " ".join(
            [f.fact.fact for f in result.facts] + [e.episode.content for e in result.episodes]
        )
        if keyword_hit(retrieved_text, q["match_keywords"]):
            hits += 1

    await memory.reset_user(user_id)

    return {
        "backend": backend_name,
        "n_writes": len(write_latencies_ms),
        "write_latency_ms": {
            "p50": round(_percentile(write_latencies_ms, 0.5), 2),
            "p95": round(_percentile(write_latencies_ms, 0.95), 2),
        },
        "n_questions": len(questions),
        "recall_at_k": round(hits / len(questions), 3),
        "search_latency_ms": {
            "p50": round(_percentile(search_latencies_ms, 0.5), 2),
            "p95": round(_percentile(search_latencies_ms, 0.95), 2),
        },
    }


def render_markdown(results: list[dict]) -> str:
    lines = [
        "## Storage backend comparison",
        "",
        "Same 54 writes (36 episodes + 18 facts) and 18 recall queries, same "
        "local sentence-transformer embeddings, run against each backend in turn.",
        "",
        "| Backend | Write p50 / p95 (ms) | Search p50 / p95 (ms) | Recall@K |",
        "|---|---|---|---|",
    ]
    for r in results:
        w, s = r["write_latency_ms"], r["search_latency_ms"]
        lines.append(
            f"| {r['backend']} | {w['p50']} / {w['p95']} | {s['p50']} / {s['p95']} | "
            f"{r['recall_at_k']:.0%} |"
        )
    return "\n".join(lines)


async def main(backends: list[str]) -> None:
    embedder = get_embedding_client()
    conversations = _load_jsonl(DATA_DIR / "conversations_v1.jsonl")
    questions = _load_jsonl(DATA_DIR / "questions_v1.jsonl")

    results = []
    for name in backends:
        print(f"Running {name}...")
        try:
            result = await run_backend(name, embedder, conversations, questions)
        except Exception as exc:  # noqa: BLE001 — report and continue to the next backend
            print(f"  {name} failed: {exc}")
            continue
        results.append(result)
        await reset_storage_backend_cache()

    report = render_markdown(results)
    print("\n" + report)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "backend_comparison.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="mnemos storage backend comparison")
    parser.add_argument("--backends", nargs="+", choices=ALL_BACKENDS, default=ALL_BACKENDS)
    args = parser.parse_args()
    asyncio.run(main(args.backends))
