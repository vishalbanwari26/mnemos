"""Evaluates mnemos's actual value as a coding assistant's persistent memory,
through the real MCP tool functions (mnemos_remember / mnemos_recall) from
mnemos.mcp_server — not a reimplementation. This is the "before vs after"
number for the Claude Code integration specifically: how often a question
about the user/project gets answered correctly with no memory at all (the
baseline every session starts from) versus with mnemos wired in over MCP.

    uv run python -m benchmark.eval_claude_code_memory --llm anthropic   # reported number
    uv run python -m benchmark.eval_claude_code_memory --llm mock        # harness smoke test only

Reuses benchmark/scoring.py's scoring machinery (same keyword-hit methodology
as the main recall-over-time benchmark) rather than inventing a second one.
"""

import argparse
import asyncio
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path

os.environ.setdefault("MNEMOS_MCP_USER_ID", "claude-code-eval")

from benchmark.scoring import BenchmarkReport, ProbeRecord, keyword_hit
from mnemos.config import get_settings
from mnemos.llm.base import Message
from mnemos.llm.factory import get_llm_client
from mnemos.mcp_server import USER_ID, _get_engine, mnemos_recall, mnemos_remember

RESULTS_DIR = Path(__file__).parent / "results"

# Realistic things a user might tell Claude Code across separate past
# sessions — preferences, project conventions, decisions — the exact
# category of thing worth persisting between sessions rather than losing
# when a session ends.
FACTS: list[tuple[str, str, list[str]]] = [
    (
        "User prefers pytest over unittest for testing.",
        "What testing framework do I prefer?",
        ["pytest"],
    ),
    (
        "This project's dependency manager is uv, not pip or poetry.",
        "What dependency manager does this project use?",
        ["uv"],
    ),
    (
        "User wants all commit messages to explain why, not what changed.",
        "What should commit messages focus on?",
        ["why"],
    ),
    (
        "User's preferred cloud provider for deployment is AWS, specifically ECS and RDS.",
        "Where do I deploy my applications?",
        ["aws"],
    ),
    (
        "User always wants new Python code to include type hints.",
        "Should new Python code include type hints?",
        ["type hint"],
    ),
    (
        "The project's default storage backend is Postgres with pgvector.",
        "What's the default storage backend for this project?",
        ["postgres"],
    ),
    (
        "User prefers small, focused pull requests over large ones.",
        "What size should pull requests be?",
        ["small"],
    ),
    ("User's editor is Neovim with a dark theme.", "What editor does the user use?", ["neovim"]),
    (
        "This project's LLM provider is Groq, using Llama 3.3 70B.",
        "What LLM provider does this project use?",
        ["groq"],
    ),
    (
        "User wants README changes to include real screenshots, not placeholder text.",
        "What should README updates include?",
        ["screenshot"],
    ),
]

BASELINE_SYSTEM = (
    "You are a coding assistant starting a fresh session with no memory of the user or "
    "this project from any past session. Answer the question from what's given alone."
)
MEMORY_SYSTEM_TEMPLATE = (
    "You are a coding assistant with persistent memory of this user and project across "
    "sessions. Here is what you remember that might be relevant to this question:\n"
    "{memories}\n"
    "Use it if relevant; don't mention that you have a memory system unless asked."
)


async def run(llm_provider: str | None) -> None:
    settings = get_settings()
    if llm_provider:
        settings = settings.model_copy(update={"llm_provider": llm_provider})
    if settings.llm_provider == "mock":
        print(
            "WARNING: --llm mock ignores context, so this run's accuracy numbers are not "
            "the reported result — it only proves the harness runs end to end.\n"
        )

    llm = get_llm_client(settings)
    engine = _get_engine()
    await engine.reset_user(USER_ID)

    print(f"Storing {len(FACTS)} things learned about the user/project (mnemos_remember)...")
    for fact, _, _ in FACTS:
        await mnemos_remember(fact)

    report = BenchmarkReport()
    print(f"Asking {len(FACTS)} questions, before vs after mnemos...\n")

    for i, (_, question, keywords) in enumerate(FACTS):
        for condition in ("no_memory", "with_memory"):
            start = time.perf_counter()
            if condition == "with_memory":
                recalled_text = await mnemos_recall(question)
                system = MEMORY_SYSTEM_TEMPLATE.format(memories=recalled_text)
                retrieval_hit = keyword_hit(recalled_text, keywords)
            else:
                system = BASELINE_SYSTEM
                retrieval_hit = False

            response = await llm.complete(
                messages=[Message(role="user", content=question)],
                system=system,
                # Generous headroom: reasoning models (e.g. Groq's gpt-oss)
                # spend completion tokens on hidden reasoning before the
                # final answer, so a tight budget can starve the answer.
                max_tokens=600,
                temperature=0,
            )
            latency_ms = (time.perf_counter() - start) * 1000

            report.add(
                ProbeRecord(
                    question_id=f"q{i}",
                    gap_days=0,
                    condition=condition,
                    retrieval_hit=retrieval_hit,
                    answer_hit=keyword_hit(response.content, keywords),
                    latency_ms=latency_ms,
                    input_tokens=response.input_tokens,
                    output_tokens=response.output_tokens,
                )
            )

    await engine.reset_user(USER_ID)

    summary = report.summary()
    acc = summary["answer_accuracy"]
    lines = [
        f"## mnemos-as-Claude-Code-memory eval "
        f"({settings.llm_provider}, n={summary['n_questions']} questions)",
        "",
        "| Condition | Answered correctly |",
        "|---|---|",
        f"| Before mnemos (no memory, today's baseline) | {acc['no_memory']:.0%} |",
        f"| After mnemos (recall via MCP) | {acc['with_memory']:.0%} |",
        f"| **Delta** | **{acc['delta']:+.0%}** |",
        "",
        f"Retrieval hit rate (relevant fact actually recalled): "
        f"{summary['retrieval_recall_at_k']['overall']:.0%}",
        f"Latency (with-memory turns): p50 {summary['latency_ms']['p50']:.0f}ms, "
        f"p95 {summary['latency_ms']['p95']:.0f}ms",
        f"Tokens: {summary['tokens']['total_input']} in / {summary['tokens']['total_output']} out, "
        f"est. cost ${summary['tokens']['estimated_cost_usd']:.4f}",
    ]
    print("\n".join(lines))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"claude_code_memory_eval_{datetime.now(UTC):%Y%m%dT%H%M%SZ}.json"
    out_path.write_text(json.dumps({"llm_provider": settings.llm_provider, **summary}, indent=2))
    print(f"\nSaved: {out_path}")

    await engine.aclose()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate mnemos as Claude Code's memory over MCP")
    parser.add_argument(
        "--llm", choices=["anthropic", "groq", "mock"], default=None, help="override LLM_PROVIDER"
    )
    args = parser.parse_args()
    asyncio.run(run(args.llm))


if __name__ == "__main__":
    main()
