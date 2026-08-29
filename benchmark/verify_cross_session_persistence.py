"""Proves the actual mechanism behind "mnemos remembers across Claude Code
sessions" — not a simulation within one running process, but two genuinely
independent OS processes that share nothing except the external storage
backend (Postgres/Qdrant/Neo4j). Process 1 writes a memory via mnemos_remember
and exits completely; process 2 starts cold afterward and recalls it via
mnemos_recall. This is exactly what a real session boundary looks like —
each Claude Code session spawns the MCP server as its own fresh subprocess,
so persistence has to survive a full process exit, not just an in-memory
cache, or it wouldn't actually work in practice.

    uv run python -m benchmark.verify_cross_session_persistence

Distinct from eval_claude_code_memory.py, which measures whether recalled
context improves answer *quality* (a percentage). This script answers a
different, binary question: does the memory survive the process boundary at
all — the plumbing check underneath that eval.
"""

import os
import subprocess
import sys
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
USER_ID = f"cross-session-verify-{uuid.uuid4().hex[:8]}"
FACT = "The user is verifying that mnemos persists memory across separate process boundaries."
QUERY = "what is the user verifying right now?"
EXPECTED_KEYWORD = "verifying"

REMEMBER_SCRIPT = f"""
import asyncio
from mnemos.mcp_server import mnemos_remember, _get_engine

async def main():
    print(await mnemos_remember({FACT!r}))
    await _get_engine().aclose()

asyncio.run(main())
"""

RECALL_SCRIPT = f"""
import asyncio
from mnemos.mcp_server import mnemos_recall, _get_engine

async def main():
    print(await mnemos_recall({QUERY!r}))
    engine = _get_engine()
    await engine.reset_user({USER_ID!r})
    await engine.aclose()

asyncio.run(main())
"""


def run_process(script: str) -> str:
    env = {**os.environ, "MNEMOS_MCP_USER_ID": USER_ID, "PYTHONPATH": str(REPO_ROOT / "src")}
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
        cwd=REPO_ROOT,
        timeout=120,
    )
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise RuntimeError(f"subprocess exited with code {result.returncode}")
    return result.stdout


def main() -> None:
    print("Process 1 — a 'session' that learns something, then exits completely...")
    out1 = run_process(REMEMBER_SCRIPT)
    print(f"  {out1.strip().splitlines()[-1]}")

    print("\nProcess 2 — a brand-new, independent 'session', shares no state with process 1...")
    out2 = run_process(RECALL_SCRIPT)
    recalled = out2.strip().splitlines()[-1]
    print(f"  {recalled}")

    if EXPECTED_KEYWORD in out2.lower():
        print("\nPASS — memory written by process 1 was recalled by process 2.")
    else:
        print("\nFAIL — process 2 did not recall the fact written by process 1.")
        sys.exit(1)


if __name__ == "__main__":
    main()
