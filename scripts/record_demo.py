"""Record a static copy of the dashboard for the GitHub Pages demo.

Seeds a fictional user ("maya", invented for the demo) with seven conversations
over 24 days, runs reflection, asks a set of questions in a fresh session, and
records every dashboard endpoint plus the retrieval trace for each question.
The static build (VITE_STATIC=1) replays these files in the browser.

Run the API first, with any storage backend and a real LLM, e.g.:

    STORAGE_BACKEND=qdrant QDRANT_LOCAL_PATH=$(mktemp -d) LLM_PROVIDER=groq \\
        uvicorn mnemos.api.main:app --app-dir src --port 8798 &
    python scripts/record_demo.py http://127.0.0.1:8798 dashboard/demo-data
"""

from __future__ import annotations

import json
import sys
import urllib.request
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

USER = "maya"

# Seven sessions over 24 days. Session 6 changes two facts (a move and a new
# test runner), so reflection has something real to merge and forget.
SESSIONS: list[tuple[int, list[tuple[str, str]]]] = [
    (24, [("user", "Hi! I'm Maya, a backend engineer. I just adopted a beagle called Biscuit."),
          ("assistant", "Congratulations on Biscuit! How is the first week going?"),
          ("user", "Chaotic but fun. I live in Lisbon, so we walk along the river every morning."),
          ("assistant", "Morning river walks sound like a great routine for a beagle.")]),
    (21, [("user", "At work I build APIs with FastAPI and PostgreSQL."),
         ("assistant", "A solid stack. Anything you're working on right now?"),
         ("user", "A billing service. I write all my tests with unittest, it's what the team uses."),
         ("assistant", "Got it, unittest for the billing service tests.")]),
    (18, [("user", "I'm vegetarian, by the way, and I cook most evenings."),
         ("assistant", "Nice. Do you have a favourite dish?"),
         ("user", "Anything with chickpeas. I make a chickpea curry at least once a week."),
         ("assistant", "Chickpea curry once a week, noted.")]),
    (14, [("user", "I signed up for the Porto half marathon in March."),
         ("assistant", "Exciting! How is training going?"),
         ("user", "Three runs a week so far, the long one on Sundays."),
         ("assistant", "Long run on Sundays, three runs a week. Good luck!")]),
    (9, [("user", "Code reviews: I like small pull requests and comments that explain why, not what."),
         ("assistant", "Small PRs and why-comments. That makes reviews much easier."),
         ("user", "We deploy everything to Fly.io from GitHub Actions."),
         ("assistant", "Fly.io via GitHub Actions, understood.")]),
    (4, [("user", "Big news: I moved from Lisbon to Porto last weekend, closer to the race."),
         ("assistant", "Congratulations on the move! How is Biscuit settling in?"),
         ("user", "He loves it. Also, the team switched from unittest to pytest this sprint."),
         ("assistant", "So pytest from now on, and you're in Porto now.")]),
    (0, [("user", "Quick one: I'm learning Portuguese, about twenty minutes a day."),
         ("assistant", "Twenty minutes a day adds up fast. Boa sorte!")]),
]

# Asked in a new session after reflection. Each also gets a retrieval trace.
QUESTIONS = [
    "What's my dog's name?",
    "Which test framework should I use for a new service?",
    "Where do I live now?",
    "What could I cook for dinner tonight?",
    "What race am I training for, and when?",
    "How should I set up deployment for a new project?",
]


def file_for(path: str) -> str:
    """/users/maya/memories/semantic?status=active -> memories/semantic__status-active.json"""
    p, _, q = path.split(f"/users/{USER}/", 1)[1].partition("?")
    return p + "".join("__" + kv.replace("=", "-") for kv in q.split("&") if kv) + ".json"


class Recorder:
    def __init__(self, base: str, out: Path) -> None:
        self.base, self.out = base.rstrip("/"), out

    def call(self, method: str, path: str, body: dict | None = None):
        req = urllib.request.Request(self.base + path, method=method,
                                     data=json.dumps(body).encode() if body is not None else None,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as r:
            raw = r.read()
        return json.loads(raw) if raw else None

    def save(self, name: str, data) -> None:
        f = self.out / name
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":"), default=str))

    def get(self, path: str):
        data = self.call("GET", path)
        self.save(file_for(path), data)
        return data


def main(base: str, out_dir: str) -> None:
    rec, u = Recorder(base, Path(out_dir)), f"/users/{USER}"
    rec.call("POST", f"{u}/reset")
    now = datetime.now(timezone.utc)
    for days_ago, turns in SESSIONS:
        rec.call("POST", f"{u}/seed", {"session_id": str(uuid.uuid4()),
                                        "occurred_at": (now - timedelta(days=days_ago)).isoformat(),
                                        "turns": [{"role": r, "content": c} for r, c in turns]})
    reflection = rec.call("POST", f"{u}/reflect")
    rec.save("reflect.json", reflection)

    session, chats, traces = str(uuid.uuid4()), [], []
    for q in QUESTIONS:
        traces.append({"query": q, "result": rec.call("POST", f"{u}/retrieval-trace", {"query": q, "as_of": None})})
        chats.append({"message": q, "response": rec.call("POST", f"{u}/messages", {"message": q, "session_id": session})})
    rec.save("presets.json", {"user": USER, "chats": chats, "traces": traces})

    rec.get(f"{u}/stats")
    rec.get(f"{u}/memories/episodic")
    for status in ("active", "merged", "forgotten"):
        rec.get(f"{u}/memories/semantic?status={status}")
    rec.get(f"{u}/reflection-log")
    rec.get(f"{u}/procedural")
    print(f"recorded demo user '{USER}' into {out_dir}: {len(chats)} chats, reflection {reflection}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
