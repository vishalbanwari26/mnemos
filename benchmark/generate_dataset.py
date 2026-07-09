"""Deterministically generates the synthetic benchmark dataset.

Run with `uv run python -m benchmark.generate_dataset` to regenerate
benchmark/data/conversations_v1.jsonl and questions_v1.jsonl from the fact
bank below. The dataset is committed, so this only needs re-running if the
fact bank changes.

Each fact is mentioned exactly once, in one session, on a specific simulated
day. Its probe question is asked `gap_days` later, in one of three buckets
(same-day / 7-day / 30-day), so the benchmark report can show whether recall
degrades as memories age — that's the entire point of the benchmark.
"""

import json
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"

# (session_day, turns, facts_embedded, question, target_facts, match_keywords, gap_days)
FACT_BANK: list[dict] = [
    dict(
        session_day=0,
        turns=[
            ("user", "I've switched to FastAPI over Flask for my backend work."),
            ("assistant", "Nice, the async support and auto docs are great."),
        ],
        facts_embedded=["prefers FastAPI over Flask for backend development"],
        question="What backend framework do I prefer?",
        match_keywords=["fastapi"],
        gap_days=30,
    ),
    dict(
        session_day=0,
        turns=[
            ("user", "I run everything on PostgreSQL these days, not MySQL."),
            ("assistant", "Postgres has better JSON support, good call."),
        ],
        facts_embedded=["prefers PostgreSQL over MySQL"],
        question="Which database do I prefer to use?",
        match_keywords=["postgres"],
        gap_days=7,
    ),
    dict(
        session_day=1,
        turns=[
            ("user", "I work remote full time, based in Berlin."),
            ("assistant", "Got it, I'll keep that in mind for scheduling."),
        ],
        facts_embedded=["works remote, based in Berlin"],
        question="Where am I based?",
        match_keywords=["berlin"],
        gap_days=30,
    ),
    dict(
        session_day=1,
        turns=[
            ("user", "My editor of choice is Neovim, I never use VSCode."),
            ("assistant", "Respect. Neovim's a steep learning curve but fast once you're in it."),
        ],
        facts_embedded=["uses Neovim as primary editor, not VSCode"],
        question="What code editor do I use?",
        match_keywords=["neovim", "vim"],
        gap_days=0,
    ),
    dict(
        session_day=2,
        turns=[
            (
                "user",
                "I'm vegetarian, so skip meat-based examples if you ever suggest restaurants.",
            ),
            ("assistant", "Noted, vegetarian options only."),
        ],
        facts_embedded=["is vegetarian"],
        question="Do I eat meat?",
        match_keywords=["vegetarian"],
        gap_days=30,
    ),
    dict(
        session_day=2,
        turns=[
            (
                "user",
                "Rust is my favorite language for side projects, even though I use Python at work.",
            ),
            ("assistant", "Rust for fun, Python for the day job, a common combo."),
        ],
        facts_embedded=["favorite language for side projects is Rust; uses Python at work"],
        question="What's my favorite programming language for side projects?",
        match_keywords=["rust"],
        gap_days=7,
    ),
    dict(
        session_day=3,
        turns=[
            ("user", "We deploy everything on AWS, mostly ECS and RDS."),
            ("assistant", "Standard AWS setup, makes sense for that stack."),
        ],
        facts_embedded=["deploys on AWS using ECS and RDS"],
        question="Which cloud provider do I deploy on?",
        match_keywords=["aws"],
        gap_days=30,
    ),
    dict(
        session_day=3,
        turns=[
            (
                "user",
                "I'm in the CET timezone, mornings before 9am my time don't work for calls.",
            ),
            ("assistant", "I'll factor that in for any scheduling suggestions."),
        ],
        facts_embedded=["is in CET timezone, unavailable for calls before 9am CET"],
        question="What timezone am I in?",
        match_keywords=["cet"],
        gap_days=0,
    ),
    dict(
        session_day=5,
        turns=[
            ("user", "I have a dog named Miso, a small brown mutt."),
            ("assistant", "Miso sounds like a great companion."),
        ],
        facts_embedded=["has a dog named Miso"],
        question="What's my dog's name?",
        match_keywords=["miso"],
        gap_days=30,
    ),
    dict(
        session_day=5,
        turns=[
            ("user", "Climbing is my main hobby outside of work, bouldering mostly."),
            ("assistant", "Bouldering's a great way to unwind."),
        ],
        facts_embedded=["hobby is bouldering/climbing"],
        question="What do I do for fun outside of work?",
        match_keywords=["climb", "boulder"],
        gap_days=7,
    ),
    dict(
        session_day=7,
        turns=[
            ("user", "I only write tests with pytest, never unittest."),
            ("assistant", "pytest's fixtures make that an easy call."),
        ],
        facts_embedded=["uses pytest for testing, not unittest"],
        question="What testing framework do I use?",
        match_keywords=["pytest"],
        gap_days=30,
    ),
    dict(
        session_day=7,
        turns=[
            ("user", "CI is all GitHub Actions on my projects."),
            ("assistant", "GitHub Actions integrates well if you're already on GitHub."),
        ],
        facts_embedded=["uses GitHub Actions for CI"],
        question="What CI tool do I use?",
        match_keywords=["github actions"],
        gap_days=0,
    ),
    dict(
        session_day=9,
        turns=[
            ("user", "Everything ships as Docker containers onto Kubernetes."),
            ("assistant", "Containerized deploys onto k8s, a solid default."),
        ],
        facts_embedded=["deploys via Docker containers on Kubernetes"],
        question="How do I deploy my applications?",
        match_keywords=["docker", "kubernetes"],
        gap_days=30,
    ),
    dict(
        session_day=10,
        turns=[
            ("user", "I keep PRs small on purpose, I hate reviewing 1000-line diffs."),
            ("assistant", "Small PRs make review a lot faster and safer."),
        ],
        facts_embedded=["prefers small, incremental pull requests"],
        question="What's my preference for pull request size?",
        match_keywords=["small"],
        gap_days=7,
    ),
    dict(
        session_day=12,
        turns=[
            ("user", "I'd rather get a Slack message than an email, I barely check email."),
            ("assistant", "Slack it is, I'll assume that's the faster channel."),
        ],
        facts_embedded=["prefers Slack over email for communication"],
        question="How do I prefer to be contacted, Slack or email?",
        match_keywords=["slack"],
        gap_days=0,
    ),
    dict(
        session_day=14,
        turns=[
            ("user", "I drink my coffee black, no sugar, no milk."),
            ("assistant", "Black coffee, noted."),
        ],
        facts_embedded=["drinks coffee black, no sugar or milk"],
        question="How do I take my coffee?",
        match_keywords=["black"],
        gap_days=30,
    ),
    dict(
        session_day=16,
        turns=[
            (
                "user",
                "Mornings are my most productive time, schedule deep-focus work before noon.",
            ),
            ("assistant", "Mornings for deep work, got it."),
        ],
        facts_embedded=["is most productive in the mornings"],
        question="When am I most productive during the day?",
        match_keywords=["morning"],
        gap_days=7,
    ),
    dict(
        session_day=18,
        turns=[
            ("user", "I mostly read sci-fi novels, rarely anything else."),
            ("assistant", "Sci-fi's a good genre for long flights."),
        ],
        facts_embedded=["reading preference is science fiction novels"],
        question="What genre of books do I usually read?",
        match_keywords=["sci-fi", "science fiction"],
        gap_days=0,
    ),
]


def generate() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conversations_path = DATA_DIR / "conversations_v1.jsonl"
    questions_path = DATA_DIR / "questions_v1.jsonl"

    user_id = "bench-user-1"

    with conversations_path.open("w") as conv_f, questions_path.open("w") as q_f:
        for i, entry in enumerate(FACT_BANK):
            session_id = f"s{i}"
            conv_f.write(
                json.dumps(
                    {
                        "day": entry["session_day"],
                        "session_id": session_id,
                        "user_id": user_id,
                        "turns": [{"role": r, "content": c} for r, c in entry["turns"]],
                        "facts_embedded": entry["facts_embedded"],
                    }
                )
                + "\n"
            )
            q_f.write(
                json.dumps(
                    {
                        "id": f"q{i}",
                        "day": entry["session_day"] + entry["gap_days"],
                        "gap_days": entry["gap_days"],
                        "question": entry["question"],
                        "target_facts": entry["facts_embedded"],
                        "match_keywords": entry["match_keywords"],
                    }
                )
                + "\n"
            )

    print(f"Wrote {len(FACT_BANK)} sessions to {conversations_path}")
    print(f"Wrote {len(FACT_BANK)} questions to {questions_path}")


if __name__ == "__main__":
    generate()
