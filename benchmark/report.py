import json
from datetime import UTC, datetime
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"


def write_report(summary: dict, *, llm_provider: str) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    json_path = RESULTS_DIR / f"run_{timestamp}.json"
    json_path.write_text(json.dumps({"llm_provider": llm_provider, **summary}, indent=2))

    print(render_markdown(summary, llm_provider=llm_provider))
    return json_path


def render_markdown(summary: dict, *, llm_provider: str) -> str:
    acc = summary["answer_accuracy"]
    recall = summary["retrieval_recall_at_k"]
    by_gap = summary["answer_accuracy_by_gap"]
    lat = summary["latency_ms"]
    tok = summary["tokens"]

    lines = [
        f"## mnemos benchmark ({llm_provider}, n={summary['n_questions']} questions)",
        "",
        "| Metric | With memory | No memory | Delta |",
        "|---|---|---|---|",
        f"| Answer accuracy | {acc['with_memory']:.0%} | {acc['no_memory']:.0%} | "
        f"{acc['delta']:+.0%} |",
        "",
        "### Answer accuracy by memory age",
        "",
        "| Gap | With memory | No memory |",
        "|---|---|---|",
    ]
    for bucket, label in [("same_day", "Same day"), ("7_day", "7 days"), ("30_day", "30 days")]:
        row = by_gap[bucket]
        lines.append(f"| {label} | {row['with_memory']:.0%} | {row['no_memory']:.0%} |")

    lines += [
        "",
        f"### Retrieval recall@K: {recall['overall']:.0%} overall "
        f"(same-day {recall['by_gap']['same_day']:.0%}, "
        f"7-day {recall['by_gap']['7_day']:.0%}, "
        f"30-day {recall['by_gap']['30_day']:.0%})",
        "",
        f"### Latency (with-memory turns): p50 {lat['p50']:.0f}ms, p95 {lat['p95']:.0f}ms",
        "",
        f"### Tokens: {tok['total_input']} in / {tok['total_output']} out, "
        f"est. cost ${tok['estimated_cost_usd']:.4f}",
    ]
    return "\n".join(lines)
