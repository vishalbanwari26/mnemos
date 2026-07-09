from dataclasses import dataclass, field

# Rough public per-token pricing for the models this benchmark uses by default,
# purely for an order-of-magnitude cost estimate in the report — not billing-accurate.
PRICE_PER_MTOK_USD = {
    "input": 3.0,
    "output": 15.0,
}


def gap_bucket(gap_days: int) -> str:
    if gap_days <= 0:
        return "same_day"
    if gap_days <= 7:
        return "7_day"
    return "30_day"


def keyword_hit(text: str, keywords: list[str]) -> bool:
    lowered = text.lower()
    return any(k.lower() in lowered for k in keywords)


@dataclass
class ProbeRecord:
    question_id: str
    gap_days: int
    condition: str  # "with_memory" | "no_memory"
    retrieval_hit: bool
    answer_hit: bool
    latency_ms: float
    input_tokens: int
    output_tokens: int

    @property
    def gap_bucket(self) -> str:
        return gap_bucket(self.gap_days)


@dataclass
class BenchmarkReport:
    records: list[ProbeRecord] = field(default_factory=list)

    def add(self, record: ProbeRecord) -> None:
        self.records.append(record)

    def _subset(self, condition: str | None = None, bucket: str | None = None) -> list[ProbeRecord]:
        return [
            r
            for r in self.records
            if (condition is None or r.condition == condition)
            and (bucket is None or r.gap_bucket == bucket)
        ]

    @staticmethod
    def _rate(records: list[ProbeRecord], field_name: str) -> float:
        if not records:
            return 0.0
        return sum(1 for r in records if getattr(r, field_name)) / len(records)

    @staticmethod
    def _percentile(values: list[float], p: float) -> float:
        if not values:
            return 0.0
        s = sorted(values)
        idx = min(int(len(s) * p), len(s) - 1)
        return s[idx]

    def summary(self) -> dict:
        with_mem = self._subset(condition="with_memory")
        no_mem = self._subset(condition="no_memory")
        buckets = ["same_day", "7_day", "30_day"]

        latencies = [r.latency_ms for r in with_mem]
        total_input = sum(r.input_tokens for r in self.records)
        total_output = sum(r.output_tokens for r in self.records)
        est_cost_usd = (
            total_input / 1_000_000 * PRICE_PER_MTOK_USD["input"]
            + total_output / 1_000_000 * PRICE_PER_MTOK_USD["output"]
        )

        return {
            "n_questions": len({r.question_id for r in self.records}),
            "answer_accuracy": {
                "with_memory": self._rate(with_mem, "answer_hit"),
                "no_memory": self._rate(no_mem, "answer_hit"),
                "delta": self._rate(with_mem, "answer_hit") - self._rate(no_mem, "answer_hit"),
            },
            "retrieval_recall_at_k": {
                "overall": self._rate(with_mem, "retrieval_hit"),
                "by_gap": {
                    bucket: self._rate(
                        self._subset(condition="with_memory", bucket=bucket), "retrieval_hit"
                    )
                    for bucket in buckets
                },
            },
            "answer_accuracy_by_gap": {
                bucket: {
                    "with_memory": self._rate(
                        self._subset(condition="with_memory", bucket=bucket), "answer_hit"
                    ),
                    "no_memory": self._rate(
                        self._subset(condition="no_memory", bucket=bucket), "answer_hit"
                    ),
                }
                for bucket in buckets
            },
            "latency_ms": {
                "p50": self._percentile(latencies, 0.5),
                "p95": self._percentile(latencies, 0.95),
            },
            "tokens": {
                "total_input": total_input,
                "total_output": total_output,
                "estimated_cost_usd": round(est_cost_usd, 4),
            },
        }
