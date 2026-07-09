from benchmark.scoring import BenchmarkReport, ProbeRecord, gap_bucket, keyword_hit


def test_gap_bucket_boundaries():
    assert gap_bucket(0) == "same_day"
    assert gap_bucket(7) == "7_day"
    assert gap_bucket(8) == "30_day"
    assert gap_bucket(30) == "30_day"


def test_keyword_hit_case_insensitive_substring():
    assert keyword_hit("I use FastAPI for everything.", ["fastapi"])
    assert not keyword_hit("I use Flask for everything.", ["fastapi"])
    assert keyword_hit("Django works fine too", ["fastapi", "django"])


def test_summary_computes_accuracy_delta_and_gap_breakdown():
    report = BenchmarkReport()
    report.add(
        ProbeRecord(
            "q1", 0, "with_memory", retrieval_hit=True, answer_hit=True,
            latency_ms=100, input_tokens=10, output_tokens=5,
        )
    )
    report.add(
        ProbeRecord(
            "q1", 0, "no_memory", retrieval_hit=False, answer_hit=False,
            latency_ms=50, input_tokens=8, output_tokens=4,
        )
    )
    report.add(
        ProbeRecord(
            "q2", 30, "with_memory", retrieval_hit=True, answer_hit=False,
            latency_ms=200, input_tokens=12, output_tokens=6,
        )
    )
    report.add(
        ProbeRecord(
            "q2", 30, "no_memory", retrieval_hit=False, answer_hit=False,
            latency_ms=60, input_tokens=9, output_tokens=4,
        )
    )

    summary = report.summary()

    assert summary["n_questions"] == 2
    assert summary["answer_accuracy"]["with_memory"] == 0.5
    assert summary["answer_accuracy"]["no_memory"] == 0.0
    assert summary["answer_accuracy"]["delta"] == 0.5
    assert summary["retrieval_recall_at_k"]["by_gap"]["same_day"] == 1.0
    assert summary["retrieval_recall_at_k"]["by_gap"]["30_day"] == 1.0
    assert summary["tokens"]["total_input"] == 39
    assert summary["latency_ms"]["p50"] > 0
