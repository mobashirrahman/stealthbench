"""T09C: latency, throughput and usage summaries (unit)."""

from __future__ import annotations

import pytest

from stealthbench.analysis.performance import (
    chars_per_second,
    response_chars,
    sample_timing,
    summarize_results,
    tokens_per_second,
)
from stealthbench.schemas.results import (
    DeliveryStatus,
    GenerationResult,
    SampleKey,
    StreamingMeasurements,
    Usage,
)

pytestmark = pytest.mark.unit


def _key(task: str, repeat: int = 1) -> SampleKey:
    return SampleKey(campaign_id="c-test", endpoint_id="ep-a", task_id=task, repeat_id=repeat)


def _accepted(
    task: str,
    response: str,
    *,
    output_tokens: int | None,
    first_content: float | None,
    first_answer: float | None,
    total: float | None,
    chunks: int | None,
    repeat: int = 1,
) -> GenerationResult:
    streaming = StreamingMeasurements(
        first_content_seconds=first_content,
        first_answer_seconds=first_answer,
        total_seconds=total,
        chunk_count=chunks,
    )
    return GenerationResult(
        attempt_id=f"att-{task}-{repeat}",
        sample_key=_key(task, repeat),
        attempt_number=1,
        delivery_status=DeliveryStatus.ACCEPTED,
        response=response,
        usage=Usage(
            input_tokens=10,
            output_tokens=output_tokens,
            provider_reported=output_tokens is not None,
        ),
        streaming=streaming,
        finish_status="stop",
    )


def test_first_content_and_first_answer_stay_distinct() -> None:
    timing = sample_timing(
        response="hello",
        usage=Usage(input_tokens=5, output_tokens=100, provider_reported=True),
        streaming=StreamingMeasurements(
            first_content_seconds=0.12,
            first_answer_seconds=0.35,
            total_seconds=1.5,
            chunk_count=7,
        ),
    )
    assert timing.first_content_seconds == pytest.approx(0.12)
    assert timing.first_answer_seconds == pytest.approx(0.35)
    assert timing.first_content_seconds != timing.first_answer_seconds
    assert timing.total_seconds == pytest.approx(1.5)
    assert timing.chunk_count == 7
    assert timing.response_chars == 5
    # Exact synthetic trace: 5 chars / 1.5 s and 100 tokens / 1.5 s.
    assert timing.chars_per_second == pytest.approx(5.0 / 1.5)
    assert timing.tokens_per_second == pytest.approx(100.0 / 1.5)


def test_unicode_length_counts_characters_not_bytes() -> None:
    text = "héllo 😀 世界"
    assert response_chars(text) == 10
    assert len(text.encode("utf-8")) > 10
    assert chars_per_second(text, 2.0) == pytest.approx(10.0 / 2.0)
    assert response_chars(None) is None
    assert chars_per_second(None, 2.0) is None


def test_absent_usage_stays_null_never_zero() -> None:
    missing = Usage(provider_reported=False)
    assert tokens_per_second(missing, 2.0) is None
    timing = sample_timing(
        response="abc",
        usage=missing,
        streaming=StreamingMeasurements(
            first_content_seconds=0.1,
            first_answer_seconds=0.2,
            total_seconds=2.0,
            chunk_count=4,
        ),
    )
    assert timing.tokens_per_second is None
    assert timing.output_tokens is None
    # Characters still measure; only the provider-token rate is absent.
    assert timing.chars_per_second == pytest.approx(3.0 / 2.0)
    # Without a measured duration neither rate exists.
    assert chars_per_second("abc", None) is None
    assert (
        tokens_per_second(Usage(input_tokens=1, output_tokens=9, provider_reported=True), None)
        is None
    )
    assert chars_per_second("abc", 0.0) is None


def test_chunks_are_not_tokens() -> None:
    timing = sample_timing(
        response="x" * 20,
        usage=Usage(input_tokens=5, output_tokens=10, provider_reported=True),
        streaming=StreamingMeasurements(
            first_content_seconds=0.05,
            first_answer_seconds=0.4,
            total_seconds=2.0,
            chunk_count=100,
        ),
    )
    # 10 provider tokens / 2 s = 5/s. Using the 100 chunks would give 50/s.
    assert timing.tokens_per_second == pytest.approx(5.0)
    assert timing.chunk_count == 100
    # A usage-only absence with many chunks still yields no token rate.
    no_usage = sample_timing(
        response="x" * 20,
        usage=Usage(provider_reported=False),
        streaming=StreamingMeasurements(
            first_content_seconds=0.05,
            first_answer_seconds=0.4,
            total_seconds=2.0,
            chunk_count=50,
        ),
    )
    assert no_usage.tokens_per_second is None


def test_missing_streaming_has_no_timings_or_rates() -> None:
    timing = sample_timing(
        response="answer",
        usage=Usage(input_tokens=3, output_tokens=7, provider_reported=True),
        streaming=None,
    )
    assert timing.first_content_seconds is None
    assert timing.first_answer_seconds is None
    assert timing.total_seconds is None
    assert timing.chunk_count is None
    assert timing.response_chars == 6
    assert timing.chars_per_second is None
    assert timing.tokens_per_second is None


def test_exact_synthetic_trace_summary() -> None:
    results = [
        _accepted(
            "t1",
            "aa",
            output_tokens=10,
            first_content=0.1,
            first_answer=0.2,
            total=1.0,
            chunks=3,
        ),
        _accepted(
            "t2",
            "bbbb",
            output_tokens=20,
            first_content=0.2,
            first_answer=0.5,
            total=2.0,
            chunks=5,
        ),
        _accepted(
            "t3",
            "cccccc",
            output_tokens=30,
            first_content=0.3,
            first_answer=0.6,
            total=3.0,
            chunks=7,
        ),
    ]
    summary = summarize_results(results)
    assert summary.n_total == 3
    assert summary.n_accepted == 3
    assert summary.failure_rate == pytest.approx(0.0)
    assert summary.median_total_seconds == pytest.approx(2.0)
    assert summary.median_first_content_seconds == pytest.approx(0.2)
    assert summary.median_first_answer_seconds == pytest.approx(0.5)
    # Per-sample chars/s: 2/1=2, 4/2=2, 6/3=2 -> median 2.
    assert summary.median_chars_per_second == pytest.approx(2.0)
    # Per-sample tokens/s: 10/1=10, 20/2=10, 30/3=10 -> median 10.
    assert summary.median_tokens_per_second == pytest.approx(10.0)


def test_failure_rate_uses_explicit_attempt_denominator() -> None:
    failed = GenerationResult(
        attempt_id="att-fail",
        sample_key=_key("t9"),
        attempt_number=1,
        delivery_status=DeliveryStatus.TRANSPORT_FAILED,
        response=None,
        usage=Usage(provider_reported=False),
        streaming=None,
        finish_status=None,
    )
    results = [
        _accepted(
            "t1",
            "ok",
            output_tokens=5,
            first_content=0.1,
            first_answer=0.2,
            total=1.0,
            chunks=2,
        ),
        failed,
        failed.model_copy(update={"attempt_id": "att-fail-2", "sample_key": _key("t8")}),
        _accepted(
            "t2",
            "ok!!",
            output_tokens=None,
            first_content=None,
            first_answer=None,
            total=None,
            chunks=None,
        ),
    ]
    summary = summarize_results(results)
    assert summary.n_total == 4
    assert summary.n_accepted == 2
    assert summary.n_transport_failed == 2
    assert summary.failure_rate == pytest.approx(2.0 / 4.0)
    # Only one accepted sample carries a measured total duration.
    assert summary.n_with_total_time == 1
    assert summary.n_with_token_rate == 1


def test_empty_results_have_null_rates() -> None:
    summary = summarize_results([])
    assert summary.n_total == 0
    assert summary.failure_rate is None
    assert summary.median_total_seconds is None
    assert summary.median_chars_per_second is None
    assert summary.median_tokens_per_second is None
