"""Endpoint latency, throughput and usage summaries (G09 T09C).

Measurement contract (``BENCHMARK_PLAN.md`` section "Endpoint performance"):

* First streamed content and first visible answer are distinct measurements.
  Reasoning may precede any visible answer, so collapsing them would hide the
  behaviour a performance report exists to show.
* Throughput is reported twice, for different purposes: characters per second
  (a common reference-token rate for speed comparisons) and provider output
  tokens per second (billing accounting). Both need a measured total
  duration; without one the rate is ``None``, never ``0``.
* An SSE chunk is not a token. ``chunk_count`` is reported as transport shape
  only and never enters a token rate.
* Response length is Unicode characters (``len(response)``), not UTF-8 bytes.
* Absent usage (``output_tokens is None``) stays ``None``. It is never
  replaced by ``0`` or by the chunk count.

Offline by construction: pure functions over stored results. No transport,
no socket, no subprocess, no credential.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from stealthbench.schemas.results import (
    DeliveryStatus,
    GenerationResult,
    StreamingMeasurements,
    Usage,
)

#: Throughput uses characters per second as the common cross-endpoint rate.
RATE_UNIT_CHARS_PER_SECOND: Final[str] = "chars_per_second"
RATE_UNIT_TOKENS_PER_SECOND: Final[str] = "provider_output_tokens_per_second"


@dataclass(frozen=True, slots=True)
class SampleTiming:
    """Per-sample timing and rate decomposition.

    ``chars_per_second`` compares speed across endpoints; ``tokens_per_second``
    uses provider token accounting for billing. ``chunk_count`` is transport
    shape and enters neither rate.
    """

    first_content_seconds: float | None
    first_answer_seconds: float | None
    total_seconds: float | None
    chunk_count: int | None
    response_chars: int | None
    output_tokens: int | None
    chars_per_second: float | None
    tokens_per_second: float | None


@dataclass(frozen=True, slots=True)
class EndpointPerformance:
    """Aggregate performance over one endpoint's delivery attempts."""

    n_total: int
    n_accepted: int
    n_transport_failed: int
    failure_rate: float | None
    n_with_total_time: int
    n_with_chars_rate: int
    n_with_token_rate: int
    median_first_content_seconds: float | None
    median_first_answer_seconds: float | None
    median_total_seconds: float | None
    median_chars_per_second: float | None
    median_tokens_per_second: float | None


def response_chars(response: str | None) -> int | None:
    """Unicode character length of a response, or ``None`` when absent.

    ``len`` counts code points, so emoji and multilingual text measure as the
    characters a reader sees, not as UTF-8 bytes.
    """
    if response is None:
        return None
    return len(response)


def chars_per_second(response: str | None, total_seconds: float | None) -> float | None:
    """Normalised output rate for cross-endpoint speed comparison.

    Returns ``None`` when the response or the measured duration is absent (or
    the duration is not positive). Never ``0`` for missing data.
    """
    chars = response_chars(response)
    if chars is None or total_seconds is None:
        return None
    if total_seconds <= 0:
        return None
    return chars / total_seconds


def tokens_per_second(usage: Usage, total_seconds: float | None) -> float | None:
    """Provider-token output rate for billing accounting.

    Requires both a reported ``output_tokens`` count and a measured duration.
    The chunk count is deliberately not an input: one SSE chunk is not one
    token. Absent usage stays ``None``.
    """
    if usage.output_tokens is None or total_seconds is None:
        return None
    if total_seconds <= 0:
        return None
    return usage.output_tokens / total_seconds


def sample_timing(
    *,
    response: str | None,
    usage: Usage,
    streaming: StreamingMeasurements | None,
) -> SampleTiming:
    """Decompose one sample into timings, lengths and the two output rates.

    First content and first answer are preserved separately; a reasoning-only
    prefix makes them differ. Exact inputs yield exact outputs: no rounding,
    no imputation.
    """
    first_content = streaming.first_content_seconds if streaming is not None else None
    first_answer = streaming.first_answer_seconds if streaming is not None else None
    total = streaming.total_seconds if streaming is not None else None
    chunks = streaming.chunk_count if streaming is not None else None
    chars = response_chars(response)
    return SampleTiming(
        first_content_seconds=first_content,
        first_answer_seconds=first_answer,
        total_seconds=total,
        chunk_count=chunks,
        response_chars=chars,
        output_tokens=usage.output_tokens,
        chars_per_second=chars_per_second(response, total),
        tokens_per_second=tokens_per_second(usage, total),
    )


def timing_of(result: GenerationResult) -> SampleTiming:
    """Timing decomposition of one stored generation result."""
    return sample_timing(response=result.response, usage=result.usage, streaming=result.streaming)


def _median(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return statistics.median(values)


def summarize_results(results: Sequence[GenerationResult]) -> EndpointPerformance:
    """Aggregate attempts into failure rate, latency medians and rate medians.

    The failure denominator is explicit: ``n_total`` attempts. Medians run
    only over samples where the underlying measurement exists; a missing
    duration or missing usage shrinks that median's sample, never contributes
    a zero.
    """
    total = len(results)
    accepted = sum(1 for result in results if result.is_accepted_sample)
    failed = sum(
        1 for result in results if result.delivery_status is DeliveryStatus.TRANSPORT_FAILED
    )
    timings = [timing_of(result) for result in results if result.is_accepted_sample]
    first_contents = [
        t.first_content_seconds for t in timings if t.first_content_seconds is not None
    ]
    first_answers = [t.first_answer_seconds for t in timings if t.first_answer_seconds is not None]
    totals = [t.total_seconds for t in timings if t.total_seconds is not None]
    chars_rates = [t.chars_per_second for t in timings if t.chars_per_second is not None]
    token_rates = [t.tokens_per_second for t in timings if t.tokens_per_second is not None]
    return EndpointPerformance(
        n_total=total,
        n_accepted=accepted,
        n_transport_failed=failed,
        failure_rate=((total - accepted) / total) if total else None,
        n_with_total_time=len(totals),
        n_with_chars_rate=len(chars_rates),
        n_with_token_rate=len(token_rates),
        median_first_content_seconds=_median(first_contents),
        median_first_answer_seconds=_median(first_answers),
        median_total_seconds=_median(totals),
        median_chars_per_second=_median(chars_rates),
        median_tokens_per_second=_median(token_rates),
    )


__all__ = [
    "RATE_UNIT_CHARS_PER_SECOND",
    "RATE_UNIT_TOKENS_PER_SECOND",
    "EndpointPerformance",
    "SampleTiming",
    "chars_per_second",
    "response_chars",
    "sample_timing",
    "summarize_results",
    "timing_of",
    "tokens_per_second",
]
