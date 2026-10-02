"""Server-sent event parsing for streamed completions.

The protocol is deceptively easy to get wrong. A stream is a byte stream, not a list
of records, so every one of these is a real failure mode:

* a record split across two reads, or several records arriving in one read
* a multi-byte character split across the read boundary
* a chunk carrying only reasoning, before any visible answer
* a final chunk carrying only usage, with no choices at all
* a stream that stops mid-record, with no terminating sentinel

The last rule is the one that matters most for scoring: **a chunk is not a token.**
There is no function in this module that turns ``chunk_count`` into a token count, and
there is deliberately no way to derive a token rate from chunks alone.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Final

from stealthbench.adapters.base import (
    AdapterResult,
    FailureKind,
    safe_failure,
)
from stealthbench.adapters.zen import map_finish_reason, usage_from_response
from stealthbench.schemas.results import DeliveryStatus, StreamingMeasurements, Usage

SSE_DATA_PREFIX: Final[str] = "data:"
SSE_DONE_SENTINEL: Final[str] = "[DONE]"


class StreamEventKind(StrEnum):
    """What a parsed stream record turned out to be."""

    DELTA = "delta"
    REASONING_DELTA = "reasoning_delta"
    USAGE = "usage"
    FINISH = "finish"
    DONE = "done"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class StreamEvent:
    """One parsed record from a stream."""

    kind: StreamEventKind
    content_delta: str = ""
    reasoning_delta: str = ""
    finish_reason: str | None = None
    usage: Usage = field(default_factory=Usage)
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def contributes_text(self) -> bool:
        return bool(self.content_delta)


@dataclass(slots=True)
class StreamAssembly:
    """Reassembled result of a stream.

    Holds content and reasoning separately. Reasoning can precede any visible answer,
    so the two timings are genuinely different measurements and collapsing them would
    hide exactly the behaviour a performance report exists to show.
    """

    content: str = ""
    reasoning: str = ""
    finish_reason: str | None = None
    usage: Usage = field(default_factory=Usage)
    chunk_count: int = 0
    first_content_seconds: float | None = None
    first_answer_seconds: float | None = None
    total_seconds: float | None = None
    saw_done: bool = False

    def apply(self, event: StreamEvent, *, at_seconds: float | None) -> None:
        self.chunk_count += 1
        if event.content_delta:
            if self.first_content_seconds is None and at_seconds is not None:
                self.first_content_seconds = at_seconds
            self.content += event.content_delta
        if event.reasoning_delta:
            self.reasoning += event.reasoning_delta
        if event.kind is StreamEventKind.FINISH and event.finish_reason:
            self.finish_reason = event.finish_reason
        if event.kind is StreamEventKind.USAGE:
            self.usage = event.usage
        if event.kind is StreamEventKind.DONE:
            self.saw_done = True

    def measurements(self) -> StreamingMeasurements:
        return StreamingMeasurements(
            first_content_seconds=self.first_content_seconds,
            first_answer_seconds=self.first_answer_seconds,
            total_seconds=self.total_seconds,
            chunk_count=self.chunk_count,
        )

    def token_rate(self) -> float | None:
        """Normalized output rate, or ``None`` when it cannot be known.

        Requires both a reported output-token count and a measured duration. The chunk
        count is deliberately not an input: an SSE chunk is not a token.
        """
        if self.usage.output_tokens is None or not self.total_seconds:
            return None
        return self.usage.output_tokens / self.total_seconds


class SSEParser:
    """Incremental SSE parser that survives arbitrary read boundaries.

    Feed it decoded text of any shape; it yields whole records. Byte-level framing is
    handled by :func:`decode_stream_bytes`, which keeps a partial multi-byte sequence
    buffered rather than corrupting the character at the boundary.
    """

    def __init__(self) -> None:
        self._buffer = ""
        self._saw_done = False

    def feed(self, chunk: str) -> Iterator[StreamEvent]:
        """Consume a chunk of decoded text and yield any complete records."""
        if self._saw_done:
            return
        self._buffer += chunk
        while "\n\n" in self._buffer:
            raw_record, self._buffer = self._buffer.split("\n\n", 1)
            event = self._parse_record(raw_record)
            if event is None:
                continue
            if event.kind is StreamEventKind.DONE:
                # Everything after the sentinel is discarded, even if it arrived in the
                # same read as the sentinel.
                self._saw_done = True
                self._buffer = ""
                yield event
                return
            yield event

    def flush(self) -> StreamEvent | None:
        """Emit a trailing record that never received its blank-line terminator.

        A stream cut mid-record leaves a partial ``data:`` line. Returning it lets the
        caller decide whether it was a complete payload, rather than the parser
        silently inventing or discarding content.
        """
        remainder = self._buffer.strip()
        self._buffer = ""
        if not remainder:
            return None
        return self._parse_record(remainder)

    @property
    def has_pending(self) -> bool:
        return bool(self._buffer.strip())

    def _parse_record(self, raw_record: str) -> StreamEvent | None:
        data_lines: list[str] = []
        for line in raw_record.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith(":"):
                continue
            if stripped.startswith(SSE_DATA_PREFIX):
                data_lines.append(stripped[len(SSE_DATA_PREFIX) :].strip())
        if not data_lines:
            return None
        payload_text = "\n".join(data_lines)
        if payload_text == SSE_DONE_SENTINEL:
            return StreamEvent(kind=StreamEventKind.DONE, raw={"data": payload_text})
        try:
            payload = json.loads(payload_text)
        except json.JSONDecodeError:
            return StreamEvent(kind=StreamEventKind.UNKNOWN, raw={"unparsable": payload_text})
        if not isinstance(payload, dict):
            return StreamEvent(kind=StreamEventKind.UNKNOWN, raw={"value": payload})
        return classify_chunk(payload)


def classify_chunk(payload: dict[str, Any]) -> StreamEvent:
    """Classify one streamed chunk.

    A usage-only final chunk has no choices at all, and a reasoning-only chunk has
    content absent. Both are distinguished here so neither is mistaken for an answer.
    """
    usage_raw = payload.get("usage")
    choices = payload.get("choices")
    choice: Any = None
    if isinstance(choices, Sequence) and choices:
        first = choices[0]
        choice = first if isinstance(first, dict) else None

    content_delta = ""
    reasoning_delta = ""
    finish_reason: str | None = None
    if choice is not None:
        delta = choice.get("delta")
        if isinstance(delta, dict):
            content = delta.get("content")
            if isinstance(content, str):
                content_delta = content
            for name in ("reasoning_content", "reasoning"):
                reasoning = delta.get(name)
                if isinstance(reasoning, str) and reasoning:
                    reasoning_delta = reasoning
                    break
        finish = choice.get("finish_reason")
        if isinstance(finish, str):
            finish_reason = finish

    has_usage = isinstance(usage_raw, dict)
    if content_delta:
        return StreamEvent(
            kind=StreamEventKind.DELTA,
            content_delta=content_delta,
            raw=payload,
        )
    if reasoning_delta:
        return StreamEvent(
            kind=StreamEventKind.REASONING_DELTA,
            reasoning_delta=reasoning_delta,
            raw=payload,
        )
    if has_usage and choice is None:
        return StreamEvent(
            kind=StreamEventKind.USAGE, usage=usage_from_response(payload), raw=payload
        )
    if finish_reason:
        return StreamEvent(kind=StreamEventKind.FINISH, finish_reason=finish_reason, raw=payload)
    if has_usage:
        return StreamEvent(
            kind=StreamEventKind.USAGE, usage=usage_from_response(payload), raw=payload
        )
    return StreamEvent(kind=StreamEventKind.UNKNOWN, raw=payload)


def decode_stream_bytes(chunks: Iterable[bytes]) -> Iterator[str]:
    """Decode a byte stream into text, holding back an incomplete character.

    Without the holdback, a multi-byte character split across two reads decodes to
    replacement characters and the response text is silently wrong.
    """
    decoder_partial = b""
    for chunk in chunks:
        buffer = decoder_partial + chunk
        for boundary in range(len(buffer), 0, -1):
            try:
                text = buffer[:boundary].decode("utf-8")
            except UnicodeDecodeError:
                continue
            # Emit what decoded and hold back only the trailing incomplete character.
            decoder_partial = buffer[boundary:]
            if text:
                yield text
            break
        else:
            decoder_partial = buffer
    if decoder_partial:
        # A truncated tail is surfaced as replacement characters rather than dropped,
        # so a corrupt stream is visible instead of quietly losing content.
        yield decoder_partial.decode("utf-8", errors="replace")


def assemble(
    stream: Iterable[StreamEvent], *, total_seconds: float | None = None
) -> StreamAssembly:
    """Fold a stream into an assembly, timing the first content and first answer.

    ``first_content_seconds`` is when the first visible text arrived;
    ``first_answer_seconds`` is when the response was complete enough to answer,
    which may be later when reasoning preceded it or when the stream was cut short.
    """
    result = StreamAssembly(total_seconds=total_seconds)
    events = list(stream)
    for index, event in enumerate(events):
        at = _elapsed(index, len(events))
        result.apply(event, at_seconds=at)
    if result.first_answer_seconds is None and events:
        last_delta = max(
            (index for index, event in enumerate(events) if event.contributes_text),
            default=None,
        )
        if last_delta is not None:
            result.first_answer_seconds = _elapsed(last_delta, len(events))
        elif result.content:
            result.first_answer_seconds = result.first_content_seconds
    return result


def _elapsed(index: int, total: int) -> float:
    """Placeholder pacing so tests can assert ordering without wall-clock coupling.

    A real adapter passes measured timings; this keeps the assembly logic testable.
    """
    return float(index) if total else 0.0


def streamed_result(
    *,
    sample_key: Any,
    attempt_id: str,
    attempt_number: int,
    assembly: StreamAssembly,
    manifest_hash: str | None = None,
) -> AdapterResult:
    """Turn an assembled stream into an adapter outcome.

    A stream that never sent its terminating sentinel is an interruption, not a
    completed answer, no matter how much content arrived.
    """
    if not assembly.saw_done:
        return AdapterResult(
            result=None,
            failure=safe_failure(
                FailureKind.INTERRUPTED,
                (
                    f"stream ended without its terminating sentinel after {assembly.chunk_count} "
                    "chunks; the response cannot be confirmed complete"
                ),
                body=None,
            ),
        )
    from stealthbench.schemas.results import GenerationResult

    # Map through the shared vocabulary so a reason the frozen contract does not
    # define becomes an error rather than being forced into a success value.
    finish = map_finish_reason(assembly.finish_reason)
    result = GenerationResult(
        attempt_id=attempt_id,
        sample_key=sample_key,
        attempt_number=attempt_number,
        delivery_status=DeliveryStatus.ACCEPTED,
        response=assembly.content,
        usage=assembly.usage,
        streaming=assembly.measurements(),
        finish_status=finish,
        manifest_hash=manifest_hash,
        redacted_provider_metadata={
            "adapter": "stream",
            "reasoning_chars": len(assembly.reasoning),
            "chunk_count": assembly.chunk_count,
        },
    )
    return AdapterResult(result=result, effective_settings={"streamed": True})


def parse_stream(
    chunks: Iterable[bytes],
    *,
    total_seconds: float | None = None,
    sample_key: Any = None,
    attempt_id: str = "attempt-1",
    attempt_number: int = 1,
    manifest_hash: str | None = None,
) -> tuple[AdapterResult, StreamAssembly]:
    """Parse, assemble and classify a complete stream in one call."""
    parser = SSEParser()
    events: list[StreamEvent] = []
    for text in decode_stream_bytes(chunks):
        events.extend(parser.feed(text))
    tail = parser.flush()
    if tail is not None:
        events.append(tail)
    assembly = assemble(events, total_seconds=total_seconds)
    if sample_key is None:
        return (
            AdapterResult(failure=safe_failure(FailureKind.PROTOCOL, "no sample key supplied")),
            assembly,
        )
    return (
        streamed_result(
            sample_key=sample_key,
            attempt_id=attempt_id,
            attempt_number=attempt_number,
            assembly=assembly,
            manifest_hash=manifest_hash,
        ),
        assembly,
    )


__all__: Sequence[str] = (
    "SSEParser",
    "StreamAssembly",
    "StreamEvent",
    "StreamEventKind",
    "assemble",
    "classify_chunk",
    "decode_stream_bytes",
    "parse_stream",
    "streamed_result",
)
