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
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from math import isfinite
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
    ERROR = "error"
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
    error: dict[str, Any] | None = None
    error_message: str | None = None
    #: Whether any frame carried content, reasoning or a usage report.
    saw_content: bool = False

    def apply(self, event: StreamEvent, *, at_seconds: float | None) -> None:
        self.chunk_count += 1
        if event.content_delta:
            if self.first_content_seconds is None and at_seconds is not None:
                self.first_content_seconds = at_seconds
            self.saw_content = True
            self.content += event.content_delta
        if event.reasoning_delta:
            self.saw_content = True
            self.reasoning += event.reasoning_delta
        # Any record may carry the terminal reason; a reason-only record and a final
        # content record both deliver it.
        if event.finish_reason:
            self.finish_reason = event.finish_reason
        if event.kind is StreamEventKind.USAGE or event.usage.provider_reported:
            self.usage = _merge_usage(self.usage, event.usage)
        if event.kind is StreamEventKind.DONE:
            self.saw_done = True
        if event.kind is StreamEventKind.ERROR:
            reported = event.raw.get("error")
            if isinstance(reported, Mapping):
                self.error = dict(reported)
                inner = reported.get("message")
                self.error_message = inner if isinstance(inner, str) else json.dumps(dict(reported))
            else:
                self.error_message = str(reported)

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


def _sse_lines(record: str) -> list[str]:
    """Split a record on CR and LF only.

    ``str.splitlines`` also breaks on U+0085, U+2028, U+2029 and the C0 separators,
    none of which terminate an SSE line. A model is free to emit them inside a JSON
    string, and splitting there truncates the record: the answer is lost and a
    mid-stream error frame silently becomes an interruption.
    """
    lines: list[str] = []
    current: list[str] = []
    index = 0
    length = len(record)
    while index < length:
        char = record[index]
        if char == "\r":
            lines.append("".join(current))
            current = []
            index += 2 if record[index + 1 : index + 2] == "\n" else 1
            continue
        if char == "\n":
            lines.append("".join(current))
            current = []
            index += 1
            continue
        current.append(char)
        index += 1
    lines.append("".join(current))
    return lines


class SSEParser:
    """Incremental SSE parser that survives arbitrary read boundaries.

    Feed it decoded text of any shape; it yields whole records. Byte-level framing is
    handled by :func:`decode_stream_bytes`, which keeps a partial multi-byte sequence
    buffered rather than corrupting the character at the boundary.
    """

    def __init__(self) -> None:
        self._buffer = ""
        self._saw_done = False
        self._pending_cr = False

    def feed(self, chunk: str) -> Iterator[StreamEvent]:
        """Consume a chunk of decoded text and yield any complete records.

        Per the SSE specification a record is dispatched on a blank line terminated by
        CRLF, LF or a bare CR, so a CRLF-framed stream would otherwise never split and
        all content would be lost.

        A CR at the very end of a read is held back rather than converted: the LF that
        completes its CRLF may arrive in the next read, and converting eagerly would
        manufacture a blank line where the wire had a single newline, splitting one
        record into two unparsable halves.
        """
        if self._saw_done:
            return
        if self._pending_cr:
            # The held-back CR terminated a line either way: CRLF or a bare CR.
            # Dropping it entirely would weld the two lines it separated together.
            chunk = chunk[1:] if chunk.startswith("\n") else chunk
            chunk = "\n" + chunk
            self._pending_cr = False
        if chunk.endswith("\r"):
            self._pending_cr = True
            chunk = chunk[:-1]
        self._buffer += chunk.replace("\r\n", "\n").replace("\r", "\n")
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
        if self._pending_cr:
            # The stream ended on a bare CR, which is itself a line terminator.
            self._buffer += "\n"
            self._pending_cr = False
        remainder = self._buffer.replace("\r\n", "\n").replace("\r", "\n").strip(" \t\n")
        self._buffer = ""
        if not remainder:
            return None
        return self._parse_record(remainder)

    @property
    def has_pending(self) -> bool:
        return bool(self._buffer.strip())

    def _parse_record(self, raw_record: str) -> StreamEvent | None:
        data_lines: list[str] = []
        for line in _sse_lines(raw_record):
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


def _merge_usage(into: Usage, newer: Usage) -> Usage:
    """Fold newly reported counts into what was already known.

    Some gateways split usage across frames, so a field the first frame left absent is
    filled in from a later one.

    A field that is already reported is kept. Last-wins would let a terminal frame
    carrying ``0`` overwrite a real reported count, turning a measurement into the
    forbidden zero; where a gateway contradicts itself the earlier report is retained
    rather than silently replaced.
    """
    return Usage(
        input_tokens=into.input_tokens if into.input_tokens is not None else newer.input_tokens,
        output_tokens=(
            into.output_tokens if into.output_tokens is not None else newer.output_tokens
        ),
        cached_input_tokens=(
            into.cached_input_tokens
            if into.cached_input_tokens is not None
            else newer.cached_input_tokens
        ),
        reasoning_tokens=(
            into.reasoning_tokens if into.reasoning_tokens is not None else newer.reasoning_tokens
        ),
        provider_reported=into.provider_reported or newer.provider_reported,
    )


def _text_from_parts(parts: Sequence[Any]) -> str:
    """Concatenate the text of a content-parts delta, ignoring non-text parts."""
    pieces: list[str] = []
    for part in parts:
        if not isinstance(part, Mapping):
            continue
        text = part.get("text")
        if isinstance(text, str):
            pieces.append(text)
    return "".join(pieces)


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
            elif isinstance(content, Sequence) and not isinstance(content, (str, bytes)):
                # The same shape the non-streaming path already understands. Losing it
                # here would accept an empty answer as a completed one.
                content_delta = _text_from_parts(content)
            for name in ("reasoning_content", "reasoning"):
                reasoning = delta.get(name)
                if isinstance(reasoning, str) and reasoning:
                    reasoning_delta = reasoning
                    break
        finish = choice.get("finish_reason")
        if isinstance(finish, str):
            finish_reason = finish

    has_usage = isinstance(usage_raw, dict)
    error_raw = payload.get("error")
    if isinstance(error_raw, (dict, str)):
        # A gateway that reports a failure mid-stream is a protocol error. Ignoring
        # the frame and calling the result "interrupted" would hide the cause and
        # misreport it to the scheduler as a retryable network event.
        return StreamEvent(kind=StreamEventKind.ERROR, raw=payload)
    if content_delta:
        # The terminal record commonly carries the last delta *and* finish_reason
        # together. The reason must travel with it, or a length-truncated answer is
        # recorded as a clean stop.
        return StreamEvent(
            kind=StreamEventKind.DELTA,
            content_delta=content_delta,
            reasoning_delta=reasoning_delta,
            finish_reason=finish_reason,
            usage=usage_from_response(payload) if has_usage else Usage(),
            raw=payload,
        )
    if reasoning_delta:
        return StreamEvent(
            kind=StreamEventKind.REASONING_DELTA,
            reasoning_delta=reasoning_delta,
            finish_reason=finish_reason,
            usage=usage_from_response(payload) if has_usage else Usage(),
            raw=payload,
        )
    if has_usage and choice is None:
        return StreamEvent(
            kind=StreamEventKind.USAGE, usage=usage_from_response(payload), raw=payload
        )
    if finish_reason:
        # A gateway may put usage on the same record as the terminal reason. Dropping
        # it would turn a billed request into a missing measurement.
        return StreamEvent(
            kind=StreamEventKind.FINISH,
            finish_reason=finish_reason,
            usage=usage_from_response(payload) if has_usage else Usage(),
            raw=payload,
        )
    if has_usage:
        return StreamEvent(
            kind=StreamEventKind.USAGE, usage=usage_from_response(payload), raw=payload
        )
    return StreamEvent(kind=StreamEventKind.UNKNOWN, raw=payload)


def decode_stream_bytes(chunks: Iterable[bytes]) -> Iterator[str]:
    """Decode a byte stream into text, holding back an incomplete character.

    Without the holdback, a multi-byte character split across two reads decodes to
    replacement characters and the response text is silently wrong.

    Only a trailing incomplete sequence is held back. Corrupt bytes are emitted as
    replacement characters immediately, so one bad byte cannot hold the rest of the
    stream hostage until end of stream.
    """
    decoder_partial = b""
    for chunk in chunks:
        buffer = decoder_partial + chunk
        decoder_partial = b""
        while buffer:
            try:
                text = buffer.decode("utf-8")
            except UnicodeDecodeError as exc:
                if exc.end == len(buffer):
                    # The failing sequence runs to the end of what we have, so it may
                    # simply be truncated: hold it back for the next read.
                    decoder_partial = buffer
                    break
                if exc.start:
                    yield buffer[: exc.start].decode("utf-8", errors="replace")
                bad = buffer[exc.start : max(exc.end, exc.start + 1)]
                yield bad.decode("utf-8", errors="replace")
                buffer = buffer[max(exc.end, exc.start + 1) :]
                continue
            yield text
            break
    if decoder_partial:
        # A truncated tail is surfaced as replacement characters rather than dropped,
        # so a corrupt stream is visible instead of quietly losing content.
        yield decoder_partial.decode("utf-8", errors="replace")


def assemble(
    stream: Iterable[StreamEvent],
    *,
    total_seconds: float | None = None,
    pacing: Callable[[int, StreamEvent], float | None] | None = None,
) -> StreamAssembly:
    """Fold a stream into an assembly, timing the first content and first answer.

    ``first_content_seconds`` is when the first visible text arrived;
    ``first_answer_seconds`` is when the response was complete enough to answer,
    which may be later when reasoning preceded it or when the stream was cut short.

    ``pacing`` supplies measured per-event offsets from a real adapter. Without it no
    per-event timing is recorded at all: an event's position is not a duration, and a
    fabricated timing is worse than a missing one.
    """
    result = StreamAssembly(total_seconds=total_seconds)
    events = list(stream)

    def offset(index: int) -> float | None:
        if pacing is not None:
            return pacing(index, events[index])
        # With no measured pacing there is no clock, and an event's position in the
        # stream is not a duration. Publishing the index in a seconds field would
        # fabricate a timing -- 0.0 for the first one, which is the substitution the
        # contract forbids.
        return None

    for index, event in enumerate(events):
        result.apply(event, at_seconds=offset(index))
    if result.first_answer_seconds is None and events:
        last_delta = max(
            (index for index, event in enumerate(events) if event.contributes_text),
            default=None,
        )
        if last_delta is not None:
            result.first_answer_seconds = offset(last_delta)
        elif result.content:
            result.first_answer_seconds = result.first_content_seconds
    return result


def _coherent_measurements(assembly: StreamAssembly) -> StreamingMeasurements:
    """Measurements that cannot all be true together are dropped, not clamped.

    A caller-supplied pacing function can pair an offset past the measured total, or
    produce a negative, infinite or NaN offset. Clamping would invent a coincidence;
    raising would take the campaign down on a timing artefact. An offset that is not a
    finite number inside the stream's own window is simply not recorded.
    """

    def usable(value: float | None) -> float | None:
        if value is None or not isfinite(value) or value < 0:
            return None
        total = assembly.total_seconds
        if total is not None and (not isfinite(total) or value > total):
            return None
        return value

    total = assembly.total_seconds
    measurements = StreamingMeasurements(
        first_content_seconds=usable(assembly.first_content_seconds),
        first_answer_seconds=usable(assembly.first_answer_seconds),
        total_seconds=usable(total),
        chunk_count=assembly.chunk_count,
    )
    # first_content cannot follow first_answer; if it somehow does, neither is sound.
    if (
        measurements.first_content_seconds is not None
        and measurements.first_answer_seconds is not None
        and measurements.first_content_seconds > measurements.first_answer_seconds
    ):
        return StreamingMeasurements(
            first_content_seconds=None,
            first_answer_seconds=measurements.first_answer_seconds,
            total_seconds=measurements.total_seconds,
            chunk_count=assembly.chunk_count,
        )
    return measurements


def streamed_result(
    *,
    sample_key: Any,
    attempt_id: str,
    attempt_number: int,
    assembly: StreamAssembly,
    route: str,
    adapter: str = "stream",
    manifest_hash: str | None = None,
    extra_secrets: Iterable[str] = (),
) -> AdapterResult:
    """Turn an assembled stream into an adapter outcome.

    A stream that never sent its terminating sentinel is an interruption, not a
    completed answer, no matter how much content arrived.
    """
    if assembly.error_message is not None:
        # The gateway said what went wrong. Reporting that is the point: an upstream
        # error is not the same failure as a connection that died mid-answer.
        return AdapterResult(
            result=None,
            failure=safe_failure(
                FailureKind.PROTOCOL,
                f"endpoint reported an error mid-stream: {assembly.error_message}",
                body=assembly.error,
                extra_secrets=frozenset(extra_secrets),
            ),
        )
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
    # A usage *block* is not an answer: an endpoint that sent `usage: {}` reported
    # nothing, so accepting it would invent a sample with no content.
    if (
        not assembly.saw_content
        and assembly.usage.input_tokens is None
        and assembly.usage.output_tokens is None
    ):
        # The stream was terminated but carried nothing at all. Accepting it would
        # invent a sample with no answer: the frames were absent or unusable, not
        # empty in fact.
        return AdapterResult(
            result=None,
            failure=safe_failure(
                FailureKind.PROTOCOL,
                "the recorded stream was terminated but carried no usable frames",
                body=None,
                extra_secrets=frozenset(extra_secrets),
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
        streaming=_coherent_measurements(assembly),
        finish_status=finish,
        manifest_hash=manifest_hash,
        redacted_provider_metadata={
            "adapter": adapter,
            # The route is recorded on every streamed result so a stream through the
            # gateway and a stream through a fixture are never conflated. Route,
            # provider and family stay independent labels.
            "route": route,
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
    route: str = "unlabeled",
    adapter: str = "stream",
    extra_secrets: Iterable[str] = (),
    pacing: Callable[[int, StreamEvent], float | None] | None = None,
) -> tuple[AdapterResult, StreamAssembly]:
    """Parse, assemble and classify a complete stream in one call.

    ``route`` and ``adapter`` are required to be stated by the caller rather than
    inferred, so a streamed result always says which route produced it.

    ``pacing`` supplies measured per-event offsets. Without it no duration is
    recorded: a replayed capture has no clock, and an event's position is not a time.
    """
    parser = SSEParser()
    events: list[StreamEvent] = []
    for text in decode_stream_bytes(chunks):
        events.extend(parser.feed(text))
    tail = parser.flush()
    if tail is not None:
        events.append(tail)
    assembly = assemble(events, total_seconds=total_seconds, pacing=pacing)
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
            route=route,
            adapter=adapter,
            manifest_hash=manifest_hash,
            extra_secrets=extra_secrets,
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
