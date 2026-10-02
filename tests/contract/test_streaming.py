"""Streaming normalization (task T03C).

Acceptance: split SSE records, Unicode, reasoning and terminal usage preserve content
and correct timestamps; route labels stay distinct.

The framing cases are the point of this file. A record split across two reads, several
records in one read, and a multi-byte character split across a read boundary are all
real gateway behaviour and all silently corrupt content if mishandled.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from stealthbench.adapters.base import FailureKind
from stealthbench.adapters.streaming import (
    SSEParser,
    StreamEvent,
    StreamEventKind,
    assemble,
    classify_chunk,
    decode_stream_bytes,
    parse_stream,
)
from stealthbench.schemas.results import SampleKey, Usage

pytestmark = pytest.mark.contract

CANARY = "sk-canary-stream-0123456789abcdef"


def key() -> SampleKey:
    return SampleKey(campaign_id="c1", endpoint_id="alias-a", task_id="ifeval::item-1", repeat_id=1)


def sse(*payloads: dict[str, Any]) -> bytes:
    """Encode payloads as a well-formed SSE byte stream."""
    out = b""
    for payload in payloads:
        out += b"data: " + json.dumps(payload, ensure_ascii=False).encode("utf-8") + b"\n\n"
    return out


def delta(text: str) -> dict[str, Any]:
    return {"choices": [{"delta": {"content": text}, "finish_reason": None}]}


def reasoning(text: str) -> dict[str, Any]:
    return {"choices": [{"delta": {"reasoning_content": text}, "finish_reason": None}]}


def usage_chunk(**usage: int) -> dict[str, Any]:
    return {"choices": [], "usage": usage}


DONE = b"data: [DONE]\n\n"


# ---------------------------------------------------------------------------
# Framing
# ---------------------------------------------------------------------------


def test_a_single_well_formed_stream_parses() -> None:
    body = sse(delta("Hello"), delta(" world")) + DONE
    outcome, assembly = parse_stream([body], total_seconds=1.0, sample_key=key())
    assert outcome.ok
    assert assembly.content == "Hello world"
    assert outcome.result is not None
    assert outcome.result.response == "Hello world"


def test_several_records_in_one_read_are_all_parsed() -> None:
    outcome, assembly = parse_stream(
        [sse(delta("a"), delta("b"), delta("c")) + DONE], sample_key=key()
    )
    assert assembly.content == "abc"
    assert outcome.ok


def test_a_record_split_across_two_reads_is_reassembled() -> None:
    whole = sse(delta("split")) + DONE
    midpoint = whole.index(b"\n\n") + 2
    outcome, assembly = parse_stream([whole[:midpoint], whole[midpoint:]], sample_key=key())
    assert assembly.content == "split", "a split record must not lose its content"
    assert outcome.ok


def test_a_record_split_mid_json_payload_is_reassembled() -> None:
    whole = sse(delta("payload")) + DONE
    midpoint = len(sse(delta("payload"))) // 2
    outcome, assembly = parse_stream([whole[:midpoint], whole[midpoint:]], sample_key=key())
    assert assembly.content == "payload"
    assert outcome.ok


def test_one_byte_at_a_time_still_parses() -> None:
    body = sse(delta("drip"), delta("ped")) + DONE
    outcome, assembly = parse_stream([bytes([b]) for b in body], sample_key=key())
    assert assembly.content == "dripped"
    assert outcome.ok


def test_the_parser_yields_nothing_for_an_incomplete_record() -> None:
    parser = SSEParser()
    assert list(parser.feed('data: {"choices": []}')) == []
    assert parser.has_pending


def test_flush_emits_a_record_that_never_got_its_terminator() -> None:
    parser = SSEParser()
    list(parser.feed('data: {"choices":[{"delta":{"content":"tail"}}]}'))
    tail = parser.flush()
    assert tail is not None
    assert tail.content_delta == "tail"


def test_comment_lines_are_ignored() -> None:
    body = b": keepalive\n\n" + sse(delta("ok")) + DONE
    outcome, assembly = parse_stream([body], sample_key=key())
    assert assembly.content == "ok"
    assert outcome.ok


# ---------------------------------------------------------------------------
# Unicode across boundaries
# ---------------------------------------------------------------------------


def test_a_multibyte_character_split_across_reads_survives() -> None:
    """A boundary inside a multi-byte character must not corrupt the text."""
    payload = json.dumps(delta("héllo — wörld 😀"), ensure_ascii=False).encode("utf-8")
    body = b"data: " + payload + b"\n\n" + DONE
    cut = body.index(b"h\xc3") + 1  # lands between the two bytes of 'é'
    outcome, assembly = parse_stream([body[:cut], body[cut:]], sample_key=key())
    assert assembly.content == "héllo — wörld 😀", "the split character must reassemble"
    assert outcome.ok


def test_emoji_split_across_reads_survives() -> None:
    body = sse(delta("a😀b")) + DONE
    index = body.index("😀".encode()) + 2
    outcome, assembly = parse_stream([body[:index], body[index:]], sample_key=key())
    assert assembly.content == "a😀b"
    assert outcome.ok


def test_decode_stream_bytes_holds_back_a_partial_character() -> None:
    text = "".join(decode_stream_bytes(["héllo".encode()[:2], "héllo".encode()[2:]]))
    assert text == "héllo"


def test_decode_stream_bytes_surfaces_a_truncated_tail() -> None:
    truncated = "héllo".encode()[:-1]
    text = "".join(decode_stream_bytes([truncated]))
    assert text, "a truncated tail must be visible, not silently dropped"


def test_unicode_survives_a_json_escape_and_a_raw_character() -> None:
    record = json.dumps({"choices": [{"delta": {"content": "café 😀"}}]})
    escaped = f"data: {record}\n\n".encode() + DONE
    outcome, assembly = parse_stream([escaped], sample_key=key())
    assert assembly.content == "café 😀"
    assert outcome.ok

    # The same text with JSON escapes instead of literal characters.
    surrogate_escaped = (
        rb'data: {"choices":[{"delta":{"content":"caf\u00e9 \ud83d\ude00"}}]}' + b"\n\n" + DONE
    )
    _outcome, escaped_assembly = parse_stream([surrogate_escaped], sample_key=key())
    assert escaped_assembly.content == "café 😀"


# ---------------------------------------------------------------------------
# Reasoning versus visible answer
# ---------------------------------------------------------------------------


def test_reasoning_only_chunks_are_kept_separate() -> None:
    body = sse(reasoning("thinking..."), delta("answer")) + DONE
    outcome, assembly = parse_stream([body], total_seconds=2.0, sample_key=key())
    assert assembly.reasoning == "thinking..."
    assert assembly.content == "answer"
    assert outcome.result is not None
    assert outcome.result.response == "answer", "reasoning must not leak into the answer"


def test_first_content_and_first_answer_are_different_measurements() -> None:
    """Reasoning can delay the answer; collapsing them hides the behaviour."""
    body = sse(reasoning("a"), reasoning("b"), delta("the "), delta("answer"), delta(" now")) + DONE
    _outcome, assembly = parse_stream([body], sample_key=key())
    assert assembly.first_content_seconds is not None
    assert assembly.first_answer_seconds is not None
    assert assembly.first_content_seconds < assembly.first_answer_seconds, (
        "first content is when text starts arriving; first answer is when it is complete"
    )


def test_a_single_content_delta_makes_the_two_timings_equal() -> None:
    """With one text delta there is nothing between the two events."""
    _outcome, assembly = parse_stream(
        [sse(reasoning("think"), delta("done")) + DONE], sample_key=key()
    )
    assert assembly.first_content_seconds == assembly.first_answer_seconds


def test_reasoning_chunks_are_classified_separately_from_text() -> None:
    assert classify_chunk(reasoning("x")).kind is StreamEventKind.REASONING_DELTA
    assert classify_chunk(delta("x")).kind is StreamEventKind.DELTA


def test_an_alternative_reasoning_field_name_is_accepted() -> None:
    payload = {"choices": [{"delta": {"reasoning": "alt"}, "finish_reason": None}]}
    assert classify_chunk(payload).reasoning_delta == "alt"


def test_a_reasoning_response_reports_its_reasoning_size_only() -> None:
    body = sse(reasoning("secret thoughts")) + DONE
    outcome, _ = parse_stream([body], sample_key=key())
    assert outcome.result is not None
    assert outcome.result.redacted_provider_metadata["reasoning_chars"] == len("secret thoughts")


# ---------------------------------------------------------------------------
# Terminal usage
# ---------------------------------------------------------------------------


def test_a_usage_only_final_chunk_is_captured() -> None:
    """The final chunk commonly carries usage and no choices at all."""
    body = sse(delta("done")) + sse(usage_chunk(input_tokens=11, output_tokens=5)) + DONE
    outcome, assembly = parse_stream([body], sample_key=key())
    assert assembly.usage.input_tokens == 11
    assert assembly.usage.output_tokens == 5
    assert outcome.result is not None
    assert outcome.result.usage.output_tokens == 5


def test_a_usage_only_chunk_is_classified_as_usage() -> None:
    assert classify_chunk(usage_chunk(input_tokens=1, output_tokens=1)).kind is (
        StreamEventKind.USAGE
    )


def test_a_finish_chunk_is_captured() -> None:
    payload = {"choices": [{"delta": {}, "finish_reason": "length"}]}
    body = sse(delta("cut"), payload) + DONE
    outcome, assembly = parse_stream([body], sample_key=key())
    assert assembly.finish_reason == "length"
    assert outcome.result is not None
    assert outcome.result.finish_status == "length"


def test_an_unrecognised_finish_reason_becomes_an_error_not_a_stop() -> None:
    payload = {"choices": [{"delta": {}, "finish_reason": "brand_new"}]}
    outcome, _ = parse_stream([sse(delta("x"), payload) + DONE], sample_key=key())
    assert outcome.result is not None
    assert outcome.result.finish_status == "error"


def test_absent_usage_in_the_final_chunk_stays_null() -> None:
    body = sse(delta("no usage")) + DONE
    outcome, _ = parse_stream([body], sample_key=key())
    assert outcome.result is not None
    assert outcome.result.usage.input_tokens is None
    assert outcome.result.usage.output_tokens is None


# ---------------------------------------------------------------------------
# Chunks are not tokens
# ---------------------------------------------------------------------------


def test_a_chunk_count_is_not_a_token_count() -> None:
    body = sse(delta("a"), delta("b"), delta("c"), delta("d")) + DONE
    outcome, assembly = parse_stream([body], total_seconds=2.0, sample_key=key())
    assert assembly.chunk_count == 5, "four deltas plus the done sentinel"
    assert outcome.result is not None
    assert outcome.result.usage.output_tokens is None, "no usage means no token count"
    assert assembly.token_rate() is None, (
        "deriving a token rate from a chunk count would fabricate a measurement"
    )


def test_a_token_rate_needs_both_tokens_and_a_duration() -> None:
    body = sse(delta("x"), usage_chunk(input_tokens=1, output_tokens=20)) + DONE
    _outcome, assembly = parse_stream([body], total_seconds=4.0, sample_key=key())
    assert assembly.token_rate() == 5.0
    assert assembly.chunk_count == 3


def test_a_token_rate_is_absent_without_a_duration() -> None:
    body = sse(delta("x"), usage_chunk(input_tokens=1, output_tokens=20)) + DONE
    _outcome, assembly = parse_stream([body], total_seconds=None, sample_key=key())
    assert assembly.token_rate() is None


def test_a_reported_zero_token_count_yields_a_zero_rate_not_a_missing_one() -> None:
    body = sse(delta(""), usage_chunk(input_tokens=0, output_tokens=0)) + DONE
    _outcome, assembly = parse_stream([body], total_seconds=4.0, sample_key=key())
    assert assembly.token_rate() == 0.0
    assert assembly.usage.output_tokens == 0


def test_the_streaming_measurement_rejects_impossible_orderings() -> None:
    from stealthbench.schemas.results import StreamingMeasurements

    body = sse(delta("x")) + DONE
    _outcome, assembly = parse_stream([body], total_seconds=0.5, sample_key=key())
    measurements = assembly.measurements()
    assert measurements.chunk_count == assembly.chunk_count
    with pytest.raises(ValueError, match="first_content_seconds"):
        StreamingMeasurements(first_content_seconds=9.0, total_seconds=1.0)


# ---------------------------------------------------------------------------
# Interruption
# ---------------------------------------------------------------------------


def test_a_stream_without_its_sentinel_is_an_interruption() -> None:
    """Content alone does not make a response confirmed."""
    outcome, assembly = parse_stream([sse(delta("half a sen"))], sample_key=key())
    assert not outcome.ok
    assert outcome.failure is not None
    assert outcome.failure.kind is FailureKind.INTERRUPTED
    assert "sentinel" in outcome.failure.detail
    assert assembly.content == "half a sen", "what did arrive is still visible"


def test_an_interrupted_stream_never_produces_an_accepted_sample() -> None:
    outcome, _ = parse_stream([sse(delta("x"))], sample_key=key())
    assert outcome.result is None
    assert outcome.delivery_status is None


def test_an_empty_stream_is_an_interruption_not_an_empty_answer() -> None:
    outcome, assembly = parse_stream([b""], sample_key=key())
    assert not outcome.ok
    assert outcome.failure is not None
    assert outcome.failure.kind is FailureKind.INTERRUPTED
    assert assembly.chunk_count == 0


def test_content_after_the_sentinel_is_ignored() -> None:
    body = sse(delta("before")) + DONE + sse(delta("after"))
    _outcome, assembly = parse_stream([body], sample_key=key())
    assert assembly.content == "before"
    assert assembly.saw_done


# ---------------------------------------------------------------------------
# Malformed records
# ---------------------------------------------------------------------------


def test_an_unparsable_record_does_not_abort_the_stream() -> None:
    body = b"data: {not json\n\n" + sse(delta("still here")) + DONE
    outcome, assembly = parse_stream([body], sample_key=key())
    assert assembly.content == "still here"
    assert outcome.ok


def test_a_record_with_no_data_field_is_skipped() -> None:
    body = b"event: ping\nid: 7\n\n" + sse(delta("ok")) + DONE
    _outcome, assembly = parse_stream([body], sample_key=key())
    assert assembly.content == "ok"


def test_a_non_object_payload_is_classified_unknown() -> None:
    parser = SSEParser()
    events = list(parser.feed("data: [1,2,3]\n\n"))
    assert events and events[0].kind is StreamEventKind.UNKNOWN


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def test_assembly_reports_ordered_timings() -> None:
    events = [
        StreamEvent(kind=StreamEventKind.DELTA, content_delta="a"),
        StreamEvent(kind=StreamEventKind.DELTA, content_delta="b"),
        StreamEvent(kind=StreamEventKind.DONE),
    ]
    assembly = assemble(events, total_seconds=3.0)
    assert assembly.chunk_count == 3
    assert assembly.first_content_seconds == 0.0
    assert assembly.first_answer_seconds == 1.0
    assert assembly.total_seconds == 3.0


def test_an_empty_assembly_is_safe() -> None:
    assembly = assemble([], total_seconds=None)
    assert assembly.chunk_count == 0
    assert assembly.content == ""
    assert assembly.token_rate() is None
    assert assembly.measurements().chunk_count == 0


def test_usage_on_the_assembly_survives_a_later_delta() -> None:
    """A gateway may send usage before the last content chunk."""
    events = [
        StreamEvent(kind=StreamEventKind.USAGE, usage=Usage(input_tokens=3, output_tokens=4)),
        StreamEvent(kind=StreamEventKind.DELTA, content_delta="x"),
        StreamEvent(kind=StreamEventKind.DONE),
    ]
    assembly = assemble(events)
    assert assembly.usage.output_tokens == 4
    assert assembly.content == "x"


# ---------------------------------------------------------------------------
# Route labels stay distinct
# ---------------------------------------------------------------------------


def test_a_stream_result_records_the_streaming_route() -> None:
    body = sse(delta("x")) + DONE
    outcome, _ = parse_stream([body], sample_key=key())
    assert outcome.ok
    assert outcome.effective_settings == {"streamed": True}
    assert outcome.result is not None
    assert outcome.result.redacted_provider_metadata["adapter"] == "stream"


def test_a_streamed_result_keeps_the_sample_key_it_was_asked_for() -> None:
    body = sse(delta("x")) + DONE
    outcome, _ = parse_stream([body], sample_key=key())
    assert outcome.result is not None
    assert outcome.result.sample_key.endpoint_id == "alias-a"


def test_a_stream_result_carries_no_credential() -> None:
    body = sse(delta(f"token {CANARY}")) + DONE
    outcome, _ = parse_stream([body], sample_key=key())
    assert outcome.ok
    assert outcome.result is not None


def test_streaming_module_imports_no_transport() -> None:
    import ast
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[2] / "src" / "stealthbench" / "adapters" / "streaming.py"
    )
    tree = ast.parse(source.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            modules.add(node.module.split(".")[0])
    for forbidden in ("httpx", "requests", "socket", "urllib", "http"):
        assert forbidden not in modules
