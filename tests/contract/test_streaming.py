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

from stealthbench.adapters.base import FailureKind, FixtureBundle, FixtureTransport
from stealthbench.adapters.streaming import (
    SSEParser,
    StreamEvent,
    StreamEventKind,
    assemble,
    classify_chunk,
    decode_stream_bytes,
    parse_stream,
    streamed_result,
)
from stealthbench.schemas.campaign import Capabilities
from stealthbench.schemas.results import ModelRequest, SampleKey, Usage

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
    """Reasoning can delay the answer; collapsing them hides the behaviour.

    Timing is only recorded from measured pacing: with no clock there is no
    duration, and publishing an event's position as one would be a fabricated
    measurement.
    """
    body = sse(reasoning("a"), reasoning("b"), delta("the "), delta("answer"), delta(" now")) + DONE
    _outcome, assembly = parse_stream([body], sample_key=key(), pacing=_by_index)
    assert assembly.first_content_seconds is not None
    assert assembly.first_answer_seconds is not None
    assert assembly.first_content_seconds < assembly.first_answer_seconds, (
        "first content is when text starts arriving; first answer is when it is complete"
    )


def test_a_single_content_delta_makes_the_two_timings_equal() -> None:
    """With one text delta there is nothing between the two events."""
    _outcome, assembly = parse_stream(
        [sse(reasoning("think"), delta("done")) + DONE], sample_key=key(), pacing=_by_index
    )
    assert assembly.first_content_seconds == assembly.first_answer_seconds


def test_no_timing_is_recorded_without_a_clock() -> None:
    """Neither adapter measures time, so a streamed artifact records no durations.

    Publishing the frame index under a seconds field would record 0.0 for the first
    event, which is exactly the substitution the contract forbids.
    """
    body = sse(reasoning("a"), delta("the answer")) + DONE
    _outcome, assembly = parse_stream([body], sample_key=key())
    assert assembly.first_content_seconds is None
    assert assembly.first_answer_seconds is None
    assert assembly.total_seconds is None
    assert assembly.chunk_count == 3


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


def test_a_truncated_terminal_record_is_not_recorded_as_a_clean_stop() -> None:
    """The real terminal shape: last content delta and finish_reason in one record.

    An OpenAI-compatible gateway sends them together, so a parser that keeps the
    reason only on an empty-delta record turns every truncated answer into a stop.
    """
    truncated = {"choices": [{"delta": {"content": "truncated ans"}, "finish_reason": "length"}]}
    outcome, assembly = parse_stream(
        [sse(delta("truncated ans"), truncated) + DONE], sample_key=key()
    )
    assert assembly.finish_reason == "length"
    assert outcome.result is not None
    assert outcome.result.finish_status == "length"
    assert outcome.result.finish_status != "stop"


@pytest.mark.parametrize(
    ("reason", "expected"),
    [("length", "length"), ("content_filter", "content_filter"), ("tool_calls", "tool_calls")],
)
def test_every_terminal_reason_survives_a_combined_final_record(reason: str, expected: str) -> None:
    combined = {"choices": [{"delta": {"content": "tail"}, "finish_reason": reason}]}
    outcome, _ = parse_stream([sse(delta("tail"), combined) + DONE], sample_key=key())
    assert outcome.result is not None
    assert outcome.result.finish_status == expected


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


def _by_index(index: int, _event: StreamEvent) -> float:
    """Stand-in for an adapter that measured each event's arrival."""
    return float(index)


def test_assembly_reports_ordered_timings() -> None:
    events = [
        StreamEvent(kind=StreamEventKind.DELTA, content_delta="a"),
        StreamEvent(kind=StreamEventKind.DELTA, content_delta="b"),
        StreamEvent(kind=StreamEventKind.DONE),
    ]
    assembly = assemble(events, total_seconds=3.0, pacing=_by_index)
    assert assembly.chunk_count == 3
    assert assembly.first_content_seconds == 0.0
    assert assembly.first_answer_seconds == 1.0
    assert assembly.total_seconds == 3.0


def test_no_offset_is_invented_against_a_measured_total() -> None:
    """Without measured pacing the per-event timings stay absent, not fabricated.

    Pairing an event's position with a real total can produce a time past the total,
    which the contract rejects -- and clamping it would invent a coincidence.
    """
    events = [StreamEvent(kind=StreamEventKind.DELTA, content_delta=f"c{i}") for i in range(9)]
    events.append(StreamEvent(kind=StreamEventKind.DONE))
    assembly = assemble(events, total_seconds=1.0)
    assert assembly.first_content_seconds is None
    assert assembly.first_answer_seconds is None
    assert assembly.total_seconds == 1.0


def test_a_measured_total_with_many_frames_still_produces_a_result() -> None:
    """Regression: fabricated offsets used to raise out of the streaming adapters."""
    frames = [
        {"choices": [{"delta": {"reasoning_content": f"think {i}"}, "finish_reason": None}]}
        for i in range(5)
    ]
    frames.append({"choices": [{"delta": {"content": "done"}, "finish_reason": "stop"}]})
    frames.append({"choices": [], "usage": {"input_tokens": 4, "output_tokens": 9}})
    body = (
        b"".join(b"data: " + json.dumps(f).encode() + b"\n\n" for f in frames) + b"data: [DONE]\n\n"
    )
    outcome, _ = parse_stream([body], total_seconds=1.0, sample_key=key(), route="zen")
    assert outcome.ok
    assert outcome.result is not None
    assert outcome.result.response == "done"
    assert outcome.result.streaming is not None
    assert outcome.result.streaming.total_seconds == 1.0
    assert outcome.result.streaming.first_content_seconds is None


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
    """A model can echo a prompt that contains a secret; the result must be clean.

    The previous version asserted only that the call succeeded, so it passed with
    redaction removed entirely.
    """
    body = sse(delta(f"token {CANARY}")) + DONE
    outcome, _ = parse_stream([body], sample_key=key())
    assert outcome.ok
    assert outcome.result is not None
    # The response legitimately echoes what the model was told, so the check is on
    # the metadata the harness itself adds, which must never carry a credential.
    assert CANARY not in json.dumps(outcome.result.redacted_provider_metadata)


def test_a_streamed_result_records_its_route() -> None:
    """T03C: route labels stay distinct across routes serving identical bytes."""
    body = sse(delta("x")) + DONE
    zen, _ = parse_stream([body], sample_key=key(), route="zen", adapter="zen")
    fixture, _ = parse_stream([body], sample_key=key(), route="fixture", adapter="fixture")
    assert zen.result is not None and fixture.result is not None
    assert zen.result.response == fixture.result.response
    assert zen.result.redacted_provider_metadata["route"] == "zen"
    assert fixture.result.redacted_provider_metadata["route"] == "fixture"
    assert zen.result.redacted_provider_metadata != fixture.result.redacted_provider_metadata


def test_a_crlframed_stream_keeps_all_of_its_content() -> None:
    """A record is dispatched on a blank line terminated by CRLF, LF or a bare CR."""
    body = (
        b"data: " + json.dumps(delta("FIRST")).encode("utf-8") + b"\r\n\r\n"
        b"data: " + json.dumps(delta("SECOND")).encode("utf-8") + b"\r\n\r\n"
        b"data: [DONE]\r\n\r\n"
    )
    outcome, assembly = parse_stream([body], sample_key=key(), route="zen")
    assert assembly.content == "FIRSTSECOND", "CRLF framing must not lose content"
    assert outcome.ok
    assert outcome.result is not None
    assert outcome.result.response == "FIRSTSECOND"


def test_a_mixed_crlf_and_lf_stream_keeps_all_of_its_content() -> None:
    body = (
        b"data: " + json.dumps(delta("A")).encode("utf-8") + b"\r\n\r\n"
        b"data: " + json.dumps(delta("B")).encode("utf-8") + b"\n\n"
        b"data: " + json.dumps(delta("C")).encode("utf-8") + b"\r\r"
        b"data: [DONE]\n\n"
    )
    outcome, assembly = parse_stream([body], sample_key=key(), route="zen")
    assert assembly.content == "ABC"
    assert outcome.ok


def test_a_crlf_stream_split_across_reads_still_parses() -> None:
    body = b"data: " + json.dumps(delta("split")).encode("utf-8") + b"\r\n\r\ndata: [DONE]\r\n\r\n"
    midpoint = body.index(b"\r\n\r\n") + 3
    outcome, assembly = parse_stream(
        [body[:midpoint], body[midpoint:]], sample_key=key(), route="zen"
    )
    assert assembly.content == "split"
    assert outcome.ok


def test_a_fixture_adapter_can_stream_and_labels_its_own_route() -> None:
    """A dispatched request must actually be able to reach the streaming path."""
    streaming_caps = Capabilities(
        streaming=True, tool_calls=False, reasoning=False, usage_reporting=False, logprobs=False
    )
    transport = FixtureTransport(
        FixtureBundle.model_validate(
            {
                "name": "streamed",
                "capabilities": streaming_caps.model_dump(),
                "exchanges": [
                    {
                        "endpoint_id": "alias-a",
                        "benchmark_id": "ifeval",
                        "item_id": "item-1",
                        "response": "streamed answer",
                        "stream_frames": [
                            {"choices": [{"delta": {"content": "streamed "}}]},
                            {
                                "choices": [
                                    {"delta": {"content": "answer"}, "finish_reason": "stop"}
                                ]
                            },
                            {"choices": [], "usage": {"input_tokens": 4, "output_tokens": 3}},
                        ],
                    }
                ],
            }
        )
    )
    outcome = transport.stream(
        sample_key=key(),
        request=ModelRequest(messages=[{"role": "user", "content": "hi"}], max_output_tokens=32),
        prompt_hash="a" * 64,
        capabilities=streaming_caps,
    )
    assert outcome.ok
    assert outcome.result is not None
    assert outcome.result.response == "streamed answer"
    assert outcome.result.usage.output_tokens == 3
    assert outcome.result.redacted_provider_metadata["route"] == "fixture"
    assert outcome.result.streaming is not None


def test_streaming_off_a_non_streaming_endpoint_is_refused() -> None:
    caps = Capabilities(
        streaming=False, tool_calls=False, reasoning=False, usage_reporting=False, logprobs=False
    )
    transport = FixtureTransport(
        FixtureBundle.model_validate(
            {
                "name": "nostream",
                "capabilities": caps.model_dump(),
                "exchanges": [
                    {
                        "endpoint_id": "alias-a",
                        "benchmark_id": "ifeval",
                        "item_id": "item-1",
                        "response": "x",
                        "stream_frames": [{"choices": [{"delta": {"content": "x"}}]}],
                    }
                ],
            }
        )
    )
    outcome = transport.stream(
        sample_key=key(),
        request=ModelRequest(messages=[{"role": "user", "content": "hi"}], max_output_tokens=32),
        prompt_hash="a" * 64,
        capabilities=caps,
    )
    assert not outcome.ok
    assert outcome.failure is not None


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


def test_a_mid_stream_error_frame_is_a_protocol_failure_not_an_interruption() -> None:
    """The gateway said what went wrong; an interruption label would hide the cause."""
    body = (
        b'data: {"choices":[{"delta":{"content":"partial"},"finish_reason":null}]}\n\n'
        b'data: {"error":{"message":"upstream overloaded","type":"server_error"}}\n\n'
    )
    outcome, assembly = parse_stream([body], sample_key=key(), route="zen")
    assert not outcome.ok
    assert outcome.result is None
    assert outcome.failure is not None
    assert outcome.failure.kind is FailureKind.PROTOCOL
    assert "upstream overloaded" in outcome.failure.detail
    assert outcome.failure.body == {"message": "upstream overloaded", "type": "server_error"}
    # The partial content is still available for diagnosis, just not accepted.
    assert assembly.content == "partial"


def test_a_bare_string_error_frame_is_still_a_protocol_failure() -> None:
    outcome, _ = parse_stream(
        [b'data: {"error":"gateway restart"}\n\n'], sample_key=key(), route="zen"
    )
    assert outcome.failure is not None
    assert outcome.failure.kind is FailureKind.PROTOCOL
    assert "gateway restart" in outcome.failure.detail


def test_a_credential_in_a_mid_stream_error_body_is_redacted() -> None:
    body = f'data: {{"error":{{"message":"rejected {CANARY}"}}}}\n\n'.encode()
    outcome, _ = parse_stream([body], sample_key=key(), route="zen", extra_secrets={CANARY})
    assert outcome.failure is not None
    assert CANARY not in outcome.failure.detail
    assert CANARY not in json.dumps(outcome.failure.to_dict())


# ---------------------------------------------------------------------------
# Regression tests for the second G03 review
# ---------------------------------------------------------------------------


def test_a_capture_without_the_sentinel_is_never_replayed_as_complete() -> None:
    """CRITICAL: the adapters used to append `data: [DONE]` to every capture.

    A connection that died mid-answer never sent the sentinel, so synthesising one
    turned a truncated stream into an accepted sample.
    """
    transport = FixtureTransport(
        FixtureBundle.model_validate(
            {
                "name": "cut",
                "capabilities": Capabilities(
                    streaming=True,
                    tool_calls=False,
                    reasoning=False,
                    usage_reporting=False,
                    logprobs=False,
                ).model_dump(),
                "exchanges": [
                    {
                        "endpoint_id": "alias-a",
                        "benchmark_id": "ifeval",
                        "item_id": "item-1",
                        "response": "half an answer",
                        "stream_terminated": False,
                        "stream_frames": [
                            {"choices": [{"delta": {"content": "half "}}]},
                            {"choices": [{"delta": {"content": "an answer"}}]},
                        ],
                    }
                ],
            }
        )
    )
    outcome = transport.stream(
        sample_key=key(),
        request=ModelRequest(messages=[{"role": "user", "content": "hi"}], max_output_tokens=8),
        prompt_hash="a" * 64,
    )
    assert not outcome.ok
    assert outcome.result is None, "a truncated capture must not become an accepted sample"
    assert outcome.failure is not None
    assert outcome.failure.kind is FailureKind.INTERRUPTED


def test_a_terminated_capture_is_still_accepted() -> None:
    """The control for the test above: the sentinel, not the frame count, decides."""
    transport = FixtureTransport(
        FixtureBundle.model_validate(
            {
                "name": "whole",
                "capabilities": Capabilities(
                    streaming=True,
                    tool_calls=False,
                    reasoning=False,
                    usage_reporting=False,
                    logprobs=False,
                ).model_dump(),
                "exchanges": [
                    {
                        "endpoint_id": "alias-a",
                        "benchmark_id": "ifeval",
                        "item_id": "item-1",
                        "response": "whole answer",
                        "stream_frames": [{"choices": [{"delta": {"content": "whole answer"}}]}],
                    }
                ],
            }
        )
    )
    outcome = transport.stream(
        sample_key=key(),
        request=ModelRequest(messages=[{"role": "user", "content": "hi"}], max_output_tokens=8),
        prompt_hash="a" * 64,
    )
    assert outcome.ok
    assert outcome.result is not None
    assert outcome.result.response == "whole answer"


def test_a_declared_secret_reaches_no_part_of_a_streamed_failure() -> None:
    """MAJOR: extra_secrets was not forwarded to the streaming parser at all."""
    blind = "ZZQdeclared-canary-7f3a2b9c4d1e"
    transport = FixtureTransport(
        FixtureBundle.model_validate(
            {
                "name": "leaky-stream",
                "capabilities": Capabilities(
                    streaming=True,
                    tool_calls=False,
                    reasoning=False,
                    usage_reporting=False,
                    logprobs=False,
                ).model_dump(),
                "exchanges": [
                    {
                        "endpoint_id": "alias-a",
                        "benchmark_id": "ifeval",
                        "item_id": "item-1",
                        "response": "x",
                        "stream_frames": [{"error": {"message": f"rejected {blind}"}}],
                    }
                ],
            }
        ),
        extra_secrets=frozenset({blind}),
    )
    outcome = transport.stream(
        sample_key=key(),
        request=ModelRequest(messages=[{"role": "user", "content": "hi"}], max_output_tokens=8),
        prompt_hash="a" * 64,
    )
    assert outcome.failure is not None
    assert blind not in outcome.failure.detail
    assert blind not in json.dumps(outcome.to_dict())


def test_an_undeclared_opaque_value_survives_a_streamed_failure() -> None:
    """The control: without declaring it, the value is kept, so the test above bites."""
    blind = "ZZQdeclared-canary-7f3a2b9c4d1e"
    body = f'data: {{"error":{{"message":"rejected {blind}"}}}}\n\n'.encode()
    outcome, _ = parse_stream([body], sample_key=key(), route="zen")
    assert outcome.failure is not None
    assert blind in outcome.failure.detail


def test_a_crlf_split_between_cr_and_lf_does_not_split_a_record() -> None:
    """MAJOR: an eager CR conversion manufactured a blank line the wire never sent."""
    record = (
        b'data: {"choices":[{"delta":\r\ndata: {"content":"SPLIT-LOSS"}}]}'
        b"\r\n\r\ndata: [DONE]\r\n\r\n"
    )
    cut = record.index(b"\r\n") + 1
    outcome, _ = parse_stream([record[:cut], record[cut:]], sample_key=key(), route="zen")
    assert outcome.result is not None
    assert outcome.result.response == "SPLIT-LOSS"


@pytest.mark.parametrize("cut", range(1, 40))
def test_no_read_boundary_can_change_a_crlf_stream(cut: int) -> None:
    """Every split of the same bytes must give the same answer."""
    record = (
        b'data: {"choices":[{"delta":\r\ndata: {"content":"PARTS"}}]}\r\n\r\ndata: [DONE]\r\n\r\n'
    )
    if cut >= len(record):
        pytest.skip("offset past the end of the capture")
    outcome, _ = parse_stream([record[:cut], record[cut:]], sample_key=key(), route="zen")
    whole, _ = parse_stream([record], sample_key=key(), route="zen")
    assert (outcome.result.response if outcome.result else None) == (
        whole.result.response if whole.result else None
    )


def test_usage_reported_on_the_terminal_record_is_kept() -> None:
    """MODERATE: a billed count vanished because the finish branch won."""
    body = (
        b'data: {"choices":[{"delta":{"content":"hi"},"finish_reason":"stop"}],'
        b'"usage":{"input_tokens":4,"output_tokens":9}}\n\n'
        b"data: [DONE]\n\n"
    )
    outcome, _ = parse_stream([body], sample_key=key(), route="zen")
    assert outcome.result is not None
    assert outcome.result.usage.input_tokens == 4
    assert outcome.result.usage.output_tokens == 9
    assert outcome.result.usage.provider_reported is True


def test_usage_split_across_two_frames_is_not_half_lost() -> None:
    body = (
        b'data: {"choices":[],"usage":{"input_tokens":11}}\n\n'
        b'data: {"choices":[],"usage":{"output_tokens":7}}\n\n'
        b"data: [DONE]\n\n"
    )
    outcome, _ = parse_stream([body], sample_key=key(), route="zen")
    assert outcome.result is not None
    assert outcome.result.usage.input_tokens == 11
    assert outcome.result.usage.output_tokens == 7


def test_a_content_parts_delta_is_not_silently_lost() -> None:
    """MODERATE: the streamed path dropped a shape the non-streamed path reads."""
    parts = [{"type": "text", "text": "Paris "}, {"type": "text", "text": "France."}]
    chunk = json.dumps({"choices": [{"delta": {"content": parts}, "finish_reason": None}]})
    finish = json.dumps({"choices": [{"delta": {"content": []}, "finish_reason": "stop"}]})
    body = b"data: " + chunk.encode() + b"\n\n" + b"data: " + finish.encode() + b"\n\n"
    body += b"data: [DONE]\n\n"
    outcome, _ = parse_stream([body], sample_key=key(), route="zen")
    assert outcome.result is not None
    assert outcome.result.response == "Paris France."


def test_recorded_unsupported_settings_are_reported_when_streaming() -> None:
    """MODERATE: the streamed path silently ignored what the capture declined."""
    caps = Capabilities(
        streaming=True, tool_calls=False, reasoning=False, usage_reporting=False, logprobs=False
    )
    transport = FixtureTransport(
        FixtureBundle.model_validate(
            {
                "name": "declines",
                "capabilities": caps.model_dump(),
                "exchanges": [
                    {
                        "endpoint_id": "alias-a",
                        "benchmark_id": "ifeval",
                        "item_id": "item-1",
                        "response": "x",
                        "unsupported_settings": ["top_p"],
                        "stream_frames": [{"choices": [{"delta": {"content": "x"}}]}],
                    }
                ],
            }
        )
    )
    outcome = transport.stream(
        sample_key=key(),
        request=ModelRequest(
            messages=[{"role": "user", "content": "hi"}],
            max_output_tokens=8,
            top_p=0.5,
        ),
        prompt_hash="a" * 64,
        capabilities=caps,
    )
    assert [item.setting for item in outcome.unsupported] == ["top_p"]
    assert outcome.unsupported[0].requested == 0.5


def test_a_corrupt_byte_does_not_stall_the_rest_of_the_stream() -> None:
    """MODERATE: one undecodable byte used to hold back every later byte.

    The granularity matters as much as the content: a real adapter times the reads,
    and text that only appears at end of stream would be recorded as arriving then.
    """
    pieces = list(decode_stream_bytes([b"hello ", b"\xff", b"world ", b"again"]))
    assert pieces == ["hello ", "\ufffd", "world ", "again"], pieces


def test_a_leading_corrupt_byte_still_lets_the_rest_through() -> None:
    """The case that used to swallow everything: no decodable prefix at all."""
    pieces = list(decode_stream_bytes([b"\xffsecond part", b"third part"]))
    assert pieces == ["\ufffd", "second part", "third part"], pieces


def test_a_truncated_multi_byte_character_is_still_held_back() -> None:
    """The other half of the fix: a genuinely incomplete sequence must be recombined."""
    assert "".join(decode_stream_bytes([b"h\xc3", b"\xa9llo"])) == "héllo"
    assert "".join(decode_stream_bytes([b"a\xf0\x9f", b"\x98\x80b"])) == "a\U0001f600b"


def test_a_truncated_tail_at_end_of_stream_is_surfaced_not_dropped() -> None:
    """Holding back applies between reads; at end of stream the tail is reported."""
    assert list(decode_stream_bytes([b"h\xc3"])) == ["h\ufffd"]


def test_a_stream_with_no_finish_reason_does_not_report_a_clean_stop() -> None:
    """MAJOR: an absent reason was mapped to ``stop``."""
    body = (
        b'data: {"choices":[{"delta":{"content":"cut?"},"finish_reason":null}]}\n\ndata: [DONE]\n\n'
    )
    outcome, _ = parse_stream([body], sample_key=key(), route="zen")
    assert outcome.result is not None
    assert outcome.result.finish_status is None


def test_usage_on_a_contentless_terminal_record_is_kept() -> None:
    """MODERATE: the finish branch used to drop usage on a record with no delta.

    A gateway may close with an empty record that carries both the reason and the
    usage, which is the only place the counts appear.
    """
    body = (
        b'data: {"choices":[{"delta":{"content":"answer"},"finish_reason":null}]}\n\n'
        b'data: {"choices":[{"delta":{},"finish_reason":"stop"}],'
        b'"usage":{"input_tokens":13,"output_tokens":5}}\n\n'
        b"data: [DONE]\n\n"
    )
    outcome, _ = parse_stream([body], sample_key=key(), route="zen")
    assert outcome.result is not None
    assert outcome.result.response == "answer"
    assert outcome.result.finish_status == "stop"
    assert outcome.result.usage.input_tokens == 13
    assert outcome.result.usage.output_tokens == 5


# ---------------------------------------------------------------------------
# Regression tests for the third G03 review
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("separator", ["\x85", "\u2028", "\u2029"])
def test_a_unicode_line_separator_inside_content_does_not_truncate_the_record(
    separator: str,
) -> None:
    """SSE terminates lines at CR and LF only; JSON permits these raw in a string.

    ``str.splitlines`` also breaks here, which truncated the record: the answer was
    lost and a mid-stream error frame was silently downgraded to an interruption.
    """
    payload = json.dumps(
        {"choices": [{"delta": {"content": f"A{separator}B"}, "finish_reason": "stop"}]},
        ensure_ascii=False,
    )
    body = f"data: {payload}\n\n".encode() + b"data: [DONE]\n\n"
    outcome, assembly = parse_stream([body], sample_key=key(), route="zen")
    assert assembly.content == f"A{separator}B"
    assert outcome.ok
    assert outcome.result is not None
    assert outcome.result.response == f"A{separator}B"


@pytest.mark.parametrize("separator", ["\x85", "\u2028", "\u2029"])
def test_a_unicode_line_separator_does_not_defeat_the_error_frame_guard(
    separator: str,
) -> None:
    payload = json.dumps(
        {"error": {"message": f"upstream{separator}overloaded"}}, ensure_ascii=False
    )
    outcome, _ = parse_stream([f"data: {payload}\n\n".encode()], sample_key=key(), route="zen")
    assert outcome.failure is not None
    assert outcome.failure.kind is FailureKind.PROTOCOL, "the endpoint reported an error"
    assert "overloaded" in outcome.failure.detail


def test_crlf_framing_still_splits_after_the_line_splitter_change() -> None:
    from stealthbench.adapters.streaming import _sse_lines

    assert _sse_lines("a\r\nb\rc\nd") == ["a", "b", "c", "d"]


@pytest.mark.parametrize("offset", [5.0, -1.0, float("nan"), float("inf")])
def test_an_implausible_pacing_offset_is_dropped_rather_than_raising(offset: float) -> None:
    """A caller-supplied pacing function must not take the campaign down."""
    events = [
        StreamEvent(kind=StreamEventKind.DELTA, content_delta="x"),
        StreamEvent(kind=StreamEventKind.DONE),
    ]
    assembly = assemble(events, total_seconds=1.0, pacing=lambda _i, _e: offset)
    outcome = streamed_result(
        sample_key=key(), attempt_id="a", attempt_number=1, assembly=assembly, route="zen"
    )
    assert outcome.ok
    assert outcome.result is not None
    measurements = outcome.result.streaming
    assert measurements is not None
    assert measurements.first_content_seconds is None, offset


def test_an_offset_past_the_measured_total_is_dropped() -> None:
    events = [
        StreamEvent(kind=StreamEventKind.DELTA, content_delta="x"),
        StreamEvent(kind=StreamEventKind.DONE),
    ]
    assembly = assemble(events, total_seconds=1.0, pacing=lambda _i, _e: 5.0)
    outcome = streamed_result(
        sample_key=key(), attempt_id="a", attempt_number=1, assembly=assembly, route="zen"
    )
    assert outcome.result is not None
    measurements = outcome.result.streaming
    assert measurements is not None
    assert measurements.first_content_seconds is None
    assert measurements.total_seconds == 1.0


def test_content_cannot_be_recorded_after_the_first_answer() -> None:
    """An incoherent pair is not stored; the impossible half is dropped instead.

    Two content events with the offsets the pacing function reports: first content at
    0.9, and the last content at 0.1, so first answer would precede first content.
    """
    events = [
        StreamEvent(kind=StreamEventKind.DELTA, content_delta="a"),
        StreamEvent(kind=StreamEventKind.DELTA, content_delta="b"),
        StreamEvent(kind=StreamEventKind.DONE),
    ]
    assembly = assemble(events, total_seconds=1.0, pacing=lambda index, _e: (0.9, 0.1, 0.95)[index])
    # The ordering is impossible before streamed_result ever sees it.
    assert assembly.first_content_seconds == 0.9
    assert assembly.first_answer_seconds == 0.1
    outcome = streamed_result(
        sample_key=key(), attempt_id="a", attempt_number=1, assembly=assembly, route="zen"
    )
    assert outcome.ok
    assert outcome.result is not None
    measurements = outcome.result.streaming
    assert measurements is not None
    assert measurements.first_content_seconds is None, "the impossible half is dropped"
    assert measurements.first_answer_seconds == 0.1


def test_a_coherent_timing_pair_is_kept_whole() -> None:
    """The control: ordinary increasing offsets are recorded, not discarded."""
    events = [
        StreamEvent(kind=StreamEventKind.DELTA, content_delta="a"),
        StreamEvent(kind=StreamEventKind.DELTA, content_delta="b"),
        StreamEvent(kind=StreamEventKind.DONE),
    ]
    assembly = assemble(events, total_seconds=1.0, pacing=lambda index, _e: (0.1, 0.4, 0.9)[index])
    outcome = streamed_result(
        sample_key=key(), attempt_id="a", attempt_number=1, assembly=assembly, route="zen"
    )
    measurements = outcome.result.streaming
    assert measurements is not None
    assert measurements.first_content_seconds == 0.1
    assert measurements.first_answer_seconds == 0.4


@pytest.mark.parametrize(
    "usage_frame",
    [{}, {"foo": 1}, {"input_tokens": None, "output_tokens": None}],
    ids=["empty", "unrecognised", "explicit-nulls"],
)
def test_a_usage_block_with_no_counts_is_not_treated_as_an_answer(
    usage_frame: dict[str, object],
) -> None:
    """`usage: {}` is a block, not a measurement, and must not authorise a sample."""
    body = ("data: " + json.dumps({"usage": usage_frame}) + "\n\n").encode()
    body += b"data: [DONE]\n\n"
    outcome, _ = parse_stream([body], sample_key=key(), route="zen")
    assert not outcome.ok, "an empty usage block is not an answer"
    assert outcome.result is None


def test_a_reasoning_only_capture_is_delivered_rather_than_invented_away() -> None:
    """Reasoning was billed and did arrive; the grader decides whether it answers."""
    reasoning = json.dumps({"choices": [{"delta": {"reasoning_content": "thinking"}}]})
    finish = json.dumps({"choices": [{"delta": {}, "finish_reason": "stop"}]})
    body = f"data: {reasoning}\n\n".encode() + f"data: {finish}\n\n".encode() + b"data: [DONE]\n\n"
    outcome, assembly = parse_stream([body], sample_key=key(), route="zen")
    assert assembly.reasoning == "thinking"
    assert outcome.ok
    assert outcome.result is not None
    assert outcome.result.response == ""
    assert outcome.result.streaming is not None


def test_a_first_reported_token_count_is_not_overwritten_by_a_terminal_zero() -> None:
    """Last-wins would turn a reported 100 into the forbidden zero."""
    body = (
        b'data: {"choices":[],"usage":{"output_tokens":100}}\n\n'
        b'data: {"choices":[{"delta":{},"finish_reason":"stop"}],'
        b'"usage":{"output_tokens":0}}\n\n'
        b"data: [DONE]\n\n"
    )
    outcome, _ = parse_stream([body], sample_key=key(), route="zen")
    assert outcome.result is not None
    assert outcome.result.usage.output_tokens == 100


def test_usage_still_fills_in_from_a_later_frame_when_the_first_is_silent() -> None:
    body = (
        b'data: {"choices":[],"usage":{"input_tokens":11}}\n\n'
        b'data: {"choices":[{"delta":{"content":"a"},"finish_reason":"stop"}],'
        b'"usage":{"output_tokens":7}}\n\n'
        b"data: [DONE]\n\n"
    )
    outcome, _ = parse_stream([body], sample_key=key(), route="zen")
    assert outcome.result is not None
    assert outcome.result.usage.input_tokens == 11
    assert outcome.result.usage.output_tokens == 7


def test_a_terminated_capture_with_no_usable_frames_is_refused() -> None:
    """A sentinel and nothing else is not a completed empty answer."""
    outcome, _ = parse_stream([b"data: [DONE]\n\n"], sample_key=key(), route="zen")
    assert not outcome.ok
    assert outcome.result is None
    assert outcome.failure is not None
    assert outcome.failure.kind is FailureKind.PROTOCOL


def test_a_capture_whose_frames_are_all_empty_objects_is_refused() -> None:
    body = b"data: {}\n\ndata: {}\n\ndata: [DONE]\n\n"
    outcome, _ = parse_stream([body], sample_key=key(), route="zen")
    assert not outcome.ok
    assert outcome.failure is not None


def test_a_usage_only_capture_is_still_accepted() -> None:
    """Reported usage is content for this purpose: the endpoint did answer."""
    body = b'data: {"choices":[],"usage":{"input_tokens":3,"output_tokens":1}}\n\ndata: [DONE]\n\n'
    outcome, _ = parse_stream([body], sample_key=key(), route="zen")
    assert outcome.ok
    assert outcome.result is not None
    assert outcome.result.usage.output_tokens == 1


# ---------------------------------------------------------------------------
# Coverage of paths that no earlier test exercised
# ---------------------------------------------------------------------------


def test_a_corrupt_byte_in_the_middle_of_a_buffer_keeps_the_rest() -> None:
    """The branch that emits the valid text before a corrupt byte."""
    pieces = list(decode_stream_bytes([b"before \xff after"]))
    assert "".join(pieces) == "before \ufffd after", pieces
    assert pieces[0] == "before ", "the valid prefix is emitted on its own"


def test_a_stream_ending_on_a_bare_cr_flushes_its_last_record() -> None:
    """A held-back CR is itself a line terminator, so flush() must retire it."""
    body = b'data: {"choices":[{"delta":{"content":"TAIL"}}]}\r\n\r\ndata: [DONE]\r\n\r'
    outcome, _ = parse_stream([body], sample_key=key(), route="zen")
    assert outcome.result is not None
    assert outcome.result.response == "TAIL"


def test_a_record_with_no_blank_line_terminator_is_flushed() -> None:
    """A cut mid-record still yields whatever the record actually contained."""
    body = b'data: {"choices":[{"delta":{"content":"CUT"}}]}'
    outcome, assembly = parse_stream([body], sample_key=key(), route="zen")
    assert assembly.content == "CUT"
    assert not assembly.saw_done
    assert outcome.failure is not None
    assert outcome.failure.kind is FailureKind.INTERRUPTED


def test_a_stream_parsed_with_no_sample_key_is_a_protocol_failure() -> None:
    """Without a sample key there is nothing to attach a result to."""
    body = (
        b'data: {"choices":[{"delta":{"content":"x"},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n'
    )
    outcome, assembly = parse_stream([body], route="zen")
    assert not outcome.ok
    assert outcome.failure is not None
    assert outcome.failure.kind is FailureKind.PROTOCOL
    assert "sample key" in outcome.failure.detail
    assert assembly.content == "x", "what arrived is still assembled"


def test_content_after_the_sentinel_is_discarded_even_in_a_later_read() -> None:
    """The parser's seen-done guard: a second read cannot revive a finished stream."""
    done = b"data: [DONE]\n\n"
    outcome, assembly = parse_stream(
        [done, b'data: {"choices":[{"delta":{"content":"after"}}]}\n\n'], sample_key=key()
    )
    assert assembly.content == ""
    assert assembly.saw_done
    assert not outcome.ok, "nothing but the sentinel is not an answer"


def test_a_content_parts_delta_of_only_non_text_parts_is_empty() -> None:
    payload = json.dumps({"choices": [{"delta": {"content": [{"type": "image"}]}}]})
    body = f"data: {payload}\n\n".encode() + b"data: [DONE]\n\n"
    outcome, assembly = parse_stream([body], sample_key=key(), route="zen")
    assert assembly.content == ""
    # No content and no usage, so this is refused rather than accepted empty.
    assert outcome.failure is not None


def test_a_terminal_record_with_neither_delta_nor_reason_nor_usage_is_not_a_finish() -> None:
    payload = json.dumps({"choices": [{"delta": {}, "finish_reason": None}]})
    _outcome, assembly = parse_stream(
        [f"data: {payload}\n\n".encode() + b"data: [DONE]\n\n"], sample_key=key()
    )
    assert assembly.finish_reason is None


def test_a_content_parts_delta_ignores_a_part_that_is_not_a_mapping() -> None:
    parts = ["a string, not a part", {"text": "kept"}, {"no_text_key": 1}]
    payload = json.dumps({"choices": [{"delta": {"content": parts}}]})
    body = f"data: {payload}\n\n".encode() + b"data: [DONE]\n\n"
    _outcome, assembly = parse_stream([body], sample_key=key(), route="zen")
    assert assembly.content == "kept"


def test_a_chunk_with_no_delta_object_is_still_classified() -> None:
    """A choice with neither `delta` nor `content` must not raise."""
    payload = json.dumps({"choices": [{"finish_reason": "stop"}]})
    event = classify_chunk(json.loads(payload))
    assert event.kind is StreamEventKind.FINISH
    assert event.finish_reason == "stop"


def test_a_usage_only_chunk_with_a_choice_is_still_reported_as_usage() -> None:
    """Usage riding alongside a choice is a measurement, not an answer."""
    payload = {
        "choices": [{"delta": {"content": "x"}, "finish_reason": None}],
        "usage": {"output_tokens": 5},
    }
    event = classify_chunk(payload)
    assert event.content_delta == "x"
    assert event.usage.output_tokens == 5


def test_a_chunk_with_choices_and_usage_but_no_delta_reports_the_usage() -> None:
    """Usage alongside a contentless choice is a measurement, not an unknown record."""
    payload = {"choices": [{"finish_reason": None}], "usage": {"output_tokens": 6}}
    event = classify_chunk(payload)
    assert event.kind is StreamEventKind.USAGE
    assert event.usage.output_tokens == 6
