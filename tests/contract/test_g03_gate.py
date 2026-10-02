"""G03 gate: adapter output matches the contract (tasks T03A, T03B, T03C).

Gate criteria, each asserted directly:

* adapter output matches the result contract
* chunks are not counted as tokens
* error bodies are redacted
* unsupported parameters are reported
* production adapter behaviour is demonstrated against fixture protocol transcripts
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from stealthbench.adapters.base import (
    AdapterResult,
    FailureKind,
    FixtureBundle,
    FixtureTransport,
    ProviderAdapter,
    UnsupportedSetting,
    check_requested_settings,
)
from stealthbench.adapters.streaming import parse_stream
from stealthbench.adapters.zen import ZenAdapter, normalize_catalog, usage_from_response
from stealthbench.schemas.campaign import Capabilities
from stealthbench.schemas.results import (
    DeliveryStatus,
    GenerationResult,
    ModelRequest,
    SampleKey,
    accepted_sample_keys,
)

pytestmark = pytest.mark.contract

CANARY = "sk-canary-g03-0123456789abcdef"
NO_CAPS = Capabilities(
    streaming=False, tool_calls=False, reasoning=False, usage_reporting=False, logprobs=False
)


def key(task: str = "ifeval::item-1", endpoint: str = "alias-a") -> SampleKey:
    return SampleKey(campaign_id="c1", endpoint_id=endpoint, task_id=task, repeat_id=1)


#: The alias the committed capture records its exchanges under.
CAPTURED_ALIAS = "zen-fast-alias"


def captured_key(task: str) -> SampleKey:
    return key(task, endpoint=CAPTURED_ALIAS)


def request_(**overrides: object) -> ModelRequest:
    payload: dict[str, object] = {
        "messages": [{"role": "user", "content": "hello"}],
        "max_output_tokens": 64,
    }
    payload.update(overrides)
    return ModelRequest.model_validate(payload)


def chat(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "id": "cmpl-1",
        "model": "alias-a",
        "choices": [{"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2},
    }
    payload.update(overrides)
    return payload


def fixture(**overrides: object) -> FixtureTransport:
    payload: dict[str, object] = {
        "name": "gate",
        "capabilities": NO_CAPS.model_dump(),
        "exchanges": [
            {
                "endpoint_id": "alias-a",
                "benchmark_id": "ifeval",
                "item_id": "item-1",
                "response": "hi",
                "usage": {"input_tokens": 5, "output_tokens": 2},
            }
        ],
    }
    payload.update(overrides)
    return FixtureTransport(FixtureBundle.model_validate(payload))


# ---------------------------------------------------------------------------
# Criterion: adapter output matches the result contract
# ---------------------------------------------------------------------------


def _adapters() -> list[FixtureTransport | ZenAdapter]:
    return [
        fixture(),
        ZenAdapter(
            catalog_payload={"data": [{"id": "alias-a"}]},
            exchanges={"ifeval::item-1": {"http_status": 200, "json": chat()}},
            endpoint_id="alias-a",
        ),
    ]


@pytest.mark.parametrize("index", [0, 1])
def test_every_adapter_produces_a_contract_valid_result(index: int) -> None:
    outcome = _adapters()[index].complete(
        sample_key=key(), request=request_(), prompt_hash="a" * 64
    )
    assert outcome.ok
    assert outcome.result is not None
    # The result must satisfy the frozen contract, not merely look plausible.
    GenerationResult.model_validate(outcome.result.model_dump(mode="json"))
    assert outcome.result.delivery_status is DeliveryStatus.ACCEPTED
    assert outcome.result.manifest_hash is None


@pytest.mark.parametrize("index", [0, 1])
def test_every_adapter_reports_failures_rather_than_raising(index: int) -> None:
    outcome = _adapters()[index].complete(
        sample_key=key("ifeval::absent"), request=request_(), prompt_hash="a" * 64
    )
    assert not outcome.ok
    assert outcome.result is None
    assert outcome.failure is not None


@pytest.mark.parametrize("index", [0, 1])
def test_adapter_outcomes_are_serialisable(index: int) -> None:
    outcome = _adapters()[index].complete(
        sample_key=key(), request=request_(), prompt_hash="a" * 64
    )
    json.dumps(outcome.to_dict())


def test_an_adapter_result_counts_as_at_most_one_sample() -> None:
    results = []
    for adapter in _adapters():
        results.append(
            adapter.complete(sample_key=key(), request=request_(), prompt_hash="a" * 64).result
        )
    assert len(accepted_sample_keys([r for r in results if r is not None])) == 1


# ---------------------------------------------------------------------------
# Criterion: chunks are not counted as tokens
# ---------------------------------------------------------------------------


def test_no_adapter_derives_tokens_from_chunks() -> None:
    """A stream of 200 chunks must not produce a token count of 200."""
    body = b""
    for _ in range(200):
        body += b'data: {"choices":[{"delta":{"content":"x"},"finish_reason":null}]}\n\n'
    body += b'data: {"choices":[],"usage":{"input_tokens":1,"output_tokens":2}}\n\n'
    body += b"data: [DONE]\n\n"

    outcome, assembly = parse_stream([body], total_seconds=10.0, sample_key=key())
    assert assembly.chunk_count == 202
    assert outcome.result is not None
    assert outcome.result.usage.output_tokens == 2, "only the reported count is a token count"
    assert assembly.token_rate() == 0.2


def test_chunk_count_is_recorded_separately_from_usage() -> None:
    body = b'data: {"choices":[{"delta":{"content":"x"},"finish_reason":null}]}\n\ndata: [DONE]\n\n'
    outcome, assembly = parse_stream([body], sample_key=key())
    assert outcome.result is not None
    assert outcome.result.streaming is not None
    assert outcome.result.streaming.chunk_count == assembly.chunk_count
    assert outcome.result.usage.output_tokens is None


# ---------------------------------------------------------------------------
# Criterion: error bodies are redacted
# ---------------------------------------------------------------------------


def test_a_recorded_error_body_and_detail_are_both_redacted() -> None:
    transport = FixtureTransport(
        FixtureBundle.model_validate(
            {
                "name": "leaky",
                "capabilities": NO_CAPS.model_dump(),
                "exchanges": [
                    {
                        "endpoint_id": "alias-a",
                        "benchmark_id": "ifeval",
                        "item_id": "item-1",
                        "outcome": "error",
                        "failure_kind": "authentication",
                        "http_status": 401,
                        "failure_detail": f"key {CANARY} rejected",
                        "error_body": {"error": {"message": f"key {CANARY} rejected"}},
                    }
                ],
            }
        ),
        extra_secrets=frozenset({CANARY}),
    )
    outcome = transport.complete(sample_key=key(), request=request_(), prompt_hash="a" * 64)
    assert CANARY not in outcome.failure.detail
    assert CANARY not in json.dumps(outcome.failure.to_dict())
    assert CANARY not in json.dumps(outcome.to_dict())


def test_a_zen_error_body_and_detail_are_both_redacted() -> None:
    adapter = ZenAdapter(
        exchanges={
            "ifeval::item-1": {
                "http_status": 500,
                "body": {"error": {"message": f"upstream rejected {CANARY}"}},
            }
        },
        extra_secrets=frozenset({CANARY}),
    )
    outcome = adapter.complete(sample_key=key(), request=request_(), prompt_hash="a" * 64)
    assert CANARY not in outcome.failure.detail
    assert CANARY not in json.dumps(outcome.to_dict())


def test_no_adapter_leaks_a_credential_through_a_known_header() -> None:
    from stealthbench.storage.events import REDACTED

    adapter = ZenAdapter(
        endpoint_id="alias-a",
        exchanges={
            "ifeval::item-1": {
                "http_status": 403,
                "headers": {"authorization": "opaque"},
                "body": {"headers": {"Authorization": "opaque", "accept": "application/json"}},
            }
        },
    )
    outcome = adapter.complete(sample_key=key(), request=request_(), prompt_hash="a" * 64)
    body = outcome.failure.body
    assert body is not None
    assert body["headers"]["Authorization"] == REDACTED
    assert body["headers"]["accept"] == "application/json"


# ---------------------------------------------------------------------------
# Criterion: unsupported parameters are reported
# ---------------------------------------------------------------------------


def test_an_unsupported_setting_is_reported_not_ignored() -> None:
    transport = FixtureTransport(
        FixtureBundle.model_validate(
            {
                "name": "unsupported",
                "capabilities": NO_CAPS.model_dump(),
                "exchanges": [
                    {
                        "endpoint_id": "alias-a",
                        "benchmark_id": "ifeval",
                        "item_id": "item-1",
                        "response": "hi",
                        "unsupported_settings": ["top_p"],
                    }
                ],
            }
        )
    )
    outcome = transport.complete(
        sample_key=key(), request=request_(top_p=0.5), prompt_hash="a" * 64
    )
    assert [item.setting for item in outcome.unsupported] == ["top_p"]
    assert outcome.unsupported[0].requested == 0.5


def test_streaming_on_an_endpoint_that_cannot_stream_is_reported() -> None:
    reported = check_requested_settings(request_(), NO_CAPS, stream=True)
    assert [item.setting for item in reported] == ["stream"]


def test_an_adapter_refuses_to_fake_a_stream() -> None:
    """A non-streamed result labelled as streamed would corrupt every timing metric."""
    assert not _adapters()[0].stream(sample_key=key(), request=request_(), prompt_hash="a" * 64).ok


def test_unsupported_settings_are_serialisable() -> None:
    item = UnsupportedSetting(setting="top_p", requested=0.5, reason="not offered")
    assert item.to_dict()["setting"] == "top_p"


# ---------------------------------------------------------------------------
# Criterion: production adapter behaviour is demonstrated against transcripts
# ---------------------------------------------------------------------------


TRANSCRIPT_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "zen"


def test_the_transcript_fixtures_are_committed_protocol_capture() -> None:
    """The capture exists on disk, in wire form, not only as inline test dictionaries."""
    expected = {
        "chat.completions.json",
        "chat.completions.stream.sse",
        "chat.completions.truncated.sse",
        "chat.completions.server_error.sse",
    }
    assert expected <= {f.name for f in TRANSCRIPT_DIR.iterdir()}

    sse = (TRANSCRIPT_DIR / "chat.completions.stream.sse").read_bytes()
    assert b"\r\n\r\n" in sse, "a real capture frames records with CRLF"
    assert b"\n\n" not in sse.replace(b"\r\n", b""), "no bare-LF framing anywhere"
    assert sse.rstrip().endswith(b"data: [DONE]")

    body = json.loads((TRANSCRIPT_DIR / "chat.completions.json").read_text(encoding="utf-8"))
    assert [r["method"] for r in body["requests"]] == ["GET", "POST", "POST", "POST"]
    assert [r["status"] for r in body["requests"]] == [200, 200, 401, 429]


def test_the_zen_adapter_runs_the_production_path_over_a_transcript() -> None:
    """A captured transcript on disk drives the real adapter, not a test double."""
    adapter = ZenAdapter.from_path(TRANSCRIPT_DIR / "chat.completions.json")
    snapshot = adapter.discover()
    assert snapshot.aliases() == ("zen-fast-alias", "zen-reason-alias", "zen-legacy-alias")
    assert snapshot.get("zen-fast-alias").capabilities.streaming is True
    assert snapshot.get("zen-reason-alias").capabilities.reasoning is True
    # The third alias advertises nothing; the snapshot must not fill it in.
    assert snapshot.get("zen-legacy-alias").capabilities.streaming is False

    outcome = adapter.complete(
        sample_key=captured_key("ifeval::capital-of-france"),
        request=request_(),
        prompt_hash="a" * 64,
    )
    assert outcome.ok
    assert outcome.result is not None
    assert outcome.result.response == "Paris is the capital of France."
    assert outcome.result.usage.input_tokens == 24
    assert outcome.result.usage.output_tokens == 7
    assert outcome.result.redacted_provider_metadata["route"] == "zen"


def test_the_captured_catalog_is_digestible_for_storage() -> None:
    from stealthbench.adapters.zen import catalog_snapshot_digest

    snapshot = ZenAdapter.from_path(TRANSCRIPT_DIR / "chat.completions.json").discover()
    assert len(catalog_snapshot_digest(snapshot)) == 64


def test_the_captured_auth_header_never_reaches_an_artifact() -> None:
    """The capture itself contains a credential-shaped value; output must not."""
    adapter = ZenAdapter.from_path(TRANSCRIPT_DIR / "chat.completions.json", extra_secrets=CANARY)
    for task in ("ifeval::capital-of-france", "ifeval::forbidden-keyword", "ifeval::rate-limited"):
        outcome = adapter.complete(sample_key=key(task), request=request_(), prompt_hash="a" * 64)
        assert CANARY not in json.dumps(outcome.to_dict()), task


def test_the_captured_error_statuses_map_to_the_right_failure_kinds() -> None:
    from stealthbench.adapters.base import FailureKind

    adapter = ZenAdapter.from_path(TRANSCRIPT_DIR / "chat.completions.json")
    unauthorized = adapter.complete(
        sample_key=captured_key("ifeval::forbidden-keyword"),
        request=request_(),
        prompt_hash="a" * 64,
    )
    assert unauthorized.failure is not None
    assert unauthorized.failure.kind is FailureKind.AUTHENTICATION

    limited = adapter.complete(
        sample_key=captured_key("ifeval::rate-limited"), request=request_(), prompt_hash="a" * 64
    )
    assert limited.failure is not None
    assert limited.failure.kind is FailureKind.RATE_LIMIT
    assert limited.failure.retry_after_seconds == 3.0


def test_the_captured_stream_is_replayed_by_the_zen_adapter() -> None:
    from stealthbench.schemas.campaign import Capabilities

    caps = Capabilities(
        streaming=True, tool_calls=False, reasoning=False, usage_reporting=True, logprobs=False
    )
    adapter = ZenAdapter.from_path(TRANSCRIPT_DIR / "chat.completions.json")
    outcome = adapter.stream(
        sample_key=captured_key("ifeval::streamed-answer"),
        request=request_(),
        prompt_hash="a" * 64,
        capabilities=caps,
    )
    assert outcome.ok
    assert outcome.result is not None
    assert outcome.result.response == "Paris is the capital of France."
    assert outcome.result.usage.output_tokens == 7
    assert outcome.result.streaming is not None
    assert outcome.result.streaming.chunk_count == 5


def test_the_captured_truncated_stream_is_not_reported_as_a_clean_stop() -> None:
    from stealthbench.adapters.streaming import parse_stream

    raw = (TRANSCRIPT_DIR / "chat.completions.truncated.sse").read_bytes()
    outcome, assembly = parse_stream([raw], sample_key=key(), route="zen")
    assert assembly.content == "The quick brown fox jumps over the lazy"
    assert outcome.result is not None
    assert outcome.result.finish_status == "length"


def test_a_captured_server_error_stream_is_reported_not_accepted() -> None:
    from stealthbench.adapters.base import FailureKind
    from stealthbench.adapters.streaming import parse_stream

    raw = (TRANSCRIPT_DIR / "chat.completions.server_error.sse").read_bytes()
    outcome, _ = parse_stream([raw], sample_key=key(), route="zen")
    assert not outcome.ok
    assert outcome.result is None
    assert outcome.failure is not None
    assert outcome.failure.kind is FailureKind.PROTOCOL


def test_every_failure_kind_the_plan_names_is_expressible() -> None:
    """The plan requires these failure paths to exist at all."""
    required = {
        FailureKind.AUTHENTICATION,
        FailureKind.PERMISSION,
        FailureKind.RATE_LIMIT,
        FailureKind.SERVER_ERROR,
        FailureKind.TIMEOUT,
        FailureKind.PROTOCOL,
        FailureKind.UNSUPPORTED_SETTING,
        FailureKind.INTERRUPTED,
    }
    assert required <= set(FailureKind)


def test_retryable_and_terminal_failures_are_distinguished() -> None:
    from stealthbench.adapters.base import RETRYABLE_FAILURES

    assert FailureKind.RATE_LIMIT in RETRYABLE_FAILURES
    assert FailureKind.SERVER_ERROR in RETRYABLE_FAILURES
    assert FailureKind.AUTHENTICATION not in RETRYABLE_FAILURES
    assert FailureKind.PROTOCOL not in RETRYABLE_FAILURES


def test_usage_absence_survives_every_adapter() -> None:
    """The single most important missing-value rule, across the whole adapter layer."""
    for adapter in _adapters():
        result = adapter.complete(sample_key=key(), request=request_(), prompt_hash="a" * 64).result
        assert result is not None
        assert result.usage.provider_reported is True

    bare = ZenAdapter(
        endpoint_id="alias-a",
        exchanges={
            "ifeval::item-1": {
                "http_status": 200,
                "json": {"choices": [{"message": {"content": "x"}, "finish_reason": "stop"}]},
            }
        },
    ).complete(sample_key=key(), request=request_(), prompt_hash="a" * 64)
    assert bare.result is not None
    assert bare.result.usage.input_tokens is None
    assert bare.result.usage.output_tokens is None
    assert json.loads(bare.result.model_dump_json())["usage"]["output_tokens"] is None


def test_the_adapter_base_class_cannot_be_instantiated() -> None:
    """An adapter must implement discovery; there is no partial implementation."""
    with pytest.raises(TypeError):
        ProviderAdapter()  # type: ignore[abstract]


def test_adapter_results_are_not_invented_for_unknown_aliases() -> None:
    snapshot = normalize_catalog({"data": []})
    assert snapshot.is_empty
    assert snapshot.to_endpoint_specs() == ()


def test_usage_from_an_empty_payload_is_absent_not_zero() -> None:
    usage = usage_from_response({})
    assert usage.input_tokens is None
    assert usage.output_tokens is None
    assert usage.provider_reported is False


def test_adapter_outcomes_require_exactly_one_outcome() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="either a generation result or a failure"):
        AdapterResult()


# ---------------------------------------------------------------------------
# Regression tests for the second G03 review
# ---------------------------------------------------------------------------


def test_the_committed_cut_capture_has_no_sentinel_and_is_refused() -> None:
    """A capture taken from a connection that died must not replay as a complete answer."""
    raw = (TRANSCRIPT_DIR / "chat.completions.cut.sse").read_bytes()
    assert b"data: [DONE]" not in raw, "the cut capture must genuinely lack the sentinel"
    assert raw.count(b"\r\n\r\n") == 2

    adapter = ZenAdapter.from_path(TRANSCRIPT_DIR / "chat.completions.json")
    outcome = adapter.stream(
        sample_key=captured_key("ifeval::cut-answer"),
        request=request_(),
        prompt_hash="a" * 64,
    )
    assert not outcome.ok
    assert outcome.result is None, "a truncated capture must not become an accepted sample"
    assert outcome.failure is not None
    assert outcome.failure.kind is FailureKind.INTERRUPTED


def test_the_committed_truncated_capture_still_carries_its_length_reason() -> None:
    raw = (TRANSCRIPT_DIR / "chat.completions.truncated.sse").read_bytes()
    assert raw.rstrip().endswith(b"data: [DONE]")
    outcome, _ = parse_stream([raw], sample_key=captured_key("ifeval::x"), route="zen")
    assert outcome.ok
    assert outcome.result is not None
    assert outcome.result.finish_status == "length"


def test_every_committed_capture_is_crlf_framed() -> None:
    """Not just the one the gate asserted before; all of them."""
    for name in sorted(TRANSCRIPT_DIR.glob("*.sse")):
        raw = name.read_bytes()
        assert b"\r\n\r\n" in raw, f"{name.name} has no CRLF-framed record"
        assert b"\n\n" not in raw.replace(b"\r\n", b""), f"{name.name} has bare-LF framing"
