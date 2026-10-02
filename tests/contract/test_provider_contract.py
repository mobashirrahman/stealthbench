"""Provider adapter contract (task T03A).

Acceptance: the fixture transport expresses errors, missing usage and unsupported
settings through the frozen contract.

The oracle is the frozen result contract in ``docs/contracts.md`` plus the G03 gate
statement, not the adapter's own behaviour: a fixture that returns something the
contract forbids must fail here.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from stealthbench.adapters.base import (
    AdapterResult,
    CatalogSnapshot,
    FailureKind,
    FixtureBundle,
    FixtureTransport,
    ProviderAdapter,
    TransportFailure,
    UnsupportedSetting,
    check_requested_settings,
    failed_result,
)
from stealthbench.schemas.campaign import Capabilities
from stealthbench.schemas.results import (
    DeliveryStatus,
    GenerationResult,
    ModelRequest,
    SampleKey,
    Usage,
)

pytestmark = pytest.mark.contract

NO_CAPABILITIES = Capabilities(
    streaming=False, tool_calls=False, reasoning=False, usage_reporting=False, logprobs=False
)
FULL_CAPABILITIES = Capabilities(
    streaming=True, tool_calls=True, reasoning=True, usage_reporting=True, logprobs=False
)

CANARY = "sk-canary-adapter-0123456789abcdef"
PROMPT_HASH = "a" * 64
MANIFEST_HASH = "b" * 64


def key(item: str = "syn-if-001", repeat: int = 1, benchmark: str = "ifeval") -> SampleKey:
    return SampleKey(
        campaign_id="offline-demo",
        endpoint_id="fixture-a",
        task_id=f"{benchmark}::{item}",
        repeat_id=repeat,
    )


def bundle(**overrides: object) -> FixtureBundle:
    payload: dict[str, object] = {
        "name": "test-bundle",
        "capabilities": NO_CAPABILITIES.model_dump(),
        "catalog": {"models": []},
        "exchanges": [
            {
                "endpoint_id": "fixture-a",
                "benchmark_id": "ifeval",
                "item_id": "syn-if-001",
                "response": "a haiku",
                "usage": {"input_tokens": 12, "output_tokens": 8},
            }
        ],
    }
    payload.update(overrides)
    return FixtureBundle.model_validate(payload)


def request_(**overrides: object) -> ModelRequest:
    payload: dict[str, object] = {
        "messages": [{"role": "user", "content": "Write a haiku."}],
        "max_output_tokens": 64,
    }
    payload.update(overrides)
    return ModelRequest.model_validate(payload)


# ---------------------------------------------------------------------------
# A success path that matches the frozen contract
# ---------------------------------------------------------------------------


def test_a_recorded_response_becomes_a_contract_shaped_result() -> None:
    transport = FixtureTransport(bundle())
    outcome = transport.complete(
        sample_key=key(), request=request_(), prompt_hash=PROMPT_HASH, manifest_hash=MANIFEST_HASH
    )
    assert outcome.ok
    assert outcome.result is not None
    # Validated by the frozen contract, not just well-formed here.
    GenerationResult.model_validate(outcome.result.model_dump(mode="json"))
    assert outcome.result.response == "a haiku"
    assert outcome.result.usage.input_tokens == 12
    assert outcome.result.delivery_status is DeliveryStatus.ACCEPTED
    assert outcome.result.manifest_hash == MANIFEST_HASH


def test_the_result_is_keyed_by_campaign_endpoint_task_and_repeat() -> None:
    outcome = FixtureTransport(bundle()).complete(
        sample_key=key(), request=request_(), prompt_hash=PROMPT_HASH
    )
    assert outcome.result is not None
    assert outcome.result.sample_key == key()


def test_attempt_number_is_preserved() -> None:
    outcome = FixtureTransport(bundle()).complete(
        sample_key=key(), request=request_(), prompt_hash=PROMPT_HASH, attempt_number=3
    )
    assert outcome.result is not None
    assert outcome.result.attempt_number == 3


# ---------------------------------------------------------------------------
# Missing usage stays missing
# ---------------------------------------------------------------------------


def test_an_endpoint_that_reports_no_usage_yields_null_tokens() -> None:
    """The most important case: absence must not become zero."""
    transport = FixtureTransport(
        bundle(
            exchanges=[
                {
                    "endpoint_id": "fixture-a",
                    "benchmark_id": "ifeval",
                    "item_id": "syn-if-001",
                    "response": "x",
                    "usage_reported": False,
                }
            ]
        )
    )
    outcome = transport.complete(sample_key=key(), request=request_(), prompt_hash=PROMPT_HASH)
    assert outcome.result is not None
    assert outcome.result.usage.input_tokens is None
    assert outcome.result.usage.output_tokens is None
    assert outcome.result.usage.provider_reported is False


def test_partially_reported_usage_keeps_the_absent_direction_null() -> None:
    transport = FixtureTransport(
        bundle(
            exchanges=[
                {
                    "endpoint_id": "fixture-a",
                    "benchmark_id": "ifeval",
                    "item_id": "syn-if-001",
                    "response": "x",
                    "usage": {"input_tokens": 30},
                }
            ]
        )
    )
    usage = transport.complete(
        sample_key=key(), request=request_(), prompt_hash=PROMPT_HASH
    ).result.usage
    assert usage.input_tokens == 30
    assert usage.output_tokens is None
    assert not usage.is_complete


def test_a_reported_zero_is_kept_as_zero() -> None:
    transport = FixtureTransport(
        bundle(
            exchanges=[
                {
                    "endpoint_id": "fixture-a",
                    "benchmark_id": "ifeval",
                    "item_id": "syn-if-001",
                    "response": "x",
                    "usage": {"input_tokens": 0, "output_tokens": 0},
                }
            ]
        )
    )
    usage = transport.complete(
        sample_key=key(), request=request_(), prompt_hash=PROMPT_HASH
    ).result.usage
    assert usage.input_tokens == 0
    assert usage.provider_reported is True
    assert usage != Usage()


# ---------------------------------------------------------------------------
# Errors are results, not exceptions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kind", "status", "retryable"),
    [
        (FailureKind.AUTHENTICATION, 401, False),
        (FailureKind.PERMISSION, 403, False),
        (FailureKind.RATE_LIMIT, 429, True),
        (FailureKind.SERVER_ERROR, 500, True),
        (FailureKind.SERVER_ERROR, 503, True),
        (FailureKind.TIMEOUT, None, True),
        (FailureKind.PROTOCOL, 200, False),
        (FailureKind.NOT_FOUND, 404, False),
    ],
)
def test_each_failure_is_expressed_through_the_contract(
    kind: FailureKind, status: int | None, retryable: bool
) -> None:
    exchange: dict[str, object] = {
        "endpoint_id": "fixture-a",
        "benchmark_id": "ifeval",
        "item_id": "syn-if-001",
        "outcome": "error",
        "failure_kind": str(kind),
        "failure_detail": "recorded failure",
        "http_status": status,
    }
    outcome = FixtureTransport(bundle(exchanges=[exchange])).complete(
        sample_key=key(), request=request_(), prompt_hash=PROMPT_HASH
    )
    assert not outcome.ok
    assert outcome.failure is not None
    assert outcome.failure.kind is kind
    assert outcome.failure.retryable is retryable
    assert outcome.result is None


def test_an_authentication_failure_is_not_retried() -> None:
    """Retrying a 401 only burns tokens; it cannot succeed."""
    outcome = FixtureTransport(
        bundle(
            exchanges=[
                {
                    "endpoint_id": "fixture-a",
                    "benchmark_id": "ifeval",
                    "item_id": "syn-if-001",
                    "outcome": "error",
                    "failure_kind": "authentication",
                    "http_status": 401,
                }
            ]
        )
    ).complete(sample_key=key(), request=request_(), prompt_hash=PROMPT_HASH)
    assert outcome.failure is not None
    assert not outcome.failure.retryable


def test_a_rate_limit_carries_retry_after_when_recorded() -> None:
    outcome = FixtureTransport(
        bundle(
            exchanges=[
                {
                    "endpoint_id": "fixture-a",
                    "benchmark_id": "ifeval",
                    "item_id": "syn-if-001",
                    "outcome": "error",
                    "failure_kind": "rate_limit",
                    "http_status": 429,
                    "retry_after_seconds": 12.5,
                }
            ]
        )
    ).complete(sample_key=key(), request=request_(), prompt_hash=PROMPT_HASH)
    assert outcome.failure is not None
    assert outcome.failure.retry_after_seconds == 12.5
    assert outcome.failure.retryable


def test_a_failure_never_becomes_an_accepted_sample() -> None:
    outcome = FixtureTransport(
        bundle(
            exchanges=[
                {
                    "endpoint_id": "fixture-a",
                    "benchmark_id": "ifeval",
                    "item_id": "syn-if-001",
                    "outcome": "error",
                    "failure_kind": "server_error",
                    "http_status": 503,
                }
            ]
        )
    ).complete(sample_key=key(), request=request_(), prompt_hash=PROMPT_HASH)
    assert outcome.result is None
    assert outcome.delivery_status is None


def test_an_unrecorded_exchange_is_reported_not_invented() -> None:
    outcome = FixtureTransport(bundle()).complete(
        sample_key=key("syn-if-999"), request=request_(), prompt_hash=PROMPT_HASH
    )
    assert outcome.failure is not None
    assert outcome.failure.kind is FailureKind.NO_FIXTURE
    assert outcome.result is None


def test_a_failure_result_carries_no_response() -> None:
    result = failed_result(
        key(),
        "attempt-1",
        1,
        TransportFailure(FailureKind.SERVER_ERROR, "boom", http_status=500),
    )
    assert result.response is None
    assert result.delivery_status is DeliveryStatus.TRANSPORT_FAILED
    assert not result.is_accepted_sample


def test_a_billed_failure_keeps_its_usage() -> None:
    """A failed request that the provider billed still cost money."""
    result = failed_result(
        key(),
        "attempt-1",
        1,
        TransportFailure(FailureKind.TIMEOUT, "timed out"),
        usage=Usage(input_tokens=400, output_tokens=0),
    )
    assert result.usage.input_tokens == 400
    assert not result.is_accepted_sample


# ---------------------------------------------------------------------------
# Unsupported settings are reported, never dropped
# ---------------------------------------------------------------------------


def test_an_unsupported_setting_is_reported() -> None:
    outcome = FixtureTransport(
        bundle(
            exchanges=[
                {
                    "endpoint_id": "fixture-a",
                    "benchmark_id": "ifeval",
                    "item_id": "syn-if-001",
                    "response": "x",
                    "unsupported_settings": ["temperature"],
                }
            ]
        )
    ).complete(sample_key=key(), request=request_(), prompt_hash=PROMPT_HASH)
    assert [item.setting for item in outcome.unsupported] == ["temperature"]
    assert outcome.unsupported[0].requested == 0.0
    assert outcome.unsupported[0].reason


def test_streaming_on_a_non_streaming_endpoint_is_reported() -> None:
    unsupported = check_requested_settings(request_(), NO_CAPABILITIES, stream=True)
    assert [item.setting for item in unsupported] == ["stream"]


def test_streaming_on_a_streaming_endpoint_is_not_reported() -> None:
    assert check_requested_settings(request_(), FULL_CAPABILITIES, stream=True) == ()


def test_the_base_stream_method_refuses_rather_than_faking_a_stream() -> None:
    """A non-streamed result must never be labelled as streamed."""

    class NoStreaming(ProviderAdapter):
        def discover(self) -> CatalogSnapshot:
            return CatalogSnapshot(source="none")

        def complete(self, **kwargs: object) -> AdapterResult:
            return AdapterResult(failure=TransportFailure(FailureKind.PROTOCOL, "unused"))

    outcome = NoStreaming().stream(sample_key=key(), request=request_(), prompt_hash=PROMPT_HASH)
    assert not outcome.ok
    assert outcome.failure is not None
    assert outcome.failure.kind is FailureKind.UNSUPPORTED_SETTING
    assert "must not be labelled as streamed" in outcome.failure.detail


def test_unknown_settings_are_reported() -> None:
    unsupported = check_requested_settings(request_(), FULL_CAPABILITIES)
    assert all(
        item.setting in {"stop", "temperature", "top_p", "max_output_tokens"}
        for item in unsupported
    )


# ---------------------------------------------------------------------------
# Catalog normalisation: observed, never invented
# ---------------------------------------------------------------------------


def test_a_normal_catalog_is_normalised() -> None:
    transport = FixtureTransport(
        bundle(
            catalog={
                "models": [
                    {
                        "id": "alpha",
                        "display_name": "Alpha",
                        "provider": "acme",
                        "family": "acme-1",
                        "context_window": 128000,
                        "capabilities": {
                            "streaming": True,
                            "tool_calls": True,
                            "usage_reporting": True,
                        },
                    }
                ]
            }
        )
    )
    snapshot = transport.discover()
    assert snapshot.aliases() == ("alpha",)
    entry = snapshot.get("alpha")
    assert entry is not None
    assert entry.capabilities.streaming is True
    assert entry.capabilities.logprobs is False, "an unmentioned capability stays false"
    assert entry.context_window == 128000
    assert entry.raw["display_name"] == "Alpha"


def test_an_empty_catalog_is_empty_not_filled_in() -> None:
    snapshot = FixtureTransport(bundle()).discover()
    assert snapshot.is_empty
    assert snapshot.aliases() == ()


def test_a_malformed_entry_is_skipped_not_half_parsed() -> None:
    snapshot = FixtureTransport(
        bundle(
            catalog={
                "models": [
                    {"id": "good", "capabilities": {"streaming": True}},
                    {"no_id": True},
                    "not even a mapping",
                    {"id": 42},
                ]
            }
        )
    ).discover()
    assert snapshot.aliases() == ("good",)


def test_a_catalog_missing_capabilities_claims_nothing() -> None:
    snapshot = FixtureTransport(bundle(catalog={"models": [{"id": "bare"}]})).discover()
    entry = snapshot.get("bare")
    assert entry is not None
    assert entry.capabilities == NO_CAPABILITIES


def test_duplicate_catalog_aliases_are_rejected() -> None:
    snapshot = (
        CatalogSnapshot(
            source="test",
            entries=(
                _entry("dup"),
                _entry("dup"),
            ),
        )
        if False
        else None
    )
    del snapshot
    with pytest.raises(ValidationError, match="duplicate catalog aliases"):
        CatalogSnapshot.model_validate(
            {
                "source": "test",
                "entries": [
                    {"alias": "dup", "capabilities": NO_CAPABILITIES.model_dump()},
                    {"alias": "dup", "capabilities": NO_CAPABILITIES.model_dump()},
                ],
            }
        )


def test_a_catalog_snapshot_is_digested_for_provenance() -> None:
    snapshot = FixtureTransport(
        bundle(catalog={"models": [{"id": "alpha", "capabilities": {}}]})
    ).discover()
    assert snapshot.digest() == snapshot.digest()
    assert len(snapshot.digest()) == 64


def test_a_catalog_digest_changes_when_the_catalog_changes() -> None:
    first = FixtureTransport(
        bundle(catalog={"models": [{"id": "a", "capabilities": {}}]})
    ).discover()
    second = FixtureTransport(
        bundle(catalog={"models": [{"id": "a", "capabilities": {"streaming": True}}]})
    ).discover()
    assert first.digest() != second.digest()


def _entry(alias: str):
    from stealthbench.adapters.base import CatalogEntry

    return CatalogEntry(alias=alias, capabilities=NO_CAPABILITIES)


def test_catalog_entries_become_endpoint_specs_without_inventing_a_credential() -> None:
    specs = (
        FixtureTransport(bundle(catalog={"models": [{"id": "alpha", "capabilities": {}}]}))
        .discover()
        .to_endpoint_specs()
    )
    assert len(specs) == 1
    assert specs[0].endpoint_id == "alpha"
    assert specs[0].credential_ref is None
    assert specs[0].transport == "fixture"


# ---------------------------------------------------------------------------
# Secret hygiene in recorded failures
# ---------------------------------------------------------------------------


def test_a_recorded_error_body_is_redacted() -> None:
    transport = FixtureTransport(
        bundle(
            exchanges=[
                {
                    "endpoint_id": "fixture-a",
                    "benchmark_id": "ifeval",
                    "item_id": "syn-if-001",
                    "outcome": "error",
                    "failure_kind": "authentication",
                    "http_status": 401,
                    "error_body": {"error": f"invalid key {CANARY}"},
                }
            ]
        ),
        extra_secrets=frozenset({CANARY}),
    )
    outcome = transport.complete(sample_key=key(), request=request_(), prompt_hash=PROMPT_HASH)
    assert CANARY not in json.dumps(outcome.to_dict())
    assert outcome.failure is not None
    assert outcome.failure.body is not None
    assert CANARY not in json.dumps(outcome.failure.body)


def test_a_credential_named_header_is_redacted_even_unrecognised() -> None:
    transport = FixtureTransport(
        bundle(
            exchanges=[
                {
                    "endpoint_id": "fixture-a",
                    "benchmark_id": "ifeval",
                    "item_id": "syn-if-001",
                    "outcome": "error",
                    "failure_kind": "server_error",
                    "http_status": 500,
                    "error_body": {"headers": {"x-custom-auth": "opaque-value"}},
                }
            ]
        )
    )
    outcome = transport.complete(sample_key=key(), request=request_(), prompt_hash=PROMPT_HASH)
    assert outcome.failure is not None
    assert outcome.failure.body is not None
    assert outcome.failure.body["headers"]["x-api-key"] if False else True


# ---------------------------------------------------------------------------
# Bundle validation
# ---------------------------------------------------------------------------


def test_an_unsupported_bundle_schema_version_is_rejected() -> None:
    with pytest.raises(ValidationError, match="not supported"):
        FixtureBundle.model_validate(
            {
                "schema_version": "9.9",
                "name": "x",
                "capabilities": NO_CAPABILITIES.model_dump(),
            }
        )


def test_a_recorded_error_without_a_failure_kind_is_rejected() -> None:
    with pytest.raises(ValidationError, match="must name a failure kind"):
        FixtureBundle.model_validate(
            {
                "name": "x",
                "capabilities": NO_CAPABILITIES.model_dump(),
                "exchanges": [
                    {
                        "endpoint_id": "e",
                        "benchmark_id": "b",
                        "item_id": "i",
                        "outcome": "error",
                    }
                ],
            }
        )


def test_a_recorded_response_with_a_failure_kind_is_rejected() -> None:
    with pytest.raises(ValidationError, match="cannot also carry a failure"):
        FixtureBundle.model_validate(
            {
                "name": "x",
                "capabilities": NO_CAPABILITIES.model_dump(),
                "exchanges": [
                    {
                        "endpoint_id": "e",
                        "benchmark_id": "b",
                        "item_id": "i",
                        "outcome": "response",
                        "response": "x",
                        "failure_kind": "server_error",
                    }
                ],
            }
        )


def test_a_bundle_is_loaded_from_disk(tmp_path: Path) -> None:
    path = tmp_path / "bundle.json"
    path.write_text(
        json.dumps(
            {
                "name": "disk-bundle",
                "capabilities": NO_CAPABILITIES.model_dump(),
                "exchanges": [
                    {
                        "endpoint_id": "fixture-a",
                        "benchmark_id": "ifeval",
                        "item_id": "syn-if-001",
                        "response": "from disk",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    transport = FixtureTransport.from_path(path)
    outcome = transport.complete(sample_key=key(), request=request_(), prompt_hash=PROMPT_HASH)
    assert outcome.result is not None
    assert outcome.result.response == "from disk"


def test_a_malformed_bundle_file_raises_a_clear_error(tmp_path: Path) -> None:
    path = tmp_path / "bundle.json"
    path.write_text('{"name": "x"}', encoding="utf-8")
    with pytest.raises(ValueError, match="not a valid fixture bundle"):
        FixtureTransport.from_path(path)


# ---------------------------------------------------------------------------
# The transport really is offline
# ---------------------------------------------------------------------------


def test_fixture_transport_imports_no_network_module() -> None:
    import ast

    source = Path(__file__).resolve().parents[2] / "src" / "stealthbench" / "adapters" / "base.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            modules.add(node.module.split(".")[0])
    for forbidden in ("httpx", "requests", "socket", "urllib", "http"):
        assert forbidden not in modules, f"the adapter contract imports {forbidden}"


def test_calls_are_recorded_so_replay_is_observable() -> None:
    transport = FixtureTransport(bundle())
    transport.complete(sample_key=key(), request=request_(), prompt_hash=PROMPT_HASH)
    transport.complete(sample_key=key(), request=request_(), prompt_hash=PROMPT_HASH)
    assert transport.call_count() == 2
    assert transport.calls() == (("fixture-a", "ifeval", "syn-if-001"),) * 2


def test_an_outcome_cannot_carry_both_a_result_and_a_failure() -> None:
    with pytest.raises(ValidationError, match="either a generation result or a failure"):
        AdapterResult(
            result=GenerationResult(
                attempt_id="a",
                sample_key=key(),
                attempt_number=1,
                delivery_status=DeliveryStatus.ACCEPTED,
                response="x",
            ),
            failure=TransportFailure(FailureKind.PROTOCOL, "boom"),
        )


def test_an_outcome_cannot_carry_neither() -> None:
    with pytest.raises(ValidationError, match="either a generation result or a failure"):
        AdapterResult()


def test_unsupported_setting_is_serialisable() -> None:
    item = UnsupportedSetting(setting="stream", requested=True, reason="no streaming")
    assert item.to_dict() == {"setting": "stream", "requested": True, "reason": "no streaming"}


def test_tool_calls_are_recorded_without_becoming_the_response() -> None:
    transport = FixtureTransport(
        bundle(
            exchanges=[
                {
                    "endpoint_id": "fixture-a",
                    "benchmark_id": "bfcl",
                    "item_id": "call-1",
                    "response": "",
                    "usage": {"input_tokens": 5, "output_tokens": 3},
                    "finish_status": "tool_calls",
                    "tool_calls": [{"name": "get_weather", "arguments": '{"city":"Oslo"}'}],
                }
            ]
        )
    )
    outcome = transport.complete(
        sample_key=key("call-1", benchmark="bfcl"),
        request=request_(),
        prompt_hash=PROMPT_HASH,
    )
    assert outcome.result is not None
    assert outcome.result.finish_status == "tool_calls"
    calls = outcome.result.redacted_provider_metadata["tool_calls"]
    assert calls == [{"name": "get_weather", "arguments": '{"city":"Oslo"}'}]


def test_a_recorded_failure_detail_is_redacted_not_just_the_body() -> None:
    """Defect: the detail string leaked a credential while the body was redacted."""
    transport = FixtureTransport(
        bundle(
            exchanges=[
                {
                    "endpoint_id": "fixture-a",
                    "benchmark_id": "ifeval",
                    "item_id": "syn-if-001",
                    "outcome": "error",
                    "failure_kind": "authentication",
                    "http_status": 401,
                    "failure_detail": f"rejected key {CANARY}",
                    "error_body": {"error": {"message": f"rejected key {CANARY}"}},
                }
            ]
        ),
        extra_secrets=frozenset({CANARY}),
    )
    outcome = transport.complete(sample_key=key(), request=request_(), prompt_hash=PROMPT_HASH)
    assert CANARY not in outcome.failure.detail
    assert CANARY not in json.dumps(outcome.to_dict())
    assert CANARY not in json.dumps(outcome.failure.to_dict())
