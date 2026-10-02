"""Zen catalog and chat adapter (task T03B).

Acceptance: catalog snapshots and normal/error/tool responses normalize without
invented capabilities.

Every alias here is a fixture name, deliberately not a real stealth alias: the point
of these tests is that the adapter learns the alias set from the snapshot, so a test
that hardcoded a real one would prove nothing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from stealthbench.adapters.base import FailureKind
from stealthbench.adapters.zen import (
    ZEN_BASE_URL,
    ZEN_CATALOG_PATH,
    ZEN_CHAT_PATH,
    ZenAdapter,
    catalog_snapshot_digest,
    effective_settings_of,
    extract_text,
    extract_tool_calls,
    finish_reason_of,
    map_finish_reason,
    map_http_status,
    normalize_catalog,
    usage_from_response,
)
from stealthbench.schemas.results import DeliveryStatus, ModelRequest, SampleKey, Usage

pytestmark = pytest.mark.contract

CANARY = "sk-canary-zen-0123456789abcdef"
PROMPT_HASH = "a" * 64


def key(alias_task: str = "ifeval::item-1", repeat: int = 1) -> SampleKey:
    return SampleKey(campaign_id="c1", endpoint_id="alias-a", task_id=alias_task, repeat_id=repeat)


def request_(**overrides: object) -> ModelRequest:
    payload: dict[str, object] = {
        "messages": [{"role": "user", "content": "hello"}],
        "max_output_tokens": 64,
    }
    payload.update(overrides)
    return ModelRequest.model_validate(payload)


def chat_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "id": "chatcmpl-123",
        "model": "alias-a",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "hello there"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 9, "completion_tokens": 4},
    }
    payload.update(overrides)
    return payload


def adapter(**overrides: object) -> ZenAdapter:
    payload: dict[str, object] = {
        "catalog_payload": {"data": []},
        "exchanges": {},
    }
    payload.update(overrides)
    return ZenAdapter(**payload)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Catalog snapshots
# ---------------------------------------------------------------------------


def test_an_openai_shaped_catalog_is_normalized() -> None:
    snapshot = normalize_catalog(
        {
            "object": "list",
            "data": [
                {
                    "id": "alias-a",
                    "display_name": "Alias A",
                    "owned_by": "some-provider",
                    "family": "some-family",
                    "context_window": 200000,
                    "capabilities": {
                        "streaming": True,
                        "tool_calls": False,
                        "usage_reporting": True,
                    },
                }
            ],
        }
    )
    assert snapshot.aliases() == ("alias-a",)
    entry = snapshot.get("alias-a")
    assert entry is not None
    assert entry.provider == "some-provider"
    assert entry.family == "some-family"
    assert entry.capabilities.streaming is True
    assert entry.capabilities.tool_calls is False


def test_a_bare_list_catalog_is_accepted() -> None:
    snapshot = normalize_catalog([{"id": "alias-b"}])
    assert snapshot.aliases() == ("alias-b",)


def test_a_models_keyed_catalog_is_accepted() -> None:
    snapshot = normalize_catalog({"models": [{"alias": "alias-c"}]})
    assert snapshot.aliases() == ("alias-c",)


def test_an_empty_catalog_is_empty_not_invented() -> None:
    for payload in ({}, {"data": []}, [], None, "nonsense", 42):
        snapshot = normalize_catalog(payload)
        assert snapshot.is_empty, f"{payload!r} should yield an empty snapshot"


def test_capabilities_are_never_inferred_from_absence() -> None:
    """A catalog that says nothing about streaming must not be treated as streaming."""
    snapshot = normalize_catalog({"data": [{"id": "quiet"}]})
    entry = snapshot.get("quiet")
    assert entry is not None
    assert entry.capabilities.streaming is False
    assert entry.capabilities.tool_calls is False
    assert entry.capabilities.reasoning is False
    assert entry.capabilities.usage_reporting is False
    assert entry.capabilities.logprobs is False


def test_a_non_boolean_capability_value_does_not_become_true() -> None:
    snapshot = normalize_catalog({"data": [{"id": "odd", "capabilities": {"streaming": "yes"}}]})
    entry = snapshot.get("odd")
    assert entry is not None
    assert entry.capabilities.streaming is False


def test_a_malformed_entry_is_skipped_and_others_survive() -> None:
    snapshot = normalize_catalog({"data": [{"id": "good"}, {"nope": 1}, None, "string", {"id": 7}]})
    assert snapshot.aliases() == ("good",)


def test_duplicate_aliases_keep_the_first_occurrence() -> None:
    snapshot = normalize_catalog(
        {
            "data": [
                {"id": "dup", "display_name": "First"},
                {"id": "dup", "display_name": "Second"},
            ]
        }
    )
    assert snapshot.aliases() == ("dup",)
    entry = snapshot.get("dup")
    assert entry is not None
    assert entry.display_name == "First"


def test_the_snapshot_is_digested_for_provenance() -> None:
    first = normalize_catalog({"data": [{"id": "a"}]})
    again = normalize_catalog({"data": [{"id": "a"}]})
    other = normalize_catalog({"data": [{"id": "a"}, {"id": "b"}]})
    assert catalog_snapshot_digest(first) == catalog_snapshot_digest(again)
    assert catalog_snapshot_digest(first) != catalog_snapshot_digest(other)


def test_the_source_is_recorded() -> None:
    assert normalize_catalog({"data": []}, source="custom").source == "custom"
    assert normalize_catalog({"data": []}).source == ZEN_BASE_URL


def test_the_documented_paths_are_recorded() -> None:
    """Provenance: the endpoints this adapter targets are documented, never dialled."""
    assert ZEN_CATALOG_PATH == "/v1/models"
    assert ZEN_CHAT_PATH == "/v1/chat/completions"
    assert ZEN_BASE_URL.startswith("https://")


def test_discover_returns_the_recorded_catalog() -> None:
    instance = adapter(catalog_payload={"data": [{"id": "alias-x"}]})
    assert instance.discover().aliases() == ("alias-x",)


def test_discover_without_a_capture_is_empty() -> None:
    instance = ZenAdapter()
    assert instance.discover().is_empty


def test_no_stealth_alias_is_hardcoded() -> None:
    """The alias set must come from the catalog, not from this source file."""
    source = Path(__file__).resolve().parents[2] / "src" / "stealthbench" / "adapters" / "zen.py"
    text = source.read_text(encoding="utf-8").lower()
    for alias in ("big-pickle", "space-bunny"):
        assert alias not in text, f"{alias} is hardcoded in the adapter"


# ---------------------------------------------------------------------------
# Normal responses
# ---------------------------------------------------------------------------


def test_a_normal_response_is_normalized() -> None:
    instance = adapter(exchanges={"ifeval::item-1": {"http_status": 200, "json": chat_payload()}})
    outcome = instance.complete(sample_key=key(), request=request_(), prompt_hash=PROMPT_HASH)
    assert outcome.ok
    assert outcome.result is not None
    assert outcome.result.response == "hello there"
    assert outcome.result.usage.input_tokens == 9
    assert outcome.result.usage.output_tokens == 4
    assert outcome.result.finish_status == "stop"
    assert outcome.result.delivery_status is DeliveryStatus.ACCEPTED


def test_the_reported_model_is_kept_separate_from_the_endpoint() -> None:
    """A gateway may serve a different model than the alias names."""
    instance = adapter(
        exchanges={
            "ifeval::item-1": {"http_status": 200, "json": chat_payload(model="something-else")}
        }
    )
    outcome = instance.complete(sample_key=key(), request=request_(), prompt_hash=PROMPT_HASH)
    assert outcome.result is not None
    assert outcome.result.sample_key.endpoint_id == "alias-a"
    assert outcome.result.redacted_provider_metadata["reported_model"] == "something-else"


def test_request_id_is_recorded_for_provenance() -> None:
    instance = adapter(exchanges={"ifeval::item-1": {"http_status": 200, "json": chat_payload()}})
    outcome = instance.complete(sample_key=key(), request=request_(), prompt_hash=PROMPT_HASH)
    assert outcome.result is not None
    assert outcome.result.redacted_provider_metadata["request_id"] == "chatcmpl-123"


def test_effective_settings_separate_requested_from_reported() -> None:
    payload = chat_payload(stealthbench_effective={"temperature": 0.9})
    instance = adapter(exchanges={"ifeval::item-1": {"http_status": 200, "json": payload}})
    outcome = instance.complete(
        sample_key=key(), request=request_(temperature=0.0), prompt_hash=PROMPT_HASH
    )
    assert outcome.result is not None
    effective = outcome.result.effective_settings
    assert effective["requested"]["temperature"] == 0.0
    assert effective["reported"]["temperature"] == 0.9, (
        "what the endpoint did must not be conflated with what we asked for"
    )


def test_requested_and_reported_settings_are_always_both_present() -> None:
    effective = effective_settings_of({}, request_(temperature=0.3))
    assert "requested" in effective
    assert "reported" not in effective, "nothing reported means nothing claimed"


# ---------------------------------------------------------------------------
# Usage normalisation
# ---------------------------------------------------------------------------


def test_openai_style_usage_keys_are_read() -> None:
    usage = usage_from_response({"usage": {"prompt_tokens": 5, "completion_tokens": 7}})
    assert usage.input_tokens == 5
    assert usage.output_tokens == 7
    assert usage.is_complete


def test_anthropic_style_usage_keys_are_read() -> None:
    usage = usage_from_response({"usage": {"input_tokens": 5, "output_tokens": 7}})
    assert usage.input_tokens == 5
    assert usage.output_tokens == 7


def test_absent_usage_stays_null() -> None:
    usage = usage_from_response({"choices": []})
    assert usage.input_tokens is None
    assert usage.output_tokens is None
    assert usage.provider_reported is False


def test_partially_reported_usage_keeps_the_gap() -> None:
    usage = usage_from_response({"usage": {"prompt_tokens": 5}})
    assert usage.input_tokens == 5
    assert usage.output_tokens is None
    assert not usage.is_complete


def test_a_reported_zero_survives() -> None:
    usage = usage_from_response({"usage": {"prompt_tokens": 0, "completion_tokens": 0}})
    assert usage.input_tokens == 0
    assert usage.output_tokens == 0
    assert usage != Usage()
    assert usage.provider_reported is True


def test_cached_and_reasoning_tokens_are_read_when_present() -> None:
    usage = usage_from_response(
        {
            "usage": {
                "input_tokens": 100,
                "output_tokens": 20,
                "cached_input_tokens": 60,
                "reasoning_tokens": 8,
            }
        }
    )
    assert usage.cached_input_tokens == 60
    assert usage.reasoning_tokens == 8


def test_usage_at_the_top_level_is_read() -> None:
    usage = usage_from_response({"input_tokens": 3, "output_tokens": 4})
    assert usage.input_tokens == 3
    assert usage.output_tokens == 4


def test_a_negative_or_boolean_token_value_is_ignored() -> None:
    usage = usage_from_response({"usage": {"prompt_tokens": -5, "completion_tokens": True}})
    assert usage.input_tokens is None
    assert usage.output_tokens is None


# ---------------------------------------------------------------------------
# Error responses
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("status", "kind"),
    [
        (401, FailureKind.AUTHENTICATION),
        (403, FailureKind.PERMISSION),
        (404, FailureKind.NOT_FOUND),
        (429, FailureKind.RATE_LIMIT),
        (500, FailureKind.SERVER_ERROR),
        (503, FailureKind.SERVER_ERROR),
        (408, FailureKind.TIMEOUT),
        (302, FailureKind.PROTOCOL),
    ],
)
def test_http_statuses_map_to_failure_kinds(status: int, kind: FailureKind) -> None:
    assert map_http_status(status) is kind


@pytest.mark.parametrize("status", [200, 201, 204, None])
def test_success_statuses_map_to_no_failure(status: int | None) -> None:
    assert map_http_status(status) is None


def test_an_authentication_failure_is_reported_with_its_error_message() -> None:
    instance = adapter(
        exchanges={
            "ifeval::item-1": {
                "http_status": 401,
                "body": {"error": {"message": "invalid api key"}},
            }
        }
    )
    outcome = instance.complete(sample_key=key(), request=request_(), prompt_hash=PROMPT_HASH)
    assert not outcome.ok
    assert outcome.failure is not None
    assert outcome.failure.kind is FailureKind.AUTHENTICATION
    assert outcome.failure.detail == "invalid api key"
    assert not outcome.failure.retryable
    assert outcome.result is None


def test_a_rate_limit_carries_retry_after() -> None:
    instance = adapter(
        exchanges={
            "ifeval::item-1": {
                "http_status": 429,
                "headers": {"retry-after": "7"},
                "body": {"error": "slow down"},
            }
        }
    )
    outcome = instance.complete(sample_key=key(), request=request_(), prompt_hash=PROMPT_HASH)
    assert outcome.failure is not None
    assert outcome.failure.kind is FailureKind.RATE_LIMIT
    assert outcome.failure.retry_after_seconds == 7.0
    assert outcome.failure.retryable


def test_an_error_body_containing_a_credential_is_redacted() -> None:
    instance = adapter(
        catalog_payload={"data": []},
        exchanges={
            "ifeval::item-1": {
                "http_status": 401,
                "body": {"error": {"message": f"bad key {CANARY}", "key": CANARY}},
            }
        },
        extra_secrets=frozenset({CANARY}),
    )
    outcome = instance.complete(sample_key=key(), request=request_(), prompt_hash=PROMPT_HASH)
    assert CANARY not in json.dumps(outcome.to_dict())


def test_a_response_without_content_or_tool_calls_is_a_protocol_error() -> None:
    instance = adapter(
        exchanges={
            "ifeval::item-1": {
                "http_status": 200,
                "json": {"id": "x", "choices": [{"message": {}}]},
            }
        }
    )
    outcome = instance.complete(sample_key=key(), request=request_(), prompt_hash=PROMPT_HASH)
    assert not outcome.ok
    assert outcome.failure is not None
    assert outcome.failure.kind is FailureKind.PROTOCOL


def test_an_exchange_with_no_json_body_is_a_protocol_error() -> None:
    instance = adapter(exchanges={"ifeval::item-1": {"http_status": 200}})
    outcome = instance.complete(sample_key=key(), request=request_(), prompt_hash=PROMPT_HASH)
    assert outcome.failure is not None
    assert outcome.failure.kind is FailureKind.PROTOCOL


def test_an_unrecorded_exchange_is_reported_not_invented() -> None:
    outcome = adapter().complete(sample_key=key(), request=request_(), prompt_hash=PROMPT_HASH)
    assert outcome.failure is not None
    assert outcome.failure.kind is FailureKind.NO_FIXTURE


# ---------------------------------------------------------------------------
# Tool responses
# ---------------------------------------------------------------------------


def test_a_tool_call_response_is_normalized() -> None:
    payload = chat_payload(
        choices=[
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": '{"city":"Oslo"}'},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        usage={"prompt_tokens": 20, "completion_tokens": 8},
    )
    instance = adapter(exchanges={"ifeval::item-1": {"http_status": 200, "json": payload}})
    outcome = instance.complete(sample_key=key(), request=request_(), prompt_hash=PROMPT_HASH)
    assert outcome.ok
    assert outcome.result is not None
    assert outcome.result.finish_status == "tool_calls"
    calls = outcome.result.redacted_provider_metadata["tool_calls"]
    assert calls[0]["function"]["name"] == "get_weather"


def test_a_tool_call_with_no_text_is_still_accepted() -> None:
    """A tool call is a valid completion even when the text content is empty."""
    payload = chat_payload(
        choices=[
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"id": "c", "type": "function", "function": {"name": "f"}}],
                },
                "finish_reason": "tool_calls",
            }
        ]
    )
    outcome = adapter(exchanges={"ifeval::item-1": {"http_status": 200, "json": payload}}).complete(
        sample_key=key(), request=request_(), prompt_hash=PROMPT_HASH
    )
    assert outcome.ok


def test_no_tool_calls_reports_an_empty_tuple_not_none() -> None:
    payload = chat_payload()
    assert extract_tool_calls(payload) == ()
    outcome = adapter(exchanges={"ifeval::item-1": {"http_status": 200, "json": payload}}).complete(
        sample_key=key(), request=request_(), prompt_hash=PROMPT_HASH
    )
    assert outcome.result is not None
    assert outcome.result.redacted_provider_metadata["tool_calls"] == []


# ---------------------------------------------------------------------------
# Finish reasons
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("reason", "expected"),
    [
        ("stop", "stop"),
        ("STOP", "stop"),
        ("length", "length"),
        ("tool_calls", "tool_calls"),
        ("function_call", "tool_calls"),
        ("content_filter", "content_filter"),
        ("something_new", "error"),
        (None, "stop"),
    ],
)
def test_finish_reasons_map_to_the_frozen_vocabulary(reason: str | None, expected: str) -> None:
    assert map_finish_reason(reason) == expected


def test_an_unrecognised_finish_reason_is_never_mapped_to_stop() -> None:
    """A truncated generation must not be counted as a clean stop."""
    assert map_finish_reason("weird") != "stop"


def test_a_missing_choices_array_is_an_error_not_a_stop() -> None:
    assert finish_reason_of({"choices": []}) == "error"
    assert finish_reason_of({}) == "error"


# ---------------------------------------------------------------------------
# Text extraction shapes
# ---------------------------------------------------------------------------


def test_content_parts_are_concatenated() -> None:
    payload = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "part one "},
                        {"type": "text", "text": "two"},
                    ],
                }
            }
        ]
    }
    assert extract_text(payload) == "part one two"


def test_absent_content_is_none_not_an_empty_string() -> None:
    """Absent and empty are different facts."""
    assert extract_text({"choices": [{"message": {}}]}) is None
    assert extract_text({"choices": [{"message": {"content": ""}}]}) == ""
    assert extract_text({"choices": []}) is None
    assert extract_text({}) is None


# ---------------------------------------------------------------------------
# Loading from disk
# ---------------------------------------------------------------------------


def test_an_adapter_is_loaded_from_a_transcript_file(tmp_path: Path) -> None:
    path = tmp_path / "zen.json"
    path.write_text(
        json.dumps(
            {
                "catalog": {"data": [{"id": "alias-disk"}]},
                "exchanges": {"ifeval::item-1": {"http_status": 200, "json": chat_payload()}},
            }
        ),
        encoding="utf-8",
    )
    instance = ZenAdapter.from_path(path)
    assert instance.discover().aliases() == ("alias-disk",)
    outcome = instance.complete(sample_key=key(), request=request_(), prompt_hash=PROMPT_HASH)
    assert outcome.ok


def test_a_transcript_file_must_be_an_object(tmp_path: Path) -> None:
    path = tmp_path / "zen.json"
    path.write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(ValueError, match="must contain a JSON object"):
        ZenAdapter.from_path(path)


def test_repeats_are_selectable_in_a_transcript() -> None:
    """The repeat must select a different recorded generation, not the first one."""
    instance = adapter(
        exchanges={
            "ifeval::item-1#r1": {"http_status": 200, "json": chat_payload()},
            "ifeval::item-1#r2": {
                "http_status": 200,
                "json": chat_payload(
                    choices=[{"message": {"content": "second"}, "finish_reason": "stop"}]
                ),
            },
        }
    )
    first = instance.complete(sample_key=key(repeat=1), request=request_(), prompt_hash=PROMPT_HASH)
    second = instance.complete(
        sample_key=key(repeat=2), request=request_(), prompt_hash=PROMPT_HASH
    )
    assert first.result is not None and first.result.response == "hello there"
    assert second.result is not None and second.result.response == "second"
