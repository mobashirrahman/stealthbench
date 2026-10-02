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


def key(
    alias_task: str = "ifeval::item-1", repeat: int = 1, endpoint: str = "alias-a"
) -> SampleKey:
    return SampleKey(campaign_id="c1", endpoint_id=endpoint, task_id=alias_task, repeat_id=repeat)


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
    ],
)
def test_finish_reasons_map_to_the_frozen_vocabulary(reason: str, expected: str) -> None:
    assert map_finish_reason(reason) == expected


def test_an_unrecognised_finish_reason_is_never_mapped_to_stop() -> None:
    """A truncated generation must not be counted as a clean stop."""
    assert map_finish_reason("weird") != "stop"


def test_an_absent_finish_reason_stays_absent() -> None:
    """No reason is not a stop. The gateway said nothing; the record must too.

    This case previously asserted ``None -> "stop"``, which is how a truncated
    generation gets counted as a completed answer.
    """
    assert map_finish_reason(None) is None
    assert finish_reason_of({"choices": [{"delta": {}, "finish_reason": None}]}) is None
    assert finish_reason_of({"choices": [{}]}) is None


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


# ---------------------------------------------------------------------------
# Regression tests for the G03 review findings
# ---------------------------------------------------------------------------


def test_a_repeat_specific_record_beats_a_generic_one() -> None:
    """Defect: the repeat-less key was yielded first, so every repeat replayed one response."""
    instance = adapter(
        exchanges={
            "ifeval::item-1": {
                "http_status": 200,
                "json": chat_payload(
                    choices=[{"message": {"content": "GENERIC"}, "finish_reason": "stop"}]
                ),
            },
            "ifeval::item-1#r2": {
                "http_status": 200,
                "json": chat_payload(
                    choices=[{"message": {"content": "REPEAT-2"}, "finish_reason": "stop"}]
                ),
            },
        }
    )
    request = request_()
    responses = {
        repeat: instance.complete(
            sample_key=key(repeat=repeat), request=request, prompt_hash=PROMPT_HASH
        ).result.response
        for repeat in (1, 2, 3)
    }
    assert responses == {1: "GENERIC", 2: "REPEAT-2", 3: "GENERIC"}


def test_every_repeat_gets_its_own_record_when_each_is_recorded() -> None:
    instance = adapter(
        exchanges={
            f"ifeval::item-1#r{repeat}": {
                "http_status": 200,
                "json": chat_payload(
                    choices=[{"message": {"content": f"r{repeat}"}, "finish_reason": "stop"}]
                ),
            }
            for repeat in (1, 2, 3)
        }
    )
    request = request_()
    responses = [
        instance.complete(
            sample_key=key(repeat=repeat), request=request, prompt_hash=PROMPT_HASH
        ).result.response
        for repeat in (1, 2, 3)
    ]
    assert responses == ["r1", "r2", "r3"]


@pytest.mark.parametrize("raw", ["false", "no", 1, "yes", 0])
def test_a_non_boolean_capability_is_never_reported_as_supported(raw: object) -> None:
    snapshot = normalize_catalog({"data": [{"id": "odd", "capabilities": {"streaming": raw}}]})
    entry = snapshot.get("odd")
    assert entry is not None
    assert entry.capabilities.streaming is False


def test_a_credential_in_effective_settings_is_redacted() -> None:
    """Defect: Zen effective_settings bypassed redact_mapping while its metadata did not."""
    instance = adapter(
        exchanges={
            "ifeval::item-1": {
                "http_status": 200,
                "json": chat_payload(stealthbench_effective={"debug": CANARY}),
            }
        },
        extra_secrets=frozenset({CANARY}),
    )
    outcome = instance.complete(sample_key=key(), request=request_(), prompt_hash=PROMPT_HASH)
    assert CANARY not in json.dumps(outcome.to_dict())


def test_a_credential_in_a_zen_tool_call_is_redacted() -> None:
    payload = chat_payload(
        choices=[
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "c",
                            "type": "function",
                            "function": {"name": "f", "arguments": f'{{"k":"{CANARY}"}}'},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ]
    )
    instance = adapter(
        exchanges={"ifeval::item-1": {"http_status": 200, "json": payload}},
        extra_secrets=frozenset({CANARY}),
    )
    outcome = instance.complete(sample_key=key(), request=request_(), prompt_hash=PROMPT_HASH)
    assert CANARY not in json.dumps(outcome.to_dict())


def test_a_credential_in_a_zen_catalog_raw_record_is_redacted() -> None:
    snapshot = ZenAdapter(
        catalog_payload={"data": [{"id": "m", "note": CANARY}]},
        extra_secrets=frozenset({CANARY}),
    ).discover()
    assert CANARY not in json.dumps(dict(snapshot.raw))
    assert CANARY not in json.dumps([e.model_dump(mode="json") for e in snapshot.entries])


def test_the_zen_adapter_can_stream_and_labels_its_own_route() -> None:
    from stealthbench.schemas.campaign import Capabilities

    caps = Capabilities(
        streaming=True, tool_calls=False, reasoning=False, usage_reporting=False, logprobs=False
    )
    instance = adapter(
        exchanges={
            "ifeval::item-1": {
                "http_status": 200,
                "stream_frames": [
                    {"choices": [{"delta": {"content": "hello "}}]},
                    {"choices": [{"delta": {"content": "world"}, "finish_reason": "stop"}]},
                    {"choices": [], "usage": {"input_tokens": 3, "output_tokens": 2}},
                ],
            }
        }
    )
    outcome = instance.stream(
        sample_key=key(),
        request=request_(),
        prompt_hash=PROMPT_HASH,
        capabilities=caps,
    )
    assert outcome.ok
    assert outcome.result is not None
    assert outcome.result.response == "hello world"
    assert outcome.result.usage.output_tokens == 2
    assert outcome.result.redacted_provider_metadata["route"] == "zen"


def test_streaming_an_exchange_without_frames_is_reported() -> None:
    instance = adapter(exchanges={"ifeval::item-1": {"http_status": 200, "json": chat_payload()}})
    outcome = instance.stream(sample_key=key(), request=request_(), prompt_hash=PROMPT_HASH)
    assert not outcome.ok
    assert outcome.failure is not None
    assert outcome.failure.kind is FailureKind.UNSUPPORTED_SETTING


def test_a_credential_in_the_reported_model_name_is_redacted() -> None:
    """A gateway can echo the credential back inside the model field."""
    instance = adapter(
        exchanges={
            "ifeval::item-1": {
                "http_status": 200,
                "json": chat_payload(model=f"alias-a-{CANARY}"),
            }
        },
        extra_secrets=frozenset({CANARY}),
    )
    outcome = instance.complete(sample_key=key(), request=request_(), prompt_hash=PROMPT_HASH)
    assert CANARY not in json.dumps(outcome.to_dict())
    assert outcome.result is not None
    assert CANARY not in str(outcome.result.redacted_provider_metadata.get("reported_model"))


# ---------------------------------------------------------------------------
# The declared-secret path must be exercised on its own
#
# An `sk-...` shaped canary is defused by the built-in credential patterns whether
# or not the adapter passes its declared secrets through, so it cannot prove the
# declared-secrets wiring. BLIND is shaped so that only an explicitly declared
# secret can remove it.
# ---------------------------------------------------------------------------

BLIND = "ZZQdeclared-canary-7f3a2b9c4d1e"


def test_the_declared_secret_path_is_wired_into_effective_settings() -> None:
    instance = adapter(
        exchanges={
            "ifeval::item-1": {
                "http_status": 200,
                "json": chat_payload(
                    model=f"alias-{BLIND}", stealthbench_effective={"debug": BLIND}
                ),
            }
        },
        extra_secrets=frozenset({BLIND}),
    )
    outcome = instance.complete(sample_key=key(), request=request_(), prompt_hash=PROMPT_HASH)
    assert BLIND not in json.dumps(outcome.to_dict())


def test_the_declared_secret_path_is_wired_into_tool_calls() -> None:
    payload = chat_payload(
        choices=[
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "c",
                            "type": "function",
                            "function": {"name": "f", "arguments": f'{{"k":"{BLIND}"}}'},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ]
    )
    instance = adapter(
        exchanges={"ifeval::item-1": {"http_status": 200, "json": payload}},
        extra_secrets=frozenset({BLIND}),
    )
    outcome = instance.complete(sample_key=key(), request=request_(), prompt_hash=PROMPT_HASH)
    assert BLIND not in json.dumps(outcome.to_dict())


def test_the_declared_secret_path_is_wired_into_the_catalog_snapshot() -> None:
    snapshot = ZenAdapter(
        catalog_payload={"data": [{"id": "m", "note": BLIND}]},
        extra_secrets=frozenset({BLIND}),
    ).discover()
    assert BLIND not in json.dumps(dict(snapshot.raw))
    assert BLIND not in json.dumps([e.model_dump(mode="json") for e in snapshot.entries])


def test_the_declared_secret_path_is_wired_into_a_failure_body() -> None:
    instance = adapter(
        exchanges={
            "ifeval::item-1": {
                "http_status": 500,
                "body": {"error": {"message": f"upstream rejected {BLIND}"}},
            }
        },
        extra_secrets=frozenset({BLIND}),
    )
    outcome = instance.complete(sample_key=key(), request=request_(), prompt_hash=PROMPT_HASH)
    assert outcome.failure is not None
    assert BLIND not in json.dumps(outcome.to_dict())


def test_a_declared_secret_that_matches_no_pattern_is_still_visible_if_undeclared() -> None:
    """The control: without declaring it, the value survives, so the tests above bite."""
    instance = adapter(
        exchanges={
            "ifeval::item-1": {
                "http_status": 200,
                "json": chat_payload(stealthbench_effective={"debug": BLIND}),
            }
        }
    )
    outcome = instance.complete(sample_key=key(), request=request_(), prompt_hash=PROMPT_HASH)
    assert BLIND in json.dumps(outcome.to_dict()), "an undeclared opaque value stays as-is"


# ---------------------------------------------------------------------------
# Regression tests for the second G03 review
# ---------------------------------------------------------------------------


def test_a_capture_without_the_sentinel_is_not_replayed_as_complete() -> None:
    """CRITICAL: the Zen stream path appended `data: [DONE]` unconditionally."""
    instance = adapter(
        exchanges={
            "ifeval::item-1": {
                "http_status": 200,
                "stream_terminated": False,
                "stream_frames": [
                    {"choices": [{"delta": {"content": "truncated mid-"}}]},
                ],
            }
        }
    )
    outcome = instance.stream(sample_key=key(), request=request_(), prompt_hash=PROMPT_HASH)
    assert not outcome.ok
    assert outcome.result is None
    assert outcome.failure is not None
    assert outcome.failure.kind is FailureKind.INTERRUPTED


def test_a_declared_secret_reaches_no_part_of_a_zen_streamed_failure() -> None:
    blind = "ZZQdeclared-canary-7f3a2b9c4d1e"
    instance = adapter(
        exchanges={
            "ifeval::item-1": {
                "http_status": 200,
                "stream_frames": [{"error": {"message": f"gateway rejected {blind}"}}],
            }
        },
        extra_secrets=frozenset({blind}),
    )
    outcome = instance.stream(sample_key=key(), request=request_(), prompt_hash=PROMPT_HASH)
    assert outcome.failure is not None
    assert blind not in outcome.failure.detail
    assert blind not in json.dumps(outcome.to_dict())


def test_recorded_unsupported_settings_are_reported_on_the_zen_route() -> None:
    """MAJOR: the Zen route could only ever report the `stream` setting."""
    instance = adapter(
        exchanges={
            "ifeval::item-1": {
                "http_status": 200,
                "json": chat_payload(),
                "unsupported_settings": ["temperature", "top_p"],
            }
        }
    )
    outcome = instance.complete(
        sample_key=key(),
        request=request_(temperature=0.7, top_p=0.5),
        prompt_hash=PROMPT_HASH,
    )
    assert sorted(item.setting for item in outcome.unsupported) == ["temperature", "top_p"]
    reported = {item.setting: item.requested for item in outcome.unsupported}
    assert reported["temperature"] == 0.7
    assert reported["top_p"] == 0.5


def test_recorded_unsupported_settings_are_reported_when_streaming() -> None:
    from stealthbench.schemas.campaign import Capabilities

    caps = Capabilities(
        streaming=True, tool_calls=False, reasoning=False, usage_reporting=False, logprobs=False
    )
    instance = adapter(
        exchanges={
            "ifeval::item-1": {
                "http_status": 200,
                "unsupported_settings": ["top_p"],
                "stream_frames": [{"choices": [{"delta": {"content": "x"}}]}],
            }
        }
    )
    outcome = instance.stream(
        sample_key=key(),
        request=request_(top_p=0.5),
        prompt_hash=PROMPT_HASH,
        capabilities=caps,
    )
    assert [item.setting for item in outcome.unsupported] == ["top_p"]


def test_an_unqualified_transcript_key_is_refused_when_several_aliases_exist() -> None:
    """MAJOR: one capture was replayed as a sample for every endpoint in the campaign."""
    instance = ZenAdapter(
        catalog_payload={"data": [{"id": "alias-a"}, {"id": "alias-b"}]},
        exchanges={"ifeval::item-1": {"http_status": 200, "json": chat_payload()}},
    )
    for endpoint in ("alias-a", "alias-b", "alias-c"):
        outcome = instance.complete(
            sample_key=key(endpoint=endpoint),
            request=request_(),
            prompt_hash=PROMPT_HASH,
        )
        assert not outcome.ok, f"{endpoint} must not inherit another endpoint's capture"
        assert outcome.failure is not None
        assert outcome.failure.kind is FailureKind.NO_FIXTURE


def test_an_unqualified_transcript_key_is_used_when_the_capture_has_one_alias() -> None:
    instance = ZenAdapter(
        catalog_payload={"data": [{"id": "alias-a"}]},
        exchanges={"ifeval::item-1": {"http_status": 200, "json": chat_payload()}},
    )
    outcome = instance.complete(sample_key=key(), request=request_(), prompt_hash=PROMPT_HASH)
    assert outcome.ok
    assert outcome.result is not None


def test_an_endpoint_qualified_key_is_used_regardless_of_alias_count() -> None:
    instance = ZenAdapter(
        catalog_payload={"data": [{"id": "alias-a"}, {"id": "alias-b"}]},
        exchanges={"alias-a:ifeval::item-1": {"http_status": 200, "json": chat_payload()}},
    )
    mine = instance.complete(
        sample_key=key(endpoint="alias-a"), request=request_(), prompt_hash=PROMPT_HASH
    )
    theirs = instance.complete(
        sample_key=key(endpoint="alias-b"), request=request_(), prompt_hash=PROMPT_HASH
    )
    assert mine.ok
    assert not theirs.ok


def test_a_catalog_request_with_a_query_string_is_still_recognised() -> None:
    from stealthbench.adapters.zen import _catalog_from_requests

    catalog = _catalog_from_requests(
        [
            {
                "method": "GET",
                "path": "/v1/models?limit=100",
                "status": 200,
                "body": {"data": [{"id": "m"}]},
            }
        ]
    )
    assert catalog == {"data": [{"id": "m"}]}


@pytest.mark.parametrize("status", [500, 401, True, "200", 302])
def test_only_a_real_2xx_catalog_response_is_accepted(status: object) -> None:
    from stealthbench.adapters.zen import _catalog_from_requests

    catalog = _catalog_from_requests(
        [{"method": "GET", "path": "/v1/models", "status": status, "body": {"data": [{"id": "m"}]}}]
    )
    assert catalog is None


def test_a_boolean_context_window_is_not_a_measurement_on_the_zen_route() -> None:
    snapshot = normalize_catalog({"data": [{"id": "m", "context_window": True}]})
    entry = snapshot.get("m")
    assert entry is not None
    assert entry.context_window is None


@pytest.mark.parametrize("route", [None, "", "   "])
def test_a_missing_route_label_does_not_become_the_string_none(route: object) -> None:
    snapshot = normalize_catalog({"data": [{"id": "m", "route": route}]})
    entry = snapshot.get("m")
    assert entry is not None
    assert entry.route == "zen", f"{route!r} must fall back, not stringify"


def test_an_explicit_route_label_survives_on_the_zen_route() -> None:
    snapshot = normalize_catalog({"data": [{"id": "m", "route": "eu-west"}]})
    entry = snapshot.get("m")
    assert entry is not None
    assert entry.route == "eu-west"
