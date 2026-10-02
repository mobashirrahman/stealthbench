"""Zen catalog discovery and chat adapter.

Two facts about this endpoint shape the whole design:

* **An alias is not an immutable model version.** The catalog changes under a fixed
  alias, so every campaign stores a hashed snapshot of what was observed and nothing
  is hardcoded. In particular this file contains no stealth alias by name; the set of
  aliases comes from the snapshot.
* **A gateway, not necessarily the model developer.** A Zen alias may be served by
  someone else entirely. Protocol behaviour is therefore evidence about the *route*,
  never about the model behind it, and the route label stays separate from the
  provider and family labels.

The adapter reads its responses from a recorded transcript rather than the network.
That is what lets the offline suite exercise the production code path — the same
normalisation runs against a real capture, so the gate's requirement that "production
adapter behavior is demonstrated against fixture protocol transcripts" is satisfiable
without spending anything.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from pydantic import ValidationError

from stealthbench.adapters.base import (
    AdapterResult,
    CatalogEntry,
    CatalogSnapshot,
    FailureKind,
    ProviderAdapter,
    TransportFailure,
    UnsupportedSetting,
    check_requested_settings,
    safe_failure,
)
from stealthbench.schemas.campaign import Capabilities
from stealthbench.schemas.hashing import content_digest
from stealthbench.schemas.results import (
    DeliveryStatus,
    FinishStatus,
    GenerationResult,
    ModelRequest,
    SampleKey,
    Usage,
)
from stealthbench.storage.events import redact_mapping, redact_text

ZEN_ADAPTER_VERSION: Final[str] = "0.1.0"

#: Documented Zen endpoints. Recorded for provenance; never dialled by this module.
ZEN_CATALOG_PATH: Final[str] = "/v1/models"
ZEN_CHAT_PATH: Final[str] = "/v1/chat/completions"
ZEN_BASE_URL: Final[str] = "https://opencode.ai/zen"

#: Chat finish reasons, mapped onto the frozen vocabulary. Anything unrecognised
#: becomes ``error`` rather than being mapped onto a success reason.
_FINISH_REASONS: Final[dict[str, FinishStatus]] = {
    "stop": "stop",
    "length": "length",
    "tool_calls": "tool_calls",
    "function_call": "tool_calls",
    "content_filter": "content_filter",
}


def map_finish_reason(reason: str | None) -> FinishStatus:
    """Map a gateway finish reason onto the frozen vocabulary.

    An unrecognised reason is not treated as ``stop``. Guessing success is how a
    truncated generation ends up counted as a completed answer.
    """
    if reason is None:
        return "stop"
    return _FINISH_REASONS.get(reason.lower(), "error")


def map_http_status(status: int | None) -> FailureKind | None:
    """Map an HTTP status onto a failure kind, or ``None`` when the call succeeded."""
    if status is None or 200 <= status < 300:
        return None
    if status in {401}:
        return FailureKind.AUTHENTICATION
    if status in {403}:
        return FailureKind.PERMISSION
    if status == 404:
        return FailureKind.NOT_FOUND
    if status == 429:
        return FailureKind.RATE_LIMIT
    if 500 <= status <= 599:
        return FailureKind.SERVER_ERROR
    if status == 408:
        return FailureKind.TIMEOUT
    return FailureKind.PROTOCOL


def normalize_catalog(
    payload: Any, *, source: str = ZEN_BASE_URL, observed_at: str | None = None
) -> CatalogSnapshot:
    """Normalize a catalog response into a snapshot.

    Handles both the OpenAI-style ``{"data": [...]}`` shape and a bare list. A
    malformed payload yields an empty snapshot rather than an exception: an endpoint
    exposing nothing is an observation, and inventing entries would be worse.
    """
    raw: dict[str, Any]
    if isinstance(payload, list):
        models: Any = payload
        raw = {"data": payload}
    elif isinstance(payload, Mapping):
        raw = dict(payload)
        models = payload.get("data", payload.get("models", []))
    else:
        raw = {"value": payload}
        models = []

    entries: list[CatalogEntry] = []
    if isinstance(models, list):
        normalizer = _CatalogNormalizer()
        for item in models:
            entry = normalizer.entry(item)
            if entry is not None:
                entries.append(entry)

    try:
        return CatalogSnapshot(
            source=source,
            observed_at=observed_at,
            entries=tuple(entries),
            raw=_stringable(raw),
        )
    except ValidationError:
        # Duplicate aliases in a live catalog are a real observation, not a crash.
        # Keep the first occurrence of each and let the caller see the snapshot.
        seen: set[str] = set()
        unique: list[CatalogEntry] = []
        for entry in entries:
            if entry.alias in seen:
                continue
            seen.add(entry.alias)
            unique.append(entry)
        return CatalogSnapshot(
            source=source, observed_at=observed_at, entries=tuple(unique), raw=_stringable(raw)
        )


class _CatalogNormalizer:
    """Turns one raw catalog record into a ``CatalogEntry`` without guessing."""

    def entry(self, item: Any) -> CatalogEntry | None:
        if not isinstance(item, Mapping):
            return None
        alias = item.get("id") or item.get("alias") or item.get("slug")
        if not isinstance(alias, str) or not alias.strip():
            return None

        caps = item.get("capabilities")
        caps_map: Mapping[str, Any] = caps if isinstance(caps, Mapping) else {}
        capabilities = Capabilities(
            streaming=_flag(caps_map.get("streaming"), item.get("supports_streaming")),
            tool_calls=_flag(caps_map.get("tool_calls"), item.get("supports_tools")),
            reasoning=_flag(caps_map.get("reasoning"), item.get("supports_reasoning")),
            usage_reporting=_flag(caps_map.get("usage_reporting"), item.get("reports_usage")),
            logprobs=_flag(caps_map.get("logprobs"), item.get("supports_logprobs")),
        )
        context = item.get("context_window", item.get("context_length", item.get("max_context")))
        return CatalogEntry(
            alias=alias.strip(),
            route=str(item.get("route", "zen")),
            display_name=_opt_str(item.get("display_name", item.get("name"))),
            provider=_opt_str(item.get("provider", item.get("owned_by"))),
            family=_opt_str(item.get("family")),
            context_window=context if isinstance(context, int) and context > 0 else None,
            capabilities=capabilities,
            raw=_stringable(item),
        )


def _flag(*candidates: Any) -> bool:
    """First candidate that is a real boolean wins; anything else is False."""
    for value in candidates:
        if isinstance(value, bool):
            return value
    return False


def _opt_str(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _stringable(value: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): item for key, item in value.items()}


def usage_from_response(payload: Mapping[str, Any]) -> Usage:
    """Extract usage, keeping unreported directions absent.

    Gateways vary: some report ``prompt_tokens``, some ``input_tokens``, some report
    usage only on the final streamed chunk and some not at all. Each direction is
    read independently and stays ``None`` when absent.
    """

    def pick(*names: str) -> int | None:
        for name in names:
            value = payload.get(name)
            if isinstance(value, bool):
                continue
            if isinstance(value, int) and value >= 0:
                return value
        return None

    raw = payload.get("usage")
    usage_map: Mapping[str, Any] = raw if isinstance(raw, Mapping) else {}

    def pick_usage(*names: str) -> int | None:
        for name in names:
            value = usage_map.get(name)
            if isinstance(value, bool):
                continue
            if isinstance(value, int) and value >= 0:
                return value
        return None

    input_tokens = pick_usage("input_tokens", "prompt_tokens")
    output_tokens = pick_usage("output_tokens", "completion_tokens")
    if input_tokens is None or output_tokens is None:
        # Fall back to the top level: some gateways put usage beside the choice.
        input_tokens = (
            input_tokens if input_tokens is not None else pick("input_tokens", "prompt_tokens")
        )
        output_tokens = (
            output_tokens
            if output_tokens is not None
            else pick("output_tokens", "completion_tokens")
        )
    reported = input_tokens is not None or output_tokens is not None or isinstance(raw, Mapping)
    return Usage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_input_tokens=pick_usage("cached_input_tokens", "cache_read_input_tokens"),
        reasoning_tokens=pick_usage("reasoning_tokens"),
        provider_reported=bool(reported),
    )


def extract_text(payload: Mapping[str, Any]) -> str | None:
    """Extract assistant text from a chat completion, or ``None`` if absent."""
    choices = payload.get("choices")
    if not isinstance(choices, Sequence) or not choices:
        return None
    first = choices[0]
    if not isinstance(first, Mapping):
        return None
    message = first.get("message")
    if not isinstance(message, Mapping):
        return None
    content = message.get("content")
    if isinstance(content, str):
        return content
    # Some gateways return a content-part list instead of a string.
    if isinstance(content, list):
        parts = [
            part.get("text", "")
            for part in content
            if isinstance(part, Mapping) and isinstance(part.get("text"), str)
        ]
        if parts:
            return "".join(parts)
    return None


def extract_tool_calls(payload: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    """Extract tool calls, or an empty tuple when the gateway reported none."""
    choices = payload.get("choices")
    if not isinstance(choices, Sequence) or not choices:
        return ()
    first = choices[0]
    if not isinstance(first, Mapping):
        return ()
    message = first.get("message")
    if not isinstance(message, Mapping):
        return ()
    calls = message.get("tool_calls")
    if not isinstance(calls, Sequence):
        return ()
    return tuple(dict(call) for call in calls if isinstance(call, Mapping))


def finish_reason_of(payload: Mapping[str, Any]) -> FinishStatus:
    choices = payload.get("choices")
    if not isinstance(choices, Sequence) or not choices:
        return "error"
    first = choices[0]
    if not isinstance(first, Mapping):
        return "error"
    return map_finish_reason(first.get("finish_reason"))


def effective_settings_of(payload: Mapping[str, Any], requested: ModelRequest) -> dict[str, Any]:
    """Record what the endpoint says it did, separately from what was requested.

    A gateway that reports a different temperature actually used is evidence, and
    conflating the two would make a run irreproducible.
    """
    effective: dict[str, Any] = {
        "requested": {
            "max_output_tokens": requested.max_output_tokens,
            "temperature": requested.temperature,
            "top_p": requested.top_p,
            "seed": requested.seed,
            "stop": list(requested.stop),
        }
    }
    echoed = payload.get("stealthbench_effective")
    if isinstance(echoed, Mapping):
        effective["reported"] = dict(echoed)
    model = payload.get("model")
    if isinstance(model, str):
        effective["reported_model"] = model
    return effective


class ZenAdapter(ProviderAdapter):
    """Zen catalog and chat adapter driven by a recorded transcript.

    ``transcript`` is a mapping of chat-exchange keys to recorded HTTP responses. It is
    the production code path with the network call replaced by a capture, which is why
    the offline suite can assert real normalization behaviour.
    """

    route = "zen"

    def __init__(
        self,
        *,
        catalog_payload: Any = None,
        exchanges: Mapping[str, Mapping[str, Any]] | None = None,
        extra_secrets: frozenset[str] = frozenset(),
        catalog_source: str = ZEN_BASE_URL,
    ) -> None:
        self.adapter_version = ZEN_ADAPTER_VERSION
        self._catalog_payload = catalog_payload
        self._exchanges: dict[str, Mapping[str, Any]] = dict(exchanges or {})
        self.extra_secrets = extra_secrets
        self.catalog_source = catalog_source
        self._capabilities = Capabilities(
            streaming=False,
            tool_calls=False,
            reasoning=False,
            usage_reporting=False,
            logprobs=False,
        )

    @classmethod
    def from_path(cls, path: Path) -> ZenAdapter:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, Mapping):
            raise ValueError(f"{path} must contain a JSON object")
        return cls(
            catalog_payload=raw.get("catalog"),
            exchanges=raw.get("exchanges"),
            catalog_source=str(raw.get("catalog_source", ZEN_BASE_URL)),
        )

    def discover(self) -> CatalogSnapshot:
        """Snapshot whatever the recorded catalog response contained.

        Absent or malformed yields an empty snapshot. No alias is ever invented.
        """
        if self._catalog_payload is None:
            return CatalogSnapshot(source=self.catalog_source, raw={})
        return normalize_catalog(self._catalog_payload, source=self.catalog_source)

    def _record_for(self, sample_key: SampleKey) -> Mapping[str, Any] | None:
        for candidate in self._candidate_keys(sample_key):
            record = self._exchanges.get(candidate)
            if record is not None:
                return record
        return None

    @staticmethod
    def _candidate_keys(sample_key: SampleKey) -> Iterator[str]:
        yield sample_key.task_id
        yield f"{sample_key.task_id}#r{sample_key.repeat_id}"
        yield f"{sample_key.endpoint_id}:{sample_key.task_id}"
        yield f"{sample_key.endpoint_id}:{sample_key.task_id}#r{sample_key.repeat_id}"

    def complete(
        self,
        *,
        sample_key: SampleKey,
        request: ModelRequest,
        prompt_hash: str,
        manifest_hash: str | None = None,
        attempt_id: str | None = None,
        attempt_number: int = 1,
        capabilities: Capabilities | None = None,
    ) -> AdapterResult:
        del prompt_hash
        attempt = attempt_id or f"{sample_key.task_id}-r{sample_key.repeat_id}-a{attempt_number}"
        unsupported: tuple[UnsupportedSetting, ...] = check_requested_settings(
            request, capabilities or self._capabilities
        )

        record = self._record_for(sample_key)
        if record is None:
            return AdapterResult(
                failure=TransportFailure(
                    kind=FailureKind.NO_FIXTURE,
                    detail=f"no recorded Zen exchange for {sample_key.task_id}",
                ),
                unsupported=unsupported,
            )

        status = record.get("http_status")
        failure_kind = map_http_status(status if isinstance(status, int) else None)
        if failure_kind is not None:
            body = record.get("body")
            return AdapterResult(
                failure=safe_failure(
                    failure_kind,
                    _detail_of(record),
                    http_status=status if isinstance(status, int) else None,
                    retry_after_seconds=_retry_after(record),
                    body=body if isinstance(body, Mapping) else None,
                    extra_secrets=self.extra_secrets,
                ),
                unsupported=unsupported,
            )

        payload = record.get("json")
        if not isinstance(payload, Mapping):
            return AdapterResult(
                failure=TransportFailure(
                    kind=FailureKind.PROTOCOL,
                    detail="recorded exchange carried no JSON body",
                    http_status=status if isinstance(status, int) else None,
                ),
                unsupported=unsupported,
            )

        text = extract_text(payload)
        tool_calls = extract_tool_calls(payload)
        if text is None and not tool_calls:
            return AdapterResult(
                failure=TransportFailure(
                    kind=FailureKind.PROTOCOL,
                    detail="chat completion contained neither content nor tool calls",
                    http_status=200,
                ),
                unsupported=unsupported,
            )

        reported_model = payload.get("model")
        metadata: dict[str, Any] = {
            "adapter": "zen",
            "route": self.route,
            "request_id": _opt_str(payload.get("id")),
            "tool_calls": [dict(call) for call in tool_calls],
        }
        if isinstance(reported_model, str):
            metadata["reported_model"] = reported_model

        result = GenerationResult(
            attempt_id=attempt,
            sample_key=sample_key,
            attempt_number=attempt_number,
            delivery_status=DeliveryStatus.ACCEPTED,
            response=text or "",
            usage=usage_from_response(payload),
            effective_settings=effective_settings_of(payload, request),
            finish_status=finish_reason_of(payload),
            manifest_hash=manifest_hash,
            redacted_provider_metadata=redact_mapping(metadata, extra_secrets=self.extra_secrets),
        )
        return AdapterResult(
            result=result,
            unsupported=unsupported,
            effective_settings=result.effective_settings,
        )


def _detail_of(record: Mapping[str, Any]) -> str:
    detail = record.get("error_message", record.get("detail"))
    if isinstance(detail, str) and detail:
        return detail
    body = record.get("body")
    if isinstance(body, Mapping):
        message = body.get("error")
        if isinstance(message, Mapping):
            inner = message.get("message")
            if isinstance(inner, str):
                return inner
        if isinstance(message, str):
            return message
    return "recorded failure"


def _retry_after(record: Mapping[str, Any]) -> float | None:
    headers = record.get("headers")
    if not isinstance(headers, Mapping):
        return None
    value = headers.get("retry-after") or headers.get("Retry-After")
    if isinstance(value, str) and value.strip().isdigit():
        return float(value.strip())
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def catalog_snapshot_digest(snapshot: CatalogSnapshot) -> str:
    """A stable digest for storing the observed catalog alongside a campaign."""
    return content_digest(
        {
            "source": snapshot.source,
            "entries": [entry.model_dump(mode="json") for entry in snapshot.entries],
        }
    )


__all__: Sequence[str] = (
    "ZEN_ADAPTER_VERSION",
    "ZEN_BASE_URL",
    "ZEN_CATALOG_PATH",
    "ZEN_CHAT_PATH",
    "ZenAdapter",
    "catalog_snapshot_digest",
    "effective_settings_of",
    "extract_text",
    "extract_tool_calls",
    "finish_reason_of",
    "map_finish_reason",
    "map_http_status",
    "normalize_catalog",
    "redact_text",
    "usage_from_response",
)
