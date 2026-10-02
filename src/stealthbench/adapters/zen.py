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

from stealthbench.adapters.base import _ABSENT as _ABSENT
from stealthbench.adapters.base import (
    AdapterResult,
    CatalogEntry,
    CatalogSnapshot,
    FailureKind,
    ProviderAdapter,
    TransportFailure,
    UnsupportedSetting,
    _alias_of,
    check_requested_settings,
    read_terminated,
    safe_failure,
    strict_flag,
)
from stealthbench.schemas.campaign import Capabilities
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


def map_finish_reason(reason: str | None) -> FinishStatus | None:
    """Map a gateway finish reason onto the frozen vocabulary.

    An absent reason stays absent. ``None`` means the endpoint never said how the
    generation ended, and reporting that as ``stop`` would turn a cut-off stream
    into a counted success -- the exact failure this function exists to prevent.

    An unrecognised reason becomes ``error`` rather than a success reason either.
    """
    if reason is None:
        return None
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
    payload: Any,
    *,
    source: str = ZEN_BASE_URL,
    observed_at: str | None = None,
    extra_secrets: frozenset[str] = frozenset(),
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
            entry = normalizer.entry(item, extra_secrets=extra_secrets)
            if entry is not None:
                entries.append(entry)

    try:
        return CatalogSnapshot(
            source=source,
            observed_at=observed_at,
            entries=tuple(entries),
            raw=redact_mapping(_stringable(raw), extra_secrets=extra_secrets),
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
            source=source,
            observed_at=observed_at,
            entries=tuple(unique),
            raw=redact_mapping(_stringable(raw), extra_secrets=extra_secrets),
        )


class _CatalogNormalizer:
    """Turns one raw catalog record into a ``CatalogEntry`` without guessing."""

    def entry(
        self, item: Any, *, extra_secrets: frozenset[str] = frozenset()
    ) -> CatalogEntry | None:
        if not isinstance(item, Mapping):
            return None
        alias = _alias_of(item)
        if alias is None:
            return None

        caps = item.get("capabilities")
        caps_map: Mapping[str, Any] = caps if isinstance(caps, Mapping) else {}
        capabilities = Capabilities(
            streaming=strict_flag(caps_map.get("streaming"), item.get("supports_streaming")),
            tool_calls=strict_flag(caps_map.get("tool_calls"), item.get("supports_tools")),
            reasoning=strict_flag(caps_map.get("reasoning"), item.get("supports_reasoning")),
            usage_reporting=strict_flag(caps_map.get("usage_reporting"), item.get("reports_usage")),
            logprobs=strict_flag(caps_map.get("logprobs"), item.get("supports_logprobs")),
        )
        context = item.get("context_window", item.get("context_length", item.get("max_context")))
        raw_entry = _stringable(item)
        return CatalogEntry(
            alias=alias.strip(),
            route=_route_label(item.get("route")),
            display_name=_opt_str(item.get("display_name", item.get("name"))),
            provider=_opt_str(item.get("provider", item.get("owned_by"))),
            family=_opt_str(item.get("family")),
            # A bool is an int in Python; `True` as a context window would be stored
            # as 1 token, which is a fabricated measurement.
            context_window=(
                context
                if isinstance(context, int) and not isinstance(context, bool) and context > 0
                else None
            ),
            capabilities=capabilities,
            raw=redact_mapping(raw_entry, extra_secrets=extra_secrets),
        )


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


def finish_reason_of(payload: Mapping[str, Any]) -> FinishStatus | None:
    choices = payload.get("choices")
    if not isinstance(choices, Sequence) or not choices:
        return "error"
    first = choices[0]
    if not isinstance(first, Mapping):
        return "error"
    return map_finish_reason(first.get("finish_reason"))


def effective_settings_of(
    payload: Mapping[str, Any],
    requested: ModelRequest,
    *,
    extra_secrets: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """Record what the endpoint says it did, separately from what was requested.

    A gateway that reports a different temperature actually used is evidence, and
    conflating the two would make a run irreproducible.

    Redaction happens here rather than at each call site: a gateway echoing a
    request that carried an authorization header would otherwise write the
    credential straight into the artifact.
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
        effective["reported"] = redact_mapping(dict(echoed), extra_secrets=extra_secrets)
    model = payload.get("model")
    if isinstance(model, str):
        effective["reported_model"] = redact_text(model, extra_secrets=extra_secrets)
    return redact_mapping(effective, extra_secrets=extra_secrets)


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
        endpoint_id: str | None = None,
    ) -> None:
        self.adapter_version = ZEN_ADAPTER_VERSION
        self._endpoint_id = endpoint_id
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
    def from_path(cls, path: Path, *, extra_secrets: frozenset[str] = frozenset()) -> ZenAdapter:
        """Load a recorded transcript.

        The catalog is taken from the recorded ``GET /v1/models`` response body when
        one is present, so a wire capture is replayed as captured instead of relying
        on a separately maintained summary of it.

        ``extra_secrets`` must be supplied by the caller when the capture contains a
        credential-shaped value; there is no way to redact what was never declared.
        """
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, Mapping):
            raise ValueError(f"{path} must contain a JSON object")
        catalog = raw.get("catalog")
        if catalog is None:
            catalog = _catalog_from_requests(raw.get("requests"))
        bound = raw.get("endpoint_id")
        return cls(
            catalog_payload=catalog,
            exchanges=raw.get("exchanges"),
            catalog_source=str(raw.get("catalog_source", ZEN_BASE_URL)),
            extra_secrets=extra_secrets,
            endpoint_id=bound if isinstance(bound, str) and bound.strip() else None,
        )

    def discover(self) -> CatalogSnapshot:
        """Snapshot whatever the recorded catalog response contained.

        Absent or malformed yields an empty snapshot. No alias is ever invented.
        """
        if self._catalog_payload is None:
            return CatalogSnapshot(source=self.catalog_source, raw={})
        return normalize_catalog(
            self._catalog_payload,
            source=self.catalog_source,
            extra_secrets=self.extra_secrets,
        )

    def _record_for(self, sample_key: SampleKey) -> Mapping[str, Any] | None:
        allow_unqualified = self._allows_unqualified(sample_key)
        for candidate in self._candidate_keys(sample_key, allow_unqualified=allow_unqualified):
            record = self._exchanges.get(candidate)
            if record is not None:
                return record
        return None

    def _sole_alias(self) -> str | None:
        """The one alias a capture can be bound to, or ``None`` if that is unclear.

        A capture whose catalog names exactly one alias binds an endpoint-unqualified
        key to that alias and to no other: an unrelated endpoint would be an endpoint
        observation that was never dispatched.
        """
        aliases = {
            entry.alias
            for entry in normalize_catalog(
                self._catalog_payload if isinstance(self._catalog_payload, Mapping) else {}
            ).entries
        }
        return next(iter(aliases)) if len(aliases) == 1 else None

    def _aliases_are_unambiguous(self) -> bool:
        """Whether an endpoint-unqualified transcript key can mean only one alias.

        A capture keyed ``ifeval::item-1`` against a catalog listing three aliases
        cannot say which endpoint produced the answer. Replaying it for each of them
        would record accepted samples for endpoint observations never dispatched.

        The aliases are read through the same normalizer the snapshot uses, so the
        two cannot disagree about what a catalog contains.
        """
        aliases = {
            entry.alias
            for entry in normalize_catalog(
                self._catalog_payload if isinstance(self._catalog_payload, Mapping) else {}
            ).entries
        }
        # No catalog evidence is ambiguity: nothing says which endpoint answered.
        return len(aliases) == 1

    def _allows_unqualified(self, sample_key: SampleKey) -> bool:
        """Whether an unqualified key is bound to this sample key's endpoint.

        Either the adapter was built for a named endpoint and this sample is for it,
        or the capture's catalog holds exactly one alias. Anything else is ambiguous.
        """
        if self._endpoint_id is not None:
            return self._endpoint_id == sample_key.endpoint_id
        return self._sole_alias() == sample_key.endpoint_id and bool(sample_key.endpoint_id)

    @staticmethod
    def _candidate_keys(sample_key: SampleKey, *, allow_unqualified: bool = True) -> Iterator[str]:
        """Transcript lookup keys, most specific first.

        A record pinned to a repeat must win over a generic one, otherwise every
        repeat replays the same response and a repeat measurement collapses into a
        duplicate.

        The endpoint-unqualified keys are only offered when the capture has a single
        alias. With several aliases in the catalog they are ambiguous, and matching
        them would replay one capture as a sample for every endpoint in the campaign.
        """
        yield f"{sample_key.endpoint_id}:{sample_key.task_id}#r{sample_key.repeat_id}"
        yield f"{sample_key.task_id}#r{sample_key.repeat_id}"
        yield f"{sample_key.endpoint_id}:{sample_key.task_id}"
        if allow_unqualified:
            yield sample_key.task_id

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
        unsupported = unsupported + _recorded_unsupported(record, request)
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
            effective_settings=effective_settings_of(
                payload, request, extra_secrets=self.extra_secrets
            ),
            finish_status=finish_reason_of(payload),
            manifest_hash=manifest_hash,
            redacted_provider_metadata=redact_mapping(metadata, extra_secrets=self.extra_secrets),
        )
        return AdapterResult(
            result=result,
            unsupported=unsupported,
            effective_settings=result.effective_settings,
        )

    def stream(
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
        """Replay a recorded Zen SSE stream.

        The route label is ``zen``, so a gateway stream is distinguishable from a
        fixture stream of the same bytes.
        """
        from stealthbench.adapters.streaming import parse_stream

        attempt = attempt_id or f"{sample_key.task_id}-r{sample_key.repeat_id}-a{attempt_number}"
        unsupported: tuple[UnsupportedSetting, ...] = check_requested_settings(
            request, capabilities or self._capabilities, stream=True
        )
        record = self._record_for(sample_key)
        if record is None:
            return AdapterResult(
                failure=safe_failure(
                    FailureKind.NO_FIXTURE,
                    f"no recorded Zen stream for {sample_key.task_id}",
                )
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
                    body=body if isinstance(body, Mapping) else None,
                    extra_secrets=self.extra_secrets,
                )
            )
        frames = record.get("stream_frames")
        if not isinstance(frames, Sequence) or not frames:
            return AdapterResult(
                failure=safe_failure(
                    FailureKind.UNSUPPORTED_SETTING,
                    "recorded Zen exchange carries no stream frames",
                    extra_secrets=self.extra_secrets,
                )
            )
        if capabilities is not None and not capabilities.streaming:
            return AdapterResult(
                failure=safe_failure(
                    FailureKind.UNSUPPORTED_SETTING,
                    "streaming was requested but this endpoint does not advertise it",
                    extra_secrets=self.extra_secrets,
                )
            )
        raw = b"".join(
            b"data: " + json.dumps(dict(frame), ensure_ascii=False).encode("utf-8") + b"\n\n"
            for frame in frames
            if isinstance(frame, Mapping)
        )
        # A capture whose connection died never sent the sentinel; replaying it as
        # terminated would report a truncated answer as a completed one.
        if "stream_terminated" not in record or _was_terminated(record["stream_terminated"]):
            raw += b"data: [DONE]\n\n"
        outcome, _assembly = parse_stream(
            [raw],
            sample_key=sample_key,
            attempt_id=attempt,
            attempt_number=attempt_number,
            manifest_hash=manifest_hash,
            route=self.route,
            adapter="zen",
            extra_secrets=self.extra_secrets,
        )
        return AdapterResult(
            result=outcome.result,
            failure=outcome.failure,
            unsupported=unsupported + _recorded_unsupported(record, request),
            effective_settings=outcome.effective_settings,
        )


def _recorded_unsupported(
    record: Mapping[str, Any] | None, request: ModelRequest
) -> tuple[UnsupportedSetting, ...]:
    """Settings the recorded gateway declined, named with what was asked for.

    The capture is the only evidence of what a gateway will not do, so it is read
    here rather than inferred from a capability that was never advertised.
    """
    if record is None:
        return ()
    names = record.get("unsupported_settings")
    if not isinstance(names, Sequence) or isinstance(names, (str, bytes)):
        return ()
    # What the request asked for, read from the request itself. Reporting a hardcoded
    # False for a setting the request set to True writes a measurement never made.
    from stealthbench.adapters.base import _requested_settings

    requested = _requested_settings(request)
    return tuple(
        UnsupportedSetting(
            setting=str(name),
            requested=requested.get(str(name)),
            reason="the recorded gateway does not offer this setting",
        )
        for name in names
        if isinstance(name, str)
    )


def _was_terminated(value: Any = _ABSENT) -> bool:
    """Whether a capture recorded its terminating sentinel.

    Delegates to the shared reader the fixture schema also uses, so the two adapters
    cannot return opposite verdicts on the same capture.
    """
    return read_terminated(value)


def _route_label(value: Any) -> str:
    """The route a record names, or the default when it names none."""
    return value.strip() if isinstance(value, str) and value.strip() else "zen"


def _catalog_from_requests(requests: Any) -> Any:
    """Extract the catalog body from recorded requests, if the capture has any.

    A wire capture records ``GET /v1/models`` as one entry among many; replaying the
    recorded body is what makes discovery-from-fixtures evidence rather than a
    convenience. The first successful models response wins; the status is checked
    so an error response is never mistaken for an empty catalog.
    """
    if not isinstance(requests, Sequence) or isinstance(requests, (str, bytes)):
        return None
    for entry in requests:
        if not isinstance(entry, Mapping):
            continue
        path = entry.get("path")
        if not isinstance(path, str):
            continue
        # A capture records whatever the client requested, query string and all.
        bare = path.split("?", 1)[0].rstrip("/")
        if not bare.endswith("/models"):
            continue
        status = entry.get("status")
        if status is not None and (
            not isinstance(status, int) or isinstance(status, bool) or not 200 <= status < 300
        ):
            continue
        body = entry.get("body")
        if isinstance(body, Mapping):
            return body
    return None


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
    """A stable digest for storing the observed catalog alongside a campaign.

    It delegates to the snapshot so the two can never disagree: a campaign that
    verifies one while storing the other could never be checked.
    """
    return snapshot.digest()


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
