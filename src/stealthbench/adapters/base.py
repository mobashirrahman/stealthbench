"""Provider adapter contract and the offline fixture transport.

Every adapter answers the same three questions — what does this endpoint offer, what
does its catalog look like, and what happened when we dispatched — and answers them
through the frozen result contracts rather than a shape of its own.

The fixture transport is the reference implementation of that contract. It exists so
that every scenario the gate cares about (a 429, a 5xx, an authentication failure, a
stream interrupted mid-flight, usage the endpoint never reported, a setting the
endpoint does not support) can be *recorded once and replayed*, and so that an offline
run exercises the same code path a live one does.

Three rules the base class enforces for every adapter:

* **Capabilities are observed, never inferred.** A capability is true only if the
  catalog or the endpoint said so.
* **Unsupported is not the same as ignored.** A requested setting the endpoint does
  not offer is reported, never silently dropped.
* **A failure is a result, not an exception.** Transport failures surface as
  ``GenerationResult`` with a non-accepted delivery status, so a failure lands in the
  denominators it belongs to instead of aborting a campaign.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Final

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from stealthbench.schemas.campaign import Capabilities, EndpointSpec
from stealthbench.schemas.hashing import content_digest
from stealthbench.schemas.results import (
    DeliveryStatus,
    FinishStatus,
    GenerationResult,
    ModelRequest,
    SampleKey,
    StreamingMeasurements,
    Usage,
)
from stealthbench.storage.events import redact_mapping, redact_text

FIXTURE_SCHEMA_VERSION: Final[str] = "1.0"

#: Generation settings an adapter may be asked to honour. Anything outside this set is
#: a caller bug rather than an unsupported feature.
#: The settings a request may carry. ``ModelRequest`` is closed, so this is a
#: restatement of the schema rather than a separate source of truth; the assertion
#: below keeps the two from drifting apart silently.
KNOWN_SETTINGS: Final[frozenset[str]] = frozenset(
    {"max_output_tokens", "temperature", "top_p", "seed", "stream", "stop"}
)
assert set(ModelRequest.model_fields) - {"messages"} == KNOWN_SETTINGS, (
    "the reported setting vocabulary must match the closed request schema"
)


class FailureKind(StrEnum):
    """Why a dispatch did not produce a usable response.

    These map onto the frozen statuses: a transport failure is never an incorrect
    answer, and never an accepted sample.
    """

    AUTHENTICATION = "authentication"
    PERMISSION = "permission"
    RATE_LIMIT = "rate_limit"
    SERVER_ERROR = "server_error"
    TIMEOUT = "timeout"
    PROTOCOL = "protocol"
    UNSUPPORTED_SETTING = "unsupported_setting"
    INTERRUPTED = "interrupted"
    NOT_FOUND = "not_found"
    NO_FIXTURE = "no_fixture"


#: Which failures are worth retrying. A 4xx that is not a rate limit, and a protocol
#: error, will not fix itself; retrying it only spends tokens and time.
RETRYABLE_FAILURES: Final[frozenset[FailureKind]] = frozenset(
    {FailureKind.RATE_LIMIT, FailureKind.SERVER_ERROR, FailureKind.TIMEOUT}
)


@dataclass(frozen=True, slots=True)
class TransportFailure:
    """A dispatch that did not produce a usable response."""

    kind: FailureKind
    detail: str
    http_status: int | None = None
    retry_after_seconds: float | None = None
    body: Mapping[str, Any] | None = None

    @property
    def retryable(self) -> bool:
        return self.kind in RETRYABLE_FAILURES

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": str(self.kind),
            "detail": self.detail,
            "http_status": self.http_status,
            "retry_after_seconds": self.retry_after_seconds,
            "retryable": self.retryable,
            "body": dict(self.body) if self.body else None,
        }


@dataclass(frozen=True, slots=True)
class UnsupportedSetting:
    """A requested generation setting the endpoint does not offer."""

    setting: str
    requested: Any
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {"setting": self.setting, "requested": self.requested, "reason": self.reason}


class CatalogEntry(BaseModel):
    """One entry as the catalog actually reported it.

    Capability booleans are required and default to nothing being claimed: an adapter
    that fails to parse a capability leaves it false rather than guessing true.
    """

    model_config = ConfigDict(extra="allow", frozen=True)

    alias: str
    route: str = "zen"
    display_name: str | None = None
    provider: str | None = None
    family: str | None = None
    context_window: int | None = Field(default=None, gt=0)
    capabilities: Capabilities
    raw: Mapping[str, Any] = Field(default_factory=dict)

    @property
    def supports_streaming(self) -> bool:
        return self.capabilities.streaming


class CatalogSnapshot(BaseModel):
    """A point-in-time view of an endpoint catalog.

    An alias is not an immutable model version, so the snapshot is hashed and stored
    with every campaign rather than being treated as configuration.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = FIXTURE_SCHEMA_VERSION
    source: str
    observed_at: str | None = None
    entries: tuple[CatalogEntry, ...] = ()
    raw: Mapping[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _aliases_are_unique(self) -> CatalogSnapshot:
        aliases = [entry.alias for entry in self.entries]
        duplicates = sorted({alias for alias in aliases if aliases.count(alias) > 1})
        if duplicates:
            raise ValueError(f"duplicate catalog aliases: {duplicates}")
        return self

    def aliases(self) -> tuple[str, ...]:
        return tuple(entry.alias for entry in self.entries)

    def get(self, alias: str) -> CatalogEntry | None:
        return next((entry for entry in self.entries if entry.alias == alias), None)

    @property
    def is_empty(self) -> bool:
        return not self.entries

    def digest(self) -> str:
        """One digest per snapshot content.

        ``raw`` is included because a per-entry copy of it is: a catalog whose raw
        record changed is a different observation even if the entries agree.
        """
        return content_digest(
            {
                "source": self.source,
                "observed_at": self.observed_at,
                "entries": [entry.model_dump(mode="json") for entry in self.entries],
                "raw": self.raw,
            }
        )

    def to_endpoint_specs(self) -> tuple[EndpointSpec, ...]:
        """Turn catalog entries into manifest endpoints.

        No credential reference is invented here: a fixture catalog carries none, and
        a live one is resolved by the operator at run time.
        """
        return tuple(
            EndpointSpec(
                endpoint_id=entry.alias,
                alias=entry.alias,
                route=entry.route,
                transport="fixture",
                capabilities=entry.capabilities,
                label_provider=entry.provider,
                label_family=entry.family,
            )
            for entry in self.entries
        )


class AdapterResult(BaseModel):
    """What one dispatch produced, successful or not.

    Wrapping the generation result rather than raising keeps a transport failure in
    the denominators it belongs to.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    result: GenerationResult | None = None
    failure: TransportFailure | None = None
    unsupported: tuple[UnsupportedSetting, ...] = ()
    effective_settings: Mapping[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _exactly_one_outcome(self) -> AdapterResult:
        if (self.result is None) == (self.failure is None):
            raise ValueError(
                "an adapter outcome carries either a generation result or a failure, not both "
                "and not neither"
            )
        return self

    @property
    def ok(self) -> bool:
        return self.failure is None and self.result is not None

    @property
    def delivery_status(self) -> DeliveryStatus | None:
        return self.result.delivery_status if self.result else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "result": self.result.model_dump(mode="json") if self.result else None,
            "failure": self.failure.to_dict() if self.failure else None,
            "unsupported": [item.to_dict() for item in self.unsupported],
            "effective_settings": dict(self.effective_settings),
        }


class ProviderAdapter(ABC):
    """Base class for every endpoint adapter."""

    #: Stable identifier for the adapter implementation, recorded in results.
    adapter_version: str = "0.0.0"
    #: Route label. Distinct routes stay distinct labels even for the same model.
    route: str = "unknown"

    @abstractmethod
    def discover(self) -> CatalogSnapshot:
        """Snapshot the endpoint catalog.

        Never hardcoded: the snapshot is what the endpoint reported at this moment.
        """

    @abstractmethod
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
        """Dispatch one non-streaming completion."""

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
        """Dispatch one streaming completion.

        The base implementation refuses rather than pretending to stream, so an
        adapter that has not implemented it cannot silently return a non-streamed
        result labelled as streamed.
        """
        del sample_key, request, prompt_hash, manifest_hash
        del attempt_id, attempt_number, capabilities
        return AdapterResult(
            failure=TransportFailure(
                kind=FailureKind.UNSUPPORTED_SETTING,
                detail=(
                    f"adapter {type(self).__name__} does not implement streaming; a non-streamed "
                    "response must not be labelled as streamed"
                ),
            )
        )


def check_requested_settings(
    request: ModelRequest, capabilities: Capabilities | None, *, stream: bool = False
) -> tuple[UnsupportedSetting, ...]:
    """Report requested settings the endpoint does not offer.

    A setting the endpoint lacks is reported, never dropped: silently ignoring
    ``stream=True`` on a non-streaming endpoint would record measurements that never
    happened.
    """
    unsupported: list[UnsupportedSetting] = []
    if capabilities is not None and stream and not capabilities.streaming:
        unsupported.append(
            UnsupportedSetting(
                setting="stream",
                requested=True,
                reason="the endpoint's catalog does not advertise streaming",
            )
        )
    return tuple(unsupported)


def failed_result(
    sample_key: SampleKey,
    attempt_id: str,
    attempt_number: int,
    failure: TransportFailure,
    *,
    status: DeliveryStatus = DeliveryStatus.TRANSPORT_FAILED,
    usage: Usage | None = None,
    manifest_hash: str | None = None,
) -> GenerationResult:
    """Build the result record for a dispatch that did not complete.

    Usage from a failed attempt is preserved when the provider reported it, because a
    billed-but-failed request still costs money and that must remain visible.
    """
    return GenerationResult(
        attempt_id=attempt_id,
        sample_key=sample_key,
        attempt_number=attempt_number,
        delivery_status=status,
        response=None,
        usage=usage or Usage(),
        manifest_hash=manifest_hash,
        redacted_provider_metadata={"failure": failure.to_dict()},
    )


# ---------------------------------------------------------------------------
# Fixture transport
# ---------------------------------------------------------------------------


class RecordedExchange(BaseModel):
    """One recorded request/response pair, or one recorded failure."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    endpoint_id: str
    benchmark_id: str
    item_id: str
    repeat_id: int = Field(default=1, ge=1)
    outcome: str = "response"
    response: str | None = None
    usage: Mapping[str, Any] | None = None
    usage_reported: bool = True
    effective_settings: Mapping[str, Any] = Field(default_factory=dict)
    streaming: Mapping[str, Any] | None = None
    #: Absent by default: a capture that does not say how the generation ended must
    #: not be recorded as a clean stop.
    finish_status: FinishStatus | None = None
    tool_calls: tuple[Mapping[str, Any], ...] = ()
    #: Recorded SSE frames, when this exchange was a streamed one.
    stream_frames: tuple[Mapping[str, Any], ...] = ()
    #: Whether the recorded stream sent its terminating ``data: [DONE]`` sentinel.
    #: A capture taken from a connection that died never did, and replaying it as
    #: complete would invent a finished answer out of a truncated one.
    stream_terminated: bool = True

    @field_validator("stream_terminated", mode="before")
    @classmethod
    def _read_terminated(cls, value: Any) -> Any:
        """Read the sentinel flag exactly as the Zen route reads it."""
        return read_terminated(value)

    failure_kind: str | None = None
    failure_detail: str | None = None
    http_status: int | None = None
    retry_after_seconds: float | None = None
    error_body: Mapping[str, Any] | None = None
    unsupported_settings: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _outcome_is_coherent(self) -> RecordedExchange:
        if self.outcome == "response" and self.failure_kind:
            raise ValueError("a recorded response cannot also carry a failure kind")
        if self.outcome == "error" and not self.failure_kind:
            raise ValueError("a recorded error must name a failure kind")
        if self.outcome == "response" and self.response is None:
            raise ValueError("a recorded response must carry a response body")
        return self


class FixtureBundle(BaseModel):
    """A recorded provider interaction, replayable offline."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = FIXTURE_SCHEMA_VERSION
    name: str
    catalog_source: str = "fixture"
    catalog: Mapping[str, Any] = Field(default_factory=dict)
    exchanges: tuple[RecordedExchange, ...] = ()
    capabilities: Capabilities

    @model_validator(mode="after")
    def _version_is_supported(self) -> FixtureBundle:
        if self.schema_version != FIXTURE_SCHEMA_VERSION:
            raise ValueError(
                f"fixture schema_version {self.schema_version!r} is not supported; "
                f"expected {FIXTURE_SCHEMA_VERSION!r}"
            )
        return self


class FixtureTransport(ProviderAdapter):
    """Replays recorded exchanges. Performs no I/O beyond reading the bundle.

    There is no socket, no credential and no network call anywhere in this class,
    which is what makes it a safe default and a real test of the adapter contract.
    """

    route = "fixture"

    def __init__(
        self, bundle: FixtureBundle, *, extra_secrets: frozenset[str] = frozenset()
    ) -> None:
        self.bundle = bundle
        self.extra_secrets = extra_secrets
        self.adapter_version = "0.1.0"
        self._calls: list[tuple[str, str, str]] = []

    @classmethod
    def from_path(
        cls, path: Path, *, extra_secrets: frozenset[str] = frozenset()
    ) -> FixtureTransport:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        try:
            bundle = FixtureBundle.model_validate(raw)
        except ValidationError as exc:
            raise ValueError(f"{path} is not a valid fixture bundle: {exc}") from exc
        return cls(bundle, extra_secrets=extra_secrets)

    def discover(self) -> CatalogSnapshot:
        """The catalog exactly as recorded. Missing or malformed yields an empty snapshot.

        An empty catalog is a legitimate observation (an endpoint may expose nothing),
        and it is reported as empty rather than filled with plausible aliases.
        """
        raw = dict(self.bundle.catalog)
        models = raw.get("models", raw.get("data", []))
        if not isinstance(models, Sequence) or isinstance(models, (str, bytes)):
            # A malformed catalog is an empty observation, not an exception: discovery
            # must never take the campaign down on one bad fixture.
            models = []
        entries: list[CatalogEntry] = []
        for index, item in enumerate(models):
            if not isinstance(item, dict):
                continue
            entry = _catalog_entry_from_raw(item, index, extra_secrets=self.extra_secrets)
            if entry is not None:
                entries.append(entry)
        return CatalogSnapshot(
            source=self.bundle.catalog_source,
            entries=tuple(entries),
            # The snapshot-level raw record is stored and folded into the catalog
            # digest, so it is redacted too, not just the per-entry copies.
            raw=redact_mapping(_safe_mapping(raw), extra_secrets=self.extra_secrets),
        )

    def capabilities(self) -> Capabilities:
        return self.bundle.capabilities

    def _find(
        self, endpoint_id: str, benchmark_id: str, item_id: str, repeat_id: int
    ) -> RecordedExchange | None:
        for exchange in self.bundle.exchanges:
            if (
                exchange.endpoint_id == endpoint_id
                and exchange.benchmark_id == benchmark_id
                and exchange.item_id == item_id
                and exchange.repeat_id == repeat_id
            ):
                return exchange
        return None

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
        del prompt_hash, capabilities
        endpoint_id = sample_key.endpoint_id
        benchmark_id, _, item_id = sample_key.task_id.partition("::")
        attempt = attempt_id or f"{sample_key.task_id}-r{sample_key.repeat_id}-a{attempt_number}"
        self._calls.append((endpoint_id, benchmark_id, item_id))

        recorded = self._find(endpoint_id, benchmark_id, item_id, sample_key.repeat_id)
        if recorded is None:
            return AdapterResult(
                failure=TransportFailure(
                    kind=FailureKind.NO_FIXTURE,
                    detail=(
                        f"no recorded exchange for {endpoint_id}/{benchmark_id}/{item_id} "
                        f"repeat {sample_key.repeat_id}"
                    ),
                )
            )

        unsupported = _unsupported_from(recorded, request, extra_secrets=self.extra_secrets)

        if recorded.outcome == "error":
            assert recorded.failure_kind is not None
            failure = safe_failure(
                FailureKind(recorded.failure_kind),
                recorded.failure_detail or "recorded failure",
                http_status=recorded.http_status,
                retry_after_seconds=recorded.retry_after_seconds,
                body=recorded.error_body,
                extra_secrets=self.extra_secrets,
            )
            return AdapterResult(failure=failure, unsupported=unsupported)

        usage = _usage_from(recorded)
        return AdapterResult(
            result=GenerationResult(
                attempt_id=attempt,
                sample_key=sample_key,
                attempt_number=attempt_number,
                delivery_status=DeliveryStatus.ACCEPTED,
                response=recorded.response,
                usage=usage,
                effective_settings=redact_mapping(
                    dict(recorded.effective_settings), extra_secrets=self.extra_secrets
                ),
                streaming=StreamingMeasurements.model_validate(recorded.streaming)
                if recorded.streaming
                else None,
                finish_status=recorded.finish_status,
                manifest_hash=manifest_hash,
                redacted_provider_metadata={
                    "adapter": "fixture",
                    "route": self.route,
                    "bundle": self.bundle.name,
                    # A model can echo a credential inside a tool-call argument, so
                    # these are redacted like any other provider text.
                    "tool_calls": redact_mapping(
                        {"calls": [dict(call) for call in recorded.tool_calls]},
                        extra_secrets=self.extra_secrets,
                    )["calls"],
                },
            ),
            unsupported=unsupported,
            effective_settings=redact_mapping(
                dict(recorded.effective_settings), extra_secrets=self.extra_secrets
            ),
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
        """Replay a recorded SSE stream.

        Implemented rather than refused, so a fixture campaign exercises the same
        streaming path a live one would. The route label is this adapter's own, so a
        fixture stream is never confused with a gateway stream.
        """
        from stealthbench.adapters.streaming import parse_stream

        endpoint_id = sample_key.endpoint_id
        benchmark_id, _, item_id = sample_key.task_id.partition("::")
        attempt = attempt_id or f"{sample_key.task_id}-r{sample_key.repeat_id}-a{attempt_number}"
        self._calls.append((endpoint_id, benchmark_id, item_id))

        recorded = self._find(endpoint_id, benchmark_id, item_id, sample_key.repeat_id)
        if recorded is None:
            return AdapterResult(
                failure=TransportFailure(
                    kind=FailureKind.NO_FIXTURE,
                    detail=f"no recorded stream for {endpoint_id}/{benchmark_id}/{item_id}",
                )
            )
        unsupported = _unsupported_from(recorded, request, extra_secrets=self.extra_secrets)
        if recorded.outcome == "error":
            # The capture records this dispatch as a failure. Replaying its frames as
            # an answer would turn a 429 that served no tokens into an accepted sample
            # with a clean stop and billed token counts.
            assert recorded.failure_kind is not None
            return AdapterResult(
                failure=safe_failure(
                    FailureKind(recorded.failure_kind),
                    recorded.failure_detail or "recorded failure",
                    http_status=recorded.http_status,
                    retry_after_seconds=recorded.retry_after_seconds,
                    body=recorded.error_body,
                    extra_secrets=self.extra_secrets,
                ),
                unsupported=unsupported,
            )
        if not recorded.stream_frames:
            return AdapterResult(
                failure=safe_failure(
                    FailureKind.UNSUPPORTED_SETTING,
                    f"recorded exchange for {benchmark_id}/{item_id} carries no stream frames",
                    extra_secrets=self.extra_secrets,
                ),
                unsupported=unsupported,
            )
        if capabilities is not None and not capabilities.streaming:
            return AdapterResult(
                failure=safe_failure(
                    FailureKind.UNSUPPORTED_SETTING,
                    "streaming was requested but the endpoint does not advertise it",
                    extra_secrets=self.extra_secrets,
                ),
                unsupported=unsupported,
            )

        frames = b"".join(
            b"data: " + json.dumps(dict(frame), ensure_ascii=False).encode("utf-8") + b"\n\n"
            for frame in recorded.stream_frames
        ) + (b"data: [DONE]\n\n" if recorded.stream_terminated else b"")
        outcome, _assembly = parse_stream(
            [frames],
            sample_key=sample_key,
            attempt_id=attempt,
            attempt_number=attempt_number,
            manifest_hash=manifest_hash,
            route=self.route,
            adapter="fixture",
            extra_secrets=self.extra_secrets,
        )
        # A setting the endpoint does not offer is reported on the streamed path too;
        # silently returning a result here would hide it from the campaign record.
        return AdapterResult(
            result=outcome.result,
            failure=outcome.failure,
            unsupported=_unsupported_from(recorded, request),
            effective_settings=outcome.effective_settings,
        )

    def call_count(self) -> int:
        return len(self._calls)

    def calls(self) -> tuple[tuple[str, str, str], ...]:
        return tuple(self._calls)


_ABSENT: Final = object()

#: Spellings a capture may use for a true value. Anything else is False: an
#: unrecognised value is not evidence that the stream completed.
_TRUE_WORDS: Final[frozenset[str]] = frozenset({"true", "t", "yes", "y", "on", "1"})


def read_terminated(value: Any = _ABSENT) -> bool:
    """Whether a capture declared that its stream sent the terminating sentinel.

    Both adapters call this, so the same capture cannot get one verdict from the
    fixture transport and the opposite one from the gateway. An absent field means
    an ordinary complete capture. A value that is present but cannot be read as a
    boolean is not completion.
    """
    if value is _ABSENT:
        return True
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        # An allowlist: an unrecognised spelling is not evidence of completion.
        return value.strip().lower() in _TRUE_WORDS
    return False


def _redacted_requested(request: ModelRequest, setting: str, extra_secrets: frozenset[str]) -> Any:
    """The requested value of a setting, defused before it can reach an artifact.

    ``UnsupportedSetting.to_dict`` serializes this field verbatim, so a stop sequence
    carrying operator-authored prompt data would otherwise bypass the declared-secret
    mechanism that every other serialized field goes through.
    """
    value = _requested_value(request, setting)
    return redact_mapping({"requested": value}, extra_secrets=extra_secrets)["requested"]


def _unsupported_from(
    recorded: RecordedExchange,
    request: ModelRequest,
    *,
    extra_secrets: frozenset[str] = frozenset(),
) -> tuple[UnsupportedSetting, ...]:
    """Settings the recorded endpoint declined, named with what was asked for."""
    return tuple(
        UnsupportedSetting(
            setting=name,
            requested=_redacted_requested(request, name, extra_secrets),
            reason="the recorded endpoint does not offer this setting",
        )
        for name in recorded.unsupported_settings
    )


def _requested_value(request: ModelRequest, setting: str) -> Any:
    """What the request actually asked for, or ``None`` when it asked for nothing.

    Reporting ``False`` for a setting the request set to ``True`` (or omitting one it
    did set) writes a measurement into the artifact that was never made.
    """
    return _requested_settings(request).get(setting)


def _requested_settings(request: ModelRequest) -> dict[str, Any]:
    return {
        "stream": request.stream,
        "max_output_tokens": request.max_output_tokens,
        "temperature": request.temperature,
        "top_p": request.top_p,
        "seed": request.seed,
        "stop": list(request.stop),
    }


def _usage_from(recorded: RecordedExchange) -> Usage:
    """Build usage, keeping unreported directions absent rather than zero.

    ``usage_reported: false`` records an endpoint that answers without usage at all,
    which is a real and common behaviour and must not become a zero.
    """
    if not recorded.usage_reported:
        return Usage(provider_reported=False)
    raw = dict(recorded.usage or {})

    def optional(name: str) -> int | None:
        return token_count(raw.get(name))

    return Usage(
        input_tokens=optional("input_tokens"),
        output_tokens=optional("output_tokens"),
        cached_input_tokens=optional("cached_input_tokens"),
        reasoning_tokens=optional("reasoning_tokens"),
        provider_reported=True,
    )


def strict_flag(*candidates: Any) -> bool:
    """Read a capability flag without ever guessing true.

    ``bool("false")`` and ``bool("no")`` are both True, so a catalog that spells a
    capability as a string would report support the endpoint never claimed. Only a
    real boolean counts.
    """
    for value in candidates:
        if isinstance(value, bool):
            return value
    return False


def token_count(value: Any) -> int | None:
    """Read one token count, or ``None`` when the value is not a usable count.

    A boolean, a float, a numeric string or a negative number is not a token count.
    Coercing any of them produces a fabricated measurement.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value >= 0:
        return value
    return None


#: Keys a catalog record may name an alias under. Both normalizers read the same set,
#: so the same payload cannot yield two different snapshots.
_ALIAS_KEYS: Final[tuple[str, ...]] = ("id", "alias", "slug", "name")


def _alias_of(item: Mapping[str, Any]) -> str | None:
    """The alias a catalog record names, or ``None`` when it names none."""
    for key in _ALIAS_KEYS:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _route_label(value: Any) -> str:
    """The route a record names, or the default when it names none.

    ``str(None)`` would label a record ``"None"``, which is worse than no label.
    """
    return value.strip() if isinstance(value, str) and value.strip() else "zen"


def _first_str(item: Mapping[str, Any], *keys: str) -> str | None:
    """The first of ``keys`` that names a non-empty string."""
    for key in keys:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _context_window_of(item: Mapping[str, Any]) -> int | None:
    """A declared context window, or ``None``.

    A bool is an int in Python; ``True`` as a context window would be recorded as one
    token, which is a fabricated measurement.
    """
    for key in ("context_window", "context_length", "max_context"):
        value = item.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    return None


def _capabilities_of(item: Mapping[str, Any]) -> Capabilities:
    """Read the capabilities a raw catalog record actually advertises.

    Both normalizers call this, so one payload cannot yield two different snapshots.
    A capability the record does not mention stays false; a string or number is never
    read as support.
    """
    raw_caps = item.get("capabilities")
    caps = raw_caps if isinstance(raw_caps, Mapping) else {}
    return Capabilities(
        streaming=strict_flag(caps.get("streaming"), item.get("supports_streaming")),
        tool_calls=strict_flag(caps.get("tool_calls"), item.get("supports_tools")),
        reasoning=strict_flag(caps.get("reasoning"), item.get("supports_reasoning")),
        usage_reporting=strict_flag(caps.get("usage_reporting"), item.get("reports_usage")),
        logprobs=strict_flag(caps.get("logprobs"), item.get("supports_logprobs")),
    )


def _catalog_entry_from_raw(
    item: Mapping[str, Any],
    index: int,
    *,
    extra_secrets: frozenset[str] = frozenset(),
) -> CatalogEntry | None:
    """Normalize one raw catalog entry, or drop it if it is not usable.

    A malformed entry is skipped rather than half-parsed, and a capability the raw
    record does not mention stays false.
    """
    alias = _alias_of(item)
    if alias is None:
        return None
    capabilities = _capabilities_of(item)
    context = _context_window_of(item)
    return CatalogEntry(
        alias=alias,
        route=_route_label(item.get("route")),
        display_name=_first_str(item, "display_name", "name"),
        provider=_first_str(item, "provider", "owned_by"),
        family=_first_str(item, "family"),
        context_window=context,
        capabilities=capabilities,
        # The raw record is retained for provenance and folded into the catalog
        # digest, so it is redacted before it is ever stored.
        raw=redact_mapping(_safe_mapping(item), extra_secrets=extra_secrets),
    )


def _safe_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): item for key, item in value.items()}


def safe_failure(
    kind: FailureKind,
    detail: str,
    *,
    http_status: int | None = None,
    retry_after_seconds: float | None = None,
    body: Mapping[str, Any] | None = None,
    extra_secrets: frozenset[str] = frozenset(),
) -> TransportFailure:
    """Build a failure with both its detail and its body redacted.

    Redacting only the body is not enough: gateways put the credential in the
    human-readable message as often as in the structured field, and a failure detail
    reaches logs and exports like any other text.
    """
    return TransportFailure(
        kind=kind,
        detail=redact_text(detail, extra_secrets=extra_secrets),
        http_status=http_status,
        retry_after_seconds=retry_after_seconds,
        body=redact_mapping(dict(body), extra_secrets=extra_secrets) if body else None,
    )


__all__: Sequence[str] = (
    "RETRYABLE_FAILURES",
    "AdapterResult",
    "CatalogEntry",
    "CatalogSnapshot",
    "FailureKind",
    "FixtureBundle",
    "FixtureTransport",
    "ProviderAdapter",
    "RecordedExchange",
    "TransportFailure",
    "UnsupportedSetting",
    "check_requested_settings",
    "failed_result",
    "safe_failure",
    "strict_flag",
    "token_count",
)
