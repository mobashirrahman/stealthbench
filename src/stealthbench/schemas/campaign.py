"""Versioned configuration contracts: campaign manifest and endpoint spec.

These models are the gate that an invalid configuration fails at. Everything is
frozen and rejects unknown fields, so a typo in a manifest is an error rather than
a silently ignored setting.

The rules encoded here that exist to stop specific, previously-plausible mistakes:

* An unknown ``schema_version`` or unknown enum member is rejected, never coerced.
* A credential *value* in a manifest is rejected. ``credential_ref`` is a name.
* An offline campaign cannot carry a spending cap or a live endpoint.
* A campaign that declares more items than its caps allow is rejected up front
  rather than failing halfway through a paid run.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from datetime import datetime
from typing import Annotated, Any, Final, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)

#: The only contract version this build accepts. An unknown version is an error,
#: never coerced forward.
SCHEMA_VERSION: Final[Literal["1.0"]] = "1.0"

#: A float that must be finite. Pydantic's ``allow_inf_nan=False`` also enforces
#: this at the model level; the alias exists so intent is visible at each field.
FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]

#: An environment-variable-shaped name is the only acceptable credential reference.
#: An allowlist, not a blocklist: an unrecognised provider key must not be accepted
#: as a "name" merely because its prefix is not in the blocklist below.
_CREDENTIAL_REF_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[A-Z][A-Z0-9_]{0,127}$")

#: Substrings that indicate a credential value was pasted where a name belongs.
#: Used for free-text fields, and to explain a credential_ref rejection.
_SECRET_VALUE_MARKERS: Final[tuple[str, ...]] = (
    "sk-",
    "sk_",
    "ghp_",
    "gho_",
    "github_pat_",
    "bearer ",
    "akia",
    "xoxb-",
    "xoxp-",
    "-----begin",
)

_ID_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
_DATASET_ID_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,255}$")

CampaignMode = Literal["offline", "live_authorized"]
Transport = Literal["zen", "reference", "fixture"]
Track = Literal["direct", "agent", "signature", "performance"]


class FrozenModel(BaseModel):
    """Base for every contract model.

    ``frozen=True`` blocks attribute assignment, so a validated model cannot be
    modified in place. It does **not** deep-freeze mutable containers: a ``dict``
    field is still mutable. Fields that live inside a hashed manifest therefore use
    an immutable representation (see ``BenchmarkSpec.scoring_protocol``) rather than a
    plain mapping, so a manifest's hash cannot drift after validation.

    ``extra="forbid"`` turns an unrecognised key into an error instead of silently
    dropping it.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
        validate_default=True,
        ser_json_inf_nan="constants",
    )


class Capabilities(FrozenModel):
    """What an endpoint actually offers.

    Every field is required and boolean. A capability is either supported or not;
    inferring support from another endpoint's behaviour is exactly the mistake the
    G03 gate forbids.
    """

    streaming: bool
    tool_calls: bool
    reasoning: bool
    usage_reporting: bool
    logprobs: bool


class PricingSnapshot(FrozenModel):
    """A price snapshot, or an explicit statement that no price is known.

    Unknown prices are ``None``. A price of ``0`` means a price was published and
    it is zero, which is a different fact.
    """

    currency: str = Field(default="USD", min_length=1, max_length=8)
    input_per_mtok: FiniteFloat | None = None
    output_per_mtok: FiniteFloat | None = None
    cached_input_per_mtok: FiniteFloat | None = None
    reasoning_per_mtok: FiniteFloat | None = None
    snapshot_id: str | None = None

    @model_validator(mode="after")
    def _prices_are_not_negative(self) -> PricingSnapshot:
        for name, value in (
            ("input_per_mtok", self.input_per_mtok),
            ("output_per_mtok", self.output_per_mtok),
            ("cached_input_per_mtok", self.cached_input_per_mtok),
            ("reasoning_per_mtok", self.reasoning_per_mtok),
        ):
            if value is not None and value < 0:
                raise ValueError(f"pricing.{name} must not be negative, got {value}")
        return self


class EndpointSpec(FrozenModel):
    """One endpoint under observation.

    The five identity labels are independent. An anonymous alias has all of them
    unknown, and a shared tokenizer is tokenization evidence only, never an
    identity claim.
    """

    endpoint_id: str
    alias: str
    route: str = Field(min_length=1, max_length=256)
    transport: Transport
    capabilities: Capabilities
    pricing: PricingSnapshot = PricingSnapshot()
    credential_ref: str | None = None
    label_provider: str | None = None
    label_family: str | None = None
    label_exact_version: str | None = None
    label_tokenizer: str | None = None
    label_route: str | None = None

    @field_validator("endpoint_id", "alias")
    @classmethod
    def _identifier_shape(cls, value: str) -> str:
        if not _ID_PATTERN.match(value):
            raise ValueError(
                f"identifier {value!r} must match {_ID_PATTERN.pattern!r} "
                "(lowercase, starting alphanumeric)"
            )
        return value

    @field_validator("credential_ref")
    @classmethod
    def _credential_ref_is_a_name_not_a_value(cls, value: str | None) -> str | None:
        """A manifest may name where a credential lives, never carry it.

        Allowlist of environment-variable names. A blocklist would let a JWT or an
        unlisted provider key through as a "name" and straight into ``manifest_hash``.
        """
        if value is None:
            return None
        if _CREDENTIAL_REF_PATTERN.match(value):
            return value
        lowered = value.lower()
        for marker in _SECRET_VALUE_MARKERS:
            if marker in lowered:
                raise ValueError(
                    "credential_ref must be the name of an environment variable, not a "
                    f"credential; it looks like a secret value (matched {marker!r})"
                )
        raise ValueError(
            f"credential_ref {value!r} must match {_CREDENTIAL_REF_PATTERN.pattern!r}: it is "
            "the name of an environment variable, never the credential itself"
        )

    @model_validator(mode="after")
    def _fixture_endpoints_carry_no_credential(self) -> EndpointSpec:
        if self.transport == "fixture" and self.credential_ref is not None:
            raise ValueError(
                f"endpoint {self.endpoint_id!r} uses the fixture transport and must not "
                "name a credential; fixture transport cannot reach a provider"
            )
        return self


class ObservationWindow(FrozenModel):
    """The UTC window in which an endpoint was observed.

    An offline campaign observes no live endpoint, so both bounds stay ``None``
    rather than carrying a placeholder timestamp.
    """

    started_at: datetime | None = None
    ended_at: datetime | None = None
    timezone: Literal["UTC"] = "UTC"

    @model_validator(mode="after")
    def _window_is_ordered_and_utc(self) -> ObservationWindow:
        for name in ("started_at", "ended_at"):
            moment = getattr(self, name)
            if moment is None:
                continue
            if moment.tzinfo is None:
                raise ValueError(f"observation_window.{name} must be timezone-aware")
            if moment.utcoffset() is None or moment.utcoffset().total_seconds() != 0:
                raise ValueError(f"observation_window.{name} must be UTC, got {moment!r}")
        if (
            self.started_at is not None
            and self.ended_at is not None
            and self.ended_at < self.started_at
        ):
            raise ValueError("observation_window.ended_at must not precede started_at")
        return self


class GenerationSettings(FrozenModel):
    """Requested generation settings.

    Requested settings are recorded separately from the effective settings the
    endpoint reports. A setting the endpoint does not support is reported as
    unsupported; it is never silently dropped.
    """

    max_output_tokens: int = Field(gt=0)
    temperature: FiniteFloat = 0.0
    top_p: FiniteFloat = 1.0
    seed: int | None = None
    stream: bool = False
    unsupported_settings_are_errors: bool = True

    @model_validator(mode="after")
    def _probabilities_are_in_range(self) -> GenerationSettings:
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError(f"temperature must be within [0, 2], got {self.temperature}")
        if not 0.0 <= self.top_p <= 1.0:
            raise ValueError(f"top_p must be within [0, 1], got {self.top_p}")
        return self


class RetryPolicy(FrozenModel):
    """Bounded retries for transport failures only.

    A wrong answer or a malformed answer is scored, never retried. Only the
    statuses listed here are retryable.
    """

    max_attempts: int = Field(ge=1, le=10)
    initial_backoff_seconds: FiniteFloat = Field(gt=0)
    multiplier: FiniteFloat = Field(ge=1.0)
    max_backoff_seconds: FiniteFloat = Field(gt=0)
    retryable_statuses: tuple[int, ...] = (408, 429, 500, 502, 503, 504)

    @field_validator("retryable_statuses")
    @classmethod
    def _only_failure_statuses_retry(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if not value:
            raise ValueError("retryable_statuses must not be empty")
        for status in value:
            if not 400 <= status <= 599:
                raise ValueError(f"status {status} is not an error status and cannot be retried")
        return tuple(sorted(set(value)))

    @model_validator(mode="after")
    def _backoff_is_bounded(self) -> RetryPolicy:
        if self.max_backoff_seconds < self.initial_backoff_seconds:
            raise ValueError("max_backoff_seconds must be at least initial_backoff_seconds")
        return self


class Limits(FrozenModel):
    """Caps for one campaign.

    ``require_cost_bounds`` is true by default: if a reliable upper bound on the
    cost of a request cannot be computed, spending-capped execution is refused
    rather than treated as free.
    """

    max_requests: int = Field(gt=0)
    max_concurrency: int = Field(gt=0)
    max_input_tokens: int = Field(gt=0)
    max_output_tokens: int = Field(gt=0)
    max_total_cost_usd: FiniteFloat | None = Field(default=None, gt=0)
    max_wall_seconds: FiniteFloat = Field(gt=0)
    require_cost_bounds: bool = True
    missingness_threshold: FiniteFloat | None = None

    @field_validator("missingness_threshold")
    @classmethod
    def _missingness_is_a_fraction(cls, value: FiniteFloat | None) -> FiniteFloat | None:
        if value is None:
            return None
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"missingness_threshold must be a fraction, got {value}")
        return value

    @model_validator(mode="after")
    def _limits_are_mutually_consistent(self) -> Limits:
        if self.max_concurrency > self.max_requests:
            raise ValueError(
                f"max_concurrency ({self.max_concurrency}) cannot exceed max_requests "
                f"({self.max_requests})"
            )
        if self.require_cost_bounds and self.max_total_cost_usd is None:
            raise ValueError(
                "require_cost_bounds is true, so max_total_cost_usd must be a number; "
                "an absent cap is not permission to spend an unknown amount"
            )
        return self


class Authorization(FrozenModel):
    """Operator authorization for a live campaign.

    ``spending_cap_usd`` is ``None`` until an operator sets it. The cap in a
    profile is a ceiling; this flag is the approval.
    """

    required: bool = True
    spending_cap_usd: FiniteFloat | None = Field(default=None, gt=0)
    authorized_by: str | None = None

    @model_validator(mode="after")
    def _approval_is_explicit(self) -> Authorization:
        if self.spending_cap_usd is not None and self.authorized_by is None:
            raise ValueError(
                "spending_cap_usd requires authorized_by; a cap without a named approver "
                "is not authorization"
            )
        return self


class BenchmarkSpec(FrozenModel):
    """One benchmark inside a campaign.

    ``item_ids`` is the frozen selection. Before the selection runs it is empty and
    ``materialized`` on the campaign is false, which makes the campaign
    undispatchable rather than dispatchable-with-no-items.
    """

    benchmark_id: str
    track: Track
    category: str = Field(min_length=1)
    adapter_version: str | None = None
    dataset_id: str | None = None
    dataset_revision: str | None = None
    evaluator_id: str | None = None
    evaluator_revision: str | None = None
    declared_item_count: int | None = Field(default=None, gt=0)
    expected_item_count: int | None = Field(default=None, gt=0)
    split_enumeration: str | None = None
    item_ids: tuple[str, ...] = ()
    repeats: int = Field(default=1, ge=1, le=10)
    core_category: bool = False
    api_modes: tuple[Literal["native_function_calling", "prompted"], ...] = ()
    #: Immutable so a hashed manifest cannot be mutated after validation. Accepts a
    #: mapping on input and serializes back to one, so profiles stay readable.
    scoring_protocol: tuple[tuple[str, bool], ...] | None = None
    # Benchmark-specific selection inputs. Declared here so the frozen selection is
    # reproducible rather than being re-derived from a runner's memory.
    release_window: str | None = None
    stratify_by: tuple[str, ...] = ()
    declared_categories: tuple[str, ...] = ()
    generation_seed: int | None = None
    task_versions: tuple[str, ...] = ()
    context_lengths: tuple[int, ...] = ()
    image_digest: str | None = None
    item_digest_manifest: str | None = None
    #: When true the whole declared official split must be enumerated. A subset is
    #: never relabelled as a full run.
    require_full_split: bool = False

    @field_validator("context_lengths")
    @classmethod
    def _context_lengths_are_positive_and_unique(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        for length in value:
            if length <= 0:
                raise ValueError(f"context length must be positive, got {length}")
        if len(value) != len(set(value)):
            raise ValueError("context_lengths must be unique")
        return value

    @field_validator("scoring_protocol", mode="before")
    @classmethod
    def _protocol_flags_are_booleans_with_identifiers(
        cls, value: object
    ) -> tuple[tuple[str, bool], ...] | None:
        """Protocol knobs are named booleans.

        A free-form value here would let a campaign declare a protocol that no
        adapter implements, which is how an unreproducible score gets published.
        """
        if value is None:
            return None
        if not isinstance(value, Mapping):
            raise ValueError("scoring_protocol must be a mapping of names to booleans")
        pairs: list[tuple[str, bool]] = []
        for key, flag in value.items():
            if not _ID_PATTERN.match(key):
                raise ValueError(f"scoring_protocol key {key!r} must match {_ID_PATTERN.pattern!r}")
            if not isinstance(flag, bool):
                raise ValueError(f"scoring_protocol[{key!r}] must be a boolean")
            pairs.append((key, flag))
        return tuple(sorted(pairs))

    @field_serializer("scoring_protocol")
    def _serialize_protocol(
        self, value: tuple[tuple[str, bool], ...] | None
    ) -> dict[str, bool] | None:
        if value is None:
            return None
        return dict(value)

    def protocol_flag(self, name: str) -> bool | None:
        """Look up one protocol flag, or ``None`` when it is not declared."""
        if self.scoring_protocol is None:
            return None
        return dict(self.scoring_protocol).get(name)

    @field_validator("benchmark_id", "category")
    @classmethod
    def _identifier_shape(cls, value: str) -> str:
        if not _ID_PATTERN.match(value):
            raise ValueError(f"identifier {value!r} must match {_ID_PATTERN.pattern!r}")
        return value

    @field_validator("dataset_id")
    @classmethod
    def _dataset_id_shape(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not _DATASET_ID_PATTERN.match(value):
            raise ValueError(f"dataset_id {value!r} is not a plausible dataset identifier")
        return value

    @field_validator("item_ids")
    @classmethod
    def _item_ids_are_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        seen: set[str] = set()
        duplicates = sorted({item for item in value if item in seen or seen.add(item)})  # type: ignore[func-returns-value]
        if duplicates:
            raise ValueError(f"duplicate item ids: {duplicates}")
        return value

    @property
    def planned_item_count(self) -> int | None:
        """Items this benchmark plans to dispatch, or ``None`` when genuinely unknown.

        The frozen selection once it exists, otherwise the declared target. A
        benchmark that declares only a ``split_enumeration`` procedure has an unknown
        size; reporting ``0`` there would silently disable the request-cap check that
        is meant to stop a paid run dying at request 500 of 600.
        """
        if self.item_ids:
            return len(self.item_ids)
        return self.declared_item_count or self.expected_item_count

    @model_validator(mode="after")
    def _counts_agree_with_the_selection(self) -> BenchmarkSpec:
        recorded = self.declared_item_count or self.expected_item_count
        if recorded is None and not self.split_enumeration:
            raise ValueError(
                f"{self.benchmark_id} must record declared_item_count, expected_item_count "
                "or split_enumeration"
            )
        if recorded is not None and recorded <= 0:
            raise ValueError(f"{self.benchmark_id} records a non-positive item count")
        if recorded is not None and self.item_ids and len(self.item_ids) != recorded:
            raise ValueError(
                f"{self.benchmark_id} records {recorded} items but lists {len(self.item_ids)}"
            )
        return self


class SignatureProbeSpec(FrozenModel):
    """The signature probe suite, kept separate from the benchmarks."""

    probe_version: str
    probe_count: int = Field(gt=0)
    repeats: int = Field(gt=0)
    separate_from_benchmarks: bool = True

    @model_validator(mode="after")
    def _probes_stay_separate(self) -> SignatureProbeSpec:
        if not self.separate_from_benchmarks:
            raise ValueError("signature probes must not be mixed into the benchmark selection")
        return self


class SetupProfile(FrozenModel):
    """The smaller campaign that runs before a pilot commits real budget.

    Its purpose is measurement, not a score: establish measured cost, protocol
    support and token accounting first.
    """

    campaign_id: str
    direct_item_count: int = Field(gt=0)
    purpose: str = Field(min_length=1)

    @field_validator("campaign_id")
    @classmethod
    def _identifier_shape(cls, value: str) -> str:
        if not _ID_PATTERN.match(value):
            raise ValueError(f"campaign_id {value!r} must match {_ID_PATTERN.pattern!r}")
        return value


class CampaignManifest(FrozenModel):
    """The frozen description of one campaign.

    Everything needed to reproduce a campaign lives here, and nothing that could
    leak a credential or a gold answer does.
    """

    schema_version: Literal["1.0"] = SCHEMA_VERSION
    campaign_id: str
    title: str = Field(min_length=1)
    description: str = Field(min_length=1)
    mode: CampaignMode
    materialized: bool
    score_version: str = Field(min_length=1)
    seed: int
    observation_window: ObservationWindow = ObservationWindow()
    endpoints: tuple[EndpointSpec, ...]
    benchmarks: tuple[BenchmarkSpec, ...]
    generation: GenerationSettings
    retry_policy: RetryPolicy
    limits: Limits
    authorization: Authorization | None = None
    signature_probes: SignatureProbeSpec | None = None
    setup_profile: SetupProfile | None = None
    budget_scope: Literal["campaign", "setup", "pilot", "full"] = "campaign"
    comparability_rule: str | None = None

    @field_validator("campaign_id")
    @classmethod
    def _identifier_shape(cls, value: str) -> str:
        if not _ID_PATTERN.match(value):
            raise ValueError(f"campaign_id {value!r} must match {_ID_PATTERN.pattern!r}")
        return value

    @field_validator("title", "description", "comparability_rule")
    @classmethod
    def _free_text_carries_no_secret(cls, value: str | None) -> str | None:
        """Human-readable fields are stored and hashed, so they must be secret-free.

        A manifest that accepted a pasted key into its description would serialize it
        into every stored artifact and fold it into the manifest hash.
        """
        if value is None:
            return None
        lowered = value.lower()
        for marker in _SECRET_VALUE_MARKERS:
            if marker in lowered:
                raise ValueError(
                    f"free-text field contains what looks like a credential (matched "
                    f"{marker!r}); a stored, hashed manifest must not carry secret values"
                )
        return value

    @field_validator("endpoints")
    @classmethod
    def _endpoint_ids_unique(cls, value: tuple[EndpointSpec, ...]) -> tuple[EndpointSpec, ...]:
        ids = [endpoint.endpoint_id for endpoint in value]
        duplicates = sorted({name for name in ids if ids.count(name) > 1})
        if duplicates:
            raise ValueError(f"duplicate endpoint ids: {duplicates}")
        return value

    @field_validator("benchmarks")
    @classmethod
    def _benchmark_ids_unique(cls, value: tuple[BenchmarkSpec, ...]) -> tuple[BenchmarkSpec, ...]:
        ids = [spec.benchmark_id for spec in value]
        duplicates = sorted({name for name in ids if ids.count(name) > 1})
        if duplicates:
            raise ValueError(f"duplicate benchmark ids: {duplicates}")
        return value

    @model_validator(mode="after")
    def _mode_is_consistent(self) -> CampaignManifest:
        if self.mode == "offline":
            if self.authorization is not None:
                raise ValueError(
                    "an offline campaign must not declare authorization or a spending cap"
                )
            live = [
                endpoint.endpoint_id
                for endpoint in self.endpoints
                if endpoint.transport != "fixture"
            ]
            if live:
                raise ValueError(
                    f"offline campaign cannot use non-fixture transport: {sorted(live)}"
                )
        else:
            if self.authorization is None:
                raise ValueError("a live_authorized campaign must declare an authorization block")
            if not self.authorization.required:
                raise ValueError("a live_authorized campaign must require authorization")
            # A null spending cap is a well-formed *document* that is not yet
            # authorized to dispatch. Conflating the two would make the shipped
            # example profiles invalid, and would make a cap look like approval.
        if not self.endpoints and self.materialized:
            raise ValueError("a materialized campaign must declare at least one endpoint")
        return self

    @model_validator(mode="after")
    def _materialization_matches_the_selection(self) -> CampaignManifest:
        for spec in self.benchmarks:
            if self.materialized and not spec.item_ids:
                raise ValueError(
                    f"campaign {self.campaign_id} is materialized but benchmark "
                    f"{spec.benchmark_id!r} has no item_ids"
                )
            if not self.materialized and spec.item_ids:
                raise ValueError(
                    f"campaign {self.campaign_id} is not materialized, so benchmark "
                    f"{spec.benchmark_id!r} must not pre-declare item_ids"
                )
        return self

    @model_validator(mode="after")
    def _caps_can_cover_the_declared_work(self) -> CampaignManifest:
        """Refuse a campaign whose own numbers make it certain to exceed its caps.

        This is the check that stops a paid run from dying at request 500 of 600.
        """
        # Request and token caps bind every transport, including fixtures: a free
        # endpoint is not an unlimited one. Only the *cost* cap is meaningless for a
        # fixture, which is why require_cost_bounds is the fixture-exempt check.
        # A materialized campaign always has non-empty item_ids (enforced above), so
        # its planned count is always known; only an unmaterialized campaign can carry
        # an unknown size, and that is reported as a dispatch blocker instead.
        planned_requests = self.total_planned_requests
        if (
            planned_requests is not None
            and planned_requests > 0
            and self.limits.max_requests < planned_requests
        ):
            raise ValueError(
                f"max_requests ({self.limits.max_requests}) is below the "
                f"{planned_requests} requests this campaign would dispatch; raise the cap or "
                "reduce the selection"
            )
        if self.limits.max_output_tokens < self.generation.max_output_tokens:
            raise ValueError(
                f"limits.max_output_tokens ({self.limits.max_output_tokens}) is below "
                f"generation.max_output_tokens ({self.generation.max_output_tokens})"
            )
        return self

    @property
    def benchmarks_with_unknown_size(self) -> tuple[str, ...]:
        """Benchmarks whose item count is not yet knowable.

        Non-empty only before the frozen selection exists for a benchmark that
        declares an enumeration procedure instead of a count.
        """
        return tuple(
            sorted(spec.benchmark_id for spec in self.benchmarks if spec.planned_item_count is None)
        )

    @property
    def total_planned_requests(self) -> int | None:
        """Requests the campaign would dispatch if nothing failed.

        ``None`` when any benchmark's size is unknown, because a partial sum would
        understate the run. Uses declared counts before materialization and the frozen
        selection after.
        """
        total = 0
        for spec in self.benchmarks:
            count = spec.planned_item_count
            if count is None:
                return None
            total += count * spec.repeats
        return total

    @property
    def known_planned_requests(self) -> int:
        """Lower bound on planned requests, for reporting alongside ``None``."""
        return sum((spec.planned_item_count or 0) * spec.repeats for spec in self.benchmarks)

    def dispatch_blockers(self) -> tuple[str, ...]:
        """Every reason this campaign may not be dispatched yet.

        Kept separate from validation on purpose. A profile can be a perfectly valid
        document while still being undispatchable: unmaterialized item selection, or a
        live campaign with no operator spending cap. Both are reported here rather
        than being treated as malformed input.
        """
        blockers: list[str] = []
        unknown_sizes = self.benchmarks_with_unknown_size
        if unknown_sizes and not self.materialized:
            blockers.append(
                f"item count is unknown for {list(unknown_sizes)}; their split must be "
                "enumerated before the request cap can be checked"
            )
        if not self.materialized:
            blockers.append(
                "item selection has not run; item_ids are empty so there is nothing to dispatch"
            )
        if self.mode == "live_authorized":
            if self.authorization is None or self.authorization.spending_cap_usd is None:
                blockers.append(
                    "live campaign has no operator spending cap; "
                    "authorization.spending_cap_usd must be a number"
                )
            live_endpoints = [
                endpoint.endpoint_id
                for endpoint in self.endpoints
                if endpoint.credential_ref is None and endpoint.transport != "fixture"
            ]
            if live_endpoints and (
                self.authorization is None or not self.authorization.authorized_by
            ):
                blockers.append(
                    f"endpoints {sorted(live_endpoints)} name no credential_ref and no "
                    "authorized_by is recorded"
                )
        return tuple(blockers)

    @property
    def is_dispatchable(self) -> bool:
        return not self.dispatch_blockers()

    def to_json_dict(self) -> dict[str, Any]:
        """A plain dict for canonical hashing and storage.

        Excludes nothing: every field that could hold a credential is rejected at
        validation, including the free-text ones.
        """
        return self.model_dump(mode="json", exclude_none=False)
