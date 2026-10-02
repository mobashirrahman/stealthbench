"""Versioned result contracts: tasks, generations, grades, events, identity, evidence.

Three properties in here carry most of the weight:

* **An absent measurement is ``None``, never ``0``.** ``Usage`` has every field
  nullable because an endpoint may report input tokens and nothing else.
* **A delivery attempt is not a sample.** ``DeliveryStatus`` and
  :func:`accepted_sample_keys` keep retries and crashes out of the denominator.
* **A request cannot carry evaluator-only data.** ``ModelRequest`` is a separate,
  closed model built from an explicit allowlist. Gold answers live in
  ``EvaluationPayload`` and there is no serialization path from one to the other.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Annotated, Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from stealthbench.schemas.campaign import SCHEMA_VERSION, FiniteFloat, ObservationWindow

if TYPE_CHECKING:
    from stealthbench.schemas.manifest import PromptRef

#: A lowercase hex sha256 digest.
Sha256Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]

#: Field names that may never appear in a model request. Kept as data so the
#: contract test can assert the closed ``ModelRequest`` schema stays clean.
EVALUATOR_ONLY_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "answer",
        "answers",
        "gold",
        "gold_answer",
        "solution",
        "expected",
        "label",
        "test_patch",
        "fail_to_pass",
        "pass_to_pass",
        "eval_script",
        "hidden_tests",
        "reference_tests",
        "eval_payload",
        "evaluator_revision",
        "scoring_protocol",
    }
)

FinishStatus = Literal["stop", "length", "tool_calls", "content_filter", "error", "cancelled"]

#: Event vocabulary. Declared explicitly rather than inferred from a name suffix, so
#: "is this a completion record" cannot change because someone added a dot.
EVENT_REQUEST_DISPATCHED: Final[str] = "request.dispatched"
EVENT_REQUEST_COMPLETED: Final[str] = "request.completed"
EVENT_REQUEST_FAILED: Final[str] = "request.failed"
EVENT_SAMPLE_ACCEPTED: Final[str] = "sample.accepted"
EVENT_GRADE_RECORDED: Final[str] = "grade.recorded"
EVENT_CAMPAIGN_COMPLETED: Final[str] = "campaign.completed"

#: An event is durable-and-complete only if one of these types was written. Anything
#: else is progress that may be lost to a crash.
COMPLETION_EVENT_TYPES: Final[frozenset[str]] = frozenset(
    {
        EVENT_REQUEST_COMPLETED,
        EVENT_SAMPLE_ACCEPTED,
        EVENT_GRADE_RECORDED,
        EVENT_CAMPAIGN_COMPLETED,
    }
)

#: Attempts: a request that was sent. An attempt is not a sample.
ATTEMPT_EVENT_TYPES: Final[frozenset[str]] = frozenset(
    {EVENT_REQUEST_DISPATCHED, EVENT_REQUEST_COMPLETED, EVENT_REQUEST_FAILED}
)


class ResultModel(BaseModel):
    """Base for result contracts: frozen, closed, finite-only."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
        validate_default=True,
    )


class Message(ResultModel):
    """One chat message. Roles are limited to what a direct chat protocol accepts."""

    role: Literal["system", "user", "assistant"]
    content: str


class ModelRequest(ResultModel):
    """The only structure permitted to leave the process toward a provider.

    Closed on purpose. An unrecognised field is an error, so a gold answer cannot be
    smuggled in under a new name without the schema refusing to build.
    """

    messages: tuple[Message, ...]
    max_output_tokens: int = Field(gt=0)
    temperature: FiniteFloat = 0.0
    top_p: FiniteFloat = 1.0
    seed: int | None = None
    stream: bool = False
    stop: tuple[str, ...] = ()

    @field_validator("messages")
    @classmethod
    def _at_least_one_message(cls, value: tuple[Message, ...]) -> tuple[Message, ...]:
        if not value:
            raise ValueError("a request must contain at least one message")
        return value

    @field_validator("stop")
    @classmethod
    def _stop_sequences_are_short_and_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for sequence in value:
            if not sequence or len(sequence) > 64:
                raise ValueError(f"stop sequence must be 1-64 characters, got {sequence!r}")
        if len(value) != len(set(value)):
            raise ValueError("stop sequences must be unique")
        return value

    def forbidden_fields_present(self) -> frozenset[str]:
        """Evaluator-only names reachable from this request's own schema.

        Always empty for a valid instance; the method exists so a contract test can
        assert it without re-deriving the schema.
        """
        declared = set(type(self).model_fields)
        return frozenset(declared & EVALUATOR_ONLY_FIELDS)


class EvaluationPayload(ResultModel):
    """Everything needed to grade a response, and nothing a provider may see."""

    gold_answer: str | None = None
    evaluator_id: str
    evaluator_revision: str
    declared_metadata: Mapping[str, Any] = Field(default_factory=dict)
    hidden_artifacts: tuple[str, ...] = ()

    def assert_absent_from(self, request: ModelRequest) -> None:
        """Raise if any evaluator-only value leaks into a serialized request."""
        serialized = request.model_dump_json()
        offenders = [
            value
            for value in (self.gold_answer, *self.hidden_artifacts)
            if value and value in serialized
        ]
        if offenders:
            raise ValueError(
                "evaluator-only data would be sent to a provider: "
                + ", ".join(repr(offender[:32]) for offender in offenders)
            )


class TaskSpec(ResultModel):
    """One benchmark item at one repeat.

    ``sample_key`` identifies the accepted sample. Repeats are distinct keys, which
    is what keeps a retry from being mistaken for a second measurement.
    """

    benchmark_id: str
    item_id: str
    campaign_id: str
    endpoint_id: str
    repeat_id: int = Field(ge=1)
    prompt_artifact_hash: Sha256Digest
    request: ModelRequest
    evaluation: EvaluationPayload
    declared_metadata: Mapping[str, Any] = Field(default_factory=dict)

    @property
    def sample_key(self) -> SampleKey:
        return SampleKey(
            campaign_id=self.campaign_id,
            endpoint_id=self.endpoint_id,
            task_id=task_id_for(self.benchmark_id, self.item_id),
            repeat_id=self.repeat_id,
        )

    def to_prompt_ref(self) -> PromptRef:
        """The addressable form of this task's request, for prompt hashing.

        Delegates to ``PromptRef.from_request`` so this and the prompt-hash path can
        never drift apart over which fields count as dispatchable.
        """
        from stealthbench.schemas.manifest import PromptRef

        return PromptRef.from_request(self.request)

    @model_validator(mode="after")
    def _verify_recorded_prompt_hash(self) -> TaskSpec:
        self.check_prompt_hash()
        return self

    def check_prompt_hash(self) -> None:
        """Raise unless the recorded provenance hash describes the held request.

        Recomputed rather than trusted, because ``model_copy(update=...)`` and
        ``model_construct`` bypass validators. Without this check any 64 hex
        characters would pass as provenance and a sample could be attributed to a
        prompt it was never built from.
        """
        from stealthbench.schemas.manifest import prompt_hash

        actual = prompt_hash(self.to_prompt_ref())
        if self.prompt_artifact_hash != actual:
            raise ValueError(
                f"prompt_artifact_hash {self.prompt_artifact_hash} does not describe this "
                f"request; the canonical prompt hash is {actual}"
            )

    def safe_request(self) -> ModelRequest:
        """The request as it may be dispatched.

        The recorded prompt hash is re-verified and evaluator-only data is asserted
        absent, rather than assumed. That way a future field added to
        ``ModelRequest`` cannot quietly carry a gold answer, and an object built
        through a validator-bypassing constructor still cannot be dispatched.
        """
        self.check_prompt_hash()
        self.evaluation.assert_absent_from(self.request)
        return self.request


class SampleKey(ResultModel):
    """Identity of one accepted sample.

    A benchmark result is keyed by campaign, endpoint observation, task and
    repetition. Retries do not add samples, so nothing else belongs in this key.
    """

    campaign_id: str
    endpoint_id: str
    task_id: str
    repeat_id: int = Field(ge=1)

    def as_tuple(self) -> tuple[str, str, str, int]:
        return (self.campaign_id, self.endpoint_id, self.task_id, self.repeat_id)


def task_id_for(benchmark_id: str, item_id: str) -> str:
    """Stable task identifier, so clustering by task works across benchmarks."""
    return f"{benchmark_id}::{item_id}"


class Usage(ResultModel):
    """Token accounting as reported by the endpoint.

    Every field is nullable and independently so. ``None`` means the endpoint did
    not report it. Substituting ``0`` would make an unreported count look like a
    free request, which is the single most expensive mistake in this file.
    """

    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    cached_input_tokens: int | None = Field(default=None, ge=0)
    reasoning_tokens: int | None = Field(default=None, ge=0)
    provider_reported: bool = False

    @property
    def is_complete(self) -> bool:
        """True only when both billed directions were reported."""
        return self.input_tokens is not None and self.output_tokens is not None


class StreamingMeasurements(ResultModel):
    """Timing and shape of a streamed completion.

    ``chunk_count`` is a count of transport events. It is deliberately not a token
    count and there is no helper that converts one to the other.
    """

    first_content_seconds: FiniteFloat | None = Field(default=None, ge=0)
    first_answer_seconds: FiniteFloat | None = Field(default=None, ge=0)
    total_seconds: FiniteFloat | None = Field(default=None, ge=0)
    chunk_count: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _timings_are_ordered(self) -> StreamingMeasurements:
        if (
            self.first_content_seconds is not None
            and self.total_seconds is not None
            and self.first_content_seconds > self.total_seconds
        ):
            raise ValueError("first_content_seconds cannot exceed total_seconds")
        return self

    def output_tokens_per_second(self, usage: Usage) -> FiniteFloat | None:
        """Normalized output rate, or ``None`` when either input is missing.

        Returns ``None`` rather than substituting the chunk count, because one SSE
        chunk is not one token.
        """
        if usage.output_tokens is None or not self.total_seconds:
            return None
        return usage.output_tokens / self.total_seconds


class DeliveryStatus(StrEnum):
    """What became of a dispatch attempt."""

    ACCEPTED = "accepted"
    TRANSPORT_FAILED = "transport_failed"
    CANCELLED = "cancelled"
    UNRESOLVED = "unresolved"


class GenerationResult(ResultModel):
    """One delivery attempt's outcome.

    Exactly one of these exists per attempt. Only ``ACCEPTED`` results with a
    response contribute a sample.
    """

    schema_version: Literal["1.0"] = SCHEMA_VERSION
    attempt_id: str
    sample_key: SampleKey
    attempt_number: int = Field(ge=1)
    delivery_status: DeliveryStatus
    response: str | None = None
    usage: Usage = Usage()
    effective_settings: Mapping[str, Any] = Field(default_factory=dict)
    streaming: StreamingMeasurements | None = None
    finish_status: FinishStatus | None = None
    redacted_provider_metadata: Mapping[str, Any] = Field(default_factory=dict)
    manifest_hash: Sha256Digest | None = None

    @model_validator(mode="after")
    def _accepted_results_have_a_response(self) -> GenerationResult:
        if self.delivery_status is DeliveryStatus.ACCEPTED and self.response is None:
            raise ValueError(
                f"attempt {self.attempt_id} is accepted but carries no response; an accepted "
                "sample with no output cannot be graded and must not be counted"
            )
        if self.delivery_status is DeliveryStatus.UNRESOLVED and self.response is not None:
            raise ValueError(
                f"attempt {self.attempt_id} is unresolved after an ambiguous dispatch and must "
                "not carry a response; it may have been billed and was never confirmed"
            )
        return self

    @property
    def is_accepted_sample(self) -> bool:
        return self.delivery_status is DeliveryStatus.ACCEPTED and self.response is not None


def accepted_sample_keys(results: Iterable[GenerationResult]) -> set[tuple[str, str, str, int]]:
    """The distinct accepted samples in a result stream.

    Retries collapse: several attempts for one sample key yield one entry. This is
    what stops a retried task from inflating a denominator.
    """
    return {result.sample_key.as_tuple() for result in results if result.is_accepted_sample}


def unresolved_attempts(results: Iterable[GenerationResult]) -> tuple[str, ...]:
    """Attempt ids dispatched with no confirmed outcome, which may still have been billed."""
    return tuple(
        sorted(
            result.attempt_id
            for result in results
            if result.delivery_status is DeliveryStatus.UNRESOLVED
        )
    )


StatusFlag = Literal["pass", "fail", "invalid", "unavailable"]


class DenominatorEligibility(ResultModel):
    """Which published numbers may include a sample.

    A first-class field rather than a filtering step at render time, so a grade
    cannot be silently included in a mean it does not belong to.
    """

    correctness: bool
    format: bool
    transport_success: bool
    evaluator_ran: bool


class GradeResult(ResultModel):
    """One grader verdict.

    The four statuses are orthogonal and are never merged: "did the request
    complete" and "was the answer right" are different questions with different
    denominators.
    """

    schema_version: Literal["1.0"] = SCHEMA_VERSION
    grader_version: str
    sample_key: SampleKey
    correctness: StatusFlag
    format: StatusFlag
    transport: StatusFlag
    evaluator: StatusFlag
    score_components: Mapping[str, FiniteFloat] = Field(default_factory=dict)
    denominator_eligibility: DenominatorEligibility

    @model_validator(mode="after")
    def _eligibility_follows_the_statuses(self) -> GradeResult:
        expected = DenominatorEligibility(
            correctness=self.correctness in {"pass", "fail"},
            format=self.format in {"pass", "fail"},
            transport_success=self.transport == "pass",
            evaluator_ran=self.evaluator == "pass",
        )
        if self.denominator_eligibility != expected:
            raise ValueError(
                f"denominator_eligibility disagrees with the statuses: "
                f"expected {expected.model_dump()}, got {self.denominator_eligibility.model_dump()}"
            )
        return self

    @property
    def counts_toward_accuracy(self) -> bool:
        """Whether this sample belongs in a published correctness figure."""
        return self.correctness in {"pass", "fail"} and self.evaluator == "pass"


class RunEvent(ResultModel):
    """An append-only log entry.

    Wall time is UTC for humans; the duration comes from a monotonic clock so it
    cannot go backwards when the wall clock is adjusted.
    """

    schema_version: Literal["1.0"] = SCHEMA_VERSION
    event_id: str
    event_type: str = Field(min_length=1)
    campaign_id: str
    #: Ordinal in the append-only log. Makes event identity unique even for identical
    #: payloads, and makes a reordered or spliced log detectable on read.
    sequence: int = Field(ge=0)
    wall_time_utc: datetime
    monotonic_duration_seconds: FiniteFloat = Field(ge=0)
    payload_hash: Sha256Digest
    payload: Mapping[str, Any] = Field(default_factory=dict)

    @field_validator("wall_time_utc")
    @classmethod
    def _wall_time_is_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("wall_time_utc must be timezone-aware")
        offset = value.utcoffset()
        if offset is None or offset.total_seconds() != 0:
            raise ValueError(f"wall_time_utc must be UTC, got {value!r}")
        return value

    @field_validator("event_type")
    @classmethod
    def _event_type_is_known(cls, value: str) -> str:
        known = COMPLETION_EVENT_TYPES | ATTEMPT_EVENT_TYPES
        if value not in known:
            raise ValueError(
                f"event_type {value!r} is not in the declared vocabulary {sorted(known)}"
            )
        return value

    @property
    def is_completion_record(self) -> bool:
        """Whether this event durably records that something finished.

        Delivery attempts and accepted samples are separate event types, and an
        interrupted write must not be able to look like a completion.
        """
        return self.event_type in COMPLETION_EVENT_TYPES

    @property
    def is_attempt_record(self) -> bool:
        """Whether this event describes a request attempt rather than a sample."""
        return self.event_type in ATTEMPT_EVENT_TYPES


class ProbeValidity(StrEnum):
    """Whether one probe produced usable evidence."""

    VALID = "valid"
    MISSING_COUNT = "missing_count"
    UNSTABLE_REPEAT = "unstable_repeat"
    CONTENT_DEPENDENT_OFFSET = "content_dependent_offset"


class SignatureResult(ResultModel):
    """Token-count and behavioural evidence from one probe suite.

    The validity mask is per probe. A missing or unstable probe marks that probe
    unavailable and is never imputed from its neighbours.
    """

    schema_version: Literal["1.0"] = SCHEMA_VERSION
    probe_version: str
    endpoint_id: str
    campaign_id: str
    input_count_vector: tuple[int | None, ...]
    validity: tuple[ProbeValidity, ...]
    behavioral_features: Mapping[str, Any] = Field(default_factory=dict)
    observation_window: ObservationWindow = ObservationWindow()

    @model_validator(mode="after")
    def _mask_matches_the_vector(self) -> SignatureResult:
        if len(self.input_count_vector) != len(self.validity):
            raise ValueError(
                f"validity mask has {len(self.validity)} entries for "
                f"{len(self.input_count_vector)} probes"
            )
        pairs = zip(self.input_count_vector, self.validity, strict=True)
        for index, (count, validity) in enumerate(pairs):
            if validity is ProbeValidity.VALID and count is None:
                raise ValueError(f"probe {index} is marked valid but has no count")
            if validity is not ProbeValidity.VALID and count is not None:
                raise ValueError(
                    f"probe {index} is {validity} yet carries a count; an unusable signal "
                    "must not present a number"
                )
        return self

    @property
    def valid_mask(self) -> tuple[bool, ...]:
        return tuple(validity is ProbeValidity.VALID for validity in self.validity)

    @property
    def usable_probes(self) -> int:
        return sum(self.valid_mask)


class SignalEvidence(ResultModel):
    """One kind of evidence about a candidate, and how much weight it carries."""

    signal: Literal[
        "input_count_vector",
        "actual_tokenization",
        "behavior",
        "capability_pattern",
        "protocol",
        "performance",
    ]
    available: bool
    comparable_observations: int = Field(ge=0)
    detail: str | None = None


class RankedSimilarity(ResultModel):
    """A similarity to a reference candidate.

    There is deliberately no ``probability`` field. A similarity of 0.9 is not a
    90% probability, and representing it as one would be the error this whole
    subsystem exists to avoid.
    """

    candidate_id: str
    candidate_library_revision: str
    similarity: FiniteFloat = Field(ge=0.0, le=1.0)
    evidence_signals: tuple[str, ...] = ()
    note: str | None = None


class IdentityReport(ResultModel):
    """An identity estimate for one endpoint observation.

    ``calibrated_probabilities`` stays ``None`` unless a calibration evidence gate
    passed. Similarity is reported either way.
    """

    schema_version: Literal["1.0"] = SCHEMA_VERSION
    endpoint_id: str
    candidate_library_revision: str
    signal_evidence: tuple[SignalEvidence, ...]
    ranked_similarities: tuple[RankedSimilarity, ...]
    abstention_reason: str | None = None
    calibrated_probabilities: Mapping[str, FiniteFloat] | None = None
    calibration_evidence_id: str | None = None
    official_reveal_label: str | None = None
    predicted_at: datetime | None = None

    @field_validator("predicted_at")
    @classmethod
    def _predicted_at_is_utc(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            raise ValueError("predicted_at must be timezone-aware")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def _probabilities_require_evidence(self) -> IdentityReport:
        if self.calibrated_probabilities is None:
            return self
        if not self.calibration_evidence_id:
            raise ValueError(
                "calibrated_probabilities requires a calibration_evidence_id; probabilities may "
                "not be displayed without the evidence gate that produced them"
            )
        total = sum(self.calibrated_probabilities.values())
        if not 0.999 <= total <= 1.001:
            raise ValueError(f"calibrated probabilities must sum to 1, got {total}")
        unknown = set(self.calibrated_probabilities) - {
            similarity.candidate_id for similarity in self.ranked_similarities
        }
        if unknown:
            raise ValueError(f"probabilities reference unranked candidates: {sorted(unknown)}")
        return self

    @model_validator(mode="after")
    def _similarities_are_ranked(self) -> IdentityReport:
        scores = [item.similarity for item in self.ranked_similarities]
        if scores != sorted(scores, reverse=True):
            raise ValueError("ranked_similarities must be ordered by descending similarity")
        return self

    @model_validator(mode="after")
    def _weak_evidence_abstains(self) -> IdentityReport:
        """A report with no usable comparable signal must say unknown."""
        usable = [
            evidence
            for evidence in self.signal_evidence
            if evidence.available and evidence.comparable_observations > 0
        ]
        if not usable and not self.abstention_reason:
            raise ValueError(
                "no signal has a comparable observation; the report must carry an "
                "abstention_reason rather than rank candidates"
            )
        return self

    @property
    def is_unknown(self) -> bool:
        return not self.ranked_similarities


class CommandRecord(ResultModel):
    """One command actually run, with the status it actually returned."""

    command: str
    exit_status: int


class GateEvidence(ResultModel):
    """Evidence that a gate's checks passed.

    A zero exit status alone cannot satisfy a check: a skip, an expected failure, an
    empty suite and a zero-exit run are all recorded, and the counts are consistent
    so a passing verdict cannot hide an uncollected suite.
    """

    schema_version: Literal["1.0"] = SCHEMA_VERSION
    gate_id: str
    source_revision: str
    manifest_hash: Sha256Digest | None = None
    commands: tuple[CommandRecord, ...] = ()
    collected: int = Field(ge=0)
    passed: int = Field(ge=0)
    failed: int = Field(ge=0)
    skipped: int = Field(ge=0)
    artifacts: tuple[str, ...] = ()
    review_decision: Literal["pass", "fail", "blocked_external"]

    @model_validator(mode="after")
    def _counts_are_consistent(self) -> GateEvidence:
        if self.passed + self.failed + self.skipped > self.collected:
            raise ValueError(
                f"passed({self.passed}) + failed({self.failed}) + skipped({self.skipped}) "
                f"exceeds collected({self.collected})"
            )
        return self

    @model_validator(mode="after")
    def _passing_requires_clean_tests_and_review(self) -> GateEvidence:
        """A pass must rest on tests that ran and commands that worked.

        `IMPLEMENTATION_PLAN.md` section 4: a skip, an expected failure, an empty
        suite and a zero exit status alone each cannot satisfy a mandatory check.
        This refuses the combinations that would let a gate certify itself on
        nothing.
        """
        if self.review_decision != "pass":
            return self
        if self.failed:
            raise ValueError(f"gate {self.gate_id} cannot pass with {self.failed} failing tests")
        if self.collected == 0:
            raise ValueError(
                f"gate {self.gate_id} cannot pass on zero collected tests; an empty suite is not "
                "evidence"
            )
        if self.passed == 0:
            raise ValueError(
                f"gate {self.gate_id} cannot pass with zero passing tests; "
                f"{self.skipped} skipped is not a pass"
            )
        if not self.commands:
            raise ValueError(f"gate {self.gate_id} cannot pass without a recorded command")
        failing = [record.command for record in self.commands if record.exit_status != 0]
        if failing:
            raise ValueError(
                f"gate {self.gate_id} cannot pass while a recorded command returned non-zero: "
                f"{failing}"
            )
        if not self.artifacts:
            raise ValueError(f"gate {self.gate_id} cannot pass without a recorded artifact")
        return self

    @property
    def has_vacuous_check(self) -> tuple[str, ...]:
        """Commands whose exit status alone would not satisfy a mandatory check."""
        warnings: list[str] = []
        if self.collected == 0:
            warnings.append("collected zero tests")
        if self.skipped:
            warnings.append(f"{self.skipped} skipped")
        if self.failed:
            warnings.append(f"{self.failed} failed")
        return tuple(warnings)


def group_by_task(results: Sequence[GenerationResult]) -> dict[str, list[GenerationResult]]:
    """Group accepted results by task, so repeats cluster for statistics.

    Reps of one item are not independent problems, and G09's intervals depend on
    this grouping being right.
    """
    grouped: dict[str, list[GenerationResult]] = {}
    for result in results:
        if not result.is_accepted_sample:
            continue
        grouped.setdefault(result.sample_key.task_id, []).append(result)
    return grouped
