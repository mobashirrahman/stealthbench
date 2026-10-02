"""Contract tests for result, identity and evidence schemas (task T01B).

Acceptance for T01B: usage remains nullable; retries and accepted samples are
distinct; evaluator-only data cannot enter requests.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from stealthbench.schemas.campaign import CampaignManifest
from stealthbench.schemas.manifest import PromptRef, prompt_hash, trace_key
from stealthbench.schemas.results import (
    EVALUATOR_ONLY_FIELDS,
    EVENT_REQUEST_COMPLETED,
    EVENT_SAMPLE_ACCEPTED,
    CommandRecord,
    DeliveryStatus,
    DenominatorEligibility,
    EvaluationPayload,
    GateEvidence,
    GenerationResult,
    GradeResult,
    IdentityReport,
    ModelRequest,
    RankedSimilarity,
    RunEvent,
    SampleKey,
    SignalEvidence,
    SignatureResult,
    TaskSpec,
    Usage,
    accepted_sample_keys,
    group_by_task,
    task_id_for,
    unresolved_attempts,
)

pytestmark = pytest.mark.contract

DIGEST = "a" * 64
GOLD_CANARY = "the-correct-answer-is-42"
REPO_ROOT = Path(__file__).resolve().parents[2]


def _manifest() -> CampaignManifest:
    return CampaignManifest.model_validate(
        json.loads((REPO_ROOT / "configs" / "offline-demo.json").read_text(encoding="utf-8"))
    )


_MANIFEST = _manifest()


@pytest.fixture
def request_() -> ModelRequest:
    return ModelRequest(
        messages=[{"role": "user", "content": "What is 6 times 7?"}],
        max_output_tokens=64,
    )


@pytest.fixture
def sample_key() -> SampleKey:
    return SampleKey(campaign_id="c1", endpoint_id="e1", task_id="ifeval::item-1", repeat_id=1)


def make_task(request: ModelRequest) -> TaskSpec:
    """Build a TaskSpec whose recorded prompt hash genuinely describes ``request``."""
    return TaskSpec(
        benchmark_id="ifeval",
        item_id="item-1",
        campaign_id="c1",
        endpoint_id="e1",
        repeat_id=1,
        prompt_artifact_hash=prompt_hash(PromptRef.from_request(request)),
        request=request,
        evaluation=EvaluationPayload(
            gold_answer=GOLD_CANARY,
            evaluator_id="instruction_following_eval",
            evaluator_revision="google-research@e49bbfe3",
            hidden_artifacts=("hidden-test-suite.py",),
        ),
    )


@pytest.fixture
def task(request_: ModelRequest) -> TaskSpec:
    return make_task(request_)


def _result(
    sample_key: SampleKey,
    status: DeliveryStatus = DeliveryStatus.ACCEPTED,
    attempt: int = 1,
    response: str | None = "42",
) -> GenerationResult:
    return GenerationResult(
        attempt_id=f"attempt-{attempt}",
        sample_key=sample_key,
        attempt_number=attempt,
        delivery_status=status,
        response=response,
    )


# ---------------------------------------------------------------------------
# Usage stays nullable
# ---------------------------------------------------------------------------


def test_absent_usage_is_null_not_zero() -> None:
    usage = Usage()
    assert usage.input_tokens is None
    assert usage.output_tokens is None
    dumped = usage.model_dump()
    assert dumped["input_tokens"] is None
    assert dumped["output_tokens"] is None
    assert json.loads(usage.model_dump_json())["output_tokens"] is None


def test_reported_zero_is_distinguishable_from_absent() -> None:
    reported = Usage(input_tokens=0, output_tokens=0, provider_reported=True)
    unreported = Usage()
    assert reported.input_tokens == 0
    assert reported != unreported
    assert reported.is_complete and not unreported.is_complete


def test_usage_fields_are_independently_nullable() -> None:
    """An endpoint may report input tokens and nothing else."""
    partial = Usage(input_tokens=120)
    assert partial.input_tokens == 120
    assert partial.output_tokens is None
    assert not partial.is_complete


def test_negative_token_count_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Usage(output_tokens=-1)


def test_chunk_count_is_not_a_token_count() -> None:
    """A streamed response's chunk count must not be usable as a token count."""
    from stealthbench.schemas.results import StreamingMeasurements

    streaming = StreamingMeasurements(chunk_count=250, total_seconds=10.0)
    assert streaming.chunk_count == 250
    assert streaming.output_tokens_per_second(Usage()) is None, (
        "deriving a token rate from chunk count would fabricate a measurement"
    )
    assert streaming.output_tokens_per_second(Usage(output_tokens=100)) == 10.0


def test_absent_timing_gives_an_absent_rate() -> None:
    from stealthbench.schemas.results import StreamingMeasurements

    streaming = StreamingMeasurements()
    assert streaming.output_tokens_per_second(Usage(output_tokens=100)) is None


def test_first_content_may_not_exceed_total() -> None:
    from stealthbench.schemas.results import StreamingMeasurements

    with pytest.raises(ValidationError, match="first_content_seconds"):
        StreamingMeasurements(first_content_seconds=5.0, total_seconds=2.0)


# ---------------------------------------------------------------------------
# Retries and accepted samples are distinct
# ---------------------------------------------------------------------------


def test_a_retry_does_not_add_a_sample(sample_key: SampleKey) -> None:
    failed = _result(sample_key, DeliveryStatus.TRANSPORT_FAILED, attempt=1, response=None)
    succeeded = _result(sample_key, attempt=2)
    keys = accepted_sample_keys([failed, succeeded])
    assert keys == {sample_key.as_tuple()}
    assert len(keys) == 1, "a retry must not inflate the denominator"


def test_repeat_id_distinguishes_measurements_from_retries(sample_key: SampleKey) -> None:
    first = sample_key
    second = sample_key.model_copy(update={"repeat_id": 2})
    keys = accepted_sample_keys([_result(first), _result(second)])
    assert len(keys) == 2, "a repeat is a distinct measurement"


def test_repeated_attempts_for_one_key_collapse_to_one_sample(sample_key: SampleKey) -> None:
    results = [
        _result(sample_key, DeliveryStatus.TRANSPORT_FAILED, 1, None),
        _result(sample_key, DeliveryStatus.TRANSPORT_FAILED, 2, None),
        _result(sample_key, DeliveryStatus.CANCELLED, 3, None),
        _result(sample_key, attempt=4),
    ]
    assert len(accepted_sample_keys(results)) == 1


def test_accepted_result_without_a_response_is_rejected(sample_key: SampleKey) -> None:
    with pytest.raises(ValidationError, match="accepted but carries no response"):
        _result(sample_key, response=None)


def test_unresolved_attempt_may_not_carry_a_response(sample_key: SampleKey) -> None:
    """A dispatched-but-unconfirmed request may have been billed; it is not a result."""
    with pytest.raises(ValidationError, match="may have been billed"):
        _result(sample_key, DeliveryStatus.UNRESOLVED, response="maybe")


def test_unresolved_attempts_are_reported_separately(sample_key: SampleKey) -> None:
    results = [
        _result(sample_key, attempt=1),
        _result(sample_key, DeliveryStatus.UNRESOLVED, 2, None),
        _result(sample_key, DeliveryStatus.UNRESOLVED, 3, None),
    ]
    assert unresolved_attempts(results) == ("attempt-2", "attempt-3")
    assert len(accepted_sample_keys(results)) == 1


def test_unresolved_attempt_is_not_an_accepted_sample(sample_key: SampleKey) -> None:
    result = _result(sample_key, DeliveryStatus.UNRESOLVED, response=None)
    assert not result.is_accepted_sample
    assert accepted_sample_keys([result]) == set()


def test_group_by_task_clusters_repeats(sample_key: SampleKey) -> None:
    first = _result(sample_key)
    second = _result(sample_key.model_copy(update={"repeat_id": 2}), attempt=2)
    third = _result(sample_key.model_copy(update={"task_id": "ifeval::item-2"}), attempt=3)
    grouped = group_by_task([first, second, third])
    assert sorted(grouped) == ["ifeval::item-1", "ifeval::item-2"]
    assert len(grouped["ifeval::item-1"]) == 2


def test_task_id_is_stable_and_namespaced() -> None:
    assert task_id_for("ifeval", "abc") == "ifeval::abc"
    assert task_id_for("ifeval", "abc") == task_id_for("ifeval", "abc")
    assert task_id_for("mmlu_pro", "abc") != task_id_for("ifeval", "abc")


def test_sample_key_from_task_spec(task: TaskSpec) -> None:
    assert task.sample_key.task_id == "ifeval::item-1"
    assert task.sample_key.repeat_id == 1


# ---------------------------------------------------------------------------
# Evaluator-only data cannot enter a request
# ---------------------------------------------------------------------------


def test_model_request_schema_contains_no_evaluator_only_field() -> None:
    leaked = set(ModelRequest.model_fields) & EVALUATOR_ONLY_FIELDS
    assert leaked == set(), f"evaluator-only fields leaked into ModelRequest: {sorted(leaked)}"


def test_model_request_is_closed_to_extra_fields(request_: ModelRequest) -> None:
    with pytest.raises(ValidationError):
        ModelRequest(
            messages=[{"role": "user", "content": "hi"}],
            max_output_tokens=8,
            gold_answer=GOLD_CANARY,
        )


def test_model_request_nested_extras_are_closed() -> None:
    """A message cannot smuggle an answer either."""
    with pytest.raises(ValidationError):
        ModelRequest(
            messages=[{"role": "user", "content": "hi", "gold": GOLD_CANARY}],
            max_output_tokens=8,
        )


def test_gold_answer_never_appears_in_the_dispatchable_request(task: TaskSpec) -> None:
    serialized = task.safe_request().model_dump_json()
    assert GOLD_CANARY not in serialized
    assert "hidden-test-suite.py" not in serialized
    assert "gold" not in serialized.lower()


def test_safe_request_detects_a_leak() -> None:
    """The guard must actually fire, not merely return the request unchanged."""
    poisoned = ModelRequest(
        messages=[{"role": "user", "content": f"What is it? {GOLD_CANARY}"}],
        max_output_tokens=8,
    )
    with pytest.raises(ValueError, match="would be sent to a provider"):
        make_task(poisoned).safe_request()


def test_a_task_built_with_a_stale_prompt_hash_cannot_be_dispatched(
    request_: ModelRequest,
) -> None:
    """Defect: prompt_artifact_hash was never checked, so any 64 hex characters passed."""
    payload = make_task(request_).model_dump()
    payload["prompt_artifact_hash"] = "b" * 64
    with pytest.raises(ValidationError, match="does not describe this request"):
        TaskSpec.model_validate(payload)


def test_a_constructed_task_rechecks_its_prompt_hash_at_dispatch(
    request_: ModelRequest,
) -> None:
    """`model_copy(update=)` and `model_construct` skip validators, so re-check on use.

    This is the defence that does not depend on the object having been built through
    validation: the recorded hash is recomputed from the request at dispatch time.
    """
    forged = make_task(request_).model_copy(update={"prompt_artifact_hash": "b" * 64})
    # ValueError, not ValidationError: this check runs outside pydantic's validator
    # machinery precisely so it still applies to an object that bypassed it.
    with pytest.raises(ValueError, match="does not describe this request"):
        forged.safe_request()


def test_stop_sequences_are_part_of_the_prompt_identity(request_: ModelRequest) -> None:
    """Defect: PromptRef dropped `stop`, so two different requests hashed alike."""
    plain = PromptRef.from_request(request_)
    stopped = PromptRef.from_request(request_.model_copy(update={"stop": ("\n\n",)}))
    assert plain.to_request() != stopped.to_request()
    assert prompt_hash(plain) != prompt_hash(stopped)


def test_prompt_ref_round_trips_every_dispatchable_field(request_: ModelRequest) -> None:
    full = request_.model_copy(
        update={
            "temperature": 0.7,
            "top_p": 0.9,
            "seed": 42,
            "stream": True,
            "stop": ("END",),
        }
    )
    assert PromptRef.from_request(full).to_request() == full


def test_two_tasks_differing_only_in_stop_have_different_trace_keys(
    request_: ModelRequest,
) -> None:
    def build(stop: tuple[str, ...]) -> TaskSpec:
        return make_task(request_.model_copy(update={"stop": stop}))

    plain, stopped = build(()), build(("\n\n",))
    assert plain.to_prompt_ref().stop != stopped.to_prompt_ref().stop
    assert trace_key(_MANIFEST, plain.to_prompt_ref()) != trace_key(
        _MANIFEST, stopped.to_prompt_ref()
    )


def test_model_construct_cannot_smuggle_a_gold_answer_past_the_dispatch_check(
    request_: ModelRequest,
) -> None:
    """`model_construct` bypasses validation, so `safe_request()` re-asserts the rule."""
    forged = TaskSpec.model_construct(
        benchmark_id="ifeval",
        item_id="i",
        campaign_id="c",
        endpoint_id="e",
        repeat_id=1,
        prompt_artifact_hash=prompt_hash(PromptRef.from_request(request_)),
        request=request_,
        evaluation=EvaluationPayload.model_construct(
            gold_answer=GOLD_CANARY, evaluator_id="e", evaluator_revision="r"
        ),
        declared_metadata={},
    )
    assert forged.evaluation.gold_answer == GOLD_CANARY
    assert GOLD_CANARY not in forged.safe_request().model_dump_json()


def test_evaluation_payload_is_not_part_of_the_request(task: TaskSpec) -> None:
    assert "evaluation" not in set(type(task.request).model_fields)
    assert task.evaluation.evaluator_revision not in task.safe_request().model_dump_json()


def test_forbidden_field_probe_reports_nothing_for_a_valid_request(
    request_: ModelRequest,
) -> None:
    assert request_.forbidden_fields_present() == frozenset()


# ---------------------------------------------------------------------------
# Grades: four orthogonal statuses
# ---------------------------------------------------------------------------


def test_transport_failure_is_not_a_wrong_answer(sample_key: SampleKey) -> None:
    grade = GradeResult(
        grader_version="1.0",
        sample_key=sample_key,
        correctness="unavailable",
        format="unavailable",
        transport="fail",
        evaluator="unavailable",
        denominator_eligibility=DenominatorEligibility(
            correctness=False, format=False, transport_success=False, evaluator_ran=False
        ),
    )
    assert not grade.counts_toward_accuracy
    assert grade.denominator_eligibility.correctness is False


def test_ineligibility_that_contradicts_the_statuses_is_rejected(sample_key: SampleKey) -> None:
    with pytest.raises(ValidationError, match="disagrees with the statuses"):
        GradeResult(
            grader_version="1.0",
            sample_key=sample_key,
            correctness="pass",
            format="pass",
            transport="pass",
            evaluator="pass",
            denominator_eligibility=DenominatorEligibility(
                correctness=False, format=True, transport_success=True, evaluator_ran=True
            ),
        )


def test_a_correct_answer_the_grader_never_ran_is_not_counted(sample_key: SampleKey) -> None:
    grade = GradeResult(
        grader_version="1.0",
        sample_key=sample_key,
        correctness="unavailable",
        format="pass",
        transport="pass",
        evaluator="fail",
        denominator_eligibility=DenominatorEligibility(
            correctness=False, format=True, transport_success=True, evaluator_ran=False
        ),
    )
    assert not grade.counts_toward_accuracy


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------


def test_wall_time_must_be_utc() -> None:
    from datetime import timedelta, timezone

    with pytest.raises(ValidationError, match="must be UTC"):
        RunEvent(
            event_id="e1",
            event_type="request.dispatched",
            campaign_id="c",
            wall_time_utc=datetime(2026, 10, 2, tzinfo=timezone(timedelta(hours=3))),
            monotonic_duration_seconds=0.1,
            payload_hash=DIGEST,
        )


def test_naive_wall_time_is_rejected() -> None:
    with pytest.raises(ValidationError, match="timezone-aware"):
        RunEvent(
            event_id="e1",
            event_type="request.dispatched",
            campaign_id="c",
            sequence=0,
            wall_time_utc=datetime(2026, 10, 2),
            monotonic_duration_seconds=0.1,
            payload_hash=DIGEST,
        )


def test_negative_duration_is_rejected() -> None:
    with pytest.raises(ValidationError):
        RunEvent(
            event_id="e1",
            event_type="request.dispatched",
            campaign_id="c",
            wall_time_utc=datetime(2026, 10, 2, tzinfo=UTC),
            monotonic_duration_seconds=-1.0,
            payload_hash=DIGEST,
        )


def test_completion_is_a_distinct_event_type() -> None:
    def event(event_type: str) -> RunEvent:
        return RunEvent(
            event_id="e",
            event_type=event_type,
            campaign_id="c",
            sequence=0,
            wall_time_utc=datetime(2026, 10, 2, tzinfo=UTC),
            monotonic_duration_seconds=0.0,
            payload_hash=DIGEST,
        )

    from stealthbench.schemas.results import EVENT_REQUEST_DISPATCHED, EVENT_REQUEST_FAILED

    assert event(EVENT_REQUEST_DISPATCHED).is_completion_record is False
    assert event(EVENT_REQUEST_FAILED).is_completion_record is False
    assert event(EVENT_REQUEST_COMPLETED).is_completion_record is True
    assert event(EVENT_SAMPLE_ACCEPTED).is_completion_record is True
    assert event(EVENT_REQUEST_DISPATCHED).is_attempt_record is True
    assert event(EVENT_SAMPLE_ACCEPTED).is_attempt_record is False


def test_payload_hash_must_be_a_digest() -> None:
    with pytest.raises(ValidationError):
        RunEvent(
            event_id="e",
            event_type="request.completed",
            campaign_id="c",
            wall_time_utc=datetime(2026, 10, 2, tzinfo=UTC),
            monotonic_duration_seconds=0.0,
            payload_hash="not-a-digest",
        )


# ---------------------------------------------------------------------------
# Signatures: unavailable stays unavailable
# ---------------------------------------------------------------------------


def test_validity_mask_must_match_the_vector() -> None:
    with pytest.raises(ValidationError, match="validity mask has"):
        SignatureResult(
            probe_version="probe-v1",
            endpoint_id="e",
            campaign_id="c",
            input_count_vector=(1, 2, 3),
            validity=("valid", "valid"),
        )


def test_a_missing_probe_carries_no_count() -> None:
    result = SignatureResult(
        probe_version="probe-v1",
        endpoint_id="e",
        campaign_id="c",
        input_count_vector=(12, None, 7),
        validity=("valid", "missing_count", "valid"),
    )
    assert result.usable_probes == 2
    assert result.valid_mask == (True, False, True)


def test_an_unusable_probe_may_not_carry_a_number() -> None:
    with pytest.raises(ValidationError, match="must not present a number"):
        SignatureResult(
            probe_version="probe-v1",
            endpoint_id="e",
            campaign_id="c",
            input_count_vector=(12, 7),
            validity=("valid", "unstable_repeat"),
        )


def test_a_valid_probe_must_carry_a_count() -> None:
    with pytest.raises(ValidationError, match="marked valid but has no count"):
        SignatureResult(
            probe_version="probe-v1",
            endpoint_id="e",
            campaign_id="c",
            input_count_vector=(None,),
            validity=("valid",),
        )


# ---------------------------------------------------------------------------
# Identity: similarity is not probability
# ---------------------------------------------------------------------------


def test_ranked_similarity_has_no_probability_field() -> None:
    assert "probability" not in RankedSimilarity.model_fields
    with pytest.raises(ValidationError):
        RankedSimilarity(
            candidate_id="c",
            candidate_library_revision="lib-v1",
            similarity=0.9,
            probability=0.9,  # type: ignore[call-arg]
        )


def test_similarity_must_be_within_range() -> None:
    with pytest.raises(ValidationError):
        RankedSimilarity(candidate_id="c", candidate_library_revision="lib-v1", similarity=1.4)


def test_similarities_must_be_ordered() -> None:
    with pytest.raises(ValidationError, match="descending"):
        IdentityReport(
            endpoint_id="e",
            candidate_library_revision="lib-v1",
            signal_evidence=(
                SignalEvidence(signal="behavior", available=True, comparable_observations=9),
            ),
            ranked_similarities=(
                RankedSimilarity(candidate_id="a", candidate_library_revision="l", similarity=0.2),
                RankedSimilarity(candidate_id="b", candidate_library_revision="l", similarity=0.8),
            ),
        )


def test_probabilities_without_evidence_are_rejected() -> None:
    with pytest.raises(ValidationError, match="calibration_evidence_id"):
        IdentityReport(
            endpoint_id="e",
            candidate_library_revision="lib-v1",
            signal_evidence=(
                SignalEvidence(signal="behavior", available=True, comparable_observations=9),
            ),
            ranked_similarities=(
                RankedSimilarity(candidate_id="a", candidate_library_revision="l", similarity=0.8),
            ),
            calibrated_probabilities={"a": 1.0},
        )


def test_probabilities_must_sum_to_one() -> None:
    with pytest.raises(ValidationError, match="sum to 1"):
        IdentityReport(
            endpoint_id="e",
            candidate_library_revision="lib-v1",
            signal_evidence=(
                SignalEvidence(signal="behavior", available=True, comparable_observations=9),
            ),
            ranked_similarities=(
                RankedSimilarity(candidate_id="a", candidate_library_revision="l", similarity=0.8),
            ),
            calibrated_probabilities={"a": 0.4},
            calibration_evidence_id="g12-holdout-1",
        )


def test_probabilities_must_reference_ranked_candidates() -> None:
    with pytest.raises(ValidationError, match="unranked candidates"):
        IdentityReport(
            endpoint_id="e",
            candidate_library_revision="lib-v1",
            signal_evidence=(
                SignalEvidence(signal="behavior", available=True, comparable_observations=9),
            ),
            ranked_similarities=(
                RankedSimilarity(candidate_id="a", candidate_library_revision="l", similarity=0.8),
            ),
            calibrated_probabilities={"a": 0.5, "z": 0.5},
            calibration_evidence_id="g12-holdout-1",
        )


def test_a_report_with_no_usable_signal_must_abstain() -> None:
    with pytest.raises(ValidationError, match="abstention_reason"):
        IdentityReport(
            endpoint_id="e",
            candidate_library_revision="lib-v1",
            signal_evidence=(
                SignalEvidence(
                    signal="input_count_vector", available=False, comparable_observations=0
                ),
            ),
            ranked_similarities=(
                RankedSimilarity(candidate_id="a", candidate_library_revision="l", similarity=0.9),
            ),
        )


def test_unknown_is_a_supported_outcome() -> None:
    report = IdentityReport(
        endpoint_id="e",
        candidate_library_revision="lib-v1",
        signal_evidence=(
            SignalEvidence(signal="performance", available=False, comparable_observations=0),
        ),
        ranked_similarities=(),
        abstention_reason="insufficient_evidence",
    )
    assert report.is_unknown
    assert report.calibrated_probabilities is None


def test_a_shared_tokenizer_is_a_tokenizer_signal_not_identity() -> None:
    """Two models sharing a tokenizer must not collapse into one identity claim."""
    report = IdentityReport(
        endpoint_id="anon-alias",
        candidate_library_revision="lib-v1",
        signal_evidence=(
            SignalEvidence(
                signal="input_count_vector",
                available=True,
                comparable_observations=180,
                detail="exact count-vector match on 60 probes",
            ),
        ),
        ranked_similarities=(
            RankedSimilarity(
                candidate_id="model-a",
                candidate_library_revision="lib-v1",
                similarity=1.0,
                evidence_signals=("input_count_vector",),
                note="tokenization evidence only; does not establish shared weights",
            ),
        ),
    )
    assert report.ranked_similarities[0].similarity == 1.0
    assert report.calibrated_probabilities is None
    assert "does not establish" in (report.ranked_similarities[0].note or "")


def test_reveal_label_is_separate_from_the_prediction() -> None:
    report = IdentityReport(
        endpoint_id="e",
        candidate_library_revision="lib-v1",
        signal_evidence=(
            SignalEvidence(signal="behavior", available=True, comparable_observations=9),
        ),
        ranked_similarities=(
            RankedSimilarity(candidate_id="a", candidate_library_revision="l", similarity=0.7),
        ),
        predicted_at=datetime(2026, 10, 2, tzinfo=UTC),
        official_reveal_label="provider/model-x 2026-09-01",
    )
    assert report.official_reveal_label is not None
    assert report.predicted_at is not None


# ---------------------------------------------------------------------------
# Gate evidence cannot claim a pass it does not have
# ---------------------------------------------------------------------------


def test_gate_cannot_pass_with_a_failing_test() -> None:
    with pytest.raises(ValidationError, match="failing tests"):
        GateEvidence(
            gate_id="G01",
            source_revision="abc",
            commands=(CommandRecord(command="pytest", exit_status=1),),
            collected=10,
            passed=9,
            failed=1,
            skipped=0,
            artifacts=("gate.json",),
            review_decision="pass",
        )


def test_gate_cannot_pass_on_an_empty_suite() -> None:
    with pytest.raises(ValidationError, match="zero collected tests"):
        GateEvidence(
            gate_id="G01",
            source_revision="abc",
            commands=(CommandRecord(command="pytest", exit_status=0),),
            collected=0,
            passed=0,
            failed=0,
            skipped=0,
            artifacts=("gate.json",),
            review_decision="pass",
        )


def test_gate_cannot_pass_without_commands_or_artifacts() -> None:
    with pytest.raises(ValidationError, match="without a recorded command"):
        GateEvidence(
            gate_id="G01",
            source_revision="abc",
            collected=5,
            passed=5,
            failed=0,
            skipped=0,
            artifacts=("gate.json",),
            review_decision="pass",
        )
    with pytest.raises(ValidationError, match="without a recorded artifact"):
        GateEvidence(
            gate_id="G01",
            source_revision="abc",
            commands=(CommandRecord(command="pytest", exit_status=0),),
            collected=5,
            passed=5,
            failed=0,
            skipped=0,
            review_decision="pass",
        )


def test_counts_must_not_exceed_collected() -> None:
    with pytest.raises(ValidationError, match="exceeds collected"):
        GateEvidence(
            gate_id="G01",
            source_revision="abc",
            commands=(CommandRecord(command="pytest", exit_status=0),),
            collected=3,
            passed=5,
            failed=0,
            skipped=0,
            artifacts=("gate.json",),
            review_decision="pass",
        )


def test_a_pass_with_skips_but_no_passes_is_rejected() -> None:
    """Every test skipped is not a pass, however clean the exit status looks."""
    with pytest.raises(ValidationError, match="zero passing tests"):
        GateEvidence(
            gate_id="G01",
            source_revision="abc",
            commands=(CommandRecord(command="pytest", exit_status=0),),
            collected=10,
            passed=0,
            failed=0,
            skipped=10,
            artifacts=("gate.json",),
            review_decision="pass",
        )


def test_a_pass_with_a_nonzero_command_status_is_rejected() -> None:
    """A recorded command that failed cannot sit under a passing verdict."""
    with pytest.raises(ValidationError, match="returned non-zero"):
        GateEvidence(
            gate_id="G01",
            source_revision="abc",
            commands=(CommandRecord(command="pytest", exit_status=1),),
            collected=10,
            passed=10,
            failed=0,
            skipped=0,
            artifacts=("gate.json",),
            review_decision="pass",
        )


def test_a_pass_with_some_passes_and_a_skip_is_allowed() -> None:
    """A skip alongside real passes is recorded, not rejected; it is simply disclosed."""
    evidence = GateEvidence(
        gate_id="G01",
        source_revision="abc",
        commands=(CommandRecord(command="pytest", exit_status=0),),
        collected=10,
        passed=9,
        failed=0,
        skipped=1,
        artifacts=("gate.json",),
        review_decision="pass",
    )
    assert evidence.passed == 9
    assert evidence.has_vacuous_check == ("1 skipped",)


def test_blocked_external_needs_no_tests_to_pass() -> None:
    evidence = GateEvidence(
        gate_id="G14",
        source_revision="abc",
        collected=0,
        passed=0,
        failed=0,
        skipped=0,
        review_decision="blocked_external",
    )
    assert evidence.review_decision == "blocked_external"


def test_unknown_event_type_is_rejected() -> None:
    """A typo in an event name must not invent a new kind of record."""
    with pytest.raises(ValidationError, match="not in the declared vocabulary"):
        RunEvent(
            event_id="e",
            event_type="sample.accept",
            campaign_id="c",
            sequence=0,
            wall_time_utc=datetime(2026, 10, 2, tzinfo=UTC),
            monotonic_duration_seconds=0.0,
            payload_hash=DIGEST,
        )


# ---------------------------------------------------------------------------
# Regression: the recorded prompt hash must describe the request it accompanies
# ---------------------------------------------------------------------------


def test_event_sequence_is_required() -> None:
    """Sequence makes event identity unique and makes a spliced log detectable."""
    with pytest.raises(ValidationError, match="sequence"):
        RunEvent(
            event_id="e",
            event_type="request.completed",
            campaign_id="c",
            wall_time_utc=datetime(2026, 10, 2, tzinfo=UTC),
            monotonic_duration_seconds=0.0,
            payload_hash=DIGEST,
        )
