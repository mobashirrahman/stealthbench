"""Contract tests for the official EvalPlus wrapper (task T07D).

Acceptance for T07D: pinned official extended-test outcomes match golden
fixtures and accepted samples define pass@1.

The expected outcomes below are hand-computed from the documented official
timeout formula, memory policy, md5 gate and unbiased pass@k estimator, not
derived from the implementation. Production wiring supplies verdicts from
the pinned official checker (``evalplus``) executed inside the G07 sandbox;
the adapter under test only maps verdicts onto ``GradeResult``.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from stealthbench.benchmarks.evalplus import (
    DEFAULT_BASE_TIMEOUT_SECONDS,
    DEFAULT_MEMORY_CAP_BYTES,
    DEFAULT_TIMEOUT_FACTOR,
    GRADER_VERSION,
    HUMANEVAL_PLUS_DATASET_ID,
    HUMANEVAL_PLUS_ITEM_COUNT,
    MBPP_PLUS_DATASET_ID,
    MBPP_PLUS_ITEM_COUNT,
    PINNED_HUMANEVAL_PLUS_VERSION,
    PINNED_MBPP_PLUS_VERSION,
    SANDBOX_BACKEND_AVAILABLE,
    EvalPlusGrade,
    dispatch_request,
    estimate_pass_at_k,
    evalplus_timeout,
    grade_accepted_sample,
    grade_generation,
    grade_with_checker,
    is_malformed,
    resolve_memory_cap,
    summarize_grades,
    verify_md5,
)
from stealthbench.schemas.manifest import PromptRef, prompt_hash
from stealthbench.schemas.results import (
    EVALUATOR_ONLY_FIELDS,
    DeliveryStatus,
    EvaluationPayload,
    GenerationResult,
    ModelRequest,
    SampleKey,
    TaskSpec,
)

pytestmark = pytest.mark.contract

GOLD_CANARY = "canary-gold-evalplus-t07d-8c44"


def _task(
    item_id: str = "HumanEvalPlus/001",
    prompt: str = "Write a function that adds two integers.",
    gold: str = GOLD_CANARY,
) -> TaskSpec:
    request = ModelRequest(
        messages=[{"role": "user", "content": prompt}],
        max_output_tokens=512,
    )
    return TaskSpec(
        benchmark_id="evalplus",
        item_id=item_id,
        campaign_id="c1",
        endpoint_id="e1",
        repeat_id=1,
        prompt_artifact_hash=prompt_hash(PromptRef.from_request(request)),
        request=request,
        evaluation=EvaluationPayload(
            gold_answer=gold,
            evaluator_id="evalplus.eval",
            evaluator_revision=(
                f"humaneval-{PINNED_HUMANEVAL_PLUS_VERSION}+mbpp-{PINNED_MBPP_PLUS_VERSION}"
            ),
        ),
    )


def _generation(
    task: TaskSpec,
    response: str | None,
    status: DeliveryStatus = DeliveryStatus.ACCEPTED,
    attempt: int = 1,
) -> GenerationResult:
    return GenerationResult(
        attempt_id=f"attempt-{attempt}",
        sample_key=task.sample_key,
        attempt_number=attempt,
        delivery_status=status,
        response=response,
    )


# ---------------------------------------------------------------------------
# Wrapper identity pins the official dataset releases
# ---------------------------------------------------------------------------


def test_grader_version_pins_both_dataset_releases() -> None:
    assert PINNED_HUMANEVAL_PLUS_VERSION == "v0.1.10"
    assert PINNED_MBPP_PLUS_VERSION == "v0.2.0"
    assert PINNED_HUMANEVAL_PLUS_VERSION in GRADER_VERSION
    assert PINNED_MBPP_PLUS_VERSION in GRADER_VERSION
    assert GRADER_VERSION.startswith("evalplus.eval@")


def test_dataset_identities_match_inventory() -> None:
    assert HUMANEVAL_PLUS_DATASET_ID == "evalplus/humanevalplus"
    assert HUMANEVAL_PLUS_ITEM_COUNT == 164
    assert MBPP_PLUS_DATASET_ID == "evalplus/mbppplus"
    assert MBPP_PLUS_ITEM_COUNT == 378
    assert DEFAULT_BASE_TIMEOUT_SECONDS == 4.0
    assert DEFAULT_TIMEOUT_FACTOR == 4.0
    assert DEFAULT_MEMORY_CAP_BYTES == 4 * 1024 * 1024 * 1024


def test_sandbox_backend_flag_reflects_sibling_availability() -> None:
    try:
        from stealthbench.sandbox import (
            policy,  # type: ignore[import-not-found] # noqa: F401
            runtime,  # type: ignore[import-not-found] # noqa: F401
        )

        expected = True
    except ImportError:
        expected = False
    assert SANDBOX_BACKEND_AVAILABLE is expected


# ---------------------------------------------------------------------------
# Per-task timeout T = max(T_base, T_gt * k)
# ---------------------------------------------------------------------------


def test_evalplus_timeout_matches_hand_computed_values() -> None:
    assert evalplus_timeout(0.0) == 4.0
    assert evalplus_timeout(0.5) == 4.0
    assert evalplus_timeout(1.0) == 4.0
    assert evalplus_timeout(1.5) == 6.0
    assert evalplus_timeout(2.0) == 8.0
    assert evalplus_timeout(0.25, base_seconds=2.0, factor=4.0) == 2.0


def test_evalplus_timeout_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="base_seconds must be positive"):
        evalplus_timeout(1.0, base_seconds=0.0)
    with pytest.raises(ValueError, match="factor must be positive"):
        evalplus_timeout(1.0, factor=-1.0)
    with pytest.raises(ValueError, match="non-negative"):
        evalplus_timeout(-0.5)


# ---------------------------------------------------------------------------
# Memory cap: min(4GB, system max); override wins; -1 means unlimited
# ---------------------------------------------------------------------------


def test_memory_cap_matches_hand_computed_values() -> None:
    four_gb = 4 * 1024 * 1024 * 1024
    assert resolve_memory_cap(system_maximum_bytes=8 * 1024**3) == four_gb
    assert resolve_memory_cap(system_maximum_bytes=1024) == 1024
    assert resolve_memory_cap(system_maximum_bytes=None) == four_gb
    assert resolve_memory_cap(system_maximum_bytes=8 * 1024**3, override_bytes=512) == 512
    assert resolve_memory_cap(system_maximum_bytes=8 * 1024**3, override_bytes=-1) is None


def test_memory_cap_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="positive"):
        resolve_memory_cap(system_maximum_bytes=0)
    with pytest.raises(ValueError, match="non-negative or -1"):
        resolve_memory_cap(system_maximum_bytes=None, override_bytes=-2)


# ---------------------------------------------------------------------------
# MD5 checksum gate mirrors the upstream dataset verification
# ---------------------------------------------------------------------------


def test_md5_gate_matches_independent_hashlib() -> None:
    payload = b"evalplus ground-truth outputs"
    expected = hashlib.md5(payload, usedforsecurity=False).hexdigest()
    assert verify_md5(payload, expected_md5=expected) == expected
    assert verify_md5(payload, expected_md5=expected.upper()) == expected


def test_md5_mismatch_refuses_to_run() -> None:
    with pytest.raises(ValueError, match="md5 mismatch"):
        verify_md5(b"tampered", expected_md5="0" * 32)


# ---------------------------------------------------------------------------
# Unbiased pass@k estimator 1 - C(n-c, k) / C(n, k)
# ---------------------------------------------------------------------------


def test_pass_at_k_matches_hand_computed_fractions() -> None:
    assert estimate_pass_at_k(10, 7, 1) == pytest.approx(0.7)
    # 1 - C(3,2)/C(5,2) = 1 - 3/10 = 0.7
    assert estimate_pass_at_k(5, 2, 2) == pytest.approx(0.7)
    assert estimate_pass_at_k(4, 4, 2) == 1.0
    assert estimate_pass_at_k(4, 0, 2) == 0.0
    # Too few failures to choose k from: success is certain.
    assert estimate_pass_at_k(3, 3, 2) == 1.0
    # Single greedy sample reduces to accuracy.
    assert estimate_pass_at_k(1, 1, 1) == 1.0
    assert estimate_pass_at_k(1, 0, 1) == 0.0


def test_pass_at_k_rejects_impossible_counts() -> None:
    with pytest.raises(ValueError, match="n must be positive"):
        estimate_pass_at_k(0, 0, 1)
    with pytest.raises(ValueError, match=r"k.*positive"):
        estimate_pass_at_k(4, 2, 0)
    with pytest.raises(ValueError, match="cannot exceed n"):
        estimate_pass_at_k(2, 1, 3)
    with pytest.raises(ValueError, match="0 <= c <= n"):
        estimate_pass_at_k(4, 5, 1)


# ---------------------------------------------------------------------------
# Golden fixtures: extended-test pass / fail / timeout / OOM
# ---------------------------------------------------------------------------


def test_extended_tests_passing_grades_as_correct() -> None:
    task = _task()
    grade = grade_accepted_sample(
        task=task,
        response="def add(a, b):\n    return a + b",
        passed=True,
    )
    assert grade.grade.correctness == "pass"
    assert grade.grade.score_components["pass_at_1"] == 1.0
    assert grade.grade.counts_toward_accuracy
    assert grade.timed_out is False
    assert grade.out_of_memory is False


def test_extended_tests_failing_grades_as_incorrect() -> None:
    task = _task()
    grade = grade_accepted_sample(
        task=task,
        response="def add(a, b):\n    return a - b",
        passed=False,
    )
    assert grade.grade.correctness == "fail"
    assert grade.grade.score_components["pass_at_1"] == 0.0
    assert grade.grade.counts_toward_accuracy
    assert grade.grade.denominator_eligibility.correctness is True


def test_timeout_is_a_failure_not_missing_data() -> None:
    task = _task()
    grade = grade_accepted_sample(
        task=task,
        response="def add(a, b):\n    while True:\n        pass",
        passed=False,
        timed_out=True,
        timeout_seconds=8.0,
    )
    assert grade.grade.correctness == "fail"
    assert grade.grade.evaluator == "pass"
    assert grade.grade.transport == "pass"
    assert grade.grade.counts_toward_accuracy
    assert grade.timed_out is True
    assert grade.timeout_seconds == 8.0


def test_out_of_memory_is_a_failure_not_missing_data() -> None:
    task = _task()
    grade = grade_accepted_sample(
        task=task,
        response="def add(a, b):\n    return a + b",
        passed=False,
        out_of_memory=True,
        memory_cap_bytes=1024,
    )
    assert grade.grade.correctness == "fail"
    assert grade.grade.counts_toward_accuracy
    assert grade.out_of_memory is True
    assert grade.memory_cap_bytes == 1024


def test_a_pass_contradicting_a_timeout_or_oom_is_rejected() -> None:
    task = _task()
    with pytest.raises(ValueError, match="contradicts"):
        grade_accepted_sample(task=task, response="x", passed=True, timed_out=True)
    with pytest.raises(ValueError, match="contradicts"):
        grade_accepted_sample(task=task, response="x", passed=True, out_of_memory=True)


def test_timeout_and_memory_are_recorded_verbatim() -> None:
    task = _task()
    first = grade_accepted_sample(
        task=task, response="x", passed=False, timed_out=True, timeout_seconds=4.0
    )
    second = grade_accepted_sample(
        task=task, response="x", passed=False, timed_out=True, timeout_seconds=12.0
    )
    assert first.timeout_seconds == 4.0
    assert second.timeout_seconds == 12.0


def test_checker_mapping_and_crash_paths() -> None:
    task = _task()
    good = grade_with_checker(
        task=task, response="def f(): pass", check=lambda _r: (True, False, False)
    )
    assert good.grade.correctness == "pass"
    slow = grade_with_checker(task=task, response="x", check=lambda _r: (False, True, False))
    assert slow.grade.correctness == "fail"
    assert slow.timed_out is True

    def crashing(_response: str) -> tuple[bool, bool, bool]:
        raise RuntimeError("official checker exploded")

    failed = grade_with_checker(task=task, response="x", check=crashing)
    assert failed.grade.evaluator == "fail"
    assert failed.grade.correctness == "unavailable"
    assert not failed.grade.counts_toward_accuracy
    # A malformed verdict shape is an evaluator failure, not a pass.
    shape_bad = grade_with_checker(task=task, response="x", check=lambda _r: "pass")
    assert shape_bad.grade.evaluator == "fail"
    assert shape_bad.grade.correctness == "unavailable"


def test_is_malformed_flags_only_empty_responses() -> None:
    assert is_malformed("")
    assert is_malformed("   \n  ")
    assert not is_malformed("def f(): pass")


def test_empty_program_is_a_format_failure_in_the_denominator() -> None:
    task = _task()
    grade = grade_accepted_sample(task=task, response="   ", passed=False)
    assert grade.grade.format == "fail"
    assert grade.grade.denominator_eligibility.format is True
    assert grade.grade.correctness == "fail"
    assert grade.grade.counts_toward_accuracy


# ---------------------------------------------------------------------------
# Transport failures are unavailable; gold never enters the request
# ---------------------------------------------------------------------------


def test_transport_failure_is_unavailable_not_incorrect() -> None:
    task = _task()
    generation = _generation(task, None, DeliveryStatus.TRANSPORT_FAILED)
    grade = grade_generation(task=task, generation=generation, passed=True)
    assert grade.grade.correctness == "unavailable"
    assert grade.grade.transport == "fail"
    assert grade.grade.evaluator == "unavailable"
    assert not grade.grade.counts_toward_accuracy


def test_generation_timeout_stays_a_failure_in_the_denominator() -> None:
    task = _task()
    grade = grade_generation(
        task=task,
        generation=_generation(task, "def f():\n    while True: pass"),
        passed=False,
        timed_out=True,
        timeout_seconds=8.0,
    )
    assert grade.grade.correctness == "fail"
    assert grade.grade.counts_toward_accuracy


def test_an_accepted_response_without_a_verdict_is_unevaluated() -> None:
    task = _task()
    grade = grade_generation(task=task, generation=_generation(task, "def f(): pass"))
    assert grade.grade.evaluator == "fail"
    assert grade.grade.correctness == "unavailable"
    assert not grade.grade.counts_toward_accuracy


def test_sample_key_mismatch_refuses_to_grade() -> None:
    task = _task(item_id="HumanEvalPlus/001")
    other = _task(item_id="HumanEvalPlus/002")
    generation = _generation(other, "def f(): pass")
    with pytest.raises(ValueError, match="does not match"):
        grade_generation(task=task, generation=generation, passed=True)


def test_gold_answer_never_appears_in_the_dispatchable_request() -> None:
    task = _task()
    serialized = dispatch_request(task).model_dump_json()
    assert GOLD_CANARY not in serialized
    assert "gold" not in serialized.lower()


def test_model_request_schema_has_no_evaluator_only_field() -> None:
    assert set(ModelRequest.model_fields) & EVALUATOR_ONLY_FIELDS == set()


def test_a_prompt_smuggling_gold_is_rejected_before_grading() -> None:
    request = ModelRequest(
        messages=[{"role": "user", "content": f"Repeat this back: {GOLD_CANARY}"}],
        max_output_tokens=512,
    )
    task = TaskSpec(
        benchmark_id="evalplus",
        item_id="HumanEvalPlus/001",
        campaign_id="c1",
        endpoint_id="e1",
        repeat_id=1,
        prompt_artifact_hash=prompt_hash(PromptRef.from_request(request)),
        request=request,
        evaluation=EvaluationPayload(
            gold_answer=GOLD_CANARY,
            evaluator_id="evalplus.eval",
            evaluator_revision=(
                f"humaneval-{PINNED_HUMANEVAL_PLUS_VERSION}+mbpp-{PINNED_MBPP_PLUS_VERSION}"
            ),
        ),
    )
    with pytest.raises(ValueError, match="would be sent to a provider"):
        dispatch_request(task)
    with pytest.raises(ValueError, match="would be sent to a provider"):
        grade_accepted_sample(task=task, response="def f(): pass", passed=True)


# ---------------------------------------------------------------------------
# pass@1 comes from accepted samples only; unknown stays null
# ---------------------------------------------------------------------------


def test_timeouts_and_transport_have_different_denominators() -> None:
    task = _task()
    timed_out = grade_generation(
        task=task,
        generation=_generation(task, "def f():\n    while True: pass"),
        passed=False,
        timed_out=True,
    )
    failed = grade_generation(
        task=task,
        generation=_generation(task, None, DeliveryStatus.TRANSPORT_FAILED, attempt=2),
        passed=True,
    )
    assert timed_out.grade.counts_toward_accuracy
    assert not failed.grade.counts_toward_accuracy
    summary = summarize_grades([timed_out, failed])
    assert summary.eligible == 1
    assert summary.passed == 0
    assert summary.pass_at_1 == 0.0


def test_summary_is_not_best_of_several_attempts() -> None:
    task_a = _task(item_id="HumanEvalPlus/001")
    task_b = _task(item_id="HumanEvalPlus/002")
    good = grade_accepted_sample(task=task_a, response="def f(): pass", passed=True)
    bad = grade_accepted_sample(task=task_b, response="def f(): pass", passed=False)
    summary = summarize_grades([good, bad])
    assert (summary.passed, summary.eligible) == (1, 2)
    assert summary.pass_at_1 == pytest.approx(0.5)


def test_unknown_rate_is_null_never_zero() -> None:
    empty = summarize_grades([])
    assert empty.pass_at_1 is None
    assert empty.pass_at_1 != 0.0  # None is not a measured zero
    task = _task()
    failed_only = summarize_grades(
        [
            grade_generation(
                task=task,
                generation=_generation(task, None, DeliveryStatus.TRANSPORT_FAILED),
            )
        ]
    )
    assert failed_only.eligible == 0
    assert failed_only.pass_at_1 is None
    dumped = json.loads(failed_only.model_dump_json())
    assert dumped["pass_at_1"] is None


def test_grade_survives_a_result_schema_round_trip() -> None:
    task = _task()
    grade = grade_accepted_sample(
        task=task, response="def f(): pass", passed=True, timeout_seconds=4.0
    )
    revived = EvalPlusGrade.model_validate(grade.model_dump(mode="json"))
    assert revived == grade
    sample: SampleKey = task.sample_key
    assert revived.grade.sample_key == sample
