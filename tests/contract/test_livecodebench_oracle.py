"""Contract tests for the official LiveCodeBench wrapper (task T07C).

Acceptance for T07C: correct, subtly incorrect and timeout fixtures match
official grading; problem dates and requested IDs persist.

The ``outcome`` strings below are hand-computed official verdicts, not
derived from the implementation. Production wiring supplies verdicts from
the pinned official checker (``lcb_runner``) executed inside the G07
sandbox; the adapter under test only maps verdicts onto ``GradeResult``.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from stealthbench.benchmarks.livecodebench import (
    DATASET_ID,
    DATE_WINDOW_END,
    DATE_WINDOW_START,
    DEFAULT_TIMEOUT_SECONDS,
    GRADER_VERSION,
    LATEST_CONTEST_DATE,
    PINNED_ITEM_COUNT,
    PINNED_LIVECODEBENCH_COMMIT,
    REPORT_START_DATE,
    SANDBOX_BACKEND_AVAILABLE,
    SPLIT,
    LiveCodeBenchGrade,
    LiveCodeBenchItem,
    LiveCodeBenchSummary,
    apply_errata_exclusions,
    dispatch_request,
    filter_by_date_window,
    grade_accepted_sample,
    grade_generation,
    grade_with_checker,
    is_malformed,
    parse_contest_date,
    selection_checksum,
    summarize_grades,
    verify_requested_ids_preserved,
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

GOLD_CANARY = "canary-gold-livecodebench-t07c-4d21"


def _task(
    item_id: str = "lcb-0001",
    prompt: str = "Write a function that returns the sum of two integers.",
    gold: str = GOLD_CANARY,
) -> TaskSpec:
    request = ModelRequest(
        messages=[{"role": "user", "content": prompt}],
        max_output_tokens=512,
    )
    return TaskSpec(
        benchmark_id="livecodebench",
        item_id=item_id,
        campaign_id="c1",
        endpoint_id="e1",
        repeat_id=1,
        prompt_artifact_hash=prompt_hash(PromptRef.from_request(request)),
        request=request,
        evaluation=EvaluationPayload(
            gold_answer=gold,
            evaluator_id="lcb_runner.evaluation.compute_scores",
            evaluator_revision=PINNED_LIVECODEBENCH_COMMIT,
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


def _item(item_id: str, contest_date: str) -> LiveCodeBenchItem:
    return LiveCodeBenchItem(
        item_id=item_id,
        prompt=f"Solve problem {item_id}.",
        contest_date=contest_date,
    )


# ---------------------------------------------------------------------------
# Wrapper identity pins the official commit and dataset window
# ---------------------------------------------------------------------------


def test_grader_version_pins_the_official_commit() -> None:
    assert PINNED_LIVECODEBENCH_COMMIT in GRADER_VERSION
    assert GRADER_VERSION.startswith("lcb_runner.evaluation.compute_scores@")


def test_dataset_identity_matches_inventory() -> None:
    assert DATASET_ID == "livecodebench/code_generation_lite"
    assert SPLIT == "release_v6"
    assert PINNED_ITEM_COUNT == 1055
    assert DATE_WINDOW_START == "2023-05-01"
    assert DATE_WINDOW_END == "2025-04-30"
    assert REPORT_START_DATE == "2023-09-01"
    assert LATEST_CONTEST_DATE == "2025-03-29"
    assert DEFAULT_TIMEOUT_SECONDS == 6.0


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
# Contest dates and date-window slicing preserve IDs in order
# ---------------------------------------------------------------------------


def test_parse_contest_date_accepts_valid_dates() -> None:
    assert parse_contest_date("2023-09-15") == "2023-09-15"
    assert parse_contest_date(LATEST_CONTEST_DATE) == LATEST_CONTEST_DATE


def test_parse_contest_date_rejects_malformed_dates() -> None:
    for bad in ["2023-9-5", "2023/09/15", "not-a-date", "2023-13-01", "2023-00-10", ""]:
        with pytest.raises(ValueError, match="contest_date"):
            parse_contest_date(bad)


def test_filter_by_date_window_preserves_order_and_bounds() -> None:
    items = (
        _item("a", "2023-05-01"),
        _item("b", "2023-09-01"),
        _item("c", "2024-06-15"),
        _item("d", "2025-04-30"),
        _item("e", "2025-05-01"),
    )
    selected = filter_by_date_window(items, start_date="2023-09-01", end_date="2025-04-30")
    assert [i.item_id for i in selected] == ["b", "c", "d"]
    # Inclusive bounds: window edges are kept.
    assert (
        filter_by_date_window(items[:1], start_date="2023-05-01", end_date="2023-05-01")[0].item_id
        == "a"
    )


def test_filter_by_date_window_rejects_an_inverted_window() -> None:
    with pytest.raises(ValueError, match="after end_date"):
        filter_by_date_window(
            (_item("a", "2024-01-01"),), start_date="2024-02-01", end_date="2024-01-01"
        )


def test_requested_ids_must_survive_in_order() -> None:
    verify_requested_ids_preserved(requested_ids=["a", "b"], selected_ids=["a", "b"])
    with pytest.raises(ValueError, match="not preserved"):
        verify_requested_ids_preserved(requested_ids=["a", "b"], selected_ids=["b", "a"])
    with pytest.raises(ValueError, match="not preserved"):
        verify_requested_ids_preserved(requested_ids=["a", "b"], selected_ids=["a"])
    with pytest.raises(ValueError, match="not preserved"):
        verify_requested_ids_preserved(requested_ids=["a"], selected_ids=["a", "extra"])


# ---------------------------------------------------------------------------
# ERRATA exclusions are declared, never silent; checksum is order-sensitive
# ---------------------------------------------------------------------------


def test_errata_exclusions_require_an_explicit_declared_set() -> None:
    items = (_item("a", "2024-01-01"), _item("b", "2024-01-02"), _item("c", "2024-01-03"))
    kept, excluded = apply_errata_exclusions(items, declared_excluded_ids=[])
    assert [i.item_id for i in kept] == ["a", "b", "c"]
    assert excluded == ()
    kept2, excluded2 = apply_errata_exclusions(items, declared_excluded_ids=["b"])
    assert [i.item_id for i in kept2] == ["a", "c"]
    assert excluded2 == ("b",)


def test_errata_exclusions_reject_duplicates_and_unknown_ids() -> None:
    items = (_item("a", "2024-01-01"),)
    with pytest.raises(ValueError, match="duplicates"):
        apply_errata_exclusions(items, declared_excluded_ids=["a", "a"])
    with pytest.raises(ValueError, match="unknown item IDs"):
        apply_errata_exclusions(items, declared_excluded_ids=["ghost"])


def test_selection_checksum_matches_independent_sha256_and_is_order_sensitive() -> None:
    ids = ["lcb-0001", "lcb-0002"]
    expected = hashlib.sha256(b"lcb-0001\nlcb-0002\n").hexdigest()
    assert selection_checksum(ids) == expected
    assert selection_checksum(list(reversed(ids))) != selection_checksum(ids)


# ---------------------------------------------------------------------------
# Golden fixtures: correct / subtly-wrong / timeout / runtime error
# ---------------------------------------------------------------------------


def test_known_correct_solution_passes() -> None:
    task = _task()
    grade = grade_accepted_sample(
        task=task,
        response="def add(a, b):\n    return a + b",
        outcome="pass",
        timeout_seconds=6.0,
        contest_date="2024-06-15",
    )
    assert grade.grade.correctness == "pass"
    assert grade.grade.score_components["pass_at_1"] == 1.0
    assert grade.grade.counts_toward_accuracy
    assert grade.outcome == "pass"
    assert grade.contest_date == "2024-06-15"
    assert grade.timeout_seconds == 6.0


def test_subtly_wrong_solution_fails_but_stays_in_the_denominator() -> None:
    task = _task()
    grade = grade_accepted_sample(
        task=task,
        response="def add(a, b):\n    return a - b",
        outcome="wrong_answer",
        timeout_seconds=6.0,
        contest_date="2024-06-15",
    )
    assert grade.grade.correctness == "fail"
    assert grade.grade.score_components["pass_at_1"] == 0.0
    assert grade.grade.counts_toward_accuracy
    assert grade.grade.denominator_eligibility.correctness is True


def test_timeout_is_a_graded_failure_not_missing_data() -> None:
    task = _task()
    grade = grade_accepted_sample(
        task=task,
        response="def add(a, b):\n    while True:\n        pass",
        outcome="timeout",
        timeout_seconds=6.0,
        contest_date="2024-06-15",
    )
    assert grade.grade.correctness == "fail"
    assert grade.grade.evaluator == "pass"
    assert grade.grade.transport == "pass"
    assert grade.grade.counts_toward_accuracy
    assert grade.outcome == "timeout"
    assert grade.timeout_seconds == 6.0


def test_runtime_error_is_a_graded_failure() -> None:
    task = _task()
    grade = grade_accepted_sample(
        task=task,
        response="def add(a, b):\n    return undefined_name",
        outcome="runtime_error",
        timeout_seconds=6.0,
    )
    assert grade.grade.correctness == "fail"
    assert grade.grade.counts_toward_accuracy


def test_timeout_is_recorded_verbatim_and_never_retuned() -> None:
    task = _task()
    fast = grade_accepted_sample(
        task=task, response="def f():\n    pass", outcome="timeout", timeout_seconds=2.0
    )
    slow = grade_accepted_sample(
        task=task, response="def f():\n    pass", outcome="timeout", timeout_seconds=30.0
    )
    assert fast.timeout_seconds == 2.0
    assert slow.timeout_seconds == 30.0
    assert fast.grade.correctness == "fail"
    assert slow.grade.correctness == "fail"


def test_unknown_outcome_and_bad_timeout_are_rejected() -> None:
    task = _task()
    with pytest.raises(ValueError, match="not one of"):
        grade_accepted_sample(task=task, response="x", outcome="maybe")
    with pytest.raises(ValueError, match="timeout_seconds must be positive"):
        grade_accepted_sample(task=task, response="x", outcome="pass", timeout_seconds=0.0)
    with pytest.raises(ValueError, match="contest_date"):
        grade_accepted_sample(task=task, response="x", outcome="pass", contest_date="not-a-date")


def test_checker_crash_is_an_evaluator_failure_not_an_incorrect_answer() -> None:
    def crashing(_response: str) -> str:
        raise RuntimeError("official checker exploded")

    task = _task()
    grade = grade_with_checker(task=task, response="def f():\n    pass", check=crashing)
    assert grade.grade.evaluator == "fail"
    assert grade.grade.correctness == "unavailable"
    assert not grade.grade.counts_toward_accuracy


def test_checker_success_maps_through_the_official_verdict() -> None:
    task = _task()
    grade = grade_with_checker(task=task, response="def f():\n    pass", check=lambda _r: "pass")
    assert grade.grade.correctness == "pass"
    wrong = grade_with_checker(
        task=task, response="def f():\n    pass", check=lambda _r: "wrong_answer"
    )
    assert wrong.grade.correctness == "fail"


def test_is_malformed_flags_only_empty_responses() -> None:
    assert is_malformed("")
    assert is_malformed("   \n  ")
    assert not is_malformed("def f(): pass")


def test_empty_program_is_a_format_failure_in_the_denominator() -> None:
    task = _task()
    grade = grade_accepted_sample(task=task, response="   ", outcome="wrong_answer")
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
    grade = grade_generation(task=task, generation=generation, outcome="pass")
    assert grade.grade.correctness == "unavailable"
    assert grade.grade.transport == "fail"
    assert grade.grade.evaluator == "unavailable"
    assert not grade.grade.counts_toward_accuracy


def test_cancelled_and_unresolved_are_not_counted_as_incorrect() -> None:
    task = _task()
    cancelled = grade_generation(
        task=task, generation=_generation(task, None, DeliveryStatus.CANCELLED)
    )
    unresolved = grade_generation(
        task=task, generation=_generation(task, None, DeliveryStatus.UNRESOLVED)
    )
    assert cancelled.grade.transport == "invalid"
    assert unresolved.grade.transport == "unavailable"
    for grade in (cancelled, unresolved):
        assert grade.grade.correctness == "unavailable"
        assert not grade.grade.counts_toward_accuracy


def test_generation_timeout_stays_a_failure_in_the_denominator() -> None:
    task = _task()
    grade = grade_generation(
        task=task,
        generation=_generation(task, "def f():\n    while True: pass"),
        outcome="timeout",
        timeout_seconds=6.0,
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
    task = _task(item_id="lcb-0001")
    other = _task(item_id="lcb-0002")
    generation = _generation(other, "def f(): pass")
    with pytest.raises(ValueError, match="does not match"):
        grade_generation(task=task, generation=generation, outcome="pass")


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
        benchmark_id="livecodebench",
        item_id="lcb-0001",
        campaign_id="c1",
        endpoint_id="e1",
        repeat_id=1,
        prompt_artifact_hash=prompt_hash(PromptRef.from_request(request)),
        request=request,
        evaluation=EvaluationPayload(
            gold_answer=GOLD_CANARY,
            evaluator_id="lcb_runner.evaluation.compute_scores",
            evaluator_revision=PINNED_LIVECODEBENCH_COMMIT,
        ),
    )
    with pytest.raises(ValueError, match="would be sent to a provider"):
        dispatch_request(task)
    with pytest.raises(ValueError, match="would be sent to a provider"):
        grade_accepted_sample(task=task, response="def f(): pass", outcome="pass")


# ---------------------------------------------------------------------------
# pass@1 comes from accepted samples only; unknown stays null
# ---------------------------------------------------------------------------


def test_incorrect_and_transport_failure_have_different_denominators() -> None:
    task = _task()
    wrong = grade_generation(
        task=task, generation=_generation(task, "def f():\n    return 0"), outcome="wrong_answer"
    )
    failed = grade_generation(
        task=task,
        generation=_generation(task, None, DeliveryStatus.TRANSPORT_FAILED, attempt=2),
        outcome="pass",
    )
    assert wrong.grade.counts_toward_accuracy
    assert not failed.grade.counts_toward_accuracy
    summary = summarize_grades([wrong, failed], timeout_seconds=6.0)
    assert summary.eligible == 1
    assert summary.passed == 0
    assert summary.pass_at_1 == 0.0
    assert summary.timeout_seconds == 6.0


def test_summary_is_not_best_of_several_attempts() -> None:
    task_a = _task(item_id="lcb-0001")
    task_b = _task(item_id="lcb-0002")
    good = grade_accepted_sample(task=task_a, response="def f():\n    pass", outcome="pass")
    bad = grade_accepted_sample(task=task_b, response="def f():\n    pass", outcome="wrong_answer")
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


def test_summary_rejects_a_non_positive_recorded_timeout() -> None:
    with pytest.raises(ValueError, match="timeout_seconds must be positive"):
        summarize_grades([], timeout_seconds=0.0)


def test_grade_survives_a_result_schema_round_trip() -> None:
    task = _task()
    grade = grade_accepted_sample(
        task=task,
        response="def f():\n    pass",
        outcome="pass",
        timeout_seconds=6.0,
        contest_date="2024-01-15",
    )
    revived = LiveCodeBenchGrade.model_validate(grade.model_dump(mode="json"))
    assert revived == grade
    sample: SampleKey = task.sample_key
    assert revived.grade.sample_key == sample
    summary = LiveCodeBenchSummary.model_validate(summarize_grades([grade]).model_dump(mode="json"))
    assert summary.pass_at_1 == 1.0
