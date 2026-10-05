"""Contract tests for the BFCL tool evaluation adapter (task T08A).

Acceptance for T08A: native and prompted modes remain separate; invalid,
irrelevant and parallel calls match category oracles.

Expected verdicts below are hand-computed from the frozen protocol stated
in ``stealthbench.benchmarks.bfcl``, not derived from the implementation.
Production wiring supplies real model tool calls; the adapter only applies
the strict AST-equivalence oracle and maps verdicts onto ``GradeResult``.
"""

from __future__ import annotations

import json

import pytest

from stealthbench.benchmarks import bfcl
from stealthbench.benchmarks.bfcl import (
    GRADER_VERSION,
    IRRELEVANCE_CATEGORIES,
    NATIVE_PROFILE,
    NON_SCORING_CATEGORIES,
    PINNED_BFCL_COMMIT,
    PROMPTED_PROFILE,
    RETIRED_CATEGORIES,
    SCORING_CATEGORIES,
    SERPAPI_REQUIRED_CATEGORIES,
    BfclGrade,
    RetiredCategoryError,
    ToolCall,
    UnknownCategoryError,
    args_equal,
    calls_match,
    check_tool_calls,
    classify_category,
    dispatch_request,
    grade_accepted_sample,
    grade_generation,
    is_supported,
    summarize,
    summarize_category,
    unsupported_reason,
    verify_bfcl_revision,
)
from stealthbench.benchmarks.datasets import DatasetRevisionMismatch
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

GOLD_CANARY = "canary-gold-bfcl-t08a-3e71"


def _task(
    item_id: str = "item-001",
    prompt: str = "Call get_weather with a city.",
    gold: str = GOLD_CANARY,
) -> TaskSpec:
    request = ModelRequest(
        messages=[{"role": "user", "content": prompt}],
        max_output_tokens=128,
    )
    return TaskSpec(
        benchmark_id="bfcl",
        item_id=item_id,
        campaign_id="c1",
        endpoint_id="e1",
        repeat_id=1,
        prompt_artifact_hash=prompt_hash(PromptRef.from_request(request)),
        request=request,
        evaluation=EvaluationPayload(
            gold_answer=gold,
            evaluator_id="bfcl-eval",
            evaluator_revision=PINNED_BFCL_COMMIT,
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


def _call(function: str, arguments: dict[str, object] | None = None) -> ToolCall:
    return ToolCall(function=function, arguments=dict(arguments or {}))


# ---------------------------------------------------------------------------
# Pins and frozen category sets
# ---------------------------------------------------------------------------


def test_grader_version_pins_the_official_commit() -> None:
    assert PINNED_BFCL_COMMIT == "f7cf7359b7ac615a0b294831c5ba2bc95ee4a000"
    assert PINNED_BFCL_COMMIT in GRADER_VERSION
    assert GRADER_VERSION.startswith("bfcl-eval@")


def test_evaluation_temperature_is_greedy() -> None:
    assert pytest.approx(0.001) == bfcl.EVALUATION_TEMPERATURE


def test_scoring_set_excludes_retired_and_non_scoring() -> None:
    assert "restful" not in SCORING_CATEGORIES
    assert "executable" not in SCORING_CATEGORIES
    for retired in RETIRED_CATEGORIES:
        assert retired not in SCORING_CATEGORIES
    for non_scoring in NON_SCORING_CATEGORIES:
        assert non_scoring not in SCORING_CATEGORIES
    assert set(NON_SCORING_CATEGORIES) == {"live_relevance", "format_sensitivity"}


def test_classify_category_oracle() -> None:
    assert classify_category("simple_python") == "scoring"
    assert classify_category("irrelevance") == "scoring"
    assert classify_category("live_relevance") == "non_scoring"
    assert classify_category("format_sensitivity") == "non_scoring"
    assert classify_category("restful") == "retired"
    assert classify_category("executable") == "retired"
    assert classify_category("Restful_Python") == "retired"
    assert classify_category("EXECUTABLE_JS") == "retired"
    assert classify_category("not_a_bfcl_category") == "unknown"


def test_web_search_requires_serpapi() -> None:
    assert "agentic_web_search" in SERPAPI_REQUIRED_CATEGORIES
    assert not is_supported("agentic_web_search", NATIVE_PROFILE, has_serpapi=False)
    assert is_supported("agentic_web_search", NATIVE_PROFILE, has_serpapi=True)
    assert unsupported_reason("agentic_web_search", NATIVE_PROFILE) is not None


def test_format_sensitivity_is_prompted_only() -> None:
    assert not is_supported("format_sensitivity", NATIVE_PROFILE)
    assert is_supported("format_sensitivity", PROMPTED_PROFILE)


# ---------------------------------------------------------------------------
# Category oracles: valid / invalid / missing / parallel / unicode
# ---------------------------------------------------------------------------


def test_valid_arguments_pass() -> None:
    task = _task()
    grade = grade_accepted_sample(
        task=task,
        predicted_calls=[_call("get_weather", {"city": "Paris"})],
        expected_calls=[_call("get_weather", {"city": "Paris"})],
        category="simple_python",
        profile=NATIVE_PROFILE,
        offered_tools=["get_weather", "get_time"],
    )
    assert grade.profile == NATIVE_PROFILE
    assert grade.category == "simple_python"
    assert grade.grade.correctness == "pass"
    assert grade.grade.counts_toward_accuracy


def test_invalid_argument_value_fails() -> None:
    task = _task()
    grade = grade_accepted_sample(
        task=task,
        predicted_calls=[_call("get_weather", {"city": "London"})],
        expected_calls=[_call("get_weather", {"city": "Paris"})],
        category="simple_python",
        profile=NATIVE_PROFILE,
    )
    assert grade.grade.correctness == "fail"
    assert grade.grade.counts_toward_accuracy


def test_invalid_argument_type_fails() -> None:
    assert not args_equal({"count": 1}, {"count": "1"})
    assert not args_equal({"count": 1}, {"count": 1.0})
    assert not args_equal({"flag": True}, {"flag": 1})
    task = _task()
    grade = grade_accepted_sample(
        task=task,
        predicted_calls=[_call("add", {"x": "1"})],
        expected_calls=[_call("add", {"x": 1})],
        category="simple_python",
        profile=NATIVE_PROFILE,
    )
    assert grade.grade.correctness == "fail"


def test_wrong_function_name_fails() -> None:
    task = _task()
    grade = grade_accepted_sample(
        task=task,
        predicted_calls=[_call("get_time", {"city": "Paris"})],
        expected_calls=[_call("get_weather", {"city": "Paris"})],
        category="simple_python",
        profile=NATIVE_PROFILE,
    )
    assert grade.grade.correctness == "fail"


def test_missing_parameter_fails() -> None:
    assert not args_equal({"a": 1, "b": 2}, {"a": 1})
    task = _task()
    grade = grade_accepted_sample(
        task=task,
        predicted_calls=[_call("book", {"city": "Paris"})],
        expected_calls=[_call("book", {"city": "Paris", "date": "2026-10-04"})],
        category="multiple",
        profile=NATIVE_PROFILE,
    )
    assert grade.grade.correctness == "fail"


def test_extra_parameter_fails() -> None:
    task = _task()
    grade = grade_accepted_sample(
        task=task,
        predicted_calls=[_call("book", {"city": "Paris", "extra": "x"})],
        expected_calls=[_call("book", {"city": "Paris"})],
        category="multiple",
        profile=NATIVE_PROFILE,
    )
    assert grade.grade.correctness == "fail"


def test_irrelevant_tools_fail_and_empty_passes() -> None:
    task = _task()
    silent = grade_accepted_sample(
        task=task,
        predicted_calls=[],
        expected_calls=[],
        category="irrelevance",
        profile=NATIVE_PROFILE,
        offered_tools=["get_weather"],
    )
    assert silent.grade.correctness == "pass"
    noisy = grade_accepted_sample(
        task=task,
        predicted_calls=[_call("get_weather", {"city": "Paris"})],
        expected_calls=[],
        category="irrelevance",
        profile=NATIVE_PROFILE,
        offered_tools=["get_weather"],
    )
    assert noisy.grade.correctness == "fail"
    assert noisy.grade.counts_toward_accuracy


def test_live_irrelevance_empty_passes() -> None:
    assert "live_irrelevance" in IRRELEVANCE_CATEGORIES
    assert check_tool_calls([], [], "live_irrelevance", ["a"])


def test_hallucinated_function_outside_offered_tools_fails() -> None:
    task = _task()
    grade = grade_accepted_sample(
        task=task,
        predicted_calls=[_call("secret_tool", {"x": 1})],
        expected_calls=[_call("secret_tool", {"x": 1})],
        category="simple_python",
        profile=NATIVE_PROFILE,
        offered_tools=["get_weather"],
    )
    assert grade.grade.correctness == "fail"


def test_parallel_calls_match_order_insensitive() -> None:
    first = _call("a", {"x": 1})
    second = _call("b", {"y": "z"})
    assert calls_match([first, second], [second, first])
    assert check_tool_calls([first, second], [second, first], "parallel")


def test_parallel_missing_or_extra_fails() -> None:
    first = _call("a", {"x": 1})
    second = _call("b", {"y": "z"})
    assert not calls_match([first], [first, second])
    assert not calls_match([first, second, _call("c", {})], [first, second])
    assert not check_tool_calls([first], [first, second], "parallel")
    assert not check_tool_calls([first, first], [first], "parallel")


def test_unicode_arguments_are_exact() -> None:
    assert args_equal({"city": "café ☃"}, {"city": "café ☃"})
    assert not args_equal({"city": "café"}, {"city": "cafe"})
    assert not args_equal({"city": "Hello"}, {"city": "hello"})
    task = _task()
    grade = grade_accepted_sample(
        task=task,
        predicted_calls=[_call("greet", {"name": "café"})],
        expected_calls=[_call("greet", {"name": "cafe"})],
        category="simple_python",
        profile=NATIVE_PROFILE,
    )
    assert grade.grade.correctness == "fail"


def test_nested_and_list_arguments_compare_exactly() -> None:
    assert args_equal({"q": {"a": [1, 2]}}, {"q": {"a": [1, 2]}})
    assert not args_equal({"q": {"a": [1, 2]}}, {"q": {"a": [2, 1]}})
    assert not args_equal({"q": [1, 2, 3]}, {"q": [1, 2]})


# ---------------------------------------------------------------------------
# Mode separation: native vs prompted are distinct tracks
# ---------------------------------------------------------------------------


def test_native_and_prompted_grades_carry_their_track() -> None:
    task = _task()
    native = grade_accepted_sample(
        task=task,
        predicted_calls=[_call("f", {"x": 1})],
        expected_calls=[_call("f", {"x": 1})],
        category="simple_python",
        profile=NATIVE_PROFILE,
    )
    prompted = grade_accepted_sample(
        task=task,
        predicted_calls=[_call("f", {"x": 1})],
        expected_calls=[_call("f", {"x": 1})],
        category="simple_python",
        profile=PROMPTED_PROFILE,
    )
    assert native.profile == "native_function_calling"
    assert prompted.profile == "prompted"
    assert native.grade == prompted.grade
    assert (native.profile, native.category) != (prompted.profile, prompted.category) or (
        native.profile != prompted.profile
    )


def test_summaries_never_average_tracks() -> None:
    task = _task()
    native = grade_accepted_sample(
        task=task,
        predicted_calls=[_call("f", {"x": 1})],
        expected_calls=[_call("f", {"x": 1})],
        category="simple_python",
        profile=NATIVE_PROFILE,
    )
    prompted = grade_accepted_sample(
        task=task,
        predicted_calls=[_call("f", {"x": 1})],
        expected_calls=[_call("f", {"x": 1})],
        category="simple_python",
        profile=PROMPTED_PROFILE,
    )
    with pytest.raises(ValueError, match="mix BFCL tracks"):
        summarize(
            [native, prompted],
            profile=NATIVE_PROFILE,
            planned_by_category={"simple_python": 2},
        )
    with pytest.raises(ValueError, match="mix BFCL tracks"):
        summarize_category(
            [native, prompted],
            category="simple_python",
            profile=NATIVE_PROFILE,
            planned=2,
        )


def test_per_track_summaries_stay_separate() -> None:
    task = _task()
    native_pass = grade_accepted_sample(
        task=task,
        predicted_calls=[_call("f", {"x": 1})],
        expected_calls=[_call("f", {"x": 1})],
        category="simple_python",
        profile=NATIVE_PROFILE,
    )
    prompted_fail = grade_accepted_sample(
        task=task,
        predicted_calls=[_call("f", {"x": 2})],
        expected_calls=[_call("f", {"x": 1})],
        category="simple_python",
        profile=PROMPTED_PROFILE,
    )
    native_summary = summarize(
        [native_pass], profile=NATIVE_PROFILE, planned_by_category={"simple_python": 1}
    )
    prompted_summary = summarize(
        [prompted_fail],
        profile=PROMPTED_PROFILE,
        planned_by_category={"simple_python": 1},
    )
    assert native_summary.per_category[0].accuracy == 1.0
    assert prompted_summary.per_category[0].accuracy == 0.0
    assert native_summary.profile != prompted_summary.profile


def test_format_sensitivity_tracks_differ() -> None:
    task = _task()
    native = grade_accepted_sample(
        task=task,
        predicted_calls=[],
        expected_calls=[],
        category="format_sensitivity",
        profile=NATIVE_PROFILE,
    )
    prompted = grade_accepted_sample(
        task=task,
        predicted_calls=[],
        expected_calls=[],
        category="format_sensitivity",
        profile=PROMPTED_PROFILE,
    )
    assert native.grade.correctness == "unavailable"
    assert not native.grade.counts_toward_accuracy
    assert prompted.grade.correctness == "pass"


# ---------------------------------------------------------------------------
# Unavailable denominators: never zero-filled, denominator always shown
# ---------------------------------------------------------------------------


def test_serpapi_gated_category_is_unavailable_without_credential() -> None:
    task = _task()
    grade = grade_accepted_sample(
        task=task,
        predicted_calls=[_call("search", {"q": "x"})],
        expected_calls=[_call("search", {"q": "x"})],
        category="agentic_web_search",
        profile=NATIVE_PROFILE,
        has_serpapi=False,
    )
    assert grade.grade.correctness == "unavailable"
    assert not grade.grade.counts_toward_accuracy
    assert grade.grade.denominator_eligibility.correctness is False


def test_unavailable_cell_shows_its_denominator_and_stays_null() -> None:
    task = _task()
    gated = grade_accepted_sample(
        task=task,
        predicted_calls=[_call("search", {"q": "x"})],
        expected_calls=[_call("search", {"q": "x"})],
        category="agentic_web_search",
        profile=NATIVE_PROFILE,
        has_serpapi=False,
    )
    cell = summarize_category(
        [gated],
        category="agentic_web_search",
        profile=NATIVE_PROFILE,
        planned=4,
    )
    assert cell.eligible == 0
    assert cell.planned == 4
    assert cell.accuracy is None
    assert cell.accuracy != 0.0  # type: ignore[comparison-overlap]
    assert cell.status == "unavailable"
    dumped = json.loads(cell.model_dump_json())
    assert dumped["accuracy"] is None
    assert dumped["planned"] == 4


def test_partial_coverage_has_no_complete_core_score() -> None:
    task = _task(item_id="item-001")
    good = grade_accepted_sample(
        task=task,
        predicted_calls=[_call("f", {"x": 1})],
        expected_calls=[_call("f", {"x": 1})],
        category="simple_python",
        profile=NATIVE_PROFILE,
    )
    planned = dict.fromkeys(SCORING_CATEGORIES, 1)
    summary = summarize([good], profile=NATIVE_PROFILE, planned_by_category=planned)
    assert summary.complete is False
    assert summary.overall_accuracy is None
    assert summary.overall_accuracy != 0.0  # type: ignore[comparison-overlap]
    assert summary.scoring_planned == len(SCORING_CATEGORIES)
    missing = next(c for c in summary.per_category if c.category == "multiple")
    assert missing.eligible == 0
    assert missing.planned == 1
    assert missing.accuracy is None


def test_full_coverage_produces_a_micro_average() -> None:
    grades: list[BfclGrade] = []
    expected_correct = 0
    for index, category in enumerate(SCORING_CATEGORIES):
        task = _task(item_id=f"item-{index:03d}")
        correct = index % 2 == 0
        if category in IRRELEVANCE_CATEGORIES:
            predicted = [] if correct else [_call("f", {"x": 1})]
            expected_calls: list[ToolCall] = []
        else:
            predicted = [_call("f", {"x": 1 if correct else 2})]
            expected_calls = [_call("f", {"x": 1})]
        grade = grade_accepted_sample(
            task=task,
            predicted_calls=predicted,
            expected_calls=expected_calls,
            category=category,
            profile=NATIVE_PROFILE,
            has_serpapi=True,
        )
        grades.append(grade)
        if grade.grade.correctness == "pass":
            expected_correct += 1
    planned = dict.fromkeys(SCORING_CATEGORIES, 1)
    summary = summarize(grades, profile=NATIVE_PROFILE, planned_by_category=planned)
    assert summary.complete is True
    assert summary.scoring_eligible == len(SCORING_CATEGORIES)
    assert summary.scoring_correct == expected_correct
    assert summary.overall_accuracy == pytest.approx(expected_correct / len(SCORING_CATEGORIES))


def test_non_scoring_cells_never_enter_the_overall_mean() -> None:
    task = _task()
    scoring = grade_accepted_sample(
        task=task,
        predicted_calls=[_call("f", {"x": 1})],
        expected_calls=[_call("f", {"x": 1})],
        category="simple_python",
        profile=PROMPTED_PROFILE,
    )
    non_scoring = grade_accepted_sample(
        task=task,
        predicted_calls=[],
        expected_calls=[_call("f", {"x": 1})],
        category="live_relevance",
        profile=PROMPTED_PROFILE,
    )
    assert non_scoring.grade.correctness == "fail"
    planned = {"simple_python": 1, "live_relevance": 5}
    summary = summarize(
        [scoring, non_scoring], profile=PROMPTED_PROFILE, planned_by_category=planned
    )
    assert summary.scoring_correct == 1
    assert summary.scoring_eligible == 1
    by_name = {cell.category: cell for cell in summary.per_category}
    assert by_name["live_relevance"].status == "non_scoring"


# ---------------------------------------------------------------------------
# Retired categories are never reported
# ---------------------------------------------------------------------------


def test_retired_categories_raise_on_every_entry_point() -> None:
    task = _task()
    for retired in ("restful", "executable", "Restful_Python", "executable_js"):
        with pytest.raises(RetiredCategoryError):
            grade_accepted_sample(
                task=task,
                predicted_calls=[],
                expected_calls=[],
                category=retired,
                profile=NATIVE_PROFILE,
            )
        with pytest.raises(RetiredCategoryError):
            summarize_category([], category=retired, profile=NATIVE_PROFILE, planned=1)
        with pytest.raises(RetiredCategoryError):
            summarize([], profile=NATIVE_PROFILE, planned_by_category={retired: 1})


def test_unknown_category_raises_instead_of_inventing() -> None:
    task = _task()
    with pytest.raises(UnknownCategoryError):
        grade_accepted_sample(
            task=task,
            predicted_calls=[],
            expected_calls=[],
            category="bfcl_v99_future",
            profile=NATIVE_PROFILE,
        )
    with pytest.raises(UnknownCategoryError):
        summarize([], profile=NATIVE_PROFILE, planned_by_category={"bfcl_v99_future": 1})


# ---------------------------------------------------------------------------
# Transport / evaluator separation, gold hygiene, revision pins
# ---------------------------------------------------------------------------


def test_transport_failure_is_unavailable_not_incorrect() -> None:
    task = _task()
    generation = _generation(task, None, DeliveryStatus.TRANSPORT_FAILED)
    grade = grade_generation(
        task=task,
        generation=generation,
        predicted_calls=[_call("f", {"x": 1})],
        expected_calls=[_call("f", {"x": 1})],
        category="simple_python",
        profile=NATIVE_PROFILE,
    )
    assert grade.grade.correctness == "unavailable"
    assert grade.grade.transport == "fail"
    assert not grade.grade.counts_toward_accuracy


def test_cancelled_and_unresolved_are_not_counted_as_incorrect() -> None:
    task = _task()
    cancelled = grade_generation(
        task=task,
        generation=_generation(task, None, DeliveryStatus.CANCELLED),
        predicted_calls=[],
        expected_calls=[],
        category="simple_python",
        profile=NATIVE_PROFILE,
    )
    unresolved = grade_generation(
        task=task,
        generation=_generation(task, None, DeliveryStatus.UNRESOLVED),
        predicted_calls=[],
        expected_calls=[],
        category="simple_python",
        profile=NATIVE_PROFILE,
    )
    assert cancelled.grade.transport == "invalid"
    assert unresolved.grade.transport == "unavailable"
    for grade in (cancelled, unresolved):
        assert grade.grade.correctness == "unavailable"
        assert not grade.grade.counts_toward_accuracy


def test_accepted_response_without_parsed_calls_is_unevaluated() -> None:
    task = _task()
    grade = grade_generation(
        task=task,
        generation=_generation(task, "some text"),
        predicted_calls=None,
        expected_calls=[_call("f", {"x": 1})],
        category="simple_python",
        profile=NATIVE_PROFILE,
    )
    assert grade.grade.evaluator == "fail"
    assert grade.grade.correctness == "unavailable"
    assert not grade.grade.counts_toward_accuracy


def test_sample_key_mismatch_refuses_to_grade() -> None:
    task = _task(item_id="item-001")
    other = _task(item_id="item-002")
    generation = _generation(other, "tool call")
    with pytest.raises(ValueError, match="does not match"):
        grade_generation(
            task=task,
            generation=generation,
            predicted_calls=[],
            expected_calls=[],
            category="simple_python",
            profile=NATIVE_PROFILE,
        )


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
        max_output_tokens=64,
    )
    task = TaskSpec(
        benchmark_id="bfcl",
        item_id="item-001",
        campaign_id="c1",
        endpoint_id="e1",
        repeat_id=1,
        prompt_artifact_hash=prompt_hash(PromptRef.from_request(request)),
        request=request,
        evaluation=EvaluationPayload(
            gold_answer=GOLD_CANARY,
            evaluator_id="bfcl-eval",
            evaluator_revision=PINNED_BFCL_COMMIT,
        ),
    )
    with pytest.raises(ValueError, match="would be sent to a provider"):
        dispatch_request(task)
    with pytest.raises(ValueError, match="would be sent to a provider"):
        grade_accepted_sample(
            task=task,
            predicted_calls=[],
            expected_calls=[],
            category="simple_python",
            profile=NATIVE_PROFILE,
        )


def test_unsupported_revision_is_rejected_not_substituted() -> None:
    with pytest.raises(DatasetRevisionMismatch):
        verify_bfcl_revision("some-unpinned-revision")
    assert verify_bfcl_revision(PINNED_BFCL_COMMIT) == PINNED_BFCL_COMMIT


def test_grade_survives_a_result_schema_round_trip() -> None:
    task = _task()
    grade = grade_accepted_sample(
        task=task,
        predicted_calls=[_call("f", {"x": 1})],
        expected_calls=[_call("f", {"x": 1})],
        category="simple_python",
        profile=NATIVE_PROFILE,
    )
    revived = BfclGrade.model_validate(grade.model_dump(mode="json"))
    assert revived == grade
    sample: SampleKey = task.sample_key
    assert revived.grade.sample_key == sample
