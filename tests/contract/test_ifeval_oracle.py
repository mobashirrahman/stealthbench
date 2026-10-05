"""Contract tests for the official IFEval wrapper (task T05B).

Acceptance for T05B: strict and loose grades match independent pinned official
golden fixtures.

The ``check`` callables below are test doubles standing in for the pinned
official instruction checkers (``instruction_following_eval``); the expected
strict/loose verdicts are hand-computed from the documented loose-variant
rule, not derived from the implementation. Production wiring supplies the real
official checkers; the adapter under test only expands variants and maps the
verdicts onto ``GradeResult``.
"""

from __future__ import annotations

import json

import pytest

from stealthbench.benchmarks.datasets import PINNED_IFEVAL_EVALUATOR_REVISION
from stealthbench.benchmarks.ifeval import (
    GRADER_VERSION,
    LOOSE_VARIANT_NAMES,
    IfEvalGradePair,
    dispatch_request,
    evaluate_loose,
    evaluate_strict,
    grade_accepted_sample,
    grade_generation,
    is_malformed,
    loose_variants,
    summarize_pairs,
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

GOLD_CANARY = "canary-gold-ifeval-t05b-9f31"


def _task(
    item_id: str = "item-001",
    prompt: str = "Write a reply containing the word blueberry.",
    gold: str = GOLD_CANARY,
) -> TaskSpec:
    request = ModelRequest(
        messages=[{"role": "user", "content": prompt}],
        max_output_tokens=64,
    )
    return TaskSpec(
        benchmark_id="ifeval",
        item_id=item_id,
        campaign_id="c1",
        endpoint_id="e1",
        repeat_id=1,
        prompt_artifact_hash=prompt_hash(PromptRef.from_request(request)),
        request=request,
        evaluation=EvaluationPayload(
            gold_answer=gold,
            evaluator_id="instruction_following_eval",
            evaluator_revision=PINNED_IFEVAL_EVALUATOR_REVISION,
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


def _contains(keyword: str):  # type: ignore[no-untyped-def]
    """Stand-in for an official keyword checker (case-insensitive)."""

    def check(response: str) -> bool:
        return keyword.lower() in response.lower()

    return check


# ---------------------------------------------------------------------------
# Wrapper identity and loose-variant contract
# ---------------------------------------------------------------------------


def test_grader_version_pins_the_official_commit() -> None:
    assert PINNED_IFEVAL_EVALUATOR_REVISION in GRADER_VERSION
    assert GRADER_VERSION.startswith("instruction_following_eval@")


def test_loose_variants_are_seven_deterministic_transforms() -> None:
    assert len(LOOSE_VARIANT_NAMES) == 7
    assert len(set(LOOSE_VARIANT_NAMES)) == 7
    response = "line1\nline2\nline3"
    assert loose_variants(response) == loose_variants(response)
    assert len(loose_variants(response)) == 7


def test_loose_variant_transforms_match_hand_computed_values() -> None:
    response = "line1\nline2\nline3"
    variants = loose_variants(response)
    by_name = dict(zip(LOOSE_VARIANT_NAMES, variants, strict=True))
    assert by_name["strip_asterisks"] == response
    assert by_name["remove_first_line"] == "line2\nline3"
    assert by_name["remove_last_line"] == "line1\nline2"
    assert by_name["remove_first_and_last_lines"] == "line2"
    assert by_name["remove_first_line_strip_asterisks"] == "line2\nline3"
    starred = "**hello**\nworld"
    starred_variants = dict(zip(LOOSE_VARIANT_NAMES, loose_variants(starred), strict=True))
    assert starred_variants["strip_asterisks"] == "hello\nworld"
    assert starred_variants["remove_first_line"] == "world"
    assert starred_variants["remove_first_line_strip_asterisks"] == "world"


def test_is_malformed_flags_only_empty_responses() -> None:
    assert is_malformed("")
    assert is_malformed("   \n  ")
    assert not is_malformed("blueberry")
    assert not is_malformed("  blueberry  ")


# ---------------------------------------------------------------------------
# Known compliant / noncompliant answers (golden fixtures)
# ---------------------------------------------------------------------------


def test_known_compliant_answer_passes_both_sides() -> None:
    task = _task()
    pair = grade_accepted_sample(
        task=task, response="Here is a fresh blueberry pie.", check=_contains("blueberry")
    )
    assert pair.strict.correctness == "pass"
    assert pair.loose.correctness == "pass"
    assert pair.strict.score_components["prompt_level_strict"] == 1.0
    assert pair.loose.score_components["prompt_level_loose"] == 1.0
    assert pair.strict.counts_toward_accuracy
    assert pair.loose.counts_toward_accuracy


def test_known_noncompliant_answer_fails_but_still_counts() -> None:
    task = _task()
    pair = grade_accepted_sample(
        task=task, response="Here is a fresh apple pie.", check=_contains("blueberry")
    )
    assert pair.strict.correctness == "fail"
    assert pair.loose.correctness == "fail"
    assert pair.strict.score_components["prompt_level_strict"] == 0.0
    # A wrong answer is still a graded answer: it belongs in the denominator.
    assert pair.strict.counts_toward_accuracy
    assert pair.strict.denominator_eligibility.correctness is True


def test_golden_fixture_table_matches_hand_computed_verdicts() -> None:
    """Independent oracle: hand-computed (strict, loose) per row, not from the code."""
    check = _contains("blueberry")
    rows = [
        ("blueberry", True, True),
        ("BLUEBERRY muffin", True, True),
        ("apple pie", False, False),
        ("", False, False),
    ]
    for response, want_strict, want_loose in rows:
        assert evaluate_strict(response, check) is want_strict
        assert evaluate_loose(response, check) is want_loose


# ---------------------------------------------------------------------------
# Strict/loose distinction fixtures
# ---------------------------------------------------------------------------


def test_first_line_removal_distinguishes_strict_from_loose() -> None:
    """A preamble line fails strict but the first-line-removed variant passes."""

    def check(response: str) -> bool:
        return response.startswith("ANSWER:")

    task = _task()
    pair = grade_accepted_sample(
        task=task, response="preamble line\nANSWER: blueberry", check=check
    )
    assert pair.strict.correctness == "fail"
    assert pair.loose.correctness == "pass"


def test_asterisk_stripping_distinguishes_strict_from_loose() -> None:
    """Markdown bold fails a strict exact match but passes after `*` stripping."""

    def check(response: str) -> bool:
        return response == "hello world"

    task = _task()
    pair = grade_accepted_sample(task=task, response="**hello world**", check=check)
    assert pair.strict.correctness == "fail"
    assert pair.loose.correctness == "pass"


def test_loose_is_monotone_over_strict() -> None:
    """Whenever strict passes, loose must pass too (official reporter invariant)."""
    check = _contains("blueberry")
    for response in [
        "blueberry",
        "a BLUEBERRY tart",
        "preamble\nblueberry inside",
        "**blueberry**",
        "no match here",
        "",
    ]:
        if evaluate_strict(response, check):
            assert evaluate_loose(response, check)


def test_strict_and_loose_are_never_merged() -> None:
    task = _task()

    def check(response: str) -> bool:
        return response.startswith("ANSWER:")

    pair = grade_accepted_sample(task=task, response="preamble\nANSWER: done", check=check)
    assert set(pair.strict.score_components) == {"prompt_level_strict"}
    assert set(pair.loose.score_components) == {"prompt_level_loose"}
    assert set(pair.strict.score_components).isdisjoint(pair.loose.score_components)
    for side in (pair.strict, pair.loose):
        for forbidden in ("combined", "average", "merged", "overall"):
            assert forbidden not in side.score_components


# ---------------------------------------------------------------------------
# Gold never enters the request
# ---------------------------------------------------------------------------


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
        benchmark_id="ifeval",
        item_id="item-001",
        campaign_id="c1",
        endpoint_id="e1",
        repeat_id=1,
        prompt_artifact_hash=prompt_hash(PromptRef.from_request(request)),
        request=request,
        evaluation=EvaluationPayload(
            gold_answer=GOLD_CANARY,
            evaluator_id="instruction_following_eval",
            evaluator_revision=PINNED_IFEVAL_EVALUATOR_REVISION,
        ),
    )
    with pytest.raises(ValueError, match="would be sent to a provider"):
        dispatch_request(task)
    with pytest.raises(ValueError, match="would be sent to a provider"):
        grade_accepted_sample(task=task, response="blueberry", check=_contains("blueberry"))


# ---------------------------------------------------------------------------
# Malformed responses: format failure, still in the denominator
# ---------------------------------------------------------------------------


def test_empty_response_is_a_format_failure_in_the_denominator() -> None:
    task = _task()
    pair = grade_accepted_sample(task=task, response="   ", check=_contains("blueberry"))
    for side in (pair.strict, pair.loose):
        assert side.format == "fail"
        assert side.denominator_eligibility.format is True
        assert side.transport == "pass"
        assert side.evaluator == "pass"
    # Delivered but empty: incorrect (fails the instruction) and malformed.
    assert pair.strict.correctness == "fail"
    assert pair.strict.counts_toward_accuracy


# ---------------------------------------------------------------------------
# Transport failures are not incorrect answers; evaluator failures are separate
# ---------------------------------------------------------------------------


def test_transport_failure_is_unavailable_not_incorrect() -> None:
    task = _task()
    generation = _generation(task, None, DeliveryStatus.TRANSPORT_FAILED)
    pair = grade_generation(task=task, generation=generation, check=_contains("x"))
    for side in (pair.strict, pair.loose):
        assert side.correctness == "unavailable"
        assert side.transport == "fail"
        assert side.evaluator == "unavailable"
        assert not side.counts_toward_accuracy
        assert side.denominator_eligibility.correctness is False


def test_cancelled_and_unresolved_are_not_counted_as_incorrect() -> None:
    task = _task()
    cancelled = grade_generation(
        task=task,
        generation=_generation(task, None, DeliveryStatus.CANCELLED, attempt=1),
    )
    unresolved = grade_generation(
        task=task,
        generation=_generation(task, None, DeliveryStatus.UNRESOLVED, attempt=2),
    )
    assert cancelled.strict.transport == "invalid"
    assert unresolved.strict.transport == "unavailable"
    for pair in (cancelled, unresolved):
        assert pair.strict.correctness == "unavailable"
        assert not pair.strict.counts_toward_accuracy


def test_incorrect_answer_and_transport_failure_have_different_denominators() -> None:
    task = _task(prompt="Say blueberry.", gold=GOLD_CANARY)
    wrong = grade_generation(
        task=task,
        generation=_generation(task, "apple"),
        check=_contains("blueberry"),
    )
    failed = grade_generation(
        task=task,
        generation=_generation(task, None, DeliveryStatus.TRANSPORT_FAILED, attempt=2),
        check=_contains("blueberry"),
    )
    assert wrong.strict.correctness == "fail"
    assert wrong.strict.counts_toward_accuracy
    assert not failed.strict.counts_toward_accuracy
    summary = summarize_pairs([wrong, failed])
    assert summary.strict_eligible == 1
    assert summary.strict_correct == 0
    assert summary.strict_accuracy == 0.0


def test_a_grader_crash_is_an_evaluator_failure_not_an_incorrect_answer() -> None:
    def crashing(_response: str) -> bool:
        raise RuntimeError("official checker exploded")

    task = _task()
    pair = grade_accepted_sample(task=task, response="blueberry", check=crashing)
    assert pair.strict.evaluator == "fail"
    assert pair.strict.correctness == "unavailable"
    assert not pair.strict.counts_toward_accuracy


def test_an_accepted_response_without_a_check_is_unevaluated() -> None:
    task = _task()
    pair = grade_generation(task=task, generation=_generation(task, "blueberry"))
    assert pair.strict.evaluator == "fail"
    assert pair.strict.correctness == "unavailable"
    assert not pair.strict.counts_toward_accuracy


def test_sample_key_mismatch_refuses_to_grade() -> None:
    task = _task(item_id="item-001")
    other = _task(item_id="item-002")
    generation = _generation(other, "blueberry")
    with pytest.raises(ValueError, match="does not match"):
        grade_generation(task=task, generation=generation, check=_contains("blueberry"))


# ---------------------------------------------------------------------------
# Aggregation keeps strict/loose separate and null stays null
# ---------------------------------------------------------------------------


def test_summary_reports_strict_and_loose_separately() -> None:
    def check(response: str) -> bool:
        return response.startswith("ANSWER:")

    responses = ["ANSWER: yes", "preamble\nANSWER: yes", "nothing here"]
    pairs = [
        grade_accepted_sample(task=_task(item_id=f"item-{i:03d}"), response=r, check=check)
        for i, r in enumerate(responses)
    ]
    summary = summarize_pairs(pairs)
    assert (summary.strict_correct, summary.strict_eligible) == (1, 3)
    assert (summary.loose_correct, summary.loose_eligible) == (2, 3)
    assert summary.strict_accuracy == pytest.approx(1 / 3)
    assert summary.loose_accuracy == pytest.approx(2 / 3)


def test_unknown_rate_is_null_never_zero() -> None:
    empty = summarize_pairs([])
    assert empty.strict_accuracy is None
    assert empty.loose_accuracy is None
    assert empty.strict_accuracy != 0.0  # None is not a measured zero

    task = _task()
    failed_only = summarize_pairs(
        [
            grade_generation(
                task=task,
                generation=_generation(task, None, DeliveryStatus.TRANSPORT_FAILED),
            )
        ]
    )
    assert failed_only.strict_eligible == 0
    assert failed_only.strict_accuracy is None
    dumped = json.loads(failed_only.model_dump_json())
    assert dumped["strict_accuracy"] is None
    assert dumped["loose_accuracy"] is None


def test_pair_survives_a_result_schema_round_trip() -> None:
    task = _task()
    pair = grade_accepted_sample(task=task, response="blueberry", check=_contains("blueberry"))
    revived = IfEvalGradePair.model_validate(pair.model_dump(mode="json"))
    assert revived == pair
    sample: SampleKey = task.sample_key
    assert revived.strict.sample_key == sample
    assert revived.loose.sample_key == sample
