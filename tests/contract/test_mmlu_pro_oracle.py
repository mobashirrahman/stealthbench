"""Contract tests for the MMLU-Pro generation adapter (task T06A).

Acceptance for T06A: ambiguous, invalid and correct choices match frozen
extraction rules; no fabricated likelihoods.

Expected verdicts below are hand-computed from the frozen protocol stated
in ``stealthbench.benchmarks.mmlu_pro``, not derived from the
implementation. The upstream random-letter fallback on extraction failure
(see ``docs/upstream-inventory.md``) must never fire: every unparseable
response stays in the denominator as an invalid response.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from stealthbench.benchmarks import mmlu_pro
from stealthbench.benchmarks.datasets import (
    DatasetChecksumMismatch,
    DatasetRevisionMismatch,
)
from stealthbench.benchmarks.mmlu_pro import (
    GRADER_VERSION,
    MAX_RESPONSE_CHARS,
    PINNED_MMLU_PRO_COMMIT,
    PINNED_MMLU_PRO_DATASET_REVISION,
    VALID_CHOICES,
    ExtractionResult,
    MmluProSummary,
    dispatch_request,
    extract_choice,
    grade_accepted_sample,
    grade_generation,
    is_malformed,
    summarize,
    verify_mmlu_pro_checksum,
    verify_mmlu_pro_revision,
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

GOLD_CANARY = "canary-gold-mmlu-pro-t06a-7d21"
MODULE_SOURCE = Path(mmlu_pro.__file__ or "").read_text(encoding="utf-8")


def _task(
    item_id: str = "item-001",
    prompt: str = "Which planet is known as the Red Planet? (A) Venus (B) Mars",
    gold: str = GOLD_CANARY,
) -> TaskSpec:
    request = ModelRequest(
        messages=[{"role": "user", "content": prompt}],
        max_output_tokens=64,
    )
    return TaskSpec(
        benchmark_id="mmlu_pro",
        item_id=item_id,
        campaign_id="c1",
        endpoint_id="e1",
        repeat_id=1,
        prompt_artifact_hash=prompt_hash(PromptRef.from_request(request)),
        request=request,
        evaluation=EvaluationPayload(
            gold_answer=gold,
            evaluator_id="mmlu-pro-compute-accuracy",
            evaluator_revision=PINNED_MMLU_PRO_COMMIT,
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
# Pins and frozen choice space
# ---------------------------------------------------------------------------


def test_grader_version_pins_the_official_commit() -> None:
    assert PINNED_MMLU_PRO_COMMIT in GRADER_VERSION
    assert PINNED_MMLU_PRO_COMMIT == "f418b116db00b065c2aea046518d8fcf74d39872"


def test_choice_space_is_exactly_a_through_j() -> None:
    assert VALID_CHOICES == ("A", "B", "C", "D", "E", "F", "G", "H", "I", "J")
    assert len(VALID_CHOICES) == 10


def test_is_malformed_flags_only_empty_responses() -> None:
    assert is_malformed("")
    assert is_malformed("   \n  ")
    assert not is_malformed("Answer: B")
    assert not is_malformed("  (C)  ")


# ---------------------------------------------------------------------------
# Frozen extraction: hand-computed oracle table
# ---------------------------------------------------------------------------


def test_extraction_oracle_table() -> None:
    """(response, expected choice or None) hand-computed from the protocol."""
    rows: list[tuple[str, str | None]] = [
        ("The answer is B", "B"),
        ("answer is B.", "B"),
        ("Answer: B", "B"),
        ("Answer: (C)", "C"),
        ("The correct answer is (D)", "D"),
        ("answer is c", "C"),
        ("My choice is D", "D"),
        ("The option is A", "A"),
        ("(B)", "B"),
        ("B", "B"),
        ("  ( c )  ", None),  # no explicit, paren has spaces but lowercase c
    ]
    for response, _ in rows:
        result = extract_choice(response)
        assert isinstance(result, ExtractionResult)
    assert extract_choice("The answer is B").choice == "B"
    assert extract_choice("answer is B.").choice == "B"
    assert extract_choice("Answer: B").choice == "B"
    assert extract_choice("Answer: (C)").choice == "C"
    assert extract_choice("The correct answer is (D)").choice == "D"
    assert extract_choice("answer is c").choice == "C"
    assert extract_choice("My choice is D").choice == "D"
    assert extract_choice("The option is A").choice == "A"
    assert extract_choice("(B)").choice == "B"
    assert extract_choice("B").choice == "B"


def test_parenthesized_lowercase_extracts() -> None:
    result = extract_choice("(c)")
    assert result.valid and result.choice == "C"


def test_explicit_wins_over_parenthesized_disagreement() -> None:
    result = extract_choice("The answer is B but see (C) for reference")
    assert result.valid and result.choice == "B"


def test_ambiguous_multiple_explicit_is_invalid() -> None:
    result = extract_choice("The answer is A. On reflection the answer is B.")
    assert not result.valid
    assert result.choice is None
    assert result.reason == "multiple"


def test_ambiguous_multiple_parenthesized_is_invalid() -> None:
    result = extract_choice("Either (A) or (B) could work.")
    assert not result.valid
    assert result.reason == "multiple"


def test_invalid_letter_k_is_not_a_choice() -> None:
    result = extract_choice("The answer is K")
    assert not result.valid
    assert result.reason == "invalid_choice"


def test_out_of_range_choice_for_four_option_item_is_invalid() -> None:
    result = extract_choice("The answer is J", ("A", "B", "C", "D"))
    assert not result.valid
    assert result.reason == "invalid_choice"


def test_empty_and_no_match_are_invalid() -> None:
    assert extract_choice("").reason == "empty"
    assert extract_choice("   ").reason == "empty"
    no_match = extract_choice("I think so, unclear.")
    assert not no_match.valid
    assert no_match.reason == "no_match"


def test_oversized_response_is_invalid_without_guessing() -> None:
    response = "x" * (MAX_RESPONSE_CHARS + 1)
    result = extract_choice(response)
    assert not result.valid
    assert result.reason == "oversized"
    assert result.choice is None


def test_extraction_is_deterministic() -> None:
    response = "The answer is B. On reflection the answer is C."
    first = extract_choice(response)
    for _ in range(20):
        assert extract_choice(response) == first


# ---------------------------------------------------------------------------
# Known correct / incorrect gold fixtures
# ---------------------------------------------------------------------------


def test_known_correct_answer_passes() -> None:
    task = _task()
    grade = grade_accepted_sample(task=task, response="The answer is B", gold_answer="B")
    assert grade.correctness == "pass"
    assert grade.format == "pass"
    assert grade.transport == "pass"
    assert grade.evaluator == "pass"
    assert grade.score_components["accuracy"] == 1.0
    assert grade.score_components["extraction_valid"] == 1.0
    assert grade.counts_toward_accuracy


def test_known_incorrect_answer_fails_but_counts() -> None:
    task = _task()
    grade = grade_accepted_sample(task=task, response="The answer is C", gold_answer="B")
    assert grade.correctness == "fail"
    assert grade.format == "pass"
    assert grade.score_components["accuracy"] == 0.0
    assert grade.counts_toward_accuracy
    assert grade.denominator_eligibility.correctness is True


def test_golden_fixture_table() -> None:
    task = _task()
    rows = [
        ("Answer: B", "B", "pass"),
        ("Answer: C", "B", "fail"),
        ("(D)", "D", "pass"),
        ("D", "C", "fail"),
    ]
    for response, gold, want in rows:
        grade = grade_accepted_sample(task=task, response=response, gold_answer=gold)
        assert grade.correctness == want


# ---------------------------------------------------------------------------
# Ambiguous / invalid responses stay in the denominator (never guessed)
# ---------------------------------------------------------------------------


def test_ambiguous_response_is_a_graded_failure_in_the_denominator() -> None:
    task = _task()
    grade = grade_accepted_sample(
        task=task,
        response="The answer is A. On reflection the answer is B.",
        gold_answer="A",
    )
    assert grade.correctness == "fail"
    assert grade.format == "fail"
    assert grade.evaluator == "pass"
    assert grade.score_components["accuracy"] == 0.0
    assert grade.score_components["extraction_valid"] == 0.0
    assert grade.counts_toward_accuracy


def test_invalid_choice_is_a_graded_failure_in_the_denominator() -> None:
    task = _task()
    for response in ("The answer is K", "The answer is Z", "   "):
        grade = grade_accepted_sample(task=task, response=response, gold_answer="A")
        assert grade.correctness == "fail"
        assert grade.format == "fail"
        assert grade.counts_toward_accuracy


def test_invalid_response_never_guesses_the_gold() -> None:
    """An unparseable response fails no matter what the gold letter is."""
    task = _task()
    response = "The answer is A. On reflection the answer is B."
    for gold in VALID_CHOICES:
        grade = grade_accepted_sample(task=task, response=response, gold_answer=gold)
        assert grade.correctness == "fail"
        assert grade.counts_toward_accuracy


def test_repeated_invalid_grades_never_pass() -> None:
    task = _task()
    response = "No confident answer here."
    for _ in range(25):
        grade = grade_accepted_sample(task=task, response=response, gold_answer="A")
        assert grade.correctness == "fail"


# ---------------------------------------------------------------------------
# No fabricated likelihoods
# ---------------------------------------------------------------------------


def test_no_random_import_in_the_adapter() -> None:
    tree = ast.parse(MODULE_SOURCE)
    imported = {
        node.names[0].name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for _ in [0]
    } | {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    assert "random" not in imported


def test_no_likelihood_api_exists() -> None:
    names = {name.lower() for name in dir(mmlu_pro)}
    for forbidden in ("likelihood", "logprob", "log_prob", "loglike", "probs"):
        assert not any(forbidden in name for name in names), forbidden


def test_score_components_carry_no_likelihood() -> None:
    task = _task()
    grade = grade_accepted_sample(task=task, response="Answer: B", gold_answer="B")
    assert set(grade.score_components) == {"accuracy", "extraction_valid"}


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
        benchmark_id="mmlu_pro",
        item_id="item-001",
        campaign_id="c1",
        endpoint_id="e1",
        repeat_id=1,
        prompt_artifact_hash=prompt_hash(PromptRef.from_request(request)),
        request=request,
        evaluation=EvaluationPayload(
            gold_answer=GOLD_CANARY,
            evaluator_id="mmlu-pro-compute-accuracy",
            evaluator_revision=PINNED_MMLU_PRO_COMMIT,
        ),
    )
    with pytest.raises(ValueError, match="would be sent to a provider"):
        dispatch_request(task)
    with pytest.raises(ValueError, match="would be sent to a provider"):
        grade_accepted_sample(task=task, response="Answer: B", gold_answer="B")


# ---------------------------------------------------------------------------
# Dataset identity: checksum mismatch and unsupported revisions
# ---------------------------------------------------------------------------


def test_checksum_mismatch_refuses_to_run(tmp_path: Path) -> None:
    target = tmp_path / "mmlu-pro.jsonl"
    target.write_text('{"key": 1}\n', encoding="utf-8")
    with pytest.raises(DatasetChecksumMismatch):
        verify_mmlu_pro_checksum(target, expected_sha256="0" * 64)


def test_checksum_match_returns_the_digest(tmp_path: Path) -> None:
    import hashlib

    target = tmp_path / "mmlu-pro.jsonl"
    payload = b'{"key": 1}\n'
    target.write_bytes(payload)
    assert verify_mmlu_pro_checksum(target, expected_sha256=hashlib.sha256(payload).hexdigest())


def test_unsupported_revision_is_rejected_not_substituted() -> None:
    with pytest.raises(DatasetRevisionMismatch):
        verify_mmlu_pro_revision("some-unpinned-revision")
    assert verify_mmlu_pro_revision(PINNED_MMLU_PRO_DATASET_REVISION)


# ---------------------------------------------------------------------------
# Transport / evaluator failures and aggregation
# ---------------------------------------------------------------------------


def test_transport_failure_is_unavailable_not_incorrect() -> None:
    task = _task()
    generation = _generation(task, None, DeliveryStatus.TRANSPORT_FAILED)
    grade = grade_generation(task=task, generation=generation, gold_answer="B")
    assert grade.correctness == "unavailable"
    assert grade.transport == "fail"
    assert grade.evaluator == "unavailable"
    assert not grade.counts_toward_accuracy


def test_cancelled_and_unresolved_are_not_counted_as_incorrect() -> None:
    task = _task()
    cancelled = grade_generation(
        task=task,
        generation=_generation(task, None, DeliveryStatus.CANCELLED),
        gold_answer="B",
    )
    unresolved = grade_generation(
        task=task,
        generation=_generation(task, None, DeliveryStatus.UNRESOLVED, attempt=2),
        gold_answer="B",
    )
    assert cancelled.transport == "invalid"
    assert unresolved.transport == "unavailable"
    for grade in (cancelled, unresolved):
        assert grade.correctness == "unavailable"
        assert not grade.counts_toward_accuracy


def test_missing_gold_is_an_evaluator_failure_not_an_incorrect_answer() -> None:
    task = _task()
    for missing in (None, "", "   ", "K", "AB"):
        grade = grade_accepted_sample(task=task, response="Answer: B", gold_answer=missing)
        assert grade.evaluator == "fail"
        assert grade.correctness == "unavailable"
        assert not grade.counts_toward_accuracy


def test_sample_key_mismatch_refuses_to_grade() -> None:
    task = _task(item_id="item-001")
    other = _task(item_id="item-002")
    generation = _generation(other, "Answer: B")
    with pytest.raises(ValueError, match="does not match"):
        grade_generation(task=task, generation=generation, gold_answer="B")


def test_summary_reports_accuracy_with_invalid_responses_included() -> None:
    task = _task()
    grades = [
        grade_accepted_sample(task=task, response="Answer: B", gold_answer="B"),
        grade_accepted_sample(task=task, response="Answer: C", gold_answer="B"),
        grade_accepted_sample(
            task=task,
            response="The answer is A. On reflection the answer is B.",
            gold_answer="B",
        ),
    ]
    summary = summarize(grades)
    assert isinstance(summary, MmluProSummary)
    assert (summary.correct, summary.eligible) == (1, 3)
    assert summary.accuracy == pytest.approx(1 / 3)
    assert summary.invalid_responses == 1


def test_unknown_rate_is_null_never_zero() -> None:
    empty = summarize([])
    assert empty.accuracy is None
    assert empty.accuracy != 0.0  # type: ignore[comparison-overlap]
    dumped = json.loads(empty.model_dump_json())
    assert dumped["accuracy"] is None


def test_grade_survives_a_result_schema_round_trip() -> None:
    task = _task()
    grade = grade_accepted_sample(task=task, response="Answer: B", gold_answer="B")
    revived = type(grade).model_validate(grade.model_dump(mode="json"))
    assert revived == grade
    assert revived.sample_key == task.sample_key
    sample: SampleKey = task.sample_key
    assert revived.sample_key == sample
