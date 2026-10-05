"""Contract tests for the MATH-500 scoring adapter (task T06B).

Acceptance for T06B: numerical and symbolic cases match pinned official
oracles; parsing is bounded and oversized inputs fail safely.

Expected verdicts are hand-computed from the frozen protocol in
``stealthbench.benchmarks.math500``. The non-deterministic auxiliary
metric from the reference harness is deliberately absent and must stay
absent.
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pytest

from stealthbench.benchmarks import math500
from stealthbench.benchmarks.datasets import (
    DatasetChecksumMismatch,
    DatasetRevisionMismatch,
)
from stealthbench.benchmarks.math500 import (
    GRADER_VERSION,
    MAX_EXPRESSION_CHARS,
    MAX_RESPONSE_CHARS,
    NUMERIC_TOLERANCE,
    PARSER_TIMEOUT_SECONDS,
    PINNED_MATH500_DATASET_REVISION,
    Math500Summary,
    MathParseError,
    OversizedInputError,
    ParserTimeoutError,
    dispatch_request,
    extract_final_answer,
    grade_accepted_sample,
    grade_generation,
    is_malformed,
    normalize_answer,
    numeric_equals,
    summarize,
    symbolic_equals,
    verify_math500_checksum,
    verify_math500_revision,
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

GOLD_CANARY = "canary-gold-math500-t06b-4e77"
MODULE_SOURCE = Path(math500.__file__ or "").read_text(encoding="utf-8")


def _task(
    item_id: str = "item-001",
    prompt: str = "Compute 6 * 7.",
    gold: str = GOLD_CANARY,
) -> TaskSpec:
    request = ModelRequest(
        messages=[{"role": "user", "content": prompt}],
        max_output_tokens=64,
    )
    return TaskSpec(
        benchmark_id="math500",
        item_id=item_id,
        campaign_id="c1",
        endpoint_id="e1",
        repeat_id=1,
        prompt_artifact_hash=prompt_hash(PromptRef.from_request(request)),
        request=request,
        evaluation=EvaluationPayload(
            gold_answer=gold,
            evaluator_id="math_500",
            evaluator_revision=PINNED_MATH500_DATASET_REVISION,
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
# Pins, tolerance and bounds
# ---------------------------------------------------------------------------


def test_grader_version_pins_the_data_revision() -> None:
    assert PINNED_MATH500_DATASET_REVISION in GRADER_VERSION
    assert PINNED_MATH500_DATASET_REVISION == "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be"


def test_numeric_tolerance_is_positive_and_recorded() -> None:
    assert NUMERIC_TOLERANCE > 0
    task = _task()
    grade = grade_accepted_sample(task=task, response="42", gold_answer="42")
    assert grade.score_components["numeric_tolerance"] == pytest.approx(NUMERIC_TOLERANCE)


def test_parser_bounds_are_positive() -> None:
    assert PARSER_TIMEOUT_SECONDS > 0
    assert MAX_EXPRESSION_CHARS > 0
    assert MAX_RESPONSE_CHARS > 0


def test_is_malformed_flags_only_empty_responses() -> None:
    assert is_malformed("")
    assert is_malformed("   \n  ")
    assert not is_malformed("42")
    assert not is_malformed("  \\boxed{42}  ")


# ---------------------------------------------------------------------------
# Normalization oracle (hand-computed)
# ---------------------------------------------------------------------------


def test_normalization_oracle_table() -> None:
    assert normalize_answer("  42  ") == "42"
    assert normalize_answer("$42$") == "42"
    assert normalize_answer("\\(42\\)") == "42"
    assert normalize_answer("\\[42\\]") == "42"
    assert normalize_answer("\\text{42}") == "42"
    assert normalize_answer("\\mathrm{x}+1") == "x+1"
    assert normalize_answer("\\left(42\\right)") == "(42)"
    assert normalize_answer("1,000") == "1000"
    assert normalize_answer("a \\, b") == "ab"
    assert normalize_answer("42.") == "42"
    assert normalize_answer("x + 1") == "x+1"


# ---------------------------------------------------------------------------
# Extraction oracle (hand-computed)
# ---------------------------------------------------------------------------


def test_boxed_extraction_wins_and_takes_the_last_box() -> None:
    assert extract_final_answer("Work. \\boxed{42}") == "42"
    assert extract_final_answer("First \\boxed{1} then \\boxed{3}") == "3"
    assert extract_final_answer("\\boxed{\\frac{1}{2}}") == "\\frac{1}{2}"


def test_dollar_and_last_line_fallbacks() -> None:
    assert extract_final_answer("So $x+1$ is it") == "x+1"
    assert extract_final_answer("line one\nline two\n  42  ") == "42"
    assert extract_final_answer("42") == "42"


def test_oversized_response_raises_without_scanning() -> None:
    with pytest.raises(OversizedInputError):
        extract_final_answer("x" * (MAX_RESPONSE_CHARS + 1))


# ---------------------------------------------------------------------------
# Known correct / incorrect fixtures
# ---------------------------------------------------------------------------


def test_known_correct_exact_match_passes() -> None:
    task = _task()
    grade = grade_accepted_sample(task=task, response="42", gold_answer="42")
    assert grade.correctness == "pass"
    assert grade.format == "pass"
    assert grade.evaluator == "pass"
    assert grade.score_components["math_correct"] == 1.0
    assert grade.counts_toward_accuracy


def test_boxed_response_matches_plain_gold() -> None:
    task = _task()
    grade = grade_accepted_sample(task=task, response="Working...\n\\boxed{42}", gold_answer="42")
    assert grade.correctness == "pass"


def test_known_incorrect_answer_fails_but_counts() -> None:
    task = _task()
    grade = grade_accepted_sample(task=task, response="43", gold_answer="42")
    assert grade.correctness == "fail"
    assert grade.format == "pass"
    assert grade.score_components["math_correct"] == 0.0
    assert grade.counts_toward_accuracy


def test_golden_fixture_table() -> None:
    task = _task()
    rows = [
        ("42", "42", "pass"),
        ("\\boxed{42}", "42", "pass"),
        ("$42$", "42", "pass"),
        ("43", "42", "fail"),
        ("0.5", "\\frac{1}{2}", "pass"),
        ("1/2", "0.5", "pass"),
    ]
    for response, gold, want in rows:
        grade = grade_accepted_sample(task=task, response=response, gold_answer=gold)
        assert grade.correctness == want, (response, gold)


# ---------------------------------------------------------------------------
# Official numerical / symbolic edge cases (hand-computed)
# ---------------------------------------------------------------------------


def test_fraction_decimal_equivalence() -> None:
    assert numeric_equals("\\frac{1}{2}", "0.5")
    assert numeric_equals("1/2", "0.5")
    assert not numeric_equals("1/3", "0.5")


def test_numeric_tolerance_boundary() -> None:
    assert numeric_equals("0.3333333", "1/3")
    assert not numeric_equals("0.333", "1/3")


def test_thousands_comma_percent_and_sqrt_edges() -> None:
    assert numeric_equals("1,000", "1000")
    assert numeric_equals("50%", "0.5")
    assert numeric_equals("\\sqrt{4}", "2")
    assert not numeric_equals("\\sqrt{5}", "2")


def test_symbolic_methods_are_reported() -> None:
    equal, method = symbolic_equals("42", "42")
    assert (equal, method) == (True, "exact")
    equal, method = symbolic_equals("0.5", "\\frac{1}{2}")
    assert equal and method == "numeric"
    equal, method = symbolic_equals("43", "42")
    assert not equal and method in ("none", "symbolic", "numeric")


def test_empty_expression_raises_parse_error() -> None:
    with pytest.raises(MathParseError):
        symbolic_equals("   ", "42")


# ---------------------------------------------------------------------------
# Bounded parsing: timeouts and oversized inputs fail safely
# ---------------------------------------------------------------------------


def test_zero_budget_always_times_out() -> None:
    with pytest.raises(ParserTimeoutError):
        symbolic_equals("42", "42", timeout_seconds=0.0)
    with pytest.raises(ParserTimeoutError):
        symbolic_equals("42", "42", timeout_seconds=-1.0)


def test_parser_timeout_is_a_graded_failure_in_the_denominator() -> None:
    task = _task()
    grade = grade_accepted_sample(task=task, response="42", gold_answer="42", timeout_seconds=0.0)
    assert grade.correctness == "fail"
    assert grade.evaluator == "pass"
    assert grade.score_components["parser_timed_out"] == 1.0
    assert grade.score_components["math_correct"] == 0.0
    assert grade.counts_toward_accuracy


def test_oversized_expression_raises() -> None:
    big = "1" * (MAX_EXPRESSION_CHARS + 1)
    with pytest.raises(OversizedInputError):
        symbolic_equals(big, "42")


def test_oversized_input_is_a_graded_failure_in_the_denominator() -> None:
    task = _task()
    response = "\\boxed{" + "1" * (MAX_EXPRESSION_CHARS + 1) + "}"
    assert len(response) < MAX_RESPONSE_CHARS  # extraction runs; parsing refuses
    grade = grade_accepted_sample(task=task, response=response, gold_answer="42")
    assert grade.correctness == "fail"
    assert grade.evaluator == "pass"
    assert grade.score_components["oversized"] == 1.0
    assert grade.counts_toward_accuracy


def test_oversized_response_is_a_graded_failure_not_a_crash() -> None:
    task = _task()
    grade = grade_accepted_sample(
        task=task, response="x" * (MAX_RESPONSE_CHARS + 1), gold_answer="42"
    )
    assert grade.correctness == "fail"
    assert grade.format == "fail"
    assert grade.counts_toward_accuracy


def test_parser_uses_no_subprocess() -> None:
    tree = ast.parse(MODULE_SOURCE)
    imported = {
        node.names[0].name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for _ in [0]
    } | {
        (node.module or "").split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for _ in [0]
    }
    assert "subprocess" not in imported
    assert "multiprocessing" not in imported
    assert "concurrent" not in imported
    assert "Popen" not in MODULE_SOURCE


# ---------------------------------------------------------------------------
# Auxiliary graded metric is absent
# ---------------------------------------------------------------------------


def test_no_auxiliary_grading_api_exists() -> None:
    names = {name.lower() for name in dir(math500)}
    assert not any("judge" in name for name in names)
    assert not any("model_graded" in name for name in names)
    assert "llm_judge" not in names


def test_score_components_carry_no_auxiliary_metric() -> None:
    task = _task()
    grade = grade_accepted_sample(task=task, response="42", gold_answer="42")
    assert set(grade.score_components) == {
        "math_correct",
        "numeric_tolerance",
        "parser_timed_out",
        "oversized",
    }


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
        benchmark_id="math500",
        item_id="item-001",
        campaign_id="c1",
        endpoint_id="e1",
        repeat_id=1,
        prompt_artifact_hash=prompt_hash(PromptRef.from_request(request)),
        request=request,
        evaluation=EvaluationPayload(
            gold_answer=GOLD_CANARY,
            evaluator_id="math_500",
            evaluator_revision=PINNED_MATH500_DATASET_REVISION,
        ),
    )
    with pytest.raises(ValueError, match="would be sent to a provider"):
        dispatch_request(task)
    with pytest.raises(ValueError, match="would be sent to a provider"):
        grade_accepted_sample(task=task, response="42", gold_answer="42")


# ---------------------------------------------------------------------------
# Dataset identity: checksum mismatch and unsupported revisions
# ---------------------------------------------------------------------------


def test_checksum_mismatch_refuses_to_run(tmp_path: Path) -> None:
    target = tmp_path / "math500.jsonl"
    target.write_text('{"problem": "1+1"}\n', encoding="utf-8")
    with pytest.raises(DatasetChecksumMismatch):
        verify_math500_checksum(target, expected_sha256="0" * 64)


def test_checksum_match_returns_the_digest(tmp_path: Path) -> None:
    target = tmp_path / "math500.jsonl"
    payload = b'{"problem": "1+1"}\n'
    target.write_bytes(payload)
    assert (
        verify_math500_checksum(target, expected_sha256=hashlib.sha256(payload).hexdigest())
        == hashlib.sha256(payload).hexdigest()
    )


def test_unsupported_revision_is_rejected_not_substituted() -> None:
    with pytest.raises(DatasetRevisionMismatch):
        verify_math500_revision("some-unpinned-revision")
    assert verify_math500_revision(PINNED_MATH500_DATASET_REVISION)


# ---------------------------------------------------------------------------
# Transport / evaluator failures and aggregation
# ---------------------------------------------------------------------------


def test_transport_failure_is_unavailable_not_incorrect() -> None:
    task = _task()
    generation = _generation(task, None, DeliveryStatus.TRANSPORT_FAILED)
    grade = grade_generation(task=task, generation=generation, gold_answer="42")
    assert grade.correctness == "unavailable"
    assert grade.transport == "fail"
    assert grade.evaluator == "unavailable"
    assert not grade.counts_toward_accuracy


def test_cancelled_and_unresolved_are_not_counted_as_incorrect() -> None:
    task = _task()
    cancelled = grade_generation(
        task=task,
        generation=_generation(task, None, DeliveryStatus.CANCELLED),
        gold_answer="42",
    )
    unresolved = grade_generation(
        task=task,
        generation=_generation(task, None, DeliveryStatus.UNRESOLVED, attempt=2),
        gold_answer="42",
    )
    assert cancelled.transport == "invalid"
    assert unresolved.transport == "unavailable"
    for grade in (cancelled, unresolved):
        assert grade.correctness == "unavailable"
        assert not grade.counts_toward_accuracy


def test_missing_gold_is_an_evaluator_failure_not_an_incorrect_answer() -> None:
    task = _task()
    for missing in (None, "", "   "):
        grade = grade_accepted_sample(task=task, response="42", gold_answer=missing)
        assert grade.evaluator == "fail"
        assert grade.correctness == "unavailable"
        assert not grade.counts_toward_accuracy


def test_sample_key_mismatch_refuses_to_grade() -> None:
    task = _task(item_id="item-001")
    other = _task(item_id="item-002")
    generation = _generation(other, "42")
    with pytest.raises(ValueError, match="does not match"):
        grade_generation(task=task, generation=generation, gold_answer="42")


def test_summary_reports_accuracy_with_failures_included() -> None:
    task = _task()
    grades = [
        grade_accepted_sample(task=task, response="42", gold_answer="42"),
        grade_accepted_sample(task=task, response="43", gold_answer="42"),
        grade_accepted_sample(task=task, response="42", gold_answer="42", timeout_seconds=0.0),
    ]
    summary = summarize(grades)
    assert isinstance(summary, Math500Summary)
    assert (summary.correct, summary.eligible) == (1, 3)
    assert summary.accuracy == pytest.approx(1 / 3)
    assert summary.timed_out == 1
    assert summary.oversized == 0


def test_unknown_rate_is_null_never_zero() -> None:
    empty = summarize([])
    assert empty.accuracy is None
    assert empty.accuracy != 0.0  # type: ignore[comparison-overlap]
    dumped = json.loads(empty.model_dump_json())
    assert dumped["accuracy"] is None


def test_grade_survives_a_result_schema_round_trip() -> None:
    task = _task()
    grade = grade_accepted_sample(task=task, response="42", gold_answer="42")
    revived = type(grade).model_validate(grade.model_dump(mode="json"))
    assert revived == grade
    assert revived.sample_key == task.sample_key
    sample: SampleKey = task.sample_key
    assert revived.sample_key == sample
