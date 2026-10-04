"""Contract tests for the RULER generation and grading adapter (task T08B).

Acceptance for T08B: seeds and task versions reproduce cases; truncation
or unsupported lengths cannot masquerade as intended context.

Expected verdicts below are hand-computed from the frozen protocol stated
in ``stealthbench.benchmarks.ruler``, not derived from the implementation.
Production wiring generates prompts with the pinned seed and grades with
the official scorer; the adapter only binds settings, checks truncation,
and maps verdicts onto ``GradeResult``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from stealthbench.benchmarks import ruler
from stealthbench.benchmarks.datasets import DatasetRevisionMismatch
from stealthbench.benchmarks.ruler import (
    GENERATION_SEED,
    GENERATION_TEMPERATURE,
    GENERATION_TOP_P,
    GRADER_VERSION,
    NEMO_SKILLS_BRANCH,
    PINNED_RULER_BRANCH,
    PINNED_RULER_COMMIT,
    PROMPTS_IDENTICAL_ACROSS_MODELS,
    RULER_SAMPLES_PER_TASK,
    RULER_TASK_COUNT,
    RULER_TOTAL_ITEMS,
    RULER_V1_TASKS,
    SUPPORTED_LENGTHS,
    TOKENIZATION_CAVEAT,
    RulerSummary,
    UnknownTaskError,
    UnsupportedLengthError,
    build_task_settings,
    deterministic_case_id,
    dispatch_request,
    generation_settings,
    grade_accepted_sample,
    grade_generation,
    is_supported_length,
    is_supported_task,
    report_length,
    score_response,
    summarize,
    summarize_cell,
    verify_ruler_branch,
    verify_ruler_revision,
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

GOLD_CANARY = "canary-gold-ruler-t08b-8a44"


def _task(
    item_id: str = "niah_single_1-000",
    prompt: str = "Find the needle in the haystack.",
    gold: str = GOLD_CANARY,
) -> TaskSpec:
    request = ModelRequest(
        messages=[{"role": "user", "content": prompt}],
        max_output_tokens=128,
    )
    return TaskSpec(
        benchmark_id="ruler",
        item_id=item_id,
        campaign_id="c1",
        endpoint_id="e1",
        repeat_id=1,
        prompt_artifact_hash=prompt_hash(PromptRef.from_request(request)),
        request=request,
        evaluation=EvaluationPayload(
            gold_answer=gold,
            evaluator_id="ruler-score",
            evaluator_revision=PINNED_RULER_COMMIT,
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
# Pins: branch, commit, pipeline, shape, greedy sampling
# ---------------------------------------------------------------------------


def test_grader_version_pins_the_official_commit() -> None:
    assert PINNED_RULER_COMMIT == "e8bbff677ca2c239640dc90f93310dcf32408c93"
    assert PINNED_RULER_COMMIT in GRADER_VERSION
    assert GRADER_VERSION.startswith("ruler-score@")


def test_branch_pin_is_rulerv1_ns() -> None:
    assert PINNED_RULER_BRANCH == "rulerv1-ns"
    assert verify_ruler_branch("rulerv1-ns") == "rulerv1-ns"
    with pytest.raises(ruler.RulerError):
        verify_ruler_branch("main")
    with pytest.raises(ruler.RulerError):
        verify_ruler_branch("rulerv2-ns")


def test_nemo_skills_branch_note_is_recorded() -> None:
    assert NEMO_SKILLS_BRANCH == "chsieh/ruler-remove-prefix"
    assert ruler.NEMO_SKILLS_REPO == "https://github.com/NVIDIA-NeMo/Skills"


def test_v1_task_set_has_thirteen_pinned_tasks() -> None:
    assert RULER_TASK_COUNT == 13
    assert len(RULER_V1_TASKS) == 13
    assert len(set(RULER_V1_TASKS)) == 13
    for required in ("vt", "cwe", "fwe", "qa_1", "niah_single_1"):
        assert required in RULER_V1_TASKS


def test_dataset_shape_matches_the_inventory() -> None:
    assert RULER_SAMPLES_PER_TASK == 100
    assert RULER_TOTAL_ITEMS == 1300
    assert RULER_TOTAL_ITEMS == RULER_TASK_COUNT * RULER_SAMPLES_PER_TASK


def test_supported_lengths_are_the_frozen_six() -> None:
    assert SUPPORTED_LENGTHS == (4096, 8192, 16384, 32768, 65536, 131072)
    assert is_supported_length(4096)
    assert is_supported_length(131072)
    assert not is_supported_length(2048)
    assert not is_supported_length(200000)


def test_generation_is_greedy_with_seed_42_recorded() -> None:
    assert GENERATION_SEED == 42
    assert GENERATION_TEMPERATURE == 0.0
    assert GENERATION_TOP_P == 1.0
    settings = generation_settings()
    assert settings == {"temperature": 0.0, "top_p": 1.0, "seed": 42}


def test_per_task_settings_carry_seed_and_greedy_values() -> None:
    settings = build_task_settings("niah_single_1", 4096)
    assert settings.ruler_task == "niah_single_1"
    assert settings.requested_length == 4096
    assert settings.seed == 42
    assert settings.temperature == 0.0
    assert settings.top_p == 1.0
    assert settings.samples_per_task == 100


def test_per_task_settings_reject_unpinned_inputs() -> None:
    with pytest.raises(UnknownTaskError):
        build_task_settings("ruler_v2_task", 4096)
    with pytest.raises(UnsupportedLengthError):
        build_task_settings("niah_single_1", 2048)


def test_is_supported_task_oracle() -> None:
    assert is_supported_task("vt")
    assert is_supported_task("cwe")
    assert not is_supported_task("unknown_task_xyz")


def test_tokenization_caveat_is_explicit() -> None:
    assert PROMPTS_IDENTICAL_ACROSS_MODELS is False
    assert "NOT identical strings" in TOKENIZATION_CAVEAT
    assert "per-model tokenization" in TOKENIZATION_CAVEAT


# ---------------------------------------------------------------------------
# Seed reproducibility
# ---------------------------------------------------------------------------


def test_deterministic_case_ids_reproduce_with_the_seed() -> None:
    first = deterministic_case_id("niah_single_1", 4096, 0, seed=42)
    for _ in range(10):
        assert deterministic_case_id("niah_single_1", 4096, 0, seed=42) == first
    assert len(first) == 64


def test_different_seeds_give_different_case_ids() -> None:
    base = deterministic_case_id("niah_single_1", 4096, 0, seed=42)
    other = deterministic_case_id("niah_single_1", 4096, 0, seed=43)
    assert base != other


def test_case_ids_vary_by_task_length_and_index() -> None:
    ids = {
        deterministic_case_id("niah_single_1", 4096, 0),
        deterministic_case_id("niah_single_1", 4096, 1),
        deterministic_case_id("niah_single_1", 8192, 0),
        deterministic_case_id("vt", 4096, 0),
    }
    assert len(ids) == 4


def test_negative_sample_index_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        deterministic_case_id("niah_single_1", 4096, -1)


# ---------------------------------------------------------------------------
# Retrieval / aggregation oracles
# ---------------------------------------------------------------------------


def test_correct_retrieval_passes() -> None:
    task = _task()
    grade = grade_accepted_sample(
        task=task,
        response="needle-42",
        expected="needle-42",
        ruler_task="niah_single_1",
        intended_length=4096,
    )
    assert grade.correctness == "pass"
    assert grade.counts_toward_accuracy
    assert grade.score_components["ruler_correct"] == 1.0
    assert grade.score_components["truncated"] == 0.0


def test_incorrect_retrieval_fails_but_counts() -> None:
    task = _task()
    grade = grade_accepted_sample(
        task=task,
        response="needle-43",
        expected="needle-42",
        ruler_task="niah_single_1",
        intended_length=4096,
    )
    assert grade.correctness == "fail"
    assert grade.counts_toward_accuracy
    assert grade.denominator_eligibility.correctness is True


def test_golden_oracle_table() -> None:
    assert score_response("needle-42", "needle-42", "niah_single_1") is True
    assert score_response("needle-43", "needle-42", "niah_single_1") is False
    assert score_response("  needle-42  ", "needle-42", "vt") is True
    assert score_response("", "needle-42", "qa_1") is False


def test_aggregation_word_sets_are_order_insensitive() -> None:
    assert score_response("apple banana cherry", "cherry apple banana", "cwe") is True
    assert score_response("apple banana", "apple banana cherry", "fwe") is False
    assert score_response("", "", "cwe") is False


def test_empty_response_is_a_graded_failure() -> None:
    task = _task()
    grade = grade_accepted_sample(
        task=task,
        response="   ",
        expected="needle-42",
        ruler_task="niah_single_1",
        intended_length=4096,
    )
    assert grade.correctness == "fail"
    assert grade.format == "fail"
    assert grade.counts_toward_accuracy


def test_unpinned_task_raises_instead_of_grading() -> None:
    task = _task()
    with pytest.raises(UnknownTaskError):
        grade_accepted_sample(
            task=task,
            response="x",
            expected="x",
            ruler_task="ruler_v2_new_task",
            intended_length=4096,
        )
    with pytest.raises(UnknownTaskError):
        score_response("x", "x", "ruler_v2_new_task")


def test_unpinned_length_raises_unless_flagged_unsupported() -> None:
    task = _task()
    with pytest.raises(UnsupportedLengthError):
        grade_accepted_sample(
            task=task,
            response="x",
            expected="x",
            ruler_task="niah_single_1",
            intended_length=2048,
        )


# ---------------------------------------------------------------------------
# Truncation and unsupported lengths: unavailable, never delivered-at-length
# ---------------------------------------------------------------------------


def test_truncated_sample_is_unavailable_not_incorrect() -> None:
    task = _task()
    grade = grade_accepted_sample(
        task=task,
        response="needle-42",
        expected="needle-42",
        ruler_task="niah_single_1",
        intended_length=131072,
        truncated=True,
    )
    assert grade.correctness == "unavailable"
    assert not grade.counts_toward_accuracy
    assert grade.score_components["truncated"] == 1.0
    assert grade.score_components["intended_length"] == pytest.approx(131072.0)


def test_unsupported_length_is_unavailable() -> None:
    task = _task()
    grade = grade_accepted_sample(
        task=task,
        response="needle-42",
        expected="needle-42",
        ruler_task="niah_single_1",
        intended_length=200000,
        unsupported_length=True,
    )
    assert grade.correctness == "unavailable"
    assert not grade.counts_toward_accuracy


def test_truncated_cell_shows_its_denominator_and_stays_null() -> None:
    task = _task()
    truncated = grade_accepted_sample(
        task=task,
        response="needle-42",
        expected="needle-42",
        ruler_task="niah_single_1",
        intended_length=131072,
        truncated=True,
    )
    cell = summarize_cell(
        [truncated],
        ruler_task="niah_single_1",
        requested_length=131072,
        planned=3,
    )
    assert cell.eligible == 0
    assert cell.planned == 3
    assert cell.accuracy is None
    assert cell.accuracy != 0.0  # type: ignore[comparison-overlap]
    assert cell.status == "unavailable"
    dumped = json.loads(cell.model_dump_json())
    assert dumped["accuracy"] is None
    assert dumped["planned"] == 3


def test_partial_cells_have_no_overall_mean() -> None:
    task = _task()
    good = grade_accepted_sample(
        task=task,
        response="a",
        expected="a",
        ruler_task="niah_single_1",
        intended_length=4096,
    )
    truncated = grade_accepted_sample(
        task=task,
        response="a",
        expected="a",
        ruler_task="vt",
        intended_length=4096,
        truncated=True,
    )
    summary = summarize(
        {("niah_single_1", 4096): [good], ("vt", 4096): [truncated]},
        planned_by_cell={("niah_single_1", 4096): 1, ("vt", 4096): 1},
    )
    assert isinstance(summary, RulerSummary)
    assert summary.complete is False
    assert summary.overall_accuracy is None
    assert summary.generation_seed == 42
    assert summary.comparability_note == TOKENIZATION_CAVEAT


def test_unweighted_mean_over_cells_when_complete() -> None:
    def _grade(value: str, want: str) -> object:
        task = _task(item_id=f"{value}-{want}")
        return grade_accepted_sample(
            task=task,
            response=value,
            expected=want,
            ruler_task="niah_single_1",
            intended_length=4096,
        )

    good = _grade("a", "a")
    bad = _grade("a", "b")
    summary = summarize(
        {
            ("niah_single_1", 4096): [good, bad],  # type: ignore[list-item]
            ("vt", 4096): [good],  # type: ignore[list-item]
        },
        planned_by_cell={("niah_single_1", 4096): 2, ("vt", 4096): 1},
    )
    assert summary.complete is True
    # Cell means are 0.5 and 1.0; the unweighted mean is 0.75.
    assert summary.overall_accuracy == pytest.approx(0.75)


def test_unknown_rate_is_null_never_zero() -> None:
    empty = summarize(
        {},
        planned_by_cell={("niah_single_1", 4096): 2},
    )
    assert empty.overall_accuracy is None
    assert empty.complete is False
    dumped = json.loads(empty.model_dump_json())
    assert dumped["overall_accuracy"] is None


def test_unplanned_cell_cannot_enter_the_mean() -> None:
    task = _task()
    grade = grade_accepted_sample(
        task=task,
        response="a",
        expected="a",
        ruler_task="niah_single_1",
        intended_length=4096,
    )
    with pytest.raises(ValueError, match="no planned denominator"):
        summarize(
            {("niah_single_1", 4096): [grade]},
            planned_by_cell={},
        )


# ---------------------------------------------------------------------------
# Length reports: provider counts vs normalized lengths stay separate
# ---------------------------------------------------------------------------


def test_delivered_length_report_keeps_counts_separate() -> None:
    report = report_length(
        ruler_task="niah_single_1",
        requested_length=8192,
        effective_length=8192,
        truncated=False,
        provider_reported_tokens=8200,
        normalized_length=8192,
        template_overhead_tokens=64,
    )
    assert report.requested_length == 8192
    assert report.effective_length == 8192
    assert report.provider_reported_tokens == 8200
    assert report.normalized_length == 8192
    assert report.template_overhead_tokens == 64
    assert report.truncated is False
    # The two counts are independent fields: changing one never moves the other.
    other = report.model_copy(update={"provider_reported_tokens": 9000})
    assert other.normalized_length == report.normalized_length


def test_truncated_report_carries_no_effective_length() -> None:
    report = report_length(
        ruler_task="vt",
        requested_length=131072,
        effective_length=None,
        truncated=True,
        provider_reported_tokens=65536,
        normalized_length=None,
        template_overhead_tokens=128,
    )
    assert report.effective_length is None
    assert report.truncated is True
    assert report.provider_reported_tokens == 65536
    assert report.normalized_length is None


def test_truncated_report_claiming_delivery_raises() -> None:
    with pytest.raises(ValueError, match="effective_length=None"):
        report_length(
            ruler_task="vt",
            requested_length=131072,
            effective_length=131072,
            truncated=True,
        )


def test_delivered_report_with_mismatched_length_raises() -> None:
    with pytest.raises(ValueError, match="effective_length == requested_length"):
        report_length(
            ruler_task="vt",
            requested_length=131072,
            effective_length=65536,
            truncated=False,
        )


def test_missing_counts_stay_missing_not_zero() -> None:
    report = report_length(
        ruler_task="vt",
        requested_length=4096,
        effective_length=4096,
        truncated=False,
    )
    assert report.provider_reported_tokens is None
    assert report.normalized_length is None
    dumped = json.loads(report.model_dump_json())
    assert dumped["provider_reported_tokens"] is None
    assert dumped["normalized_length"] is None


# ---------------------------------------------------------------------------
# Transport / evaluator separation, gold hygiene, revision pins
# ---------------------------------------------------------------------------


def test_transport_failure_is_unavailable_not_incorrect() -> None:
    task = _task()
    generation = _generation(task, None, DeliveryStatus.TRANSPORT_FAILED)
    grade = grade_generation(
        task=task,
        generation=generation,
        expected="needle-42",
        ruler_task="niah_single_1",
        intended_length=4096,
    )
    assert grade.correctness == "unavailable"
    assert grade.transport == "fail"
    assert not grade.counts_toward_accuracy


def test_cancelled_and_unresolved_are_not_counted_as_incorrect() -> None:
    task = _task()
    cancelled = grade_generation(
        task=task,
        generation=_generation(task, None, DeliveryStatus.CANCELLED),
        expected="x",
        ruler_task="vt",
        intended_length=4096,
    )
    unresolved = grade_generation(
        task=task,
        generation=_generation(task, None, DeliveryStatus.UNRESOLVED),
        expected="x",
        ruler_task="vt",
        intended_length=4096,
    )
    assert cancelled.transport == "invalid"
    assert unresolved.transport == "unavailable"
    for grade in (cancelled, unresolved):
        assert grade.correctness == "unavailable"
        assert not grade.counts_toward_accuracy


def test_missing_expected_is_an_evaluator_failure() -> None:
    task = _task()
    for missing in (None, "", "   "):
        grade = grade_accepted_sample(
            task=task,
            response="needle-42",
            expected=missing,
            ruler_task="niah_single_1",
            intended_length=4096,
        )
        assert grade.evaluator == "fail"
        assert grade.correctness == "unavailable"
        assert not grade.counts_toward_accuracy


def test_sample_key_mismatch_refuses_to_grade() -> None:
    task = _task(item_id="niah_single_1-000")
    other = _task(item_id="niah_single_1-001")
    generation = _generation(other, "needle-42")
    with pytest.raises(ValueError, match="does not match"):
        grade_generation(
            task=task,
            generation=generation,
            expected="needle-42",
            ruler_task="niah_single_1",
            intended_length=4096,
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
        benchmark_id="ruler",
        item_id="niah_single_1-000",
        campaign_id="c1",
        endpoint_id="e1",
        repeat_id=1,
        prompt_artifact_hash=prompt_hash(PromptRef.from_request(request)),
        request=request,
        evaluation=EvaluationPayload(
            gold_answer=GOLD_CANARY,
            evaluator_id="ruler-score",
            evaluator_revision=PINNED_RULER_COMMIT,
        ),
    )
    with pytest.raises(ValueError, match="would be sent to a provider"):
        dispatch_request(task)
    with pytest.raises(ValueError, match="would be sent to a provider"):
        grade_accepted_sample(
            task=task,
            response="needle-42",
            expected="needle-42",
            ruler_task="niah_single_1",
            intended_length=4096,
        )


def test_unsupported_revision_is_rejected_not_substituted() -> None:
    with pytest.raises(DatasetRevisionMismatch):
        verify_ruler_revision("some-unpinned-revision")
    assert verify_ruler_revision(PINNED_RULER_COMMIT) == PINNED_RULER_COMMIT


def test_checksum_mismatch_refuses_to_run(tmp_path: Path) -> None:
    from stealthbench.benchmarks.ruler import verify_ruler_checksum

    target = tmp_path / "ruler.jsonl"
    target.write_text('{"key": 1}\n', encoding="utf-8")
    with pytest.raises(Exception, match=r"hashes to|mismatch"):
        verify_ruler_checksum(target, expected_sha256="0" * 64)


def test_grade_survives_a_result_schema_round_trip() -> None:
    task = _task()
    grade = grade_accepted_sample(
        task=task,
        response="needle-42",
        expected="needle-42",
        ruler_task="niah_single_1",
        intended_length=4096,
    )
    revived = type(grade).model_validate(grade.model_dump(mode="json"))
    assert revived == grade
    sample: SampleKey = task.sample_key
    assert revived.sample_key == sample
