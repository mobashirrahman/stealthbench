"""T09A: benchmark denominators and StealthBench Core v1 oracles (unit)."""

from __future__ import annotations

import pytest

from stealthbench.analysis.scoring import (
    CORE_CATEGORIES_V1,
    average_benchmark_accuracies,
    benchmark_score,
    category_score_0_100,
    core_index_v1,
    score_campaign,
    summarize_grades,
    task_accuracy,
)
from stealthbench.schemas.results import (
    DenominatorEligibility,
    GradeResult,
    SampleKey,
    StatusFlag,
)

pytestmark = pytest.mark.unit


def _key(benchmark: str, item: str, repeat: int = 1) -> SampleKey:
    return SampleKey(
        campaign_id="c-test",
        endpoint_id="ep-a",
        task_id=f"{benchmark}::{item}",
        repeat_id=repeat,
    )


def _grade(
    benchmark: str,
    item: str,
    *,
    repeat: int = 1,
    correctness: StatusFlag,
    format: StatusFlag,
    transport: StatusFlag,
    evaluator: StatusFlag,
) -> GradeResult:
    eligibility = DenominatorEligibility(
        correctness=correctness in ("pass", "fail"),
        format=format in ("pass", "fail"),
        transport_success=transport == "pass",
        evaluator_ran=evaluator == "pass",
    )
    return GradeResult(
        grader_version="test-grader-v1",
        sample_key=_key(benchmark, item, repeat),
        correctness=correctness,
        format=format,
        transport=transport,
        evaluator=evaluator,
        score_components={},
        denominator_eligibility=eligibility,
    )


def _graded_pass(benchmark: str, item: str, repeat: int = 1) -> GradeResult:
    return _grade(
        benchmark,
        item,
        repeat=repeat,
        correctness="pass",
        format="pass",
        transport="pass",
        evaluator="pass",
    )


def _graded_fail(benchmark: str, item: str, repeat: int = 1) -> GradeResult:
    return _grade(
        benchmark,
        item,
        repeat=repeat,
        correctness="fail",
        format="pass",
        transport="pass",
        evaluator="pass",
    )


def test_core_declares_six_direct_categories() -> None:
    assert CORE_CATEGORIES_V1 == (
        "code_generation",
        "mathematical_reasoning",
        "instruction_following",
        "general_knowledge",
        "tool_use",
        "long_context",
    )


def test_hand_computed_category_mean_and_core_index() -> None:
    # livecodebench: 2 correct of 3 single-repeat tasks -> 2/3.
    livecodebench = [
        _graded_pass("livecodebench", "i1"),
        _graded_pass("livecodebench", "i2"),
        _graded_fail("livecodebench", "i3"),
    ]
    # evalplus: 1 correct of 2 -> 1/2.
    evalplus = [
        _graded_pass("evalplus", "j1"),
        _graded_fail("evalplus", "j2"),
    ]
    live_score = benchmark_score("livecodebench", "code_generation", livecodebench)
    eval_score = benchmark_score("evalplus", "code_generation", evalplus)
    assert live_score.accuracy == pytest.approx(2.0 / 3.0)
    assert eval_score.accuracy == pytest.approx(1.0 / 2.0)
    # Equal-weight category mean, normalised to 0-100:
    # ((2/3) + (1/2)) / 2 * 100 == 58.333...
    expected_category = ((2.0 / 3.0) + (1.0 / 2.0)) / 2.0 * 100.0
    category = category_score_0_100(
        {"livecodebench": live_score.accuracy, "evalplus": eval_score.accuracy},
        declared=("livecodebench", "evalplus"),
    )
    assert category == pytest.approx(expected_category)
    # Hand-computed core over six categories.
    cats = {
        "code_generation": expected_category,
        "mathematical_reasoning": 70.0,
        "instruction_following": 60.0,
        "general_knowledge": 50.0,
        "tool_use": 40.0,
        "long_context": 30.0,
    }
    expected_core = sum(cats.values()) / 6.0
    assert core_index_v1(cats) == pytest.approx(expected_core)


def test_incorrect_versus_missing_have_different_denominators() -> None:
    grades = [
        _graded_pass("ifeval", "i1"),
        _graded_fail("ifeval", "i2"),
        # Transport failure: no gradeable response, excluded from accuracy.
        _grade(
            "ifeval",
            "i3",
            correctness="unavailable",
            format="unavailable",
            transport="fail",
            evaluator="unavailable",
        ),
        # Evaluator failure: delivered but ungraded, excluded from accuracy.
        _grade(
            "ifeval",
            "i4",
            correctness="unavailable",
            format="pass",
            transport="pass",
            evaluator="fail",
        ),
    ]
    summary = summarize_grades(grades)
    assert summary.total == 4
    assert summary.graded == 2
    assert summary.correct == 1
    assert summary.incorrect == 1
    assert summary.missing == 2
    # Incorrect stays in the denominator (1/2); missing does not (not 1/4).
    assert summary.accuracy == pytest.approx(1.0 / 2.0)
    # End-to-end counts every declared sample: 1/4.
    assert summary.end_to_end_success == pytest.approx(1.0 / 4.0)
    # Endpoint success over known transport: 3 pass of 4 known -> 3/4.
    assert summary.endpoint_success_rate == pytest.approx(3.0 / 4.0)


def test_uneven_categories_use_equal_benchmark_weight() -> None:
    # Four correct tasks in one benchmark, one incorrect task in another.
    # Pooled over items this would be 4/5 = 0.8; the category must be 0.5.
    big = [_graded_pass("livecodebench", f"i{n}") for n in range(4)]
    small = [_graded_fail("evalplus", "j1")]
    big_score = benchmark_score("livecodebench", "code_generation", big)
    small_score = benchmark_score("evalplus", "code_generation", small)
    assert big_score.accuracy == pytest.approx(1.0)
    assert small_score.accuracy == pytest.approx(0.0)
    category = category_score_0_100(
        {"livecodebench": big_score.accuracy, "evalplus": small_score.accuracy},
        declared=("livecodebench", "evalplus"),
    )
    assert category == pytest.approx(50.0)
    pooled = (4.0 / 5.0) * 100.0
    assert category != pytest.approx(pooled)


def test_incomplete_coverage_yields_none_never_zero() -> None:
    assert core_index_v1({}) is None
    partial = {
        "code_generation": 80.0,
        "mathematical_reasoning": 70.0,
        "instruction_following": 60.0,
        "general_knowledge": 50.0,
        "tool_use": 40.0,
        # long_context absent
    }
    assert core_index_v1(partial) is None
    with_none = dict(partial)
    with_none["long_context"] = None
    assert core_index_v1(with_none) is None
    # A category with no scored benchmark is None, not 0.
    assert category_score_0_100({}, declared=("livecodebench",)) is None
    assert (
        category_score_0_100({"livecodebench": None}, declared=("livecodebench", "evalplus"))
        is None
    )
    assert average_benchmark_accuracies({}, declared=None) is None


def test_repeated_tasks_cluster_by_task() -> None:
    grades = [
        _graded_pass("math500", "t1", repeat=1),
        _graded_pass("math500", "t1", repeat=2),
        _graded_fail("math500", "t1", repeat=3),
        _graded_pass("math500", "t2", repeat=1),
    ]
    entry = benchmark_score("math500", "mathematical_reasoning", grades)
    # Task means: t1 = 2/3, t2 = 1. Clustered = (2/3 + 1)/2 = 5/6.
    assert entry.n_tasks_total == 2
    assert entry.n_tasks_graded == 2
    assert entry.accuracy == pytest.approx(5.0 / 6.0)
    # The naive pooled ratio (3/4) must differ, proving clustering.
    assert entry.accuracy_pooled == pytest.approx(3.0 / 4.0)
    assert entry.accuracy != pytest.approx(entry.accuracy_pooled)  # type: ignore[arg-type]
    # Direct task helper agrees.
    assert task_accuracy(grades[:3]) == pytest.approx(2.0 / 3.0)
    assert task_accuracy([]) is None


def test_task_with_only_missing_repeats_is_unscored() -> None:
    missing = _grade(
        "ruler",
        "t1",
        correctness="unavailable",
        format="pass",
        transport="pass",
        evaluator="fail",
    )
    assert task_accuracy([missing]) is None
    entry = benchmark_score("ruler", "long_context", [missing])
    assert entry.accuracy is None
    assert entry.accuracy_pooled is None
    assert entry.n_tasks_graded == 0


def test_empty_grade_list_has_no_rates() -> None:
    summary = summarize_grades([])
    assert summary.total == 0
    assert summary.accuracy is None
    assert summary.endpoint_success_rate is None
    assert summary.end_to_end_success is None
    assert summary.format_valid_rate is None
    entry = benchmark_score("bfcl", "tool_use", [])
    assert entry.accuracy is None


def test_score_campaign_matches_hand_computed_oracle() -> None:
    grades = [
        _graded_pass("livecodebench", "i1"),
        _graded_fail("livecodebench", "i2"),
        _graded_pass("evalplus", "j1"),
        _graded_pass("evalplus", "j2"),
        _graded_pass("ifeval", "k1"),
        _graded_fail("mmlu_pro", "m1"),
        _graded_pass("math500", "n1"),
        _graded_pass("bfcl", "b1"),
        _graded_pass("ruler", "r1"),
    ]
    mapping = {
        "livecodebench": "code_generation",
        "evalplus": "code_generation",
        "ifeval": "instruction_following",
        "mmlu_pro": "general_knowledge",
        "math500": "mathematical_reasoning",
        "bfcl": "tool_use",
        "ruler": "long_context",
    }
    declared = {
        "code_generation": ("livecodebench", "evalplus"),
        "instruction_following": ("ifeval",),
        "general_knowledge": ("mmlu_pro",),
        "mathematical_reasoning": ("math500",),
        "tool_use": ("bfcl",),
        "long_context": ("ruler",),
    }
    campaign = score_campaign(grades, mapping, declared_benchmarks_per_category=declared)
    by_id = {entry.benchmark_id: entry for entry in campaign.benchmark_scores}
    assert by_id["livecodebench"].accuracy == pytest.approx(1.0 / 2.0)
    assert by_id["evalplus"].accuracy == pytest.approx(1.0)
    by_cat = {entry.category: entry for entry in campaign.category_scores}
    # code_generation = ((1/2) + 1)/2 * 100 = 75.
    assert by_cat["code_generation"].score == pytest.approx(75.0)
    assert by_cat["instruction_following"].score == pytest.approx(100.0)
    assert by_cat["general_knowledge"].score == pytest.approx(0.0)
    expected_core = (75.0 + 100.0 + 0.0 + 100.0 + 100.0 + 100.0) / 6.0
    assert campaign.core_index == pytest.approx(expected_core)
    assert campaign.summary.total == 9


def test_score_campaign_without_one_category_has_no_core_index() -> None:
    grades = [_graded_pass("ifeval", "k1")]
    mapping = {"ifeval": "instruction_following"}
    campaign = score_campaign(grades, mapping)
    # Only one category is present; the six-category headline is absent.
    assert campaign.core_index is None
