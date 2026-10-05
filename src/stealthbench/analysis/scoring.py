"""Benchmark denominators and the StealthBench Core v1 index (G09 T09A).

Contracts implemented here (see ``docs/contracts.md`` section 2 and
``BENCHMARK_PLAN.md``):

* The four per-sample statuses (``correctness`` / ``format`` / ``transport`` /
  ``evaluator``) are never merged. Each published rate has its own explicit
  denominator.
* An incorrect answer (``correctness == "fail"`` with ``evaluator == "pass"``)
  stays in the correctness denominator. A missing grade (transport failure,
  evaluator failure, or ``correctness`` unavailable) is excluded from that
  denominator and counted as missing.
* ``StealthBench Core v1`` is an equal-weight mean of the six pilot categories
  (each normalised to 0-100). Within a category, declared component
  (benchmark) accuracies are averaged with equal weight, so a benchmark with
  more items does not silently dominate. The headline index is ``None`` when
  any declared category is absent or unscored -- never ``0``.
* Repeated trials of one task are clustered: a task contributes its mean over
  graded repeats, and a benchmark contributes the mean over its graded tasks.
  Repeating one item three times does not triple its weight.

Offline by construction: pure functions over stored grades. No transport,
no socket, no subprocess, no credential.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from stealthbench.schemas.results import GradeResult

#: The six direct pilot categories that enter the headline index. Agent tracks
#: never enter it (see ``tests/contract/test_profile_spec.py``).
CORE_CATEGORIES_V1: Final[tuple[str, ...]] = (
    "code_generation",
    "mathematical_reasoning",
    "instruction_following",
    "general_knowledge",
    "tool_use",
    "long_context",
)

#: Project-defined index identifier. Pilot and full-suite runs use distinct
#: score versions so a pilot mean is never compared as a full-suite index.
CORE_INDEX_VERSION: Final[str] = "stealthbench-core-v1"
PILOT_SCORE_VERSION: Final[str] = "stealthbench-core-pilot-v1"
FULL_SCORE_VERSION: Final[str] = "stealthbench-core-full-v1"


@dataclass(frozen=True, slots=True)
class GradeSummary:
    """Counts and rates over one grade list, each rate with its own denominator.

    * ``accuracy`` (response-conditional): correct / graded. Transport and
      evaluator failures are excluded, not counted as incorrect.
    * ``endpoint_success_rate``: transport ``pass`` / known transport
      (``pass`` + ``fail``). ``invalid`` / ``unavailable`` transport stays out.
    * ``end_to_end_success``: correct / total samples. Transport failures count
      as non-success here, which is why this differs from ``accuracy``.
    * ``format_valid_rate``: reported separately; format never enters the
      correctness denominator.
    """

    total: int
    graded: int
    correct: int
    incorrect: int
    missing: int
    transport_success: int
    transport_failed: int
    evaluator_ran: int
    format_eligible: int
    format_valid: int
    accuracy: float | None
    endpoint_success_rate: float | None
    end_to_end_success: float | None
    format_valid_rate: float | None


@dataclass(frozen=True, slots=True)
class BenchmarkScore:
    """One benchmark's score with task-clustered and pooled views.

    ``accuracy`` is the headline: the mean of per-task means over graded tasks
    (0-1), so repeats of one item cluster. ``accuracy_pooled`` is the naive
    correct / graded ratio, reported for diagnostics only.
    """

    benchmark_id: str
    category: str
    n_samples_total: int
    n_samples_graded: int
    n_correct: int
    n_incorrect: int
    n_missing: int
    n_tasks_total: int
    n_tasks_graded: int
    accuracy: float | None
    accuracy_pooled: float | None


@dataclass(frozen=True, slots=True)
class CategoryScore:
    """One category's 0-100 score as an equal-weight mean of its benchmarks."""

    category: str
    declared_benchmarks: tuple[str, ...]
    scored_benchmarks: tuple[str, ...]
    score: float | None


@dataclass(frozen=True, slots=True)
class CampaignScores:
    """Per-benchmark, per-category and headline scores for one endpoint."""

    benchmark_scores: tuple[BenchmarkScore, ...]
    category_scores: tuple[CategoryScore, ...]
    core_index: float | None
    score_version: str
    summary: GradeSummary


def is_graded(grade: GradeResult) -> bool:
    """Whether a grade belongs in a correctness denominator."""
    return grade.counts_toward_accuracy


def is_correct(grade: GradeResult) -> bool:
    """Whether a grade is a graded correct answer (not merely unscored)."""
    return grade.counts_toward_accuracy and grade.correctness == "pass"


def task_accuracy(grades: Sequence[GradeResult]) -> float | None:
    """Mean correctness over graded repeats of one task, or ``None``.

    Incorrect (``fail``) contributes 0. Missing (ungraded) repeats are excluded.
    ``None`` means the task has no graded repeat and must not enter a mean.
    """
    graded = [grade for grade in grades if grade.counts_toward_accuracy]
    if not graded:
        return None
    correct = sum(1 for grade in graded if grade.correctness == "pass")
    return correct / len(graded)


def summarize_grades(grades: Sequence[GradeResult]) -> GradeSummary:
    """Count denominators separately and derive each rate over its own base.

    Incorrect versus missing is the load-bearing distinction: ``incorrect``
    lowers ``accuracy`` while ``missing`` only shrinks its denominator.
    Every rate is ``None`` when its denominator is zero -- never ``0`` for
    "no data".
    """
    total = len(grades)
    graded = sum(1 for grade in grades if grade.counts_toward_accuracy)
    correct = sum(1 for grade in grades if is_correct(grade))
    incorrect = graded - correct
    missing = total - graded
    transport_success = sum(1 for grade in grades if grade.transport == "pass")
    transport_failed = sum(1 for grade in grades if grade.transport == "fail")
    transport_known = transport_success + transport_failed
    evaluator_ran = sum(1 for grade in grades if grade.evaluator == "pass")
    format_eligible = sum(1 for grade in grades if grade.format in ("pass", "fail"))
    format_valid = sum(1 for grade in grades if grade.format == "pass")
    return GradeSummary(
        total=total,
        graded=graded,
        correct=correct,
        incorrect=incorrect,
        missing=missing,
        transport_success=transport_success,
        transport_failed=transport_failed,
        evaluator_ran=evaluator_ran,
        format_eligible=format_eligible,
        format_valid=format_valid,
        accuracy=(correct / graded) if graded else None,
        endpoint_success_rate=((transport_success / transport_known) if transport_known else None),
        end_to_end_success=(correct / total) if total else None,
        format_valid_rate=(format_valid / format_eligible) if format_eligible else None,
    )


def benchmark_score(
    benchmark_id: str, category: str, grades: Sequence[GradeResult]
) -> BenchmarkScore:
    """Score one benchmark by clustering repeats within each task first.

    Per-task accuracy is the mean over that task's graded repeats; the
    benchmark accuracy is the mean over graded tasks. An uneven repeat count
    (e.g. one task repeated three times, another once) therefore cannot
    overweight the repeated item.
    """
    grouped: dict[str, list[GradeResult]] = {}
    for grade in grades:
        grouped.setdefault(grade.sample_key.task_id, []).append(grade)
    per_task: list[float] = []
    for task_grades in grouped.values():
        mean = task_accuracy(task_grades)
        if mean is not None:
            per_task.append(mean)
    summary = summarize_grades(grades)
    return BenchmarkScore(
        benchmark_id=benchmark_id,
        category=category,
        n_samples_total=summary.total,
        n_samples_graded=summary.graded,
        n_correct=summary.correct,
        n_incorrect=summary.incorrect,
        n_missing=summary.missing,
        n_tasks_total=len(grouped),
        n_tasks_graded=len(per_task),
        accuracy=(sum(per_task) / len(per_task)) if per_task else None,
        accuracy_pooled=summary.accuracy,
    )


def average_benchmark_accuracies(
    accuracies: Mapping[str, float | None],
    *,
    declared: Collection[str] | None = None,
) -> float | None:
    """Equal-weight mean of benchmark accuracies (0-1 scale).

    When ``declared`` is given, every declared benchmark must be present and
    scored, otherwise the category is incomplete and the result is ``None``.
    Without ``declared``, the mean runs over whatever is scored (``None``
    when nothing is). Missing is ``None``, never ``0``.
    """
    if declared is not None:
        values: list[float] = []
        for name in declared:
            value = accuracies.get(name)
            if value is None:
                return None
            values.append(value)
        if not values:
            return None
        return sum(values) / len(values)
    scored = [value for value in accuracies.values() if value is not None]
    if not scored:
        return None
    return sum(scored) / len(scored)


def category_score_0_100(
    benchmark_accuracies: Mapping[str, float | None],
    *,
    declared: Collection[str] | None = None,
) -> float | None:
    """Equal-weight category score normalised to 0-100.

    Benchmark accuracies are 0-1; the category is their mean scaled by 100 so
    a benchmark with more items cannot dominate. ``None`` on incomplete
    coverage, never ``0``.
    """
    mean = average_benchmark_accuracies(benchmark_accuracies, declared=declared)
    if mean is None:
        return None
    return mean * 100.0


def core_index_v1(
    category_scores: Mapping[str, float | None],
    *,
    declared: Collection[str] = CORE_CATEGORIES_V1,
) -> float | None:
    """StealthBench Core v1: equal-weight mean of the six categories (0-100).

    Every declared category must be present and scored. A missing or unscored
    category yields ``None`` -- an incomplete core has no headline index.
    """
    values: list[float] = []
    for category in declared:
        value = category_scores.get(category)
        if value is None:
            return None
        values.append(value)
    if not values:
        return None
    return sum(values) / len(values)


def benchmark_id_of(grade: GradeResult) -> str:
    """Benchmark owning a grade, from the ``benchmark::item`` task id."""
    return grade.sample_key.task_id.split("::", 1)[0]


def score_campaign(
    grades: Sequence[GradeResult],
    benchmark_to_category: Mapping[str, str],
    *,
    declared_benchmarks_per_category: Mapping[str, Collection[str]] | None = None,
    declared_categories: Collection[str] = CORE_CATEGORIES_V1,
    score_version: str = CORE_INDEX_VERSION,
) -> CampaignScores:
    """Aggregate one endpoint's grades into benchmark, category and core scores.

    Grades are grouped by benchmark (from each grade's task id), scored with
    task clustering, then averaged with equal weight into categories (0-100)
    and into the headline index (0-100). Unsupported or unrun benchmarks stay
    absent: they produce ``None`` scores and, when declared, a ``None`` core
    index rather than a silently reduced denominator.
    """
    by_benchmark: dict[str, list[GradeResult]] = {}
    for grade in grades:
        by_benchmark.setdefault(benchmark_id_of(grade), []).append(grade)
    benchmark_scores: list[BenchmarkScore] = []
    accuracies_by_category: dict[str, dict[str, float | None]] = {}
    for benchmark_id, category in benchmark_to_category.items():
        scores = by_benchmark.get(benchmark_id, [])
        entry = benchmark_score(benchmark_id, category, scores)
        benchmark_scores.append(entry)
        accuracies_by_category.setdefault(category, {})[benchmark_id] = entry.accuracy
    category_scores: list[CategoryScore] = []
    headline_inputs: dict[str, float | None] = {}
    categories = sorted(set(benchmark_to_category.values()) | set(headline_inputs))
    for category in sorted(set(benchmark_to_category.values())):
        accuracies = accuracies_by_category.get(category, {})
        declared = (
            declared_benchmarks_per_category.get(category)
            if declared_benchmarks_per_category is not None
            else None
        )
        # When no explicit declaration exists, the observed benchmarks are the
        # declared set: dropping one would silently reduce the denominator.
        effective_declared: Collection[str] | None = (
            declared if declared is not None else tuple(sorted(accuracies))
        )
        score = category_score_0_100(accuracies, declared=effective_declared)
        scored = tuple(sorted(name for name, value in accuracies.items() if value is not None))
        declared_tuple = tuple(sorted(effective_declared)) if effective_declared else ()
        category_scores.append(
            CategoryScore(
                category=category,
                declared_benchmarks=declared_tuple,
                scored_benchmarks=scored,
                score=score,
            )
        )
        headline_inputs[category] = score
    _ = categories  # kept for readability of the category universe above
    return CampaignScores(
        benchmark_scores=tuple(sorted(benchmark_scores, key=lambda entry: entry.benchmark_id)),
        category_scores=tuple(sorted(category_scores, key=lambda entry: entry.category)),
        core_index=core_index_v1(headline_inputs, declared=declared_categories),
        score_version=score_version,
        summary=summarize_grades(grades),
    )


__all__ = [
    "CORE_CATEGORIES_V1",
    "CORE_INDEX_VERSION",
    "FULL_SCORE_VERSION",
    "PILOT_SCORE_VERSION",
    "BenchmarkScore",
    "CampaignScores",
    "CategoryScore",
    "GradeSummary",
    "average_benchmark_accuracies",
    "benchmark_id_of",
    "benchmark_score",
    "category_score_0_100",
    "core_index_v1",
    "is_correct",
    "is_graded",
    "score_campaign",
    "summarize_grades",
    "task_accuracy",
]
