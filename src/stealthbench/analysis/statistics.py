"""Paired task-level bootstrap intervals (G09 T09B).

Fairness contract (``BENCHMARK_PLAN.md``): compare endpoints on shared
declared items, and bootstrap at the task level so repetitions of one item
are not treated as independent problems.

Clustering rule: each task contributes its mean over graded repeats. The
bootstrap resamples tasks (not repeats) with replacement using a seeded RNG.
Shared tasks are sorted before resampling, so input ordering cannot change
the interval. The same seed always yields the same interval.

Unavailable, with a reason, beats a fabricated number: too few shared tasks
yields ``available=False`` rather than a degenerate interval.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

#: Minimum shared tasks for a task-level interval by default. One shared task
#: cannot support an interval; the caller may raise the floor explicitly.
MIN_SHARED_TASKS: Final[int] = 2

#: Default bootstrap replicates for a 95% interval.
DEFAULT_RESAMPLES: Final[int] = 2000


@dataclass(frozen=True, slots=True)
class PairedDifference:
    """Paired mean difference (A minus B) over shared tasks with a CI."""

    available: bool
    reason: str | None
    observed_difference: float | None
    ci_low: float | None
    ci_high: float | None
    confidence: float
    n_shared_tasks: int
    n_resamples: int
    seed: int


@dataclass(frozen=True, slots=True)
class MeanInterval:
    """Single-endpoint task-clustered mean with a bootstrap CI."""

    available: bool
    reason: str | None
    observed_mean: float | None
    ci_low: float | None
    ci_high: float | None
    confidence: float
    n_tasks: int
    n_resamples: int
    seed: int


def task_means(outcomes: Mapping[str, Sequence[float]]) -> dict[str, float]:
    """Per-task means over repeats, skipping tasks with no repeats.

    Each task's repeats cluster into one number first; tasks with an empty
    repeat list are excluded rather than scored as zero.
    """
    means: dict[str, float] = {}
    for task_id, repeats in outcomes.items():
        if not repeats:
            continue
        means[task_id] = sum(repeats) / len(repeats)
    return means


def shared_task_ids(
    first: Mapping[str, Sequence[float]],
    second: Mapping[str, Sequence[float]],
) -> tuple[list[str], int, int]:
    """Sorted shared tasks plus exclusive counts (never cherry-picked).

    Returns ``(shared_sorted, only_in_first, only_in_second)``. Sorting makes
    every downstream resample order-invariant.
    """
    first_means = task_means(first)
    second_means = task_means(second)
    shared = sorted(set(first_means) & set(second_means))
    only_first = len(set(first_means) - set(second_means))
    only_second = len(set(second_means) - set(first_means))
    return shared, only_first, only_second


def _check_interval_args(confidence: float, n_resamples: int) -> None:
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be within (0, 1), got {confidence}")
    if n_resamples < 1:
        raise ValueError(f"n_resamples must be positive, got {n_resamples}")


def _percentile(sorted_values: Sequence[float], fraction: float) -> float:
    """Linear-interpolation percentile over an already sorted sequence."""
    if not sorted_values:
        raise ValueError("cannot take a percentile of an empty sequence")
    if fraction <= 0.0:
        return sorted_values[0]
    if fraction >= 1.0:
        return sorted_values[-1]
    position = fraction * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _bootstrap_ci(
    values: Sequence[float],
    *,
    seed: int,
    n_resamples: int,
    confidence: float,
) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean, seeded and deterministic."""
    rng = random.Random(seed)
    count = len(values)
    replicates: list[float] = []
    for _ in range(n_resamples):
        total = 0.0
        for _ in range(count):
            total += values[rng.randrange(count)]
        replicates.append(total / count)
    replicates.sort()
    tail = (1.0 - confidence) / 2.0
    return _percentile(replicates, tail), _percentile(replicates, 1.0 - tail)


def paired_difference_ci(
    first: Mapping[str, Sequence[float]],
    second: Mapping[str, Sequence[float]],
    *,
    seed: int,
    n_resamples: int = DEFAULT_RESAMPLES,
    confidence: float = 0.95,
    min_tasks: int = MIN_SHARED_TASKS,
) -> PairedDifference:
    """Paired bootstrap CI of the mean difference over shared tasks.

    Only tasks scored by both endpoints enter the comparison. Each task
    contributes ``mean(first repeats) - mean(second repeats)``; the bootstrap
    resamples those per-task differences at the task level. Identical paired
    outcomes yield a zero-width interval at the observed difference (typically
    0.0), not an error. Fewer than ``min_tasks`` shared tasks yields an
    unavailable result with a reason.
    """
    _check_interval_args(confidence, n_resamples)
    if min_tasks < 1:
        raise ValueError(f"min_tasks must be positive, got {min_tasks}")
    first_means = task_means(first)
    second_means = task_means(second)
    shared = sorted(set(first_means) & set(second_means))
    if not shared:
        return PairedDifference(
            available=False,
            reason="no shared tasks: endpoints have no jointly scored item",
            observed_difference=None,
            ci_low=None,
            ci_high=None,
            confidence=confidence,
            n_shared_tasks=0,
            n_resamples=n_resamples,
            seed=seed,
        )
    if len(shared) < min_tasks:
        return PairedDifference(
            available=False,
            reason=(
                f"only {len(shared)} shared tasks; need at least {min_tasks} "
                "for a task-level interval"
            ),
            observed_difference=None,
            ci_low=None,
            ci_high=None,
            confidence=confidence,
            n_shared_tasks=len(shared),
            n_resamples=n_resamples,
            seed=seed,
        )
    differences = [first_means[task_id] - second_means[task_id] for task_id in shared]
    observed = sum(differences) / len(differences)
    low, high = _bootstrap_ci(
        differences, seed=seed, n_resamples=n_resamples, confidence=confidence
    )
    return PairedDifference(
        available=True,
        reason=None,
        observed_difference=observed,
        ci_low=low,
        ci_high=high,
        confidence=confidence,
        n_shared_tasks=len(shared),
        n_resamples=n_resamples,
        seed=seed,
    )


def mean_ci(
    outcomes: Mapping[str, Sequence[float]],
    *,
    seed: int,
    n_resamples: int = DEFAULT_RESAMPLES,
    confidence: float = 0.95,
    min_tasks: int = MIN_SHARED_TASKS,
) -> MeanInterval:
    """Bootstrap CI of one endpoint's task-clustered mean.

    Tasks cluster first (mean over repeats), then the bootstrap resamples
    tasks. All-zero or all-one outcomes stay available with a degenerate
    interval; too few tasks is unavailable with a reason.
    """
    _check_interval_args(confidence, n_resamples)
    if min_tasks < 1:
        raise ValueError(f"min_tasks must be positive, got {min_tasks}")
    means = task_means(outcomes)
    ordered = sorted(means)
    if not ordered:
        return MeanInterval(
            available=False,
            reason="no scored tasks: nothing to average",
            observed_mean=None,
            ci_low=None,
            ci_high=None,
            confidence=confidence,
            n_tasks=0,
            n_resamples=n_resamples,
            seed=seed,
        )
    if len(ordered) < min_tasks:
        return MeanInterval(
            available=False,
            reason=(
                f"only {len(ordered)} scored tasks; need at least {min_tasks} "
                "for a task-level interval"
            ),
            observed_mean=None,
            ci_low=None,
            ci_high=None,
            confidence=confidence,
            n_tasks=len(ordered),
            n_resamples=n_resamples,
            seed=seed,
        )
    values = [means[task_id] for task_id in ordered]
    observed = sum(values) / len(values)
    low, high = _bootstrap_ci(values, seed=seed, n_resamples=n_resamples, confidence=confidence)
    return MeanInterval(
        available=True,
        reason=None,
        observed_mean=observed,
        ci_low=low,
        ci_high=high,
        confidence=confidence,
        n_tasks=len(ordered),
        n_resamples=n_resamples,
        seed=seed,
    )


__all__ = [
    "DEFAULT_RESAMPLES",
    "MIN_SHARED_TASKS",
    "MeanInterval",
    "PairedDifference",
    "mean_ci",
    "paired_difference_ci",
    "shared_task_ids",
    "task_means",
]
