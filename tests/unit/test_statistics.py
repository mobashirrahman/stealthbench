"""T09B: paired task-level bootstrap intervals (unit)."""

from __future__ import annotations

import pytest

from stealthbench.analysis.statistics import (
    mean_ci,
    paired_difference_ci,
    shared_task_ids,
    task_means,
)

pytestmark = pytest.mark.unit


def test_task_means_cluster_repeats_and_skip_empty() -> None:
    means = task_means({"t1": [1.0, 1.0, 0.0], "t2": [1.0], "t3": []})
    assert means["t1"] == pytest.approx(2.0 / 3.0)
    assert means["t2"] == pytest.approx(1.0)
    assert "t3" not in means


def test_shared_tasks_are_sorted_and_exclusive_counts_reported() -> None:
    first = {"t3": [1.0], "t1": [0.0], "t2": [1.0]}
    second = {"t2": [1.0], "t4": [0.0], "t3": [0.0]}
    shared, only_first, only_second = shared_task_ids(first, second)
    assert shared == ["t2", "t3"]
    assert only_first == 1
    assert only_second == 1


def test_paired_comparison_uses_shared_items_only() -> None:
    first = {"t1": [1.0], "t2": [1.0], "t3": [0.0], "t4": [1.0]}
    second = {"t2": [0.0], "t3": [0.0], "t5": [1.0]}
    result = paired_difference_ci(first, second, seed=7, n_resamples=500)
    assert result.available is True
    assert result.n_shared_tasks == 2
    # Shared means: t2: 1-0=1, t3: 0-0=0 -> observed (1+0)/2 = 0.5.
    assert result.observed_difference == pytest.approx(0.5)
    assert result.reason is None
    assert result.ci_low is not None and result.ci_high is not None
    assert result.ci_low <= result.observed_difference <= result.ci_high  # type: ignore[operator]


def test_repetitions_are_clustered_not_independent() -> None:
    # t1 has three repeats, t2 has one. Clustered task means give differences
    # +1 (t1) and -1 (t2) -> observed 0. A repeat-pooled mean would give 0.5.
    first = {"t1": [1.0, 1.0, 1.0], "t2": [0.0]}
    second = {"t1": [0.0, 0.0, 0.0], "t2": [1.0]}
    result = paired_difference_ci(first, second, seed=11, n_resamples=500)
    assert result.available is True
    assert result.observed_difference == pytest.approx(0.0)


def test_seeded_resampling_is_deterministic_and_order_invariant() -> None:
    first = {"t1": [1.0], "t2": [0.0], "t3": [1.0], "t4": [0.0], "t5": [1.0]}
    second = {"t1": [0.0], "t2": [0.0], "t3": [0.0], "t4": [1.0], "t5": [1.0]}
    once = paired_difference_ci(first, second, seed=123, n_resamples=500)
    twice = paired_difference_ci(first, second, seed=123, n_resamples=500)
    assert once.ci_low == twice.ci_low
    assert once.ci_high == twice.ci_high
    assert once.observed_difference == twice.observed_difference
    # Same content in a different insertion order must agree exactly.
    first_shuffled = {"t5": [1.0], "t3": [1.0], "t1": [1.0], "t4": [0.0], "t2": [0.0]}
    second_shuffled = {"t4": [1.0], "t2": [0.0], "t5": [1.0], "t1": [0.0], "t3": [0.0]}
    shuffled = paired_difference_ci(first_shuffled, second_shuffled, seed=123, n_resamples=500)
    assert shuffled.observed_difference == once.observed_difference
    assert shuffled.ci_low == once.ci_low
    assert shuffled.ci_high == once.ci_high


def test_identical_paired_outcomes_give_zero_width_interval() -> None:
    tasks = {f"t{n}": [1.0] if n % 2 == 0 else [0.0] for n in range(6)}
    result = paired_difference_ci(tasks, dict(tasks), seed=5, n_resamples=500)
    assert result.available is True
    assert result.observed_difference == pytest.approx(0.0)
    assert result.ci_low == pytest.approx(0.0)
    assert result.ci_high == pytest.approx(0.0)


def test_zero_versus_one_success_stays_available() -> None:
    low = {f"t{n}": [0.0] for n in range(5)}
    high = {f"t{n}": [1.0] for n in range(5)}
    result = paired_difference_ci(low, high, seed=9, n_resamples=500)
    assert result.available is True
    assert result.observed_difference == pytest.approx(-1.0)
    assert result.ci_low == pytest.approx(-1.0)
    assert result.ci_high == pytest.approx(-1.0)
    # Single-endpoint extremes are also defined, not errors.
    zeros = mean_ci(low, seed=9, n_resamples=500)
    ones = mean_ci(high, seed=9, n_resamples=500)
    assert zeros.available is True and zeros.observed_mean == pytest.approx(0.0)
    assert zeros.ci_low == pytest.approx(0.0) and zeros.ci_high == pytest.approx(0.0)
    assert ones.available is True and ones.observed_mean == pytest.approx(1.0)
    assert ones.ci_low == pytest.approx(1.0) and ones.ci_high == pytest.approx(1.0)


def test_too_few_tasks_is_unavailable_with_reason() -> None:
    one_each = paired_difference_ci({"t1": [1.0]}, {"t1": [0.0]}, seed=1)
    assert one_each.available is False
    assert one_each.observed_difference is None
    assert one_each.ci_low is None and one_each.ci_high is None
    assert one_each.reason is not None and "1" in one_each.reason
    empty = paired_difference_ci({}, {"t1": [1.0]}, seed=1)
    assert empty.available is False
    assert empty.reason is not None and "no shared tasks" in empty.reason
    single = mean_ci({"only": [1.0]}, seed=1)
    assert single.available is False
    assert single.observed_mean is None
    assert single.reason is not None
    assert mean_ci({}, seed=1).available is False


def test_disjoint_endpoints_have_no_shared_interval() -> None:
    result = paired_difference_ci({"t1": [1.0]}, {"t2": [1.0]}, seed=3)
    assert result.available is False
    assert result.n_shared_tasks == 0


def test_hand_computed_observed_difference() -> None:
    first = {"t1": [1.0], "t2": [0.0], "t3": [1.0, 0.0]}
    second = {"t1": [0.0], "t2": [0.0], "t3": [0.0, 0.0]}
    # Task means: first 1.0, 0.0, 0.5; second 0.0, 0.0, 0.0.
    # Differences: 1.0, 0.0, 0.5 -> observed (1.5)/3 = 0.5.
    result = paired_difference_ci(first, second, seed=42, n_resamples=500)
    assert result.observed_difference == pytest.approx(0.5)
    solo = mean_ci(first, seed=42, n_resamples=500)
    assert solo.observed_mean == pytest.approx((1.0 + 0.0 + 0.5) / 3.0)


def test_invalid_interval_arguments_raise() -> None:
    with pytest.raises(ValueError):
        paired_difference_ci({"t1": [1.0], "t2": [0.0]}, {"t1": [1.0]}, seed=1, confidence=1.5)
    with pytest.raises(ValueError):
        mean_ci({"t1": [1.0], "t2": [0.0]}, seed=1, n_resamples=0)
