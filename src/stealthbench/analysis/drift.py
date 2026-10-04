"""Canary scheduling, drift detection and reveal history (G12 T12C).

Contracts (``BENCHMARK_PLAN.md`` "Track every anonymous alias over time",
``IMPLEMENTATION_PLAN.md`` G12, ``docs/contracts.md`` ``IdentityReport``):

* A small fixed canary suite reruns on a schedule while an alias is
  active. :func:`due_canaries` is pure over stored timestamps and
  :func:`canary_job_key` is idempotent per (alias, slot), so an offline
  scheduled canary never double-dispatches.
* Stable-control simulations must meet the preregistered false-alert
  target and change simulations the declared detection target
  (:func:`evaluate_stable_controls`, :func:`evaluate_change_detection`).
* A control-wide shift (many controls alerting together) is reported as
  suspected infrastructure change, never as per-alias model drift
  (:func:`classify_alias_alerts`).
* Reveal history is append-only: :func:`record_reveal` returns a new
  history and never rewrites an earlier prediction or its evidence.
  :func:`apply_reveal` records the official label on a report while
  preserving the original ranking and evidence.

Offline by construction: pure functions over stored scores and labels.
No transport, no socket, no subprocess, no credential.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final, Literal

from stealthbench.schemas.results import IdentityReport

__all__ = [
    "DEFAULT_CONTROL_WIDE_FRACTION",
    "DEFAULT_DETECTION_TARGET",
    "DEFAULT_FALSE_ALERT_TARGET",
    "AliasClassification",
    "CanarySchedule",
    "ChangeEvaluation",
    "DriftDecision",
    "DriftSpec",
    "RevealEvent",
    "RevealHistory",
    "StableEvaluation",
    "apply_reveal",
    "canary_due",
    "canary_job_key",
    "classify_alias_alerts",
    "detect_drift",
    "due_canaries",
    "evaluate_change_detection",
    "evaluate_stable_controls",
    "record_reveal",
    "shift_statistic",
    "slot_for",
]

#: Preregistered defaults. A campaign freezes its own :class:`DriftSpec`;
#: these are the starting values, not silently applied gates.
DEFAULT_FALSE_ALERT_TARGET: Final[float] = 0.05
DEFAULT_DETECTION_TARGET: Final[float] = 0.90
DEFAULT_CONTROL_WIDE_FRACTION: Final[float] = 0.5


@dataclass(frozen=True, slots=True)
class CanarySchedule:
    """Rerun schedule for one alias's fixed canary suite."""

    alias: str
    interval_seconds: float
    last_run_at: float | None

    def __post_init__(self) -> None:
        if not self.alias:
            raise ValueError("alias must be a non-empty string")
        if not math.isfinite(self.interval_seconds) or self.interval_seconds <= 0:
            raise ValueError(
                f"interval_seconds must be positive finite, got {self.interval_seconds}"
            )
        if self.last_run_at is not None and not math.isfinite(self.last_run_at):
            raise ValueError(f"last_run_at must be finite or None, got {self.last_run_at}")


def canary_due(schedule: CanarySchedule, now: float) -> bool:
    """Whether the alias's canary suite is due at ``now``.

    A suite that never ran is due immediately; otherwise it is due once
    a full interval has elapsed since the last run.
    """
    if not math.isfinite(now):
        raise ValueError(f"now must be finite, got {now}")
    if schedule.last_run_at is None:
        return True
    return (now - schedule.last_run_at) >= schedule.interval_seconds


def due_canaries(schedules: Sequence[CanarySchedule], now: float) -> tuple[str, ...]:
    """Aliases whose canary suites are due, in sorted order."""
    return tuple(sorted(s.alias for s in schedules if canary_due(s, now)))


def slot_for(schedule: CanarySchedule, now: float) -> int:
    """The schedule slot containing ``now`` (one job per slot)."""
    if not math.isfinite(now):
        raise ValueError(f"now must be finite, got {now}")
    return int(now // schedule.interval_seconds)


def canary_job_key(alias: str, slot: int) -> str:
    """Idempotency key for one scheduled canary run.

    The same (alias, slot) always yields the same key, so retrying or
    re-listing a scheduled run cannot dispatch it twice.
    """
    if not alias:
        raise ValueError("alias must be a non-empty string")
    if slot < 0:
        raise ValueError(f"slot must be non-negative, got {slot}")
    return f"canary:{alias}:{slot:08d}"


@dataclass(frozen=True, slots=True)
class DriftSpec:
    """Preregistered drift targets for one monitoring campaign."""

    false_alert_target: float
    detection_target: float
    control_wide_fraction: float = DEFAULT_CONTROL_WIDE_FRACTION

    def __post_init__(self) -> None:
        if not 0.0 < self.false_alert_target < 1.0:
            raise ValueError(
                f"false_alert_target must lie in (0, 1), got {self.false_alert_target}"
            )
        if not 0.0 < self.detection_target <= 1.0:
            raise ValueError(f"detection_target must lie in (0, 1], got {self.detection_target}")
        if not 0.0 < self.control_wide_fraction <= 1.0:
            raise ValueError(
                f"control_wide_fraction must lie in (0, 1], got {self.control_wide_fraction}"
            )


def _checked_scores(scores: Sequence[float], name: str) -> tuple[float, ...]:
    values = tuple(scores)
    if not values:
        raise ValueError(f"{name} must not be empty")
    if any(not math.isfinite(value) for value in values):
        raise ValueError(f"{name} must be finite; missing scores stay missing upstream")
    return values


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _pooled_sd(first: Sequence[float], second: Sequence[float]) -> float:
    pooled = (*first, *second)
    mean = _mean(pooled)
    variance = sum((value - mean) ** 2 for value in pooled) / len(pooled)
    return math.sqrt(variance)


def shift_statistic(baseline: Sequence[float], current: Sequence[float]) -> float:
    """Standardized mean shift between baseline and current canary scores.

    ``|mean(current) - mean(baseline)| / pooled_sd``. Zero spread with
    equal means is no shift (0.0); zero spread with different means is an
    unambiguous shift (infinity), never a division error.
    """
    base = _checked_scores(baseline, "baseline")
    present = _checked_scores(current, "current")
    spread = _pooled_sd(base, present)
    gap = abs(_mean(present) - _mean(base))
    if spread == 0.0:
        return 0.0 if gap == 0.0 else math.inf
    return gap / spread


@dataclass(frozen=True, slots=True)
class DriftDecision:
    """One drift verdict for one alias."""

    alert: bool
    statistic: float
    threshold: float
    detail: str | None


def detect_drift(
    baseline: Sequence[float],
    current: Sequence[float],
    *,
    threshold: float,
) -> DriftDecision:
    """Alert when the shift statistic strictly exceeds ``threshold``.

    The threshold is frozen before monitoring starts; tuning it on the
    alerts under review is the failure this signature exists to prevent.
    """
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError(f"threshold must be a non-negative finite value, got {threshold}")
    statistic = shift_statistic(baseline, current)
    alert = statistic > threshold
    return DriftDecision(
        alert=alert,
        statistic=statistic,
        threshold=threshold,
        detail=(f"shift {statistic:.4f} {'>' if alert else '<='} threshold {threshold:.4f}"),
    )


@dataclass(frozen=True, slots=True)
class StableEvaluation:
    """False-alert measurement over stable (unchanged) controls."""

    n_alerts: int
    n_controls: int
    false_alert_rate: float
    target: float
    meets_target: bool


def evaluate_stable_controls(n_alerts: int, n_controls: int, spec: DriftSpec) -> StableEvaluation:
    """Check stable controls against the preregistered false-alert target."""
    if n_controls <= 0:
        raise ValueError(f"n_controls must be positive, got {n_controls}")
    if not 0 <= n_alerts <= n_controls:
        raise ValueError(f"n_alerts must lie in [0, n_controls], got {n_alerts}")
    rate = n_alerts / n_controls
    return StableEvaluation(
        n_alerts=n_alerts,
        n_controls=n_controls,
        false_alert_rate=rate,
        target=spec.false_alert_target,
        meets_target=rate <= spec.false_alert_target,
    )


@dataclass(frozen=True, slots=True)
class ChangeEvaluation:
    """Detection measurement over simulated changed endpoints."""

    n_detected: int
    n_changed: int
    detection_rate: float
    target: float
    meets_target: bool


def evaluate_change_detection(n_detected: int, n_changed: int, spec: DriftSpec) -> ChangeEvaluation:
    """Check changed-endpoint simulations against the detection target."""
    if n_changed <= 0:
        raise ValueError(f"n_changed must be positive, got {n_changed}")
    if not 0 <= n_detected <= n_changed:
        raise ValueError(f"n_detected must lie in [0, n_changed], got {n_detected}")
    rate = n_detected / n_changed
    return ChangeEvaluation(
        n_detected=n_detected,
        n_changed=n_changed,
        detection_rate=rate,
        target=spec.detection_target,
        meets_target=rate >= spec.detection_target,
    )


@dataclass(frozen=True, slots=True)
class AliasClassification:
    """What an alert pattern means across aliases and controls."""

    kind: Literal["no_change", "alias_change", "control_wide_shift"]
    changed_aliases: tuple[str, ...]
    detail: str


def classify_alias_alerts(alerts: Mapping[str, bool], spec: DriftSpec) -> AliasClassification:
    """Classify one monitoring round's alerts.

    When at least ``control_wide_fraction`` of monitored aliases alert
    together, the round is a suspected control-wide infrastructure shift:
    per-alias identity-change claims are suppressed (no alias is named)
    and the detail says so. Isolated alerts name their aliases.
    """
    if not alerts:
        raise ValueError("classify_alias_alerts needs at least one monitored alias")
    total = len(alerts)
    firing = sorted(alias for alias, alert in alerts.items() if alert)
    if not firing:
        return AliasClassification(
            kind="no_change",
            changed_aliases=(),
            detail=f"0/{total} aliases alerted",
        )
    if len(firing) / total >= spec.control_wide_fraction:
        return AliasClassification(
            kind="control_wide_shift",
            changed_aliases=(),
            detail=(
                f"{len(firing)}/{total} aliases alerted together "
                f"(fraction >= {spec.control_wide_fraction}); suspected "
                "control-wide infrastructure shift, not per-alias model drift"
            ),
        )
    return AliasClassification(
        kind="alias_change",
        changed_aliases=tuple(firing),
        detail=f"{len(firing)}/{total} aliases alerted: {', '.join(firing)}",
    )


@dataclass(frozen=True, slots=True)
class RevealEvent:
    """One official identity reveal for one alias.

    The prediction and the evidence behind it are preserved verbatim;
    the reveal is added alongside, never written over.
    """

    alias: str
    predicted_label: str | None
    predicted_evidence_id: str | None
    revealed_label: str
    revealed_at: float
    note: str | None = None

    def __post_init__(self) -> None:
        if not self.alias:
            raise ValueError("alias must be a non-empty string")
        if not self.revealed_label:
            raise ValueError("revealed_label must be a non-empty string")
        if not math.isfinite(self.revealed_at):
            raise ValueError(f"revealed_at must be finite, got {self.revealed_at}")


@dataclass(frozen=True, slots=True)
class RevealHistory:
    """Append-only reveal log. Events are only ever appended."""

    events: tuple[RevealEvent, ...] = ()


def record_reveal(history: RevealHistory, event: RevealEvent) -> RevealHistory:
    """Append one reveal event, returning the new history.

    The input history is never mutated: the caller keeps the old object
    as the pre-reveal record. Re-revealing an alias appends a second
    event rather than overwriting the first.
    """
    return RevealHistory(events=(*history.events, event))


def apply_reveal(report: IdentityReport, revealed_label: str) -> IdentityReport:
    """Record the official reveal label on a report, preserving the rest.

    The ranking, signal evidence, abstention state and any probabilities
    stay exactly as predicted; only ``official_reveal_label`` is set, so
    a later reveal cannot rewrite history.
    """
    if not revealed_label:
        raise ValueError("revealed_label must be a non-empty string")
    payload = report.model_dump()
    payload["official_reveal_label"] = revealed_label
    return IdentityReport(**payload)
