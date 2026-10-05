"""Deterministic scheduling and bounded retries.

The scheduler decides *what to dispatch, in what order, and how many times*. It does
not talk to a provider and it does not decide what an answer is worth.

Three properties matter and each is enforced here rather than by convention:

* **Determinism.** The same campaign, seed and task list produce the same plan. The
  interleaving is drawn from a seeded generator, never from a clock or from set
  ordering, so a run is reproducible byte for byte.
* **Bounded retries.** Only the transport failures the contract declares retryable
  are retried, at most ``max_attempts`` times, with backoff read from an injected
  clock. A task that failed for any other reason is finished, not retried.
* **Task failures are not transport failures.** A wrong answer or a grader refusal
  ends that attempt. Retrying it would inflate the sample count and the cost.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from stealthbench.adapters.base import RETRYABLE_FAILURES, FailureKind
from stealthbench.schemas.campaign import RetryPolicy
from stealthbench.schemas.hashing import content_digest
from stealthbench.schemas.results import SampleKey, TaskSpec, task_id_for

__all__ = [
    "AttemptPlan",
    "PlanItem",
    "RetryDecision",
    "RunPlan",
    "backoff_seconds",
    "build_plan",
    "plan_digest",
    "should_retry",
]


class SkipReason(StrEnum):
    """Why an item never reached a provider.

    A skipped item is not a failed one. The distinction matters because a skipped
    item is absent from a denominator rather than counted against it.
    """

    ALREADY_ACCEPTED = "already_accepted"
    DUPLICATE_IN_PLAN = "duplicate_in_plan"
    NOT_ELIGIBLE = "not_eligible"


@dataclass(frozen=True, slots=True)
class AttemptPlan:
    """One dispatch: a sample key, which attempt it is, and the backoff before it."""

    sample_key: SampleKey
    attempt_number: int
    delay_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempt_number": self.attempt_number,
            "delay_seconds": self.delay_seconds,
            "sample_key": list(self.sample_key.as_tuple()),
        }


@dataclass(frozen=True, slots=True)
class PlanItem:
    """A sample and the attempts planned for it, in dispatch order."""

    sample_key: SampleKey
    attempts: tuple[AttemptPlan, ...]
    skipped: SkipReason | None = None
    skip_detail: str | None = None

    @property
    def is_skipped(self) -> bool:
        return self.skipped is not None

    @property
    def total_delay_seconds(self) -> float:
        return sum(attempt.delay_seconds for attempt in self.attempts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempts": [attempt.to_dict() for attempt in self.attempts],
            "sample_key": list(self.sample_key.as_tuple()),
            "skipped": None if self.skipped is None else str(self.skipped),
            "skip_detail": self.skip_detail,
        }


@dataclass(frozen=True, slots=True)
class RunPlan:
    """The whole campaign's dispatch order.

    ``plan_digest`` covers the order and the attempts, so two plans that dispatch the
    same work in the same sequence are verifiably the same plan.
    """

    campaign_id: str
    seed: int
    items: tuple[PlanItem, ...]
    policy: RetryPolicy

    @property
    def dispatches(self) -> int:
        return sum(len(item.attempts) for item in self.items if not item.is_skipped)

    @property
    def skipped(self) -> tuple[PlanItem, ...]:
        return tuple(item for item in self.items if item.is_skipped)

    @property
    def eligible(self) -> tuple[PlanItem, ...]:
        return tuple(item for item in self.items if not item.is_skipped)

    def digest(self) -> str:
        return plan_digest(self)

    def to_dict(self) -> dict[str, Any]:
        return {
            "campaign_id": self.campaign_id,
            "digest": self.digest(),
            "dispatches": self.dispatches,
            "items": [item.to_dict() for item in self.items],
            "policy": self.policy.model_dump(mode="json"),
            "seed": self.seed,
        }


def plan_digest(plan: RunPlan) -> str:
    """A stable digest of the dispatch order and the attempts it contains."""
    return content_digest(
        {
            "campaign_id": plan.campaign_id,
            "items": [item.to_dict() for item in plan.items],
            "policy": plan.policy.model_dump(mode="json"),
            "seed": plan.seed,
        }
    )


def backoff_seconds(policy: RetryPolicy, attempt_number: int) -> float:
    """Seconds to wait before ``attempt_number``, clamped to the declared maximum.

    Deterministic: no jitter is added here. A jittered backoff would make the plan
    depend on the clock, and a plan that cannot be replayed cannot be verified.
    """
    if attempt_number < 1:
        return 0.0
    delay = policy.initial_backoff_seconds * (policy.multiplier ** (attempt_number - 1))
    return min(delay, policy.max_backoff_seconds)


@dataclass(frozen=True, slots=True)
class RetryDecision:
    """Whether another attempt is permitted, and if not, why."""

    retry: bool
    reason: str
    delay_seconds: float = 0.0
    attempt_number: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempt_number": self.attempt_number,
            "delay_seconds": self.delay_seconds,
            "reason": self.reason,
            "retry": self.retry,
        }


def should_retry(
    *,
    failure_kind: FailureKind | str | None,
    attempt_number: int,
    policy: RetryPolicy,
    cancelled: bool = False,
) -> RetryDecision:
    """Decide whether a failed dispatch may be attempted again.

    Only the transport failures the contract declares retryable qualify. Everything
    else -- including a task that was answered and graded wrong -- is final, because
    retrying it would create a second sample for one (campaign, endpoint, task, repeat).
    """
    if cancelled:
        return RetryDecision(False, "cancelled", attempt_number=attempt_number)
    if failure_kind is None:
        return RetryDecision(False, "not a failure", attempt_number=attempt_number)
    kind = failure_kind if isinstance(failure_kind, FailureKind) else _as_failure_kind(failure_kind)
    if kind is None:
        return RetryDecision(
            False, f"unknown failure kind {failure_kind!r}", attempt_number=attempt_number
        )
    if kind not in RETRYABLE_FAILURES:
        return RetryDecision(
            False, f"{kind} is not a retryable transport failure", attempt_number=attempt_number
        )
    if attempt_number >= policy.max_attempts:
        return RetryDecision(
            False,
            f"attempt {attempt_number} is the last the policy allows",
            attempt_number=attempt_number,
        )
    return RetryDecision(
        True,
        f"{kind} is retryable",
        delay_seconds=backoff_seconds(policy, attempt_number),
        attempt_number=attempt_number + 1,
    )


def _as_failure_kind(name: str) -> FailureKind | None:
    try:
        return FailureKind(name)
    except ValueError:
        return None


@dataclass(slots=True)
class _Accumulator:
    """Mutable bookkeeping while a plan is built."""

    items: list[PlanItem] = field(default_factory=list)
    seen: set[tuple[str, str, str, int]] = field(default_factory=set)


def build_plan(
    *,
    campaign_id: str,
    tasks: Sequence[TaskSpec],
    policy: RetryPolicy,
    seed: int,
    already_accepted: Iterable[SampleKey] = (),
    eligible: Callable[[TaskSpec], bool] | None = None,
) -> RunPlan:
    """Build the whole campaign's dispatch plan.

    ``already_accepted`` names samples an earlier run accepted. They are skipped, not
    re-dispatched: resuming must never produce a second accepted sample for one
    (campaign, endpoint, task, repeat).

    The interleaving is a seeded permutation, so a campaign's endpoints are visited in
    a reproducible order that is not simply "sorted by alias".
    """
    accepted = {key.as_tuple() for key in already_accepted}
    generator = random.Random(f"{campaign_id}:{seed}".encode())

    ordered = list(tasks)
    generator.shuffle(ordered)

    acc = _Accumulator()
    for task in ordered:
        key = SampleKey(
            campaign_id=task.campaign_id,
            endpoint_id=task.endpoint_id,
            task_id=task_id_for(task.benchmark_id, task.item_id),
            repeat_id=task.repeat_id,
        )
        identity = key.as_tuple()
        if identity in accepted:
            acc.items.append(
                PlanItem(
                    sample_key=key,
                    attempts=(),
                    skipped=SkipReason.ALREADY_ACCEPTED,
                    skip_detail="an earlier run already accepted this sample",
                )
            )
            continue
        if identity in acc.seen:
            acc.items.append(
                PlanItem(
                    sample_key=key,
                    attempts=(),
                    skipped=SkipReason.DUPLICATE_IN_PLAN,
                    skip_detail="the same sample appears twice in the task list",
                )
            )
            continue
        acc.seen.add(identity)
        if eligible is not None and not eligible(task):
            acc.items.append(
                PlanItem(
                    sample_key=key,
                    attempts=(),
                    skipped=SkipReason.NOT_ELIGIBLE,
                    skip_detail="the manifest marks this task ineligible",
                )
            )
            continue
        acc.items.append(
            PlanItem(
                sample_key=key,
                attempts=tuple(
                    AttemptPlan(
                        sample_key=key,
                        attempt_number=attempt,
                        delay_seconds=backoff_seconds(policy, attempt - 1),
                    )
                    for attempt in range(1, policy.max_attempts + 1)
                ),
            )
        )
    return RunPlan(campaign_id=campaign_id, seed=seed, items=tuple(acc.items), policy=policy)


def stable_seed(campaign_id: str, seed: int) -> bytes:
    """The exact bytes a plan's generator is seeded from, for reproducibility checks."""
    return f"{campaign_id}:{seed}".encode()
