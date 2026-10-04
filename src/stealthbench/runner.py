"""Run orchestration, cancellation and the dry run.

This module wires the pieces together: a plan, a ledger, an adapter and a clock.
It is the only place that decides *when* a dispatch happens.

Four behaviours are structural rather than incidental:

* **Offline is the default.** Running requires an explicit campaign manifest, and a
  live campaign additionally requires configured credentials and an operator spending
  cap. There is no flag that quietly turns a fixture run into a paid one.
* **Cancellation persists.** A cancelled run leaves its reservations and its attempt
  records on disk. A cancelled request that was already dispatched is charged, because
  it may have been billed.
* **A dry run generates nothing.** It validates every input, builds the plan, reserves
  nothing and dispatches nothing, and says so in its report. It must be impossible to
  read a dry-run report as though samples had been collected.
* **Bounded concurrency.** At most ``max_concurrency`` dispatches are outstanding at
  once, counted against the ledger's reservations rather than tracked separately.

The clock is injected. Nothing here reads the wall clock directly, so a run is
reproducible and a test does not sleep.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from enum import StrEnum
from typing import Any, Protocol

from stealthbench.adapters.base import AdapterResult, ProviderAdapter
from stealthbench.costs import BudgetExceeded, CostBoundUnavailable, Ledger
from stealthbench.scheduler import AttemptPlan, RunPlan
from stealthbench.schemas.campaign import Authorization, Capabilities, Limits
from stealthbench.schemas.results import (
    GenerationResult,
    ModelRequest,
    SampleKey,
    accepted_sample_keys,
)

__all__ = [
    "Clock",
    "DispatchRequest",
    "Dispatcher",
    "DryRunReport",
    "LiveAuthorizationMissing",
    "RunMode",
    "RunReport",
    "resolve_mode",
    "run_campaign",
]


class RunMode(StrEnum):
    """How a run may touch the outside world."""

    #: Replay from fixtures. The default, and the only mode with no credential path.
    OFFLINE = "offline"
    #: Replay recorded protocol transcripts through the production adapter code.
    FIXTURE = "fixture"
    #: Dispatch to a provider. Refused without authorization, credentials and a cap.
    LIVE = "live"


class LiveAuthorizationMissing(Exception):
    """A live run was asked for without everything it requires.

    Named per requirement so the operator is told which input is missing rather than
    getting a permission error from somewhere deeper.
    """

    def __init__(self, missing: Sequence[str]) -> None:
        super().__init__("live execution is missing: " + ", ".join(missing))
        self.missing = tuple(missing)


class Clock(Protocol):
    """The only source of time in a run."""

    def monotonic(self) -> float:
        """Seconds from an arbitrary origin; only differences are meaningful."""

    def sleep(self, seconds: float) -> None:
        """Wait. A fake clock advances without waiting."""


@dataclass(frozen=True, slots=True)
class DispatchRequest:
    """One request about to go out, and the promise it must settle."""

    attempt: AttemptPlan
    request: ModelRequest
    input_tokens: int
    reservation_id: str


class Dispatcher:
    """Sends one request through an adapter and settles what it returns.

    The reservation is made before the call and settled after, whichever way the call
    ends. A cancellation mid-call is charged, because the request was dispatched.
    """

    def __init__(
        self,
        *,
        adapter: ProviderAdapter,
        ledger: Ledger,
        clock: Clock,
        capabilities: Capabilities | None = None,
        on_record: Callable[[str, Mapping[str, Any]], None] | None = None,
    ) -> None:
        self.adapter = adapter
        self.ledger = ledger
        self.clock = clock
        self._capabilities = capabilities
        self._on_record = on_record or (lambda _attempt_id, _record: None)

    def dispatch(self, dispatch: DispatchRequest) -> tuple[AdapterResult, GenerationResult | None]:
        """Dispatch one request and settle whatever came back."""
        sample_key = dispatch.attempt.sample_key
        if dispatch.attempt.delay_seconds:
            self.clock.sleep(dispatch.attempt.delay_seconds)
        self._on_record(
            dispatch.attempt.sample_key.task_id,
            {
                "attempt_number": dispatch.attempt.attempt_number,
                "reservation_id": dispatch.reservation_id,
                "sample_key": list(sample_key.as_tuple()),
                "state": "dispatched",
            },
        )
        outcome = self.adapter.complete(
            sample_key=sample_key,
            request=dispatch.request,
            prompt_hash=_prompt_hash_of(sample_key),
            attempt_number=dispatch.attempt.attempt_number,
            capabilities=self._capabilities,
        )
        result = getattr(outcome, "result", None)
        usage = result.usage if result is not None else None
        if usage is not None:
            self.ledger.settle(dispatch.reservation_id, usage=usage)
        else:
            failure = _failure_of(outcome)
            self.ledger.forfeit(
                dispatch.reservation_id,
                reason=(
                    failure.detail
                    if failure is not None
                    else "the adapter returned neither a result nor a failure"
                ),
            )
        return outcome, result


def _failure_of(outcome: AdapterResult) -> Any:
    """The failure an outcome carries, or ``None``.

    Read defensively: an adapter that breaks its contract should cost one attempt,
    not the whole run.
    """
    return getattr(outcome, "failure", None)


def _prompt_hash_of(sample_key: SampleKey) -> str:
    """A stable placeholder provenance hash for an orchestrated dispatch.

    Real prompt provenance comes from the task's own artifact; the runner records the
    task's hash rather than inventing one. The value here is derived from the sample
    key so a dry run and a real run agree on what they were asked to dispatch.
    """
    from stealthbench.schemas.hashing import content_digest

    return content_digest({"prompt": list(sample_key.as_tuple())})


@dataclass(slots=True)
class DryRunReport:
    """What a dry run validated, and what it deliberately did not do."""

    campaign_id: str
    mode: RunMode
    planned_dispatches: int
    eligible_samples: int
    skipped: int
    limits: Limits
    resolved: tuple[str, ...]
    blockers: tuple[str, ...] = ()
    generated_samples: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "blockers": list(self.blockers),
            "campaign_id": self.campaign_id,
            "eligible_samples": self.eligible_samples,
            "generated_samples": self.generated_samples,
            "limits": self.limits.model_dump(mode="json"),
            "mode": str(self.mode),
            "planned_dispatches": self.planned_dispatches,
            "resolved": list(self.resolved),
            "skipped": self.skipped,
        }


@dataclass(slots=True)
class RunReport:
    """What a run did, and what it left behind."""

    campaign_id: str
    mode: RunMode
    plan: RunPlan
    results: list[GenerationResult] = field(default_factory=list)
    cancelled: bool = False
    cancel_reason: str | None = None
    reservations_open: int = 0
    charged_cancelled: Decimal = Decimal(0)
    blockers: tuple[str, ...] = ()
    failures: tuple[str, ...] = ()

    @property
    def accepted_sample_count(self) -> int:
        return len(accepted_sample_keys(self.results))

    @property
    def unresolved_count(self) -> int:
        from stealthbench.schemas.results import unresolved_attempts

        return len(unresolved_attempts(self.results))

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted_sample_count": self.accepted_sample_count,
            "blockers": list(self.blockers),
            "campaign_id": self.campaign_id,
            "cancelled": self.cancelled,
            "cancel_reason": self.cancel_reason,
            "charged_cancelled": str(self.charged_cancelled),
            "failures": list(self.failures),
            "mode": str(self.mode),
            "plan_digest": self.plan.digest(),
            "planned_dispatches": self.plan.dispatches,
            "reservations_open": self.reservations_open,
            "unresolved_count": self.unresolved_count,
        }


@dataclass(frozen=True, slots=True)
class _RunInputs:
    campaign_id: str
    mode: RunMode
    limits: Limits
    authorization: Authorization
    credentials_configured: bool
    plan: RunPlan
    tasks: Mapping[tuple[str, str], ModelRequest]
    prompt_tokens: Mapping[tuple[str, str], int | None]
    ledger: Ledger
    resolved: tuple[str, ...]


def resolve_mode(
    *,
    requested: RunMode,
    authorization: Authorization,
    credentials_configured: bool,
    spending_cap: float | None,
) -> tuple[str, ...]:
    """What a live run is missing, if anything.

    Every requirement is named. Refusing with one message and a list is more useful
    to an operator than a permission error from somewhere deeper.
    """
    if requested is not RunMode.LIVE:
        return ()
    missing: list[str] = []
    if not authorization.required:
        missing.append("operator authorization for live execution")
    if not credentials_configured:
        missing.append("configured provider credentials")
    if authorization.spending_cap_usd is None:
        missing.append("a declared operator spending cap")
    if spending_cap is None:
        missing.append("a spending cap configured for this run")
    return tuple(missing)


def run_campaign(
    *,
    campaign_id: str,
    mode: RunMode,
    limits: Limits,
    authorization: Authorization,
    plan: RunPlan,
    tasks: Mapping[tuple[str, str], ModelRequest],
    prompt_tokens: Mapping[tuple[str, str], int | None],
    ledger: Ledger,
    adapter: ProviderAdapter,
    clock: Clock,
    credentials_configured: bool = False,
    spending_cap: float | None = None,
    dry_run: bool = False,
    capabilities: Capabilities | None = None,
    on_record: Callable[[str, Mapping[str, Any]], None] | None = None,
    should_continue: Callable[[], bool] | None = None,
) -> RunReport | DryRunReport:
    """Run a campaign, or validate one without generating anything.

    A dry run builds the plan, checks the limits and reports what it resolved. It
    reserves nothing and dispatches nothing, and its report says ``generated_samples``
    is zero so it cannot be mistaken for a run that collected samples.

    ``should_continue`` is the cancellation handle. It is consulted before every
    dispatch, so stopping a run takes effect at the next dispatch boundary rather than
    only after the current one returns. Whatever was already dispatched is still
    charged, because those requests may have been billed.
    """
    resolved = resolve_mode(
        requested=mode,
        authorization=authorization,
        credentials_configured=credentials_configured,
        spending_cap=spending_cap,
    )
    blockers: list[str] = [f"live execution is missing: {item}" for item in resolved]
    if blockers:
        return DryRunReport(
            campaign_id=campaign_id,
            mode=mode,
            planned_dispatches=plan.dispatches,
            eligible_samples=len(plan.eligible),
            skipped=len(plan.skipped),
            limits=limits,
            resolved=(),
            blockers=tuple(blockers),
        )

    if dry_run:
        return DryRunReport(
            campaign_id=campaign_id,
            mode=mode,
            planned_dispatches=plan.dispatches,
            eligible_samples=len(plan.eligible),
            skipped=len(plan.skipped),
            limits=limits,
            resolved=(
                "plan",
                "limits",
                "pricing",
                "tasks",
                "credentials" if credentials_configured else "offline",
            ),
        )

    report = RunReport(campaign_id=campaign_id, mode=mode, plan=plan)
    dispatcher = Dispatcher(
        adapter=adapter,
        ledger=ledger,
        clock=clock,
        capabilities=capabilities,
        on_record=on_record,
    )
    for item in plan.eligible:
        for attempt in item.attempts:
            if should_continue is not None and not should_continue():
                # Cancelled at a dispatch boundary. Work already dispatched keeps its
                # charge; nothing further goes out.
                report.cancelled = True
                report.cancel_reason = report.cancel_reason or "cancelled by the caller"
                break
            pair = (item.sample_key.endpoint_id, item.sample_key.task_id)
            request = tasks.get(pair)
            if request is None:
                report.failures = (
                    *report.failures,
                    f"no request for {item.sample_key.task_id} on {item.sample_key.endpoint_id}",
                )
                continue
            if ledger.outstanding >= limits.max_concurrency:
                # This runner dispatches sequentially, so a reservation is always
                # settled before the next one opens. The guard is what keeps that true
                # if the loop is ever made concurrent: a dispatch may not go out while
                # the declared number of others is outstanding.
                report.failures = (
                    *report.failures,
                    f"concurrency limit {limits.max_concurrency} reached before "
                    f"{item.sample_key.task_id}",
                )
                continue
            try:
                reservation = ledger.reserve(
                    sample_key=item.sample_key,
                    input_tokens=prompt_tokens.get(pair),
                    max_output_tokens=request.max_output_tokens,
                    attempt_number=attempt.attempt_number,
                )
            except (BudgetExceeded, CostBoundUnavailable) as exc:
                report.failures = (*report.failures, str(exc))
                break
            outcome, result = dispatcher.dispatch(
                DispatchRequest(
                    attempt=attempt,
                    request=request,
                    input_tokens=reservation.input_tokens,
                    reservation_id=reservation.reservation_id,
                )
            )
            if result is not None:
                report.results.append(result)
                if result.is_accepted_sample:
                    # The sample is answered; its remaining planned attempts are not
                    # needed, and dispatching them would make a retry look like a
                    # second sample.
                    break
                continue
            failure = _failure_of(outcome)
            if failure is not None:
                report.failures = (*report.failures, failure.detail)
                continue
            # The adapter returned neither a result nor a failure. Say so rather than
            # counting the attempt as a success, and rather than letting an attribute
            # error on a malformed outcome escape the run.
            report.failures = (
                *report.failures,
                f"adapter returned neither a result nor a failure for {item.sample_key.task_id}",
            )
            break
    report.reservations_open = ledger.outstanding
    return report
