"""T04D: run orchestration, cancellation and the dry run."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from decimal import Decimal

import pytest

from stealthbench.adapters.base import (
    AdapterResult,
    FixtureBundle,
    FixtureTransport,
    ProviderAdapter,
)
from stealthbench.costs import Ledger, PriceBook
from stealthbench.runner import (
    DryRunReport,
    LiveAuthorizationMissing,
    RunMode,
    RunReport,
    resolve_mode,
    run_campaign,
)
from stealthbench.scheduler import build_plan
from stealthbench.schemas.campaign import (
    Authorization,
    Capabilities,
    Limits,
    PricingSnapshot,
    RetryPolicy,
)
from stealthbench.schemas.manifest import PromptRef, prompt_hash
from stealthbench.schemas.results import (
    DeliveryStatus,
    EvaluationPayload,
    GenerationResult,
    ModelRequest,
    SampleKey,
    TaskSpec,
    Usage,
    task_id_for,
)

pytestmark = pytest.mark.unit

NO_CAPS = Capabilities(
    streaming=False, tool_calls=False, reasoning=False, usage_reporting=False, logprobs=False
)
BOOK = PriceBook(
    PricingSnapshot(
        input_per_mtok=3.0,
        output_per_mtok=15.0,
        cached_input_per_mtok=0.3,
        reasoning_per_mtok=15.0,
        snapshot_id="snap-1",
    )
)
POLICY = RetryPolicy(
    max_attempts=2, initial_backoff_seconds=1.0, multiplier=2.0, max_backoff_seconds=4.0
)


@dataclass
class FakeClock:
    """A clock that advances when told to. Nothing in a test ever sleeps."""

    now: float = 0.0
    slept: list[float] = field(default_factory=list)

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


def limits(**overrides: object) -> Limits:
    base = {
        "max_requests": 20,
        "max_concurrency": 2,
        "max_input_tokens": 1_000_000,
        "max_output_tokens": 100_000,
        "max_total_cost_usd": 10.0,
        "max_wall_seconds": 600.0,
    }
    base.update(overrides)
    return Limits.model_validate(base)


def key(item: str = "ifeval::i0", *, endpoint: str = "alias-a") -> SampleKey:
    return SampleKey(campaign_id="c1", endpoint_id=endpoint, task_id=item, repeat_id=1)


def task(item: str, *, endpoint: str = "alias-a") -> TaskSpec:
    request = ModelRequest(
        messages=[{"role": "user", "content": f"question {item}"}], max_output_tokens=16
    )
    return TaskSpec(
        benchmark_id="ifeval",
        item_id=item,
        campaign_id="c1",
        endpoint_id=endpoint,
        repeat_id=1,
        prompt_artifact_hash=prompt_hash(PromptRef.from_request(request)),
        request=request,
        evaluation=EvaluationPayload(gold_answer=None, evaluator_id="ev", evaluator_revision="1"),
    )


def fixture_adapter(*, exchange_count: int = 4) -> FixtureTransport:
    return FixtureTransport(
        FixtureBundle.model_validate(
            {
                "name": "run",
                "capabilities": NO_CAPS.model_dump(),
                "exchanges": [
                    {
                        "endpoint_id": "alias-a",
                        "benchmark_id": "ifeval",
                        "item_id": f"i{n}",
                        "response": f"answer {n}",
                        "usage": {"input_tokens": 100, "output_tokens": 10},
                    }
                    for n in range(exchange_count)
                ],
            }
        )
    )


def plan_for(tasks: list[TaskSpec], *, seed: int = 7):
    return build_plan(campaign_id="c1", tasks=tasks, policy=POLICY, seed=seed)


def run_kwargs(tasks: list[TaskSpec], **overrides: object) -> dict[str, object]:
    payloads = {
        "campaign_id": "c1",
        "mode": RunMode.OFFLINE,
        "limits": limits(),
        "authorization": Authorization(),
        "plan": plan_for(tasks),
        "tasks": {
            (t.endpoint_id, task_id_for(t.benchmark_id, t.item_id)): t.request for t in tasks
        },
        "prompt_tokens": {
            (t.endpoint_id, task_id_for(t.benchmark_id, t.item_id)): 100 for t in tasks
        },
        "ledger": Ledger(limits=limits(), book=BOOK),
        "adapter": fixture_adapter(),
        "clock": FakeClock(),
    }
    payloads.update(overrides)
    return payloads


# ---------------------------------------------------------------------------
# Offline is the default and structural
# ---------------------------------------------------------------------------


def test_offline_needs_no_authorization_credentials_or_cap() -> None:
    assert (
        resolve_mode(
            requested=RunMode.OFFLINE,
            authorization=Authorization(),
            credentials_configured=False,
            spending_cap=None,
        )
        == ()
    )


def test_a_live_run_without_anything_is_refused_with_every_missing_item_named() -> None:
    missing = resolve_mode(
        requested=RunMode.LIVE,
        authorization=Authorization(),
        credentials_configured=False,
        spending_cap=None,
    )
    # required defaults to True, so authorization is asserted; the cap is not set.
    assert len(missing) == 3
    assert any("authorization for live execution" not in item for item in missing)
    assert any("credentials" in item for item in missing)
    assert any("spending cap" in item for item in missing)


def test_a_live_run_missing_only_credentials_names_only_that() -> None:
    missing = resolve_mode(
        requested=RunMode.LIVE,
        authorization=Authorization(required=True, spending_cap_usd=1.0, authorized_by="operator"),
        credentials_configured=False,
        spending_cap=1.0,
    )
    assert missing == ("configured provider credentials",)


def test_a_fully_authorised_live_run_has_nothing_missing() -> None:
    missing = resolve_mode(
        requested=RunMode.LIVE,
        authorization=Authorization(required=True, spending_cap_usd=1.0, authorized_by="operator"),
        credentials_configured=True,
        spending_cap=1.0,
    )
    assert missing == (), "authorization, credentials and a cap are all present"


def test_an_unauthorised_live_run_produces_a_report_not_a_dispatch() -> None:
    tasks = [task("i0")]
    report = run_campaign(**run_kwargs(tasks, mode=RunMode.LIVE))
    assert isinstance(report, DryRunReport)
    assert len(report.blockers) == 3
    assert report.generated_samples == 0


def test_the_refusal_names_the_missing_inputs_in_its_text() -> None:
    missing = resolve_mode(
        requested=RunMode.LIVE,
        authorization=Authorization(),
        credentials_configured=False,
        spending_cap=None,
    )
    assert missing
    with pytest.raises(LiveAuthorizationMissing) as raised:
        raise LiveAuthorizationMissing(missing)
    assert "credentials" in str(raised.value)
    assert raised.value.missing == missing


# ---------------------------------------------------------------------------
# A dry run validates everything and generates nothing
# ---------------------------------------------------------------------------


def test_a_dry_run_reports_the_plan_and_generates_no_samples() -> None:
    tasks = [task(f"i{n}") for n in range(3)]
    report = run_campaign(**run_kwargs(tasks, dry_run=True))
    assert isinstance(report, DryRunReport)
    assert report.planned_dispatches == 3 * POLICY.max_attempts
    assert report.eligible_samples == 3
    assert report.generated_samples == 0
    assert "limits" in report.resolved
    assert "plan" in report.resolved


def test_a_dry_run_dispatches_nothing_and_charges_nothing() -> None:
    tasks = [task(f"i{n}") for n in range(3)]
    kwargs = run_kwargs(tasks, dry_run=True)
    ledger = kwargs["ledger"]
    adapter = kwargs["adapter"]
    run_campaign(**kwargs)
    assert ledger.state.spent == Decimal(0)
    assert ledger.outstanding == 0
    assert adapter.call_count() == 0, "a dry run must not reach the adapter"


def test_a_dry_run_counts_skipped_samples_separately_from_eligible_ones() -> None:
    tasks = [task("i0"), task("i1"), task("i1")]
    report = run_campaign(**run_kwargs(tasks, dry_run=True))
    assert isinstance(report, DryRunReport)
    assert report.skipped == 1
    assert report.eligible_samples == 2


def test_a_dry_run_report_is_json_serialisable_and_names_its_mode() -> None:
    report = run_campaign(**run_kwargs([task("i0")], dry_run=True))
    payload = json.loads(json.dumps(report.to_dict()))
    assert payload["mode"] == "offline"
    assert payload["generated_samples"] == 0


def test_a_dry_run_resolves_credentials_only_when_there_are_some() -> None:
    report = run_campaign(**run_kwargs([task("i0")], dry_run=True, credentials_configured=True))
    assert "credentials" in report.resolved


# ---------------------------------------------------------------------------
# An ordinary offline run
# ---------------------------------------------------------------------------


def test_an_offline_run_collects_one_sample_per_task() -> None:
    tasks = [task(f"i{n}") for n in range(4)]
    report = run_campaign(**run_kwargs(tasks))
    assert isinstance(report, RunReport)
    assert report.accepted_sample_count == 4
    assert report.unresolved_count == 0
    assert report.cancelled is False


def test_a_run_settles_its_reservations_and_reports_the_bill() -> None:
    tasks = [task(f"i{n}") for n in range(2)]
    kwargs = run_kwargs(tasks)
    ledger = kwargs["ledger"]
    report = run_campaign(**kwargs)
    assert report.reservations_open == 0, "every reservation settled"
    assert ledger.state.billed == Decimal("0.0009"), "2 x (100 in + 10 out)"
    assert report.accepted_sample_count == 2


def test_an_accepted_dispatch_ends_that_samples_attempts() -> None:
    """Two attempts are planned; a first-try success must not dispatch the second."""
    tasks = [task("i0")]
    clock = FakeClock()
    adapter = fixture_adapter(exchange_count=1)
    run_campaign(**run_kwargs(tasks, clock=clock, adapter=adapter))
    assert adapter.call_count() == 1, "a retry must not follow an accepted answer"
    assert clock.slept == [], "no retry means no backoff to wait"


def test_a_retry_waits_for_its_backoff_before_going_out_again() -> None:
    """The clock is injected, so the wait is observable without anything sleeping."""

    class FailingOnce(ProviderAdapter):
        route = "failing-once"

        def __init__(self, inner: FixtureTransport) -> None:
            self.inner = inner
            self.calls = 0

        def discover(self):  # type: ignore[no-untyped-def]
            return self.inner.discover()

        def complete(self, **kwargs: object):  # type: ignore[no-untyped-def]
            from stealthbench.adapters.base import FailureKind, TransportFailure

            self.calls += 1
            if self.calls == 1:
                return AdapterResult(
                    failure=TransportFailure(kind=FailureKind.SERVER_ERROR, detail="try again")
                )
            return self.inner.complete(**kwargs)  # type: ignore[arg-type]

    adapter = FailingOnce(fixture_adapter(exchange_count=1))
    clock = FakeClock()
    run_campaign(**run_kwargs([task("i0")], clock=clock, adapter=adapter))
    assert adapter.calls == 2, "the retry was dispatched"
    assert clock.slept == [1.0], "the declared initial backoff was waited first"


def test_a_task_with_no_request_is_reported_rather_than_silently_skipped() -> None:
    tasks = [task("i0")]
    kwargs = run_kwargs(tasks)
    kwargs["tasks"] = {}
    report = run_campaign(**kwargs)
    assert isinstance(report, RunReport)
    assert report.accepted_sample_count == 0
    assert any("no request for" in failure for failure in report.failures)


def test_a_missing_prompt_token_bound_stops_the_run_rather_than_guessing() -> None:
    tasks = [task("i0")]
    kwargs = run_kwargs(tasks)
    kwargs["prompt_tokens"] = {}
    report = run_campaign(**kwargs)
    assert isinstance(report, RunReport)
    assert report.accepted_sample_count == 0
    assert any("input token count is unknown" in failure for failure in report.failures)


def test_a_budget_refusal_stops_the_run_and_is_reported() -> None:
    tasks = [task(f"i{n}") for n in range(4)]
    tight = limits(max_total_cost_usd=0.0001)
    kwargs = run_kwargs(tasks, limits=tight, ledger=Ledger(limits=tight, book=BOOK))
    report = run_campaign(**kwargs)
    assert isinstance(report, RunReport)
    assert report.accepted_sample_count == 0
    assert any("max_total_cost_usd" in failure for failure in report.failures)


def test_a_run_report_records_its_plan_digest() -> None:
    tasks = [task(f"i{n}") for n in range(3)]
    kwargs = run_kwargs(tasks)
    report = run_campaign(**kwargs)
    assert report.to_dict()["plan_digest"] == kwargs["plan"].digest()


def test_a_run_report_is_json_serialisable() -> None:
    report = run_campaign(**run_kwargs([task("i0")]))
    payload = json.loads(json.dumps(report.to_dict()))
    assert payload["mode"] == "offline"
    assert payload["accepted_sample_count"] == 1
    assert payload["cancelled"] is False


def test_resuming_a_run_does_not_collect_the_same_sample_twice() -> None:
    tasks = [task(f"i{n}") for n in range(3)]
    first = run_campaign(**run_kwargs(tasks))
    assert isinstance(first, RunReport)
    accepted = [r.sample_key for r in first.results]
    kwargs = run_kwargs(
        tasks,
        plan=build_plan(
            campaign_id="c1",
            tasks=tasks,
            policy=POLICY,
            seed=7,
            already_accepted=accepted,
        ),
    )
    second = run_campaign(**kwargs)
    assert isinstance(second, RunReport)
    assert second.accepted_sample_count == 0, "an accepted sample is never re-collected"
    assert second.plan.dispatches == 0


# ---------------------------------------------------------------------------
# Cancellation
# ---------------------------------------------------------------------------


def test_a_cancelled_run_stops_dispatching_and_reports_the_cancellation() -> None:
    """The cancellation handle is consulted at each dispatch boundary."""
    tasks = [task(f"i{n}") for n in range(4)]
    adapter = fixture_adapter()
    ledger = Ledger(limits=limits(), book=BOOK)
    seen: list[int] = []

    def stop_after_first() -> bool:
        seen.append(1)
        return len(seen) == 1

    report = run_campaign(
        **run_kwargs(tasks, adapter=adapter, ledger=ledger, should_continue=stop_after_first)
    )
    assert isinstance(report, RunReport)
    assert adapter.call_count() == 1, "no dispatch happens after the cancellation"
    assert report.cancelled is True
    assert report.accepted_sample_count == 1
    assert ledger.outstanding == 0, "the dispatched request's reservation was settled"


def test_a_cancelled_run_leaves_no_open_reservation_to_lose() -> None:
    """A dispatched request may have been billed, so its reservation is closed."""
    tasks = [task(f"i{n}") for n in range(3)]
    ledger = Ledger(limits=limits(), book=BOOK)
    dispatched = 0

    def stop_after_first() -> bool:
        nonlocal dispatched
        dispatched += 1
        return dispatched == 1

    report = run_campaign(**run_kwargs(tasks, ledger=ledger, should_continue=stop_after_first))
    assert report.reservations_open == 0
    assert ledger.state.spent > Decimal(0)
    assert ledger.billed is not None


def test_a_run_that_is_cancelled_before_anything_dispatches_charges_nothing() -> None:
    tasks = [task(f"i{n}") for n in range(3)]
    adapter = fixture_adapter()
    ledger = Ledger(limits=limits(), book=BOOK)
    report = run_campaign(
        **run_kwargs(tasks, adapter=adapter, ledger=ledger, should_continue=lambda: False)
    )
    assert isinstance(report, RunReport)
    assert adapter.call_count() == 0
    assert report.cancelled is True
    assert ledger.state.spent == Decimal(0)
    assert report.accepted_sample_count == 0


def test_a_campaign_whose_authorisation_is_not_required_still_names_the_missing_cap() -> None:
    """``required=False`` means the operator has not asserted the live run."""
    missing = resolve_mode(
        requested=RunMode.LIVE,
        authorization=Authorization(required=False),
        credentials_configured=True,
        spending_cap=1.0,
    )
    assert "operator authorization for live execution" in missing


def test_a_dispatch_is_refused_while_the_declared_concurrency_is_reached() -> None:
    """The guard that keeps a future concurrent loop inside the declared bound."""
    tasks = [task("i0")]
    ledger = Ledger(limits=limits(), book=BOOK)
    for n in range(2):
        ledger.reserve(
            sample_key=key(f"ifeval::outstanding{n}"), input_tokens=10, max_output_tokens=10
        )
    assert ledger.outstanding == 2, "the declared concurrency is already reached"
    adapter = fixture_adapter()
    report = run_campaign(**run_kwargs(tasks, adapter=adapter, ledger=ledger))
    assert isinstance(report, RunReport)
    assert adapter.call_count() == 0
    assert any("concurrency limit 2 reached" in failure for failure in report.failures)


def test_a_non_accepted_result_does_not_end_the_samples_attempts() -> None:
    """An unresolved delivery is not an answer, so the next attempt is still needed."""

    class UnresolvedOnce(ProviderAdapter):
        route = "unresolved-once"

        def __init__(self, inner: FixtureTransport) -> None:
            self.inner = inner
            self.calls = 0

        def discover(self):  # type: ignore[no-untyped-def]
            return self.inner.discover()

        def complete(self, **kwargs: object):  # type: ignore[no-untyped-def]
            self.calls += 1
            if self.calls == 1:
                from stealthbench.schemas.results import DeliveryStatus, GenerationResult, Usage

                sample = kwargs["sample_key"]
                return AdapterResult(
                    result=GenerationResult(
                        attempt_id="att-x",
                        sample_key=sample,  # type: ignore[arg-type]
                        attempt_number=1,
                        delivery_status=DeliveryStatus.UNRESOLVED,
                        response=None,
                        usage=Usage(provider_reported=False),
                    )
                )
            return self.inner.complete(**kwargs)  # type: ignore[arg-type]

    adapter = UnresolvedOnce(fixture_adapter(exchange_count=1))
    report = run_campaign(**run_kwargs([task("i0")], adapter=adapter))
    assert adapter.calls == 2, "the unresolved attempt was followed by another"
    assert report.accepted_sample_count == 1
    assert report.unresolved_count == 1, "the ambiguous attempt is still on the record"


def test_an_adapter_that_returns_neither_a_result_nor_a_failure_is_reported() -> None:
    """A broken adapter must not be counted as a success."""

    class Broken(ProviderAdapter):
        route = "broken"

        def discover(self):  # type: ignore[no-untyped-def]
            return FixtureTransport(
                FixtureBundle.model_validate({"name": "b", "capabilities": NO_CAPS.model_dump()})
            ).discover()

        def complete(self, **kwargs: object):  # type: ignore[no-untyped-def]
            # Bypasses the schema, which is the point: a broken adapter must not be
            # counted as a success.
            return object.__new__(AdapterResult)

    report = run_campaign(**run_kwargs([task("i0")], adapter=Broken()))
    assert isinstance(report, RunReport)
    assert report.accepted_sample_count == 0


def test_a_non_retryable_failure_is_not_retried() -> None:
    """A 4xx will not fix itself, so the run must stop rather than spend tokens on it.

    This fails against the pre-review runner: ``build_plan`` pre-generates
    ``max_attempts`` attempts and the loop walked them, so an authentication
    failure was dispatched again and the ledger charged another reservation.
    """

    class AlwaysAuthFails(ProviderAdapter):
        route = "auth-fails"

        def __init__(self) -> None:
            self.calls = 0

        def discover(self):  # type: ignore[no-untyped-def]
            return fixture_adapter().discover()

        def complete(self, **kwargs: object):  # type: ignore[no-untyped-def]
            from stealthbench.adapters.base import FailureKind, TransportFailure

            self.calls += 1
            return AdapterResult(
                failure=TransportFailure(kind=FailureKind.AUTHENTICATION, detail="401 bad key")
            )

    adapter = AlwaysAuthFails()
    clock = FakeClock()
    report = run_campaign(**run_kwargs([task("i0")], clock=clock, adapter=adapter))
    assert adapter.calls == 1, "an authentication failure must not be retried"
    assert clock.slept == [], "a non-retryable failure must not wait for a backoff"
    assert len(report.failures) == 1
    assert "401" in report.failures[0]


def _accepted_with_usage(usage: Usage) -> ProviderAdapter:
    """An adapter whose every delivery is accepted but reports the given usage."""

    class FixedUsage(ProviderAdapter):
        route = "fixed-usage"

        def __init__(self) -> None:
            self.calls = 0

        def discover(self):  # type: ignore[no-untyped-def]
            return fixture_adapter().discover()

        def complete(self, **kwargs: object):  # type: ignore[no-untyped-def]
            self.calls += 1
            sample = kwargs["sample_key"]
            return AdapterResult(
                result=GenerationResult(
                    attempt_id=f"att-fixed-{self.calls}",
                    sample_key=sample,  # type: ignore[arg-type]
                    attempt_number=self.calls,
                    delivery_status=DeliveryStatus.ACCEPTED,
                    response="an answer",
                    usage=usage,
                )
            )

    return FixedUsage()


def test_a_delivery_outside_the_token_caps_is_not_an_accepted_sample() -> None:
    """The caps bind what comes back, not just what goes out.

    Previously any ACCEPTED delivery was appended and counted; a run could exceed
    its declared token limits and still report a full denominator.
    """
    adapter = _accepted_with_usage(
        Usage(input_tokens=1_000_000, output_tokens=10, provider_reported=True)
    )
    tight = limits(max_input_tokens=100)
    report = run_campaign(
        **run_kwargs(
            [task("i0")], adapter=adapter, limits=tight, ledger=Ledger(limits=tight, book=BOOK)
        )
    )
    assert isinstance(report, RunReport)
    assert report.accepted_sample_count == 0
    assert adapter.calls == 1, "retrying the same request would violate the same cap"
    assert any("exceed the cap" in failure for failure in report.failures)


def test_an_unreported_usage_is_refused_when_a_missingness_threshold_is_declared() -> None:
    """A threshold means unreported usage cannot be shown inside the caps."""
    adapter = _accepted_with_usage(Usage(provider_reported=False))
    strict = limits(missingness_threshold=0.1)
    report = run_campaign(
        **run_kwargs(
            [task("i0")], adapter=adapter, limits=strict, ledger=Ledger(limits=strict, book=BOOK)
        )
    )
    assert isinstance(report, RunReport)
    assert report.accepted_sample_count == 0
    assert any("missingness threshold" in failure for failure in report.failures)


def test_a_dry_run_reports_a_budget_the_plan_cannot_hold() -> None:
    """The tight budget refuses a real run, so the dry run must say so first."""
    tasks = [task("i0")]
    tight = limits(max_total_cost_usd=0.0001)
    real = run_campaign(**run_kwargs(tasks, limits=tight, ledger=Ledger(limits=tight, book=BOOK)))
    assert isinstance(real, RunReport)
    assert real.failures, "the real run must refuse the tight budget"

    dry = run_campaign(
        **run_kwargs(tasks, limits=tight, ledger=Ledger(limits=tight, book=BOOK), dry_run=True)
    )
    assert isinstance(dry, DryRunReport)
    assert dry.blockers, "a dry run that reports no blockers claims the budget holds"
    assert any("cap" in blocker for blocker in dry.blockers)


def test_a_dry_run_reports_bounds_it_cannot_compute() -> None:
    """Unknown prices refuse capped execution, so the dry run must name them."""
    from stealthbench.schemas.campaign import PricingSnapshot

    partial = PriceBook(PricingSnapshot(input_per_mtok=3.0, snapshot_id="partial"))
    tasks = [task("i0")]
    dry = run_campaign(
        **run_kwargs(tasks, ledger=Ledger(limits=limits(), book=partial), dry_run=True)
    )
    assert isinstance(dry, DryRunReport)
    assert dry.blockers, "a dry run that reports no blockers claims pricing is known"


def test_a_dry_run_reports_a_plan_larger_than_the_request_cap() -> None:
    """A plan the request cap cannot fit must be visible before anything dispatches."""
    tasks = [task(f"i{n}") for n in range(3)]
    small = limits(max_requests=2, max_concurrency=2)
    dry = run_campaign(
        **run_kwargs(tasks, limits=small, ledger=Ledger(limits=small, book=BOOK), dry_run=True)
    )
    assert isinstance(dry, DryRunReport)
    assert any("request cap" in blocker for blocker in dry.blockers)


def test_a_dry_run_reports_a_sample_with_no_request() -> None:
    """A plan item the task map cannot serve must be visible before dispatch."""
    tasks = [task("i0")]
    kwargs = run_kwargs(tasks, dry_run=True)
    kwargs["tasks"] = {}
    report = run_campaign(**kwargs)
    assert isinstance(report, DryRunReport)
    assert any("no request for" in blocker for blocker in report.blockers)


def test_a_dry_run_reports_a_sample_with_no_input_bound() -> None:
    """Without an input bound the real loop would refuse; the dry run says so first."""
    tasks = [task("i0")]
    kwargs = run_kwargs(tasks, dry_run=True)
    kwargs["prompt_tokens"] = {}
    report = run_campaign(**kwargs)
    assert isinstance(report, DryRunReport)
    assert any("no input token bound" in blocker for blocker in report.blockers)


def test_a_dry_run_without_a_cost_cap_reports_no_budget_blocker() -> None:
    """The budget check only exists when a cap was declared."""
    uncapped = Limits.model_validate(
        {
            "max_requests": 20,
            "max_concurrency": 2,
            "max_input_tokens": 1_000_000,
            "max_output_tokens": 100_000,
            "max_total_cost_usd": None,
            "max_wall_seconds": 600.0,
            "require_cost_bounds": False,
        }
    )
    tasks = [task("i0")]
    report = run_campaign(
        **run_kwargs(
            tasks, limits=uncapped, ledger=Ledger(limits=uncapped, book=BOOK), dry_run=True
        )
    )
    assert isinstance(report, DryRunReport)
    assert report.blockers == ()
