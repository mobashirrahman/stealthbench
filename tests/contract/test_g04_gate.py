"""G04 gate: bounded runs, no retry of task failures, no duplicate samples on resume.

Oracle here is IMPLEMENTATION_PLAN.md section G04. Each test drives the documented
pieces together -- scheduler, ledger, adapter, clock -- through ``run_campaign``
rather than asserting any one module in isolation. A gate that passes while the
wiring between modules is broken is not a gate.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from stealthbench.adapters.base import (
    AdapterResult,
    FailureKind,
    FixtureTransport,
    ProviderAdapter,
    TransportFailure,
)
from stealthbench.costs import Ledger, PriceBook
from stealthbench.runner import RunMode, RunReport, run_campaign
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
    EvaluationPayload,
    ModelRequest,
    TaskSpec,
    accepted_sample_keys,
    task_id_for,
)

pytestmark = pytest.mark.contract

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
    max_attempts=3, initial_backoff_seconds=1.0, multiplier=2.0, max_backoff_seconds=4.0
)


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
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


def fixture_adapter(*, exchange_count: int = 8) -> FixtureTransport:
    from stealthbench.adapters.base import FixtureBundle

    return FixtureTransport(
        FixtureBundle.model_validate(
            {
                "name": "gate",
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


def run_kwargs(tasks: list[TaskSpec], **overrides: object) -> dict[str, object]:
    already = overrides.pop("already_accepted", ())
    payloads: dict[str, object] = {
        "campaign_id": "c1",
        "mode": RunMode.OFFLINE,
        "limits": limits(),
        "authorization": Authorization(),
        "plan": build_plan(
            campaign_id="c1", tasks=tasks, policy=POLICY, seed=7, already_accepted=already
        ),
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


def test_a_controlled_run_cannot_exceed_its_declared_limits() -> None:
    """The gate's first sentence, executed rather than asserted per module."""
    tasks = [task(f"i{n}") for n in range(6)]
    tight = limits(max_total_cost_usd=0.001, max_requests=20)
    ledger = Ledger(limits=tight, book=BOOK)
    report = run_campaign(**run_kwargs(tasks, limits=tight, ledger=ledger))
    assert isinstance(report, RunReport)
    totals = ledger.totals()
    assert totals["requests"] <= 20
    assert Decimal(totals["spent"]) <= Decimal("0.001"), (
        "reservations bound the spend before dispatch, so a tight cap stops the run, "
        "not the invoice"
    )
    assert any("max_total_cost_usd" in failure for failure in report.failures)
    assert ledger.outstanding == 0, "every reservation settled or was never made"


def test_a_terminal_failure_is_dispatched_exactly_once_whatever_the_policy_allows() -> None:
    """Task failures are not retried as if they were transport errors."""

    class AlwaysForbidden(ProviderAdapter):
        route = "forbidden"

        def __init__(self) -> None:
            self.calls = 0

        def discover(self):  # type: ignore[no-untyped-def]
            return fixture_adapter().discover()

        def complete(self, **kwargs: object):  # type: ignore[no-untyped-def]
            self.calls += 1
            return AdapterResult(
                failure=TransportFailure(kind=FailureKind.PERMISSION, detail="403 no")
            )

    adapter = AlwaysForbidden()
    report = run_campaign(**run_kwargs([task("i0")], adapter=adapter))
    assert isinstance(report, RunReport)
    assert adapter.calls == 1, "max_attempts=3 must not matter for a 403"
    assert report.accepted_sample_count == 0


def test_resume_cannot_silently_create_a_second_accepted_sample() -> None:
    """Run to completion, resume with the accepted keys, and nothing goes out again."""
    tasks = [task(f"i{n}") for n in range(3)]
    first_adapter = fixture_adapter()
    first = run_campaign(**run_kwargs(tasks, adapter=first_adapter))
    assert isinstance(first, RunReport)
    assert first.accepted_sample_count == 3

    accepted = [r.sample_key for r in first.results if r.is_accepted_sample]
    assert len(accepted) == 3
    resumed_adapter = fixture_adapter()
    resumed = run_campaign(
        **run_kwargs(
            tasks,
            adapter=resumed_adapter,
            already_accepted=accepted,
            ledger=Ledger(limits=limits(), book=BOOK),
        )
    )
    assert isinstance(resumed, RunReport)
    assert resumed_adapter.call_count() == 0, "an accepted sample is never re-dispatched"
    assert len(resumed.plan.skipped) == 3, "resumed samples are skipped, not eligible"
    assert resumed.accepted_sample_count == 0
    assert accepted_sample_keys(resumed.results) == set()
