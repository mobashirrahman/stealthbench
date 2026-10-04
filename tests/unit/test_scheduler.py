"""T04A: deterministic scheduling and bounded retries."""

from __future__ import annotations

import pytest

from stealthbench.adapters.base import FailureKind
from stealthbench.scheduler import (
    SkipReason,
    backoff_seconds,
    build_plan,
    plan_digest,
    should_retry,
    stable_seed,
)
from stealthbench.schemas.campaign import RetryPolicy
from stealthbench.schemas.manifest import PromptRef, prompt_hash
from stealthbench.schemas.results import (
    EvaluationPayload,
    ModelRequest,
    SampleKey,
    TaskSpec,
    task_id_for,
)

pytestmark = pytest.mark.unit

POLICY = RetryPolicy(
    max_attempts=3,
    initial_backoff_seconds=1.0,
    multiplier=2.0,
    max_backoff_seconds=5.0,
)


def task(
    item: str,
    *,
    endpoint: str = "alias-a",
    repeat: int = 1,
    benchmark: str = "ifeval",
    content: str = "question",
) -> TaskSpec:
    request = ModelRequest(messages=[{"role": "user", "content": content}], max_output_tokens=32)
    # The provenance hash is derived from the request itself; the schema recomputes it
    # on construction, so a placeholder would be refused.
    return TaskSpec(
        benchmark_id=benchmark,
        item_id=item,
        campaign_id="c1",
        endpoint_id=endpoint,
        repeat_id=repeat,
        prompt_artifact_hash=prompt_hash(PromptRef.from_request(request)),
        request=request,
        evaluation=EvaluationPayload(gold_answer=None, evaluator_id="ev", evaluator_revision="1"),
    )


def key(item: str, *, endpoint: str = "alias-a", repeat: int = 1) -> SampleKey:
    return SampleKey(
        campaign_id="c1",
        endpoint_id=endpoint,
        task_id=task_id_for("ifeval", item),
        repeat_id=repeat,
    )


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_the_same_seed_produces_the_same_plan() -> None:
    tasks = [task(f"i{n}") for n in range(12)]
    first = build_plan(campaign_id="c1", tasks=tasks, policy=POLICY, seed=7)
    second = build_plan(campaign_id="c1", tasks=tasks, policy=POLICY, seed=7)
    assert first.digest() == second.digest()
    assert [i.sample_key for i in first.items] == [i.sample_key for i in second.items]


def test_a_different_seed_produces_a_different_interleaving() -> None:
    tasks = [task(f"i{n}") for n in range(12)]
    a = build_plan(campaign_id="c1", tasks=tasks, policy=POLICY, seed=1)
    b = build_plan(campaign_id="c1", tasks=tasks, policy=POLICY, seed=2)
    assert a.digest() != b.digest(), "the seed must actually drive the interleaving"


def test_the_plan_is_not_merely_the_input_order() -> None:
    """A shuffle that reproduced the input order would prove nothing about seeding."""
    tasks = [task(f"i{n}") for n in range(16)]
    plan = build_plan(campaign_id="c1", tasks=tasks, policy=POLICY, seed=7)
    assert [i.sample_key.task_id for i in plan.items] != [i.sample_key.task_id for i in tasks]


def test_the_plan_digest_changes_when_the_order_changes() -> None:
    tasks = [task(f"i{n}") for n in range(6)]
    a = build_plan(campaign_id="c1", tasks=tasks, policy=POLICY, seed=7)
    b = build_plan(campaign_id="c1", tasks=tasks, policy=POLICY, seed=8)
    assert a.digest() != b.digest()


def test_the_plan_digest_is_the_free_function_too() -> None:
    plan = build_plan(campaign_id="c1", tasks=[task("i0")], policy=POLICY, seed=1)
    assert plan_digest(plan) == plan.digest()


def test_the_generator_is_seeded_from_identifiable_bytes() -> None:
    assert stable_seed("c1", 7) == b"c1:7"


def test_a_plan_with_no_tasks_is_empty_not_broken() -> None:
    plan = build_plan(campaign_id="c1", tasks=[], policy=POLICY, seed=1)
    assert plan.items == ()
    assert plan.dispatches == 0
    assert len(plan.digest()) == 64


# ---------------------------------------------------------------------------
# Resuming must not create a second accepted sample
# ---------------------------------------------------------------------------


def test_an_already_accepted_sample_is_skipped_rather_than_re_dispatched() -> None:
    tasks = [task("i0"), task("i1")]
    plan = build_plan(
        campaign_id="c1", tasks=tasks, policy=POLICY, seed=7, already_accepted=[key("i0")]
    )
    assert [i.sample_key.task_id for i in plan.skipped] == [task_id_for("ifeval", "i0")]
    assert plan.skipped[0].skipped is SkipReason.ALREADY_ACCEPTED
    assert plan.dispatches == POLICY.max_attempts


def test_a_skipped_sample_costs_no_dispatch() -> None:
    plan = build_plan(
        campaign_id="c1", tasks=[task("i0")], policy=POLICY, seed=7, already_accepted=[key("i0")]
    )
    assert plan.eligible == ()
    assert plan.dispatches == 0


def test_the_same_sample_listed_twice_is_dispatched_once() -> None:
    plan = build_plan(campaign_id="c1", tasks=[task("i0"), task("i0")], policy=POLICY, seed=7)
    dispatched = [i for i in plan.items if not i.is_skipped]
    skipped = [i for i in plan.items if i.is_skipped]
    assert len(dispatched) == 1
    assert len(skipped) == 1
    assert skipped[0].skipped is SkipReason.DUPLICATE_IN_PLAN


def test_a_repeat_is_a_distinct_sample_from_its_first_attempt() -> None:
    plan = build_plan(
        campaign_id="c1", tasks=[task("i0", repeat=1), task("i0", repeat=2)], policy=POLICY, seed=1
    )
    assert len(plan.eligible) == 2, "repeats are separate samples, not retries"


def test_an_ineligible_task_is_skipped_with_a_reason() -> None:
    plan = build_plan(
        campaign_id="c1",
        tasks=[task("i0"), task("i1")],
        policy=POLICY,
        seed=1,
        eligible=lambda spec: spec.item_id != "i1",
    )
    assert [i.sample_key.task_id for i in plan.eligible] == [task_id_for("ifeval", "i0")]
    assert plan.skipped[0].skipped is SkipReason.NOT_ELIGIBLE
    assert "ineligible" in (plan.skipped[0].skip_detail or "")


# ---------------------------------------------------------------------------
# Attempts and backoff
# ---------------------------------------------------------------------------


def test_a_plan_reserves_exactly_the_policy_attempt_count() -> None:
    plan = build_plan(campaign_id="c1", tasks=[task("i0")], policy=POLICY, seed=1)
    attempts = plan.eligible[0].attempts
    assert [a.attempt_number for a in attempts] == [1, 2, 3]


def test_backoff_grows_geometrically_and_is_capped() -> None:
    assert [backoff_seconds(POLICY, n) for n in range(1, 7)] == [1.0, 2.0, 4.0, 5.0, 5.0, 5.0]
    assert backoff_seconds(POLICY, 0) == 0.0, "before the first attempt nothing is waited"


def test_the_first_attempt_of_a_plan_waits_for_nothing() -> None:
    plan = build_plan(campaign_id="c1", tasks=[task("i0")], policy=POLICY, seed=1)
    assert plan.eligible[0].attempts[0].delay_seconds == 0.0


def test_a_plan_reports_the_backoff_it_embeds() -> None:
    plan = build_plan(campaign_id="c1", tasks=[task("i0")], policy=POLICY, seed=1)
    assert plan.eligible[0].total_delay_seconds == 0.0 + 1.0 + 2.0


# ---------------------------------------------------------------------------
# Bounded retries: only declared transport failures retry
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kind", [FailureKind.RATE_LIMIT, FailureKind.SERVER_ERROR, FailureKind.TIMEOUT]
)
def test_a_declared_transport_failure_retries(kind: FailureKind) -> None:
    decision = should_retry(failure_kind=kind, attempt_number=1, policy=POLICY)
    assert decision.retry is True
    assert decision.attempt_number == 2
    assert decision.delay_seconds == 1.0


@pytest.mark.parametrize(
    "kind",
    [
        FailureKind.AUTHENTICATION,
        FailureKind.PERMISSION,
        FailureKind.NOT_FOUND,
        FailureKind.PROTOCOL,
        FailureKind.UNSUPPORTED_SETTING,
        FailureKind.NO_FIXTURE,
        FailureKind.INTERRUPTED,
    ],
)
def test_a_non_retryable_transport_failure_is_final(kind: FailureKind) -> None:
    decision = should_retry(failure_kind=kind, attempt_number=1, policy=POLICY)
    assert decision.retry is False
    assert "not a retryable transport failure" in decision.reason


def test_a_task_failure_is_never_retried_as_if_it_were_a_transport_error() -> None:
    """The gate's explicit requirement: a wrong answer is a result, not a fault."""
    decision = should_retry(failure_kind=None, attempt_number=1, policy=POLICY)
    assert decision.retry is False
    assert decision.reason == "not a failure"


def test_retries_stop_at_the_declared_maximum() -> None:
    for attempt in range(1, POLICY.max_attempts):
        assert should_retry(
            failure_kind=FailureKind.SERVER_ERROR, attempt_number=attempt, policy=POLICY
        ).retry
    final = should_retry(
        failure_kind=FailureKind.SERVER_ERROR,
        attempt_number=POLICY.max_attempts,
        policy=POLICY,
    )
    assert final.retry is False
    assert "last the policy allows" in final.reason


def test_a_single_attempt_policy_never_retries() -> None:
    policy = RetryPolicy(
        max_attempts=1,
        initial_backoff_seconds=0.1,
        multiplier=1.0,
        max_backoff_seconds=1.0,
    )
    assert (
        should_retry(failure_kind=FailureKind.RATE_LIMIT, attempt_number=1, policy=policy).retry
        is False
    )


def test_cancellation_stops_immediately_whatever_failed() -> None:
    decision = should_retry(
        failure_kind=FailureKind.RATE_LIMIT, attempt_number=1, policy=POLICY, cancelled=True
    )
    assert decision.retry is False
    assert decision.reason == "cancelled"


def test_an_unknown_failure_kind_is_not_retried_rather_than_guessed() -> None:
    decision = should_retry(failure_kind="meteor_strike", attempt_number=1, policy=POLICY)
    assert decision.retry is False
    assert "unknown failure kind" in decision.reason


def test_a_failure_kind_may_be_given_by_name() -> None:
    assert should_retry(failure_kind="rate_limit", attempt_number=1, policy=POLICY).retry is True


def test_the_backoff_a_retry_decision_uses_matches_the_plan() -> None:
    decision = should_retry(failure_kind=FailureKind.SERVER_ERROR, attempt_number=2, policy=POLICY)
    assert decision.delay_seconds == backoff_seconds(POLICY, 2)


def test_decisions_and_plans_are_serialisable() -> None:
    import json

    plan = build_plan(campaign_id="c1", tasks=[task("i0")], policy=POLICY, seed=1)
    decision = should_retry(failure_kind=FailureKind.RATE_LIMIT, attempt_number=1, policy=POLICY)
    json.dumps(plan.to_dict())
    json.dumps(decision.to_dict())


def test_a_single_attempt_plan_costs_one_dispatch_not_the_whole_policy() -> None:
    policy = RetryPolicy(
        max_attempts=1, initial_backoff_seconds=0.1, multiplier=1.0, max_backoff_seconds=1.0
    )
    plan = build_plan(campaign_id="c1", tasks=[task("i0")], policy=policy, seed=1)
    assert plan.dispatches == 1
