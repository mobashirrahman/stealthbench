"""T04C: crash recovery and accepted-sample accounting."""

from __future__ import annotations

import json
from decimal import Decimal

import pytest

from stealthbench.costs import Ledger, PriceBook, priced_usage
from stealthbench.recovery import (
    AcceptedSampleIndex,
    DuplicateRisk,
    open_reservations_of,
    reconcile_accepted,
    recover,
    report_json,
    result_record,
    unresolved_record,
)
from stealthbench.schemas.campaign import Limits, PricingSnapshot
from stealthbench.schemas.results import (
    DeliveryStatus,
    GenerationResult,
    SampleKey,
    Usage,
)

pytestmark = pytest.mark.unit

BOOK = PriceBook(
    PricingSnapshot(
        input_per_mtok=3.0,
        output_per_mtok=15.0,
        cached_input_per_mtok=0.3,
        reasoning_per_mtok=15.0,
        snapshot_id="snap-1",
    )
)


def ledger(**overrides: object) -> Ledger:
    base = {
        "max_requests": 20,
        "max_concurrency": 2,
        "max_input_tokens": 1_000_000,
        "max_output_tokens": 100_000,
        "max_total_cost_usd": 10.0,
        "max_wall_seconds": 600.0,
    }
    base.update(overrides)
    return Ledger(limits=Limits.model_validate(base), book=BOOK)


def key(item: str = "ifeval::i0", *, endpoint: str = "alias-a", repeat: int = 1) -> SampleKey:
    return SampleKey(campaign_id="c1", endpoint_id=endpoint, task_id=item, repeat_id=repeat)


def unresolved_result(
    sample: SampleKey, *, attempt_id: str = "att-x", attempt_number: int = 1
) -> GenerationResult:
    """An attempt dispatched with no confirmed outcome.

    The schema refuses a response on such an attempt, which is the point: an
    ambiguous dispatch must not carry an answer nobody received.
    """
    return GenerationResult(
        attempt_id=attempt_id,
        sample_key=sample,
        attempt_number=attempt_number,
        delivery_status=DeliveryStatus.UNRESOLVED,
        response=None,
        usage=Usage(provider_reported=False),
    )


def result(
    sample: SampleKey,
    *,
    attempt_id: str = "att-1",
    attempt_number: int = 1,
    status: DeliveryStatus = DeliveryStatus.ACCEPTED,
    response: str = "an answer",
) -> GenerationResult:
    return GenerationResult(
        attempt_id=attempt_id,
        sample_key=sample,
        attempt_number=attempt_number,
        delivery_status=status,
        response=response,
        usage=priced_usage(1000, 100),
    )


def attempt_record(
    sample: SampleKey | None,
    reservation_id: str | None,
    *,
    attempt_number: int = 1,
    reason: str = "",
) -> dict[str, object]:
    return {
        "sample_key": None if sample is None else list(sample.as_tuple()),
        "attempt_number": attempt_number,
        "reservation_id": reservation_id,
        "reason": reason or "dispatched with no durable response",
    }


# ---------------------------------------------------------------------------
# An ambiguous dispatch is unresolved, not retried and not dropped
# ---------------------------------------------------------------------------


def test_a_dispatched_attempt_with_no_response_is_unresolved_by_default() -> None:
    book = ledger()
    sample = key()
    reservation = book.reserve(sample_key=sample, input_tokens=1000, max_output_tokens=100)
    report = recover(
        campaign_id="c1",
        ledger=book,
        attempt_records={"att-1": attempt_record(sample, reservation.reservation_id)},
    )
    assert len(report.unresolved) == 1
    assert report.unresolved[0].attempt_id == "att-1"
    assert report.unresolved[0].may_have_been_billed is True
    assert report.resubmitted == (), "recovery never resubmits on its own"


def test_an_ambiguous_dispatch_is_charged_because_it_may_have_been_billed() -> None:
    book = ledger()
    reservation = book.reserve(sample_key=key(), input_tokens=1000, max_output_tokens=100)
    before = book.state.spent
    recover(
        campaign_id="c1",
        ledger=book,
        attempt_records={"att-1": attempt_record(key(), reservation.reservation_id)},
    )
    assert book.state.spent == before + reservation.amount
    assert book.reserved == Decimal(0), "the reservation is closed, not left hanging"


def test_recovery_never_submits_anything() -> None:
    """The report's resubmitted list exists so that its emptiness is a fact, not an absence."""
    book = ledger()
    reservation = book.reserve(sample_key=key(), input_tokens=10, max_output_tokens=10)
    report = recover(
        campaign_id="c1",
        ledger=book,
        attempt_records={"att-1": attempt_record(key(), reservation.reservation_id)},
    )
    assert report.resubmitted == ()


def test_an_attempt_that_came_back_is_not_unresolved() -> None:
    book = ledger()
    reservation = book.reserve(sample_key=key(), input_tokens=1000, max_output_tokens=100)
    book.settle(reservation.reservation_id, usage=priced_usage(1000, 100))
    report = recover(
        campaign_id="c1",
        ledger=book,
        results=[result(key())],
        attempt_records={"att-1": attempt_record(key(), reservation.reservation_id)},
    )
    assert report.unresolved == ()
    assert report.accepted_count == 1


def test_a_crash_before_dispatch_leaves_no_reservation_and_no_charge() -> None:
    """Nothing went out, so nothing may be charged."""
    book = ledger()
    report = recover(campaign_id="c1", ledger=book, attempt_records={})
    assert report.charged_ambiguous == 0.0
    assert report.unresolved == ()
    assert report.open_reservations == 0


def test_an_unresolved_record_without_a_usable_sample_key_is_still_recorded() -> None:
    """An invented endpoint id would read as an observation that never happened."""
    book = ledger()
    reservation = book.reserve(sample_key=key(), input_tokens=10, max_output_tokens=10)
    report = recover(
        campaign_id="c1",
        ledger=book,
        attempt_records={"att-1": attempt_record(None, reservation.reservation_id)},
    )
    assert len(report.unresolved) == 1
    assert report.unresolved[0].sample_key is None


def test_a_second_dispatch_for_an_accepted_sample_is_a_duplicate_not_a_sample() -> None:
    """A *different* attempt id for a sample that is already accepted."""
    book = ledger()
    reservation = book.reserve(sample_key=key(), input_tokens=10, max_output_tokens=10)
    report = recover(
        campaign_id="c1",
        ledger=book,
        results=[result(key(), attempt_id="att-1")],
        attempt_records={"att-2": attempt_record(key(), reservation.reservation_id)},
    )
    assert report.accepted_count == 1, "recovery cannot add a second accepted sample"
    assert report.unresolved[0].may_have_been_billed is False
    assert "duplicate" in report.unresolved[0].reason


def test_an_attempt_that_came_back_is_never_also_reported_unresolved() -> None:
    """Its record exists because it was dispatched, not because it is unknown."""
    book = ledger()
    reservation = book.reserve(sample_key=key(), input_tokens=10, max_output_tokens=10)
    report = recover(
        campaign_id="c1",
        ledger=book,
        results=[result(key(), attempt_id="att-1")],
        attempt_records={"att-1": attempt_record(key(), reservation.reservation_id)},
    )
    assert report.unresolved == ()
    assert report.accepted_count == 1


# ---------------------------------------------------------------------------
# Resume cannot duplicate an accepted sample
# ---------------------------------------------------------------------------


def test_an_accepted_sample_survives_a_restart_as_exactly_one() -> None:
    index = AcceptedSampleIndex([result(key())])
    assert len(index) == 1
    assert index.is_accepted(key())
    assert index.keys == {key().as_tuple()}


def test_several_attempts_for_one_sample_collapse_to_one_accepted_sample() -> None:
    """A retry is another attempt at one sample, not a second sample."""
    index = AcceptedSampleIndex(
        [
            result(key(), attempt_id="att-1", attempt_number=1),
            result(key(), attempt_id="att-2", attempt_number=2),
            result(key(), attempt_id="att-3", attempt_number=3),
        ]
    )
    assert len(index) == 1
    assert index.attempt_ids == frozenset({"att-1", "att-2", "att-3"})


def test_the_first_accepted_result_is_the_one_kept() -> None:
    index = AcceptedSampleIndex(
        [
            result(key(), attempt_id="att-1", response="first"),
            result(key(), attempt_id="att-2", response="second"),
        ]
    )
    assert index.results[0].response == "first"


def test_a_repeat_is_a_separate_sample_not_a_duplicate() -> None:
    index = AcceptedSampleIndex([result(key(repeat=1)), result(key(repeat=2))])
    assert len(index) == 2


def test_a_different_endpoint_is_a_separate_sample() -> None:
    index = AcceptedSampleIndex([result(key(endpoint="a")), result(key(endpoint="b"))])
    assert len(index) == 2


def test_reconcile_reads_accepted_and_unresolved_from_results_alone() -> None:
    accepted, unresolved = reconcile_accepted(
        [
            result(key(), attempt_id="att-1"),
            unresolved_result(key("ifeval::i1"), attempt_id="att-2"),
        ]
    )
    assert len(accepted) == 1
    assert "att-2" in unresolved


# ---------------------------------------------------------------------------
# An explicit retry keeps the history and discloses the duplicate risk
# ---------------------------------------------------------------------------


def test_an_explicit_retry_is_authorized_when_nothing_was_accepted() -> None:
    index = AcceptedSampleIndex([])
    decision = index.authorize_retry(key(), attempt_number=1, max_attempts=3)
    assert decision.authorized is True
    assert decision.risk is DuplicateRisk.MAY_BE_BILLED_TWICE
    assert "may still have been billed" in decision.reason


def test_an_explicit_retry_of_an_accepted_sample_is_refused() -> None:
    index = AcceptedSampleIndex([result(key())])
    decision = index.authorize_retry(key(), attempt_number=2, max_attempts=3)
    assert decision.authorized is False
    assert decision.risk is DuplicateRisk.ALREADY_ACCEPTED
    assert "duplicate" in decision.reason


def test_a_retry_beyond_the_declared_maximum_is_refused() -> None:
    index = AcceptedSampleIndex([])
    decision = index.authorize_retry(key(), attempt_number=3, max_attempts=3)
    assert decision.authorized is False
    assert decision.risk is DuplicateRisk.NONE


def test_an_attempt_id_can_be_found_for_a_sample_and_number() -> None:
    index = AcceptedSampleIndex([result(key(), attempt_number=2)])
    assert index.attempt_for(key(), 2) is not None
    assert index.attempt_for(key(), 1) is None
    assert index.attempt_for(key("ifeval::other"), 2) is None


def test_the_cost_history_of_a_retried_attempt_is_kept_not_replaced() -> None:
    """The first attempt's charge stays; the retry is additional."""
    book = ledger()
    first = book.reserve(sample_key=key(), input_tokens=1000, max_output_tokens=100)
    book.settle(first.reservation_id, usage=priced_usage(1000, 100))
    first_spend = book.state.spent
    second = book.reserve(sample_key=key(), input_tokens=1000, max_output_tokens=100)
    book.settle(second.reservation_id, usage=priced_usage(1000, 50))
    assert book.state.spent == first_spend + Decimal("0.00375")
    assert len(book.state.settlements) == 2, "both attempts remain in the record"


# ---------------------------------------------------------------------------
# Persistence across a restart
# ---------------------------------------------------------------------------


def test_an_open_reservation_survives_a_restart_and_is_still_charged() -> None:
    book = ledger()
    reservation = book.reserve(sample_key=key(), input_tokens=1000, max_output_tokens=100)
    from stealthbench.costs import LedgerState

    restored = Ledger(
        limits=book.limits, book=BOOK, state=LedgerState.from_dict(book.snapshot().to_dict())
    )
    assert restored.outstanding == 1
    report = recover(
        campaign_id="c1",
        ledger=restored,
        attempt_records={"att-1": attempt_record(key(), reservation.reservation_id)},
    )
    assert report.charged_ambiguous == float(reservation.amount)
    assert restored.outstanding == 0


def test_open_reservations_are_listed_for_a_restart_to_respect() -> None:
    book = ledger()
    book.reserve(sample_key=key("ifeval::a"), input_tokens=10, max_output_tokens=10)
    book.reserve(sample_key=key("ifeval::b"), input_tokens=10, max_output_tokens=10)
    assert [r.sample_key.task_id for r in open_reservations_of(book.state)] == [
        "ifeval::a",
        "ifeval::b",
    ]


def test_recovery_is_idempotent() -> None:
    """Running it twice must not charge twice."""
    book = ledger()
    reservation = book.reserve(sample_key=key(), input_tokens=1000, max_output_tokens=100)
    records = {"att-1": attempt_record(key(), reservation.reservation_id)}
    first = recover(campaign_id="c1", ledger=book, attempt_records=records)
    spent = book.state.spent
    second = recover(campaign_id="c1", ledger=book, attempt_records=records)
    assert book.state.spent == spent, "the reservation was already closed"
    assert second.charged_ambiguous == 0.0
    assert len(second.unresolved) == len(first.unresolved)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def test_recovery_reports_are_json_serialisable() -> None:
    book = ledger()
    reservation = book.reserve(sample_key=key(), input_tokens=10, max_output_tokens=10)
    report = recover(
        campaign_id="c1",
        ledger=book,
        results=[result(key("ifeval::ok"), attempt_id="att-ok")],
        attempt_records={"att-1": attempt_record(key(), reservation.reservation_id)},
    )
    payload = json.loads(report_json(report))
    assert payload["campaign_id"] == "c1"
    assert payload["accepted_count"] == 1
    assert len(payload["unresolved"]) == 1
    assert payload["resubmitted"] == []


def test_an_unresolved_attempt_record_serialises_its_risk() -> None:
    book = ledger()
    reservation = book.reserve(sample_key=key(), input_tokens=10, max_output_tokens=10)
    report = recover(
        campaign_id="c1",
        ledger=book,
        attempt_records={"att-1": attempt_record(key(), reservation.reservation_id)},
    )
    payload = unresolved_record(report.unresolved[0])
    assert payload["may_have_been_billed"] is True
    assert payload["sample_key"] == list(key().as_tuple())
    json.dumps(payload)


def test_a_result_record_names_the_attempt_and_its_usage() -> None:
    record = result_record(result(key()))
    assert record["attempt_id"] == "att-1"
    assert record["delivery_status"] == str(DeliveryStatus.ACCEPTED)
    assert record["usage"]["output_tokens"] == 100
    json.dumps(record)


def test_a_retry_authorization_serialises() -> None:
    index = AcceptedSampleIndex([])
    payload = index.authorize_retry(key(), 1, max_attempts=3).to_dict()
    assert payload["authorized"] is True
    assert payload["risk"] == "may_be_billed_twice"
    json.dumps(payload)


def test_an_index_can_be_extended_as_results_arrive() -> None:
    index = AcceptedSampleIndex([result(key("ifeval::a"))])
    index.add(result(key("ifeval::b")))
    assert len(index) == 2
    assert not index.is_accepted(key("ifeval::c"))


def test_an_unresolved_result_is_not_counted_as_an_accepted_sample() -> None:
    index = AcceptedSampleIndex([unresolved_result(key())])
    assert len(index) == 0, "no confirmed delivery means no accepted sample"


def test_an_unresolved_attempt_carries_no_usage_and_no_response() -> None:
    """Nothing was confirmed, so there is nothing to report as measured."""
    unresolved = unresolved_result(key())
    assert unresolved.usage.provider_reported is False
    assert unresolved.usage.input_tokens is None
    assert unresolved.response is None
    assert unresolved.is_accepted_sample is False


def test_a_resolved_attempt_record_serialises_its_outcome() -> None:
    """The mirror of an unresolved record: what came back, and how it ended."""
    from stealthbench.recovery import ResolvedAttempt

    record = ResolvedAttempt(
        attempt_id="att-1",
        sample_key=key(),
        attempt_number=1,
        accepted=True,
        finish_status="stop",
        usage=priced_usage(1000, 100),
    )
    payload = record.to_dict()
    assert payload["accepted"] is True
    assert payload["finish_status"] == "stop"
    assert payload["usage"]["output_tokens"] == 100
    json.dumps(payload)


def test_an_unusable_record_shapes_yields_no_sample_key_rather_than_a_guess() -> None:
    """A malformed durable record must not be turned into a plausible-looking key."""
    book = ledger()
    for record in (
        {"sample_key": "not-a-tuple", "attempt_number": 1},
        {"sample_key": [1, 2], "attempt_number": 1},
        {"sample_key": ["c", "e", "t", "not-an-int"], "attempt_number": 1},
        {"attempt_number": 1},
    ):
        report = recover(campaign_id="c1", ledger=book, attempt_records={"att-x": record})
        assert report.unresolved[0].sample_key is None, record


def test_a_record_with_no_reservation_id_still_records_the_attempt() -> None:
    """The attempt is unknown whether or not the reservation can be located."""
    book = ledger()
    report = recover(
        campaign_id="c1",
        ledger=book,
        attempt_records={"att-x": {"sample_key": list(key().as_tuple()), "attempt_number": 2}},
    )
    assert len(report.unresolved) == 1
    assert report.unresolved[0].reserved is None
    assert report.unresolved[0].attempt_number == 2


def test_a_reservation_id_that_does_not_exist_yields_no_amount() -> None:
    book = ledger()
    report = recover(
        campaign_id="c1",
        ledger=book,
        attempt_records={
            "att-x": {
                "sample_key": list(key().as_tuple()),
                "attempt_number": 1,
                "reservation_id": "res-does-not-exist",
            }
        },
    )
    assert report.unresolved[0].reserved is None


def test_a_reservation_id_that_is_not_a_string_yields_no_amount() -> None:
    book = ledger()
    report = recover(
        campaign_id="c1",
        ledger=book,
        attempt_records={
            "att-x": {
                "sample_key": list(key().as_tuple()),
                "attempt_number": 1,
                "reservation_id": 7,
            }
        },
    )
    assert report.unresolved[0].reserved is None
