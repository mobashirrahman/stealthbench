"""T04B: the persistent spending and token ledger."""

from __future__ import annotations

import json
from decimal import Decimal

import pytest

from stealthbench.costs import (
    BudgetExceeded,
    CostBoundUnavailable,
    Ledger,
    LedgerState,
    PriceBook,
    PriceMissing,
    estimate_upper_bound,
    priced_usage,
    quantize,
    summed,
    usage_within_caps,
)
from stealthbench.schemas.campaign import Limits, PricingSnapshot
from stealthbench.schemas.results import SampleKey, Usage

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


def limits(**overrides: object) -> Limits:
    base = {
        "max_requests": 10,
        "max_concurrency": 2,
        "max_input_tokens": 100_000,
        "max_output_tokens": 4_000,
        "max_total_cost_usd": 1.0,
        "max_wall_seconds": 600.0,
    }
    base.update(overrides)
    return Limits.model_validate(base)


def key(item: str = "ifeval::i0", *, endpoint: str = "alias-a") -> SampleKey:
    return SampleKey(campaign_id="c1", endpoint_id=endpoint, task_id=item, repeat_id=1)


def ledger(**overrides: object) -> Ledger:
    return Ledger(limits=limits(**overrides), book=BOOK)


# ---------------------------------------------------------------------------
# An unknown price is not permission to spend an unknown amount
# ---------------------------------------------------------------------------


def test_a_missing_output_price_refuses_a_reservation() -> None:
    book = PriceBook(PricingSnapshot(input_per_mtok=3.0, snapshot_id="partial"))
    with pytest.raises(CostBoundUnavailable, match="no output price"):
        estimate_upper_bound(input_tokens=1000, max_output_tokens=100, book=book)


def test_a_missing_input_price_refuses_a_reservation() -> None:
    book = PriceBook(PricingSnapshot(output_per_mtok=15.0, snapshot_id="partial"))
    with pytest.raises(CostBoundUnavailable, match="no input price"):
        estimate_upper_bound(input_tokens=1000, max_output_tokens=100, book=book)


def test_an_unknown_input_token_count_refuses_a_reservation() -> None:
    with pytest.raises(CostBoundUnavailable, match="input token count is unknown"):
        estimate_upper_bound(input_tokens=None, max_output_tokens=100, book=BOOK)


def test_a_reservation_names_the_sample_it_refused_to_cover() -> None:
    with pytest.raises(CostBoundUnavailable, match="ifeval::i0"):
        ledger().reserve(sample_key=key(), input_tokens=None, max_output_tokens=10)


def test_a_complete_price_book_reports_every_direction_known() -> None:
    assert BOOK.is_complete() is True
    assert all(BOOK.known().values())


def test_an_incomplete_price_book_names_the_direction_it_lacks() -> None:
    book = PriceBook(PricingSnapshot(input_per_mtok=3.0, output_per_mtok=15.0, snapshot_id="p"))
    assert book.is_complete() is False
    assert book.known() == {
        "input": True,
        "output": True,
        "cached_input": False,
        "reasoning": False,
    }


def test_asking_for_an_absent_price_raises_rather_than_returning_zero() -> None:
    """A direction the snapshot does not price has no value, not a value of zero."""
    partial = PriceBook(PricingSnapshot(input_per_mtok=3.0, output_per_mtok=15.0, snapshot_id="p"))
    with pytest.raises(PriceMissing, match="reasoning"):
        partial._per_mtok("reasoning")
    with pytest.raises(PriceMissing, match="cached_input"):
        partial._per_mtok("cached_input")


def test_cost_of_an_unpriced_usage_report_is_absent_not_zero() -> None:
    book = PriceBook(PricingSnapshot(input_per_mtok=3.0, snapshot_id="partial"))
    usage = priced_usage(input_tokens=1000, output_tokens=10)
    assert book.cost_of(usage) is None, "an unknown cost must not read as free"


# ---------------------------------------------------------------------------
# The upper bound is a worst case
# ---------------------------------------------------------------------------


def test_the_upper_bound_prices_the_worst_case_not_the_expected_case() -> None:
    """A reservation covers every output token the request could produce."""
    bound = estimate_upper_bound(input_tokens=1_000_000, max_output_tokens=1000, book=BOOK)
    assert bound == Decimal("3.0") + Decimal("0.015")


def test_the_upper_bound_is_zero_only_when_both_counts_are_zero() -> None:
    with pytest.raises(CostBoundUnavailable):
        estimate_upper_bound(input_tokens=100, max_output_tokens=0, book=BOOK)


def test_a_negative_token_count_is_refused_rather_than_cheaper_than_zero() -> None:
    with pytest.raises(CostBoundUnavailable, match="non-negative"):
        estimate_upper_bound(input_tokens=-1, max_output_tokens=10, book=BOOK)


# ---------------------------------------------------------------------------
# Reserved versus billed
# ---------------------------------------------------------------------------


def test_a_reservation_is_held_and_counts_against_the_cap_before_settlement() -> None:
    book = ledger(max_total_cost_usd=100.0)
    reservation = book.reserve(sample_key=key(), input_tokens=1_000_000, max_output_tokens=1000)
    assert book.reserved == reservation.amount
    assert book.state.spent == Decimal(0), "held is not spent"
    assert book.outstanding == 1


def test_settling_releases_the_reservation_and_records_the_bill_separately() -> None:
    book = ledger(max_total_cost_usd=100.0)
    reservation = book.reserve(sample_key=key(), input_tokens=1_000_000, max_output_tokens=1000)
    assert reservation.amount == Decimal("3.015")
    settlement = book.settle(reservation.reservation_id, usage=priced_usage(1_000_000, 100))
    assert settlement.billed == Decimal("3.0") + Decimal("0.0015")
    assert book.reserved == Decimal(0)
    assert book.state.spent == settlement.billed, "spent becomes the bill once known"
    assert book.outstanding == 0


def test_an_unreported_usage_leaves_the_bill_unknown_and_keeps_the_reservation_spent() -> None:
    """The request went out, so the money may have gone even though nobody said."""
    book = ledger(max_total_cost_usd=100.0)
    reservation = book.reserve(sample_key=key(), input_tokens=1_000_000, max_output_tokens=1000)
    settlement = book.settle(reservation.reservation_id, usage=Usage(provider_reported=False))
    assert settlement.billed is None
    assert settlement.settled is False
    assert book.state.spent == reservation.amount, "pessimistic: assume it was spent"
    assert book.billed == Decimal(0), "the total bill is still exactly zero, not unknown"
    assert book.state.usage_missing == 1


def test_billed_and_spent_are_never_the_same_number_by_accident() -> None:
    book = ledger(max_total_cost_usd=100.0)
    book.limits = limits(max_total_cost_usd=100.0)
    a = book.reserve(sample_key=key("ifeval::a"), input_tokens=1_000_000, max_output_tokens=1000)
    book.settle(a.reservation_id, usage=priced_usage(1_000_000, 0))
    b = book.reserve(sample_key=key("ifeval::b"), input_tokens=1_000_000, max_output_tokens=1000)
    book.settle(b.reservation_id, usage=Usage(provider_reported=False))
    assert book.billed == Decimal("3.0")
    assert book.state.spent > book.billed, "the unknown one was charged at its worst case"


def test_a_forfeited_reservation_is_charged_and_says_why() -> None:
    book = ledger(max_total_cost_usd=100.0)
    reservation = book.reserve(sample_key=key(), input_tokens=1_000_000, max_output_tokens=1000)
    settlement = book.forfeit(reservation.reservation_id, reason="no response before the crash")
    assert settlement.billed is None
    assert settlement.note == "no response before the crash"
    assert book.state.spent == reservation.amount


def test_settling_an_unknown_reservation_is_an_error_not_a_silent_no_op() -> None:
    with pytest.raises(KeyError):
        ledger().settle("res-nope", usage=priced_usage(1, 1))
    with pytest.raises(KeyError):
        ledger().forfeit("res-nope", reason="x")


def test_reservation_ids_are_unique() -> None:
    book = ledger()
    ids = {
        book.reserve(
            sample_key=key(f"ifeval::i{n}"), input_tokens=10, max_output_tokens=10
        ).reservation_id
        for n in range(5)
    }
    assert len(ids) == 5


def test_a_reservation_records_the_price_snapshot_it_used() -> None:
    reservation = ledger().reserve(sample_key=key(), input_tokens=10, max_output_tokens=10)
    assert reservation.price_snapshot_id == "snap-1"


def test_a_price_change_after_reservation_does_not_silently_reattribute() -> None:
    """The reservation stands at the price it was made under; both ids are recorded."""
    book = ledger(max_total_cost_usd=100.0)
    reservation = book.reserve(sample_key=key(), input_tokens=1_000_000, max_output_tokens=1000)
    dearer = PriceBook(
        PricingSnapshot(
            input_per_mtok=300.0,
            output_per_mtok=1500.0,
            cached_input_per_mtok=30.0,
            reasoning_per_mtok=1500.0,
            snapshot_id="snap-2",
        )
    )
    book.book = dearer
    settlement = book.settle(reservation.reservation_id, usage=priced_usage(1_000_000, 0))
    assert settlement.price_snapshot_id == "snap-1", "the reservation's snapshot is what bound it"
    assert settlement.billed == Decimal("300.0"), "the new price bills what it billed"
    assert book.state.spent == Decimal("300.0"), "and the spend follows the actual price"


def test_cached_and_reasoning_tokens_are_priced_separately() -> None:
    usage = Usage(
        input_tokens=1_000_000,
        output_tokens=0,
        cached_input_tokens=1_000_000,
        reasoning_tokens=1_000_000,
        provider_reported=True,
    )
    assert BOOK.cost_of(usage) == Decimal("3.0") + Decimal("0.3") + Decimal("15.0")


def test_totals_over_settlements_are_unknown_when_any_one_is() -> None:
    book = ledger()
    a = book.reserve(sample_key=key("ifeval::a"), input_tokens=1000, max_output_tokens=10)
    known = book.settle(a.reservation_id, usage=priced_usage(1000, 1))
    assert known.billed == Decimal("0.003") + Decimal("0.000015")
    b = book.reserve(sample_key=key("ifeval::b"), input_tokens=1000, max_output_tokens=10)
    unknown = book.settle(b.reservation_id, usage=Usage(provider_reported=False))
    assert summed([known]) is not None
    assert summed([known, unknown]) is None


# ---------------------------------------------------------------------------
# Concurrent reservations must not collectively exceed the cap
# ---------------------------------------------------------------------------


def test_concurrent_reservations_cannot_collectively_exceed_the_cap() -> None:
    """A burst must not collectively overrun, which is what a reservation is for."""
    book = ledger(max_total_cost_usd=1.0)
    first = book.reserve(sample_key=key("ifeval::a"), input_tokens=200_000, max_output_tokens=1000)
    assert first.amount == Decimal("0.615"), "200k in at $3/Mtok plus 1k out at $15/Mtok"
    with pytest.raises(BudgetExceeded, match="max_total_cost_usd"):
        book.reserve(sample_key=key("ifeval::b"), input_tokens=200_000, max_output_tokens=1000)


def test_a_burst_of_three_is_bounded_by_the_cap_too() -> None:
    book = ledger(max_total_cost_usd=2.0)
    for n in range(3):
        book.reserve(sample_key=key(f"ifeval::i{n}"), input_tokens=200_000, max_output_tokens=1000)
    assert book.reserved == Decimal("1.845")
    with pytest.raises(BudgetExceeded, match="max_total_cost_usd"):
        book.reserve(sample_key=key("ifeval::i3"), input_tokens=200_000, max_output_tokens=1000)


def test_a_reservation_just_inside_the_cap_is_allowed() -> None:
    book = ledger(max_total_cost_usd=1.0)
    reservation = book.reserve(sample_key=key(), input_tokens=200_000, max_output_tokens=1000)
    assert book.committed == reservation.amount <= Decimal("1.0")


def test_settling_frees_budget_for_the_next_reservation() -> None:
    book = ledger(max_total_cost_usd=1.0)
    first = book.reserve(sample_key=key("ifeval::a"), input_tokens=200_000, max_output_tokens=1000)
    with pytest.raises(BudgetExceeded):
        book.reserve(sample_key=key("ifeval::b"), input_tokens=200_000, max_output_tokens=1000)
    # Settling releases the whole reservation, whatever the request actually cost.
    book.settle(first.reservation_id, usage=priced_usage(200_000, 0))
    assert book.reserved == Decimal(0)
    assert book.state.spent == Decimal("0.6"), "the actual bill, not the held bound"
    assert book.committed == Decimal("0.6"), "what is already spent still counts"
    with pytest.raises(BudgetExceeded):
        book.reserve(sample_key=key("ifeval::c"), input_tokens=200_000, max_output_tokens=1000)

    # A second reservation does fit when the first actually cost far less, which is
    # exactly the budget a settlement is supposed to return.
    book2 = ledger(max_total_cost_usd=1.0)
    a = book2.reserve(sample_key=key("ifeval::a"), input_tokens=200_000, max_output_tokens=1000)
    book2.settle(a.reservation_id, usage=priced_usage(1, 0))
    assert book2.state.spent == Decimal("0.000003")
    book2.reserve(sample_key=key("ifeval::b"), input_tokens=200_000, max_output_tokens=1000)
    assert book2.committed <= Decimal("1.0")


def test_a_ledger_with_no_cost_cap_still_refuses_an_unknown_input_bound() -> None:
    book = Ledger(limits=limits(max_total_cost_usd=None, require_cost_bounds=False), book=BOOK)
    with pytest.raises(CostBoundUnavailable, match="unknown"):
        book.reserve(sample_key=key(), input_tokens=None, max_output_tokens=10)
    reservation = book.reserve(sample_key=key(), input_tokens=10, max_output_tokens=10)
    assert reservation.amount == Decimal(0), "no cap means nothing is reserved"


def test_outstanding_reservations_count_against_the_request_cap() -> None:
    book = ledger(max_requests=2)
    book.reserve(sample_key=key("ifeval::a"), input_tokens=10, max_output_tokens=10)
    book.reserve(sample_key=key("ifeval::b"), input_tokens=10, max_output_tokens=10)
    with pytest.raises(BudgetExceeded, match="max_requests"):
        book.reserve(sample_key=key("ifeval::c"), input_tokens=10, max_output_tokens=10)


def test_bounds_required_false_still_refuses_a_missing_input_bound() -> None:
    """Without a bound the request's cost is unknown, which a report cannot fix."""
    book = Ledger(limits=limits(require_cost_bounds=False), book=BOOK)
    with pytest.raises(CostBoundUnavailable, match="unknown"):
        book.reserve(sample_key=key(), input_tokens=None, max_output_tokens=10)


# ---------------------------------------------------------------------------
# Token caps, including aliases that cost nothing
# ---------------------------------------------------------------------------


def test_usage_inside_both_token_caps_is_allowed() -> None:
    usage = priced_usage(input_tokens=10, output_tokens=20)
    decision = usage_within_caps(usage, limits(max_input_tokens=100, max_output_tokens=100))
    assert decision.allowed is True


@pytest.mark.parametrize(
    ("usage", "expected"),
    [
        (priced_usage(input_tokens=101, output_tokens=1), "input tokens"),
        (priced_usage(input_tokens=1, output_tokens=101), "output tokens"),
    ],
)
def test_usage_over_a_token_cap_is_refused(usage: Usage, expected: str) -> None:
    decision = usage_within_caps(usage, limits(max_input_tokens=100, max_output_tokens=100))
    assert decision.allowed is False
    assert expected in decision.reason


def test_a_free_alias_still_obeys_the_token_caps() -> None:
    """A zero price is not an exemption from a declared limit."""
    free = PriceBook(
        PricingSnapshot(
            input_per_mtok=0.0,
            output_per_mtok=0.0,
            cached_input_per_mtok=0.0,
            reasoning_per_mtok=0.0,
            snapshot_id="free",
        )
    )
    book = Ledger(limits=limits(), book=free)
    reservation = book.reserve(sample_key=key(), input_tokens=10_000, max_output_tokens=10)
    assert reservation.amount == Decimal(0)
    assert book.check_usage(priced_usage(input_tokens=10, output_tokens=10)).allowed is True
    assert (
        book.check_usage(priced_usage(input_tokens=10_000_000, output_tokens=10)).allowed is False
    )


def test_a_missingness_threshold_refuses_an_unreported_usage() -> None:
    book = ledger(missingness_threshold=0.1)
    decision = book.check_usage(Usage(provider_reported=False))
    assert decision.allowed is False
    assert "missingness threshold" in decision.reason


def test_without_a_threshold_an_unreported_usage_is_allowed_but_counted() -> None:
    book = ledger()
    assert book.check_usage(Usage(provider_reported=False)).allowed is True


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def test_a_ledger_round_trips_through_its_serialized_state() -> None:
    book = ledger(max_total_cost_usd=100.0)
    book.limits = limits(max_total_cost_usd=100.0)
    a = book.reserve(sample_key=key("ifeval::a"), input_tokens=1_000_000, max_output_tokens=1000)
    book.settle(a.reservation_id, usage=priced_usage(1_000_000, 100))
    b = book.reserve(sample_key=key("ifeval::b"), input_tokens=1_000_000, max_output_tokens=1000)
    restored = Ledger(
        limits=limits(), book=BOOK, state=LedgerState.from_dict(book.snapshot().to_dict())
    )
    assert restored.state.spent == book.state.spent
    assert restored.reserved == book.reserved, "an outstanding reservation survives a restart"
    assert restored.outstanding == 1
    restored.settle(b.reservation_id, usage=priced_usage(1_000_000, 50))
    assert restored.state.spent == book.state.spent + Decimal("3.00075"), (
        "the restored ledger bills the later request at its own usage"
    )


def test_ledger_state_is_json_serialisable() -> None:
    book = ledger()
    book.reserve(sample_key=key(), input_tokens=10, max_output_tokens=10)
    restored = json.loads(json.dumps(book.snapshot().to_dict()))
    assert LedgerState.from_dict(restored).spent == Decimal(0)


def test_a_bounds_report_says_what_cannot_be_bounded() -> None:
    partial = PriceBook(PricingSnapshot(input_per_mtok=3.0, snapshot_id="partial"))
    book = Ledger(limits=limits(), book=partial)
    report = book.bounds_report()
    assert report["bounds_available"]["cost"] is False
    assert report["bounds_available"]["price_directions"]["output"] is False
    assert report["bounds_available"]["input_tokens"] is True


def test_costs_are_quantized_only_for_display() -> None:
    """The stored amount keeps full precision; the rounded form is for a report."""
    book = ledger(max_total_cost_usd=100.0)
    reservation = book.reserve(sample_key=key(), input_tokens=1, max_output_tokens=1)
    assert reservation.amount == Decimal("0.0000180"), "1 in at $3/Mtok plus 1 out at $15/Mtok"
    rounded = str(quantize(reservation.amount))
    assert rounded == "0.000018000000"
    assert rounded != str(reservation.amount), "the display form is padded, not truncated"


def test_totals_report_counts_separately_from_costs() -> None:
    book = ledger(max_total_cost_usd=100.0)
    a = book.reserve(sample_key=key("ifeval::a"), input_tokens=1000, max_output_tokens=100)
    book.settle(a.reservation_id, usage=priced_usage(1000, 50))
    totals = book.totals()
    assert totals["input_tokens"] == 1000
    assert totals["output_tokens"] == 50
    assert totals["requests"] == 1, "derived from the records, not a separate counter"
    assert totals["billed"] == "0.003750"


def test_a_fallback_book_supplies_a_price_the_newer_snapshot_lacks() -> None:
    """A price may be carried forward explicitly rather than read as zero."""
    newer = PriceBook(PricingSnapshot(input_per_mtok=4.0, output_per_mtok=20.0, snapshot_id="new"))
    older = PriceBook(
        PricingSnapshot(
            input_per_mtok=3.0,
            output_per_mtok=15.0,
            cached_input_per_mtok=0.3,
            reasoning_per_mtok=15.0,
            snapshot_id="old",
        )
    )
    carried = PriceBook(newer.snapshot, fallback=older)
    assert carried._per_mtok("input") == Decimal("4.0"), "the newer price wins where it exists"
    assert carried._per_mtok("cached_input") == Decimal("0.3"), "the fallback fills the gap"
    assert carried.is_complete() is True
    with pytest.raises(PriceMissing):
        PriceBook(PricingSnapshot(snapshot_id="empty"))._per_mtok("output")


def test_a_settlement_note_records_the_declared_missingness_threshold() -> None:
    book = ledger(missingness_threshold=0.5)
    reservation = book.reserve(sample_key=key(), input_tokens=10, max_output_tokens=10)
    settlement = book.settle(reservation.reservation_id, usage=Usage(provider_reported=False))
    assert settlement.billed is None
    assert "missingness threshold" in (settlement.note or "")


def test_a_usage_decision_serialises_with_its_counts() -> None:
    usage = priced_usage(10, 20)
    decision = usage_within_caps(usage, limits(max_input_tokens=100, max_output_tokens=100))
    payload = decision.to_dict()
    assert payload["allowed"] is True
    assert payload["input_tokens"] == 10
    assert payload["output_tokens"] == 20
    json.dumps(payload)
