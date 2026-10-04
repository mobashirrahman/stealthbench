"""Persistent spending and token ledger.

The rule this module exists to enforce: **an unknown price is not permission to spend
an unknown amount.** If a reliable upper bound on a request's cost cannot be computed,
spending-capped execution is refused until explicit bounds exist.

Two kinds of number are tracked and never mixed:

* **Reserved** — a conservative upper bound taken *before* dispatch, so a burst of
  concurrent requests cannot collectively exceed the cap.
* **Billed** — what the endpoint actually reported afterwards. Where the endpoint
  reports nothing, the billed amount stays ``None``. It does not become the estimate,
  and the estimate does not become the bill.

A reservation covers the whole worst case: every output token the request could
produce, at the price that will apply when it does. If the price changes between
reservation and settlement, the reservation still stands, and the difference is
reported.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Final

from stealthbench.schemas.campaign import Limits, PricingSnapshot
from stealthbench.schemas.results import SampleKey, Usage

__all__ = [
    "BudgetExceeded",
    "CostBoundUnavailable",
    "Ledger",
    "LedgerState",
    "PriceBook",
    "PriceMissing",
    "Reservation",
    "Settlement",
    "UsageDecision",
    "estimate_upper_bound",
    "usage_within_caps",
]

#: Costs are compared as Decimal so a float rounding step can never decide a cap.
_CENT: Final[Decimal] = Decimal("0.000000000001")
_TOKENS_PER_MTOK: Final[Decimal] = Decimal(1_000_000)


class PriceMissing(Exception):
    """A price this reservation needs is not known.

    Raised rather than defaulted. Substituting zero would turn "we do not know what
    this costs" into "this is free", which is the failure this module prevents.
    """


class CostBoundUnavailable(Exception):
    """A reliable upper bound could not be computed, so capped execution is refused."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class BudgetExceeded(Exception):
    """A declared limit would be exceeded by the requested work."""

    def __init__(self, limit: str, requested: Decimal, allowed: Decimal) -> None:
        super().__init__(f"{limit}: requested {requested}, only {allowed} remains")
        self.limit = limit
        self.requested = requested
        self.allowed = allowed


@dataclass(frozen=True, slots=True)
class PriceBook:
    """Prices as observed, with the snapshot they came from.

    ``None`` for a direction means the price is unknown, never zero. A snapshot whose
    ``snapshot_id`` differs from the one in force is a price change, and the caller
    must decide what to do about it rather than have it applied silently.
    """

    snapshot: PricingSnapshot
    fallback: PriceBook | None = None

    def _per_mtok(self, direction: str) -> Decimal:
        value = getattr(self.snapshot, f"{direction}_per_mtok")
        if value is not None:
            return Decimal(str(value))
        if self.fallback is not None:
            return self.fallback._per_mtok(direction)
        raise PriceMissing(f"no {direction} price in snapshot {self.snapshot.snapshot_id!r}")

    def known(self) -> dict[str, bool]:
        """Which directions this book can price, for a bounds report."""
        report: dict[str, bool] = {}
        for direction in ("input", "output", "cached_input", "reasoning"):
            try:
                self._per_mtok(direction)
            except PriceMissing:
                report[direction] = False
            else:
                report[direction] = True
        return report

    def is_complete(self) -> bool:
        """Whether every direction needed for a worst-case bound is known."""
        return all(self.known().values())

    def cost_of(self, usage: Usage) -> Decimal | None:
        """What a usage report costs, or ``None`` when a needed price is unknown.

        ``None`` is not zero: it means the amount is unknown, which is a different
        statement from free.
        """
        try:
            return _cost_of(usage, self)
        except PriceMissing:
            return None


def _cost_of(usage: Usage, book: PriceBook) -> Decimal:
    total = Decimal(0)
    directions = (
        ("input_tokens", "input"),
        ("output_tokens", "output"),
        ("cached_input_tokens", "cached_input"),
        ("reasoning_tokens", "reasoning"),
    )
    for field_name, direction in directions:
        tokens = getattr(usage, field_name)
        if tokens is None:
            continue
        total += Decimal(tokens) / _TOKENS_PER_MTOK * book._per_mtok(direction)
    return total


def estimate_upper_bound(
    *,
    input_tokens: int | None,
    max_output_tokens: int,
    book: PriceBook,
) -> Decimal:
    """The worst-case cost of one request, before it is dispatched.

    Input tokens are taken from the prompt when known. When they are not, the caller
    must supply a bound; this function refuses rather than assuming zero, because a
    zero input assumption makes the reservation unbounded in practice.
    """
    if input_tokens is None:
        raise CostBoundUnavailable(
            "input token count is unknown, so no upper bound on this request's cost"
        )
    if input_tokens < 0 or max_output_tokens <= 0:
        raise CostBoundUnavailable("token counts must be non-negative and output positive")
    try:
        input_price = book._per_mtok("input")
        output_price = book._per_mtok("output")
    except PriceMissing as exc:
        raise CostBoundUnavailable(str(exc)) from exc
    return (
        Decimal(input_tokens) / _TOKENS_PER_MTOK * input_price
        + Decimal(max_output_tokens) / _TOKENS_PER_MTOK * output_price
    )


@dataclass(frozen=True, slots=True)
class UsageDecision:
    """Whether a usage report stays inside the declared token caps."""

    allowed: bool
    reason: str
    input_tokens: int | None = None
    output_tokens: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "reason": self.reason,
        }


def usage_within_caps(usage: Usage, limits: Limits) -> UsageDecision:
    """Whether reported usage is inside the campaign's token caps.

    A cap that an unreported count might have exceeded is reported as unknown rather
    than passed: the report cannot say the cap held.
    """
    if usage.input_tokens is not None and usage.input_tokens > limits.max_input_tokens:
        return UsageDecision(
            False,
            f"input tokens {usage.input_tokens} exceed the cap {limits.max_input_tokens}",
            usage.input_tokens,
            usage.output_tokens,
        )
    if usage.output_tokens is not None and usage.output_tokens > limits.max_output_tokens:
        return UsageDecision(
            False,
            f"output tokens {usage.output_tokens} exceed the cap {limits.max_output_tokens}",
            usage.input_tokens,
            usage.output_tokens,
        )
    if limits.missingness_threshold is not None and not usage.provider_reported:
        return UsageDecision(
            False,
            "the endpoint reported no usage and the manifest declares a missingness threshold",
            usage.input_tokens,
            usage.output_tokens,
        )
    return UsageDecision(
        True, "within the declared token caps", usage.input_tokens, usage.output_tokens
    )


@dataclass(frozen=True, slots=True)
class Reservation:
    """A worst-case amount held before a dispatch goes out.

    Held rather than spent: it is released on settlement, or converted to spent if the
    request went out and no response came back. An unreleased reservation is what
    bounds a burst of concurrent requests.
    """

    reservation_id: str
    sample_key: SampleKey
    input_tokens: int
    max_output_tokens: int
    amount: Decimal
    price_snapshot_id: str | None
    attempt_number: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "amount": str(self.amount),
            "attempt_number": self.attempt_number,
            "input_tokens": self.input_tokens,
            "max_output_tokens": self.max_output_tokens,
            "price_snapshot_id": self.price_snapshot_id,
            "reservation_id": self.reservation_id,
            "sample_key": list(self.sample_key.as_tuple()),
        }


@dataclass(frozen=True, slots=True)
class Settlement:
    """What a dispatched request actually cost, as distinct from what was held."""

    reservation_id: str
    billed: Decimal | None
    usage: Usage
    price_snapshot_id: str | None
    released: Decimal
    note: str | None = None
    settled_price_snapshot_id: str | None = None

    @property
    def settled(self) -> bool:
        return self.billed is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "billed": None if self.billed is None else str(self.billed),
            "note": self.note,
            "price_snapshot_id": self.price_snapshot_id,
            "released": str(self.released),
            "reservation_id": self.reservation_id,
            "settled": self.settled,
            "settled_price_snapshot_id": self.settled_price_snapshot_id,
            "usage": self.usage.model_dump(mode="json"),
        }


@dataclass(slots=True)
class LedgerState:
    """The ledger's durable content.

    Reservations that have not settled are kept: they are the budget a resumed run
    must still respect, because those requests may have been billed.
    """

    reservations: dict[str, Reservation] = field(default_factory=dict)
    settlements: dict[str, Settlement] = field(default_factory=dict)
    spent: Decimal = Decimal(0)
    billed: Decimal | None = Decimal(0)
    input_tokens: int = 0
    output_tokens: int = 0
    usage_missing: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "billed": None if self.billed is None else str(self.billed),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "reservations": {k: v.to_dict() for k, v in sorted(self.reservations.items())},
            "settlements": {k: v.to_dict() for k, v in sorted(self.settlements.items())},
            "spent": str(self.spent),
            "usage_missing": self.usage_missing,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> LedgerState:
        billed = raw.get("billed")
        return cls(
            reservations={
                k: Reservation(
                    reservation_id=v["reservation_id"],
                    sample_key=SampleKey(
                        campaign_id=v["sample_key"][0],
                        endpoint_id=v["sample_key"][1],
                        task_id=v["sample_key"][2],
                        repeat_id=v["sample_key"][3],
                    ),
                    input_tokens=v["input_tokens"],
                    max_output_tokens=v["max_output_tokens"],
                    amount=Decimal(v["amount"]),
                    price_snapshot_id=v["price_snapshot_id"],
                    attempt_number=v.get("attempt_number", 1),
                )
                for k, v in raw.get("reservations", {}).items()
            },
            settlements={
                k: Settlement(
                    reservation_id=v["reservation_id"],
                    billed=None if v["billed"] is None else Decimal(v["billed"]),
                    usage=Usage.model_validate(v["usage"]),
                    price_snapshot_id=v["price_snapshot_id"],
                    released=Decimal(v["released"]),
                    note=v.get("note"),
                    settled_price_snapshot_id=v.get("settled_price_snapshot_id"),
                )
                for k, v in raw.get("settlements", {}).items()
            },
            spent=Decimal(raw.get("spent", "0")),
            billed=None if billed is None else Decimal(billed),
            input_tokens=int(raw.get("input_tokens", 0)),
            output_tokens=int(raw.get("output_tokens", 0)),
            usage_missing=int(raw.get("usage_missing", 0)),
        )


class Ledger:
    """Reserves before dispatch, settles after, and refuses what cannot be bounded."""

    def __init__(
        self,
        *,
        limits: Limits,
        book: PriceBook,
        state: LedgerState | None = None,
        sequence: Iterable[str] | None = None,
    ) -> None:
        self.limits = limits
        self.book = book
        self.state = state or LedgerState()
        self._sequence = iter(sequence or ())
        self._counter = len(self.state.reservations) + len(self.state.settlements)

    # -- reading -----------------------------------------------------------

    @property
    def reserved(self) -> Decimal:
        """Held but not yet settled, including every outstanding concurrent request."""
        return sum((r.amount for r in self.state.reservations.values()), Decimal(0))

    @property
    def committed(self) -> Decimal:
        """What the cap must cover: everything already spent plus everything held."""
        return self.state.spent + self.reserved

    @property
    def billed(self) -> Decimal | None:
        """What providers actually billed, or ``None`` if nothing reported yet."""
        return self.state.billed

    @property
    def outstanding(self) -> int:
        return len(self.state.reservations)

    @property
    def requests(self) -> int:
        """Requests made, derived rather than counted.

        Every reservation becomes either a settlement or an outstanding hold, so the
        two record sets are the count. A separate counter could drift from them and
        let a run exceed its request cap without any record showing why.
        """
        return len(self.state.settlements) + self.outstanding

    def snapshot(self) -> LedgerState:
        return LedgerState(
            reservations=dict(self.state.reservations),
            settlements=dict(self.state.settlements),
            spent=self.state.spent,
            billed=self.state.billed,
            input_tokens=self.state.input_tokens,
            output_tokens=self.state.output_tokens,
            usage_missing=self.state.usage_missing,
        )

    # -- reserving ---------------------------------------------------------

    def reserve(
        self,
        *,
        sample_key: SampleKey,
        input_tokens: int | None,
        max_output_tokens: int,
        attempt_number: int = 1,
    ) -> Reservation:
        """Hold the worst case for one dispatch, or refuse.

        The worst case is held whenever it can be computed, whether or not bounds
        are required: a computable bound the ledger does not hold is a hole in the
        cap accounting. Refusal happens when the bound cannot be computed and bounds
        are required, when a cap would be exceeded, or when no input bound exists.
        Each is a different reason and the caller is told which.
        """
        try:
            amount = estimate_upper_bound(
                input_tokens=input_tokens,
                max_output_tokens=max_output_tokens,
                book=self.book,
            )
        except CostBoundUnavailable as exc:
            if self.limits.require_cost_bounds and self.limits.max_total_cost_usd is not None:
                raise CostBoundUnavailable(
                    f"cannot reserve for {sample_key.task_id}: {exc.reason}"
                ) from exc
            amount = Decimal(0)
            if input_tokens is None:
                raise CostBoundUnavailable(
                    f"cannot reserve for {sample_key.task_id}: input token count is unknown"
                ) from exc

        self._counter += 1
        reservation_id = next(self._sequence, f"res-{self._counter:06d}")

        if self.limits.max_total_cost_usd is not None:
            cap = Decimal(str(self.limits.max_total_cost_usd))
            if self.committed + amount > cap:
                raise BudgetExceeded("max_total_cost_usd", amount, cap - self.committed)

        # Outstanding reservations count against the request cap: a request that has
        # not reported yet was still made.
        projected_requests = self.requests + 1
        if projected_requests > self.limits.max_requests:
            raise BudgetExceeded(
                "max_requests",
                Decimal(1),
                Decimal(self.limits.max_requests - self.requests),
            )

        reservation = Reservation(
            reservation_id=reservation_id,
            sample_key=sample_key,
            input_tokens=input_tokens or 0,
            max_output_tokens=max_output_tokens,
            amount=amount,
            price_snapshot_id=self.book.snapshot.snapshot_id,
            attempt_number=attempt_number,
        )
        self.state.reservations[reservation_id] = reservation
        return reservation

    # -- settling ----------------------------------------------------------

    def settle(
        self,
        reservation_id: str,
        *,
        usage: Usage,
        price_snapshot_id: str | None = None,
    ) -> Settlement:
        """Release a reservation and record what was actually billed.

        A report with no counts leaves the billed amount unknown and the whole
        reservation spent: the request went out, so the money may have been spent even
        though nobody said how much.

        The reservation's snapshot stays the bound. The snapshot actually used to bill
        is recorded separately; when the two differ, or when the billed amount exceeds
        the reserved bound, the settlement note says so instead of silently absorbing
        the change.
        """
        reservation = self.state.reservations.pop(reservation_id, None)
        if reservation is None:
            raise KeyError(f"no open reservation {reservation_id!r}")

        settled_snapshot_id = price_snapshot_id or self.book.snapshot.snapshot_id
        billed = self.book.cost_of(usage) if usage.provider_reported else None
        notes: list[str] = []
        if not usage.provider_reported:
            notes.append("the endpoint reported no usage; the reservation is recorded as spent")
            self.state.spent += reservation.amount
        else:
            self.state.spent += billed or Decimal(0)

        if billed is not None:
            self.state.billed = (self.state.billed or Decimal(0)) + billed

        for field_name, counter in (
            ("input_tokens", "input_tokens"),
            ("output_tokens", "output_tokens"),
        ):
            value = getattr(usage, field_name)
            if value is not None:
                setattr(self.state, counter, getattr(self.state, counter) + value)
        if not usage.provider_reported:
            self.state.usage_missing += 1
        if (
            reservation.price_snapshot_id is not None
            and settled_snapshot_id is not None
            and settled_snapshot_id != reservation.price_snapshot_id
        ):
            if billed is not None:
                notes.append(
                    "price changed between reservation "
                    f"({reservation.price_snapshot_id}) and settlement "
                    f"({settled_snapshot_id}); billed {billed} against reserved "
                    f"{reservation.amount} (difference {billed - reservation.amount})"
                )
            else:
                notes.append(
                    "price changed between reservation "
                    f"({reservation.price_snapshot_id}) and settlement "
                    f"({settled_snapshot_id}); the billed amount is unknown, so the "
                    "reservation stands as spent"
                )
        if billed is not None and billed > reservation.amount:
            notes.append(
                f"actual cost {billed} exceeded reserved bound {reservation.amount} "
                f"by {billed - reservation.amount}"
            )
        if self.limits.max_total_cost_usd is not None:
            cap = Decimal(str(self.limits.max_total_cost_usd))
            if self.state.spent > cap:
                notes.append(
                    f"spent {self.state.spent} exceeds declared cost cap {cap}; "
                    "no further capped dispatches may be reserved"
                )
        if self.limits.missingness_threshold is not None and not usage.provider_reported:
            notes.append("the manifest declares a missingness threshold")

        settlement = Settlement(
            reservation_id=reservation_id,
            billed=billed,
            usage=usage,
            price_snapshot_id=reservation.price_snapshot_id,
            released=reservation.amount,
            note="; ".join(notes) if notes else None,
            settled_price_snapshot_id=settled_snapshot_id,
        )
        self.state.settlements[reservation_id] = settlement
        return settlement

    def forfeit(self, reservation_id: str, *, reason: str) -> Settlement:
        """Close a reservation without a response: the request may still have been billed.

        The reservation becomes spent and the billed amount stays unknown. This is
        the path a crash recovery takes, and it is deliberately pessimistic: the money
        may have been spent, so the budget must assume it was.
        """
        reservation = self.state.reservations.pop(reservation_id, None)
        if reservation is None:
            raise KeyError(f"no open reservation {reservation_id!r}")
        self.state.spent += reservation.amount
        settlement = Settlement(
            reservation_id=reservation_id,
            billed=None,
            usage=Usage(provider_reported=False),
            price_snapshot_id=reservation.price_snapshot_id,
            released=reservation.amount,
            note=reason,
        )
        self.state.settlements[reservation_id] = settlement
        self.state.usage_missing += 1
        return settlement

    # -- reporting ---------------------------------------------------------

    def check_usage(self, usage: Usage) -> UsageDecision:
        return usage_within_caps(usage, self.limits)

    def bounds_report(self) -> dict[str, Any]:
        """What the ledger can and cannot bound, for an operator to act on."""
        return {
            "bounds_available": {
                "cost": self.limits.max_total_cost_usd is None or self.book.is_complete(),
                "input_tokens": True,
                "output_tokens": True,
                "price_directions": self.book.known(),
            },
            "billed": None if self.billed is None else str(self.billed),
            "limits": self.limits.model_dump(mode="json"),
            "outstanding_reservations": self.outstanding,
            "reserved": str(self.reserved),
            "spent": str(self.state.spent),
        }

    def totals(self) -> dict[str, Any]:
        return {
            "billed": None if self.billed is None else str(self.billed),
            "input_tokens": self.state.input_tokens,
            "output_tokens": self.state.output_tokens,
            "requests": self.requests,
            "spent": str(self.state.spent),
            "usage_missing": self.state.usage_missing,
        }


def priced_usage(
    input_tokens: int | None,
    output_tokens: int | None,
    *,
    reported: bool = True,
) -> Usage:
    """Build a usage report, keeping an unreported direction absent rather than zero."""
    return Usage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        provider_reported=reported,
    )


def quantize(amount: Decimal) -> Decimal:
    """Round a cost for display; comparisons always use the unrounded value."""
    return amount.quantize(_CENT, rounding=ROUND_HALF_UP)


def summed(settlements: Sequence[Settlement]) -> Decimal | None:
    """Total billed across settlements, or ``None`` if any of them is unknown."""
    total = Decimal(0)
    for settlement in settlements:
        if settlement.billed is None:
            return None
        total += settlement.billed
    return total
