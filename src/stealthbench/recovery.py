"""Crash recovery and accepted-sample accounting.

A dispatched request whose response never reached durable storage is **ambiguous**:
it may have been billed, and it may have produced an answer that is simply gone. The
default recovery does not guess. It records the attempt as unresolved and leaves the
budget charged, because both possibilities are real and only one of them is free.

Three rules make recovery safe to run twice:

* **Unresolved by default.** A crash between dispatch and storage leaves an attempt
  unresolved. Recovery never resubmits it on its own.
* **An explicit retry keeps the history.** Retrying an unresolved attempt does not
  erase the first one: its reservation, its cost and its duplicate risk are all still
  in the record, and the retry is marked as a retry of a specific attempt.
* **Resume cannot duplicate an accepted sample.** An accepted sample is keyed by
  (campaign, endpoint, task, repeat) and stays accepted; nothing recovery does can
  produce a second one for the same key.

Retries and duplicates are kept as distinct concepts throughout. A retry is another
attempt at one sample; a duplicate would be two accepted samples for one sample key,
which the accepted-sample accounting is what prevents.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from stealthbench.costs import Ledger, LedgerState, Reservation
from stealthbench.schemas.results import (
    GenerationResult,
    SampleKey,
    Usage,
    accepted_sample_keys,
    unresolved_attempts,
)

__all__ = [
    "AcceptedSampleIndex",
    "DuplicateRisk",
    "RecoveryReport",
    "ResolvedAttempt",
    "UnresolvedAttempt",
    "recover",
]


@dataclass(frozen=True, slots=True)
class UnresolvedAttempt:
    """A dispatch whose outcome is not known.

    ``may_have_been_billed`` is the reason this state exists. The attempt is not
    dropped and not retried: it is recorded, and the budget keeps the charge.
    """

    attempt_id: str
    sample_key: SampleKey | None
    attempt_number: int
    reason: str
    reserved: float | None
    may_have_been_billed: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempt_id": self.attempt_id,
            "attempt_number": self.attempt_number,
            "may_have_been_billed": self.may_have_been_billed,
            "reason": self.reason,
            "reserved": self.reserved,
            "sample_key": None if self.sample_key is None else list(self.sample_key.as_tuple()),
        }


@dataclass(frozen=True, slots=True)
class ResolvedAttempt:
    """An attempt that came back, one way or the other."""

    attempt_id: str
    sample_key: SampleKey
    attempt_number: int
    accepted: bool
    finish_status: str | None
    usage: Usage

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "attempt_id": self.attempt_id,
            "attempt_number": self.attempt_number,
            "finish_status": self.finish_status,
            "sample_key": None if self.sample_key is None else list(self.sample_key.as_tuple()),
            "usage": self.usage.model_dump(mode="json"),
        }


class DuplicateRisk(StrEnum):
    """What an explicit retry risks, stated rather than assumed."""

    NONE = "none"
    MAY_BE_BILLED_TWICE = "may_be_billed_twice"
    ALREADY_ACCEPTED = "already_accepted"


@dataclass(frozen=True, slots=True)
class RetryAuthorization:
    """Permission to resubmit a specific attempt, with its risk disclosed."""

    authorized: bool
    reason: str
    risk: DuplicateRisk
    retry_of: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "authorized": self.authorized,
            "reason": self.reason,
            "retry_of": self.retry_of,
            "risk": str(self.risk),
        }


class AcceptedSampleIndex:
    """Which samples are accepted, so recovery cannot create a second one.

    Built from results rather than from bookkeeping, so it cannot disagree with the
    results themselves. Attempt ids are kept alongside because an unresolved attempt
    has no result at all.
    """

    def __init__(self, results: Iterable[GenerationResult] = ()) -> None:
        self._accepted: dict[tuple[str, str, str, int], GenerationResult] = {}
        self._attempts: dict[str, tuple[SampleKey, int]] = {}
        for result in results:
            self.add(result)

    def add(self, result: GenerationResult) -> None:
        key = result.sample_key.as_tuple()
        if result.is_accepted_sample:
            # First acceptance wins: a later attempt for the same key is a duplicate,
            # and recording it here would inflate the denominator.
            self._accepted.setdefault(key, result)
        self._attempts.setdefault(result.attempt_id, (result.sample_key, result.attempt_number))

    @property
    def keys(self) -> set[tuple[str, str, str, int]]:
        return set(self._accepted)

    @property
    def attempt_ids(self) -> frozenset[str]:
        """Every attempt id seen, whether or not its sample was accepted."""
        return frozenset(self._attempts)

    @property
    def results(self) -> tuple[GenerationResult, ...]:
        return tuple(self._accepted[key] for key in sorted(self._accepted))

    def __len__(self) -> int:
        return len(self._accepted)

    def is_accepted(self, key: SampleKey) -> bool:
        return key.as_tuple() in self._accepted

    def attempt_for(self, sample_key: SampleKey, attempt_number: int) -> str | None:
        for attempt_id, (key, number) in self._attempts.items():
            if key.as_tuple() == sample_key.as_tuple() and number == attempt_number:
                return attempt_id
        return None

    def authorize_retry(
        self, sample_key: SampleKey, attempt_number: int, *, max_attempts: int
    ) -> RetryAuthorization:
        """Whether one specific attempt may be submitted again, and at what risk."""
        if self.is_accepted(sample_key):
            return RetryAuthorization(
                False,
                "this sample is already accepted; another attempt would duplicate it",
                DuplicateRisk.ALREADY_ACCEPTED,
            )
        if attempt_number >= max_attempts:
            return RetryAuthorization(
                False,
                f"attempt {attempt_number} is the last the policy allows",
                DuplicateRisk.NONE,
            )
        return RetryAuthorization(
            True,
            "the sample has no accepted answer; the earlier attempt may still have been billed",
            DuplicateRisk.MAY_BE_BILLED_TWICE,
        )


@dataclass(frozen=True, slots=True)
class RecoveryReport:
    """What a recovery pass found and what it changed.

    ``resubmitted`` is always empty: this module never resubmits on its own. An
    explicit retry is a separate, authorised act recorded elsewhere.
    """

    campaign_id: str
    accepted_sample_keys: tuple[tuple[str, str, str, int], ...]
    unresolved: tuple[UnresolvedAttempt, ...]
    charged_ambiguous: float
    open_reservations: int
    resubmitted: tuple[str, ...] = ()

    @property
    def accepted_count(self) -> int:
        return len(self.accepted_sample_keys)

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted_count": self.accepted_count,
            "accepted_sample_keys": [list(key) for key in self.accepted_sample_keys],
            "campaign_id": self.campaign_id,
            "charged_ambiguous": str(self.charged_ambiguous),
            "open_reservations": self.open_reservations,
            "resubmitted": list(self.resubmitted),
            "unresolved": [attempt.to_dict() for attempt in self.unresolved],
        }


@dataclass(slots=True)
class _Recovery:
    ledger: Ledger
    index: AcceptedSampleIndex
    campaign_id: str
    unresolved: dict[str, UnresolvedAttempt] = field(default_factory=dict)
    charged: float = 0.0


def recover(
    *,
    campaign_id: str,
    ledger: Ledger,
    results: Iterable[GenerationResult] = (),
    attempt_records: Mapping[str, Mapping[str, Any]] | None = None,
) -> RecoveryReport:
    """Reconcile a ledger and its results after a crash.

    An attempt record is the durable note written *before* a dispatch goes out. If one
    exists for an attempt that never produced a result, the request was dispatched and
    its outcome is unknown: it is recorded unresolved and its reservation is charged,
    because the money may have been spent.

    Nothing is resubmitted. An explicit retry needs
    :meth:`AcceptedSampleIndex.authorize_retry`, which discloses the duplicate risk.
    """
    index = AcceptedSampleIndex(results)
    records = dict(attempt_records or {})
    state: _Recovery = _Recovery(ledger=ledger, index=index, campaign_id=campaign_id)

    for attempt_id, record in sorted(records.items()):
        _resolve(state, attempt_id, record)

    _charge_open_reservations(state)

    return RecoveryReport(
        campaign_id=campaign_id,
        accepted_sample_keys=tuple(sorted(index.keys)),
        unresolved=tuple(state.unresolved[key] for key in sorted(state.unresolved)),
        charged_ambiguous=state.charged,
        open_reservations=ledger.outstanding,
        resubmitted=(),
    )


def _resolve(state: _Recovery, attempt_id: str, record: Mapping[str, Any]) -> None:
    if attempt_id in state.index.attempt_ids:
        # The attempt came back: its result is already in the index.
        return
    sample_key = _sample_key_of(record)
    reservation_id = record.get("reservation_id")
    reserved = _amount_of(state.ledger, reservation_id)
    if sample_key is not None and state.index.is_accepted(sample_key):
        reason = "the sample is already accepted, so this attempt is a duplicate, not a sample"
        risk_billed = False
    else:
        reason = str(record.get("reason", "dispatched with no durable response"))
        risk_billed = True
    if sample_key is None:
        # A record with no usable sample key still names an unresolved attempt; the
        # key is left empty rather than invented, because an invented endpoint id
        # would read as a dispatched observation.
        state.unresolved[attempt_id] = UnresolvedAttempt(
            attempt_id=attempt_id,
            sample_key=None,
            attempt_number=int(record.get("attempt_number", 1)),
            reason=reason,
            reserved=reserved,
            may_have_been_billed=risk_billed,
        )
        return
    state.unresolved[attempt_id] = UnresolvedAttempt(
        attempt_id=attempt_id,
        sample_key=sample_key,
        attempt_number=int(record.get("attempt_number", 1)),
        reason=reason,
        reserved=reserved,
        may_have_been_billed=risk_billed,
    )


def _charge_open_reservations(state: _Recovery) -> None:
    """Charge every reservation whose attempt never produced a response."""
    for reservation_id in sorted(state.ledger.state.reservations):
        settlement = state.ledger.forfeit(
            reservation_id,
            reason="no durable response was recorded; the request may have been billed",
        )
        state.charged += float(settlement.released)


def _sample_key_of(record: Mapping[str, Any]) -> SampleKey | None:
    raw = record.get("sample_key")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or len(raw) != 4:
        return None
    campaign, endpoint, task, repeat = raw
    try:
        return SampleKey(
            campaign_id=str(campaign),
            endpoint_id=str(endpoint),
            task_id=str(task),
            repeat_id=int(repeat),
        )
    except (ValueError, TypeError):
        return None


def _amount_of(ledger: Ledger, reservation_id: Any) -> float | None:
    if not isinstance(reservation_id, str):
        return None
    reservation = ledger.state.reservations.get(reservation_id)
    return None if reservation is None else float(reservation.amount)


def open_reservations_of(state: LedgerState) -> tuple[Reservation, ...]:
    """The reservations a restart must still respect."""
    return tuple(state.reservations[k] for k in sorted(state.reservations))


def reconcile_accepted(
    results: Sequence[GenerationResult],
) -> tuple[set[tuple[str, str, str, int]], tuple[str, ...]]:
    """Accepted sample keys and unresolved attempt ids, from results alone."""
    return accepted_sample_keys(results), unresolved_attempts(results)


def result_record(
    result: GenerationResult,
) -> dict[str, Any]:
    """The durable record of an attempt that came back."""
    return {
        "attempt_id": result.attempt_id,
        "attempt_number": result.attempt_number,
        "delivery_status": str(result.delivery_status),
        "sample_key": list(result.sample_key.as_tuple()),
        "usage": result.usage.model_dump(mode="json"),
    }


def unresolved_record(attempt: UnresolvedAttempt) -> dict[str, Any]:
    return attempt.to_dict()


def report_json(report: RecoveryReport) -> str:
    return json.dumps(report.to_dict(), sort_keys=True)
