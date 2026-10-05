"""Deterministic replay of stored generations.

Replay is the property that makes an offline demonstration possible and a published
score re-checkable. Three guarantees:

* **No external calls.** The reader only touches the log, the artifact store and the
  index. There is no transport, no socket and no credential in this module.
* **Provenance is verified, not assumed.** Every reconstructed result carries the
  manifest hash the event recorded, and a sample whose payload does not verify is
  reported rather than returned.
* **Identical input yields identical output.** Replay is a pure function of the stored
  artifacts, so a regrade cannot quietly differ from the original run.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from stealthbench.schemas.hashing import content_digest
from stealthbench.schemas.results import (
    DeliveryStatus,
    GenerationResult,
    GradeResult,
    RunEvent,
    SampleKey,
    accepted_sample_keys,
)
from stealthbench.storage.events import EventLog
from stealthbench.storage.index import CampaignIndex


class ReplayRejected(Exception):
    """Stored provenance does not support reconstructing the requested samples."""


@dataclass(frozen=True, slots=True)
class ReplayReport:
    """What replay did, and anything that stopped it."""

    results: tuple[GenerationResult, ...] = ()
    events_read: int = 0
    damage: tuple[str, ...] = ()
    rejected: tuple[str, ...] = ()
    integrity: dict[str, Any] = field(default_factory=dict)

    @property
    def accepted_samples(self) -> int:
        return len(accepted_sample_keys(self.results))

    def to_dict(self) -> dict[str, Any]:
        return {
            "events_read": self.events_read,
            "results": len(self.results),
            "accepted_samples": self.accepted_samples,
            "damage": list(self.damage),
            "rejected": list(self.rejected),
            "integrity": self.integrity,
        }


class ReplayReader:
    """Rebuilds generation results from stored artifacts.

    Deliberately constructed from paths only. There is no way to hand it a provider
    client, which is the structural reason replay cannot contact one.
    """

    #: Events whose payload carries a stored generation result.
    #: Every event type whose payload can carry a generation result. A dispatched
    #: attempt is included because it is how an unresolved (dispatched, unconfirmed)
    #: attempt is recorded; excluding it would drop that state from replay entirely.
    RESULT_EVENT_TYPES: tuple[str, ...] = (
        "request.dispatched",
        "request.completed",
        "request.failed",
        "sample.accepted",
    )

    def __init__(self, root: Path, *, extra_secrets: frozenset[str] = frozenset()) -> None:
        self.root = Path(root)
        self.log = EventLog(self.root, extra_secrets=extra_secrets)
        self.extra_secrets = extra_secrets

    def read(self, *, strict: bool = True) -> ReplayReport:
        """Reconstruct every stored result.

        With ``strict`` (the default) any damaged record raises, because a partial
        reconstruction presented as complete is exactly the failure this module
        exists to prevent. With ``strict=False`` the damage is reported instead.
        """
        events, integrity = self.log.read_all()
        results: list[GenerationResult] = []
        rejected: list[str] = []

        for event in events:
            if event.event_type not in self.RESULT_EVENT_TYPES:
                continue
            try:
                results.append(self._result_from(event))
            except ReplayRejected as exc:
                rejected.append(str(exc))

        damage = tuple(str(item.kind) for item in integrity.damage)
        if strict and (damage or rejected):
            raise ReplayRejected(
                "stored provenance does not support a complete replay: "
                f"damage={list(damage)} rejected={rejected}"
            )

        return ReplayReport(
            results=tuple(results),
            events_read=len(events),
            damage=damage,
            rejected=tuple(rejected),
            integrity=integrity.to_dict(),
        )

    def _result_from(self, event: RunEvent) -> GenerationResult:
        """Rebuild one result, refusing anything whose provenance does not check out."""
        payload = self.log.load_payload(event)  # verified by read_all
        if not isinstance(payload, dict):
            raise ReplayRejected(f"event {event.event_id}: payload is not an object")

        sample = payload.get("sample_key")
        if not isinstance(sample, dict):
            raise ReplayRejected(f"event {event.event_id}: payload has no sample_key")
        try:
            key = SampleKey.model_validate(sample)
        except ValidationError as exc:
            raise ReplayRejected(
                f"event {event.event_id}: sample_key is unusable ({exc.error_count()} errors)"
            ) from exc

        # The payload cannot claim a different campaign than the event that carries it.
        # Without this check a crafted payload could attribute a sample to a campaign
        # the event was never part of.
        if key.campaign_id != event.campaign_id:
            raise ReplayRejected(
                f"event {event.event_id}: payload claims campaign {key.campaign_id!r} but the "
                f"event belongs to {event.campaign_id!r}"
            )

        body = payload.get("result", payload)
        if not isinstance(body, dict):
            raise ReplayRejected(f"event {event.event_id}: result is not an object")

        declared_status = payload.get("delivery_status", _default_status(event.event_type))
        if event.event_type == "sample.accepted" and declared_status != str(
            DeliveryStatus.ACCEPTED
        ):
            raise ReplayRejected(
                f"event {event.event_id}: sample.accepted carries delivery_status "
                f"{declared_status!r}"
            )
        if event.event_type == "request.failed" and declared_status != str(
            DeliveryStatus.TRANSPORT_FAILED
        ):
            raise ReplayRejected(
                f"event {event.event_id}: request.failed carries delivery_status "
                f"{declared_status!r}"
            )

        candidate = {
            "attempt_id": payload.get("attempt_id", event.event_id),
            "sample_key": key.model_dump(),
            "attempt_number": payload.get("attempt_number", 1),
            "delivery_status": payload.get("delivery_status", _default_status(event.event_type)),
            "response": body.get("response"),
            "usage": body.get("usage", {}),
            "effective_settings": body.get("effective_settings", {}),
            "streaming": body.get("streaming"),
            "finish_status": body.get("finish_status"),
            "redacted_provider_metadata": body.get("redacted_provider_metadata", {}),
            "manifest_hash": payload.get("manifest_hash"),
        }
        try:
            return GenerationResult.model_validate(candidate)
        except ValidationError as exc:
            raise ReplayRejected(
                f"event {event.event_id}: stored result does not match the contract "
                f"({exc.error_count()} errors)"
            ) from exc

    def grades(
        self,
        results: Sequence[GenerationResult],
        grader: Callable[[GenerationResult], Mapping[str, Any] | GradeResult],
    ) -> dict[str, Any]:
        """Apply an evaluator to stored responses, producing a stable digest.

        No model is involved. The digest covers the canonical projection of the
        grades, so an identical store and grader give an identical result.
        """
        grades: list[dict[str, Any]] = []
        for result in sorted(results, key=lambda r: (r.sample_key.as_tuple(), r.attempt_id)):
            if not result.is_accepted_sample:
                continue
            verdict = grader(result)
            grade = (
                verdict if isinstance(verdict, GradeResult) else GradeResult.model_validate(verdict)
            )
            grades.append(
                {
                    "sample_key": grade.sample_key.model_dump(),
                    "correctness": grade.correctness,
                    "format": grade.format,
                    "transport": grade.transport,
                    "evaluator": grade.evaluator,
                    "score_components": dict(grade.score_components),
                    "denominator_eligibility": grade.denominator_eligibility.model_dump(),
                }
            )
        return {
            "grades": grades,
            "grade_digest": content_digest(grades),
            "accepted_samples": len(grades),
        }

    def stored_results(self) -> list[GenerationResult]:
        """Accepted samples reconstructed from the store, in a stable order."""
        report = self.read(strict=False)
        return sorted(
            (result for result in report.results if result.is_accepted_sample),
            key=lambda result: (result.sample_key.as_tuple(), result.attempt_id),
        )

    def index(self) -> CampaignIndex:
        """Build or refresh the derived SQLite index from the same store."""
        index = CampaignIndex(self.root, extra_secrets=self.extra_secrets)
        index.ingest(self.log, list(self.read(strict=False).results))
        return index


def _default_status(event_type: str) -> str:
    mapping = {
        "request.failed": str(DeliveryStatus.TRANSPORT_FAILED),
        "sample.accepted": str(DeliveryStatus.ACCEPTED),
    }
    return mapping.get(event_type, str(DeliveryStatus.ACCEPTED))


def replay_digest(results: Sequence[GenerationResult]) -> str:
    """A stable digest over the accepted-sample projection of a replay."""
    projection = [
        {
            "sample_key": list(result.sample_key.as_tuple()),
            "attempt_id": result.attempt_id,
            "response": result.response,
            "input_tokens": result.usage.input_tokens,
            "output_tokens": result.usage.output_tokens,
        }
        for result in sorted(results, key=lambda r: (r.sample_key.as_tuple(), r.attempt_id))
        if result.is_accepted_sample
    ]
    return content_digest(projection)


def write_fixture_transcript(
    root: Path,
    *,
    extra_secrets: frozenset[str] = frozenset(),
    campaign_id: str,
    item_id: str,
    endpoint_id: str,
    response: str,
    benchmark_id: str = "ifeval",
    repeat_id: int = 1,
    input_tokens: int | None = 10,
    output_tokens: int | None = 20,
    manifest_hash: str | None = None,
) -> Path:
    """Record one synthetic generation into a fixture log.

    This is how an offline demonstration is built: responses are recorded once as
    artifacts and every later run reads them back through replay.
    """
    log = EventLog(root, extra_secrets=extra_secrets)
    key = SampleKey(
        campaign_id=campaign_id,
        endpoint_id=endpoint_id,
        task_id=f"{benchmark_id}::{item_id}",
        repeat_id=repeat_id,
    )
    log.append(
        "sample.accepted",
        campaign_id,
        {
            "attempt_id": f"{key.task_id}-r{repeat_id}",
            "attempt_number": 1,
            "sample_key": key.model_dump(),
            "manifest_hash": manifest_hash,
            "result": {
                "response": response,
                "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
                "effective_settings": {"temperature": 0.0, "top_p": 1.0},
                "finish_status": "stop",
                "redacted_provider_metadata": {"transport": "fixture"},
            },
        },
    )
    return log.path


def describe_store(root: Path) -> dict[str, Any]:
    """A summary of a store, for an operator or a report."""
    log = EventLog(root)
    events, integrity = log.read_all()
    return {
        "root": str(root),
        "event_count": len(events),
        "completion_count": sum(1 for event in events if event.is_completion_record),
        "integrity": integrity.to_dict(),
        "canonical_event_digest": content_digest(
            [event.model_dump(mode="json") for event in events]
        ),
    }


__all__ = [
    "ReplayReader",
    "ReplayRejected",
    "ReplayReport",
    "describe_store",
    "replay_digest",
    "write_fixture_transcript",
]
