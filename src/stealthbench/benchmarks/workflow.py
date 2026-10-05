"""Offline run / grade / replay / export workflow (G05 T05C).

The vertical slice: a fixture-only campaign runs to completion, grades with the
official IFEval wrapper, persists redacted events, and replays without a
provider. Three structural guarantees:

* **No provider socket.** Dispatch goes through ``FixtureTransport`` only.
  Replay goes through ``ReplayReader`` only, which cannot hold a client.
* **Caps before dispatch.** Every dispatch reserves worst-case cost first via
  ``Ledger``; only retryable transport failures retry via ``scheduler``; one
  accepted sample per key via ``AcceptedSampleIndex`` and resume skipping.
* **A dry run generates nothing.** ``dry_run=True`` validates and returns a
  report without creating events, artifacts, or exports.

Offline fixture cost is zero: the fixture transport performs no billed request,
so the ledger uses an explicit zero price book. Request, token, and
concurrency caps are still enforced; an unknown size still blocks.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any, Final

from stealthbench.adapters.base import (
    FailureKind,
    FixtureBundle,
    FixtureTransport,
    RecordedExchange,
    TransportFailure,
    failed_result,
    safe_failure,
)
from stealthbench.benchmarks.datasets import DatasetItem, build_dispatch_tasks
from stealthbench.benchmarks.ifeval import (
    IfEvalGradePair,
    grade_generation,
    summarize_pairs,
)
from stealthbench.costs import Ledger, LedgerState, PriceBook
from stealthbench.recovery import AcceptedSampleIndex, recover
from stealthbench.runner import RunMode, run_campaign
from stealthbench.scheduler import build_plan
from stealthbench.schemas.campaign import (
    CampaignManifest,
    Capabilities,
    PricingSnapshot,
)
from stealthbench.schemas.hashing import canonical_json, content_digest
from stealthbench.schemas.manifest import manifest_hash
from stealthbench.schemas.results import (
    GenerationResult,
    SampleKey,
    accepted_sample_keys,
)
from stealthbench.storage.events import EventLog, redact_mapping
from stealthbench.storage.index import CampaignIndex
from stealthbench.storage.replay import ReplayReader

__all__ = [
    "OfflineReport",
    "ReplaySummary",
    "build_default_fixture_bundle",
    "default_check",
    "export_redacted_jsonl",
    "load_fixture_bundle",
    "replay_offline",
    "run_offline",
    "synthetic_items_for_manifest",
]

#: Zero-price book for fixture transport. Explicit zeros, never nulls: a fixture
#: dispatch performs no billed request, so its worst-case cost is genuinely 0.
_FIXTURE_BOOK: Final[PriceBook] = PriceBook(
    snapshot=PricingSnapshot(
        currency="USD",
        input_per_mtok=0.0,
        output_per_mtok=0.0,
        cached_input_per_mtok=0.0,
        reasoning_per_mtok=0.0,
        snapshot_id="fixture-zero",
    )
)

#: Keyword the default synthetic prompts and checks agree on.
_DEFAULT_KEYWORD: Final[str] = "blueberry"


class _OfflineClock:
    """Injected clock for offline runs. Never sleeps: backoff is recorded, not waited."""

    def __init__(self) -> None:
        self.slept: float = 0.0

    def monotonic(self) -> float:
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        self.slept += max(0.0, seconds)


def default_check(response: str) -> bool:
    """Default IFEval stand-in: the response contains the task keyword."""
    return _DEFAULT_KEYWORD in response.lower()


def _estimate_input_tokens(prompt: str) -> int:
    """Deterministic input-token bound: word count plus framing, at least 1."""
    return max(1, len(prompt.split()) + 8)


def synthetic_items_for_manifest(
    manifest: CampaignManifest,
) -> dict[str, tuple[DatasetItem, ...]]:
    """Build deterministic synthetic items from a manifest's frozen ``item_ids``.

    Each prompt instructs the model to include the default keyword, so the
    default check grades default fixture responses as compliant. Gold stays
    evaluator-only (kept in ``DatasetItem.gold_answer``, never in the prompt).
    """
    out: dict[str, tuple[DatasetItem, ...]] = {}
    for spec in manifest.benchmarks:
        items: list[DatasetItem] = []
        for item_id in spec.item_ids:
            prompt = (
                f"Instruction for {spec.benchmark_id} {item_id}: "
                f"write one sentence containing the word {_DEFAULT_KEYWORD}."
            )
            # Gold is evaluator-only and must not appear in the prompt: use an
            # item code rather than the keyword itself.
            items.append(
                DatasetItem(
                    item_id=item_id,
                    prompt=prompt,
                    gold_answer=f"expected-{spec.benchmark_id}-{item_id}",
                    metadata={"synthetic": True, "benchmark_id": spec.benchmark_id},
                )
            )
        out[spec.benchmark_id] = tuple(items)
    return out


def build_default_fixture_bundle(
    manifest: CampaignManifest,
    tasks_by_id: Mapping[str, Any] | None = None,
) -> FixtureBundle:
    """Build a compliant default bundle: every planned item answers once.

    Responses contain the default keyword so default grading passes. Usage is
    reported so token caps apply. Used when ``stealthbench run --offline`` is
    invoked without an explicit ``--fixture`` file.
    """
    del tasks_by_id
    capabilities = (
        manifest.endpoints[0].capabilities
        if manifest.endpoints
        else Capabilities(
            streaming=False,
            tool_calls=False,
            reasoning=False,
            usage_reporting=True,
            logprobs=False,
        )
    )
    exchanges: list[RecordedExchange] = []
    for spec in manifest.benchmarks:
        for endpoint in manifest.endpoints:
            for item_id in spec.item_ids:
                for repeat_id in range(1, spec.repeats + 1):
                    prompt_words = 16 + len(item_id)
                    exchanges.append(
                        RecordedExchange(
                            endpoint_id=endpoint.endpoint_id,
                            benchmark_id=spec.benchmark_id,
                            item_id=item_id,
                            repeat_id=repeat_id,
                            outcome="response",
                            response=(
                                f"A compliant answer for {item_id} "
                                f"containing the word {_DEFAULT_KEYWORD}."
                            ),
                            usage={
                                "input_tokens": prompt_words,
                                "output_tokens": 12,
                            },
                            usage_reported=True,
                            effective_settings={"temperature": 0.0, "top_p": 1.0},
                            finish_status="stop",
                        )
                    )
    return FixtureBundle(
        name=f"{manifest.campaign_id}-default-fixture",
        catalog_source="fixture",
        catalog={},
        exchanges=tuple(exchanges),
        capabilities=capabilities,
    )


def load_fixture_bundle(path: Path) -> FixtureBundle:
    """Load a recorded fixture bundle from disk (offline: file read only)."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return FixtureBundle.model_validate(raw)


@dataclass(slots=True)
class OfflineReport:
    """What an offline run did, with explicit denominators."""

    campaign_id: str
    artifacts_dir: str
    manifest_hash: str
    planned: int
    eligible: int
    skipped: int
    dispatched: int
    accepted_samples: int
    transport_failed: int
    strict_correct: int
    strict_eligible: int
    strict_accuracy: float | None
    loose_correct: int
    loose_eligible: int
    loose_accuracy: float | None
    grade_digest: str
    failures: tuple[str, ...] = ()
    adapter_calls: int = 0
    dry_run: bool = False
    generated_samples: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "campaign_id": self.campaign_id,
            "artifacts_dir": self.artifacts_dir,
            "manifest_hash": self.manifest_hash,
            "planned": self.planned,
            "eligible": self.eligible,
            "skipped": self.skipped,
            "dispatched": self.dispatched,
            "accepted_samples": self.accepted_samples,
            "transport_failed": self.transport_failed,
            "strict_correct": self.strict_correct,
            "strict_eligible": self.strict_eligible,
            "strict_accuracy": self.strict_accuracy,
            "loose_correct": self.loose_correct,
            "loose_eligible": self.loose_eligible,
            "loose_accuracy": self.loose_accuracy,
            "grade_digest": self.grade_digest,
            "failures": list(self.failures),
            "adapter_calls": self.adapter_calls,
            "dry_run": self.dry_run,
            "generated_samples": self.generated_samples,
        }


@dataclass(slots=True)
class ReplaySummary:
    """What a replay reconstructed, without contacting a provider."""

    campaign_id: str
    artifacts_dir: str
    accepted_samples: int
    grade_digest: str
    strict_accuracy: float | None
    loose_accuracy: float | None
    events_read: int
    damage: tuple[str, ...] = field(default_factory=tuple)
    rejected: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "campaign_id": self.campaign_id,
            "artifacts_dir": self.artifacts_dir,
            "accepted_samples": self.accepted_samples,
            "grade_digest": self.grade_digest,
            "strict_accuracy": self.strict_accuracy,
            "loose_accuracy": self.loose_accuracy,
            "events_read": self.events_read,
            "damage": list(self.damage),
            "rejected": list(self.rejected),
        }


def _tasks_for_manifest(
    manifest: CampaignManifest,
    items_by_benchmark: Mapping[str, Sequence[DatasetItem]],
) -> tuple[Any, ...]:
    """Build TaskSpecs for every endpoint in frozen ``item_ids`` order."""
    tasks: list[Any] = []
    for endpoint in manifest.endpoints:
        tasks.extend(
            build_dispatch_tasks(
                manifest,
                items_by_benchmark=items_by_benchmark,
                endpoint_id=endpoint.endpoint_id,
                max_output_tokens=manifest.generation.max_output_tokens,
            )
        )
    return tuple(tasks)


def _grade_all(
    tasks_by_key: Mapping[tuple[str, str, str, int], Any],
    generations: Sequence[GenerationResult],
    checks: Mapping[str, Any] | None,
) -> list[IfEvalGradePair]:
    """Grade every generation, keeping transport failures out of accuracy."""
    pairs: list[IfEvalGradePair] = []
    for generation in sorted(generations, key=lambda g: (g.sample_key.as_tuple(), g.attempt_id)):
        task = tasks_by_key.get(generation.sample_key.as_tuple())
        if task is None:
            continue
        check = (checks or {}).get(generation.sample_key.task_id, default_check)
        pairs.append(grade_generation(task=task, generation=generation, check=check))
    return pairs


def export_redacted_jsonl(
    path: Path,
    records: Sequence[Mapping[str, Any]],
    *,
    extra_secrets: frozenset[str] = frozenset(),
) -> Path:
    """Write redacted JSONL atomically: one object per line, secrets removed."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for record in records:
        safe = redact_mapping(dict(record), extra_secrets=extra_secrets)
        lines.append(json.dumps(safe, sort_keys=True, ensure_ascii=False))
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    tmp.replace(target)
    return target


def _persist_results(
    log: EventLog,
    manifest: CampaignManifest,
    digest: str,
    generations: Sequence[GenerationResult],
) -> None:
    """Durably record each generation: accepted as samples, else as failures."""
    for generation in sorted(generations, key=lambda g: (g.sample_key.as_tuple(), g.attempt_id)):
        key = generation.sample_key
        payload: dict[str, Any] = {
            "attempt_id": generation.attempt_id,
            "attempt_number": generation.attempt_number,
            "sample_key": key.model_dump(mode="json"),
            "manifest_hash": digest,
            "delivery_status": str(generation.delivery_status),
            "result": {
                "response": generation.response,
                "usage": generation.usage.model_dump(mode="json"),
                "effective_settings": dict(generation.effective_settings),
                "finish_status": generation.finish_status,
                "redacted_provider_metadata": dict(generation.redacted_provider_metadata),
            },
        }
        if generation.is_accepted_sample:
            log.append("sample.accepted", manifest.campaign_id, payload)
        else:
            log.append("request.failed", manifest.campaign_id, payload)


def run_offline(
    manifest: CampaignManifest,
    *,
    artifacts_dir: Path,
    fixture_bundle: FixtureBundle | None = None,
    items_by_benchmark: Mapping[str, Sequence[DatasetItem]] | None = None,
    checks: Mapping[str, Any] | None = None,
    dry_run: bool = False,
    already_accepted: Iterable[SampleKey] = (),
    extra_secrets: frozenset[str] = frozenset(),
) -> OfflineReport:
    """Run a fixture-only campaign, grade it, and export redacted artifacts.

    Reserve-before-dispatch, retryable-only retries, and one-accepted-sample
    semantics come from ``runner.run_campaign``; this function supplies the
    offline inputs (synthetic items, zero-price ledger, fixture transport) and
    persists the durable record. ``dry_run=True`` validates without writing.
    """
    root = Path(artifacts_dir)
    digest = manifest_hash(manifest)
    items = (
        dict(items_by_benchmark)
        if items_by_benchmark is not None
        else synthetic_items_for_manifest(manifest)
    )
    tasks = _tasks_for_manifest(manifest, items)
    tasks_by_key = {t.sample_key.as_tuple(): t for t in tasks}

    bundle = (
        fixture_bundle if fixture_bundle is not None else build_default_fixture_bundle(manifest)
    )
    adapter = FixtureTransport(bundle, extra_secrets=extra_secrets)

    # Resume: previously accepted samples are skipped, never re-dispatched.
    # When resuming into an existing store, pick up its accepted keys unless the
    # caller supplied an explicit set.
    accepted_list = list(already_accepted)
    ledger_state: LedgerState | None = None
    if not accepted_list and root.joinpath("events.jsonl").exists():
        try:
            prior = ReplayReader(root, extra_secrets=extra_secrets).read(strict=False)
            accepted_list = [
                GenerationResult.model_validate(
                    {
                        "attempt_id": r.attempt_id,
                        "sample_key": r.sample_key.model_dump(),
                        "attempt_number": r.attempt_number,
                        "delivery_status": str(r.delivery_status),
                        "response": r.response,
                        "usage": r.usage.model_dump(mode="json"),
                        "effective_settings": dict(r.effective_settings),
                        "finish_status": r.finish_status,
                        "redacted_provider_metadata": dict(r.redacted_provider_metadata),
                        "manifest_hash": r.manifest_hash,
                    }
                ).sample_key
                for r in prior.results
                if r.is_accepted_sample
            ]
        except Exception:
            accepted_list = []
    ledger_file = root / "ledger.json"
    if ledger_file.exists():
        try:
            ledger_state = LedgerState.from_dict(
                json.loads(ledger_file.read_text(encoding="utf-8"))
            )
        except Exception:
            ledger_state = None

    plan = build_plan(
        campaign_id=manifest.campaign_id,
        tasks=tasks,
        policy=manifest.retry_policy,
        seed=manifest.seed,
        already_accepted=accepted_list,
    )

    request_map = {
        (t.endpoint_id, f"{t.benchmark_id}::{t.item_id}"): t.safe_request() for t in tasks
    }
    # prompt_tokens keyed the same way the runner looks them up.
    prompt_tokens: dict[tuple[str, str], int | None] = {}
    for t in tasks:
        content = " ".join(m.content for m in t.request.messages)
        prompt_tokens[(t.endpoint_id, f"{t.benchmark_id}::{t.item_id}")] = _estimate_input_tokens(
            content
        )

    ledger = Ledger(limits=manifest.limits, book=_FIXTURE_BOOK, state=ledger_state)
    clock = _OfflineClock()
    attempt_records: dict[str, dict[str, Any]] = {}

    def _on_record(task_id: str, record: Mapping[str, Any]) -> None:
        raw_key = record.get("sample_key")
        attempt_no = int(record.get("attempt_number", 1))
        key_str = str(task_id)
        attempt_id = f"{key_str}-a{attempt_no}"
        if isinstance(raw_key, (list, tuple)) and len(raw_key) == 4:
            attempt_id = f"{raw_key[2]}-r{raw_key[3]}-a{attempt_no}"
        attempt_records[attempt_id] = {
            "sample_key": list(raw_key) if isinstance(raw_key, (list, tuple)) else raw_key,
            "attempt_number": attempt_no,
            "reservation_id": record.get("reservation_id"),
            "reason": "dispatched with no durable response",
        }

    from stealthbench.schemas.campaign import Authorization

    outcome = run_campaign(
        campaign_id=manifest.campaign_id,
        mode=RunMode.OFFLINE,
        limits=manifest.limits,
        authorization=manifest.authorization or Authorization(required=False),
        plan=plan,
        tasks=request_map,
        prompt_tokens=prompt_tokens,
        ledger=ledger,
        adapter=adapter,
        clock=clock,
        dry_run=dry_run,
        capabilities=manifest.endpoints[0].capabilities if manifest.endpoints else None,
        on_record=_on_record,
    )

    if dry_run:
        # Validates without generating: no files, no samples, no dispatches.
        return OfflineReport(
            campaign_id=manifest.campaign_id,
            artifacts_dir=str(root),
            manifest_hash=digest,
            planned=len(plan.eligible),
            eligible=len(plan.eligible),
            skipped=len(plan.skipped),
            dispatched=0,
            accepted_samples=0,
            transport_failed=0,
            strict_correct=0,
            strict_eligible=0,
            strict_accuracy=None,
            loose_correct=0,
            loose_eligible=0,
            loose_accuracy=None,
            grade_digest=content_digest([]),
            failures=tuple(getattr(outcome, "blockers", ())),
            adapter_calls=0,
            dry_run=True,
            generated_samples=0,
        )

    from stealthbench.runner import DryRunReport as _DryRunReport

    assert not isinstance(outcome, _DryRunReport), "offline run must produce a RunReport"
    results: list[GenerationResult] = list(outcome.results)

    # Synthesize transport-failed generations for planned samples with no
    # accepted result, so failure denominators are explicit and replayable.
    accepted = accepted_sample_keys(results)
    supplemented: list[GenerationResult] = list(results)
    bundle_by_key: dict[tuple[str, str, str, int], RecordedExchange] = {}
    for exchange in bundle.exchanges:
        task_id = f"{exchange.benchmark_id}::{exchange.item_id}"
        key = (
            manifest.campaign_id,
            exchange.endpoint_id,
            task_id,
            exchange.repeat_id,
        )
        bundle_by_key[key] = exchange
    for item in plan.eligible:
        key = item.sample_key.as_tuple()
        if key in accepted:
            continue
        recorded = bundle_by_key.get(key)
        detail = "no recorded exchange produced an accepted sample"
        kind: FailureKind = FailureKind.NO_FIXTURE
        body: Mapping[str, Any] | None = None
        http_status: int | None = None
        retry_after: float | None = None
        if recorded is not None and recorded.outcome == "error":
            detail = recorded.failure_detail or detail
            try:
                kind = FailureKind(recorded.failure_kind or "server_error")
            except ValueError:
                kind = FailureKind.SERVER_ERROR
            body = recorded.error_body
            http_status = recorded.http_status
            retry_after = recorded.retry_after_seconds
        failure: TransportFailure = safe_failure(
            kind,
            detail,
            http_status=http_status,
            retry_after_seconds=retry_after,
            body=body,
            extra_secrets=extra_secrets,
        )
        task = tasks_by_key.get(key)
        attempt_no = 1
        attempt_id = f"{key[2]}-r{key[3]}-a{attempt_no}"
        sample_key = (
            task.sample_key
            if task is not None
            else SampleKey(
                campaign_id=key[0],
                endpoint_id=key[1],
                task_id=key[2],
                repeat_id=key[3],
            )
        )
        supplemented.append(
            failed_result(
                sample_key,
                attempt_id,
                attempt_no,
                failure,
                manifest_hash=digest,
            )
        )

    # One accepted sample per key, even if the runner or a resumed store
    # somehow produced duplicates: first acceptance wins.
    index = AcceptedSampleIndex(s for s in supplemented if s.is_accepted_sample)
    deduped_accepted = list(index.results)
    failed_only = [g for g in supplemented if not g.is_accepted_sample]
    # Keep failed generations whose sample key was never accepted.
    failed_kept = [g for g in failed_only if g.sample_key.as_tuple() not in index.keys]
    all_generations = [*deduped_accepted, *failed_kept]

    pairs = _grade_all(tasks_by_key, all_generations, checks)
    summary = summarize_pairs(pairs)
    grade_digest = content_digest(
        [
            {
                "sample_key": list(p.strict.sample_key.as_tuple()),
                "strict": p.strict.correctness,
                "loose": p.loose.correctness,
            }
            for p in sorted(pairs, key=lambda q: q.strict.sample_key.as_tuple())
        ]
    )

    # Durable record: events first, then derived index, then exports.
    log = EventLog(root, extra_secrets=extra_secrets)
    _persist_results(log, manifest, digest, all_generations)
    for pair in sorted(pairs, key=lambda p: p.strict.sample_key.as_tuple()):
        log.append(
            "grade.recorded",
            manifest.campaign_id,
            {
                "sample_key": pair.strict.sample_key.model_dump(mode="json"),
                "grader_version": pair.strict.grader_version,
                "strict": pair.strict.model_dump(mode="json"),
                "loose": pair.loose.model_dump(mode="json"),
            },
        )
    recovery_report = recover(
        campaign_id=manifest.campaign_id,
        ledger=ledger,
        results=deduped_accepted,
        attempt_records=attempt_records,
    )
    log.append(
        "campaign.completed",
        manifest.campaign_id,
        {
            "manifest_hash": digest,
            "plan_digest": plan.digest(),
            "accepted_samples": len(deduped_accepted),
            "grade_digest": grade_digest,
            "recovery": recovery_report.to_dict(),
            "ledger": ledger.snapshot().to_dict(),
        },
    )

    index_db = CampaignIndex(root, extra_secrets=extra_secrets)
    index_db.ingest(log, deduped_accepted)

    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(
        canonical_json(manifest.model_dump(mode="json")) + "\n", encoding="utf-8"
    )
    (root / "ledger.json").write_text(
        canonical_json(ledger.snapshot().to_dict()) + "\n", encoding="utf-8"
    )
    grade_records: list[dict[str, Any]] = []
    for pair in sorted(pairs, key=lambda p: p.strict.sample_key.as_tuple()):
        grade_records.append(
            {
                "sample_key": pair.strict.sample_key.model_dump(mode="json"),
                "grader_version": pair.strict.grader_version,
                "strict_correctness": pair.strict.correctness,
                "loose_correctness": pair.loose.correctness,
                "transport": pair.strict.transport,
                "evaluator": pair.strict.evaluator,
                "score_components": {
                    **dict(pair.strict.score_components),
                    **dict(pair.loose.score_components),
                },
            }
        )
    export_redacted_jsonl(root / "grades.jsonl", grade_records, extra_secrets=extra_secrets)

    sample_records: list[dict[str, Any]] = []
    for generation in sorted(
        all_generations, key=lambda g: (g.sample_key.as_tuple(), g.attempt_id)
    ):
        sample_records.append(json.loads(generation.model_dump_json()))
    export_redacted_jsonl(root / "export.jsonl", sample_records, extra_secrets=extra_secrets)

    (root / "summary.json").write_text(
        canonical_json(
            {
                "campaign_id": manifest.campaign_id,
                "manifest_hash": digest,
                "plan_digest": plan.digest(),
                "planned": len(plan.eligible) + len(plan.skipped),
                "eligible": len(plan.eligible),
                "skipped": len(plan.skipped),
                "dispatched": adapter.call_count(),
                "accepted_samples": len(deduped_accepted),
                "transport_failed": len(failed_kept),
                "strict_correct": summary.strict_correct,
                "strict_eligible": summary.strict_eligible,
                "strict_accuracy": summary.strict_accuracy,
                "loose_correct": summary.loose_correct,
                "loose_eligible": summary.loose_eligible,
                "loose_accuracy": summary.loose_accuracy,
                "grade_digest": grade_digest,
                "ledger": ledger.totals(),
                "failures": list(getattr(outcome, "failures", ())),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    transport_failed = len(failed_kept)
    return OfflineReport(
        campaign_id=manifest.campaign_id,
        artifacts_dir=str(root),
        manifest_hash=digest,
        planned=len(plan.eligible) + len(plan.skipped),
        eligible=len(plan.eligible),
        skipped=len(plan.skipped),
        dispatched=adapter.call_count(),
        accepted_samples=len(deduped_accepted),
        transport_failed=transport_failed,
        strict_correct=summary.strict_correct,
        strict_eligible=summary.strict_eligible,
        strict_accuracy=summary.strict_accuracy,
        loose_correct=summary.loose_correct,
        loose_eligible=summary.loose_eligible,
        loose_accuracy=summary.loose_accuracy,
        grade_digest=grade_digest,
        failures=tuple(getattr(outcome, "failures", ())),
        adapter_calls=adapter.call_count(),
        dry_run=False,
        generated_samples=len(deduped_accepted),
    )


def replay_offline(
    artifacts_dir: Path,
    *,
    checks: Mapping[str, Any] | None = None,
    extra_secrets: frozenset[str] = frozenset(),
) -> ReplaySummary:
    """Reconstruct stored samples and regrade them without any provider.

    Reads ``events.jsonl`` + artifacts only. The fixture transport is never
    constructed here, so there is no code path that could dial out.
    """
    root = Path(artifacts_dir)
    reader = ReplayReader(root, extra_secrets=extra_secrets)
    report = reader.read(strict=True)
    accepted = [r for r in report.results if r.is_accepted_sample]
    # Grade every reconstructed generation (accepted + transport-failed) so the
    # digest matches the original run, which also grades failures as
    # correctness-unavailable rather than dropping them.
    to_grade = sorted(report.results, key=lambda r: (r.sample_key.as_tuple(), r.attempt_id))

    # Rebuild tasks from the stored manifest + synthetic prompts so grading
    # uses the same TaskSpecs (and gold-leakage checks) as the original run.
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        manifest = CampaignManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
        items = synthetic_items_for_manifest(manifest)
        # If the original run used custom items, the synthetic prompts still
        # match on (benchmark, item) identity; grading only needs the TaskSpec
        # keys and safe_request, both derived deterministically.
        tasks = _tasks_for_manifest(manifest, items)
        tasks_by_key = {t.sample_key.as_tuple(): t for t in tasks}
    else:
        tasks_by_key = {}

    pairs: list[IfEvalGradePair] = []
    for generation in to_grade:
        task = tasks_by_key.get(generation.sample_key.as_tuple())
        check = (checks or {}).get(generation.sample_key.task_id, default_check)
        if task is None:
            # No stored manifest: grade the response alone against the check
            # when accepted, else record an unavailable transport pair.
            if not generation.is_accepted_sample:
                from stealthbench.benchmarks.ifeval import grade_generation as _gg
                from stealthbench.schemas.manifest import PromptRef, prompt_hash
                from stealthbench.schemas.results import (
                    EvaluationPayload,
                    ModelRequest,
                    TaskSpec,
                )

                request = ModelRequest.model_validate(
                    {
                        "messages": [{"role": "user", "content": "replay"}],
                        "max_output_tokens": 64,
                    }
                )
                fallback_task = TaskSpec(
                    benchmark_id=generation.sample_key.task_id.split("::")[0]
                    if "::" in generation.sample_key.task_id
                    else "ifeval",
                    item_id=generation.sample_key.task_id.split("::")[-1],
                    campaign_id=generation.sample_key.campaign_id,
                    endpoint_id=generation.sample_key.endpoint_id,
                    repeat_id=generation.sample_key.repeat_id,
                    prompt_artifact_hash=prompt_hash(PromptRef.from_request(request)),
                    request=request,
                    evaluation=EvaluationPayload(
                        evaluator_id="instruction_following_eval",
                        evaluator_revision="e49bbfe381c9c0e564b937f1c4e163a2273c65cc",
                    ),
                )
                pairs.append(_gg(task=fallback_task, generation=generation, check=check))
                continue
            from stealthbench.benchmarks.ifeval import grade_accepted_sample
            from stealthbench.schemas.manifest import PromptRef, prompt_hash
            from stealthbench.schemas.results import (
                EvaluationPayload,
                ModelRequest,
                TaskSpec,
            )

            request = ModelRequest.model_validate(
                {"messages": [{"role": "user", "content": "replay"}], "max_output_tokens": 64}
            )
            fallback_task = TaskSpec(
                benchmark_id=generation.sample_key.task_id.split("::")[0]
                if "::" in generation.sample_key.task_id
                else "ifeval",
                item_id=generation.sample_key.task_id.split("::")[-1],
                campaign_id=generation.sample_key.campaign_id,
                endpoint_id=generation.sample_key.endpoint_id,
                repeat_id=generation.sample_key.repeat_id,
                prompt_artifact_hash=prompt_hash(PromptRef.from_request(request)),
                request=request,
                evaluation=EvaluationPayload(
                    evaluator_id="instruction_following_eval",
                    evaluator_revision="e49bbfe381c9c0e564b937f1c4e163a2273c65cc",
                ),
            )
            pairs.append(
                grade_accepted_sample(
                    task=fallback_task,
                    response=generation.response or "",
                    check=check,
                )
            )
        else:
            pairs.append(grade_generation(task=task, generation=generation, check=check))
    summary = summarize_pairs(pairs)
    digest = content_digest(
        [
            {
                "sample_key": list(p.strict.sample_key.as_tuple()),
                "strict": p.strict.correctness,
                "loose": p.loose.correctness,
            }
            for p in sorted(pairs, key=lambda q: q.strict.sample_key.as_tuple())
        ]
    )
    campaign_id = accepted[0].sample_key.campaign_id if accepted else root.name
    return ReplaySummary(
        campaign_id=campaign_id,
        artifacts_dir=str(root),
        accepted_samples=len(accepted),
        grade_digest=digest,
        strict_accuracy=summary.strict_accuracy,
        loose_accuracy=summary.loose_accuracy,
        events_read=report.events_read,
        damage=tuple(report.damage),
        rejected=tuple(report.rejected),
    )


def _unused_transport_guard() -> Decimal:
    """Keep Decimal imported for ledger totals typing; no runtime effect."""
    return Decimal(0)
