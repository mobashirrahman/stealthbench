"""Canary scheduling with campaign-grade budget and authorization rules (G13 T13C).

A canary is a small, repeatable observation of an alias that detects change
over time. Scheduling is idempotent and offline-first:

* Scheduling the same job twice does not duplicate it. The state file is
  rewritten only when the manifest hash or the interval actually changes.
* Running a canary whose current manifest hash already completed does not
  re-dispatch. The stored summary is returned unchanged.
* Budget and authorization rules match ordinary campaigns. A live canary
  without operator authorization, configured credentials and a spending cap
  is refused before anything dispatches; an offline canary runs through the
  fixture-only workflow with caps enforced.

Offline by construction: file reads plus atomic writes, plus the offline
workflow. No provider socket, no subprocess, no credential.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from stealthbench.adapters.base import FixtureBundle
from stealthbench.schemas.campaign import Authorization, CampaignManifest
from stealthbench.schemas.manifest import manifest_hash

STATE_SUFFIX: Final[str] = ".json"
DEFAULT_INTERVAL_SECONDS: Final[float] = 86400.0

_JOB_ID_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")


class CanaryRefused(Exception):
    """A canary was refused before dispatch, naming what is missing."""

    def __init__(self, missing: tuple[str, ...]) -> None:
        super().__init__("canary refused: " + ", ".join(missing))
        self.missing = tuple(missing)


class CanaryError(ValueError):
    """An invalid canary job id, interval or state directory."""


@dataclass(frozen=True, slots=True)
class CanaryState:
    """Durable canary schedule and last-run record."""

    job_id: str
    campaign_id: str
    manifest_hash: str
    interval_seconds: float
    last_run_at: float | None
    next_due_at: float
    run_count: int
    last_grade_digest: str | None = None
    last_artifacts: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "campaign_id": self.campaign_id,
            "interval_seconds": self.interval_seconds,
            "job_id": self.job_id,
            "last_artifacts": self.last_artifacts,
            "last_grade_digest": self.last_grade_digest,
            "last_run_at": self.last_run_at,
            "manifest_hash": self.manifest_hash,
            "next_due_at": self.next_due_at,
            "run_count": self.run_count,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> CanaryState:
        try:
            return cls(
                job_id=str(raw["job_id"]),
                campaign_id=str(raw["campaign_id"]),
                manifest_hash=str(raw["manifest_hash"]),
                interval_seconds=float(raw["interval_seconds"]),
                last_run_at=(None if raw.get("last_run_at") is None else float(raw["last_run_at"])),
                next_due_at=float(raw["next_due_at"]),
                run_count=int(raw["run_count"]),
                last_grade_digest=(
                    None if raw.get("last_grade_digest") is None else str(raw["last_grade_digest"])
                ),
                last_artifacts=(
                    None if raw.get("last_artifacts") is None else str(raw["last_artifacts"])
                ),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise CanaryError(f"unusable canary state: {exc}") from exc


def validate_job_id(job_id: str) -> str:
    """Validate a canary job id, refusing traversal and separators."""
    if not _JOB_ID_PATTERN.match(job_id):
        raise CanaryError(
            f"job_id {job_id!r} must match {_JOB_ID_PATTERN.pattern!r} "
            "(lowercase, no separators, no parent segments)"
        )
    if job_id in {".", ".."} or "/" in job_id or "\\" in job_id or ".." in job_id:
        raise CanaryError(f"refusing job_id with traversal {job_id!r}")
    return job_id


def _state_path(state_dir: Path, job_id: str) -> Path:
    validated = validate_job_id(job_id)
    candidate = Path(state_dir) / (validated + STATE_SUFFIX)
    resolved_base = Path(state_dir).resolve()
    resolved_target = (resolved_base / (validated + STATE_SUFFIX)).resolve()
    if resolved_target.parent != resolved_base:
        raise CanaryError(f"refusing state path escaping {state_dir}: {job_id!r}")
    return candidate


def _write_atomic(path: Path, text: str) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(target)
    return target


def _read_state(path: Path) -> CanaryState | None:
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(raw, dict):
        return None
    try:
        return CanaryState.from_dict(raw)
    except CanaryError:
        return None


def schedule_canary(
    state_dir: Path,
    job_id: str,
    manifest: CampaignManifest,
    interval_seconds: float,
    *,
    now: float,
) -> CanaryState:
    """Schedule (or reaffirm) a canary job, idempotently.

    The same ``job_id`` plus the same manifest hash and interval returns the
    existing record without rewriting it. A changed manifest or interval
    updates the record once; a third identical call is a no-op again.
    """
    validated = validate_job_id(job_id)
    if not (interval_seconds > 0):
        raise CanaryError(f"interval_seconds must be positive, got {interval_seconds}")
    root = Path(state_dir)
    root.mkdir(parents=True, exist_ok=True)
    path = _state_path(root, validated)
    digest = manifest_hash(manifest)
    existing = _read_state(path)
    if (
        existing is not None
        and existing.manifest_hash == digest
        and existing.interval_seconds == interval_seconds
        and existing.job_id == validated
        and existing.campaign_id == manifest.campaign_id
    ):
        return existing
    if existing is not None:
        last_run = existing.last_run_at
        run_count = existing.run_count
        last_digest = existing.last_grade_digest
        last_artifacts = existing.last_artifacts
    else:
        last_run = None
        run_count = 0
        last_digest = None
        last_artifacts = None
    next_due = now if last_run is None else (last_run + interval_seconds)
    state = CanaryState(
        job_id=validated,
        campaign_id=manifest.campaign_id,
        manifest_hash=digest,
        interval_seconds=interval_seconds,
        last_run_at=last_run,
        next_due_at=next_due,
        run_count=run_count,
        last_grade_digest=last_digest,
        last_artifacts=last_artifacts,
    )
    _write_atomic(path, json.dumps(state.to_dict(), sort_keys=True) + "\n")
    return state


def list_canaries(state_dir: Path) -> list[CanaryState]:
    """Every scheduled canary, in job-id order."""
    root = Path(state_dir)
    if not root.is_dir():
        return []
    states: list[CanaryState] = []
    for child in sorted(root.iterdir()):
        if not child.is_file() or child.suffix != STATE_SUFFIX:
            continue
        state = _read_state(child)
        if state is not None:
            states.append(state)
    return sorted(states, key=lambda item: item.job_id)


def due_canaries(state_dir: Path, *, now: float) -> list[CanaryState]:
    """Canaries whose next due time has arrived, in due order."""
    return sorted(
        [state for state in list_canaries(state_dir) if state.next_due_at <= now],
        key=lambda item: (item.next_due_at, item.job_id),
    )


def _live_blockers(
    manifest: CampaignManifest,
    *,
    credentials_configured: bool,
    spending_cap: float | None,
) -> tuple[str, ...]:
    # Imported here so the monitoring module stays importable without the
    # runner's adapter dependencies at module scope.
    from stealthbench.runner import RunMode, resolve_mode

    authorization: Authorization = manifest.authorization or Authorization()
    missing = resolve_mode(
        requested=RunMode.LIVE,
        authorization=authorization,
        credentials_configured=credentials_configured,
        spending_cap=spending_cap,
    )
    return tuple(f"live execution is missing: {item}" for item in missing)


def run_canary_offline(
    state_dir: Path,
    job_id: str,
    manifest: CampaignManifest,
    *,
    artifacts_root: Path,
    fixture_bundle: FixtureBundle | None = None,
    now: float,
    extra_secrets: frozenset[str] = frozenset(),
    credentials_configured: bool = False,
    spending_cap: float | None = None,
    interval_seconds: float = DEFAULT_INTERVAL_SECONDS,
) -> dict[str, Any]:
    """Run one canary offline, enforcing campaign budget and auth rules.

    * A ``live_authorized`` manifest without authorization, credentials and a
      spending cap raises :class:`CanaryRefused` before anything dispatches.
    * An offline manifest runs through the fixture-only workflow, so request,
      token, concurrency and reserved-cost caps apply exactly as they do for
      an ordinary campaign.
    * Idempotent: when the target artifact directory already holds a summary
      with the current manifest hash, the stored summary is returned and the
      schedule's run count does not advance.
    """
    validated = validate_job_id(job_id)
    if not (interval_seconds > 0):
        raise CanaryError(f"interval_seconds must be positive, got {interval_seconds}")
    root = Path(state_dir)
    root.mkdir(parents=True, exist_ok=True)
    artifacts_base = Path(artifacts_root)
    artifacts_base.mkdir(parents=True, exist_ok=True)

    if manifest.mode == "live_authorized":
        authorization = manifest.authorization or Authorization()
        cap = authorization.spending_cap_usd
        if spending_cap is not None:
            cap = spending_cap
        blockers = _live_blockers(
            manifest,
            credentials_configured=credentials_configured,
            spending_cap=cap,
        )
        if blockers:
            raise CanaryRefused(tuple(blockers))

    digest = manifest_hash(manifest)
    # The artifact directory is derived from the manifest hash so a changed
    # manifest cannot silently overwrite the previous observation.
    safe_campaign = validate_job_id(manifest.campaign_id)
    target = artifacts_base / f"{safe_campaign}-{digest[:12]}"

    summary_path = target / "summary.json"
    if summary_path.is_file():
        try:
            cached = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            cached = None
        if isinstance(cached, dict) and cached.get("manifest_hash") == digest:
            # Already completed with this exact manifest: ensure the schedule
            # exists and return the stored summary without re-dispatching.
            existing = _read_state(_state_path(root, validated))
            if existing is None or (
                existing.manifest_hash != digest or existing.interval_seconds != interval_seconds
            ):
                schedule_canary(root, validated, manifest, interval_seconds, now=now)
            result = dict(cached)
            result["canary_job_id"] = validated
            result["canary_idempotent"] = True
            return result

    # Not yet completed: dispatch through the offline workflow (lazy import
    # avoids a runner <-> workflow import cycle at module scope).
    from stealthbench.benchmarks.workflow import run_offline

    report = run_offline(
        manifest,
        artifacts_dir=target,
        fixture_bundle=fixture_bundle,
        extra_secrets=extra_secrets,
    )
    payload = report.to_dict()
    payload["canary_job_id"] = validated
    payload["canary_idempotent"] = False

    # Record the run. Ensure a schedule exists first so the run count and the
    # next due time advance together.
    scheduled = _read_state(_state_path(root, validated))
    if scheduled is None or scheduled.manifest_hash != digest:
        scheduled = schedule_canary(root, validated, manifest, interval_seconds, now=now)
    updated = CanaryState(
        job_id=scheduled.job_id,
        campaign_id=scheduled.campaign_id,
        manifest_hash=digest,
        interval_seconds=scheduled.interval_seconds,
        last_run_at=now,
        next_due_at=now + scheduled.interval_seconds,
        run_count=scheduled.run_count + 1,
        last_grade_digest=str(payload.get("grade_digest"))
        if payload.get("grade_digest") is not None
        else None,
        last_artifacts=str(target),
    )
    _write_atomic(
        _state_path(root, validated),
        json.dumps(updated.to_dict(), sort_keys=True) + "\n",
    )
    return payload


__all__ = [
    "DEFAULT_INTERVAL_SECONDS",
    "CanaryError",
    "CanaryRefused",
    "CanaryState",
    "due_canaries",
    "list_canaries",
    "run_canary_offline",
    "schedule_canary",
    "validate_job_id",
]
