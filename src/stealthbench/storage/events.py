"""Durable append-only event log and content-addressed artifact store.

The durability rule this module exists to enforce: **an interrupted write must never
look like a completed sample.** Three mechanisms combine:

1. Payloads are content-addressed and written atomically (temp file, fsync, rename),
   so a payload is either wholly present or wholly absent.
2. An event line is appended only after its payload is durable, and the line carries
   the payload's digest.
3. A reader treats a truncated final line, an unparsable line, a digest mismatch and
   a missing payload as *damage to report* — never as a completed sample.

Redaction lives here too, because a secret must never reach the log in the first
place, and because provider error bodies routinely echo credentials back.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Final

from pydantic import ValidationError

from stealthbench.schemas.hashing import canonical_json, content_digest, file_digest
from stealthbench.schemas.results import RunEvent

EVENT_LOG_NAME: Final[str] = "events.jsonl"
ARTIFACT_DIR_NAME: Final[str] = "artifacts"

#: Placeholder substituted for a redacted value. Deliberately obvious, so a leaked
#: redaction is visible in a diff rather than looking like real data.
REDACTED: Final[str] = "[REDACTED]"

#: Environment variables whose values are secrets. Read at redaction time rather than
#: captured at import, so a test can set one after the module loads.
_SECRET_ENV_SUFFIXES: Final[tuple[str, ...]] = (
    "API_KEY",
    "APIKEY",
    "TOKEN",
    "SECRET",
    "PASSWORD",
    "CREDENTIAL",
    "BEARER",
)

#: Header names whose values are always redacted regardless of content.
_SECRET_HEADER_NAMES: Final[frozenset[str]] = frozenset(
    {
        "authorization",
        "proxy-authorization",
        "x-api-key",
        "api-key",
        "apikey",
        "x-auth-token",
        "cookie",
        "set-cookie",
        "x-openai-api-key",
        "x-goog-api-key",
    }
)

#: Credential-shaped substrings, matched case-insensitively anywhere in free text.
_SECRET_VALUE_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(r"\bsk-[A-Za-z0-9_\-]{8,}"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{16,}"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}"),
    re.compile(r"\bAIza[0-9A-Za-z_\-]{20,}"),
    re.compile(r"\bAKIA[0-9A-Z]{12,}"),
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]{8,}"),
    re.compile(r"(?i)\b(?:api[_-]?key|token|secret|password)\s*[=:]\s*\S{6,}"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    # A JWT: the eyJ header prefix plus three base64url segments. The prefix keeps
    # this from matching a hash, a digest or a dotted version string.
    re.compile(r"\beyJ[A-Za-z0-9_\-]{2,}\.[A-Za-z0-9_\-]{2,}\.[A-Za-z0-9_\-]{2,}"),
)


class DamageKind(StrEnum):
    """How a log or artifact is unusable. Reported, never silently repaired."""

    TRUNCATED_FINAL_LINE = "truncated_final_line"
    INVALID_ENCODING = "invalid_encoding"
    EVENT_ID_MISMATCH = "event_id_mismatch"
    SEQUENCE_MISMATCH = "sequence_mismatch"
    UNPARSABLE_LINE = "unparsable_line"
    SCHEMA_INVALID = "schema_invalid"
    PAYLOAD_HASH_MISMATCH = "payload_hash_mismatch"
    MISSING_PAYLOAD = "missing_payload"
    PAYLOAD_DIGEST_MISMATCH = "payload_digest_mismatch"


@dataclass(frozen=True, slots=True)
class Damage:
    """One unusable record, located precisely enough to investigate."""

    kind: DamageKind
    detail: str
    line_number: int | None = None
    event_id: str | None = None
    path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": str(self.kind),
            "detail": self.detail,
            "line_number": self.line_number,
            "event_id": self.event_id,
            "path": self.path,
        }


@dataclass(slots=True)
class IntegrityReport:
    """What a reader found when it opened a log."""

    total_lines: int = 0
    valid_events: int = 0
    damage: list[Damage] = field(default_factory=list)
    #: A final record that parsed cleanly but whose newline was lost. Not damage:
    #: the record is complete, so it is read, but the anomaly is still disclosed.
    missing_trailing_newline: bool = False

    @property
    def is_clean(self) -> bool:
        return not self.damage

    @property
    def damaged_lines(self) -> int:
        return len(self.damage)

    def kinds(self) -> set[str]:
        return {str(item.kind) for item in self.damage}

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_lines": self.total_lines,
            "valid_events": self.valid_events,
            "is_clean": self.is_clean,
            "missing_trailing_newline": self.missing_trailing_newline,
            "damage": [item.to_dict() for item in self.damage],
        }


# ---------------------------------------------------------------------------
# Redaction
# ---------------------------------------------------------------------------


def secret_env_values() -> set[str]:
    """Values of credential-shaped environment variables currently set.

    Only values long enough to be a real credential are collected, so an unrelated
    variable named ``TOKEN_COUNT`` does not cause innocent text to be redacted.
    """
    found: set[str] = set()
    for name, value in os.environ.items():
        upper = name.upper()
        if any(upper.endswith(suffix) for suffix in _SECRET_ENV_SUFFIXES) and len(value) >= 8:
            found.add(value)
    return found


def redact_text(text: str, *, extra_secrets: frozenset[str] = frozenset()) -> str:
    """Replace credential-looking substrings with a visible placeholder.

    Longest secrets are replaced first so an overlapping secret is not left partly
    exposed by a shorter pattern match inside it.
    """
    result = text
    for secret in sorted({*secret_env_values(), *extra_secrets}, key=len, reverse=True):
        if secret:
            result = result.replace(secret, REDACTED)
    for pattern in _SECRET_VALUE_PATTERNS:
        result = pattern.sub(REDACTED, result)
    return result


def redact_mapping(
    mapping: Mapping[str, Any], *, extra_secrets: frozenset[str] = frozenset()
) -> dict[str, Any]:
    """Redact a mapping recursively, treating known header names as always secret."""
    out: dict[str, Any] = {}
    for key, value in mapping.items():
        if isinstance(key, str) and key.lower() in _SECRET_HEADER_NAMES:
            out[key] = REDACTED
            continue
        # Keys carry data too: a gateway error body often echoes a credential as a
        # key, and an unredacted key would persist in the artifact and every export.
        safe_key = redact_text(key, extra_secrets=extra_secrets) if isinstance(key, str) else key
        out[safe_key] = _redact_any(value, extra_secrets)
    return out


def _redact_any(value: Any, extra_secrets: frozenset[str]) -> Any:
    if isinstance(value, str):
        return redact_text(value, extra_secrets=extra_secrets)
    if isinstance(value, Mapping):
        return redact_mapping(value, extra_secrets=extra_secrets)
    if isinstance(value, (list, tuple)):
        return [_redact_any(item, extra_secrets) for item in value]
    return value


def contains_secret(text: str, *, extra_secrets: frozenset[str] = frozenset()) -> bool:
    """Whether a credential survives in a string. Used to assert redaction worked."""
    return redact_text(text, extra_secrets=extra_secrets) != text


# ---------------------------------------------------------------------------
# Artifact store
# ---------------------------------------------------------------------------


class ArtifactStore:
    """Content-addressed store for event payloads.

    Writes are atomic: a temporary file in the same directory is fsynced and then
    renamed, so a reader never observes a half-written artifact.
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def path_for(self, digest: str) -> Path:
        return self.root / digest[:2] / digest

    def put_bytes(self, data: bytes) -> str:
        """Store bytes and return their digest. Idempotent by content."""
        digest = file_digest_of_bytes(data)
        target = self.path_for(digest)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            return digest
        # Write to a sibling temp file, fsync it, then rename into place. A reader
        # therefore sees either the previous state or the complete artifact.
        fd, temp_name = tempfile.mkstemp(dir=target.parent, suffix=".tmp")
        temp_path = Path(temp_name)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            temp_path.replace(target)
        except BaseException:
            temp_path.unlink(missing_ok=True)
            raise
        return digest

    def put_json(self, value: Any) -> str:
        return self.put_bytes(canonical_json(value).encode("utf-8"))

    def get_bytes(self, digest: str) -> bytes:
        path = self.path_for(digest)
        if not path.exists():
            raise FileNotFoundError(f"no artifact for digest {digest}")
        return path.read_bytes()

    def get_json(self, digest: str) -> Any:
        return json.loads(self.get_bytes(digest).decode("utf-8"))

    def has(self, digest: str) -> bool:
        return self.path_for(digest).exists()

    def verify(self, digest: str) -> Damage | None:
        """Re-hash a stored artifact. Returns damage rather than raising."""
        path = self.path_for(digest)
        if not path.exists():
            return Damage(
                DamageKind.MISSING_PAYLOAD, f"artifact {digest} is absent", path=str(path)
            )
        actual = file_digest(path)
        if actual != digest:
            return Damage(
                DamageKind.PAYLOAD_DIGEST_MISMATCH,
                f"artifact {digest} hashes to {actual}",
                path=str(path),
            )
        return None


def file_digest_of_bytes(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# Event log
# ---------------------------------------------------------------------------


class EventLog:
    """Append-only JSONL log of run events.

    Append-only in the strong sense: no method rewrites or deletes a line. Re-opening
    for append continues the file; nothing can silently rewrite history.
    """

    def __init__(self, root: Path, *, extra_secrets: frozenset[str] = frozenset()) -> None:
        self.root = Path(root)
        self.path = self.root / EVENT_LOG_NAME
        self.artifacts = ArtifactStore(self.root / ARTIFACT_DIR_NAME)
        self.extra_secrets = extra_secrets

    # -- writing ---------------------------------------------------------

    def append(
        self,
        event_type: str,
        campaign_id: str,
        payload: Mapping[str, Any],
        *,
        monotonic_duration_seconds: float = 0.0,
        wall_time_utc: datetime | None = None,
    ) -> RunEvent:
        """Durably record one event and return it.

        Ordering is load-bearing: the payload becomes durable first, then the event
        line naming its digest. A crash between the two leaves an unreferenced
        artifact, which is harmless. The reverse order would leave an event claiming a
        payload that does not exist.

        The event id is always derived, never supplied. Deriving it from the payload
        digest *and* the record's ordinal means two identical payloads still get
        distinct ids, and a reader can detect a record whose digest was rewritten to
        point at a different artifact.
        """
        safe_payload = redact_mapping(payload, extra_secrets=self.extra_secrets)
        digest = self.artifacts.put_json(safe_payload)
        sequence = self._next_sequence()
        record = RunEvent(
            event_id=_event_id(campaign_id, event_type, digest, sequence),
            event_type=event_type,
            campaign_id=campaign_id,
            sequence=sequence,
            wall_time_utc=wall_time_utc or datetime.now(UTC),
            monotonic_duration_seconds=monotonic_duration_seconds,
            payload_hash=digest,
            payload={},
        )
        self._append_line(record.model_dump(mode="json"))
        return record

    def _next_sequence(self) -> int:
        """Ordinal of this record, derived from what is already on disk.

        Read from the file rather than kept in memory so a second process appending to
        the same log cannot reuse an ordinal.
        """
        if not self.path.exists():
            return 0
        raw = self.path.read_bytes().decode("utf-8", errors="replace")
        if not raw:
            return 0
        return len(raw.split("\n")) - 1 if raw.endswith("\n") else len(raw.split("\n"))

    def _append_line(self, dumped: Mapping[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        # If a previous append lost its trailing newline, terminate that record before
        # writing this one. Without this the new line would be concatenated onto the
        # previous record, corrupting both.
        prefix = "\n" if self._needs_newline() else ""
        line = prefix + json.dumps(dumped, sort_keys=True, ensure_ascii=False) + "\n"
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())

    def _needs_newline(self) -> bool:
        if not self.path.exists():
            return False
        size = self.path.stat().st_size
        if size == 0:
            return False
        with self.path.open("rb") as handle:
            handle.seek(-1, 2)
            return handle.read(1) != b"\n"

    # -- reading ---------------------------------------------------------

    def read_all(self) -> tuple[list[RunEvent], IntegrityReport]:
        """Return every intact event plus a report on what could not be read.

        Damaged records are reported, never yielded. A truncated or corrupt record can
        therefore never be mistaken for a completed sample.
        """
        report = IntegrityReport()
        events: list[RunEvent] = []
        if not self.path.exists():
            return events, report

        raw_bytes = self.path.read_bytes()
        try:
            # Strict decoding: a corrupt byte must be reported, never silently turned
            # into a replacement character inside an otherwise valid-looking record.
            raw = raw_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            report.damage.append(
                Damage(
                    DamageKind.INVALID_ENCODING,
                    f"log is not valid UTF-8 at byte {exc.start}; no record can be trusted",
                    path=str(self.path),
                )
            )
            return events, report

        terminated = raw.endswith("\n")
        segments = raw.split("\n")
        if terminated or (segments and not segments[-1].strip()):
            segments = segments[:-1]
        else:
            # The final segment has no newline. It may still be a complete record whose
            # newline was lost, so it is parsed like any other; only if it fails is it
            # reported as an interrupted write.
            report.missing_trailing_newline = True

        final_index = len(segments)
        for number, line in enumerate(segments, start=1):
            report.total_lines += 1
            if not line.strip():
                continue
            event, damage = self._parse_line(
                line,
                number,
                is_unterminated_final=(number == final_index) and report.missing_trailing_newline,
            )
            if damage is not None:
                report.damage.append(damage)
                continue
            assert event is not None
            events.append(event)
            report.valid_events += 1

        return events, report

    def _parse_line(
        self, line: str, number: int, *, is_unterminated_final: bool = False
    ) -> tuple[RunEvent | None, Damage | None]:
        try:
            data = json.loads(line)
        except json.JSONDecodeError as exc:
            kind = (
                DamageKind.TRUNCATED_FINAL_LINE
                if is_unterminated_final
                else DamageKind.UNPARSABLE_LINE
            )
            detail = (
                "the final record was interrupted mid-write"
                if is_unterminated_final
                else f"line {number} is not valid JSON: {exc}"
            )
            return None, Damage(kind, detail, number)
        if not isinstance(data, dict):
            return None, Damage(
                DamageKind.UNPARSABLE_LINE, f"line {number} is not a JSON object", number
            )
        try:
            event = RunEvent.model_validate(data)
        except ValidationError as exc:
            return None, Damage(
                DamageKind.SCHEMA_INVALID,
                f"line {number} does not match the event contract: {exc.error_count()} errors",
                number,
                str(data.get("event_id")),
            )

        # The record's identity is derived from its own fields plus its ordinal, so
        # rewriting a digest to point at a different artifact is detectable, and two
        # records cannot silently collapse into one id.
        if event.sequence != number - 1:
            return None, Damage(
                DamageKind.SEQUENCE_MISMATCH,
                f"line {number} claims sequence {event.sequence}; a reordering or an "
                "insertion is not an append-only log",
                number,
                event.event_id,
            )
        expected_id = _event_id(
            event.campaign_id, event.event_type, event.payload_hash, event.sequence
        )
        if event.event_id != expected_id:
            return None, Damage(
                DamageKind.EVENT_ID_MISMATCH,
                f"event id {event.event_id} does not derive from its own campaign, type, "
                f"payload digest and sequence (expected {expected_id})",
                number,
                event.event_id,
            )

        damage = self.artifacts.verify(event.payload_hash)
        if damage is not None:
            return None, Damage(
                damage.kind,
                f"event {event.event_id}: {damage.detail}",
                number,
                event.event_id,
                damage.path,
            )
        return event, None

    def iter_events(self) -> Iterator[RunEvent]:
        """Intact events only. Damage is discarded here; use read_all to see it."""
        events, _ = self.read_all()
        yield from events

    def completion_events(self) -> tuple[list[RunEvent], IntegrityReport]:
        """Events that durably record something finishing."""
        events, report = self.read_all()
        return [event for event in events if event.is_completion_record], report

    def load_payload(self, event: RunEvent) -> Any:
        return self.artifacts.get_json(event.payload_hash)

    def integrity(self) -> IntegrityReport:
        _, report = self.read_all()
        return report


def _event_id(campaign_id: str, event_type: str, payload_hash: str, sequence: int) -> str:
    return content_digest(
        {
            "campaign_id": campaign_id,
            "event_type": event_type,
            "payload_hash": payload_hash,
            "sequence": sequence,
        }
    )[:24]
