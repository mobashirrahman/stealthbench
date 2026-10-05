"""SQLite index over the event log, and redacted export.

The index is a *derived* view: the JSONL log and the artifact store are the record of
truth, and SQLite exists to make them queryable. Two consequences shape the code:

* Ingestion is idempotent. Re-indexing the same log yields the same rows, and a
  duplicate delivery attempt never becomes a second sample.
* Nothing reaches the index un-redacted. Exports and index contents pass through the
  same redaction as the log, because an index row is still a place a secret can leak.
"""

from __future__ import annotations

import csv
import io
import json
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from stealthbench.schemas.hashing import canonical_json
from stealthbench.schemas.results import GenerationResult, accepted_sample_keys
from stealthbench.storage.events import (
    EventLog,
    redact_mapping,
    redact_text,
)

SCHEMA_VERSION: Final[str] = "1.0"
INDEX_FILENAME: Final[str] = "index.sqlite3"

_SCHEMA: Final[str] = """
CREATE TABLE IF NOT EXISTS index_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    event_id                TEXT PRIMARY KEY,
    event_type              TEXT NOT NULL,
    campaign_id             TEXT NOT NULL,
    wall_time_utc           TEXT NOT NULL,
    monotonic_duration_s    REAL NOT NULL,
    payload_hash            TEXT NOT NULL,
    is_completion           INTEGER NOT NULL,
    is_attempt              INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS events_by_campaign ON events (campaign_id, wall_time_utc);
CREATE INDEX IF NOT EXISTS events_by_type ON events (event_type);

CREATE TABLE IF NOT EXISTS samples (
    campaign_id        TEXT NOT NULL,
    endpoint_id        TEXT NOT NULL,
    task_id            TEXT NOT NULL,
    repeat_id          INTEGER NOT NULL,
    attempt_id         TEXT NOT NULL,
    attempt_number     INTEGER NOT NULL,
    delivery_status    TEXT NOT NULL,
    is_accepted        INTEGER NOT NULL,
    has_response       INTEGER NOT NULL,
    input_tokens       INTEGER,
    output_tokens      INTEGER,
    manifest_hash      TEXT,
    k                  TEXT,
    PRIMARY KEY (campaign_id, endpoint_id, task_id, repeat_id, attempt_id)
);
CREATE INDEX IF NOT EXISTS samples_by_task ON samples (task_id);

CREATE TABLE IF NOT EXISTS damage (
    kind        TEXT NOT NULL,
    detail      TEXT NOT NULL,
    line_number INTEGER,
    event_id    TEXT
);
"""


@dataclass(frozen=True, slots=True)
class IndexStats:
    """What an indexing pass did, for the gate evidence record."""

    events_indexed: int
    events_skipped: int
    samples_indexed: int
    accepted_samples: int
    damaged_records: int
    #: Rows a rebuild removed that the caller did not re-supply. Non-zero means the
    #: caller passed a subset, which is a caller error; it is reported rather than silent.
    dropped_rows: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "events_indexed": self.events_indexed,
            "events_skipped": self.events_skipped,
            "samples_indexed": self.samples_indexed,
            "accepted_samples": self.accepted_samples,
            "damaged_records": self.damaged_records,
            "dropped_rows": self.dropped_rows,
        }


class CampaignIndex:
    """Queryable, idempotent index built from an event log.

    Usage is intentionally narrow: ``sqlite3`` connections are opened per operation
    so a long-lived handle cannot outlive a cancelled run.
    """

    def __init__(self, root: Path, *, extra_secrets: frozenset[str] = frozenset()) -> None:
        self.root = Path(root)
        self.path = self.root / INDEX_FILENAME
        self.extra_secrets = extra_secrets
        self.root.mkdir(parents=True, exist_ok=True)
        self._initialise()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialise(self) -> None:
        with self._connect() as connection:
            connection.executescript(_SCHEMA)
            connection.execute(
                "INSERT OR IGNORE INTO index_meta (key, value) VALUES ('schema_version', ?)",
                (SCHEMA_VERSION,),
            )
            connection.commit()

    # -- ingestion -------------------------------------------------------

    def ingest(self, log: EventLog, results: Sequence[GenerationResult] = ()) -> IndexStats:
        """Index a log and its sample results.

        Idempotent by construction: every insert is keyed, so running this twice over
        the same log leaves identical rows.
        """
        events, report = log.read_all()
        with self._connect() as connection:
            connection.execute("DELETE FROM events")
            connection.execute("DELETE FROM damage")

            # Samples are a full rebuild from the caller's result set, which is itself
            # derived from this log. Re-ingesting with a subset is a caller error, so it
            # is counted rather than silent: `dropped_rows` reports any row the rebuild
            # removed that the caller did not re-supply.
            prior_keys = {str(row[0]) for row in connection.execute("SELECT k FROM samples")}
            incoming_keys = {_sample_key_text(self._sample_row(r)) for r in results}
            connection.execute("DELETE FROM samples")

            connection.executemany(
                "INSERT OR REPLACE INTO events "
                "(event_id, event_type, campaign_id, wall_time_utc, monotonic_duration_s, "
                " payload_hash, is_completion, is_attempt) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        event.event_id,
                        event.event_type,
                        event.campaign_id,
                        event.wall_time_utc.isoformat(),
                        event.monotonic_duration_seconds,
                        event.payload_hash,
                        int(event.is_completion_record),
                        int(event.is_attempt_record),
                    )
                    for event in events
                ],
            )

            connection.executemany(
                "INSERT OR REPLACE INTO samples "
                "(campaign_id, endpoint_id, task_id, repeat_id, attempt_id, attempt_number, "
                " delivery_status, is_accepted, has_response, input_tokens, output_tokens, "
                " manifest_hash, k) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [self._sample_row(result) for result in results],
            )

            connection.executemany(
                "INSERT INTO damage (kind, detail, line_number, event_id) VALUES (?, ?, ?, ?)",
                [
                    (str(item.kind), item.detail, item.line_number, item.event_id)
                    for item in report.damage
                ],
            )
            connection.commit()

        accepted = accepted_sample_keys(results)
        dropped = len(prior_keys - incoming_keys)
        return IndexStats(
            events_indexed=len(events),
            events_skipped=len(report.damage),
            samples_indexed=len(results),
            accepted_samples=len(accepted),
            damaged_records=len(report.damage),
            dropped_rows=dropped,
        )

    def _sample_row(self, result: GenerationResult) -> tuple[Any, ...]:
        key = result.sample_key
        row = (
            key.campaign_id,
            key.endpoint_id,
            key.task_id,
            key.repeat_id,
            result.attempt_id,
            result.attempt_number,
            str(result.delivery_status),
            int(result.is_accepted_sample),
            int(result.response is not None),
            result.usage.input_tokens,
            result.usage.output_tokens,
            result.manifest_hash,
        )
        return (*row, _sample_key_text(row))

    # -- queries ---------------------------------------------------------

    def counts(self) -> dict[str, int]:
        with self._connect() as connection:

            def scalar(query: str) -> int:
                return int(connection.execute(query).fetchone()[0])

            return {
                "events": scalar("SELECT COUNT(*) FROM events"),
                "samples": scalar("SELECT COUNT(*) FROM samples"),
                # Distinct sample keys, so a retry never inflates the denominator.
                # Counting rows here would contradict accepted_sample_count().
                "accepted_samples": scalar(
                    "SELECT COUNT(*) FROM (SELECT DISTINCT campaign_id, endpoint_id, task_id, "
                    "repeat_id FROM samples WHERE is_accepted = 1)"
                ),
                "damage": scalar("SELECT COUNT(*) FROM damage"),
            }

    def accepted_sample_count(self) -> int:
        """Distinct accepted samples, so retries never inflate a denominator."""
        with self._connect() as connection:
            return int(
                connection.execute(
                    "SELECT COUNT(*) FROM (SELECT DISTINCT campaign_id, endpoint_id, task_id, "
                    "repeat_id FROM samples WHERE is_accepted = 1)"
                ).fetchone()[0]
            )

    def sample_usage_totals(self) -> dict[str, int | None]:
        """Sums token columns, keeping NULL distinct from zero.

        Returns ``None`` for a direction with no reported values at all, rather than
        0: "nobody reported usage" is not "usage was zero".
        """
        with self._connect() as connection:
            row = connection.execute(
                "SELECT COUNT(input_tokens), COALESCE(SUM(input_tokens), 0), "
                "COUNT(output_tokens), COALESCE(SUM(output_tokens), 0) FROM samples"
            ).fetchone()
        input_reported, input_total, output_reported, output_total = row
        return {
            "input_tokens": input_total if input_reported else None,
            "output_tokens": output_total if output_reported else None,
            "reported_inputs": input_reported,
            "reported_outputs": output_reported,
        }

    def damage_kinds(self) -> list[str]:
        with self._connect() as connection:
            return [
                row[0]
                for row in connection.execute("SELECT DISTINCT kind FROM damage ORDER BY kind")
            ]

    def events_of_type(self, event_type: str) -> list[sqlite3.Row]:
        with self._connect() as connection:
            return list(
                connection.execute(
                    "SELECT * FROM events WHERE event_type = ? ORDER BY wall_time_utc, event_id",
                    (event_type,),
                )
            )

    def samples_for_task(self, task_id: str) -> list[sqlite3.Row]:
        with self._connect() as connection:
            return list(
                connection.execute(
                    "SELECT * FROM samples WHERE task_id = ? ORDER BY repeat_id, attempt_number",
                    (task_id,),
                )
            )

    def has_damage(self) -> bool:
        return self.counts()["damage"] > 0


# ---------------------------------------------------------------------------
# Redacted exports
# ---------------------------------------------------------------------------


def export_json(index: CampaignIndex, log: EventLog, results: Sequence[GenerationResult]) -> str:
    """A JSON export of everything, redacted.

    Sample rows are built from the validated models rather than from index rows, so
    a null token count stays null through the export.
    """
    events, report = log.read_all()
    payload = {
        "schema_version": SCHEMA_VERSION,
        "index_counts": index.counts(),
        "accepted_samples": index.accepted_sample_count(),
        "integrity": report.to_dict(),
        "events": [
            {
                "event_id": event.event_id,
                "event_type": event.event_type,
                "campaign_id": event.campaign_id,
                "wall_time_utc": event.wall_time_utc.isoformat(),
                "monotonic_duration_seconds": event.monotonic_duration_seconds,
                "payload_hash": event.payload_hash,
                "payload": redact_mapping(
                    _safe_load(log, event), extra_secrets=index.extra_secrets
                ),
            }
            for event in events
        ],
        "samples": [
            redact_mapping(json.loads(result.model_dump_json()), extra_secrets=index.extra_secrets)
            for result in results
        ],
    }
    return canonical_json(payload)


def export_csv(index: CampaignIndex, results: Sequence[GenerationResult]) -> str:
    """A CSV export of sample rows.

    Two CSV-specific hazards are handled here rather than at render time: a cell
    beginning ``=``, ``+``, ``-`` or ``@`` is prefixed so a spreadsheet cannot execute
    it as a formula, and a missing token count is written as empty, not ``0``.
    """
    buffer = io.StringIO()
    columns = [
        "campaign_id",
        "endpoint_id",
        "task_id",
        "repeat_id",
        "attempt_id",
        "delivery_status",
        "is_accepted",
        "input_tokens",
        "output_tokens",
        "manifest_hash",
    ]
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for result in results:
        key = result.sample_key
        writer.writerow(
            {
                "campaign_id": _csv_cell(key.campaign_id, index.extra_secrets),
                "endpoint_id": _csv_cell(key.endpoint_id, index.extra_secrets),
                "task_id": _csv_cell(key.task_id, index.extra_secrets),
                "repeat_id": key.repeat_id,
                "attempt_id": _csv_cell(result.attempt_id, index.extra_secrets),
                "delivery_status": str(result.delivery_status),
                "is_accepted": int(result.is_accepted_sample),
                "input_tokens": ""
                if result.usage.input_tokens is None
                else result.usage.input_tokens,
                "output_tokens": ""
                if result.usage.output_tokens is None
                else result.usage.output_tokens,
                "manifest_hash": _csv_cell(result.manifest_hash or "", index.extra_secrets),
            }
        )
    return buffer.getvalue()


def _sample_key_text(row: Sequence[Any]) -> str:
    """A single string identity for a sample row, used for replace-on-upsert."""
    return "\x1f".join(str(value) for value in row[:5])


_CSV_FORMULA_PREFIXES: Final[tuple[str, ...]] = ("=", "+", "-", "@")


def _csv_cell(value: str, extra_secrets: frozenset[str]) -> str:
    """Redact, then defuse spreadsheet formula injection."""
    cleaned = redact_text(value, extra_secrets=extra_secrets)
    if cleaned.startswith(_CSV_FORMULA_PREFIXES):
        return "'" + cleaned
    return cleaned


def _safe_load(log: EventLog, event: Any) -> Mapping[str, Any]:
    """Load a payload defensively, for export where a missing file must not abort."""
    try:
        payload = log.load_payload(event)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"__unavailable__": True}
    return payload if isinstance(payload, dict) else {"value": payload}


def write_export(path: Path, text: str) -> Path:
    """Write an export atomically, so a partial file is never left behind."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)
    return path


def export_bundle(
    root: Path, index: CampaignIndex, log: EventLog, results: Sequence[GenerationResult]
) -> dict[str, Path]:
    """Write the JSON and CSV exports into ``root``."""
    return {
        "json": write_export(root / "export.json", export_json(index, log, results)),
        "csv": write_export(root / "export.csv", export_csv(index, results)),
    }
