"""Redacted JSON/CSV report exports with injection and traversal guards (G13 T13B).

Contracts (``docs/contracts.md`` sections 5-6, ``IMPLEMENTATION_PLAN.md`` G13):

* Default downloads are redacted. Secret values never appear in a serialized
  export; ``credential_ref`` stays a name and any canary value is replaced.
* CSV cells beginning with ``=``, ``+``, ``-`` or ``@`` are defused so a
  spreadsheet cannot execute them as a formula. Missing measurements stay
  empty/``null``, never ``0``.
* Output paths are confined to the report directory. An artifact link or an
  endpoint label containing ``..`` or a separator can never escape it.
* Counts round-trip: reloading the written JSON/CSV yields the same
  accepted-sample and grade counts the source artifacts reported.

Offline by construction: file reads plus atomic writes. No transport,
no socket, no subprocess, no credential.
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any, Final

from stealthbench.schemas.hashing import canonical_json
from stealthbench.storage.events import REDACTED, redact_text

#: CSV cells starting with one of these are a spreadsheet formula when opened.
CSV_FORMULA_PREFIXES: Final[tuple[str, ...]] = ("=", "+", "-", "@")

REPORT_JSON_NAME: Final[str] = "report.json"
REPORT_CSV_NAME: Final[str] = "report.csv"


class ExportTraversal(ValueError):
    """An export filename would escape the report directory."""


def sanitize_csv_cell(value: str, *, extra_secrets: frozenset[str] = frozenset()) -> str:
    """Redact secrets, then defuse spreadsheet formula injection.

    Redaction runs first so a secret that itself starts with ``=`` cannot
    survive inside a defused cell. Ordinary identifiers (``ifeval::syn-if-001``)
    pass through unchanged.
    """
    cleaned = redact_text(value, extra_secrets=extra_secrets)
    if cleaned.startswith(CSV_FORMULA_PREFIXES):
        return "'" + cleaned
    return cleaned


def safe_output_path(output_dir: Path, filename: str) -> Path:
    """Join ``filename`` onto ``output_dir``, refusing any traversal.

    Only a plain basename is accepted: absolute paths, separators and
    parent-directory segments all raise. The returned path is guaranteed to
    resolve inside ``output_dir``.
    """
    if not filename or filename in {".", ".."}:
        raise ExportTraversal(f"refusing empty or dot-only export name {filename!r}")
    candidate = Path(filename)
    if candidate.is_absolute():
        raise ExportTraversal(f"refusing absolute export path {filename!r}")
    if len(candidate.parts) != 1:
        raise ExportTraversal(f"refusing export path with separators {filename!r}")
    text = candidate.name
    if text in {".", ".."} or ".." in text:
        raise ExportTraversal(f"refusing parent-directory export name {filename!r}")
    if "/" in filename or "\\" in filename:
        raise ExportTraversal(f"refusing export path with separators {filename!r}")
    resolved_base = output_dir.resolve()
    resolved_target = (resolved_base / text).resolve()
    if resolved_target != resolved_base / text and resolved_target.parent != resolved_base:
        raise ExportTraversal(f"refusing export path escaping {output_dir}: {filename!r}")
    if resolved_target.parent != resolved_base:
        raise ExportTraversal(f"refusing export path escaping {output_dir}: {filename!r}")
    return output_dir / text


def write_text_atomic(path: Path, text: str) -> Path:
    """Write ``text`` atomically so a partial file is never left behind."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(target)
    return target


def report_payload_to_json(payload: dict[str, Any]) -> str:
    """Canonical JSON for a report payload (sorted keys, no whitespace drift)."""
    return canonical_json(payload)


def rows_to_csv(
    rows: list[dict[str, Any]],
    columns: list[str],
    *,
    extra_secrets: frozenset[str] = frozenset(),
) -> str:
    """Render report rows as CSV with redaction, defusing and null handling.

    String cells are redacted then defused. ``None`` renders as empty (a
    missing measurement, never ``0``). Non-string scalars render plainly.
    """
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        safe: dict[str, Any] = {}
        for column in columns:
            value = row.get(column)
            if value is None:
                safe[column] = ""
            elif isinstance(value, str):
                safe[column] = sanitize_csv_cell(value, extra_secrets=extra_secrets)
            elif isinstance(value, float):
                safe[column] = redact_text(repr(value), extra_secrets=extra_secrets)
            else:
                text = str(value)
                safe[column] = sanitize_csv_cell(text, extra_secrets=extra_secrets)
        writer.writerow(safe)
    return buffer.getvalue()


def write_report_json(
    output_dir: Path,
    payload: dict[str, Any],
    *,
    filename: str = REPORT_JSON_NAME,
    extra_secrets: frozenset[str] = frozenset(),
) -> Path:
    """Write the redacted report JSON into ``output_dir``.

    The payload is redacted as serialized text so a secret embedded in any
    nested value cannot survive. Returns the written path.
    """
    _ = extra_secrets
    text = report_payload_to_json(payload)
    redacted = redact_text(text, extra_secrets=extra_secrets)
    # Redaction must be a fixpoint: if a secret spans a JSON escape boundary
    # the first pass could miss it, so re-check by comparing.
    if redact_text(redacted, extra_secrets=extra_secrets) != redacted:
        redacted = redact_text(redacted, extra_secrets=extra_secrets)
    target = safe_output_path(Path(output_dir), filename)
    return write_text_atomic(target, redacted + "\n")


def write_report_csv(
    output_dir: Path,
    rows: list[dict[str, Any]],
    columns: list[str],
    *,
    filename: str = REPORT_CSV_NAME,
    extra_secrets: frozenset[str] = frozenset(),
) -> Path:
    """Write report rows as a defused, redacted CSV into ``output_dir``."""
    text = rows_to_csv(rows, columns, extra_secrets=extra_secrets)
    if redact_text(text, extra_secrets=extra_secrets) != text:
        raise ValueError("CSV redaction failed to converge; refusing to write")
    target = safe_output_path(Path(output_dir), filename)
    return write_text_atomic(target, text)


def read_report_json(path: Path) -> dict[str, Any]:
    """Reload a written report JSON payload."""
    raw = Path(path).read_text(encoding="utf-8")
    loaded = json.loads(raw)
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return loaded


def read_report_csv(path: Path) -> list[dict[str, str]]:
    """Reload a written report CSV as string rows (empty stays empty)."""
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def contains_secret(text: str, *, extra_secrets: frozenset[str] = frozenset()) -> bool:
    """Whether a credential survives in ``text`` (used to assert redaction)."""
    return redact_text(text, extra_secrets=extra_secrets) != text


def redacted_marker_present(text: str) -> bool:
    """Whether the visible redaction placeholder appears in ``text``."""
    return REDACTED in text


__all__ = [
    "CSV_FORMULA_PREFIXES",
    "REPORT_CSV_NAME",
    "REPORT_JSON_NAME",
    "ExportTraversal",
    "contains_secret",
    "read_report_csv",
    "read_report_json",
    "redacted_marker_present",
    "report_payload_to_json",
    "rows_to_csv",
    "safe_output_path",
    "sanitize_csv_cell",
    "write_report_csv",
    "write_report_json",
    "write_text_atomic",
]
