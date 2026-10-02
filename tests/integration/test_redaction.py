"""Indexing and redacted export (task T02B).

Acceptance for T02B: duplicate ingestion is idempotent; headers, errors and exports
contain no canary secrets.
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import pytest

from stealthbench.schemas.results import (
    DeliveryStatus,
    GenerationResult,
    SampleKey,
    Usage,
)
from stealthbench.storage.events import REDACTED, EventLog
from stealthbench.storage.index import (
    CampaignIndex,
    export_bundle,
    export_csv,
    export_json,
)

pytestmark = pytest.mark.integration

HEADER_CANARY = "ghp_canary_0123456789abcdefghij"
ERROR_CANARY = "sk-canary-abcdefghijklmnopqrstuvwx"
BODY_CANARY = "totallyUnrecognisedCredentialValue"


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return tmp_path / "run"


@pytest.fixture
def log(root: Path) -> EventLog:
    return EventLog(root, extra_secrets=frozenset({BODY_CANARY}))


def make_result(
    key: SampleKey,
    *,
    attempt: int = 1,
    status: DeliveryStatus = DeliveryStatus.ACCEPTED,
    response: str | None = "ok",
    input_tokens: int | None = 10,
    output_tokens: int | None = 20,
) -> GenerationResult:
    return GenerationResult(
        attempt_id=f"attempt-{attempt}",
        sample_key=key,
        attempt_number=attempt,
        delivery_status=status,
        response=response,
        usage=Usage(input_tokens=input_tokens, output_tokens=output_tokens),
        manifest_hash="a" * 64,
    )


def sample_key(repeat: int = 1, item: str = "item-1") -> SampleKey:
    return SampleKey(
        campaign_id="c1", endpoint_id="e1", task_id=f"ifeval::{item}", repeat_id=repeat
    )


# ---------------------------------------------------------------------------
# Idempotence
# ---------------------------------------------------------------------------


def test_ingest_is_idempotent(root: Path, log: EventLog) -> None:
    """Re-indexing the same log must leave identical rows, never duplicates."""
    log.append("sample.accepted", "c1", {"response": "ok"})
    results = [make_result(sample_key())]
    index = CampaignIndex(root)

    first = index.ingest(log, results)
    after_first = index.counts()
    second = index.ingest(log, results)
    after_second = index.counts()

    assert first.events_indexed == second.events_indexed
    assert after_first == after_second
    assert index.counts()["events"] == 1
    assert index.counts()["samples"] == 1


def test_ingest_twice_does_not_double_count_samples(root: Path, log: EventLog) -> None:
    log.append("sample.accepted", "c1", {"response": "ok"})
    index = CampaignIndex(root)
    results = [make_result(sample_key())]
    for _ in range(5):
        index.ingest(log, results)
    assert index.accepted_sample_count() == 1


def test_retries_are_indexed_as_attempts_but_counted_as_one_sample(
    root: Path, log: EventLog
) -> None:
    log.append("sample.accepted", "c1", {"response": "ok"})
    key = sample_key()
    results = [
        make_result(key, attempt=1, status=DeliveryStatus.TRANSPORT_FAILED, response=None),
        make_result(key, attempt=2, status=DeliveryStatus.TRANSPORT_FAILED, response=None),
        make_result(key, attempt=3),
    ]
    index = CampaignIndex(root)
    stats = index.ingest(log, results)

    assert stats.samples_indexed == 3, "every attempt is retained"
    assert index.counts()["samples"] == 3
    assert index.accepted_sample_count() == 1, "but the denominator counts one sample"


def test_repeats_are_distinct_samples(root: Path, log: EventLog) -> None:
    log.append("sample.accepted", "c1", {"response": "ok"})
    index = CampaignIndex(root)
    index.ingest(log, [make_result(sample_key(repeat=1)), make_result(sample_key(repeat=2))])
    assert index.accepted_sample_count() == 2


def test_unresolved_attempts_are_not_accepted(root: Path, log: EventLog) -> None:
    log.append("request.dispatched", "c1", {"attempt": 1})
    index = CampaignIndex(root)
    index.ingest(log, [make_result(sample_key(), status=DeliveryStatus.UNRESOLVED, response=None)])
    assert index.accepted_sample_count() == 0
    assert index.counts()["samples"] == 1


def test_index_survives_a_log_being_rebuilt_from_scratch(root: Path, log: EventLog) -> None:
    """A full re-ingest replaces, rather than accumulating, derived rows."""
    for index_no in range(4):
        log.append("request.dispatched", "c1", {"seq": index_no})
    index = CampaignIndex(root)
    index.ingest(log, [])
    assert index.counts()["events"] == 4
    index.ingest(log, [])
    assert index.counts()["events"] == 4


# ---------------------------------------------------------------------------
# Damage is recorded, not hidden
# ---------------------------------------------------------------------------


def test_damage_is_recorded_in_the_index(root: Path, log: EventLog) -> None:
    log.append("sample.accepted", "c1", {"response": "ok"})
    with log.path.open("a", encoding="utf-8") as handle:
        handle.write("{interrupted")
    index = CampaignIndex(root)
    stats = index.ingest(log, [])
    assert stats.damaged_records == 1
    assert "truncated_final_line" in index.damage_kinds()
    assert index.has_damage()


def test_a_missing_payload_is_indexed_as_damage_not_as_a_sample(root: Path, log: EventLog) -> None:
    event = log.append("sample.accepted", "c1", {"response": "ok"})
    log.artifacts.path_for(event.payload_hash).unlink()
    index = CampaignIndex(root)
    stats = index.ingest(log, [])
    assert index.counts()["events"] == 0
    assert stats.events_indexed == 0
    assert "missing_payload" in index.damage_kinds()


# ---------------------------------------------------------------------------
# Usage nullability through the index
# ---------------------------------------------------------------------------


def test_absent_usage_is_null_not_zero_in_totals(root: Path, log: EventLog) -> None:
    log.append("sample.accepted", "c1", {"response": "ok"})
    index = CampaignIndex(root)
    index.ingest(log, [make_result(sample_key(), input_tokens=None, output_tokens=None)])
    totals = index.sample_usage_totals()
    assert totals["input_tokens"] is None, "no reported usage is not zero usage"
    assert totals["output_tokens"] is None
    assert totals["reported_inputs"] == 0


def test_reported_zero_usage_survives_as_zero(root: Path, log: EventLog) -> None:
    log.append("sample.accepted", "c1", {"response": "ok"})
    index = CampaignIndex(root)
    index.ingest(log, [make_result(sample_key(), input_tokens=0, output_tokens=0)])
    totals = index.sample_usage_totals()
    assert totals["input_tokens"] == 0
    assert totals["reported_inputs"] == 1


def test_partially_reported_usage_sums_only_the_reported_direction(
    root: Path, log: EventLog
) -> None:
    log.append("sample.accepted", "c1", {"response": "ok"})
    index = CampaignIndex(root)
    index.ingest(
        log,
        [
            make_result(sample_key(repeat=1), input_tokens=100, output_tokens=None),
            make_result(sample_key(repeat=2), input_tokens=50, output_tokens=None),
        ],
    )
    totals = index.sample_usage_totals()
    assert totals["input_tokens"] == 150
    assert totals["output_tokens"] is None, "an unreported direction must not sum to 0"


# ---------------------------------------------------------------------------
# Redaction in the index and exports
# ---------------------------------------------------------------------------


def test_a_credential_bearing_error_is_absent_from_the_export(root: Path, log: EventLog) -> None:
    log.append(
        "request.failed",
        "c1",
        {
            "error": {"message": f"bad key {ERROR_CANARY}", "body": f'{{"key":"{ERROR_CANARY}"}}'},
            "response_headers": {"authorization": f"Bearer {HEADER_CANARY}"},
        },
    )
    index = CampaignIndex(root)
    index.ingest(log, [])
    exported = export_json(index, log, [])

    assert ERROR_CANARY not in exported
    assert HEADER_CANARY not in exported
    assert REDACTED in exported


def test_credential_headers_never_reach_the_index(root: Path, log: EventLog) -> None:
    log.append(
        "request.completed",
        "c1",
        {"response_headers": {"x-api-key": ERROR_CANARY, "content-type": "application/json"}},
    )
    index = CampaignIndex(root)
    index.ingest(log, [])
    payload = json.loads(export_json(index, log, []))
    body = json.dumps(payload)
    assert ERROR_CANARY not in body
    header = payload["events"][0]["payload"]["response_headers"]
    assert header["x-api-key"] == REDACTED
    assert header["content-type"] == "application/json"


def test_a_recognizable_canary_in_a_response_body_never_reaches_an_export(
    root: Path, log: EventLog
) -> None:
    """A model can echo a prompt that contains a secret; the export must not carry it."""
    log.append(
        "sample.accepted",
        "c1",
        {"response": f"Sure, here it is: {BODY_CANARY}", "prompt_hash": "b" * 64},
    )
    index = CampaignIndex(root)
    index.ingest(log, [])
    for path in export_bundle(root / "out", index, log, []).values():
        assert BODY_CANARY not in path.read_text(encoding="utf-8")


def test_csv_export_contains_no_canary(root: Path, log: EventLog) -> None:
    log.append("sample.accepted", "c1", {"response": f"token {ERROR_CANARY}"})
    index = CampaignIndex(root)
    index.ingest(log, [])
    assert ERROR_CANARY not in export_csv(index, [])


def test_csv_writes_missing_usage_as_empty_not_zero(root: Path, log: EventLog) -> None:
    log.append("sample.accepted", "c1", {"response": "ok"})
    index = CampaignIndex(root)
    index.ingest(log, [make_result(sample_key(), input_tokens=None, output_tokens=None)])
    rows = list(
        csv.DictReader(
            io.StringIO(
                export_csv(
                    index, [make_result(sample_key(), input_tokens=None, output_tokens=None)]
                )
            )
        )
    )
    assert rows[0]["input_tokens"] == ""
    assert rows[0]["output_tokens"] == ""


def test_csv_defuses_formula_injection(root: Path, log: EventLog) -> None:
    """A task id beginning with '=' must not execute when the export is opened."""
    hostile = SampleKey(
        campaign_id="c1",
        endpoint_id="e1",
        task_id="=cmd|'/c calc'!A1",
        repeat_id=1,
    )
    log.append("sample.accepted", "c1", {"response": "ok"})
    index = CampaignIndex(root)
    index.ingest(log, [make_result(hostile)])
    rows = list(csv.DictReader(io.StringIO(export_csv(index, [make_result(hostile)]))))
    assert rows[0]["task_id"].startswith("'"), "a formula cell must be neutralised"
    assert not rows[0]["task_id"].startswith("=")


@pytest.mark.parametrize("prefix", ["=", "+", "-", "@"])
def test_every_formula_prefix_is_defused(root: Path, log: EventLog, prefix: str) -> None:
    hostile = SampleKey(
        campaign_id="c1", endpoint_id="e1", task_id=f"{prefix}HYPERLINK(1)", repeat_id=1
    )
    index = CampaignIndex(root)
    log.append("sample.accepted", "c1", {"response": "ok"})
    index.ingest(log, [make_result(hostile)])
    rows = list(csv.DictReader(io.StringIO(export_csv(index, [make_result(hostile)]))))
    assert rows[0]["task_id"] == f"'{prefix}HYPERLINK(1)"


def test_csv_keeps_ordinary_identifiers_intact(root: Path, log: EventLog) -> None:
    """Defusing must not mangle every value that happens to start with a dash."""
    log.append("sample.accepted", "c1", {"response": "ok"})
    index = CampaignIndex(root)
    key = sample_key(item="syn-if-001")
    index.ingest(log, [make_result(key)])
    rows = list(csv.DictReader(io.StringIO(export_csv(index, [make_result(key)]))))
    assert rows[0]["task_id"] == "ifeval::syn-if-001"


def test_export_preserves_null_usage_as_json_null(root: Path, log: EventLog) -> None:
    log.append("sample.accepted", "c1", {"response": "ok"})
    index = CampaignIndex(root)
    result = make_result(sample_key(), input_tokens=None)
    index.ingest(log, [result])
    payload = json.loads(export_json(index, log, [result]))
    usage = payload["samples"][0]["usage"]
    assert usage["input_tokens"] is None
    assert usage["output_tokens"] == 20


def test_export_is_stable_for_identical_inputs(root: Path, log: EventLog) -> None:
    log.append("sample.accepted", "c1", {"response": "ok"})
    index = CampaignIndex(root)
    results = [make_result(sample_key())]
    index.ingest(log, results)
    assert export_json(index, log, results) == export_json(index, log, results)
    assert export_csv(index, results) == export_csv(index, results)


def test_export_bundle_writes_both_files(root: Path, log: EventLog) -> None:
    log.append("sample.accepted", "c1", {"response": "ok"})
    index = CampaignIndex(root)
    results = [make_result(sample_key())]
    index.ingest(log, results)
    paths = export_bundle(root / "out", index, log, results)
    assert set(paths) == {"json", "csv"}
    assert paths["json"].is_file() and paths["csv"].is_file()
    json.loads(paths["json"].read_text(encoding="utf-8"))
    assert not list((root / "out").glob("*.tmp")), "exports are written atomically"


def test_index_counts_survive_a_reopen(root: Path, log: EventLog) -> None:
    log.append("sample.accepted", "c1", {"response": "ok"})
    index = CampaignIndex(root)
    index.ingest(log, [make_result(sample_key())])
    reopened = CampaignIndex(root)
    assert reopened.counts() == index.counts()
    assert reopened.accepted_sample_count() == 1


def test_empty_campaign_exports_cleanly(root: Path, log: EventLog) -> None:
    index = CampaignIndex(root)
    stats = index.ingest(log, [])
    assert stats.events_indexed == 0
    assert index.counts()["events"] == 0
    payload = json.loads(export_json(index, log, []))
    assert payload["events"] == []
    assert payload["samples"] == []
    assert payload["integrity"]["is_clean"] is True


def test_events_of_type_and_samples_for_task_are_queryable(root: Path, log: EventLog) -> None:
    log.append("request.dispatched", "c1", {"seq": 1})
    log.append("request.completed", "c1", {"seq": 2})
    index = CampaignIndex(root)
    key = sample_key()
    index.ingest(log, [make_result(key)])

    assert len(index.events_of_type("request.dispatched")) == 1
    assert len(index.events_of_type("request.completed")) == 1
    rows = index.samples_for_task("ifeval::item-1")
    assert len(rows) == 1
    assert rows[0]["is_accepted"] == 1


# ---------------------------------------------------------------------------
# Regression tests for the G02 review findings
# ---------------------------------------------------------------------------


def test_both_accepted_sample_counts_agree(root: Path, log: EventLog) -> None:
    """Defect: the index published a row count beside a distinct count and they differed."""
    log.append("sample.accepted", "c1", {"response": "ok"})
    key = sample_key()
    results = [make_result(key, attempt=1), make_result(key, attempt=2)]
    index = CampaignIndex(root)
    index.ingest(log, results)
    assert index.counts()["accepted_samples"] == index.accepted_sample_count() == 1
    assert index.counts()["samples"] == 2


def test_export_does_not_publish_two_different_accepted_counts(root: Path, log: EventLog) -> None:
    log.append("sample.accepted", "c1", {"response": "ok"})
    key = sample_key()
    results = [make_result(key, attempt=1), make_result(key, attempt=2)]
    index = CampaignIndex(root)
    index.ingest(log, results)
    payload = json.loads(export_json(index, log, results))
    assert payload["index_counts"]["accepted_samples"] == payload["accepted_samples"] == 1


def test_a_hostile_attempt_id_is_redacted_and_defused_in_csv(root: Path, log: EventLog) -> None:
    """Defect: attempt_id bypassed _csv_cell, leaking both a secret and a formula."""
    log.append("sample.accepted", "c1", {"response": "ok"})
    index = CampaignIndex(root, extra_secrets=frozenset({ERROR_CANARY}))
    result = make_result(sample_key()).model_copy(
        update={"attempt_id": f"=cmd|'/c calc'!A1 {ERROR_CANARY}"}
    )
    index.ingest(log, [result])
    text = export_csv(index, [result])
    assert ERROR_CANARY not in text
    rows = list(csv.DictReader(io.StringIO(text)))
    assert rows[0]["attempt_id"].startswith("'")
    assert not rows[0]["attempt_id"].startswith("=")


def test_a_subset_rebuild_reports_what_it_dropped(root: Path, log: EventLog) -> None:
    """Defect: a shorter re-ingest silently dropped rows. The loss is now counted."""
    log.append("sample.accepted", "c1", {"response": "ok"})
    index = CampaignIndex(root)
    first = [make_result(sample_key(repeat=1)), make_result(sample_key(repeat=2))]
    assert index.ingest(log, first).dropped_rows == 0

    partial = index.ingest(log, [first[0]])
    assert partial.dropped_rows == 1, "the removed row must be reported, not silent"
    assert index.counts()["samples"] == 1
    assert partial.to_dict()["dropped_rows"] == 1


def test_a_full_rebuild_drops_nothing(root: Path, log: EventLog) -> None:
    log.append("sample.accepted", "c1", {"response": "ok"})
    index = CampaignIndex(root)
    results = [make_result(sample_key(repeat=1)), make_result(sample_key(repeat=2))]
    for _ in range(3):
        assert index.ingest(log, results).dropped_rows == 0
        assert index.counts()["samples"] == 2


def test_events_skipped_counts_damaged_records_not_blank_lines(root: Path, log: EventLog) -> None:
    """Defect: blank lines inflated events_skipped and the truncated line was uncounted."""
    log.append("sample.accepted", "c1", {"response": "ok"})
    with log.path.open("a", encoding="utf-8") as handle:
        handle.write("\n\n")
        handle.write("{interrupted")
    index = CampaignIndex(root)
    stats = index.ingest(log, [])
    assert stats.events_skipped == 1
    assert stats.damaged_records == 1
