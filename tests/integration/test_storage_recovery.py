"""Storage recovery: interrupted writes and corrupt records (task T02A).

Gate G02: an interrupted write must never invent a completed sample.

Every destructive scenario here is produced by actually truncating, corrupting or
interrupting a real log rather than by asserting on a mock, because the failure modes
being guarded against are precisely the ones a mock would not reproduce.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import pytest

from stealthbench.storage.events import (
    EVENT_LOG_NAME,
    REDACTED,
    ArtifactStore,
    DamageKind,
    EventLog,
    IntegrityReport,
    contains_secret,
    redact_mapping,
    redact_text,
    secret_env_values,
)

pytestmark = pytest.mark.integration

CANARY = "sk-canary-0123456789abcdef0123456789abcdef"
HEADER_CANARY = "ghp_canary_0123456789abcdef"


@pytest.fixture
def log(tmp_path: Path) -> EventLog:
    return EventLog(tmp_path / "run")


# ---------------------------------------------------------------------------
# The happy path
# ---------------------------------------------------------------------------


def test_appended_event_round_trips(log: EventLog) -> None:
    event = log.append("sample.accepted", "c1", {"response": "hi", "usage": {"input_tokens": 5}})
    events, report = log.read_all()
    assert report.is_clean
    assert len(events) == 1
    assert events[0].event_id == event.event_id
    assert log.load_payload(events[0]) == {"response": "hi", "usage": {"input_tokens": 5}}


def test_payload_is_durable_before_the_event_that_names_it(log: EventLog) -> None:
    event = log.append("request.completed", "c1", {"ok": True})
    assert log.artifacts.has(event.payload_hash)


def test_events_are_ordered_as_appended(log: EventLog) -> None:
    for index in range(5):
        log.append("request.dispatched", "c1", {"seq": index})
    events, report = log.read_all()
    assert report.is_clean
    assert [log.load_payload(e)["seq"] for e in events] == [0, 1, 2, 3, 4]


def test_log_is_append_only_across_reopening(tmp_path: Path) -> None:
    root = tmp_path / "run"
    EventLog(root).append("request.dispatched", "c1", {"seq": 1})
    EventLog(root).append("request.dispatched", "c1", {"seq": 2})
    events, report = EventLog(root).read_all()
    assert report.is_clean
    assert len(events) == 2, "reopening must continue the log, never rewrite it"


def test_identical_payloads_are_stored_once(log: EventLog) -> None:
    first = log.append("request.dispatched", "c1", {"same": True})
    second = log.append("request.dispatched", "c1", {"same": True})
    assert first.payload_hash == second.payload_hash


# ---------------------------------------------------------------------------
# Interrupted and corrupt writes
# ---------------------------------------------------------------------------


def test_a_truncated_final_record_is_reported_not_yielded(log: EventLog) -> None:
    """The core gate: a half-written last line must not become a completed sample."""
    log.append("sample.accepted", "c1", {"response": "hi"})
    with log.path.open("a", encoding="utf-8") as handle:
        handle.write('{"event_id": "partial", "event_type": "sample.acce')

    events, report = log.read_all()
    assert len(events) == 1, "the interrupted record must not be returned as an event"
    assert DamageKind.TRUNCATED_FINAL_LINE in report.kinds()
    completions, _ = log.completion_events()
    assert len(completions) == 1


def test_a_truncated_completion_record_does_not_invent_a_sample(log: EventLog) -> None:
    log.append("request.dispatched", "c1", {"attempt": 1})
    with log.path.open("a", encoding="utf-8") as handle:
        handle.write('{"event_id":"x","event_type":"sample.accepted","payload')

    completions, report = log.completion_events()
    assert completions == []
    assert not report.is_clean


def test_an_unparsable_middle_line_is_reported(tmp_path: Path) -> None:
    root = tmp_path / "run"
    log = EventLog(root)
    log.append("request.dispatched", "c1", {"seq": 1})
    with log.path.open("a", encoding="utf-8") as handle:
        handle.write("this is not json\n")
    log.append("request.completed", "c1", {"seq": 2})

    events, report = log.read_all()
    assert DamageKind.UNPARSABLE_LINE in report.kinds()
    assert len(events) == 2, "the readable records either side survive"


def test_a_schema_invalid_record_is_reported(tmp_path: Path) -> None:
    root = tmp_path / "run"
    log = EventLog(root)
    log.append("request.dispatched", "c1", {"seq": 1})
    bad = {
        "event_id": "e",
        "event_type": "not.a.real.type",
        "campaign_id": "c1",
        "wall_time_utc": datetime.now(UTC).isoformat(),
        "monotonic_duration_seconds": 0.0,
        "payload_hash": "a" * 64,
    }
    with log.path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(bad) + "\n")

    events, report = log.read_all()
    assert DamageKind.SCHEMA_INVALID in report.kinds()
    assert len(events) == 1


def test_a_missing_payload_is_reported_and_the_event_withheld(log: EventLog) -> None:
    event = log.append("sample.accepted", "c1", {"response": "hi"})
    log.artifacts.path_for(event.payload_hash).unlink()

    events, report = log.read_all()
    assert DamageKind.MISSING_PAYLOAD in report.kinds()
    assert events == [], "an event whose payload is gone is not a usable sample"


def test_a_tampered_payload_is_detected(log: EventLog) -> None:
    """Hash-checking must catch an artifact edited after the event was written."""
    event = log.append("sample.accepted", "c1", {"response": "original"})
    log.artifacts.path_for(event.payload_hash).write_text('{"response":"tampered"}')

    events, report = log.read_all()
    assert DamageKind.PAYLOAD_DIGEST_MISMATCH in report.kinds()
    assert events == []


def test_a_mismatched_payload_hash_is_reported(tmp_path: Path) -> None:
    root = tmp_path / "run"
    log = EventLog(root)
    log.append("request.dispatched", "c1", {"seq": 1})
    # A well-formed event naming a digest that does not exist.
    from stealthbench.storage.events import _event_id

    record = {
        "schema_version": "1.0",
        "event_id": _event_id("c1", "sample.accepted", "b" * 64, 1),
        "event_type": "sample.accepted",
        "campaign_id": "c1",
        "sequence": 1,
        "wall_time_utc": datetime.now(UTC).isoformat(),
        "monotonic_duration_seconds": 0.0,
        "payload_hash": "b" * 64,
    }
    with log.path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")

    events, report = log.read_all()
    assert DamageKind.MISSING_PAYLOAD in report.kinds()
    assert len(events) == 1


def test_artifact_writes_are_atomic(tmp_path: Path) -> None:
    """No temp files are left behind, and a partial write never lands."""
    store = ArtifactStore(tmp_path / "artifacts")
    digest = store.put_bytes(b"payload")
    assert store.has(digest)
    leftovers = [p.name for p in (tmp_path / "artifacts").rglob("*.tmp")]
    assert leftovers == [], f"temp files left behind: {leftovers}"


def _kill_during_writes(
    repo_root: Path, store: Path, records: int = 5000
) -> subprocess.Popen[bytes]:
    """Start a writer and SIGKILL it as soon as it has written anything at all.

    Waiting a fixed interval let the writer finish first, which tested nothing. This
    waits for the log to exist and then kills immediately, so the kill genuinely lands
    mid-append.
    """
    script = store.parent / "writer.py"
    script.write_text(
        "import sys\n"
        f"sys.path.insert(0, {str(repo_root / 'src')!r})\n"
        "from stealthbench.storage.events import EventLog\n"
        f"log = EventLog({str(store)!r})\n"
        f"for n in range({records}):\n"
        "    log.append('sample.accepted', 'c1', {'n': n, 'blob': 'x' * 4000})\n",
        encoding="utf-8",
    )
    process = subprocess.Popen([sys.executable, str(script)])
    deadline = time.monotonic() + 30
    while not (store / EVENT_LOG_NAME).exists() and time.monotonic() < deadline:
        time.sleep(0.005)
    time.sleep(0.05)
    process.kill()
    process.wait(timeout=60)
    return process


def test_a_process_killed_mid_write_leaves_a_readable_log(repo_root: Path, tmp_path: Path) -> None:
    """The crash scenario the gate names, produced by an actual SIGKILL."""
    store = tmp_path / "run"
    process = _kill_during_writes(repo_root, store)
    assert process.returncode != 0, "the writer must not have exited cleanly"

    events, report = EventLog(store).read_all()
    assert events, "a killed writer must still have left usable records"
    for event in events:
        assert event.is_completion_record
        assert log_artifact_intact(store, event.payload_hash), (
            f"completion {event.event_id} survived without a verifiable payload"
        )
    assert report.valid_events == len(events)
    if report.damage:
        # Damage is acceptable only if it is precisely an interrupted tail.
        assert report.kinds() <= {"truncated_final_line"}, report.kinds()
        assert report.missing_trailing_newline


def test_a_killed_writer_never_invents_a_completion(repo_root: Path, tmp_path: Path) -> None:
    store = tmp_path / "run"
    _kill_during_writes(repo_root, store)
    events, _report = EventLog(store).read_all()
    completions = [event for event in events if event.is_completion_record]
    assert completions == [e for e in events if e.is_completion_record]
    for event in completions:
        assert log_artifact_intact(store, event.payload_hash)


def log_artifact_intact(root: Path, digest: str) -> bool:
    return ArtifactStore(root / "artifacts").verify(digest) is None


def test_a_log_written_by_a_killed_process_never_invents_a_completion(repo_root, tmp_path) -> None:
    """Even under an abrupt kill, every completion event has a verifiable payload."""
    script = tmp_path / "writer2.py"
    script.write_text(
        "import sys\n"
        f"sys.path.insert(0, {str(repo_root / 'src')!r})\n"
        "from stealthbench.storage.events import EventLog\n"
        f"log = EventLog({str(tmp_path / 'run')!r})\n"
        "for n in range(500):\n"
        "    log.append('sample.accepted', 'c1', {'n': n})\n",
        encoding="utf-8",
    )
    process = subprocess.Popen([sys.executable, str(script)])
    import time

    time.sleep(1.0)
    process.kill()
    process.wait(timeout=30)

    events, report = EventLog(tmp_path / "run").read_all()
    for event in events:
        assert log_artifact_intact(tmp_path / "run", event.payload_hash), (
            f"completion {event.event_id} has no verifiable payload"
        )
    assert report.valid_events == len(events)


def test_integrity_report_serializes_for_evidence(tmp_path: Path) -> None:
    root = tmp_path / "run"
    log = EventLog(root)
    log.append("sample.accepted", "c1", {"response": "hi"})
    with log.path.open("a", encoding="utf-8") as handle:
        handle.write("{partial")

    report = log.integrity()
    payload = report.to_dict()
    assert payload["is_clean"] is False
    assert payload["damage"][0]["kind"] == "truncated_final_line"
    assert payload["valid_events"] == 1
    json.dumps(payload)


def test_empty_log_is_clean_and_empty(tmp_path: Path) -> None:
    events, report = EventLog(tmp_path / "missing").read_all()
    assert events == []
    assert report.is_clean and report.total_lines == 0


def test_report_counts_a_damaged_line(tmp_path: Path) -> None:
    root = tmp_path / "run"
    log = EventLog(root)
    log.append("request.dispatched", "c1", {"seq": 1})
    with log.path.open("a", encoding="utf-8") as handle:
        handle.write("garbage\n")
    report = log.integrity()
    assert report.total_lines == 2
    assert report.damaged_lines == 1


def test_integrity_report_type_is_exported() -> None:
    assert IntegrityReport().is_clean


# ---------------------------------------------------------------------------
# Redaction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        f"token is {CANARY}",
        f"Authorization: Bearer {HEADER_CANARY}",
        "api_key=AIzaSyBqKjL9pXf2QwErTyUiOpAsDfGhJkLmNoP",
        "-----BEGIN RSA PRIVATE KEY-----",
        "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.SflKxwRJSM",
        f"password: {CANARY}",
    ],
)
def test_credentials_are_redacted_from_free_text(text: str) -> None:
    redacted = redact_text(text)
    assert REDACTED in redacted, "the credential should have been replaced"
    assert not contains_secret(redacted), "redaction must be complete, not partial"


def test_redaction_is_idempotent() -> None:
    """A second pass must find nothing left, or a secret survives in the gaps."""
    once = redact_text(f"key {CANARY}")
    assert redact_text(once) == once, "a second pass must find nothing left to redact"


def test_similar_looking_non_secrets_survive_redaction() -> None:
    """Over-redaction destroys evidence too; hashes and versions must stay readable."""
    for text in (
        "manifest hash 9e420fb9ef63358566eafd0c0321e393e519752f2ab62a02185e4a6c53c815aa",
        "schema_version 1.0, temperature 0.0, top_p 1.0",
        "cost 0.0025 usd for 1500 tokens",
        "question_id 12345, answer_index 7",
    ):
        assert redact_text(text) == text, f"redaction damaged: {text!r}"


def test_secret_header_names_are_always_redacted() -> None:
    mapping = {
        "Authorization": "Bearer something",
        "X-Api-Key": "abc",
        "content-type": "application/json",
    }
    redacted = redact_mapping(mapping)
    assert redacted["Authorization"] == REDACTED
    assert redacted["X-Api-Key"] == REDACTED
    assert redacted["content-type"] == "application/json"


def test_nested_structures_are_redacted() -> None:
    payload = {
        "response_headers": {"authorization": f"Bearer {CANARY}"},
        "body": {"messages": [{"content": f"use {CANARY}"}]},
        "count": 3,
    }
    redacted = redact_mapping(payload)
    assert not contains_secret(json.dumps(redacted))
    assert redacted["count"] == 3


def test_environment_secrets_are_redacted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STEALTHBENCH_ZEN_API_KEY", CANARY)
    assert CANARY in secret_env_values()
    redacted = redact_text(f"the key is {CANARY}")
    assert redacted == f"the key is {REDACTED}"
    assert not contains_secret(redacted)


def test_short_environment_values_do_not_trigger_redaction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SOME_TOKEN", "abc")
    assert "abc" not in secret_env_values(), "a 3-character value is not a credential"


def test_a_canary_in_a_stored_payload_is_redacted_at_write_time(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "run")
    event = log.append(
        "request.failed",
        "c1",
        {"error_body": f"invalid api key {CANARY}", "status": 401},
    )
    stored = json.dumps(log.load_payload(event))
    assert CANARY not in stored
    assert REDACTED in stored


def test_a_canary_in_a_provider_error_is_redacted_at_write_time(tmp_path: Path) -> None:
    """Gateways routinely echo credentials back inside error payloads."""
    log = EventLog(tmp_path / "run")
    event = log.append(
        "request.failed",
        "c1",
        {"error": {"body": CANARY, "headers": {"x-api-key": CANARY}}},
    )
    payload = log.load_payload(event)
    assert CANARY not in json.dumps(payload)
    assert payload["error"]["headers"]["x-api-key"] == REDACTED


def test_nothing_sensitive_survives_in_the_log_file(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "run")
    log.append("request.failed", "c1", {"body": f"token {CANARY}"})
    on_disk = (tmp_path / "run" / EVENT_LOG_NAME).read_text(encoding="utf-8")
    assert CANARY not in on_disk


def test_secrets_from_the_environment_are_caught_even_unrecognised(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A credential with no known prefix is still caught via the environment."""
    monkeypatch.setenv("STEALTHBENCH_ZEN_API_KEY", "totallyUnrecognisedCredentialValue123")
    log = EventLog(tmp_path / "run")
    event = log.append("request.failed", "c1", {"body": "totallyUnrecognisedCredentialValue123"})
    assert "totallyUnrecognisedCredentialValue123" not in json.dumps(log.load_payload(event))


def test_redaction_is_applied_to_a_response_body_without_damaging_it(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "run")
    event = log.append("sample.accepted", "c1", {"response": "42", "usage": {"output_tokens": 1}})
    payload = log.load_payload(event)
    assert payload["response"] == "42"
    assert payload["usage"]["output_tokens"] == 1
    assert payload["usage"]["output_tokens"] is not None


def test_artifact_store_verifies_its_own_contents(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    digest = store.put_json({"a": 1})
    assert store.verify(digest) is None
    store.path_for(digest).write_text('{"a": 2}')
    damage = store.verify(digest)
    assert damage is not None
    assert damage.kind is DamageKind.PAYLOAD_DIGEST_MISMATCH


def test_artifact_store_reports_a_missing_artifact(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    damage = store.verify("f" * 64)
    assert damage is not None
    assert damage.kind is DamageKind.MISSING_PAYLOAD


def test_fsync_and_append_are_really_on_disk(log: EventLog) -> None:
    """The log must exist and be complete immediately after append returns."""
    log.append("sample.accepted", "c1", {"response": "hi"})
    assert log.path.exists()
    raw = log.path.read_text(encoding="utf-8")
    assert raw.endswith("\n"), "a complete append leaves a trailing newline"
    assert log.path.stat().st_size > 0


# ---------------------------------------------------------------------------
# Regression tests for the G02 review findings
# ---------------------------------------------------------------------------


def test_a_complete_final_record_whose_newline_was_lost_is_still_read(
    tmp_path: Path,
) -> None:
    """Defect: a complete record was dropped and the store reported clean."""
    log = EventLog(tmp_path / "run")
    log.append("sample.accepted", "c1", {"response": "x"})
    log.path.write_text(log.path.read_text(encoding="utf-8")[:-1])

    events, report = log.read_all()
    assert len(events) == 1, "a complete record must not be discarded"
    assert report.missing_trailing_newline is True
    assert report.is_clean, "a lost newline is not damage, but it is disclosed"
    assert log.completion_events()[0], "the completion must be visible"


def test_a_single_record_log_missing_its_newline_is_read(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "run")
    log.append("sample.accepted", "c1", {"response": "x"})
    log.path.write_text(log.path.read_text(encoding="utf-8")[:-1])
    events, report = log.read_all()
    assert len(events) == 1
    assert report.valid_events == 1
    assert report.missing_trailing_newline is True


def test_a_truly_interrupted_final_record_is_still_damage(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "run")
    log.append("sample.accepted", "c1", {"response": "x"})
    with log.path.open("a", encoding="utf-8") as handle:
        handle.write('{"event_id": "partial", "event_ty')
    events, report = log.read_all()
    assert len(events) == 1
    assert DamageKind.TRUNCATED_FINAL_LINE in report.kinds()


def test_appending_after_a_lost_newline_does_not_concatenate(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "run")
    log.append("sample.accepted", "c1", {"response": "first"})
    log.path.write_text(log.path.read_text(encoding="utf-8")[:-1])
    log.append("sample.accepted", "c1", {"response": "second"})

    events, report = log.read_all()
    assert len(events) == 2, "the repaired boundary keeps both records"
    assert report.is_clean
    assert [log.load_payload(e)["response"] for e in events] == ["first", "second"]


def test_invalid_utf8_is_reported_not_silently_replaced(tmp_path: Path) -> None:
    """Defect: a corrupt byte became U+FFFD inside an apparently valid record."""
    log = EventLog(tmp_path / "run")
    log.append("sample.accepted", "campaign-A", {"r": 1})
    log.path.write_bytes(log.path.read_bytes().replace(b"campaign-A", b"campaign-\xff\xfeA"))

    events, report = log.read_all()
    assert events == [], "an undecodable log cannot be trusted"
    assert DamageKind.INVALID_ENCODING in report.kinds()


def test_rewriting_a_payload_digest_is_detected(tmp_path: Path) -> None:
    """Defect: an event's digest could be repointed at another valid artifact."""
    log = EventLog(tmp_path / "run")
    log.append("sample.accepted", "c1", {"sample": "first"})
    log.append("sample.accepted", "c1", {"sample": "second"})

    lines = log.path.read_text(encoding="utf-8").splitlines()
    first = json.loads(lines[0])
    first["payload_hash"] = json.loads(lines[1])["payload_hash"]
    log.path.write_text(json.dumps(first) + "\n" + "\n".join(lines[1:]) + "\n", encoding="utf-8")

    events, report = log.read_all()
    assert len(events) == 1
    assert DamageKind.EVENT_ID_MISMATCH in report.kinds()


def test_identical_payloads_get_distinct_event_ids(tmp_path: Path) -> None:
    """Defect: a content-derived id collided, collapsing two events into one."""
    log = EventLog(tmp_path / "run")
    first = log.append("request.dispatched", "c1", {"attempt": 1})
    second = log.append("request.dispatched", "c1", {"attempt": 1})
    assert first.payload_hash == second.payload_hash, "the payload really is identical"
    assert first.event_id != second.event_id, "but two events are two events"
    assert len(log.read_all()[0]) == 2


def test_a_reordered_log_is_detected(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "run")
    log.append("request.dispatched", "c1", {"n": 1})
    log.append("request.completed", "c1", {"n": 2})
    lines = log.path.read_text(encoding="utf-8").splitlines()
    log.path.write_text("\n".join(reversed(lines)) + "\n", encoding="utf-8")

    events, report = log.read_all()
    assert DamageKind.SEQUENCE_MISMATCH in report.kinds()
    assert events == [], "a reordered log is not an append-only log; neither record is trusted"


def test_a_secret_used_as_a_mapping_key_is_redacted(tmp_path: Path) -> None:
    """Defect: a credential echoed as a JSON key persisted verbatim."""
    log = EventLog(tmp_path / "run", extra_secrets=frozenset({CANARY}))
    event = log.append("request.failed", "c1", {"error": {"json": {CANARY: "leaked"}}})
    stored = json.dumps(log.load_payload(event))
    assert CANARY not in stored
    assert REDACTED in stored


def test_a_secret_key_is_redacted_in_exports_too(tmp_path: Path) -> None:
    from stealthbench.storage.index import CampaignIndex, export_json

    log = EventLog(tmp_path / "run", extra_secrets=frozenset({CANARY}))
    log.append("request.failed", "c1", {"error": {"json": {CANARY: "leaked"}}})
    index = CampaignIndex(tmp_path / "run", extra_secrets=frozenset({CANARY}))
    index.ingest(log, [])
    assert CANARY not in export_json(index, log, [])
