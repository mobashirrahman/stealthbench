"""G02 gate: durability, redaction and replay (tasks T02A, T02B, T02C).

Gate criteria, each with a direct test rather than an inference:

* an interrupted write cannot invent a completed sample
* redacted exports contain no canary secrets
* replaying saved generations does not contact a provider
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from stealthbench.storage.events import REDACTED, EventLog, contains_secret
from stealthbench.storage.index import CampaignIndex, export_csv, export_json
from stealthbench.storage.replay import (
    ReplayReader,
    ReplayRejected,
    replay_digest,
    write_fixture_transcript,
)

pytestmark = pytest.mark.contract

CANARY = "sk-canary-g02-0123456789abcdefghijkl"


@pytest.fixture
def store(tmp_path: Path) -> Path:
    root = tmp_path / "store"
    for index in range(3):
        write_fixture_transcript(
            root,
            campaign_id="offline-demo",
            item_id=f"syn-if-00{index}",
            endpoint_id="fixture-a",
            response=f"response {index}",
        )
    return root


# ---------------------------------------------------------------------------
# Criterion: an interrupted write cannot invent a completed sample
# ---------------------------------------------------------------------------


def test_truncation_never_produces_a_completion(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "run")
    log.append("request.dispatched", "c1", {"attempt": 1})
    with log.path.open("a", encoding="utf-8") as handle:
        handle.write('{"event_id":"x","event_type":"sample.accepted","payload_h')

    completions, report = log.completion_events()
    assert completions == [], "an interrupted completion must not be reported as one"
    assert not report.is_clean


def test_payload_precedes_the_event_that_names_it(tmp_path: Path) -> None:
    """The write order is what makes the crash window safe."""
    log = EventLog(tmp_path / "run")
    event = log.append("sample.accepted", "c1", {"response": "ok"})
    assert log.artifacts.has(event.payload_hash), "payload must be durable first"


def test_artifact_digests_are_verified_on_read(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "run")
    event = log.append("sample.accepted", "c1", {"response": "ok"})
    log.artifacts.path_for(event.payload_hash).write_text('{"response":"changed"}')
    assert log.integrity().kinds() == {"payload_digest_mismatch"}
    assert log.read_all()[0] == []


def test_index_records_damage_and_withholds_the_sample(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "run")
    event = log.append("sample.accepted", "c1", {"response": "ok"})
    log.artifacts.path_for(event.payload_hash).unlink()
    index = CampaignIndex(tmp_path / "run")
    index.ingest(log, [])
    assert index.counts()["events"] == 0
    assert index.has_damage()


# ---------------------------------------------------------------------------
# Criterion: redacted exports contain no canary secrets
# ---------------------------------------------------------------------------


def test_no_canary_survives_any_export_path(tmp_path: Path) -> None:
    root = tmp_path / "run"
    log = EventLog(root)
    log.append(
        "request.failed",
        "c1",
        {
            "error_body": f"invalid key {CANARY}",
            "response_headers": {"authorization": f"Bearer {CANARY}"},
            "nested": {"list": [f"value {CANARY}"]},
        },
    )
    index = CampaignIndex(root)
    index.ingest(log, [])

    artifacts = {
        "events.jsonl": (root / "events.jsonl").read_text(encoding="utf-8"),
        "json export": export_json(index, log, []),
        "csv export": export_csv(index, []),
    }
    for name, text in artifacts.items():
        assert CANARY not in text, f"{name} leaked the canary"
        assert not contains_secret(text), f"{name} still contains something secret-shaped"


def test_a_header_named_like_a_credential_is_always_redacted() -> None:
    log_payload = {"headers": {"Authorization": "opaque", "Cookie": "opaque", "Accept": "*/*"}}
    from stealthbench.storage.events import redact_mapping

    redacted = redact_mapping(log_payload)
    assert redacted["headers"]["Authorization"] == REDACTED
    assert redacted["headers"]["Cookie"] == REDACTED
    assert redacted["headers"]["Accept"] == "*/*"


def test_redaction_does_not_damage_diagnostic_content() -> None:
    """Over-redaction would destroy the evidence a redaction test protects."""
    from stealthbench.storage.events import redact_text

    for text in (
        "manifest_hash=9e420fb9ef63358566eafd0c0321e393e519752f2ab62a02185e4a6c53c815aa",
        "item syn-if-001 category instruction_following repeats 1",
        "cost 0.0125 usd, 4800 tokens, temperature 0.0",
    ):
        assert redact_text(text) == text


# ---------------------------------------------------------------------------
# Criterion: replay does not contact a provider
# ---------------------------------------------------------------------------


def test_replay_reconstructs_without_any_transport(store: Path) -> None:
    report = ReplayReader(store).read()
    assert report.accepted_samples == 3
    assert report.damage == ()
    assert [r.response for r in report.results] == ["response 0", "response 1", "response 2"]


def test_replay_is_deterministic(store: Path) -> None:
    assert replay_digest(ReplayReader(store).stored_results()) == replay_digest(
        ReplayReader(store).stored_results()
    )


def test_replay_module_has_no_network_imports() -> None:
    source = Path(__file__).resolve().parents[2] / "src" / "stealthbench" / "storage" / "replay.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            modules.add(node.module.split(".")[0])
    for forbidden in ("httpx", "requests", "socket", "urllib", "http"):
        assert forbidden not in modules, f"replay imports {forbidden}"


def test_replay_class_exposes_no_client_attribute() -> None:
    """A provider client cannot be attached, so it cannot be called."""
    reader = ReplayReader(Path("/tmp/diagnostic-only"))
    for attribute in ("client", "transport", "session", "endpoint", "api_key"):
        assert not hasattr(reader, attribute)


def test_strict_replay_refuses_corrupt_provenance(store: Path) -> None:
    with (store / "events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write('{"event_id": "broken"')
    with pytest.raises(ReplayRejected):
        ReplayReader(store).read(strict=True)


# ---------------------------------------------------------------------------
# Cross-cutting: unknown stays unknown through the whole storage path
# ---------------------------------------------------------------------------


def test_absent_usage_survives_log_index_export_and_replay(tmp_path: Path) -> None:
    root = tmp_path / "run"
    write_fixture_transcript(
        root,
        campaign_id="c",
        item_id="i",
        endpoint_id="e",
        response="x",
        input_tokens=None,
        output_tokens=None,
    )
    reader = ReplayReader(root)
    result = reader.stored_results()[0]
    assert result.usage.input_tokens is None

    index = reader.index()
    assert index.sample_usage_totals()["input_tokens"] is None

    exported = json.loads(export_json(index, reader.log, [result]))
    assert exported["samples"][0]["usage"]["input_tokens"] is None
    assert index.accepted_sample_count() == 1
