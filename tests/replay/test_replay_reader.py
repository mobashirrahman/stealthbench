"""Deterministic replay (task T02C).

Gate G02: replaying saved generations does not contact a provider.

The no-network property is tested structurally (this module has no transport and the
test process has its sockets denied) and behaviourally (replay of a store works with
every credential environment variable set to something that would fail loudly if it
were ever used).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from stealthbench.schemas.results import (
    DeliveryStatus,
    DenominatorEligibility,
    GradeResult,
    SampleKey,
)
from stealthbench.storage.events import EventLog
from stealthbench.storage.replay import (
    ReplayReader,
    ReplayRejected,
    describe_store,
    replay_digest,
    write_fixture_transcript,
)

pytestmark = pytest.mark.replay

CANARY = "sk-canary-replay-0123456789abcdef"


@pytest.fixture
def store(tmp_path: Path) -> Path:
    root = tmp_path / "store"
    write_fixture_transcript(
        root,
        campaign_id="offline-demo",
        item_id="syn-if-001",
        endpoint_id="fixture-a",
        response="a haiku",
        input_tokens=12,
        output_tokens=8,
    )
    write_fixture_transcript(
        root,
        campaign_id="offline-demo",
        item_id="syn-if-002",
        endpoint_id="fixture-a",
        response="another haiku",
        input_tokens=15,
        output_tokens=9,
    )
    return root


# ---------------------------------------------------------------------------
# No external calls
# ---------------------------------------------------------------------------


def test_replay_reads_a_store_without_a_transport(store: Path) -> None:
    report = ReplayReader(store).read()
    assert report.accepted_samples == 2
    assert report.damage == ()


def test_replay_works_with_credentials_set_and_network_denied(
    store: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sockets are denied by the autouse guard; credentials present must change nothing.

    If replay ever reached for a provider, the denied socket layer or the bogus
    credential would fail the run rather than silently succeeding.
    """
    monkeypatch.setenv("STEALTHBENCH_ZEN_API_KEY", CANARY)
    monkeypatch.setenv("STEALTHBENCH_SPENDING_CAP_USD", "0.01")
    report = ReplayReader(store, extra_secrets=frozenset({CANARY})).read()
    assert report.accepted_samples == 2


def test_replay_module_imports_no_transport() -> None:
    """Structural guarantee: the reader cannot hold a provider client."""
    import ast

    module_path = Path(ReplayReader.__module__.replace(".", "/")).with_suffix(".py")
    source = Path(__file__).resolve().parents[2] / "src" / "stealthbench" / "storage" / "replay.py"
    assert source.is_file() and module_path.name == "replay.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imported.add(node.module.split(".")[0])
    assert "httpx" not in imported
    assert "requests" not in imported
    assert "socket" not in imported
    assert "urllib" not in imported


# ---------------------------------------------------------------------------
# Reconstruction
# ---------------------------------------------------------------------------


def test_reconstructed_result_matches_what_was_stored(store: Path) -> None:
    results = ReplayReader(store).stored_results()
    assert [result.response for result in results] == ["a haiku", "another haiku"]
    first = results[0]
    assert first.usage.input_tokens == 12
    assert first.usage.output_tokens == 8
    assert first.finish_status == "stop"
    assert first.effective_settings["temperature"] == 0.0


def test_absent_usage_survives_replay(tmp_path: Path) -> None:
    root = tmp_path / "store"
    write_fixture_transcript(
        root,
        campaign_id="c",
        item_id="i",
        endpoint_id="e",
        response="x",
        input_tokens=None,
        output_tokens=None,
    )
    result = ReplayReader(root).stored_results()[0]
    assert result.usage.input_tokens is None
    assert result.usage.output_tokens is None
    assert json.loads(result.model_dump_json())["usage"]["input_tokens"] is None


def test_unresolved_attempts_are_replayed_as_unresolved(store: Path) -> None:
    """Defect: an unresolved attempt was silently dropped, losing the ambiguous state."""
    from stealthbench.schemas.results import unresolved_attempts

    log = EventLog(store)
    key = SampleKey(campaign_id="c1", endpoint_id="e", task_id="ifeval::ghost", repeat_id=1)
    log.append(
        "request.dispatched",
        "c1",
        {
            "attempt_id": "ghost-1",
            "attempt_number": 1,
            "delivery_status": str(DeliveryStatus.UNRESOLVED),
            "sample_key": key.model_dump(),
            "result": {"response": None},
        },
    )
    report = ReplayReader(store).read()
    ghost = [r for r in report.results if r.sample_key.task_id == "ifeval::ghost"]
    assert len(ghost) == 1, "the unresolved attempt must be reconstructed, not dropped"
    assert ghost[0].delivery_status is DeliveryStatus.UNRESOLVED
    assert ghost[0].is_accepted_sample is False
    assert unresolved_attempts(report.results) == ("ghost-1",)
    assert all(
        r.sample_key.task_id != "ifeval::ghost" for r in report.results if r.is_accepted_sample
    )


def test_unresolved_attempts_reach_the_index_and_export(store: Path) -> None:
    from stealthbench.storage.index import export_json

    log = EventLog(store)
    key = SampleKey(campaign_id="c1", endpoint_id="e", task_id="ifeval::ghost", repeat_id=1)
    log.append(
        "request.dispatched",
        "c1",
        {
            "attempt_id": "ghost-1",
            "delivery_status": str(DeliveryStatus.UNRESOLVED),
            "sample_key": key.model_dump(),
            "result": {"response": None},
        },
    )
    reader = ReplayReader(store)
    results = list(reader.read().results)
    index = reader.index()
    assert index.counts()["samples"] == len(results)
    assert index.accepted_sample_count() == 2, "the ghost is unresolved, so it is not accepted"
    exported = json.loads(export_json(index, reader.log, results))
    statuses = {s["delivery_status"] for s in exported["samples"]}
    assert "unresolved" in statuses


def test_replay_is_deterministic_across_reads(store: Path) -> None:
    reader = ReplayReader(store)
    first = replay_digest(reader.stored_results())
    second = replay_digest(ReplayReader(store).stored_results())
    assert first == second


def test_replay_digest_changes_when_a_response_changes(store: Path) -> None:
    before = replay_digest(ReplayReader(store).stored_results())
    log = EventLog(store)
    write_fixture_transcript(
        store,
        campaign_id="offline-demo",
        item_id="syn-if-001",
        endpoint_id="fixture-a",
        response="a DIFFERENT haiku",
    )
    del log
    assert replay_digest(ReplayReader(store).stored_results()) != before


def test_store_description_is_stable(store: Path) -> None:
    first = describe_store(store)
    second = describe_store(store)
    assert first == second
    assert first["event_count"] == 2
    assert first["completion_count"] == 2


# ---------------------------------------------------------------------------
# Provenance is verified, not assumed
# ---------------------------------------------------------------------------


def test_strict_replay_refuses_a_damaged_store(store: Path) -> None:
    with (store / "events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("{interrupted")
    with pytest.raises(ReplayRejected, match="does not support a complete replay"):
        ReplayReader(store).read(strict=True)


def test_lenient_replay_reports_damage_instead(store: Path) -> None:
    with (store / "events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("{interrupted")
    report = ReplayReader(store).read(strict=False)
    assert "truncated_final_line" in report.damage
    assert report.accepted_samples == 2, "intact records are still usable"


def test_a_payload_without_a_sample_key_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "store"
    log = EventLog(root)
    log.append("sample.accepted", "c1", {"result": {"response": "orphan"}})
    with pytest.raises(ReplayRejected, match="no sample_key"):
        ReplayReader(root).read()


def test_a_payload_with_an_unusable_sample_key_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "store"
    log = EventLog(root)
    log.append("sample.accepted", "c1", {"sample_key": {"campaign_id": "c"}, "result": {}})
    with pytest.raises(ReplayRejected, match="sample_key is unusable"):
        ReplayReader(root).read()


def test_a_result_that_breaks_its_contract_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "store"
    log = EventLog(root)
    log.append(
        "sample.accepted",
        "c1",
        {
            "attempt_id": "a",
            "attempt_number": 0,  # contract requires >= 1
            "sample_key": SampleKey(
                campaign_id="c1", endpoint_id="e", task_id="t", repeat_id=1
            ).model_dump(),
            "result": {"response": "x"},
        },
    )
    with pytest.raises(ReplayRejected, match="does not match the contract"):
        ReplayReader(root).read()


def test_an_accepted_event_with_no_response_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "store"
    log = EventLog(root)
    log.append(
        "sample.accepted",
        "c1",
        {
            "attempt_id": "a",
            "sample_key": SampleKey(
                campaign_id="c1", endpoint_id="e", task_id="t", repeat_id=1
            ).model_dump(),
            "result": {"response": None},
        },
    )
    with pytest.raises(ReplayRejected):
        ReplayReader(root).read()


def test_a_tampered_payload_is_caught_before_replay(store: Path) -> None:
    log = EventLog(store)
    event = next(iter(log.iter_events()))
    log.artifacts.path_for(event.payload_hash).write_text('{"tampered": true}')
    with pytest.raises(ReplayRejected):
        ReplayReader(store).read()


def test_replay_reports_which_records_it_refused(tmp_path: Path) -> None:
    root = tmp_path / "store"
    log = EventLog(root)
    log.append("sample.accepted", "c1", {"result": {"response": "orphan"}})
    report = ReplayReader(root).read(strict=False)
    assert report.rejected
    assert "no sample_key" in report.rejected[0]
    assert report.accepted_samples == 0


def test_replay_of_an_empty_store_is_empty_not_an_error(tmp_path: Path) -> None:
    report = ReplayReader(tmp_path / "nothing").read()
    assert report.results == ()
    assert report.accepted_samples == 0
    assert report.damage == ()


# ---------------------------------------------------------------------------
# Regrading without a model
# ---------------------------------------------------------------------------


def _exact_match_grader(result):
    response = (result.response or "").strip()
    correct = "haiku" in response
    return GradeResult(
        grader_version="test-1.0",
        sample_key=result.sample_key,
        correctness="pass" if correct else "fail",
        format="pass",
        transport="pass",
        evaluator="pass",
        score_components={"matched": 1.0 if correct else 0.0},
        denominator_eligibility=DenominatorEligibility(
            correctness=True, format=True, transport_success=True, evaluator_ran=True
        ),
    )


def test_regrade_produces_identical_results_for_identical_input(store: Path) -> None:
    """Gate G02: replaying saved generations reproduces the grades exactly."""
    reader = ReplayReader(store)
    first = reader.grades(reader.stored_results(), _exact_match_grader)
    second = reader.grades(ReplayReader(store).stored_results(), _exact_match_grader)
    assert first["grade_digest"] == second["grade_digest"]
    assert first["accepted_samples"] == 2


def test_regrade_ignores_non_accepted_results(store: Path) -> None:
    log = EventLog(store)
    key = SampleKey(campaign_id="c1", endpoint_id="e", task_id="ifeval::x", repeat_id=1)
    log.append(
        "request.failed",
        "c1",
        {
            "attempt_id": "failed-1",
            "delivery_status": str(DeliveryStatus.TRANSPORT_FAILED),
            "sample_key": key.model_dump(),
            "result": {"response": None},
        },
    )
    reader = ReplayReader(store)
    graded = reader.grades(reader.stored_results(), _exact_match_grader)
    assert graded["accepted_samples"] == 2, "a failed attempt must not be graded"


def test_regrade_digest_changes_when_the_grader_changes(store: Path) -> None:
    reader = ReplayReader(store)

    def always_fail(result):
        return GradeResult(
            grader_version="test-2.0",
            sample_key=result.sample_key,
            correctness="fail",
            format="pass",
            transport="pass",
            evaluator="pass",
            denominator_eligibility=DenominatorEligibility(
                correctness=True, format=True, transport_success=True, evaluator_ran=True
            ),
        )

    assert (
        reader.grades(reader.stored_results(), _exact_match_grader)["grade_digest"]
        != reader.grades(reader.stored_results(), always_fail)["grade_digest"]
    )


def test_index_is_rebuilt_from_the_store(store: Path) -> None:
    index = ReplayReader(store).index()
    assert index.counts()["events"] == 2
    assert index.accepted_sample_count() == 2
    assert index.counts()["damage"] == 0


def test_replay_never_leaks_a_secret_into_a_rebuilt_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("STEALTHBENCH_ZEN_API_KEY", CANARY)
    root = tmp_path / "store"
    write_fixture_transcript(
        root, campaign_id="c", item_id="i", endpoint_id="e", response=f"key {CANARY}"
    )
    reader = ReplayReader(root, extra_secrets=frozenset({CANARY}))
    index = reader.index()
    from stealthbench.storage.index import export_json

    assert CANARY not in export_json(index, reader.log, reader.stored_results())


def test_a_payload_may_not_claim_a_different_campaign(tmp_path: Path) -> None:
    """Defect: a payload could attribute a sample to a campaign its event was not part of."""
    log = EventLog(tmp_path / "store")
    key = SampleKey(campaign_id="campaign-B", endpoint_id="e", task_id="t", repeat_id=1)
    log.append(
        "sample.accepted",
        "campaign-A",
        {"attempt_id": "a", "sample_key": key.model_dump(), "result": {"response": "x"}},
    )
    with pytest.raises(ReplayRejected, match="payload claims campaign"):
        ReplayReader(tmp_path / "store").read()


def test_a_completion_event_may_not_carry_a_failed_status(tmp_path: Path) -> None:
    """Defect: sample.accepted carrying transport_failed was still indexed as a completion."""
    log = EventLog(tmp_path / "store")
    key = SampleKey(campaign_id="c1", endpoint_id="e", task_id="t", repeat_id=1)
    log.append(
        "sample.accepted",
        "c1",
        {
            "attempt_id": "a",
            "delivery_status": str(DeliveryStatus.TRANSPORT_FAILED),
            "sample_key": key.model_dump(),
            "result": {"response": "x"},
        },
    )
    with pytest.raises(ReplayRejected, match=r"sample\.accepted carries delivery_status"):
        ReplayReader(tmp_path / "store").read()


def test_a_failure_event_may_not_carry_accepted_status(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "store")
    key = SampleKey(campaign_id="c1", endpoint_id="e", task_id="t", repeat_id=1)
    log.append(
        "request.failed",
        "c1",
        {
            "attempt_id": "a",
            "delivery_status": str(DeliveryStatus.ACCEPTED),
            "sample_key": key.model_dump(),
            "result": {"response": "x"},
        },
    )
    with pytest.raises(ReplayRejected, match=r"request\.failed carries delivery_status"):
        ReplayReader(tmp_path / "store").read()
