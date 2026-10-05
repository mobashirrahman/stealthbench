"""Operational canary scheduling (G13 T13C).

Acceptance: jobs are idempotent, observe caps and require configured live
authorization; fixture mode runs offline.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from stealthbench.benchmarks.workflow import run_offline
from stealthbench.runner.monitoring import (
    CanaryError,
    CanaryRefused,
    due_canaries,
    list_canaries,
    run_canary_offline,
    schedule_canary,
    validate_job_id,
)
from stealthbench.schemas.campaign import CampaignManifest
from tests.conftest import REPO_ROOT

pytestmark = pytest.mark.integration


def _manifest(campaign_id: str) -> CampaignManifest:
    payload = json.loads((REPO_ROOT / "configs" / "offline-demo.json").read_text(encoding="utf-8"))
    payload["campaign_id"] = campaign_id
    payload.pop("authorization", None)
    return CampaignManifest.model_validate(payload)


def _live_manifest(campaign_id: str) -> CampaignManifest:
    payload = json.loads((REPO_ROOT / "configs" / "offline-demo.json").read_text(encoding="utf-8"))
    payload["campaign_id"] = campaign_id
    payload["mode"] = "live_authorized"
    payload["endpoints"][0]["transport"] = "zen"
    payload["endpoints"][0]["credential_ref"] = "STEALTHBENCH_API_KEY"
    payload["authorization"] = {
        "required": True,
        "spending_cap_usd": None,
        "authorized_by": None,
    }
    return CampaignManifest.model_validate(payload)


def test_schedule_is_idempotent(tmp_path: Path) -> None:
    manifest = _manifest("c-canary-idem")
    state_dir = tmp_path / "state"
    first = schedule_canary(state_dir, "nightly", manifest, 3600.0, now=1000.0)
    second = schedule_canary(state_dir, "nightly", manifest, 3600.0, now=1000.0)
    assert first == second
    assert len(list(state_dir.glob("*.json"))) == 1
    # A changed interval updates once, then is idempotent again.
    third = schedule_canary(state_dir, "nightly", manifest, 7200.0, now=1000.0)
    assert third.interval_seconds == 7200.0
    fourth = schedule_canary(state_dir, "nightly", manifest, 7200.0, now=1000.0)
    assert third == fourth


def test_due_canaries(tmp_path: Path) -> None:
    manifest = _manifest("c-canary-due")
    state_dir = tmp_path / "state"
    schedule_canary(state_dir, "job-a", manifest, 3600.0, now=1000.0)
    assert [item.job_id for item in due_canaries(state_dir, now=1000.0)] == ["job-a"]
    assert due_canaries(state_dir, now=999.0) == []
    assert len(list_canaries(state_dir)) == 1


def test_offline_run_is_idempotent(tmp_path: Path) -> None:
    manifest = _manifest("c-canary-run")
    state_dir = tmp_path / "state"
    artifacts_root = tmp_path / "canaries"
    schedule_canary(state_dir, "nightly", manifest, 3600.0, now=1000.0)

    first = run_canary_offline(
        state_dir, "nightly", manifest, artifacts_root=artifacts_root, now=1100.0
    )
    assert first["canary_idempotent"] is False
    assert first["accepted_samples"] == 3
    events_path = artifacts_root / f"c-canary-run-{first['manifest_hash'][:12]}" / "events.jsonl"
    before_text = events_path.read_text(encoding="utf-8")

    second = run_canary_offline(
        state_dir, "nightly", manifest, artifacts_root=artifacts_root, now=1200.0
    )
    assert second["canary_idempotent"] is True
    assert second["grade_digest"] == first["grade_digest"]
    assert second["manifest_hash"] == first["manifest_hash"]
    # No second dispatch: the event log is unchanged.
    after_text = events_path.read_text(encoding="utf-8")
    assert after_text == before_text
    states = list_canaries(state_dir)
    assert states[0].run_count == 1, "idempotent rerun must not advance the run count"


def test_live_without_authorization_is_refused(tmp_path: Path) -> None:
    manifest = _live_manifest("c-canary-live")
    with pytest.raises(CanaryRefused, match="live execution is missing"):
        run_canary_offline(
            tmp_path / "state",
            "nightly",
            manifest,
            artifacts_root=tmp_path / "canaries",
            now=1000.0,
            credentials_configured=False,
        )
    # Nothing dispatched: no artifact directory appears.
    assert not list((tmp_path / "canaries").glob("*")) if (tmp_path / "canaries").exists() else True


def test_caps_match_ordinary_campaigns(tmp_path: Path) -> None:
    # A canary with a request cap below its planned work is rejected by the
    # same validation ordinary campaigns use, before anything dispatches.
    from pydantic import ValidationError

    manifest = _manifest("c-canary-caps")
    dumped = manifest.model_dump(mode="json")
    dumped["limits"]["max_requests"] = 1
    dumped["limits"]["max_concurrency"] = 1
    with pytest.raises(ValidationError, match=r"max_requests|below"):
        CampaignManifest.model_validate(dumped)
    # A valid canary stays within its caps offline.
    result = run_canary_offline(
        tmp_path / "state",
        "nightly",
        manifest,
        artifacts_root=tmp_path / "canaries",
        now=1000.0,
    )
    assert result["accepted_samples"] == 3


def test_job_id_traversal_blocked() -> None:
    for hostile in ("../evil", "/abs", "a/b", "..", ""):
        with pytest.raises(CanaryError):
            validate_job_id(hostile)
    assert validate_job_id("nightly-01") == "nightly-01"


def test_fixture_mode_runs_offline_without_network(tmp_path: Path) -> None:
    manifest = _manifest("c-canary-offline")
    artifacts = tmp_path / "direct"
    direct = run_offline(manifest, artifacts_dir=artifacts)
    via_canary = run_canary_offline(
        tmp_path / "state",
        "nightly",
        manifest,
        artifacts_root=tmp_path / "canaries",
        now=1000.0,
    )
    assert via_canary["grade_digest"] == direct.grade_digest
    assert via_canary["accepted_samples"] == direct.accepted_samples


def test_complete_partial_failed_states(tmp_path: Path) -> None:
    from stealthbench.adapters.base import FixtureBundle, RecordedExchange
    from stealthbench.schemas.campaign import Capabilities

    # Failed: one transport error among three items.
    manifest = _manifest("c-canary-failed")
    endpoint = manifest.endpoints[0].endpoint_id
    bundle = FixtureBundle(
        name="canary-failed",
        catalog_source="fixture",
        catalog={},
        exchanges=(
            RecordedExchange(
                endpoint_id=endpoint,
                benchmark_id="ifeval",
                item_id="syn-if-001",
                repeat_id=1,
                outcome="response",
                response="blueberry ok",
                usage={"input_tokens": 10, "output_tokens": 5},
                usage_reported=True,
                effective_settings={},
                finish_status="stop",
            ),
            RecordedExchange(
                endpoint_id=endpoint,
                benchmark_id="ifeval",
                item_id="syn-if-002",
                repeat_id=1,
                outcome="response",
                response="blueberry ok",
                usage={"input_tokens": 10, "output_tokens": 5},
                usage_reported=True,
                effective_settings={},
                finish_status="stop",
            ),
            RecordedExchange(
                endpoint_id=endpoint,
                benchmark_id="ifeval",
                item_id="syn-if-003",
                repeat_id=1,
                outcome="error",
                failure_kind="server_error",
                failure_detail="500 boom",
                http_status=500,
            ),
        ),
        capabilities=Capabilities(
            streaming=False,
            tool_calls=False,
            reasoning=False,
            usage_reporting=True,
            logprobs=False,
        ),
    )
    result = run_canary_offline(
        tmp_path / "state",
        "failed-job",
        manifest,
        artifacts_root=tmp_path / "canaries",
        fixture_bundle=bundle,
        now=1000.0,
    )
    assert result["transport_failed"] == 1
    assert result["accepted_samples"] == 2
