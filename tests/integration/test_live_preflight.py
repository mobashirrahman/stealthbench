"""Live readiness preflight (G14 T14B).

Mandatory checks before live dispatch: credentials, authorization plus cap,
revisions, endpoint modes, runtime isolation, and price snapshots plus caps.
A planning request alone never authorizes: the manifest's spending ceiling is
not approval, and these tests prove the outside-the-manifest flag is required.

All offline: no socket, no dispatch, no spend. Container expectations use the
real probe and never fake-pass.
"""

from __future__ import annotations

from typing import Any

import pytest

from stealthbench.runner.preflight import (
    AUTHORIZATION_ENV,
    live_authorization_granted,
    run_preflight,
)
from stealthbench.sandbox.runtime import container_runtime
from stealthbench.schemas.campaign import CampaignManifest

pytestmark = pytest.mark.integration

CREDENTIAL_NAME = "STEALTHBENCH_TEST_KEY"


def _live_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "campaign_id": "preflight-probe",
        "title": "Preflight probe campaign",
        "description": "Synthetic live-readiness probe with no public outcome.",
        "mode": "live_authorized",
        "materialized": True,
        "score_version": "probe-v1",
        "seed": 20261004,
        "observation_window": {"started_at": None, "ended_at": None, "timezone": "UTC"},
        "endpoints": [
            {
                "endpoint_id": "probe-endpoint-1",
                "alias": "probe-endpoint-1",
                "route": "https://example.invalid/chat",
                "transport": "zen",
                "capabilities": {
                    "streaming": True,
                    "tool_calls": False,
                    "reasoning": False,
                    "usage_reporting": True,
                    "logprobs": False,
                },
                "pricing": {
                    "currency": "USD",
                    "input_per_mtok": 2.0,
                    "output_per_mtok": 8.0,
                    "cached_input_per_mtok": None,
                    "reasoning_per_mtok": None,
                    "snapshot_id": "snapshot-1",
                },
                "credential_ref": CREDENTIAL_NAME,
            }
        ],
        "benchmarks": [
            {
                "benchmark_id": "ifeval",
                "track": "direct",
                "category": "instruction_following",
                "dataset_id": "google/IFEval",
                "dataset_revision": "sha966cd89545d6b6acfd7638bc708b98261ca58e84",
                "evaluator_id": "instruction_following_eval",
                "evaluator_revision": "e49bbfe381c9c0e564b937f1c4e163a2273c65cc",
                "expected_item_count": 1,
                "item_ids": ["syn-if-001"],
                "repeats": 1,
                "core_category": True,
            }
        ],
        "generation": {
            "max_output_tokens": 512,
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": 20261004,
            "stream": False,
            "unsupported_settings_are_errors": True,
        },
        "retry_policy": {
            "max_attempts": 3,
            "initial_backoff_seconds": 1.0,
            "multiplier": 2.0,
            "max_backoff_seconds": 30.0,
            "retryable_statuses": [408, 429, 500, 502, 503, 504],
        },
        "limits": {
            "max_requests": 100,
            "max_concurrency": 2,
            "max_input_tokens": 100000,
            "max_output_tokens": 100000,
            "max_total_cost_usd": 10.0,
            "max_wall_seconds": 900,
            "require_cost_bounds": True,
            "missingness_threshold": 0.05,
        },
        "authorization": {
            "required": True,
            "spending_cap_usd": 25.0,
            "authorized_by": "operator-test",
        },
        "budget_scope": "campaign",
    }
    payload.update(overrides)
    return payload


def _manifest(**overrides: Any) -> CampaignManifest:
    return CampaignManifest.model_validate(_live_payload(**overrides))


def _env(**extra: str) -> dict[str, str]:
    return {CREDENTIAL_NAME: "present", **extra}


def test_planning_request_alone_never_authorizes() -> None:
    """A complete manifest without the operator flag still refuses dispatch."""
    manifest = _manifest()
    report = run_preflight(manifest, env=_env(), live_authorization=False)
    assert report.passed is False
    assert any(
        "planning request alone does not authorize" in blocker for blocker in report.blockers
    )
    assert live_authorization_granted({AUTHORIZATION_ENV: "1"}) is True
    assert live_authorization_granted({}) is False


def test_fully_configured_manifest_passes() -> None:
    manifest = _manifest()
    report = run_preflight(manifest, env=_env(), live_authorization=True, require_container=False)
    assert report.passed is True
    assert report.blockers == ()
    assert all(check.passed for check in report.checks)
    assert report.manifest_hash
    payload = report.to_dict()
    assert payload["passed"] is True
    assert payload["campaign_id"] == "preflight-probe"


def test_missing_credential_blocks() -> None:
    manifest = _manifest()
    report = run_preflight(manifest, env={}, live_authorization=True, require_container=False)
    assert report.passed is False
    assert any("credentials" in blocker for blocker in report.blockers)
    assert any(CREDENTIAL_NAME in blocker for blocker in report.blockers)


def test_endpoint_without_credential_ref_blocks() -> None:
    payload = _live_payload()
    payload["endpoints"][0]["credential_ref"] = None
    manifest = CampaignManifest.model_validate(payload)
    report = run_preflight(manifest, env=_env(), live_authorization=True, require_container=False)
    assert report.passed is False
    assert any("credential_ref" in blocker for blocker in report.blockers)


def test_missing_spending_cap_blocks() -> None:
    payload = _live_payload()
    payload["authorization"] = {"required": True, "spending_cap_usd": None}
    manifest = CampaignManifest.model_validate(payload)
    report = run_preflight(manifest, env=_env(), live_authorization=True, require_container=False)
    assert report.passed is False
    assert any("spending cap" in blocker for blocker in report.blockers)


def test_operator_cap_must_cover_campaign_ceiling() -> None:
    payload = _live_payload()
    payload["authorization"] = {
        "required": True,
        "spending_cap_usd": 1.0,
        "authorized_by": "operator-test",
    }
    manifest = CampaignManifest.model_validate(payload)
    report = run_preflight(manifest, env=_env(), live_authorization=True, require_container=False)
    assert report.passed is False
    assert any("below" in blocker for blocker in report.blockers)


def test_null_revisions_block() -> None:
    payload = _live_payload()
    payload["benchmarks"][0]["dataset_revision"] = None
    payload["benchmarks"][0]["evaluator_revision"] = None
    manifest = CampaignManifest.model_validate(payload)
    report = run_preflight(manifest, env=_env(), live_authorization=True, require_container=False)
    assert report.passed is False
    assert any("revisions" in blocker for blocker in report.blockers)


def test_fixture_transport_refused_for_live() -> None:
    payload = _live_payload()
    payload["endpoints"][0]["transport"] = "fixture"
    payload["endpoints"][0]["credential_ref"] = None
    manifest = CampaignManifest.model_validate(payload)
    report = run_preflight(manifest, env=_env(), live_authorization=True, require_container=False)
    assert report.passed is False
    assert any("endpoint_modes" in blocker for blocker in report.blockers)


def test_unmaterialized_profile_blocks() -> None:
    payload = _live_payload()
    payload["materialized"] = False
    payload["benchmarks"][0]["item_ids"] = []
    del payload["benchmarks"][0]["expected_item_count"]
    payload["benchmarks"][0]["declared_item_count"] = 1
    manifest = CampaignManifest.model_validate(payload)
    report = run_preflight(manifest, env=_env(), live_authorization=True, require_container=False)
    assert report.passed is False
    assert any("materialization" in blocker for blocker in report.blockers)


def test_unknown_price_blocks_spending_capped_execution() -> None:
    payload = _live_payload()
    payload["endpoints"][0]["pricing"] = {
        "currency": "USD",
        "input_per_mtok": None,
        "output_per_mtok": None,
        "cached_input_per_mtok": None,
        "reasoning_per_mtok": None,
        "snapshot_id": None,
    }
    manifest = CampaignManifest.model_validate(payload)
    report = run_preflight(manifest, env=_env(), live_authorization=True, require_container=False)
    assert report.passed is False
    assert any("pricing" in blocker for blocker in report.blockers)


def test_unfrozen_missingness_threshold_blocks() -> None:
    payload = _live_payload()
    payload["limits"]["missingness_threshold"] = None
    manifest = CampaignManifest.model_validate(payload)
    report = run_preflight(manifest, env=_env(), live_authorization=True, require_container=False)
    assert report.passed is False
    assert any("missingness_threshold" in blocker for blocker in report.blockers)


def test_offline_campaign_never_passes_live_preflight() -> None:
    import json

    from tests.conftest import REPO_ROOT

    payload = json.loads((REPO_ROOT / "configs" / "offline-demo.json").read_text(encoding="utf-8"))
    manifest = CampaignManifest.model_validate(payload)
    report = run_preflight(manifest, env=_env(), live_authorization=True)
    assert report.passed is False
    assert any("authorization" in blocker for blocker in report.blockers)


def test_runtime_probe_reports_blocked_external_honestly() -> None:
    """The real container probe decides; absence never fake-passes."""
    manifest = _manifest()
    report = run_preflight(manifest, env=_env(), live_authorization=True, require_container=True)
    runtime_checks = [check for check in report.checks if check.name == "runtime"]
    assert len(runtime_checks) == 1
    if container_runtime() is None:
        assert runtime_checks[0].passed is False
        assert any(
            "blocked_external" in blocker and "runtime" in blocker for blocker in report.blockers
        )
    else:
        assert runtime_checks[0].passed is True
        assert runtime_checks[0].detail is not None
