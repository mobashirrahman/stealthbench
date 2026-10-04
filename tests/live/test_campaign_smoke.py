"""Capped live setup and pilot campaign smoke (G14 T14C).

Authorized live execution only. Without
``STEALTHBENCH_LIVE_AUTHORIZATION=1`` every test skips as
``blocked_external`` — it never fake-passes. No specific model result is
required to pass an infrastructure check: these tests assert declared
profile shape, preflight refusal reasons, and qualified coverage, never an
accuracy number.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from stealthbench.runner.preflight import AUTHORIZATION_ENV, run_preflight
from stealthbench.schemas.campaign import CampaignManifest

pytestmark = pytest.mark.live

REPO_ROOT = Path(__file__).resolve().parents[2]
CREDENTIAL_ENV = "STEALTHBENCH_ZEN_API_KEY"


def _require_authorization() -> None:
    if os.environ.get(AUTHORIZATION_ENV) != "1":
        pytest.skip(f"blocked_external: {AUTHORIZATION_ENV} is not set to 1")


def test_setup_profile_declares_capped_measurement_first() -> None:
    """Setup runs before the pilot commits budget: 80 direct items plus probes."""
    _require_authorization()
    pilot = json.loads((REPO_ROOT / "configs" / "pilot.json").read_text(encoding="utf-8"))
    setup = pilot["setup_profile"]
    assert setup["direct_item_count"] == 80
    assert setup["purpose"], "the setup purpose must say what is measured, not scored"
    probes = pilot["signature_probes"]
    assert probes["probe_count"] == 60
    assert probes["repeats"] == 3
    assert probes["separate_from_benchmarks"] is True


def test_pilot_declares_600_item_core_and_50_task_agents() -> None:
    """Pilot shape is exact; counts are declared targets, not measured results."""
    _require_authorization()
    manifest = CampaignManifest.model_validate_json(
        (REPO_ROOT / "configs" / "pilot.json").read_text(encoding="utf-8")
    )
    direct = sum(s.planned_item_count or 0 for s in manifest.benchmarks if s.track == "direct")
    agent = sum(s.planned_item_count or 0 for s in manifest.benchmarks if s.track == "agent")
    assert direct == 600
    assert agent == 50
    assert manifest.limits.missingness_threshold is not None, (
        "missingness thresholds are frozen before dispatch, not fitted after"
    )


def test_pilot_preflight_reports_blockers_without_dispatch() -> None:
    """Authorized but unmaterialized: preflight refuses, names blockers, spends nothing."""
    _require_authorization()
    manifest = CampaignManifest.model_validate_json(
        (REPO_ROOT / "configs" / "pilot.json").read_text(encoding="utf-8")
    )
    assert manifest.materialized is False
    report = run_preflight(manifest)
    assert report.passed is False, (
        "an unmaterialized pilot without endpoints, revisions or a spending cap "
        "must not pass preflight"
    )
    assert report.blockers, "a refusal must say what is missing, not just that it is missing"
    assert any("materialization" in blocker for blocker in report.blockers)
    assert any("revisions" in blocker for blocker in report.blockers)
    # Infrastructure outcome only: measured cost and coverage stay unclaimed
    # until an authorized, materialized dispatch produces them.
    assert "accuracy" not in json.dumps(report.to_dict()), (
        "preflight must never emit an accuracy claim"
    )


def test_live_smoke_needs_credentials_and_cap_when_authorized() -> None:
    """Authorization alone is not readiness: missing inputs stay blocked_external."""
    _require_authorization()
    missing: list[str] = []
    if not os.environ.get(CREDENTIAL_ENV):
        missing.append(f"{CREDENTIAL_ENV} is not present in the environment")
    if not os.environ.get("STEALTHBENCH_SPENDING_CAP_USD"):
        missing.append("STEALTHBENCH_SPENDING_CAP_USD is not set")
    if missing:
        pytest.skip(f"blocked_external: {'; '.join(missing)}")
