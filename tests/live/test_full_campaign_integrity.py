"""Full-suite execution and final campaign report integrity (G14 T14D).

Authorized live execution only. Without
``STEALTHBENCH_LIVE_AUTHORIZATION=1`` every test skips as
``blocked_external`` — it never fake-passes. The full gate requires whole
declared splits under a new immutable campaign; partial coverage and
unvalidated attribution stay visibly qualified. No accuracy number appears
here: integrity is about provenance and qualification, not outcomes.
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

EXPECTED_FULL_SPLIT_SIZES = {
    "livecodebench": 1055,
    "evalplus": 164,
    "ifeval": 541,
    "mmlu_pro": 12032,
    "math500": 500,
    "swebench": 500,
    "terminalbench": 66,
}


def _require_authorization() -> None:
    if os.environ.get(AUTHORIZATION_ENV) != "1":
        pytest.skip(f"blocked_external: {AUTHORIZATION_ENV} is not set to 1")


def test_full_profile_requires_whole_splits() -> None:
    """Every benchmark demands its full declared split, not a pilot subset."""
    _require_authorization()
    manifest = CampaignManifest.model_validate_json(
        (REPO_ROOT / "configs" / "full.json").read_text(encoding="utf-8")
    )
    assert manifest.materialized is False
    assert all(spec.require_full_split for spec in manifest.benchmarks)
    by_id = {spec.benchmark_id: spec for spec in manifest.benchmarks}
    for benchmark_id, expected in EXPECTED_FULL_SPLIT_SIZES.items():
        assert by_id[benchmark_id].expected_item_count == expected
    for benchmark_id in ("bfcl", "ruler"):
        assert by_id[benchmark_id].split_enumeration, (
            f"{benchmark_id} must document its full-split enumeration"
        )
        assert by_id[benchmark_id].planned_item_count is None, (
            "an unenumerated split is unknown, never zero"
        )


def test_full_campaign_uses_a_new_immutable_campaign() -> None:
    """Full runs under a new campaign id; the pilot report is preserved, not relabelled."""
    _require_authorization()
    full = json.loads((REPO_ROOT / "configs" / "full.json").read_text(encoding="utf-8"))
    pilot = json.loads((REPO_ROOT / "configs" / "pilot.json").read_text(encoding="utf-8"))
    assert full["campaign_id"] != pilot["campaign_id"]
    assert full["campaign_id"] == "full"
    assert full["score_version"] != pilot["score_version"]
    assert full["comparability_rule"], (
        "external comparisons require equivalent revision, prompt, settings, "
        "scaffold and grading; the rule must be recorded"
    )


def test_partial_coverage_stays_qualified_not_full() -> None:
    """Unknown split sizes mean no complete total: partial stays partial."""
    _require_authorization()
    manifest = CampaignManifest.model_validate_json(
        (REPO_ROOT / "configs" / "full.json").read_text(encoding="utf-8")
    )
    assert manifest.total_planned_requests is None
    assert manifest.known_planned_requests == sum(EXPECTED_FULL_SPLIT_SIZES.values())
    assert manifest.benchmarks_with_unknown_size == ("bfcl", "ruler")
    report = run_preflight(manifest)
    assert report.passed is False
    assert any("materialization" in blocker for blocker in report.blockers), (
        "an unmaterialized full profile must be refused as unmaterialized, "
        "never relabelled as a complete run"
    )


def test_unvalidated_attribution_stays_disabled() -> None:
    """Family probabilities require the G12 evidence gate; similarity is not probability."""
    _require_authorization()
    contracts = (REPO_ROOT / "docs" / "contracts.md").read_text(encoding="utf-8")
    assert "calibrated_probabilities" in contracts
    assert "unless the G12 evidence gate passed" in contracts
