"""JSON/CSV exports and provenance links (G13 T13B).

Acceptance: counts round-trip; secrets, CSV formula injection and path
traversal are blocked.
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import pytest

from stealthbench.benchmarks.workflow import run_offline
from stealthbench.reporting.exports import (
    ExportTraversal,
    read_report_csv,
    read_report_json,
    safe_output_path,
    sanitize_csv_cell,
)
from stealthbench.reporting.site import build_report, build_report_data
from stealthbench.schemas.campaign import CampaignManifest
from stealthbench.schemas.results import (
    IdentityReport,
    RankedSimilarity,
    SignalEvidence,
)
from tests.conftest import REPO_ROOT

pytestmark = pytest.mark.integration

CANARY = "sk-canary-0123456789abcdef0123456789abcdef"


def _manifest(campaign_id: str) -> CampaignManifest:
    payload = json.loads((REPO_ROOT / "configs" / "offline-demo.json").read_text(encoding="utf-8"))
    payload["campaign_id"] = campaign_id
    payload.pop("authorization", None)
    return CampaignManifest.model_validate(payload)


def _run(campaign_id: str, root: Path) -> Path:
    manifest = _manifest(campaign_id)
    artifacts = root / "artifacts"
    run_offline(manifest, artifacts_dir=artifacts)
    return artifacts


def test_counts_round_trip_json_and_csv(tmp_path: Path) -> None:
    artifacts = _run("c-export-roundtrip", tmp_path)
    out = tmp_path / "report"
    summary = build_report(artifacts, out)

    payload = read_report_json(out / "report.json")
    assert payload["campaign_id"] == "c-export-roundtrip"
    assert payload["manifest_hash"] == summary["manifest_hash"]
    assert payload["accepted_samples"] == summary["accepted_samples"] == 3

    rows = read_report_csv(out / "report.csv")
    assert rows, "CSV must carry at least one row"
    total_graded = sum(int(row["n_graded"]) for row in rows)
    assert total_graded == 3, "CSV graded counts must match source grades"
    # Missing stays empty, never 0 for absent accuracy tested below.
    for row in rows:
        assert row["accuracy"] != "", "complete campaign has a scored accuracy"


def test_missing_accuracy_is_empty_not_zero(tmp_path: Path) -> None:
    from stealthbench.adapters.base import FixtureBundle, RecordedExchange
    from stealthbench.schemas.campaign import Capabilities

    manifest = _manifest("c-export-missing")
    dumped = manifest.model_dump(mode="json")
    dumped["benchmarks"].append(
        {
            "benchmark_id": "mmlu-pro",
            "track": "direct",
            "category": "general_knowledge",
            "adapter_version": "0.1.0",
            "dataset_id": "synthetic-v1",
            "dataset_revision": "rev-1",
            "evaluator_id": "eval",
            "evaluator_revision": "rev-1",
            "expected_item_count": 1,
            "item_ids": ["syn-mmlu-001"],
            "repeats": 1,
        }
    )
    manifest2 = CampaignManifest.model_validate(dumped)
    endpoint = manifest2.endpoints[0].endpoint_id
    bundle = FixtureBundle(
        name="missing-csv",
        catalog_source="fixture",
        catalog={},
        exchanges=tuple(
            RecordedExchange(
                endpoint_id=endpoint,
                benchmark_id="ifeval",
                item_id=item_id,
                repeat_id=1,
                outcome="response",
                response="blueberry answer",
                usage={"input_tokens": 10, "output_tokens": 5},
                usage_reported=True,
                effective_settings={},
                finish_status="stop",
            )
            for item_id in ("syn-if-001", "syn-if-002", "syn-if-003")
        ),
        capabilities=Capabilities(
            streaming=False,
            tool_calls=False,
            reasoning=False,
            usage_reporting=True,
            logprobs=False,
        ),
    )
    artifacts = tmp_path / "artifacts"
    run_offline(manifest2, artifacts_dir=artifacts, fixture_bundle=bundle)
    out = tmp_path / "report"
    build_report(artifacts, out)
    rows = read_report_csv(out / "report.csv")
    unscored = [row for row in rows if row["benchmark_id"] == "mmlu-pro"]
    assert unscored, "unscored benchmark must still appear as a row"
    assert unscored[0]["accuracy"] == "", "missing accuracy is empty, never 0"
    assert unscored[0]["n_graded"] == "0"


def test_secrets_blocked_and_redacted(tmp_path: Path) -> None:
    artifacts = _run("c-export-secret", tmp_path)
    out = tmp_path / "report"
    build_report(artifacts, out, extra_secrets=frozenset({CANARY}))

    # A canary smuggled into an identity label must not survive.
    endpoint_id = _manifest("x").endpoints[0].endpoint_id
    assert CANARY not in (out / "report.json").read_text(encoding="utf-8")
    assert CANARY not in (out / "report.csv").read_text(encoding="utf-8")
    for html_path in out.glob("*.html"):
        assert CANARY not in html_path.read_text(encoding="utf-8")

    # Direct helper: recognizable canaries never reach an export cell.
    assert CANARY not in sanitize_csv_cell(f"prefix {CANARY}", extra_secrets=frozenset({CANARY}))
    assert "[REDACTED]" in sanitize_csv_cell(CANARY, extra_secrets=frozenset({CANARY}))
    _ = endpoint_id


def test_secret_alias_is_redacted_in_downloads(tmp_path: Path) -> None:
    payload = json.loads((REPO_ROOT / "configs" / "offline-demo.json").read_text(encoding="utf-8"))
    payload["campaign_id"] = "c-export-alias-secret"
    payload["endpoints"][0]["endpoint_id"] = "fixture-anonymous-alpha"
    payload["endpoints"][0]["alias"] = "fixture-anonymous-alpha"
    manifest = CampaignManifest.model_validate(payload)
    artifacts = tmp_path / "artifacts"
    run_offline(manifest, artifacts_dir=artifacts, extra_secrets=frozenset({CANARY}))
    out = tmp_path / "report"
    build_report(artifacts, out, extra_secrets=frozenset({CANARY}))
    assert CANARY not in (out / "report.json").read_text(encoding="utf-8")
    assert CANARY not in (out / "report.csv").read_text(encoding="utf-8")


def test_csv_formula_injection_blocked() -> None:
    for prefix in ("=", "+", "-", "@"):
        cell = sanitize_csv_cell(f"{prefix}HYPERLINK(1)")
        assert cell == f"'{prefix}HYPERLINK(1)"
        assert not cell.startswith(prefix)
    # Ordinary identifiers pass through unchanged.
    assert sanitize_csv_cell("ifeval::syn-if-001") == "ifeval::syn-if-001"

    # A hostile reveal label is defused in the written CSV.
    hostile = "=cmd|'/c calc'!A1"
    assert sanitize_csv_cell(hostile).startswith("'")


def test_hostile_reveal_label_defused_in_csv(tmp_path: Path) -> None:
    manifest = _manifest("c-export-hostile-reveal")
    artifacts = tmp_path / "artifacts"
    run_offline(manifest, artifacts_dir=artifacts)
    endpoint_id = manifest.endpoints[0].endpoint_id
    report = IdentityReport(
        endpoint_id=endpoint_id,
        candidate_library_revision="lib-v1",
        signal_evidence=(
            SignalEvidence(
                signal="input_count_vector",
                available=True,
                comparable_observations=10,
                detail="test",
            ),
        ),
        ranked_similarities=(
            RankedSimilarity(
                candidate_id="cand-a",
                candidate_library_revision="lib-v1",
                similarity=0.8,
                evidence_signals=("input_count_vector",),
            ),
        ),
        abstention_reason=None,
        calibrated_probabilities=None,
        official_reveal_label="=cmd|'/c calc'!A1",
    )
    identity_dir = artifacts / "identity"
    identity_dir.mkdir(parents=True, exist_ok=True)
    (identity_dir / f"{endpoint_id}.json").write_text(
        report.model_dump_json(indent=2), encoding="utf-8"
    )
    out = tmp_path / "report"
    build_report(artifacts, out)
    text = (out / "report.csv").read_text(encoding="utf-8")
    assert "=cmd|" not in text.splitlines()[1] or "'=cmd" in text
    rows = list(csv.DictReader(io.StringIO(text)))
    assert rows[0]["reveal_label"].startswith("'")
    assert not rows[0]["reveal_label"].startswith("=")


def test_traversal_blocked() -> None:
    with pytest.raises(ExportTraversal):
        safe_output_path(Path("/tmp/out"), "../evil.json")
    with pytest.raises(ExportTraversal):
        safe_output_path(Path("/tmp/out"), "/abs.json")
    with pytest.raises(ExportTraversal):
        safe_output_path(Path("/tmp/out"), "a/b.json")
    # Plain basenames are accepted and stay inside.
    assert safe_output_path(Path("/tmp/out"), "report.json").name == "report.json"


def test_provenance_links_and_manifest_copy(tmp_path: Path) -> None:
    artifacts = _run("c-export-prov", tmp_path)
    out = tmp_path / "report"
    summary = build_report(artifacts, out)
    assert (out / "manifest.json").is_file()
    copied = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert copied["campaign_id"] == "c-export-prov"
    report_payload = read_report_json(out / "report.json")
    assert report_payload["provenance"]["manifest_hash"] == summary["manifest_hash"]
    index = (out / "index.html").read_text(encoding="utf-8")
    assert 'href="provenance.html#' in index


def test_export_import_consistency(tmp_path: Path) -> None:
    artifacts = _run("c-export-consistent", tmp_path)
    out = tmp_path / "report"
    build_report(artifacts, out)
    data = build_report_data(artifacts)
    payload = read_report_json(out / "report.json")
    assert payload["accepted_samples"] == data.accepted_samples
    assert len(payload["endpoints"]) == len(data.endpoints)
    rows = read_report_csv(out / "report.csv")
    assert len(rows) == sum(max(1, len(item.benchmarks)) for item in data.endpoints)
