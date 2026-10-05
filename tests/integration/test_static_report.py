"""Static model pages, comparisons and history (G13 T13A).

Acceptance: complete, incomplete and unknown results render accurately with
escaped content, provenance and observation windows.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from stealthbench.benchmarks.workflow import run_offline
from stealthbench.cli import EXIT_OK, main
from stealthbench.reporting.site import build_report, build_report_data, safe_slug, serve_report
from stealthbench.schemas.campaign import CampaignManifest
from stealthbench.schemas.results import (
    IdentityReport,
    RankedSimilarity,
    SignalEvidence,
)
from tests.conftest import REPO_ROOT

pytestmark = pytest.mark.integration

CANARY = "sk-canary-0123456789abcdef0123456789abcdef"


def _base_manifest(campaign_id: str) -> CampaignManifest:
    payload = json.loads((REPO_ROOT / "configs" / "offline-demo.json").read_text(encoding="utf-8"))
    payload["campaign_id"] = campaign_id
    payload.pop("authorization", None)
    return CampaignManifest.model_validate(payload)


def _manifest_with_alias(campaign_id: str, alias: str, endpoint_id: str) -> CampaignManifest:
    manifest = _base_manifest(campaign_id)
    dumped = manifest.model_dump(mode="json")
    dumped["endpoints"][0]["alias"] = alias
    dumped["endpoints"][0]["endpoint_id"] = endpoint_id
    return CampaignManifest.model_validate(dumped)


def _two_endpoint_manifest(campaign_id: str) -> CampaignManifest:
    manifest = _base_manifest(campaign_id)
    dumped = manifest.model_dump(mode="json")
    second = dict(dumped["endpoints"][0])
    second["endpoint_id"] = "fixture-anonymous-beta"
    second["alias"] = "fixture-anonymous-beta"
    dumped["endpoints"] = [dumped["endpoints"][0], second]
    return CampaignManifest.model_validate(dumped)


def _write_identity(artifacts: Path, endpoint_id: str, report: IdentityReport) -> None:
    target_dir = artifacts / "identity"
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / f"{endpoint_id}.json").write_text(
        report.model_dump_json(indent=2), encoding="utf-8"
    )


def _known_identity(
    endpoint_id: str, predicted: str = "cand-a", reveal: str | None = "cand-b"
) -> IdentityReport:
    return IdentityReport(
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
                candidate_id=predicted,
                candidate_library_revision="lib-v1",
                similarity=0.9,
                evidence_signals=("input_count_vector",),
            ),
            RankedSimilarity(
                candidate_id="cand-other",
                candidate_library_revision="lib-v1",
                similarity=0.4,
                evidence_signals=("input_count_vector",),
            ),
        ),
        abstention_reason=None,
        calibrated_probabilities=None,
        official_reveal_label=reveal,
    )


def test_complete_campaign_renders_with_provenance(tmp_path: Path) -> None:
    manifest = _base_manifest("c-report-complete")
    artifacts = tmp_path / "artifacts"
    run_offline(manifest, artifacts_dir=artifacts)
    out = tmp_path / "report"
    summary = build_report(artifacts, out)

    assert summary["status"] == "ok"
    assert summary["campaign_id"] == "c-report-complete"
    assert summary["accepted_samples"] == 3
    for name in ("index.html", "comparison.html", "history.html", "provenance.html"):
        assert (out / name).is_file(), f"missing {name}"
    endpoint_pages = list(out.glob("endpoint-*.html"))
    assert len(endpoint_pages) == 1

    index = (out / "index.html").read_text(encoding="utf-8")
    assert "fixture-anonymous-alpha" in index
    assert 'href="provenance.html#' in index, "every score must link provenance"
    assert "Observation window" in index
    assert "95% CI" in index or "CI" in index
    assert "unknown" in index, "no identity file means unknown"

    page = endpoint_pages[0].read_text(encoding="utf-8")
    for heading in (
        "Observation dates",
        "Sample sizes",
        "Costs",
        "Missing coverage",
        "Identity resolution",
        "Predicted identity",
        "Revealed identity",
    ):
        assert heading in page, f"model page missing {heading}"
    assert 'href="provenance.html#' in page

    provenance = (out / "provenance.html").read_text(encoding="utf-8")
    assert summary["manifest_hash"] in provenance
    assert str(summary["grade_digest"]) in provenance


def test_partial_coverage_renders_without_headline(tmp_path: Path) -> None:
    manifest = _base_manifest("c-report-partial")
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
            "expected_item_count": 2,
            "item_ids": ["syn-mmlu-001", "syn-mmlu-002"],
            "repeats": 1,
        }
    )
    manifest2 = CampaignManifest.model_validate(dumped)
    artifacts = tmp_path / "artifacts"
    # Default bundle only answers declared items; the second benchmark has no
    # recorded exchanges for its items, so it stays unscored (partial).
    from stealthbench.adapters.base import FixtureBundle, RecordedExchange
    from stealthbench.schemas.campaign import Capabilities

    endpoint = manifest2.endpoints[0].endpoint_id
    bundle = FixtureBundle(
        name="partial",
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
    run_offline(manifest2, artifacts_dir=artifacts, fixture_bundle=bundle)
    out = tmp_path / "report"
    data = build_report_data(artifacts)
    assert len(data.endpoints) == 1
    endpoint = data.endpoints[0]
    assert endpoint.coverage == "partial"
    assert endpoint.n_missing + endpoint.n_transport_failed > 0
    unscored = [cell for cell in endpoint.benchmarks if cell.benchmark_id == "mmlu-pro"]
    assert unscored and unscored[0].accuracy is None
    assert unscored[0].n_missing > 0
    build_report(artifacts, out)
    page = next((out).glob("endpoint-*.html")).read_text(encoding="utf-8")
    assert "partial" in page
    assert "n/a" in page, "unscored benchmark must show n/a, never 0"


def test_failed_transport_denominators_explicit(tmp_path: Path) -> None:
    from stealthbench.adapters.base import FixtureBundle, RecordedExchange
    from stealthbench.schemas.campaign import Capabilities

    manifest = _base_manifest("c-report-failed")
    endpoint = manifest.endpoints[0].endpoint_id
    bundle = FixtureBundle(
        name="one-failure",
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
                response="no keyword here",
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
    artifacts = tmp_path / "artifacts"
    run_offline(manifest, artifacts_dir=artifacts, fixture_bundle=bundle)
    out = tmp_path / "report"
    data = build_report_data(artifacts)
    endpoint_summary = data.endpoints[0]
    assert endpoint_summary.n_transport_failed == 1
    assert endpoint_summary.n_graded == 2
    build_report(artifacts, out)
    page = next(out.glob("endpoint-*.html")).read_text(encoding="utf-8")
    assert "transport failures" in page
    assert "1" in page


def test_tied_or_uncertain_rankings(tmp_path: Path) -> None:
    manifest = _two_endpoint_manifest("c-report-tied")
    artifacts = tmp_path / "artifacts"
    run_offline(manifest, artifacts_dir=artifacts)
    out = tmp_path / "report"
    data = build_report_data(artifacts)
    assert len(data.endpoints) == 2
    assert data.endpoints[0].accuracy == pytest.approx(data.endpoints[1].accuracy)
    assert data.comparisons, "two endpoints must produce a comparison"
    comparison = data.comparisons[0]
    assert comparison.verdict in {"tied_or_uncertain", "unavailable"}
    build_report(artifacts, out)
    text = (out / "comparison.html").read_text(encoding="utf-8")
    assert "tied_or_uncertain" in text or "uncertain" in text


def test_unknown_identity_has_no_probability(tmp_path: Path) -> None:
    manifest = _base_manifest("c-report-unknown-id")
    artifacts = tmp_path / "artifacts"
    run_offline(manifest, artifacts_dir=artifacts)
    out = tmp_path / "report"
    build_report(artifacts, out)
    page = next(out.glob("endpoint-*.html")).read_text(encoding="utf-8")
    assert "unknown" in page
    assert "insufficient_evidence" in page
    assert "probability" not in page.lower() or "not probability" in page.lower()


def test_stale_evidence_flagged(tmp_path: Path) -> None:
    manifest = _base_manifest("c-report-stale")
    dumped = manifest.model_dump(mode="json")
    dumped["observation_window"] = {
        "started_at": "2020-01-01T00:00:00+00:00",
        "ended_at": "2020-01-02T00:00:00+00:00",
        "timezone": "UTC",
    }
    stale_manifest = CampaignManifest.model_validate(dumped)
    artifacts = tmp_path / "artifacts"
    run_offline(stale_manifest, artifacts_dir=artifacts)
    data = build_report_data(artifacts)
    assert data.stale_campaign is True
    out = tmp_path / "report"
    build_report(artifacts, out)
    assert "Stale evidence" in (out / "index.html").read_text(encoding="utf-8")
    assert "stale" in (out / "history.html").read_text(encoding="utf-8").lower()


def test_prediction_and_reveal_are_separate(tmp_path: Path) -> None:
    manifest = _base_manifest("c-report-pre-reveal")
    artifacts = tmp_path / "artifacts"
    run_offline(manifest, artifacts_dir=artifacts)
    endpoint_id = manifest.endpoints[0].endpoint_id
    _write_identity(artifacts, endpoint_id, _known_identity(endpoint_id))
    out = tmp_path / "report"
    build_report(artifacts, out)
    page = next(out.glob("endpoint-*.html")).read_text(encoding="utf-8")
    assert "Predicted identity" in page
    assert "Revealed identity" in page
    assert "cand-a" in page, "predicted ranking must appear"
    assert "cand-b" in page, "reveal label must appear separately"
    # Sections are distinct headings, not a merged claim.
    assert page.index("Predicted identity") < page.index("Revealed identity")


def test_script_bearing_identity_content_escaped(tmp_path: Path) -> None:
    manifest = _base_manifest("c-report-xss")
    artifacts = tmp_path / "artifacts"
    run_offline(manifest, artifacts_dir=artifacts)
    endpoint_id = manifest.endpoints[0].endpoint_id
    hostile = "<script>alert(1)</script>"
    _write_identity(artifacts, endpoint_id, _known_identity(endpoint_id, predicted=hostile))
    out = tmp_path / "report"
    build_report(artifacts, out)
    for html_path in out.glob("*.html"):
        text = html_path.read_text(encoding="utf-8")
        assert hostile not in text, f"raw script survived in {html_path.name}"
    page = next(out.glob("endpoint-*.html")).read_text(encoding="utf-8")
    assert "&lt;script&gt;" in page


def test_traversal_helpers_block_escape(tmp_path: Path) -> None:
    from stealthbench.reporting.exports import ExportTraversal, safe_output_path

    assert safe_slug("../../etc/passwd") == "etc_passwd"
    assert "/" not in safe_slug("a/b")
    with pytest.raises(ExportTraversal):
        safe_output_path(tmp_path, "../evil.html")
    with pytest.raises(ExportTraversal):
        safe_output_path(tmp_path, "/abs.html")
    with pytest.raises(ExportTraversal):
        safe_output_path(tmp_path, "sub/dir.html")

    # A normal build writes everything inside the output directory.
    manifest = _base_manifest("c-report-traversal")
    artifacts = tmp_path / "artifacts"
    run_offline(manifest, artifacts_dir=artifacts)
    out = tmp_path / "report"
    summary = build_report(artifacts, out)
    for name in summary["files"]:
        assert ".." not in name
        assert (out / name).resolve().parent == out.resolve()


def test_every_score_links_provenance(tmp_path: Path) -> None:
    manifest = _base_manifest("c-report-links")
    artifacts = tmp_path / "artifacts"
    run_offline(manifest, artifacts_dir=artifacts)
    out = tmp_path / "report"
    build_report(artifacts, out)
    for html_path in (out / "index.html", next(out.glob("endpoint-*.html"))):
        text = Path(html_path).read_text(encoding="utf-8")
        assert 'href="provenance.html#' in text


def test_cli_report_builds_offline_exit_zero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest = _base_manifest("c-report-cli")
    artifacts = tmp_path / "artifacts"
    run_offline(manifest, artifacts_dir=artifacts)
    out = tmp_path / "report"
    assert main(["report", str(artifacts), "--output", str(out)]) == EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ok"
    assert payload["campaign_id"] == "c-report-cli"
    assert (out / "index.html").is_file()
    assert (out / "report.json").is_file()

    # run/replay behavior intact: offline run and replay still exit 0.
    manifest_path = tmp_path / "campaign.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    assert main(["replay", str(artifacts)]) == EXIT_OK
    capsys.readouterr()


def test_serve_report_refuses_non_loopback(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="non-loopback"):
        serve_report(tmp_path, host="0.0.0.0")
