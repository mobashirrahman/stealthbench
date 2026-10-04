"""Vertical offline workflow: run / grade / replay / export (task T05C).

Gate G05: a complete fixture-only campaign grades stored answers without
regeneration and exposes failure denominators.

Coverage:

* Complete offline run on a synthetic 3-item fixture (2 accepted, 1 transport
  failure), graded via the IFEval wrapper, with redacted JSONL exports.
* Regrade without regeneration: replay reconstructs identical grades with the
  fixture transport disabled.
* Failure denominators explicit: transport failures never enter accuracy.
* Resume de-dup: a second run into the same store creates no second accepted
  sample for any key.
* Dry run generates nothing.
* CLI: ``run --offline`` exits 0 with artifacts; ``replay`` rebuilds offline.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from stealthbench.adapters.base import FixtureBundle, RecordedExchange
from stealthbench.benchmarks.workflow import (
    ReplaySummary,
    replay_offline,
    run_offline,
    synthetic_items_for_manifest,
)
from stealthbench.cli import EXIT_OK, main
from stealthbench.schemas.campaign import CampaignManifest
from stealthbench.storage.replay import ReplayReader
from tests.conftest import REPO_ROOT

pytestmark = pytest.mark.integration

CANARY = "sk-canary-0123456789abcdef0123456789abcdef"


def _manifest(campaign_id: str) -> CampaignManifest:
    payload = json.loads((REPO_ROOT / "configs" / "offline-demo.json").read_text(encoding="utf-8"))
    payload["campaign_id"] = campaign_id
    payload.pop("authorization", None)
    return CampaignManifest.model_validate(payload)


def _three_item_bundle(manifest: CampaignManifest) -> FixtureBundle:
    from stealthbench.schemas.campaign import Capabilities

    endpoint = manifest.endpoints[0].endpoint_id
    return FixtureBundle(
        name="synthetic-3-item",
        catalog_source="fixture",
        catalog={},
        exchanges=(
            RecordedExchange(
                endpoint_id=endpoint,
                benchmark_id="ifeval",
                item_id="syn-if-001",
                repeat_id=1,
                outcome="response",
                response=f"Compliant answer containing blueberry {CANARY}",
                usage={"input_tokens": 20, "output_tokens": 12},
                usage_reported=True,
                effective_settings={"temperature": 0.0, "top_p": 1.0},
                finish_status="stop",
            ),
            RecordedExchange(
                endpoint_id=endpoint,
                benchmark_id="ifeval",
                item_id="syn-if-002",
                repeat_id=1,
                outcome="response",
                response="A noncompliant answer with no keyword.",
                usage={"input_tokens": 20, "output_tokens": 12},
                usage_reported=True,
                effective_settings={"temperature": 0.0, "top_p": 1.0},
                finish_status="stop",
            ),
            RecordedExchange(
                endpoint_id=endpoint,
                benchmark_id="ifeval",
                item_id="syn-if-003",
                repeat_id=1,
                outcome="error",
                failure_kind="rate_limit",
                failure_detail="429 rate limited",
                http_status=429,
                retry_after_seconds=1.0,
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


def test_complete_offline_workflow_grades_and_exports(tmp_path: Path) -> None:
    manifest = _manifest("c-vertical-001")
    bundle = _three_item_bundle(manifest)
    root = tmp_path / "artifacts"

    report = run_offline(manifest, artifacts_dir=root, fixture_bundle=bundle)

    assert report.planned == 3
    assert report.eligible == 3
    assert report.accepted_samples == 2, "two responses accepted, one transport failure"
    assert report.transport_failed == 1
    assert report.strict_eligible == 2, "transport failures stay out of accuracy"
    assert report.strict_correct == 1
    assert report.strict_accuracy == pytest.approx(0.5)
    assert report.loose_eligible == 2
    assert (root / "events.jsonl").exists()
    assert (root / "export.jsonl").exists()
    assert (root / "grades.jsonl").exists()
    assert (root / "summary.json").exists()
    assert (root / "manifest.json").exists()

    # One accepted sample per key: distinct keys, no duplicates.
    replayed = ReplayReader(root).read(strict=False)
    keys = {r.sample_key.as_tuple() for r in replayed.results if r.is_accepted_sample}
    assert len(keys) == 2

    # Redaction: the canary in the stored response never reaches the export.
    export_text = (root / "export.jsonl").read_text(encoding="utf-8")
    assert CANARY not in export_text
    assert "[REDACTED]" in export_text
    grades_text = (root / "grades.jsonl").read_text(encoding="utf-8")
    assert CANARY not in grades_text

    # Caps: the zero-price fixture ledger stays within the declared ceiling.
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    assert summary["accepted_samples"] == 2
    assert summary["transport_failed"] == 1


def test_retryable_transport_retries_but_wrong_answers_do_not(tmp_path: Path) -> None:
    """Only retryable transport failures retry; graded answers never do."""
    manifest = _manifest("c-vertical-retry")
    bundle = _three_item_bundle(manifest)
    root = tmp_path / "artifacts"

    report = run_offline(manifest, artifacts_dir=root, fixture_bundle=bundle)

    # 001 and 002 accepted on first attempt (1 call each); 003 is a retryable
    # 429 retried to the policy limit of 3. A wrong answer is never retried.
    assert report.dispatched == 5, f"expected 1+1+3 dispatches, got {report.dispatched}"
    assert report.accepted_samples == 2


def test_non_retryable_transport_does_not_retry(tmp_path: Path) -> None:
    from stealthbench.schemas.campaign import Capabilities

    manifest = _manifest("c-vertical-noretry")
    endpoint = manifest.endpoints[0].endpoint_id
    bundle = FixtureBundle(
        name="noretry",
        catalog_source="fixture",
        catalog={},
        exchanges=(
            RecordedExchange(
                endpoint_id=endpoint,
                benchmark_id="ifeval",
                item_id="syn-if-001",
                repeat_id=1,
                outcome="response",
                response="blueberry answer",
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
                response="blueberry answer",
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
                failure_kind="authentication",
                failure_detail="401 bad key",
                http_status=401,
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
    root = tmp_path / "artifacts"
    report = run_offline(manifest, artifacts_dir=root, fixture_bundle=bundle)
    assert report.dispatched == 3, "a 401 must not be retried"
    assert report.transport_failed == 1


def test_regrade_without_regeneration_is_identical(tmp_path: Path) -> None:
    manifest = _manifest("c-vertical-regrade")
    bundle = _three_item_bundle(manifest)
    root = tmp_path / "artifacts"
    first = run_offline(manifest, artifacts_dir=root, fixture_bundle=bundle)

    # Disable the transport entirely: replay must not need it.
    import stealthbench.adapters.base as base

    def _forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("replay contacted the fixture transport")

    original = base.FixtureTransport.complete
    base.FixtureTransport.complete = _forbidden  # type: ignore[method-assign]
    try:
        second: ReplaySummary = replay_offline(root)
    finally:
        base.FixtureTransport.complete = original  # type: ignore[method-assign]

    assert second.accepted_samples == first.accepted_samples == 2
    assert second.grade_digest == first.grade_digest
    assert second.strict_accuracy == first.strict_accuracy == pytest.approx(0.5)


def test_failure_denominators_are_explicit(tmp_path: Path) -> None:
    manifest = _manifest("c-vertical-denoms")
    bundle = _three_item_bundle(manifest)
    root = tmp_path / "artifacts"
    report = run_offline(manifest, artifacts_dir=root, fixture_bundle=bundle)

    assert report.transport_failed == 1
    assert report.strict_eligible == 2
    # Accuracy is over graded responses only, not over all planned items.
    assert report.strict_accuracy == pytest.approx(0.5)
    assert report.strict_accuracy != pytest.approx(1 / 3)

    grades = [
        json.loads(line)
        for line in (root / "grades.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(grades) == 3, "grades cover accepted plus transport-failed samples"
    failed = [g for g in grades if g["transport"] != "pass"]
    assert len(failed) == 1
    assert failed[0]["strict_correctness"] == "unavailable"


def test_resume_does_not_duplicate_accepted_samples(tmp_path: Path) -> None:
    manifest = _manifest("c-vertical-resume")
    bundle = _three_item_bundle(manifest)
    root = tmp_path / "artifacts"

    first = run_offline(manifest, artifacts_dir=root, fixture_bundle=bundle)
    assert first.accepted_samples == 2
    events_before = (root / "events.jsonl").read_text(encoding="utf-8").count("\n")

    second = run_offline(manifest, artifacts_dir=root, fixture_bundle=bundle)

    # The failed sample is retried; accepted samples are skipped, never doubled.
    assert second.skipped == 2
    combined = ReplayReader(root).read(strict=False)
    keys = [r.sample_key.as_tuple() for r in combined.results if r.is_accepted_sample]
    assert len(keys) == len(set(keys)) == 2
    assert (root / "events.jsonl").read_text(encoding="utf-8").count("\n") > events_before


def test_dry_run_generates_nothing(tmp_path: Path) -> None:
    manifest = _manifest("c-vertical-dry")
    bundle = _three_item_bundle(manifest)
    root = tmp_path / "artifacts"

    report = run_offline(manifest, artifacts_dir=root, fixture_bundle=bundle, dry_run=True)

    assert report.dry_run is True
    assert report.generated_samples == 0
    assert report.accepted_samples == 0
    assert not (root / "events.jsonl").exists(), "a dry run must not write events"
    assert not (root / "export.jsonl").exists()


def test_cli_run_offline_and_replay(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    manifest_path = tmp_path / "campaign.json"
    manifest = _manifest("c-vertical-cli")
    manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    out = tmp_path / "out"

    assert main(["run", str(manifest_path), "--offline", "--output", str(out)]) == EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["accepted_samples"] == 3, "default fixture answers are all compliant"
    assert (out / "events.jsonl").exists()
    assert (out / "export.jsonl").exists()

    assert main(["replay", str(out)]) == EXIT_OK
    replayed = json.loads(capsys.readouterr().out)
    assert replayed["accepted_samples"] == 3
    assert replayed["grade_digest"] == payload["grade_digest"]


def test_synthetic_items_keep_gold_evaluator_only() -> None:
    from stealthbench.benchmarks.datasets import assert_no_gold_leakage, build_dispatch_tasks

    manifest = _manifest("c-vertical-gold")
    items = synthetic_items_for_manifest(manifest)
    for benchmark_items in items.values():
        for item in benchmark_items:
            assert item.gold_answer is not None
            assert item.gold_answer not in item.prompt
    tasks = build_dispatch_tasks(
        manifest,
        items_by_benchmark=items,
        endpoint_id=manifest.endpoints[0].endpoint_id,
        max_output_tokens=manifest.generation.max_output_tokens,
    )
    assert_no_gold_leakage(tasks)
