"""Fresh-install release and regression evidence (G14 T14A).

The software release gate: a fresh installation performs discovery from
fixtures, benchmark generation, grading, signature matching, agent fixture
execution, resume, replay and report generation. Full-set configurations
enumerate the entire declared official split rather than a pilot selection.
No public outcome is asserted anywhere here: the offline campaign uses
synthetic fixture answers, and the profile checks assert declared counts,
not measured results.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from stealthbench.agents.opencode import (
    AgentActionKind,
    AgentLimits,
    AgentStatus,
    ModelBroker,
    StepContext,
    StepOutcome,
    run_agent_task,
)
from stealthbench.benchmarks.workflow import (
    build_default_fixture_bundle,
    replay_offline,
    run_offline,
)
from stealthbench.cli import EXIT_OK, main
from stealthbench.fingerprints.probes import (
    PROBE_BANK_VERSION,
    PROBE_COUNT,
    PROBE_REPEATS,
    build_probe_bank,
    probe_bank_digest,
    probe_bank_manifest,
    validate_probe_bank,
)
from stealthbench.schemas.campaign import CampaignManifest
from stealthbench.storage.replay import ReplayReader
from tests.conftest import REPO_ROOT

pytestmark = pytest.mark.integration


def _manifest(campaign_id: str) -> CampaignManifest:
    payload = json.loads((REPO_ROOT / "configs" / "offline-demo.json").read_text(encoding="utf-8"))
    payload["campaign_id"] = campaign_id
    payload.pop("authorization", None)
    return CampaignManifest.model_validate(payload)


def _artifact_lines(root: Path, name: str) -> list[str]:
    return [line for line in (root / name).read_text(encoding="utf-8").splitlines() if line.strip()]


def test_fresh_offline_workflow_from_discovery_to_replay(tmp_path: Path) -> None:
    """Discovery, generation, grading, resume, replay and report artifacts."""
    manifest = _manifest("c-release-001")
    bundle = build_default_fixture_bundle(manifest)
    assert bundle.catalog_source == "fixture"
    assert len(bundle.exchanges) == 3, "one recorded response per planned item"

    root = tmp_path / "artifacts"
    report = run_offline(manifest, artifacts_dir=root, fixture_bundle=bundle)

    assert report.planned == 3
    assert report.accepted_samples == 3
    assert report.transport_failed == 0
    assert report.strict_eligible == 3
    # Artifact layout per docs/operations.md section 5.
    for name in (
        "events.jsonl",
        "export.jsonl",
        "grades.jsonl",
        "summary.json",
        "manifest.json",
        "ledger.json",
    ):
        assert (root / name).exists(), f"missing artifact {name}"

    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    assert summary["accepted_samples"] == report.accepted_samples == 3
    assert summary["planned"] == 3
    assert len(_artifact_lines(root, "export.jsonl")) == 3
    assert len(_artifact_lines(root, "grades.jsonl")) == 3
    assert len(_artifact_lines(root, "events.jsonl")) >= 3

    # Regrade without regeneration: identical digest, no provider involved.
    replayed = replay_offline(root)
    assert replayed.accepted_samples == 3
    assert replayed.grade_digest == report.grade_digest
    assert replayed.damage == ()
    assert replayed.rejected == ()

    # Resume de-duplicates: a second run skips accepted samples.
    second = run_offline(manifest, artifacts_dir=root, fixture_bundle=bundle)
    assert second.skipped == 3
    combined = ReplayReader(root).read(strict=False)
    keys = [r.sample_key.as_tuple() for r in combined.results if r.is_accepted_sample]
    assert len(keys) == len(set(keys)) == 3

    # Dry run generates nothing.
    dry_root = tmp_path / "dry"
    dry = run_offline(manifest, artifacts_dir=dry_root, fixture_bundle=bundle, dry_run=True)
    assert dry.dry_run is True
    assert dry.generated_samples == 0
    assert not (dry_root / "events.jsonl").exists()


def test_signature_bank_matches_pinned_profile() -> None:
    """The 60-probe bank is versioned, separate, and reproducible."""
    bank = build_probe_bank()
    validate_probe_bank(bank)
    assert len(bank) == PROBE_COUNT == 60

    manifest = probe_bank_manifest()
    assert manifest["probe_version"] == PROBE_BANK_VERSION == "probe-v1"
    assert manifest["probe_count"] == 60
    assert manifest["repeats"] == PROBE_REPEATS == 3
    assert manifest["separate_from_benchmarks"] is True
    assert probe_bank_digest() == probe_bank_digest(dict(probe_bank_manifest()))

    pilot = json.loads((REPO_ROOT / "configs" / "pilot.json").read_text(encoding="utf-8"))
    assert pilot["signature_probes"]["probe_version"] == PROBE_BANK_VERSION
    assert pilot["signature_probes"]["probe_count"] == 60
    assert pilot["signature_probes"]["repeats"] == 3
    assert pilot["signature_probes"]["separate_from_benchmarks"] is True


def test_agent_fixture_trajectory_is_one_attempt(tmp_path: Path) -> None:
    """A complete fixture agent trajectory is one attempt with attribution."""

    def step_fn(_ctx: StepContext) -> StepOutcome:
        return StepOutcome(
            kind=AgentActionKind.FINISH,
            patch="diff --git a/f b/f\n+fix\n",
            input_tokens=10,
            output_tokens=5,
        )

    result = run_agent_task(
        task_id="release-agent-001",
        attempt_id="attempt-1",
        prompt_id="release-prompt-v1",
        model="fixture-model",
        limits=AgentLimits(max_steps=5, max_wall_seconds=30.0),
        broker=ModelBroker(credential_ref="STEALTHBENCH_BROKER_TOKEN"),
        step_fn=step_fn,
        workdir_parent=tmp_path,
    )
    assert result.status is AgentStatus.SUCCESS
    assert result.steps_taken == 1
    assert result.prompt_id == "release-prompt-v1"
    assert result.attribution().prompt_id == "release-prompt-v1"


def test_cli_run_replay_and_pending_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """CLI fresh-install path: run, replay and report generation (G13 landed)."""
    from stealthbench.cli import EXIT_OK as _OK

    manifest_path = tmp_path / "campaign.json"
    manifest_path.write_text(_manifest("c-release-cli").model_dump_json(indent=2), encoding="utf-8")
    out = tmp_path / "out"

    assert main(["run", str(manifest_path), "--offline", "--output", str(out)]) == _OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["accepted_samples"] == 3
    assert (out / "events.jsonl").exists()
    assert (out / "summary.json").exists()

    assert main(["replay", str(out)]) == _OK
    replayed = json.loads(capsys.readouterr().out)
    assert replayed["accepted_samples"] == 3
    assert replayed["grade_digest"] == payload["grade_digest"]

    report_out = tmp_path / "report"
    assert main(["report", str(out), "--output", str(report_out)]) == _OK
    reported = json.loads(capsys.readouterr().out)
    assert reported["status"] == "ok"
    assert (report_out / "index.html").is_file()


def test_doctor_reports_availability_offline(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`doctor` exits 0 from the repo root and names container state honestly."""
    monkeypatch.chdir(REPO_ROOT)
    assert main(["doctor"]) == EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ok"
    assert payload["schema_version"] == "1.0"
    assert payload["configs"]["offline-demo"] == "valid-dispatchable"
    assert payload["configs"]["pilot"] == "valid-unmaterialized"
    assert payload["configs"]["full"] == "valid-unmaterialized"
    assert payload["container_state"] in ("ready", "blocked_external")
    assert payload["datasets"]["offline_demo"] == "available"
    assert "blocked_external" in payload["datasets"]["official"]


def test_pilot_declares_600_item_core_plus_50_task_agents() -> None:
    """Pilot profile: 600 direct items plus 50 agent tasks, unmaterialized."""
    manifest = CampaignManifest.model_validate_json(
        (REPO_ROOT / "configs" / "pilot.json").read_text(encoding="utf-8")
    )
    assert manifest.materialized is False
    assert manifest.mode == "live_authorized"
    assert manifest.dispatch_blockers(), "an unmaterialized pilot must not be dispatchable"

    direct = sum(
        spec.planned_item_count or 0 for spec in manifest.benchmarks if spec.track == "direct"
    )
    agent = sum(
        spec.planned_item_count or 0 for spec in manifest.benchmarks if spec.track == "agent"
    )
    assert direct == 600, f"pilot direct core must be 600 items, got {direct}"
    assert agent == 50, f"pilot agent profile must be 50 tasks, got {agent}"
    assert manifest.total_planned_requests == 650


EXPECTED_FULL_SPLIT_SIZES = {
    "livecodebench": 1055,
    "evalplus": 164,
    "ifeval": 541,
    "mmlu_pro": 12032,
    "math500": 500,
    "swebench": 500,
    "terminalbench": 66,
}


def test_full_profile_enumerates_declared_splits_not_pilot() -> None:
    """Full profile requires whole official splits; it is not a relabelled pilot."""
    full = CampaignManifest.model_validate_json(
        (REPO_ROOT / "configs" / "full.json").read_text(encoding="utf-8")
    )
    pilot = CampaignManifest.model_validate_json(
        (REPO_ROOT / "configs" / "pilot.json").read_text(encoding="utf-8")
    )
    assert full.materialized is False
    assert all(spec.require_full_split for spec in full.benchmarks)
    assert not any(spec.require_full_split for spec in pilot.benchmarks)

    full_by_id = {spec.benchmark_id: spec for spec in full.benchmarks}
    for benchmark_id, expected in EXPECTED_FULL_SPLIT_SIZES.items():
        assert full_by_id[benchmark_id].expected_item_count == expected
    # BFCL/RULER counts are genuinely unknown until G08 freezes them.
    for benchmark_id in ("bfcl", "ruler"):
        spec = full_by_id[benchmark_id]
        assert spec.planned_item_count is None
        assert spec.split_enumeration, f"{benchmark_id} must document its enumeration"
    assert full.total_planned_requests is None, "unknown sizes must not sum to a number"
    assert full.known_planned_requests == sum(EXPECTED_FULL_SPLIT_SIZES.values())

    pilot_by_id = {spec.benchmark_id: spec for spec in pilot.benchmarks}
    for benchmark_id, expected in EXPECTED_FULL_SPLIT_SIZES.items():
        pilot_count = pilot_by_id[benchmark_id].planned_item_count or 0
        assert pilot_count < expected, (
            f"pilot {benchmark_id} ({pilot_count}) must be a subset of "
            f"the full split ({expected}), not the full split itself"
        )
