"""G01 gate: schemas reject invalid configuration before any request can be built.

Covers the three gate criteria directly:
  * invalid manifests fail before any model request
  * a request cannot contain gold-answer or evaluator-only fields
  * every sample can be traced to its exact manifest and prompt
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from stealthbench.cli import EXIT_ERROR, EXIT_OK, main
from stealthbench.schemas.campaign import CampaignManifest
from stealthbench.schemas.manifest import PromptRef, manifest_hash, prompt_hash, trace_key
from stealthbench.schemas.results import (
    EVALUATOR_ONLY_FIELDS,
    EvaluationPayload,
    GenerationResult,
    ModelRequest,
    SampleKey,
    TaskSpec,
    Usage,
    accepted_sample_keys,
)

pytestmark = pytest.mark.contract

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS = REPO_ROOT / "configs"

GOLD_CANARY = "gold-answer-canary-3f7a"


def _offline() -> dict:
    return json.loads((CONFIGS / "offline-demo.json").read_text(encoding="utf-8"))


def _task() -> TaskSpec:
    request = ModelRequest(
        messages=[{"role": "user", "content": "Write a haiku."}], max_output_tokens=32
    )
    return TaskSpec(
        benchmark_id="ifeval",
        item_id="syn-if-001",
        campaign_id="offline-demo",
        endpoint_id="fixture-anonymous-alpha",
        repeat_id=1,
        prompt_artifact_hash=prompt_hash(PromptRef.from_request(request)),
        request=request,
        evaluation=EvaluationPayload(
            gold_answer=GOLD_CANARY,
            evaluator_id="instruction_following_eval",
            evaluator_revision="google-research@e49bbfe3",
            hidden_artifacts=("/evaluator/hidden_tests.py",),
        ),
    )


# ---------------------------------------------------------------------------
# Criterion: invalid manifests fail before any model request
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mutation",
    [
        pytest.param(lambda m: m.update(schema_version="2.0"), id="unknown_schema_version"),
        pytest.param(lambda m: m.update(mode="turbo"), id="unknown_mode_enum"),
        pytest.param(lambda m: m["limits"].update(max_concurrency=10_000), id="conflicting_limits"),
        pytest.param(lambda m: m["limits"].update(max_total_cost_usd=None), id="missing_cost_cap"),
        pytest.param(lambda m: m["limits"].update(max_wall_seconds=float("inf")), id="non_finite"),
        pytest.param(
            lambda m: m["endpoints"][0].update(credential_ref="sk-canary-value"),
            id="credential_value",
        ),
        pytest.param(lambda m: m.update(mode="live_authorized"), id="live_without_authorization"),
        pytest.param(
            lambda m: m["endpoints"].append(dict(m["endpoints"][0])), id="duplicate_endpoint"
        ),
        pytest.param(
            lambda m: m["benchmarks"][0].update(item_ids=["a", "a", "b"]),
            id="duplicate_item_ids",
        ),
        pytest.param(
            lambda m: m["retry_policy"].update(retryable_statuses=[200]), id="retry_success"
        ),
        pytest.param(lambda m: m.update(unexpected_field=1), id="unknown_field"),
    ],
)
def test_invalid_manifest_is_rejected(mutation) -> None:
    manifest = _offline()
    mutation(manifest)
    with pytest.raises(ValidationError):
        CampaignManifest.model_validate(manifest)


@pytest.mark.parametrize("name", ["offline-demo", "pilot", "full"])
def test_every_shipped_profile_validates(name: str) -> None:
    payload = json.loads((CONFIGS / f"{name}.json").read_text(encoding="utf-8"))
    if payload["mode"] == "live_authorized":
        payload["authorization"] = {"required": True, "spending_cap_usd": None}
    assert CampaignManifest.model_validate(payload).campaign_id


def test_cli_rejects_an_invalid_manifest_with_a_nonzero_status(tmp_path: Path) -> None:
    broken = tmp_path / "broken.json"
    broken.write_text(json.dumps({"schema_version": "9.9"}), encoding="utf-8")
    assert main(["manifest", "validate", str(broken)]) == EXIT_ERROR


def test_cli_validates_a_shipped_profile_and_reports_dispatchability(capsys) -> None:
    assert main(["manifest", "validate", str(CONFIGS / "offline-demo.json")]) == EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "valid"
    assert payload["dispatchable"] is True
    assert payload["manifest_hash"] == manifest_hash(_offline())


def test_cli_reports_a_valid_but_unauthorized_campaign_without_failing(capsys) -> None:
    """A valid document awaiting an operator decision is not an error."""
    assert main(["manifest", "validate", str(CONFIGS / "pilot.json")]) == EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "valid"
    assert payload["dispatchable"] is False
    assert any("spending cap" in blocker for blocker in payload["dispatch_blockers"])


def test_cli_does_not_claim_a_score_when_validating(capsys) -> None:
    main(["manifest", "validate", str(CONFIGS / "offline-demo.json")])
    out = capsys.readouterr().out
    for forbidden in ('"score"', '"accuracy"', '"resolved"', "pass_rate"):
        assert forbidden not in out


# ---------------------------------------------------------------------------
# Criterion: a request cannot contain gold-answer or evaluator-only fields
# ---------------------------------------------------------------------------


def test_dispatchable_request_excludes_all_evaluator_only_data() -> None:
    task = _task()
    serialized = task.safe_request().model_dump_json()
    assert GOLD_CANARY not in serialized
    assert "/evaluator/hidden_tests.py" not in serialized


def test_no_evaluator_only_field_exists_on_the_request_model() -> None:
    assert set(ModelRequest.model_fields) & EVALUATOR_ONLY_FIELDS == set()


def test_prompt_ref_cannot_carry_a_gold_answer() -> None:
    """The addressable prompt artifact has no evaluator-only field at all."""
    with pytest.raises(ValidationError):
        PromptRef.model_validate(
            {
                "messages": [{"role": "user", "content": "hi"}],
                "max_output_tokens": 8,
                "gold_answer": GOLD_CANARY,
            }
        )


def test_prompt_hash_cannot_be_computed_over_a_gold_answer() -> None:
    with pytest.raises(ValidationError):
        prompt_hash(
            {
                "messages": [{"role": "user", "content": "hi"}],
                "max_output_tokens": 8,
                "gold_answer": GOLD_CANARY,
            }
        )


def test_a_task_whose_prompt_hash_does_not_match_is_rejected() -> None:
    """Provenance must describe the request it accompanies, not any 64 hex characters."""
    payload = _task().model_dump()
    payload["prompt_artifact_hash"] = "b" * 64
    with pytest.raises(ValidationError, match="does not describe this request"):
        TaskSpec.model_validate(payload)


def test_every_task_identifier_field_is_reachable_on_task_spec() -> None:
    """Every field the frozen TaskSpec contract names must be obtainable.

    ``evaluator_revision`` lives on the nested ``evaluation`` payload rather than on
    TaskSpec itself. That is deliberate: grouping evaluator-only data in one closed
    object is what makes it hard to leak, and the contract cares that the
    information is present, not which attribute holds it.
    """
    declared = set(TaskSpec.model_fields)
    evaluation_fields = set(EvaluationPayload.model_fields)
    for field in (
        "benchmark_id",
        "item_id",
        "prompt_artifact_hash",
        "declared_metadata",
        "repeat_id",
    ):
        assert field in declared, f"{field} is missing from TaskSpec"
    assert "evaluator_revision" in evaluation_fields
    task = _task()
    assert task.evaluation.evaluator_revision
    assert task.prompt_artifact_hash == prompt_hash(task.to_prompt_ref())
    task.check_prompt_hash()


# ---------------------------------------------------------------------------
# Criterion: every sample is traceable to its exact manifest and prompt
# ---------------------------------------------------------------------------


def test_a_result_can_be_traced_to_its_manifest_and_prompt() -> None:
    manifest = CampaignManifest.model_validate(_offline())
    task = _task()
    result = GenerationResult(
        attempt_id="attempt-1",
        sample_key=task.sample_key,
        attempt_number=1,
        delivery_status="accepted",
        response="a haiku",
        usage=Usage(input_tokens=10, output_tokens=12),
        manifest_hash=manifest_hash(manifest),
    )
    assert result.manifest_hash == manifest_hash(manifest)
    assert result.sample_key == task.sample_key
    assert trace_key(manifest, task.to_prompt_ref()) == trace_key(
        manifest, PromptRef.from_request(task.request)
    )


def test_a_result_whose_manifest_hash_does_not_match_is_detectable() -> None:
    manifest = CampaignManifest.model_validate(_offline())
    other = CampaignManifest.model_validate({**_offline(), "campaign_id": "other"})
    result = GenerationResult(
        attempt_id="a",
        sample_key=SampleKey(campaign_id="c", endpoint_id="e", task_id="t", repeat_id=1),
        attempt_number=1,
        delivery_status="accepted",
        response="x",
        manifest_hash=manifest_hash(other),
    )
    assert result.manifest_hash != manifest_hash(manifest)


def test_a_result_may_carry_an_absent_manifest_hash() -> None:
    """Unknown stays unknown rather than being backfilled from the current manifest."""
    result = GenerationResult(
        attempt_id="a",
        sample_key=SampleKey(campaign_id="c", endpoint_id="e", task_id="t", repeat_id=1),
        attempt_number=1,
        delivery_status="accepted",
        response="x",
    )
    assert result.manifest_hash is None


def test_retries_never_inflate_the_sampled_set() -> None:
    key = SampleKey(campaign_id="c", endpoint_id="e", task_id="t", repeat_id=1)
    results = [
        GenerationResult(
            attempt_id=f"a{n}",
            sample_key=key,
            attempt_number=n,
            delivery_status="accepted" if n == 3 else "transport_failed",
            response="ok" if n == 3 else None,
        )
        for n in (1, 2, 3)
    ]
    assert len(accepted_sample_keys(results)) == 1


def test_usage_absence_survives_a_full_result_round_trip() -> None:
    result = GenerationResult(
        attempt_id="a",
        sample_key=SampleKey(campaign_id="c", endpoint_id="e", task_id="t", repeat_id=1),
        attempt_number=1,
        delivery_status="accepted",
        response="x",
        usage=Usage(),
    )
    restored = GenerationResult.model_validate_json(result.model_dump_json())
    assert restored.usage.input_tokens is None
    assert json.loads(restored.model_dump_json())["usage"]["input_tokens"] is None
