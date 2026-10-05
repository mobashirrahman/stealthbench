"""Contract tests for the official SWE-bench Verified wrapper (T10B, G10).

Acceptance for T10B: known resolving and non-resolving patches match
official grading in fresh pinned environments.

Expected outcomes below are hand-computed from the documented
``RESOLVED_FULL`` rule (every FAIL_TO_PASS and every PASS_TO_PASS must
pass), not derived from the implementation. Production wiring supplies
verdicts from the pinned harness
(``swebench/harness/grading.py@02e7a74f``) executed in a fresh container
per task; the adapter under test only applies the rule and maps verdicts
onto ``GradeResult``.
"""

from __future__ import annotations

import pytest

from stealthbench.agents.swebench import (
    GRADER_VERSION,
    IMAGE_PATTERN,
    PINNED_DATASET_ID,
    PINNED_HARNESS_COMMIT,
    PINNED_ITEM_COUNT,
    RunCache,
    SweBenchGrade,
    dispatch_request,
    grade_accepted_sample,
    grade_generation,
    image_reference,
    is_malformed,
    is_resolved,
    live_grade_status,
    new_run_id,
    parse_image_digest,
    patch_hash,
    require_container_for_live_grade,
    summarize_grades,
)
from stealthbench.schemas.manifest import PromptRef, prompt_hash
from stealthbench.schemas.results import (
    EVALUATOR_ONLY_FIELDS,
    DeliveryStatus,
    EvaluationPayload,
    GenerationResult,
    ModelRequest,
    SampleKey,
    TaskSpec,
)

pytestmark = pytest.mark.contract

GOLD_CANARY = "canary-gold-swebench-t10b-1a77"
IMAGE_TAG = "swebench/sweb.eval.x86_64.django__django-11039:latest"
IMAGE_DIGEST = "sha256:" + "ab" * 32


def _task(
    item_id: str = "django__django-11039",
    prompt: str = "Fix the failing test in this repository.",
    gold: str = GOLD_CANARY,
) -> TaskSpec:
    request = ModelRequest(
        messages=[{"role": "user", "content": prompt}],
        max_output_tokens=512,
    )
    return TaskSpec(
        benchmark_id="swebench",
        item_id=item_id,
        campaign_id="c1",
        endpoint_id="e1",
        repeat_id=1,
        prompt_artifact_hash=prompt_hash(PromptRef.from_request(request)),
        request=request,
        evaluation=EvaluationPayload(
            gold_answer=gold,
            evaluator_id="swebench.harness.grading",
            evaluator_revision=PINNED_HARNESS_COMMIT,
        ),
    )


def _generation(
    task: TaskSpec,
    response: str | None,
    status: DeliveryStatus = DeliveryStatus.ACCEPTED,
) -> GenerationResult:
    return GenerationResult(
        attempt_id="attempt-1",
        sample_key=task.sample_key,
        attempt_number=1,
        delivery_status=status,
        response=response,
    )


# ---------------------------------------------------------------------------
# Wrapper identity pins the official harness and dataset
# ---------------------------------------------------------------------------


def test_grader_version_pins_the_official_harness_commit() -> None:
    assert PINNED_HARNESS_COMMIT == "02e7a74ffd0b707aab73d203fe87bdc7c76afc8e"
    assert PINNED_HARNESS_COMMIT in GRADER_VERSION
    assert GRADER_VERSION.startswith("swebench.harness.grading@")


def test_dataset_identity_matches_inventory() -> None:
    assert PINNED_DATASET_ID == "SWE-bench/SWE-bench_Verified"
    assert PINNED_ITEM_COUNT == 500
    assert IMAGE_PATTERN.endswith(":latest")


# ---------------------------------------------------------------------------
# Image identity is a digest, never a mutable tag
# ---------------------------------------------------------------------------


def test_image_digest_must_be_sha256_hex() -> None:
    assert parse_image_digest(IMAGE_DIGEST) == IMAGE_DIGEST
    with pytest.raises(ValueError, match="mutable"):
        parse_image_digest("swebench/sweb.eval.x: latest".replace(" ", ""))
    with pytest.raises(ValueError, match="must match"):
        parse_image_digest("sha256:ZZZ")


def test_image_reference_requires_a_digest() -> None:
    ref = image_reference(image_tag=IMAGE_TAG, image_digest=IMAGE_DIGEST)
    assert ref.endswith(f"@{IMAGE_DIGEST}")
    assert ":latest" not in ref
    with pytest.raises(ValueError, match="mutable"):
        image_reference(image_tag=IMAGE_TAG, image_digest=None)


# ---------------------------------------------------------------------------
# Fresh run_id per regrade; stale reuse is refused
# ---------------------------------------------------------------------------


def test_new_run_ids_are_unique() -> None:
    assert new_run_id() != new_run_id()


def test_same_run_id_with_a_changed_patch_is_refused() -> None:
    cache = RunCache()
    run_id = new_run_id()
    cache.register(run_id=run_id, instance_id="i-1", patch="patch v1")
    # Same patch under the same id is idempotent.
    cache.register(run_id=run_id, instance_id="i-1", patch="patch v1")
    with pytest.raises(ValueError, match="new_run_id"):
        cache.register(run_id=run_id, instance_id="i-1", patch="patch v2")


def test_regrade_with_a_fresh_run_id_is_allowed() -> None:
    cache = RunCache()
    task = _task()
    first = grade_accepted_sample(
        task=task,
        patch="patch v1",
        fail_to_pass={"t1": True},
        pass_to_pass={"t2": True},
        run_id=new_run_id(),
        image_digest=IMAGE_DIGEST,
        cache=cache,
    )
    second = grade_accepted_sample(
        task=task,
        patch="patch v2",
        fail_to_pass={"t1": False},
        pass_to_pass={"t2": True},
        run_id=new_run_id(),
        image_digest=IMAGE_DIGEST,
        cache=cache,
    )
    assert first.grade.correctness == "pass"
    assert second.grade.correctness == "fail"


def test_patch_hash_is_stable_sha256() -> None:
    assert patch_hash("abc") == patch_hash("abc")
    assert patch_hash("abc") != patch_hash("abd")
    assert len(patch_hash("abc")) == 64


# ---------------------------------------------------------------------------
# RESOLVED_FULL: every FAIL_TO_PASS and every PASS_TO_PASS
# ---------------------------------------------------------------------------


def test_resolved_requires_every_listed_test() -> None:
    assert is_resolved({"t1": True, "t2": True}, {"p1": True}) is True
    assert is_resolved({"t1": True, "t2": False}, {"p1": True}) is False
    assert is_resolved({"t1": True}, {"p1": True, "p2": False}) is False


def test_fail_only_grading_ignores_pass_to_pass() -> None:
    assert is_resolved({"t1": True}, {"p1": False}, fail_only=True) is True
    assert is_resolved({"t1": False}, {"p1": True}, fail_only=True) is False


def test_empty_fail_to_pass_is_refused() -> None:
    with pytest.raises(ValueError, match="FAIL_TO_PASS"):
        is_resolved({}, {"p1": True})


# ---------------------------------------------------------------------------
# Known resolving / non-resolving golden fixtures
# ---------------------------------------------------------------------------


def test_known_resolving_patch_grades_as_resolved() -> None:
    task = _task()
    grade = grade_accepted_sample(
        task=task,
        patch="diff --git a/a b/a\n+fixed\n",
        fail_to_pass={"test_issue": True, "test_repro": True},
        pass_to_pass={"test_existing": True},
        run_id=new_run_id(),
        image_digest=IMAGE_DIGEST,
    )
    assert grade.resolved is True
    assert grade.grade.correctness == "pass"
    assert grade.grade.score_components["resolved"] == 1.0
    assert grade.grade.counts_toward_accuracy
    assert grade.run_id
    assert grade.image_digest == IMAGE_DIGEST


def test_known_non_resolving_patch_grades_as_unresolved() -> None:
    task = _task()
    grade = grade_accepted_sample(
        task=task,
        patch="diff --git a/a b/a\n+still broken\n",
        fail_to_pass={"test_issue": True, "test_repro": False},
        pass_to_pass={"test_existing": True},
        run_id=new_run_id(),
        image_digest=IMAGE_DIGEST,
    )
    assert grade.resolved is False
    assert grade.grade.correctness == "fail"
    assert grade.grade.score_components["resolved"] == 0.0
    assert grade.grade.counts_toward_accuracy


def test_broken_pass_to_pass_is_unresolved() -> None:
    task = _task()
    grade = grade_accepted_sample(
        task=task,
        patch="diff\n+fix with regression\n",
        fail_to_pass={"test_issue": True},
        pass_to_pass={"test_existing": False},
        run_id=new_run_id(),
        image_digest=IMAGE_DIGEST,
    )
    assert grade.resolved is False
    assert grade.grade.correctness == "fail"


def test_empty_patch_is_a_format_failure_in_the_denominator() -> None:
    assert is_malformed("   ")
    assert not is_malformed("diff --git a")
    task = _task()
    grade = grade_accepted_sample(
        task=task,
        patch="   ",
        fail_to_pass={"t1": False},
        pass_to_pass={},
        run_id=new_run_id(),
        image_digest=IMAGE_DIGEST,
    )
    assert grade.grade.format == "fail"
    assert grade.grade.correctness == "fail"
    assert grade.grade.counts_toward_accuracy


# ---------------------------------------------------------------------------
# Transport / evaluator failures stay out of the denominator
# ---------------------------------------------------------------------------


def test_transport_failure_is_unavailable_not_unresolved() -> None:
    task = _task()
    generation = _generation(task, None, DeliveryStatus.TRANSPORT_FAILED)
    grade = grade_generation(task=task, generation=generation)
    assert grade.grade.correctness == "unavailable"
    assert grade.grade.transport == "fail"
    assert grade.grade.evaluator == "unavailable"
    assert not grade.grade.counts_toward_accuracy


def test_accepted_trajectory_without_verdicts_is_unevaluated() -> None:
    task = _task()
    grade = grade_generation(task=task, generation=_generation(task, "diff\n+x\n"))
    assert grade.grade.evaluator == "fail"
    assert grade.grade.correctness == "unavailable"
    assert not grade.grade.counts_toward_accuracy


def test_sample_key_mismatch_refuses_to_grade() -> None:
    task = _task(item_id="repo__a-1")
    other = _task(item_id="repo__b-2")
    with pytest.raises(ValueError, match="does not match"):
        grade_generation(
            task=task,
            generation=_generation(other, "diff"),
            fail_to_pass={"t": True},
            pass_to_pass={},
        )


def test_gold_never_appears_in_the_dispatchable_request() -> None:
    task = _task()
    serialized = dispatch_request(task).model_dump_json()
    assert GOLD_CANARY not in serialized
    assert "gold" not in serialized.lower()


def test_model_request_schema_has_no_evaluator_only_field() -> None:
    assert set(ModelRequest.model_fields) & EVALUATOR_ONLY_FIELDS == set()


# ---------------------------------------------------------------------------
# One trajectory is one attempt; resolution rate over graded samples
# ---------------------------------------------------------------------------


def test_two_trajectories_are_two_attempts_not_best_of() -> None:
    task = _task()
    good = grade_accepted_sample(
        task=task,
        patch="good",
        fail_to_pass={"t": True},
        pass_to_pass={},
        run_id=new_run_id(),
        image_digest=IMAGE_DIGEST,
    )
    bad = grade_accepted_sample(
        task=task,
        patch="bad",
        fail_to_pass={"t": False},
        pass_to_pass={},
        run_id=new_run_id(),
        image_digest=IMAGE_DIGEST,
    )
    summary = summarize_grades([good, bad])
    assert (summary.resolved, summary.eligible) == (1, 2)
    assert summary.resolution_rate == pytest.approx(0.5)


def test_unknown_rate_is_null_never_zero() -> None:
    empty = summarize_grades([])
    assert empty.resolution_rate is None
    assert empty.resolution_rate != 0.0
    task = _task()
    failed_only = summarize_grades(
        [
            grade_generation(
                task=task,
                generation=_generation(task, None, DeliveryStatus.TRANSPORT_FAILED),
            )
        ]
    )
    assert failed_only.eligible == 0
    assert failed_only.resolution_rate is None


def test_grade_survives_a_result_schema_round_trip() -> None:
    task = _task()
    grade = grade_accepted_sample(
        task=task,
        patch="diff",
        fail_to_pass={"t": True},
        pass_to_pass={},
        run_id=new_run_id(),
        image_digest=IMAGE_DIGEST,
    )
    revived = SweBenchGrade.model_validate(grade.model_dump(mode="json"))
    assert revived == grade
    sample: SampleKey = task.sample_key
    assert revived.grade.sample_key == sample


def test_live_harness_grading_reports_blocked_external_without_runtime() -> None:
    status = live_grade_status("swe-bench live grading")
    assert status["state"] in {"ready", "blocked_external"}
    try:
        name = require_container_for_live_grade("swe-bench live grading")
    except Exception as exc:
        assert "blocked_external" in str(exc)
    else:
        assert name in {"docker", "podman"}
