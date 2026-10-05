"""Contract tests for the official Terminal-Bench wrapper (T10C, G10).

Acceptance for T10C: known pass/fail artifacts match official grading; one
trajectory remains one attempt; task images are pinned.

Expected outcomes below are hand-computed from the documented binary-reward
rule (``1.0`` only when every claim is correct), not derived from the
implementation. Production wiring supplies rewards from the separate
verifier container (``tests/test_scoring.py`` reading its own
``expected.json``); the adapter under test only validates the payload and
maps it onto ``GradeResult``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from stealthbench.agents.terminalbench import (
    AGENT_TIMEOUT_SECONDS,
    GRADER_VERSION,
    HARBOR_DATASET,
    HARBOR_VERSION,
    PINNED_COMMIT,
    PINNED_TAG,
    PINNED_TASK_COUNT,
    VERIFIER_TIMEOUT_SECONDS,
    TerminalBenchGrade,
    agent_container_args,
    assert_agent_verifier_separation,
    dispatch_request,
    evaluator_layout_for_task,
    grade_accepted_sample,
    grade_generation,
    grade_with_verifier_payload,
    harbor_binary,
    image_reference,
    is_malformed,
    is_pass,
    live_grade_status,
    parse_image_digest,
    parse_reward_json,
    parse_reward_text,
    parse_task_digest,
    require_harbor_for_live_grade,
    summarize_grades,
    task_content_digest,
    verifier_container_args,
    verify_task_digest,
    verify_task_set,
)
from stealthbench.sandbox.runtime import SandboxLimits
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

GOLD_CANARY = "canary-gold-terminalbench-t10c-7e19"
TASK_DIGEST = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
IMAGE_DIGEST = "sha256:" + "cd" * 32
IMAGE_TAG = "terminal-bench/task-foo:latest"


def _task(
    item_id: str = "task-hello-world",
    prompt: str = "Create /app/output.txt containing hello.",
    gold: str = GOLD_CANARY,
) -> TaskSpec:
    request = ModelRequest(
        messages=[{"role": "user", "content": prompt}],
        max_output_tokens=512,
    )
    return TaskSpec(
        benchmark_id="terminalbench",
        item_id=item_id,
        campaign_id="c1",
        endpoint_id="e1",
        repeat_id=1,
        prompt_artifact_hash=prompt_hash(PromptRef.from_request(request)),
        request=request,
        evaluation=EvaluationPayload(
            gold_answer=gold,
            evaluator_id="terminal-bench.verifier",
            evaluator_revision=f"{PINNED_TAG}+{PINNED_COMMIT}",
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
# Wrapper identity pins the official 4.0 release
# ---------------------------------------------------------------------------


def test_release_pins_match_inventory() -> None:
    assert PINNED_TAG == "v4.0.0"
    assert PINNED_COMMIT == "452bf305c6daa62fc59061d22133a7cbc7c1572e"
    assert PINNED_TAG in GRADER_VERSION
    assert HARBOR_DATASET == "terminal-bench/terminal-bench@4.0.0"
    assert HARBOR_VERSION == "harbor==0.23.0"
    assert PINNED_TASK_COUNT == 66
    assert AGENT_TIMEOUT_SECONDS == 8 * 3600
    assert VERIFIER_TIMEOUT_SECONDS == 300.0


# ---------------------------------------------------------------------------
# Task identity is a per-task sha256, never a moving tag
# ---------------------------------------------------------------------------


def test_task_digest_must_be_64_hex() -> None:
    assert parse_task_digest(TASK_DIGEST) == TASK_DIGEST
    with pytest.raises(ValueError, match="64"):
        parse_task_digest("not-a-digest")
    with pytest.raises(ValueError, match="64"):
        parse_task_digest("ABCDEF")


def test_task_content_digest_matches_hand_computed_sha256() -> None:
    assert task_content_digest(b"hello") == (
        "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
    )
    assert verify_task_digest(task_id="t", content=b"", expected_sha256=TASK_DIGEST) == TASK_DIGEST


def test_drifted_task_content_is_refused() -> None:
    with pytest.raises(ValueError, match="drifted"):
        verify_task_digest(task_id="t", content=b"changed", expected_sha256=TASK_DIGEST)


def test_task_set_validation() -> None:
    ids = verify_task_set({"b": TASK_DIGEST, "a": TASK_DIGEST})
    assert ids == ("a", "b")
    with pytest.raises(ValueError, match="empty"):
        verify_task_set({})
    with pytest.raises(ValueError, match="64"):
        verify_task_set({"a": "bad"})


# ---------------------------------------------------------------------------
# Binary reward: 1.0 passes, anything else fails; malformed is unevaluated
# ---------------------------------------------------------------------------


def test_reward_text_parsing_is_binary() -> None:
    assert parse_reward_text("1.0\n") == 1.0
    assert parse_reward_text("0.0") == 0.0
    with pytest.raises(ValueError, match="binary"):
        parse_reward_text("0.5")
    with pytest.raises(ValueError, match="binary"):
        parse_reward_text("pass")


def test_reward_json_parsing_is_binary() -> None:
    assert parse_reward_json('{"reward": 1.0}') == 1.0
    assert parse_reward_json({"reward": 0.0}) == 0.0
    with pytest.raises(ValueError, match="binary"):
        parse_reward_json('{"reward": 0.5}')
    with pytest.raises(ValueError, match="'reward'"):
        parse_reward_json('{"score": 1.0}')


def test_is_pass_only_for_reward_one() -> None:
    assert is_pass(1.0) is True
    assert is_pass(0.0) is False


def test_known_passing_artifact_grades_as_pass() -> None:
    task = _task()
    grade = grade_accepted_sample(
        task=task,
        artifact="created /app/output.txt",
        reward=1.0,
        task_digest=TASK_DIGEST,
        image_digest=IMAGE_DIGEST,
    )
    assert grade.grade.correctness == "pass"
    assert grade.grade.score_components["reward"] == 1.0
    assert grade.grade.counts_toward_accuracy
    assert grade.reward == 1.0


def test_known_failing_artifact_grades_as_fail() -> None:
    task = _task()
    grade = grade_accepted_sample(
        task=task,
        artifact="created the wrong file",
        reward=0.0,
        task_digest=TASK_DIGEST,
        image_digest=IMAGE_DIGEST,
    )
    assert grade.grade.correctness == "fail"
    assert grade.grade.score_components["reward"] == 0.0
    assert grade.grade.counts_toward_accuracy


def test_non_binary_reward_is_refused() -> None:
    task = _task()
    with pytest.raises(ValueError, match="binary"):
        grade_accepted_sample(
            task=task,
            artifact="x",
            reward=0.5,
            task_digest=TASK_DIGEST,
            image_digest=IMAGE_DIGEST,
        )


def test_verifier_payloads_must_agree() -> None:
    task = _task()
    both_pass = grade_with_verifier_payload(
        task=task,
        artifact="done",
        reward_text="1.0",
        reward_json='{"reward": 1.0}',
        task_digest=TASK_DIGEST,
        image_digest=IMAGE_DIGEST,
    )
    assert both_pass.grade.correctness == "pass"
    disagree = grade_with_verifier_payload(
        task=task,
        artifact="done",
        reward_text="1.0",
        reward_json='{"reward": 0.0}',
        task_digest=TASK_DIGEST,
        image_digest=IMAGE_DIGEST,
    )
    assert disagree.grade.evaluator == "fail"
    assert disagree.grade.correctness == "unavailable"
    missing = grade_with_verifier_payload(
        task=task,
        artifact="done",
        task_digest=TASK_DIGEST,
        image_digest=IMAGE_DIGEST,
    )
    assert missing.grade.evaluator == "fail"


def test_empty_artifact_is_a_format_failure_in_the_denominator() -> None:
    assert is_malformed("  \n ")
    assert not is_malformed("created file")
    task = _task()
    grade = grade_accepted_sample(
        task=task,
        artifact="   ",
        reward=0.0,
        task_digest=TASK_DIGEST,
        image_digest=IMAGE_DIGEST,
    )
    assert grade.grade.format == "fail"
    assert grade.grade.correctness == "fail"
    assert grade.grade.counts_toward_accuracy


# ---------------------------------------------------------------------------
# Image digests pinned, not tags
# ---------------------------------------------------------------------------


def test_image_reference_requires_a_digest() -> None:
    ref = image_reference(image_tag=IMAGE_TAG, image_digest=IMAGE_DIGEST)
    assert ref.endswith(f"@{IMAGE_DIGEST}")
    assert ":latest" not in ref
    with pytest.raises(ValueError, match="mutable"):
        image_reference(image_tag=IMAGE_TAG, image_digest=None)
    with pytest.raises(ValueError, match="sha256"):
        parse_image_digest("latest")


# ---------------------------------------------------------------------------
# Verifier runs in a separate container
# ---------------------------------------------------------------------------


def test_agent_and_verifier_layouts_are_siblings(tmp_path: Path) -> None:
    layout = evaluator_layout_for_task(tmp_path / "eval")
    assert layout.verifier_dir != layout.workdir
    assert layout.verifier_dir.parent == layout.workdir.parent
    assert_agent_verifier_separation(layout.workdir, layout.verifier_dir)
    with pytest.raises(Exception, match=r"sibling|overlap"):
        assert_agent_verifier_separation(layout.workdir, layout.workdir)


def test_agent_container_never_mounts_the_verifier(tmp_path: Path) -> None:
    args = agent_container_args(
        runtime="docker",
        image="terminal-bench/task:latest",
        workdir=tmp_path,
        command=["bash", "/work/run.sh"],
        limits=SandboxLimits(wall_seconds=60.0),
    )
    assert "--network" in args and "none" in args
    assert f"{tmp_path}:/work:ro" in args
    assert "verifier" not in " ".join(args).lower()


def test_verifier_container_mounts_only_the_verifier(tmp_path: Path) -> None:
    work = tmp_path / "work"
    verifier = tmp_path / "verifier"
    work.mkdir()
    verifier.mkdir()
    args = verifier_container_args(
        runtime="docker", image="tb-verifier:latest", verifier_dir=verifier
    )
    assert f"{verifier}:/verifier:ro" in args
    assert str(work) not in " ".join(args)


# ---------------------------------------------------------------------------
# Transport / evaluator failures stay out of the denominator
# ---------------------------------------------------------------------------


def test_transport_failure_is_unavailable_not_failed() -> None:
    task = _task()
    generation = _generation(task, None, DeliveryStatus.TRANSPORT_FAILED)
    grade = grade_generation(task=task, generation=generation, reward=1.0)
    assert grade.grade.correctness == "unavailable"
    assert grade.grade.transport == "fail"
    assert not grade.grade.counts_toward_accuracy


def test_accepted_trajectory_without_a_verdict_is_unevaluated() -> None:
    task = _task()
    grade = grade_generation(task=task, generation=_generation(task, "did work"))
    assert grade.grade.evaluator == "fail"
    assert grade.grade.correctness == "unavailable"
    assert not grade.grade.counts_toward_accuracy


def test_sample_key_mismatch_refuses_to_grade() -> None:
    task = _task(item_id="task-a")
    other = _task(item_id="task-b")
    with pytest.raises(ValueError, match="does not match"):
        grade_generation(
            task=task,
            generation=_generation(other, "did work"),
            reward=1.0,
            task_digest=TASK_DIGEST,
            image_digest=IMAGE_DIGEST,
        )


def test_gold_never_appears_in_the_dispatchable_request() -> None:
    task = _task()
    serialized = dispatch_request(task).model_dump_json()
    assert GOLD_CANARY not in serialized
    assert "gold" not in serialized.lower()


def test_model_request_schema_has_no_evaluator_only_field() -> None:
    assert set(ModelRequest.model_fields) & EVALUATOR_ONLY_FIELDS == set()


# ---------------------------------------------------------------------------
# One trajectory is one attempt; pass rate over evaluated tasks
# ---------------------------------------------------------------------------


def test_two_trajectories_are_two_entries_not_best_of() -> None:
    task = _task()
    good = grade_accepted_sample(
        task=task,
        artifact="good",
        reward=1.0,
        task_digest=TASK_DIGEST,
        image_digest=IMAGE_DIGEST,
    )
    bad = grade_accepted_sample(
        task=task,
        artifact="bad",
        reward=0.0,
        task_digest=TASK_DIGEST,
        image_digest=IMAGE_DIGEST,
    )
    summary = summarize_grades([good, bad])
    assert (summary.passed, summary.eligible) == (1, 2)
    assert summary.pass_rate == pytest.approx(0.5)


def test_unknown_rate_is_null_never_zero() -> None:
    empty = summarize_grades([])
    assert empty.pass_rate is None
    assert empty.pass_rate != 0.0
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
    assert failed_only.pass_rate is None


def test_grade_survives_a_result_schema_round_trip() -> None:
    task = _task()
    grade = grade_accepted_sample(
        task=task,
        artifact="done",
        reward=1.0,
        task_digest=TASK_DIGEST,
        image_digest=IMAGE_DIGEST,
    )
    revived = TerminalBenchGrade.model_validate(grade.model_dump(mode="json"))
    assert revived == grade
    sample: SampleKey = task.sample_key
    assert revived.grade.sample_key == sample


def test_live_harbor_grading_reports_blocked_external_without_harbor() -> None:
    status = live_grade_status("terminal-bench live grading")
    assert status["state"] in {"ready", "blocked_external"}
    if harbor_binary() is None:
        assert status["state"] == "blocked_external"
        assert "harbor" in (status["reason"] or "").lower()
    try:
        name = require_harbor_for_live_grade("terminal-bench live grading")
    except Exception as exc:
        assert "blocked_external" in str(exc)
    else:
        assert name in {"docker", "podman"}
