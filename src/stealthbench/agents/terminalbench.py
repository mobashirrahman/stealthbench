"""Official Terminal-Bench wrapper: thin deterministic adapter (G10 T10C).

Official source (see ``docs/upstream-inventory.md``):

* Repo ``https://github.com/harbor-framework/terminal-bench`` at tag
  ``v4.0.0`` (commit ``452bf305c6daa62fc59061d22133a7cbc7c1572e``).
* Dataset package ``terminal-bench/terminal-bench@4.0.0`` (66 tasks);
  ``tasks/dataset.toml`` carries a sha256 digest per task.
* Evaluator: ``tests/test_scoring.py`` in a **separate** verifier container,
  invoked as ``harbor run -d terminal-bench/terminal-bench@4.0.0`` with the
  ``harbor==0.23.0`` CLI. The retired PyPI ``terminal-bench`` (0.2.18) harness
  must not be used.
* Every task in 4.0 has a flat 8-hour agent timeout; the verifier timeout is
  300s for the inspected task. 3.0 results are not comparable.
* Reward is binary per task: ``1.0`` only when every claim is correct,
  written to ``/logs/verifier/reward.txt`` and ``reward.json``.
* Some tasks use an LLM judge (see ``task.toml [metadata]``); verifier
  behaviour is version-specific and verifier tampering is an acknowledged
  upstream risk with a 4.1 remediation milestone.
* Task data carries a canary GUID and must never appear in training corpora.

Deliberate divergence (recorded in the inventory): tasks are identified by
their per-task sha256 digest so the task set cannot drift under a moving
version tag.

What this module does and does not do:

* It does **not** reimplement ``test_scoring.py``. The caller supplies the
  binary reward (in production: the separate verifier container reading its
  own ``expected.json``); the adapter only validates the reward shape and
  maps it onto the frozen ``GradeResult`` contract.
* Task identity is digest-pinned: any grade without a valid per-task sha256
  raises instead of trusting a moving version tag.
* The verifier runs in a separate container: the agent container argv never
  mounts the verifier directory, and the verifier argv never mounts agent
  state. :func:`assert_agent_verifier_separation` enforces the sibling
  layout.
* Reward is binary: only ``1.0`` passes; any other value (including
  ``0.5``) is ``correctness="fail"``. Malformed reward payloads are
  evaluator failures, never passes.
* One trajectory is one attempt (``docs/contracts.md``): the summary counts
  each graded trajectory once and never selects the best of several.

Offline by construction: pure functions over local reward payloads and
contracts. Live runs require the ``harbor`` CLI plus a container runtime;
without them the live helpers report ``blocked_external``.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from pydantic import Field

from stealthbench.sandbox.policy import (
    EvaluatorLayout,
    assert_verifier_separation,
)
from stealthbench.sandbox.runtime import (
    SandboxLimits,
    container_check_status,
    container_exec_args,
    require_container_runtime,
)
from stealthbench.schemas.results import (
    DeliveryStatus,
    DenominatorEligibility,
    GenerationResult,
    GradeResult,
    ModelRequest,
    ResultModel,
    StatusFlag,
    TaskSpec,
)

#: Grader identity: the official verifier at the pinned 4.0 tag.
PINNED_TAG: Final[str] = "v4.0.0"
PINNED_COMMIT: Final[str] = "452bf305c6daa62fc59061d22133a7cbc7c1572e"
GRADER_VERSION: Final[str] = f"terminal-bench.verifier@{PINNED_TAG}+{PINNED_COMMIT}"
OFFICIAL_VERIFIER_MODULE: Final[str] = "tests/test_scoring.py"
OFFICIAL_COMMAND: Final[str] = "harbor run -d terminal-bench/terminal-bench@4.0.0"

#: Dataset identity from docs/upstream-inventory.md.
HARBOR_DATASET: Final[str] = "terminal-bench/terminal-bench@4.0.0"
HARBOR_VERSION: Final[str] = "harbor==0.23.0"
PINNED_TASK_COUNT: Final[int] = 66

#: Official timeouts: flat 8h agent timeout per task; 300s verifier timeout.
AGENT_TIMEOUT_SECONDS: Final[float] = 8 * 3600
VERIFIER_TIMEOUT_SECONDS: Final[float] = 300.0

_TASK_DIGEST_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{64}$")


def parse_task_digest(digest: str) -> str:
    """Validate a per-task sha256 digest (64 lowercase hex)."""
    if not _TASK_DIGEST_PATTERN.match(digest):
        raise ValueError(
            f"task digest {digest!r} must be 64 lowercase hex (tasks/dataset.toml sha256)"
        )
    return digest


def task_content_digest(content: bytes) -> str:
    """sha256 of raw task content, for pinning the task set."""
    return hashlib.sha256(content).hexdigest()


def verify_task_digest(*, task_id: str, content: bytes, expected_sha256: str) -> str:
    """Return the content digest, or raise when it differs from the pin."""
    parse_task_digest(expected_sha256)
    actual = task_content_digest(content)
    if actual.lower() != expected_sha256.lower():
        raise ValueError(
            f"task {task_id!r} hashes to {actual} but dataset.toml records "
            f"{expected_sha256}; refusing a drifted task set"
        )
    return actual


def verify_task_set(task_digests: Mapping[str, str]) -> tuple[str, ...]:
    """Validate a task-id -> digest map; return the sorted task ids.

    Every digest must be valid sha256 and every id non-empty. An empty map
    raises: an unknown task set is not an empty score.
    """
    if not task_digests:
        raise ValueError("task set is empty; refusing to grade zero tasks")
    for task_id, digest in task_digests.items():
        if not task_id:
            raise ValueError("task id must be non-empty")
        parse_task_digest(digest)
    return tuple(sorted(task_digests))


def parse_reward_text(text: str) -> float:
    """Parse ``/logs/verifier/reward.txt``: only ``1.0`` or ``0.0``."""
    stripped = text.strip()
    if stripped not in {"1.0", "0.0", "1", "0"}:
        raise ValueError(f"reward text {stripped!r} is not binary; expected '1.0' or '0.0'")
    return 1.0 if stripped in {"1.0", "1"} else 0.0


def parse_reward_json(payload: str | Mapping[str, Any]) -> float:
    """Parse ``reward.json``: ``{\"reward\": 1.0}`` with a binary value."""
    if isinstance(payload, str):
        try:
            data: Any = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ValueError(f"reward.json is not valid JSON: {exc}") from exc
    else:
        data = dict(payload)
    if not isinstance(data, dict) or "reward" not in data:
        raise ValueError("reward.json must carry a 'reward' field")
    reward = data["reward"]
    if reward not in (0, 1, 0.0, 1.0):
        raise ValueError(f"reward {reward!r} is not binary; expected 0.0 or 1.0")
    return float(reward)


def is_pass(reward: float) -> bool:
    """Binary reward rule: ``1.0`` passes, anything else fails."""
    return reward == 1.0


def parse_image_digest(digest: str) -> str:
    """Validate a ``sha256:<64hex>`` task-image digest."""
    if not re.match(r"^sha256:[0-9a-f]{64}$", digest):
        raise ValueError(f"task image digest {digest!r} must match 'sha256:<64 lowercase hex>'")
    return digest


def image_reference(*, image_tag: str, image_digest: str | None) -> str:
    """Return the digest-pinned task-image reference.

    Without a digest this raises instead of trusting a mutable tag.
    """
    if not image_tag:
        raise ValueError("image_tag must be non-empty for provenance")
    if image_digest is None:
        raise ValueError(f"task image {image_tag!r} records no digest; refusing a mutable tag")
    parse_image_digest(image_digest)
    repository = image_tag.split(":")[0]
    return f"{repository}@{image_digest}"


def dispatch_request(task: TaskSpec) -> ModelRequest:
    """Return the dispatchable request for ``task``, proving gold stays out."""
    return task.safe_request()


def is_malformed(artifact: str) -> bool:
    """True when a delivered artifact carries no gradeable content."""
    return not artifact.strip()


def harbor_binary() -> str | None:
    """Absolute path of the ``harbor`` CLI, or ``None`` when absent."""
    return shutil.which("harbor")


def live_grade_status(purpose: str) -> dict[str, str | None]:
    """Report readiness for live harbor grading without raising."""
    if harbor_binary() is None:
        return {
            "state": "blocked_external",
            "runtime": None,
            "reason": f"no 'harbor' CLI on PATH for {purpose}",
        }
    return container_check_status(purpose)


def require_harbor_for_live_grade(purpose: str) -> str:
    """Return the container runtime for a live grade, or raise blocked."""
    if harbor_binary() is None:
        from stealthbench.sandbox.runtime import ContainerUnavailable

        raise ContainerUnavailable(f"blocked_external: no 'harbor' CLI on PATH for {purpose}")
    return require_container_runtime(purpose)


def assert_agent_verifier_separation(workdir: Path, verifier_dir: Path) -> None:
    """Require the agent workdir and verifier dir to be siblings, never nested."""
    assert_verifier_separation(workdir, verifier_dir)


def agent_container_args(
    *,
    runtime: str,
    image: str,
    workdir: Path,
    command: Sequence[str],
    limits: SandboxLimits,
) -> list[str]:
    """Locked-down agent-container argv: work mounted read-only, no verifier."""
    args = container_exec_args(
        runtime=runtime,
        image=image,
        workdir=workdir,
        command=command,
        limits=limits,
    )
    joined = " ".join(args).lower()
    if "verifier" in joined:
        raise ValueError("agent container argv must never reference the verifier")
    return args


def verifier_container_args(
    *,
    runtime: str,
    image: str,
    verifier_dir: Path,
    command: Sequence[str] = ("python", "/verifier/tests/test_scoring.py"),
    limits: SandboxLimits | None = None,
) -> list[str]:
    """Separate verifier-container argv: verifier mounted, agent never mounted."""
    active = limits or SandboxLimits(wall_seconds=VERIFIER_TIMEOUT_SECONDS)
    args = [
        runtime,
        "run",
        "--rm",
        "--network",
        "none",
        "--workdir",
        "/verifier",
        "--volume",
        f"{verifier_dir}:/verifier:ro",
    ]
    if active.memory_bytes is not None:
        args += ["--memory", f"{active.memory_bytes}b"]
    args.append(image)
    args.extend(command)
    return args


class TerminalBenchGrade(ResultModel):
    """One graded Terminal-Bench task with its recorded run conditions."""

    grade: GradeResult
    task_id: str = Field(min_length=1)
    reward: float = Field(ge=0.0, le=1.0)
    task_digest: str = Field(min_length=64, max_length=64)
    image_digest: str = Field(min_length=1)


class TerminalBenchSummary(ResultModel):
    """Binary pass rate over evaluated tasks only."""

    passed: int = Field(ge=0)
    eligible: int = Field(ge=0)
    pass_rate: float | None


def _graded_result(
    task: TaskSpec,
    *,
    reward: float,
    task_digest: str,
    image_digest: str,
    malformed: bool,
) -> TerminalBenchGrade:
    parse_task_digest(task_digest)
    parse_image_digest(image_digest)
    passed = is_pass(reward)
    correctness: StatusFlag = "pass" if passed else "fail"
    format_status: StatusFlag = "fail" if malformed else "pass"
    grade = GradeResult.model_validate(
        {
            "grader_version": GRADER_VERSION,
            "sample_key": task.sample_key.model_dump(mode="json"),
            "correctness": correctness,
            "format": format_status,
            "transport": "pass",
            "evaluator": "pass",
            "score_components": {"reward": reward},
            "denominator_eligibility": DenominatorEligibility(
                correctness=True,
                format=True,
                transport_success=True,
                evaluator_ran=True,
            ).model_dump(mode="json"),
        }
    )
    return TerminalBenchGrade(
        grade=grade,
        task_id=task.item_id,
        reward=reward,
        task_digest=task_digest,
        image_digest=image_digest,
    )


def _transport_grade(task: TaskSpec, *, transport: StatusFlag) -> TerminalBenchGrade:
    eligibility = DenominatorEligibility(
        correctness=False,
        format=False,
        transport_success=False,
        evaluator_ran=False,
    )
    grade = GradeResult.model_validate(
        {
            "grader_version": GRADER_VERSION,
            "sample_key": task.sample_key.model_dump(mode="json"),
            "correctness": "unavailable",
            "format": "unavailable",
            "transport": transport,
            "evaluator": "unavailable",
            "score_components": {},
            "denominator_eligibility": eligibility.model_dump(mode="json"),
        }
    )
    return TerminalBenchGrade(
        grade=grade,
        task_id=task.item_id,
        reward=0.0,
        task_digest="0" * 64,
        image_digest="sha256:" + "0" * 64,
    )


def _evaluator_failed_grade(task: TaskSpec, *, malformed: bool) -> TerminalBenchGrade:
    format_status: StatusFlag = "fail" if malformed else "pass"
    eligibility = DenominatorEligibility(
        correctness=False,
        format=True,
        transport_success=True,
        evaluator_ran=False,
    )
    grade = GradeResult.model_validate(
        {
            "grader_version": GRADER_VERSION,
            "sample_key": task.sample_key.model_dump(mode="json"),
            "correctness": "unavailable",
            "format": format_status,
            "transport": "pass",
            "evaluator": "fail",
            "score_components": {},
            "denominator_eligibility": eligibility.model_dump(mode="json"),
        }
    )
    return TerminalBenchGrade(
        grade=grade,
        task_id=task.item_id,
        reward=0.0,
        task_digest="0" * 64,
        image_digest="sha256:" + "0" * 64,
    )


def grade_accepted_sample(
    *,
    task: TaskSpec,
    artifact: str,
    reward: float,
    task_digest: str,
    image_digest: str,
) -> TerminalBenchGrade:
    """Grade one delivered artifact against its binary verifier reward.

    ``reward`` is the separate-verifier verdict (``1.0`` pass, ``0.0`` fail).
    Non-binary rewards raise instead of grading.
    """
    task.safe_request()
    if reward not in (0.0, 1.0):
        raise ValueError(f"reward {reward!r} is not binary; expected 0.0 or 1.0")
    return _graded_result(
        task,
        reward=float(reward),
        task_digest=task_digest,
        image_digest=image_digest,
        malformed=is_malformed(artifact),
    )


def grade_with_verifier_payload(
    *,
    task: TaskSpec,
    artifact: str,
    reward_text: str | None = None,
    reward_json: str | Mapping[str, Any] | None = None,
    task_digest: str,
    image_digest: str,
) -> TerminalBenchGrade:
    """Parse ``reward.txt``/``reward.json`` payloads, then grade.

    At least one payload is required; when both are given they must agree.
    A payload that does not parse is an evaluator failure, never a pass.
    """
    task.safe_request()
    if reward_text is None and reward_json is None:
        return _evaluator_failed_grade(task, malformed=is_malformed(artifact))
    try:
        rewards: list[float] = []
        if reward_text is not None:
            rewards.append(parse_reward_text(reward_text))
        if reward_json is not None:
            rewards.append(parse_reward_json(reward_json))
    except ValueError:
        return _evaluator_failed_grade(task, malformed=is_malformed(artifact))
    if len(rewards) == 2 and rewards[0] != rewards[1]:
        return _evaluator_failed_grade(task, malformed=is_malformed(artifact))
    return _graded_result(
        task,
        reward=rewards[0],
        task_digest=task_digest,
        image_digest=image_digest,
        malformed=is_malformed(artifact),
    )


def grade_generation(
    *,
    task: TaskSpec,
    generation: GenerationResult,
    reward: float | None = None,
    task_digest: str = "0" * 64,
    image_digest: str = "sha256:" + "0" * 64,
) -> TerminalBenchGrade:
    """Grade one delivery attempt (one trajectory is one attempt).

    * Sample-key mismatch raises rather than grading across tasks.
    * A non-accepted generation yields ``correctness="unavailable"``.
    * An accepted generation with ``reward=None`` (verifier never ran)
      yields ``evaluator="fail"`` with ``correctness="unavailable"``.
    """
    task.safe_request()
    if generation.sample_key != task.sample_key:
        raise ValueError(
            "generation sample_key "
            f"{generation.sample_key.model_dump()} does not match "
            f"task sample_key {task.sample_key.model_dump()}; refusing to grade"
        )
    if not generation.is_accepted_sample or generation.response is None:
        transport: StatusFlag
        if generation.delivery_status is DeliveryStatus.TRANSPORT_FAILED:
            transport = "fail"
        elif generation.delivery_status is DeliveryStatus.CANCELLED:
            transport = "invalid"
        else:
            transport = "unavailable"
        return _transport_grade(task, transport=transport)
    if reward is None:
        return _evaluator_failed_grade(task, malformed=is_malformed(generation.response))
    return grade_accepted_sample(
        task=task,
        artifact=generation.response,
        reward=reward,
        task_digest=task_digest,
        image_digest=image_digest,
    )


def summarize_grades(grades: Sequence[TerminalBenchGrade]) -> TerminalBenchSummary:
    """Aggregate binary pass rate over evaluated tasks only.

    Each trajectory counts once; two attempts for one task are two entries,
    never collapsed to the better one.
    """
    eligible = sum(1 for item in grades if item.grade.counts_toward_accuracy)
    passed = sum(
        1
        for item in grades
        if item.grade.counts_toward_accuracy and item.grade.correctness == "pass"
    )
    return TerminalBenchSummary(
        passed=passed,
        eligible=eligible,
        pass_rate=(passed / eligible) if eligible else None,
    )


def evaluator_layout_for_task(root: Path) -> EvaluatorLayout:
    """Create sibling ``work/`` + ``verifier/`` dirs for one agent task."""
    layout = EvaluatorLayout.create(root)
    layout.assert_separation()
    return layout


__all__ = [
    "AGENT_TIMEOUT_SECONDS",
    "GRADER_VERSION",
    "HARBOR_DATASET",
    "HARBOR_VERSION",
    "OFFICIAL_COMMAND",
    "OFFICIAL_VERIFIER_MODULE",
    "PINNED_COMMIT",
    "PINNED_TAG",
    "PINNED_TASK_COUNT",
    "VERIFIER_TIMEOUT_SECONDS",
    "TerminalBenchGrade",
    "TerminalBenchSummary",
    "agent_container_args",
    "assert_agent_verifier_separation",
    "dispatch_request",
    "evaluator_layout_for_task",
    "grade_accepted_sample",
    "grade_generation",
    "grade_with_verifier_payload",
    "harbor_binary",
    "image_reference",
    "is_malformed",
    "is_pass",
    "live_grade_status",
    "parse_image_digest",
    "parse_reward_json",
    "parse_reward_text",
    "parse_task_digest",
    "require_harbor_for_live_grade",
    "summarize_grades",
    "task_content_digest",
    "verifier_container_args",
    "verify_task_digest",
    "verify_task_set",
]
