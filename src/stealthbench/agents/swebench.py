"""Official SWE-bench Verified wrapper: thin deterministic adapter (G10 T10B).

Official source (see ``docs/upstream-inventory.md``):

* Repo ``https://github.com/SWE-bench/SWE-bench`` at pinned harness commit
  ``02e7a74ffd0b707aab73d203fe87bdc7c76afc8e``.
* Dataset ``SWE-bench/SWE-bench_Verified`` (500 instances, released
  2024-08-13), superseding both the 2,294-item test set and Lite.
* Evaluator module ``swebench/harness/grading.py`` via
  ``swebench.harness.run_evaluation``, invoked as
  ``swebench eval verified -p <predictions> --run-id <id> -j <n>``.
* ``RESOLVED_FULL`` requires every ``FAIL_TO_PASS`` and every
  ``PASS_TO_PASS`` test to pass. Six JavaScript repositories use fail-only
  grading because their reporter only records failing tests.
* The harness caches verdicts by ``run_id`` + ``instance_id`` only: regrading
  a changed patch requires a **new** ``run_id``, otherwise a stale verdict is
  silently reused.
* Image tags end in ``:latest`` (mutable), so image identity must be recorded
  as a digest at run time.
* Docker is mandatory (x86_64 host, ~120GB storage, 16GB RAM, 8 cores).

Deliberate divergence (recorded in the inventory): the harness never calls a
model API; it grades patches produced elsewhere. StealthBench runs one
complete agent trajectory per instance and treats that trajectory as a
single attempt (``docs/contracts.md``).

What this module does and does not do:

* It does **not** reimplement ``grading.py``. The caller supplies the
  per-test verdicts (in production: the pinned harness executed in a fresh
  container per task); the adapter only applies the official
  ``RESOLVED_FULL`` rule and maps the verdict onto the frozen ``GradeResult``
  contract.
* Image identity is digest-pinned: any grade without a ``sha256:`` digest
  raises instead of recording a mutable ``:latest`` tag.
* Re-grading a changed patch under the same ``run_id`` raises instead of
  reusing a stale verdict. Fresh runs use :func:`new_run_id`.
* A failed evaluation is not a crash: harness errors are reported as
  evaluator failures, never as resolved/unresolved.
* Gold patches and hidden tests never enter a model request
  (``task.safe_request()`` first).

Offline by construction: pure functions over local verdicts and contracts.
No socket, no credential. Live grading requires a container runtime and the
``swebench==5.0.2`` environment; without one the live helpers report
``blocked_external`` instead of fake-passing.
"""

from __future__ import annotations

import hashlib
import re
import uuid
from collections.abc import Mapping, Sequence
from typing import Final

from pydantic import Field

from stealthbench.sandbox.runtime import (
    container_check_status,
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

#: Grader identity: the official module at the pinned inventory commit.
PINNED_HARNESS_COMMIT: Final[str] = "02e7a74ffd0b707aab73d203fe87bdc7c76afc8e"
GRADER_VERSION: Final[str] = f"swebench.harness.grading@{PINNED_HARNESS_COMMIT}"
OFFICIAL_EVALUATOR_MODULE: Final[str] = "swebench/harness/grading.py"
OFFICIAL_COMMAND: Final[str] = "swebench eval verified -p <predictions> --run-id <id> -j <n>"

#: Dataset identity from docs/upstream-inventory.md.
PINNED_DATASET_ID: Final[str] = "SWE-bench/SWE-bench_Verified"
PINNED_DATASET_SPLIT: Final[str] = "test"
PINNED_ITEM_COUNT: Final[int] = 500
PINNED_RELEASE_DATE: Final[str] = "2024-08-13"

#: Official image pattern; the trailing ``:latest`` is mutable by design.
IMAGE_PATTERN: Final[str] = "swebench/sweb.eval.<arch>.<repo>_<version>_<instance_id>:latest"

#: Pinned installer version carrying the harness.
PINNED_SWEBENCH_VERSION: Final[str] = "swebench==5.0.2"

_IMAGE_DIGEST_PATTERN: Final[re.Pattern[str]] = re.compile(r"^sha256:[0-9a-f]{64}$")


def parse_image_digest(digest: str) -> str:
    """Validate a ``sha256:<64hex>`` image digest, returning it unchanged."""
    if not _IMAGE_DIGEST_PATTERN.match(digest):
        raise ValueError(
            f"image digest {digest!r} must match 'sha256:<64 lowercase hex>'; "
            "a mutable ':latest' tag is not image identity"
        )
    return digest


def image_reference(*, image_tag: str, image_digest: str | None) -> str:
    """Return the digest-pinned image reference for a task environment.

    ``image_tag`` is recorded for provenance but never trusted as identity:
    without a digest this raises instead of pinning a mutable ``:latest``.
    """
    if not image_tag:
        raise ValueError("image_tag must be non-empty for provenance")
    if image_digest is None:
        raise ValueError(f"image {image_tag!r} records no digest; refusing a mutable ':latest' tag")
    parse_image_digest(image_digest)
    repository = image_tag.split(":")[0]
    return f"{repository}@{image_digest}"


def patch_hash(patch: str) -> str:
    """Stable sha256 identity of a patch, for run-id reuse detection."""
    return hashlib.sha256(patch.encode("utf-8")).hexdigest()


def new_run_id() -> str:
    """Return a fresh harness ``run_id`` for one grading run.

    Fresh per regrade: the harness caches by ``run_id`` + ``instance_id``,
    so a changed patch under an old id would silently reuse the old verdict.
    """
    return f"stealthbench-{uuid.uuid4().hex}"


class RunCache:
    """Detect stale-verdict reuse: one ``run_id`` grades one patch per instance.

    The harness keys its cache on ``(run_id, instance_id)`` only. Registering
    the same pair with a different patch hash raises instead of returning
    the cached verdict.
    """

    def __init__(self) -> None:
        self._entries: dict[tuple[str, str], str] = {}

    def register(self, *, run_id: str, instance_id: str, patch: str) -> None:
        """Record ``(run_id, instance_id)`` grading ``patch``."""
        if not run_id:
            raise ValueError("run_id must be non-empty")
        if not instance_id:
            raise ValueError("instance_id must be non-empty")
        digest = patch_hash(patch)
        key = (run_id, instance_id)
        previous = self._entries.get(key)
        if previous is not None and previous != digest:
            raise ValueError(
                f"run_id {run_id!r} for instance {instance_id!r} already graded a "
                "different patch; use new_run_id() for any regrade"
            )
        self._entries[key] = digest

    def __len__(self) -> int:
        return len(self._entries)


def is_resolved(
    fail_to_pass: Mapping[str, bool],
    pass_to_pass: Mapping[str, bool],
    *,
    fail_only: bool = False,
) -> bool:
    """Apply the official ``RESOLVED_FULL`` rule to per-test verdicts.

    ``True`` only when every ``FAIL_TO_PASS`` test passed and -- unless
    ``fail_only`` (the six JavaScript repositories whose reporter only
    records failures) -- every ``PASS_TO_PASS`` test passed. An empty
    ``FAIL_TO_PASS`` set raises: resolution without a failing-then-passing
    test is not evidence.
    """
    if not fail_to_pass:
        raise ValueError("FAIL_TO_PASS must be non-empty; refusing vacuous resolution")
    if not all(fail_to_pass.values()):
        return False
    if fail_only:
        return True
    return all(pass_to_pass.values())


def dispatch_request(task: TaskSpec) -> ModelRequest:
    """Return the dispatchable request for ``task``, proving gold stays out."""
    return task.safe_request()


def is_malformed(patch: str) -> bool:
    """True when a delivered patch carries no gradeable diff."""
    return not patch.strip()


class SweBenchGrade(ResultModel):
    """One graded SWE-bench instance with its recorded run conditions.

    ``grade`` is the frozen contract verdict. ``run_id``, ``image_digest``
    and the per-test maps preserve exactly what the official harness saw so
    a later reader can tell a regrade used a fresh id and a pinned image.
    """

    grade: GradeResult
    instance_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    image_digest: str = Field(min_length=1)
    fail_to_pass: dict[str, bool] = Field(default_factory=dict)
    pass_to_pass: dict[str, bool] = Field(default_factory=dict)
    resolved: bool = False
    fail_only: bool = False


class SweBenchSummary(ResultModel):
    """Resolution rate over graded (evaluated) samples only."""

    resolved: int = Field(ge=0)
    eligible: int = Field(ge=0)
    resolution_rate: float | None


def _graded_result(
    task: TaskSpec,
    *,
    resolved: bool,
    run_id: str,
    image_digest: str,
    fail_to_pass: Mapping[str, bool],
    pass_to_pass: Mapping[str, bool],
    fail_only: bool,
    malformed: bool,
) -> SweBenchGrade:
    parse_image_digest(image_digest)
    correctness: StatusFlag = "pass" if resolved else "fail"
    format_status: StatusFlag = "fail" if malformed else "pass"
    grade = GradeResult.model_validate(
        {
            "grader_version": GRADER_VERSION,
            "sample_key": task.sample_key.model_dump(mode="json"),
            "correctness": correctness,
            "format": format_status,
            "transport": "pass",
            "evaluator": "pass",
            "score_components": {"resolved": 1.0 if resolved else 0.0},
            "denominator_eligibility": DenominatorEligibility(
                correctness=True,
                format=True,
                transport_success=True,
                evaluator_ran=True,
            ).model_dump(mode="json"),
        }
    )
    return SweBenchGrade(
        grade=grade,
        instance_id=task.item_id,
        run_id=run_id,
        image_digest=image_digest,
        fail_to_pass=dict(fail_to_pass),
        pass_to_pass=dict(pass_to_pass),
        resolved=resolved,
        fail_only=fail_only,
    )


def _transport_grade(task: TaskSpec, *, transport: StatusFlag) -> SweBenchGrade:
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
    return SweBenchGrade(
        grade=grade,
        instance_id=task.item_id,
        run_id="ungraded",
        image_digest="sha256:" + "0" * 64,
        fail_to_pass={},
        pass_to_pass={},
        resolved=False,
        fail_only=False,
    )


def _evaluator_failed_grade(task: TaskSpec, *, malformed: bool) -> SweBenchGrade:
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
    return SweBenchGrade(
        grade=grade,
        instance_id=task.item_id,
        run_id="evaluator-failed",
        image_digest="sha256:" + "0" * 64,
        fail_to_pass={},
        pass_to_pass={},
        resolved=False,
        fail_only=False,
    )


def grade_accepted_sample(
    *,
    task: TaskSpec,
    patch: str,
    fail_to_pass: Mapping[str, bool],
    pass_to_pass: Mapping[str, bool],
    run_id: str,
    image_digest: str,
    fail_only: bool = False,
    cache: RunCache | None = None,
) -> SweBenchGrade:
    """Grade one delivered patch against its official per-test verdicts.

    ``fail_to_pass`` / ``pass_to_pass`` are the harness verdicts (in
    production: from the pinned harness in a fresh container). ``run_id``
    must be fresh for a changed patch: when ``cache`` is given the pair is
    registered and a reused id with a different patch raises.
    """
    task.safe_request()
    if not run_id:
        raise ValueError("run_id must be non-empty; use new_run_id() per grading run")
    if cache is not None:
        cache.register(run_id=run_id, instance_id=task.item_id, patch=patch)
    resolved = is_resolved(fail_to_pass, pass_to_pass, fail_only=fail_only)
    return _graded_result(
        task,
        resolved=resolved,
        run_id=run_id,
        image_digest=image_digest,
        fail_to_pass=fail_to_pass,
        pass_to_pass=pass_to_pass,
        fail_only=fail_only,
        malformed=is_malformed(patch),
    )


def grade_generation(
    *,
    task: TaskSpec,
    generation: GenerationResult,
    fail_to_pass: Mapping[str, bool] | None = None,
    pass_to_pass: Mapping[str, bool] | None = None,
    run_id: str = "ungraded",
    image_digest: str = "sha256:" + "0" * 64,
    fail_only: bool = False,
    cache: RunCache | None = None,
) -> SweBenchGrade:
    """Grade one delivery attempt (one trajectory is one attempt).

    * Sample-key mismatch raises rather than grading across instances.
    * A non-accepted generation yields ``correctness="unavailable"``.
    * An accepted generation without verdicts (grader never ran) yields
      ``evaluator="fail"`` with ``correctness="unavailable"``.
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
    if fail_to_pass is None or pass_to_pass is None:
        return _evaluator_failed_grade(task, malformed=is_malformed(generation.response))
    return grade_accepted_sample(
        task=task,
        patch=generation.response,
        fail_to_pass=fail_to_pass,
        pass_to_pass=pass_to_pass,
        run_id=run_id,
        image_digest=image_digest,
        fail_only=fail_only,
        cache=cache,
    )


def summarize_grades(grades: Sequence[SweBenchGrade]) -> SweBenchSummary:
    """Aggregate resolution rate over evaluated samples only.

    Only grades with ``counts_toward_accuracy`` enter the denominator, so
    transport failures and unevaluated samples are excluded rather than
    counted as unresolved. Each trajectory counts once: two attempts for one
    task are two entries, never best-of-N relabelled as one.
    """
    eligible = sum(1 for item in grades if item.grade.counts_toward_accuracy)
    resolved = sum(1 for item in grades if item.grade.counts_toward_accuracy and item.resolved)
    return SweBenchSummary(
        resolved=resolved,
        eligible=eligible,
        resolution_rate=(resolved / eligible) if eligible else None,
    )


def live_grade_status(purpose: str) -> dict[str, str | None]:
    """Report readiness for live harness grading without raising."""
    return container_check_status(purpose)


def require_container_for_live_grade(purpose: str) -> str:
    """Return the container runtime or raise with a ``blocked_external`` reason."""
    return require_container_runtime(purpose)


__all__ = [
    "GRADER_VERSION",
    "IMAGE_PATTERN",
    "OFFICIAL_COMMAND",
    "OFFICIAL_EVALUATOR_MODULE",
    "PINNED_DATASET_ID",
    "PINNED_DATASET_SPLIT",
    "PINNED_HARNESS_COMMIT",
    "PINNED_ITEM_COUNT",
    "PINNED_RELEASE_DATE",
    "PINNED_SWEBENCH_VERSION",
    "RunCache",
    "SweBenchGrade",
    "SweBenchSummary",
    "dispatch_request",
    "grade_accepted_sample",
    "grade_generation",
    "image_reference",
    "is_malformed",
    "is_resolved",
    "live_grade_status",
    "new_run_id",
    "parse_image_digest",
    "patch_hash",
    "require_container_for_live_grade",
    "summarize_grades",
]
