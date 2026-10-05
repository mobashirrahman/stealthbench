"""Official EvalPlus wrapper: thin deterministic adapter over the pinned grader (G07 T07D).

Official source (see ``docs/upstream-inventory.md``):

* Repo ``https://github.com/evalplus/evalplus``; dataset identity is pinned
  by data release rather than harness commit: HumanEval+ ``v0.1.10`` (164
  tasks) and MBPP+ ``v0.2.0`` (378 tasks). ``HUMANEVAL_PLUS_VERSION`` and
  ``MBPP_PLUS_VERSION`` are the values the official loader checksums and
  caches on.
* Evaluator module ``evalplus/eval/__init__.py`` (``untrusted_check``,
  ``estimate_pass_at_k``) invoked as ``evalplus.evaluate --dataset humaneval
  --samples <f>``.
* Per-task timeout ``T = max(T_base, T_gt * k)`` with ``T_base`` 4s and
  ``k`` 4 by default, where ``T_gt`` is the profiled ground-truth runtime.
* Memory is capped per process at ``min(4GB, system maximum)``;
  ``EVALPLUS_MAX_MEMORY_BYTES`` overrides, ``-1`` means unlimited.
* Timeouts and out-of-memory are graded as failures, not as missing data.
* No container is enforced by default; ``docs/execution.md`` recommends
  Docker and calls native execution unsafe.
* MD5 checksum verification of the dataset is present upstream, so dataset
  identity is verifiable.

Deliberate divergence (recorded in the inventory): execution runs inside
the disposable sandbox backend (G07) instead of the host process, with the
same timeout and memory policy recorded explicitly.

What this module does and does not do:

* It does **not** reimplement ``untrusted_check``. The caller supplies the
  extended-test verdict (in production: the pinned official checker executed
  inside the G07 disposable sandbox); the adapter maps it onto the frozen
  ``GradeResult`` contract and computes the unbiased ``pass@k`` estimator
  exactly as ``estimate_pass_at_k`` does.
* ``pass@1`` comes from accepted samples only. Selecting the best of several
  attempts and labelling it ``pass@1`` is refused by construction: the
  summary divides graded passes by the graded denominator.
* Timeouts and OOMs are ``correctness="fail"`` in the denominator, never
  ``"unavailable"``. Only transport failures (no delivered program) and
  unevaluated samples stay out of the denominator.
* Gold answers never enter a model request (``task.safe_request()`` first).

Sandbox integration is defensive: ``stealthbench.sandbox.runtime`` and
``stealthbench.sandbox.policy`` are owned by a sibling task (T07A/T07B) and
may be absent. When absent the adapter runs purely on injected verdicts and
reports ``SANDBOX_BACKEND_AVAILABLE`` as ``False``.

Offline by construction: pure functions over local values and contracts
unless the caller explicitly passes a sandbox runner. No socket, no
credential.
"""

from __future__ import annotations

import hashlib
import importlib
import math
from collections.abc import Callable, Sequence
from typing import Any, Final

from pydantic import Field

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


def _optional_sandbox_module(name: str) -> Any | None:
    """Import a sibling-owned sandbox module, returning ``None`` when absent.

    ``stealthbench.sandbox.runtime`` / ``policy`` are owned by T07A/T07B and
    may not exist yet. A dynamic import keeps this wrapper importable in both
    worlds: offline grading uses injected verdicts and reports
    ``SANDBOX_BACKEND_AVAILABLE`` as ``False`` until the backend lands.
    """
    try:
        return importlib.import_module(name)
    except ImportError:
        return None


_sandbox_policy: Any | None = _optional_sandbox_module("stealthbench.sandbox.policy")
_sandbox_runtime: Any | None = _optional_sandbox_module("stealthbench.sandbox.runtime")

#: True only when the sibling sandbox backend modules import cleanly.
SANDBOX_BACKEND_AVAILABLE: Final[bool] = (
    _sandbox_policy is not None and _sandbox_runtime is not None
)

#: Grader identity: the official module at the pinned dataset releases.
PINNED_HUMANEVAL_PLUS_VERSION: Final[str] = "v0.1.10"
PINNED_MBPP_PLUS_VERSION: Final[str] = "v0.2.0"
GRADER_VERSION: Final[str] = (
    f"evalplus.eval@humaneval-{PINNED_HUMANEVAL_PLUS_VERSION}+mbpp-{PINNED_MBPP_PLUS_VERSION}"
)
OFFICIAL_EVALUATOR_MODULE: Final[str] = "evalplus/eval/__init__.py"
OFFICIAL_COMMAND: Final[str] = "evalplus.evaluate --dataset humaneval --samples <f>"

#: Dataset identities from docs/upstream-inventory.md.
HUMANEVAL_PLUS_DATASET_ID: Final[str] = "evalplus/humanevalplus"
HUMANEVAL_PLUS_ITEM_COUNT: Final[int] = 164
MBPP_PLUS_DATASET_ID: Final[str] = "evalplus/mbppplus"
MBPP_PLUS_ITEM_COUNT: Final[int] = 378

#: Official timeout policy: T = max(T_base, T_gt * k).
DEFAULT_BASE_TIMEOUT_SECONDS: Final[float] = 4.0
DEFAULT_TIMEOUT_FACTOR: Final[float] = 4.0

#: Official memory policy: min(4GB, system maximum) unless overridden.
DEFAULT_MEMORY_CAP_BYTES: Final[int] = 4 * 1024 * 1024 * 1024
UNLIMITED_MEMORY_SENTINEL: Final[int] = -1


class EvalPlusGrade(ResultModel):
    """One graded EvalPlus sample with its recorded run conditions.

    ``grade`` is the frozen contract verdict. ``timeout_seconds`` and
    ``memory_cap_bytes`` record the exact per-run policy so a later reader
    can tell timeouts were never tuned post-hoc. ``timed_out`` and
    ``out_of_memory`` preserve why a failure failed; both are graded
    failures, never missing data.
    """

    grade: GradeResult
    timeout_seconds: float = Field(gt=0)
    memory_cap_bytes: int | None = None
    timed_out: bool = False
    out_of_memory: bool = False


class EvalPlusSummary(ResultModel):
    """pass@1 over accepted samples only, with the recorded run conditions."""

    passed: int = Field(ge=0)
    eligible: int = Field(ge=0)
    pass_at_1: float | None
    timeout_seconds: float = Field(gt=0)


def evalplus_timeout(
    ground_truth_runtime_seconds: float,
    *,
    base_seconds: float = DEFAULT_BASE_TIMEOUT_SECONDS,
    factor: float = DEFAULT_TIMEOUT_FACTOR,
) -> float:
    """Return the official per-task timeout ``T = max(T_base, T_gt * k)``."""
    if not base_seconds > 0:
        raise ValueError(f"base_seconds must be positive, got {base_seconds}")
    if not factor > 0:
        raise ValueError(f"factor must be positive, got {factor}")
    if not ground_truth_runtime_seconds >= 0:
        raise ValueError(
            f"ground_truth_runtime_seconds must be non-negative, got {ground_truth_runtime_seconds}"
        )
    return max(base_seconds, ground_truth_runtime_seconds * factor)


def resolve_memory_cap(
    *,
    system_maximum_bytes: int | None,
    override_bytes: int | None = None,
) -> int | None:
    """Resolve the official per-process memory cap.

    Default: ``min(4GB, system maximum)``. ``override_bytes`` mirrors
    ``EVALPLUS_MAX_MEMORY_BYTES``: a non-negative value wins, ``-1`` means
    unlimited (``None``). A system maximum of ``None`` means the host limit
    is unknown, so the default 4GB cap applies.
    """
    if override_bytes is not None:
        if override_bytes == UNLIMITED_MEMORY_SENTINEL:
            return None
        if override_bytes < 0:
            raise ValueError(f"override_bytes must be non-negative or -1, got {override_bytes}")
        return override_bytes
    if system_maximum_bytes is not None:
        if system_maximum_bytes <= 0:
            raise ValueError(f"system_maximum_bytes must be positive, got {system_maximum_bytes}")
        return min(DEFAULT_MEMORY_CAP_BYTES, system_maximum_bytes)
    return DEFAULT_MEMORY_CAP_BYTES


def verify_md5(payload: bytes, *, expected_md5: str) -> str:
    """Return the payload md5, or raise when it differs from ``expected_md5``.

    Mirrors the upstream dataset checksum gate: identity is verified, never
    assumed. Comparison is case-insensitive hex.
    """
    actual = hashlib.md5(payload, usedforsecurity=False).hexdigest()
    if actual.lower() != expected_md5.lower():
        raise ValueError(
            f"md5 mismatch: payload hashes to {actual} "
            f"but the dataset records {expected_md5}; refusing to run"
        )
    return actual


def estimate_pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased ``pass@k`` estimator ``1 - C(n-c, k) / C(n, k)``.

    Exactly the official ``estimate_pass_at_k`` formula. With the default
    single greedy sample (``n=1, k=1``) it reduces to single-sample accuracy.
    ``c > n`` or non-positive ``n``/``k`` raise rather than inventing a rate.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if k > n:
        raise ValueError(f"k ({k}) cannot exceed n ({n})")
    if not 0 <= c <= n:
        raise ValueError(f"c ({c}) must satisfy 0 <= c <= n ({n})")
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def dispatch_request(task: TaskSpec) -> ModelRequest:
    """Return the dispatchable request for ``task``, proving gold stays out."""
    return task.safe_request()


def is_malformed(response: str) -> bool:
    """True when a delivered response carries no gradeable code."""
    return not response.strip()


def _graded_evalplus_result(
    task: TaskSpec,
    *,
    passed: bool,
    timed_out: bool,
    out_of_memory: bool,
    timeout_seconds: float,
    memory_cap_bytes: int | None,
    malformed: bool,
) -> EvalPlusGrade:
    """Build a graded result: timeouts/OOM are failures, never missing data."""
    if not timeout_seconds > 0:
        raise ValueError(f"timeout_seconds must be positive, got {timeout_seconds}")
    if memory_cap_bytes is not None and memory_cap_bytes < 0:
        raise ValueError(f"memory_cap_bytes must be non-negative or None, got {memory_cap_bytes}")
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
            "score_components": {"pass_at_1": 1.0 if passed else 0.0},
            "denominator_eligibility": DenominatorEligibility(
                correctness=True,
                format=True,
                transport_success=True,
                evaluator_ran=True,
            ).model_dump(mode="json"),
        }
    )
    return EvalPlusGrade(
        grade=grade,
        timeout_seconds=timeout_seconds,
        memory_cap_bytes=memory_cap_bytes,
        timed_out=timed_out,
        out_of_memory=out_of_memory,
    )


def _transport_evalplus_grade(
    task: TaskSpec,
    *,
    transport: StatusFlag,
    timeout_seconds: float,
) -> EvalPlusGrade:
    """Build a grade for a generation that never delivered a program."""
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
    return EvalPlusGrade(
        grade=grade,
        timeout_seconds=timeout_seconds,
        memory_cap_bytes=None,
        timed_out=False,
        out_of_memory=False,
    )


def _evaluator_failed_evalplus_grade(
    task: TaskSpec,
    *,
    malformed: bool,
    timeout_seconds: float,
) -> EvalPlusGrade:
    """Build a grade for when the official checker could not run to a verdict."""
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
    return EvalPlusGrade(
        grade=grade,
        timeout_seconds=timeout_seconds,
        memory_cap_bytes=None,
        timed_out=False,
        out_of_memory=False,
    )


def grade_accepted_sample(
    *,
    task: TaskSpec,
    response: str,
    passed: bool,
    timed_out: bool = False,
    out_of_memory: bool = False,
    timeout_seconds: float = DEFAULT_BASE_TIMEOUT_SECONDS,
    memory_cap_bytes: int | None = DEFAULT_MEMORY_CAP_BYTES,
) -> EvalPlusGrade:
    """Grade one delivered program against its extended-test verdict.

    ``passed`` is the official extended-test outcome: every hidden test
    passed within the recorded timeout and memory cap. ``timed_out`` and
    ``out_of_memory`` are recorded explicitly but never excuse the sample:
    a timeout with ``passed=False`` is ``correctness="fail"`` in the
    denominator, not ``"unavailable"``. A sample that both passed and timed
    out is contradictory and raises.
    """
    task.safe_request()
    if passed and (timed_out or out_of_memory):
        raise ValueError(
            "passed=True contradicts timed_out/out_of_memory=True; "
            "a passing run completed within its limits"
        )
    return _graded_evalplus_result(
        task,
        passed=passed,
        timed_out=timed_out,
        out_of_memory=out_of_memory,
        timeout_seconds=timeout_seconds,
        memory_cap_bytes=memory_cap_bytes,
        malformed=is_malformed(response),
    )


def grade_with_checker(
    *,
    task: TaskSpec,
    response: str,
    check: Callable[[str], Any],
    timeout_seconds: float = DEFAULT_BASE_TIMEOUT_SECONDS,
    memory_cap_bytes: int | None = DEFAULT_MEMORY_CAP_BYTES,
) -> EvalPlusGrade:
    """Apply an official extended-test ``check`` returning ``(passed, timed_out, oom)``.

    ``check`` is the injected official ``untrusted_check`` stand-in. A check
    that raises is an evaluator failure, never an incorrect answer.
    """
    task.safe_request()
    malformed = is_malformed(response)
    if not callable(check):
        raise ValueError("check must be callable")
    try:
        verdict = check(response)
    except Exception:
        return _evaluator_failed_evalplus_grade(
            task, malformed=malformed, timeout_seconds=timeout_seconds
        )
    if not isinstance(verdict, tuple) or len(verdict) != 3:
        return _evaluator_failed_evalplus_grade(
            task, malformed=malformed, timeout_seconds=timeout_seconds
        )
    passed_raw, timed_out_raw, oom_raw = verdict
    if not isinstance(passed_raw, bool):
        return _evaluator_failed_evalplus_grade(
            task, malformed=malformed, timeout_seconds=timeout_seconds
        )
    if not isinstance(timed_out_raw, bool) or not isinstance(oom_raw, bool):
        return _evaluator_failed_evalplus_grade(
            task, malformed=malformed, timeout_seconds=timeout_seconds
        )
    if passed_raw and (timed_out_raw or oom_raw):
        return _evaluator_failed_evalplus_grade(
            task, malformed=malformed, timeout_seconds=timeout_seconds
        )
    return _graded_evalplus_result(
        task,
        passed=passed_raw,
        timed_out=timed_out_raw,
        out_of_memory=oom_raw,
        timeout_seconds=timeout_seconds,
        memory_cap_bytes=memory_cap_bytes,
        malformed=malformed,
    )


def grade_generation(
    *,
    task: TaskSpec,
    generation: GenerationResult,
    passed: bool | None = None,
    timed_out: bool = False,
    out_of_memory: bool = False,
    timeout_seconds: float = DEFAULT_BASE_TIMEOUT_SECONDS,
    memory_cap_bytes: int | None = DEFAULT_MEMORY_CAP_BYTES,
) -> EvalPlusGrade:
    """Grade one delivery attempt, keeping transport failures distinct.

    * Sample-key mismatch raises rather than grading across tasks.
    * A non-accepted generation yields ``correctness="unavailable"``.
    * An accepted generation with ``passed=None`` (grader never ran) yields
      ``evaluator="fail"`` with ``correctness="unavailable"``.
    * Any concrete ``passed`` verdict -- including timeouts and OOMs -- is a
      graded failure in the denominator, never missing data.
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
        return _transport_evalplus_grade(task, transport=transport, timeout_seconds=timeout_seconds)
    if passed is None:
        return _evaluator_failed_evalplus_grade(
            task,
            malformed=is_malformed(generation.response),
            timeout_seconds=timeout_seconds,
        )
    if passed and (timed_out or out_of_memory):
        raise ValueError(
            "passed=True contradicts timed_out/out_of_memory=True; "
            "a passing run completed within its limits"
        )
    return grade_accepted_sample(
        task=task,
        response=generation.response,
        passed=passed,
        timed_out=timed_out,
        out_of_memory=out_of_memory,
        timeout_seconds=timeout_seconds,
        memory_cap_bytes=memory_cap_bytes,
    )


def summarize_grades(
    grades: Sequence[EvalPlusGrade],
    *,
    timeout_seconds: float = DEFAULT_BASE_TIMEOUT_SECONDS,
) -> EvalPlusSummary:
    """Aggregate pass@1 over accepted (graded) samples only.

    Only grades with ``counts_toward_accuracy`` enter the denominator, so
    transport failures and unevaluated samples are excluded rather than
    counted as incorrect. Timeouts and OOMs stay in the denominator as
    failures. An empty denominator yields ``pass_at_1=None``: an unknown
    rate stays ``null``, never ``0.0``.
    """
    if not timeout_seconds > 0:
        raise ValueError(f"timeout_seconds must be positive, got {timeout_seconds}")
    eligible = sum(1 for item in grades if item.grade.counts_toward_accuracy)
    passed = sum(
        1
        for item in grades
        if item.grade.counts_toward_accuracy and item.grade.correctness == "pass"
    )
    return EvalPlusSummary(
        passed=passed,
        eligible=eligible,
        pass_at_1=(passed / eligible) if eligible else None,
        timeout_seconds=timeout_seconds,
    )


__all__ = [
    "DEFAULT_BASE_TIMEOUT_SECONDS",
    "DEFAULT_MEMORY_CAP_BYTES",
    "DEFAULT_TIMEOUT_FACTOR",
    "GRADER_VERSION",
    "HUMANEVAL_PLUS_DATASET_ID",
    "HUMANEVAL_PLUS_ITEM_COUNT",
    "MBPP_PLUS_DATASET_ID",
    "MBPP_PLUS_ITEM_COUNT",
    "OFFICIAL_COMMAND",
    "OFFICIAL_EVALUATOR_MODULE",
    "PINNED_HUMANEVAL_PLUS_VERSION",
    "PINNED_MBPP_PLUS_VERSION",
    "SANDBOX_BACKEND_AVAILABLE",
    "UNLIMITED_MEMORY_SENTINEL",
    "EvalPlusGrade",
    "EvalPlusSummary",
    "dispatch_request",
    "estimate_pass_at_k",
    "evalplus_timeout",
    "grade_accepted_sample",
    "grade_generation",
    "grade_with_checker",
    "is_malformed",
    "resolve_memory_cap",
    "summarize_grades",
    "verify_md5",
]
