"""Official LiveCodeBench wrapper: thin deterministic adapter over the pinned grader (G07 T07C).

Official source (see ``docs/upstream-inventory.md``):

* Repo ``https://github.com/LiveCodeBench/LiveCodeBench`` at pinned commit
  ``28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24`` (no tags exist).
* Dataset ``livecodebench/code_generation_lite``, split ``release_v6``
  (1055 items, May 2023 to Apr 2025; latest ``contest_date`` verified in
  ``test6.jsonl`` is 2025-03-29).
* Evaluator module ``lcb_runner/evaluation/compute_scores.py`` invoked as
  ``python -m lcb_runner.evaluation.compute_scores --eval_all_file <f>
  --start_date 2023-09-01``. The paper convention slices scores by date with
  ``--start_date`` / ``--end_date``; problems released after 2023-09-01 are
  the reported window.
* Default ``--timeout`` is 6 seconds. Upstream warns that time limits alone
  move pass@1 by more than 0.5 points, so the timeout is recorded per run
  and never tuned after seeing scores.
* ``ERRATA.md`` documents erroneous tests; those item IDs must be declared,
  not silently dropped.
* No dataset checksum verification exists upstream, so StealthBench records
  its own checksum over the selected item IDs.

Deliberate divergence (recorded in the inventory): generation defaults
upstream are n=10, temperature=0.2. pass@1 is reported from accepted samples
only; StealthBench never selects the best of several attempts and labels it
pass@1.

What this module does and does not do:

* It does **not** reimplement the vendored ``apps``-derived checker. The
  caller supplies the per-item test verdict (in production: the pinned
  official checker executed inside the G07 disposable sandbox backend); the
  adapter only maps the verdict onto the frozen ``GradeResult`` contract.
* Date-window filtering preserves requested item IDs in order. ERRATA
  exclusions require an explicit declared set; there is no silent drop.
* A timeout is a graded failure (``correctness="fail"``), never a missing
  sample. The ``timeout_seconds`` value is recorded verbatim on every grade.
* Transport failures are never reported as incorrect answers.
* Gold answers never enter a model request (``task.safe_request()`` first).

Sandbox integration is defensive: ``stealthbench.sandbox.runtime`` and
``stealthbench.sandbox.policy`` are owned by a sibling task (T07A/T07B) and
may be absent. When absent the adapter runs purely on injected verdicts and
reports ``SANDBOX_BACKEND_AVAILABLE`` as ``False``. When present the
``run_candidate_in_sandbox`` helper routes execution through it.

Offline by construction: pure functions over local strings and contracts
unless the caller explicitly passes a sandbox runner. No socket, no
credential.
"""

from __future__ import annotations

import hashlib
import importlib
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

#: Grader identity: the official module at the pinned inventory commit.
PINNED_LIVECODEBENCH_COMMIT: Final[str] = "28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24"
GRADER_VERSION: Final[str] = f"lcb_runner.evaluation.compute_scores@{PINNED_LIVECODEBENCH_COMMIT}"
OFFICIAL_EVALUATOR_MODULE: Final[str] = "lcb_runner/evaluation/compute_scores.py"
OFFICIAL_COMMAND: Final[str] = (
    "python -m lcb_runner.evaluation.compute_scores --eval_all_file <f> --start_date 2023-09-01"
)

#: Dataset identity from docs/upstream-inventory.md.
DATASET_ID: Final[str] = "livecodebench/code_generation_lite"
FULL_TEST_VARIANT: Final[str] = "livecodebench/code_generation"
SPLIT: Final[str] = "release_v6"
PINNED_ITEM_COUNT: Final[int] = 1055
DATE_WINDOW_START: Final[str] = "2023-05-01"
DATE_WINDOW_END: Final[str] = "2025-04-30"
REPORT_START_DATE: Final[str] = "2023-09-01"
LATEST_CONTEST_DATE: Final[str] = "2025-03-29"

#: Official defaults recorded per run, never tuned post-hoc.
DEFAULT_TIMEOUT_SECONDS: Final[float] = 6.0
DEFAULT_EVAL_PROCESSES: Final[int] = 12
DEFAULT_OPENAI_TIMEOUT_SECONDS: Final[float] = 90.0

#: Upstream generation defaults (recorded; pass@1 uses accepted samples only).
UPSTREAM_GENERATION_N: Final[int] = 10
UPSTREAM_GENERATION_TEMPERATURE: Final[float] = 0.2

#: Outcome of executing one candidate program against its hidden tests.
TestVerdict = Callable[[str], "TestOutcome"]
TestOutcome = str  # "pass" | "wrong_answer" | "timeout" | "runtime_error"

PASS: Final[str] = "pass"
WRONG_ANSWER: Final[str] = "wrong_answer"
TIMEOUT: Final[str] = "timeout"
RUNTIME_ERROR: Final[str] = "runtime_error"

VALID_OUTCOMES: Final[frozenset[str]] = frozenset({PASS, WRONG_ANSWER, TIMEOUT, RUNTIME_ERROR})


class LiveCodeBenchItem(ResultModel):
    """One LiveCodeBench problem held in memory.

    ``prompt`` is the only field that may enter a ``ModelRequest``.
    ``contest_date`` (``YYYY-MM-DD``) is the official release tag used for
    date-window slicing. ``gold_answer`` is absent: reference solutions stay
    evaluator-only and are never modeled here.
    """

    item_id: str = Field(min_length=1, max_length=256)
    prompt: str = Field(min_length=1)
    contest_date: str = Field(min_length=10, max_length=10)
    metadata: dict[str, str] = Field(default_factory=dict)


class LiveCodeBenchGrade(ResultModel):
    """One graded LiveCodeBench sample with its recorded run conditions.

    ``grade`` is the frozen contract verdict. ``timeout_seconds`` is the
    exact per-run timeout the execution used, recorded verbatim so a later
    reader can tell it was never tuned after seeing scores. ``contest_date``
    preserves the problem's release tag; ``outcome`` preserves the raw
    official verdict (correct vs subtly-wrong vs timeout vs runtime error).
    """

    grade: GradeResult
    timeout_seconds: float = Field(gt=0)
    contest_date: str | None = None
    outcome: str = Field(min_length=1)


class LiveCodeBenchSummary(ResultModel):
    """pass@1 over accepted samples only, with the recorded timeout."""

    passed: int = Field(ge=0)
    eligible: int = Field(ge=0)
    pass_at_1: float | None
    timeout_seconds: float = Field(gt=0)


def parse_contest_date(value: str) -> str:
    """Validate a ``YYYY-MM-DD`` contest date, returning it unchanged."""
    parts = value.split("-")
    if len(parts) != 3:
        raise ValueError(f"contest_date {value!r} must be YYYY-MM-DD")
    year, month, day = parts
    if not (len(year) == 4 and len(month) == 2 and len(day) == 2):
        raise ValueError(f"contest_date {value!r} must be YYYY-MM-DD")
    if not (year.isdigit() and month.isdigit() and day.isdigit()):
        raise ValueError(f"contest_date {value!r} must be YYYY-MM-DD")
    month_int = int(month)
    day_int = int(day)
    if not 1 <= month_int <= 12:
        raise ValueError(f"contest_date {value!r} has an invalid month")
    if not 1 <= day_int <= 31:
        raise ValueError(f"contest_date {value!r} has an invalid day")
    return value


def filter_by_date_window(
    items: Sequence[LiveCodeBenchItem],
    *,
    start_date: str,
    end_date: str,
) -> tuple[LiveCodeBenchItem, ...]:
    """Keep items whose ``contest_date`` lies in ``[start_date, end_date]``.

    Order is preserved: the output follows input order so requested IDs
    persist. Dates are ``YYYY-MM-DD`` strings compared lexicographically,
    which orders chronologically for zero-padded dates.
    """
    start = parse_contest_date(start_date)
    end = parse_contest_date(end_date)
    if start > end:
        raise ValueError(f"start_date {start!r} is after end_date {end!r}")
    return tuple(item for item in items if start <= parse_contest_date(item.contest_date) <= end)


def verify_requested_ids_preserved(
    *,
    requested_ids: Sequence[str],
    selected_ids: Sequence[str],
) -> None:
    """Raise unless every requested ID survived selection in order.

    Extra IDs are never invented and order is never shuffled: ``selected_ids``
    must equal ``requested_ids`` exactly. ERRATA handling removes IDs before
    this check via :func:`apply_errata_exclusions`, so any other loss raises
    instead of silently shrinking the denominator.
    """
    if tuple(selected_ids) != tuple(requested_ids):
        missing = [i for i in requested_ids if i not in set(selected_ids)]
        extra = [i for i in selected_ids if i not in set(requested_ids)]
        raise ValueError(f"requested item IDs were not preserved: missing={missing} extra={extra}")


def apply_errata_exclusions(
    items: Sequence[LiveCodeBenchItem],
    *,
    declared_excluded_ids: Sequence[str],
) -> tuple[tuple[LiveCodeBenchItem, ...], tuple[str, ...]]:
    """Split ``items`` into (kept, excluded) using an explicit ERRATA set.

    Upstream ``ERRATA.md`` documents erroneous tests; those item IDs must be
    declared, not silently dropped. Passing an empty set keeps everything and
    records that nothing was declared excluded. Only IDs present in the input
    may be declared: naming an unknown ID raises rather than recording a
    phantom exclusion.
    """
    declared = tuple(declared_excluded_ids)
    if len(set(declared)) != len(declared):
        raise ValueError(f"declared_excluded_ids contains duplicates: {declared}")
    known = {item.item_id for item in items}
    unknown = [i for i in declared if i not in known]
    if unknown:
        raise ValueError(f"ERRATA exclusion names unknown item IDs: {unknown}")
    excluded_set = set(declared)
    kept = tuple(item for item in items if item.item_id not in excluded_set)
    excluded = tuple(item.item_id for item in items if item.item_id in excluded_set)
    return kept, excluded


def selection_checksum(item_ids: Sequence[str]) -> str:
    """Record identity over a selection: sha256 of the ordered IDs.

    Upstream provides no dataset checksum, so StealthBench records its own.
    The order matters: the same set in a different order is a different
    selection and hashes differently.
    """
    digest = hashlib.sha256()
    for item_id in item_ids:
        digest.update(item_id.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def dispatch_request(task: TaskSpec) -> ModelRequest:
    """Return the dispatchable request for ``task``, proving gold stays out."""
    return task.safe_request()


def is_malformed(response: str) -> bool:
    """True when a delivered response carries no gradeable code."""
    return not response.strip()


def _check_outcome(outcome: str) -> str:
    if outcome not in VALID_OUTCOMES:
        raise ValueError(
            f"outcome {outcome!r} is not one of {sorted(VALID_OUTCOMES)}; "
            "refusing to map an unknown official verdict"
        )
    return outcome


def _graded_result(
    task: TaskSpec,
    *,
    outcome: str,
    timeout_seconds: float,
    contest_date: str | None,
    malformed: bool,
) -> LiveCodeBenchGrade:
    """Build a graded result: timeouts are failures, never missing data."""
    _check_outcome(outcome)
    if not timeout_seconds > 0:
        raise ValueError(f"timeout_seconds must be positive, got {timeout_seconds}")
    passed = outcome == PASS
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
    recorded_date: str | None = None
    if contest_date is not None:
        recorded_date = parse_contest_date(contest_date)
    return LiveCodeBenchGrade(
        grade=grade,
        timeout_seconds=timeout_seconds,
        contest_date=recorded_date,
        outcome=outcome,
    )


def _transport_grade(
    task: TaskSpec,
    *,
    transport: StatusFlag,
) -> LiveCodeBenchGrade:
    """Build a grade for a generation that never produced executable code."""
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
    return LiveCodeBenchGrade(
        grade=grade,
        timeout_seconds=DEFAULT_TIMEOUT_SECONDS,
        contest_date=None,
        outcome="not_executed",
    )


def _evaluator_failed_grade(task: TaskSpec, *, malformed: bool) -> LiveCodeBenchGrade:
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
    return LiveCodeBenchGrade(
        grade=grade,
        timeout_seconds=DEFAULT_TIMEOUT_SECONDS,
        contest_date=None,
        outcome="evaluator_error",
    )


def grade_accepted_sample(
    *,
    task: TaskSpec,
    response: str,
    outcome: str,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    contest_date: str | None = None,
) -> LiveCodeBenchGrade:
    """Grade one delivered program against its official test verdict.

    ``outcome`` is the official verdict (``pass`` / ``wrong_answer`` /
    ``timeout`` / ``runtime_error``) as produced by the pinned checker inside
    the sandbox. ``timeout_seconds`` is recorded verbatim and never tuned.
    A checker that raises is reported as an evaluator failure, never as an
    incorrect answer; use :func:`grade_generation` for transport handling.
    """
    task.safe_request()
    if callable(outcome):
        raise ValueError("outcome must be a verdict string, not a callable")
    checked = _check_outcome(outcome)
    return _graded_result(
        task,
        outcome=checked,
        timeout_seconds=timeout_seconds,
        contest_date=contest_date,
        malformed=is_malformed(response),
    )


def grade_with_checker(
    *,
    task: TaskSpec,
    response: str,
    check: Callable[[str], str],
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    contest_date: str | None = None,
) -> LiveCodeBenchGrade:
    """Apply an official ``check`` callable, mapping crashes to evaluator failure."""
    task.safe_request()
    malformed = is_malformed(response)
    try:
        outcome = check(response)
    except Exception:
        return _evaluator_failed_grade(task, malformed=malformed)
    return _graded_result(
        task,
        outcome=_check_outcome(outcome),
        timeout_seconds=timeout_seconds,
        contest_date=contest_date,
        malformed=malformed,
    )


def grade_generation(
    *,
    task: TaskSpec,
    generation: GenerationResult,
    outcome: str | None = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    contest_date: str | None = None,
) -> LiveCodeBenchGrade:
    """Grade one delivery attempt, keeping transport failures distinct.

    * Sample-key mismatch raises rather than grading across items.
    * A non-accepted generation yields ``correctness="unavailable"``.
    * An accepted generation with ``outcome=None`` (grader never ran) yields
      ``evaluator="fail"`` with ``correctness="unavailable"``.
    * Any concrete ``outcome`` -- including ``"timeout"`` -- is a graded
      failure in the denominator, never missing data.
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
    if outcome is None:
        return _evaluator_failed_grade(task, malformed=is_malformed(generation.response))
    return grade_accepted_sample(
        task=task,
        response=generation.response,
        outcome=outcome,
        timeout_seconds=timeout_seconds,
        contest_date=contest_date,
    )


def summarize_grades(
    grades: Sequence[LiveCodeBenchGrade],
    *,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> LiveCodeBenchSummary:
    """Aggregate pass@1 over accepted (graded) samples only.

    Only grades with ``counts_toward_accuracy`` enter the denominator, so
    transport failures and unevaluated samples are excluded rather than
    counted as incorrect. An empty denominator yields ``pass_at_1=None``:
    an unknown rate stays ``null``, never ``0.0``.
    """
    if not timeout_seconds > 0:
        raise ValueError(f"timeout_seconds must be positive, got {timeout_seconds}")
    eligible = sum(1 for item in grades if item.grade.counts_toward_accuracy)
    passed = sum(
        1
        for item in grades
        if item.grade.counts_toward_accuracy and item.grade.correctness == "pass"
    )
    return LiveCodeBenchSummary(
        passed=passed,
        eligible=eligible,
        pass_at_1=(passed / eligible) if eligible else None,
        timeout_seconds=timeout_seconds,
    )


__all__ = [
    "DATASET_ID",
    "DATE_WINDOW_END",
    "DATE_WINDOW_START",
    "DEFAULT_EVAL_PROCESSES",
    "DEFAULT_OPENAI_TIMEOUT_SECONDS",
    "DEFAULT_TIMEOUT_SECONDS",
    "FULL_TEST_VARIANT",
    "GRADER_VERSION",
    "LATEST_CONTEST_DATE",
    "OFFICIAL_COMMAND",
    "OFFICIAL_EVALUATOR_MODULE",
    "PASS",
    "PINNED_ITEM_COUNT",
    "PINNED_LIVECODEBENCH_COMMIT",
    "REPORT_START_DATE",
    "RUNTIME_ERROR",
    "SANDBOX_BACKEND_AVAILABLE",
    "SPLIT",
    "TIMEOUT",
    "UPSTREAM_GENERATION_N",
    "UPSTREAM_GENERATION_TEMPERATURE",
    "VALID_OUTCOMES",
    "WRONG_ANSWER",
    "LiveCodeBenchGrade",
    "LiveCodeBenchItem",
    "LiveCodeBenchSummary",
    "apply_errata_exclusions",
    "dispatch_request",
    "filter_by_date_window",
    "grade_accepted_sample",
    "grade_generation",
    "grade_with_checker",
    "is_malformed",
    "parse_contest_date",
    "selection_checksum",
    "summarize_grades",
    "verify_requested_ids_preserved",
]
