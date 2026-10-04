"""Official IFEval wrapper: thin deterministic adapter over the pinned scorer (G05 T05B).

Official source (see ``docs/upstream-inventory.md``):

* Evaluator module: ``instruction_following_eval/evaluation_main.py`` at pinned
  commit ``e49bbfe381c9c0e564b937f1c4e163a2273c65cc`` (no tags exist).
* The official tool emits ``eval_results_strict.jsonl`` and
  ``eval_results_loose.jsonl`` and prints both accuracies.
* Loose scoring accepts any of 7 response variants: the first line removed, the
  last line removed, both removed, each with and without ``*`` stripped, plus
  the ``*``-stripped original. Strict scoring grades the raw response only.
* Response language is detected with ``langdetect``, so grading is not purely
  lexical. This adapter does not reimplement language detection: language
  sensitivity lives inside the injected official check, never in the variant
  expansion below (which is pure text manipulation).

Deliberate divergence (recorded in the inventory): the official checker is
reused unchanged. StealthBench only supplies the request/response pairing and
reports strict and loose metrics separately.

What this module does and does not do:

* It does **not** reimplement any of the 29 instruction checkers. The caller
  supplies the official ``check`` callable for the item; the adapter only
  expands the 7 loose variants, applies the check deterministically, and maps
  the verdicts onto the frozen ``GradeResult`` contract.
* Strict and loose are returned as two separate ``GradeResult`` objects
  (``IfEvalGradePair``). They are never merged, averaged, or combined.
* Transport failures are never reported as incorrect answers: a generation
  that was never accepted yields ``correctness="unavailable"`` on both sides.
* Malformed (empty/whitespace-only) responses are delivered answers, so they
  stay in the denominator: ``correctness="fail"`` **and** ``format="fail"``.
* Gold answers never enter a model request. Grading calls
  ``task.safe_request()`` first, so a task whose prompt carries evaluator-only
  data raises instead of grading.

Offline by construction: pure functions over local strings and contracts. No
transport, no socket, no subprocess, no credential.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Final

from stealthbench.benchmarks.datasets import PINNED_IFEVAL_EVALUATOR_REVISION
from stealthbench.schemas.results import (
    DeliveryStatus,
    DenominatorEligibility,
    GenerationResult,
    GradeResult,
    ModelRequest,
    ResultModel,
    SampleKey,
    StatusFlag,
    TaskSpec,
)

#: Grader identity: the official module at the pinned inventory commit.
GRADER_VERSION: Final[str] = f"instruction_following_eval@{PINNED_IFEVAL_EVALUATOR_REVISION}"
OFFICIAL_EVALUATOR_MODULE: Final[str] = "instruction_following_eval/evaluation_main.py"
OFFICIAL_STRICT_FILENAME: Final[str] = "eval_results_strict.jsonl"
OFFICIAL_LOOSE_FILENAME: Final[str] = "eval_results_loose.jsonl"

#: The 7 official loose variants, in a fixed order. ``loose_variants`` returns
#: them positionally in this order.
LOOSE_VARIANT_NAMES: Final[tuple[str, ...]] = (
    "strip_asterisks",
    "remove_first_line",
    "remove_first_line_strip_asterisks",
    "remove_last_line",
    "remove_last_line_strip_asterisks",
    "remove_first_and_last_lines",
    "remove_first_and_last_lines_strip_asterisks",
)

#: The official per-item verdict function. Supplied by the caller (in
#: production: the pinned official checker); never reimplemented here.
InstructionCheck = Callable[[str], bool]


def _strip_asterisks(response: str) -> str:
    """Remove ``*`` characters (markdown bold markers), mirroring the official loose path."""
    return response.replace("*", "")


def _remove_first_line(response: str) -> str:
    lines = response.split("\n")
    return "\n".join(lines[1:])


def _remove_last_line(response: str) -> str:
    lines = response.split("\n")
    return "\n".join(lines[:-1])


def _remove_first_and_last_lines(response: str) -> str:
    lines = response.split("\n")
    return "\n".join(lines[1:-1])


def loose_variants(response: str) -> tuple[str, ...]:
    """Return the 7 official loose transforms of ``response``, in documented order.

    Pure and deterministic: the same input always yields the same 7 outputs.
    Language-agnostic by design; ``langdetect`` behaviour stays inside the
    injected official check.
    """
    no_first = _remove_first_line(response)
    no_last = _remove_last_line(response)
    no_first_last = _remove_first_and_last_lines(response)
    return (
        _strip_asterisks(response),
        no_first,
        _strip_asterisks(no_first),
        no_last,
        _strip_asterisks(no_last),
        no_first_last,
        _strip_asterisks(no_first_last),
    )


def is_malformed(response: str) -> bool:
    """True when a delivered response carries no gradeable content."""
    return not response.strip()


def dispatch_request(task: TaskSpec) -> ModelRequest:
    """Return the dispatchable request for ``task``, proving gold stays out.

    Re-verifies the recorded prompt hash and asserts evaluator-only data is
    absent; raises instead of returning a request that would leak gold.
    """
    return task.safe_request()


def evaluate_strict(response: str, check: InstructionCheck) -> bool:
    """Apply the official ``check`` to the raw response (strict path)."""
    return bool(check(response))


def evaluate_loose(response: str, check: InstructionCheck) -> bool:
    """Apply the official ``check`` loosely: strict pass or any variant passes.

    The ``or`` with the strict verdict keeps the loose-monotonicity invariant
    (loose passes whenever strict passes), matching the official reporter where
    loose accuracy is never below strict accuracy on the same responses.
    """
    if check(response):
        return True
    return any(bool(check(variant)) for variant in loose_variants(response))


class IfEvalGradePair(ResultModel):
    """One strict and one loose verdict for the same accepted sample.

    The two sides share the ``sample_key`` but are separate ``GradeResult``
    objects with disjoint ``score_components`` keys. There is deliberately no
    combined, merged, or averaged field: downstream code aggregates each side
    over its own denominator (see :func:`summarize_pairs`).
    """

    strict: GradeResult
    loose: GradeResult


class IfEvalSummary(ResultModel):
    """Separate strict/loose accuracies over their own eligible denominators.

    An accuracy is ``None`` when its denominator is zero: an unknown rate stays
    ``null``, it is never reported as ``0.0``.
    """

    strict_correct: int
    strict_eligible: int
    strict_accuracy: float | None
    loose_correct: int
    loose_eligible: int
    loose_accuracy: float | None


def _graded_pair(
    sample_key: SampleKey,
    *,
    strict_pass: bool,
    loose_pass: bool,
    malformed: bool,
) -> IfEvalGradePair:
    """Build the strict/loose ``GradeResult`` pair for a genuinely graded sample."""
    format_status: StatusFlag = "fail" if malformed else "pass"
    strict_score = 1.0 if strict_pass else 0.0
    loose_score = 1.0 if loose_pass else 0.0
    base = {
        "grader_version": GRADER_VERSION,
        "sample_key": sample_key,
        "transport": "pass",
        "evaluator": "pass",
    }
    return IfEvalGradePair(
        strict=GradeResult.model_validate(
            {
                **base,
                "correctness": "pass" if strict_pass else "fail",
                "format": format_status,
                "score_components": {"prompt_level_strict": strict_score},
                "denominator_eligibility": DenominatorEligibility(
                    correctness=True,
                    format=True,
                    transport_success=True,
                    evaluator_ran=True,
                ),
            }
        ),
        loose=GradeResult.model_validate(
            {
                **base,
                "correctness": "pass" if loose_pass else "fail",
                "format": format_status,
                "score_components": {"prompt_level_loose": loose_score},
                "denominator_eligibility": DenominatorEligibility(
                    correctness=True,
                    format=True,
                    transport_success=True,
                    evaluator_ran=True,
                ),
            }
        ),
    )


def _transport_pair(sample_key: SampleKey, *, transport: StatusFlag) -> IfEvalGradePair:
    """Build a pair for a generation that never produced a gradeable response.

    ``correctness`` stays ``"unavailable"``: a delivery failure is not an
    incorrect answer and must not enter a correctness denominator.
    """
    eligibility = DenominatorEligibility(
        correctness=False,
        format=False,
        transport_success=False,
        evaluator_ran=False,
    )
    kwargs = {
        "grader_version": GRADER_VERSION,
        "sample_key": sample_key,
        "correctness": "unavailable",
        "format": "unavailable",
        "transport": transport,
        "evaluator": "unavailable",
        "score_components": {},
        "denominator_eligibility": eligibility,
    }
    return IfEvalGradePair(
        strict=GradeResult.model_validate(dict(kwargs)),
        loose=GradeResult.model_validate(dict(kwargs)),
    )


def _evaluator_failed_pair(sample_key: SampleKey, *, malformed: bool) -> IfEvalGradePair:
    """Build a pair for when the official checker could not run to a verdict.

    ``correctness`` stays ``"unavailable"`` so an ungraded sample never enters
    an accuracy denominator. ``format`` still records whether the delivered
    response was well formed.
    """
    format_status = "fail" if malformed else "pass"
    eligibility = DenominatorEligibility(
        correctness=False,
        format=True,
        transport_success=True,
        evaluator_ran=False,
    )
    kwargs = {
        "grader_version": GRADER_VERSION,
        "sample_key": sample_key,
        "correctness": "unavailable",
        "format": format_status,
        "transport": "pass",
        "evaluator": "fail",
        "score_components": {},
        "denominator_eligibility": eligibility,
    }
    return IfEvalGradePair(
        strict=GradeResult.model_validate(dict(kwargs)),
        loose=GradeResult.model_validate(dict(kwargs)),
    )


def grade_accepted_sample(
    *,
    task: TaskSpec,
    response: str,
    check: InstructionCheck,
) -> IfEvalGradePair:
    """Grade one delivered response, returning the strict/loose pair.

    Calls ``task.safe_request()`` first so gold in the prompt blocks grading
    instead of silently flowing past. A check that raises is reported as an
    evaluator failure, never as an incorrect answer.
    """
    task.safe_request()
    sample_key = task.sample_key
    malformed = is_malformed(response)
    try:
        strict_pass = evaluate_strict(response, check)
        loose_pass = evaluate_loose(response, check)
    except Exception:
        return _evaluator_failed_pair(sample_key, malformed=malformed)
    return _graded_pair(
        sample_key,
        strict_pass=strict_pass,
        loose_pass=loose_pass,
        malformed=malformed,
    )


def grade_generation(
    *,
    task: TaskSpec,
    generation: GenerationResult,
    check: InstructionCheck | None = None,
) -> IfEvalGradePair:
    """Grade one delivery attempt, keeping transport/evaluator failures distinct.

    * The generation's ``sample_key`` must equal the task's; a mismatch raises
      rather than grading one item's response against another's checks.
    * A non-accepted generation (transport failure, cancellation, unresolved
      dispatch) yields ``correctness="unavailable"`` on both sides.
    * An accepted generation graded with ``check=None`` (grader never ran)
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
        return _transport_pair(generation.sample_key, transport=transport)
    if check is None:
        return _evaluator_failed_pair(
            generation.sample_key, malformed=is_malformed(generation.response)
        )
    return grade_accepted_sample(task=task, response=generation.response, check=check)


def summarize_pairs(pairs: Sequence[IfEvalGradePair]) -> IfEvalSummary:
    """Aggregate strict and loose accuracies over separate eligible denominators.

    Only samples with ``counts_toward_accuracy`` on that side enter its
    denominator, so transport failures and unevaluated samples are excluded
    from both rates rather than counted as incorrect.
    """
    strict_eligible = sum(1 for pair in pairs if pair.strict.counts_toward_accuracy)
    strict_passes = sum(
        1
        for pair in pairs
        if pair.strict.counts_toward_accuracy and pair.strict.correctness == "pass"
    )
    loose_eligible = sum(1 for pair in pairs if pair.loose.counts_toward_accuracy)
    loose_passes = sum(
        1
        for pair in pairs
        if pair.loose.counts_toward_accuracy and pair.loose.correctness == "pass"
    )
    return IfEvalSummary(
        strict_correct=strict_passes,
        strict_eligible=strict_eligible,
        strict_accuracy=(strict_passes / strict_eligible) if strict_eligible else None,
        loose_correct=loose_passes,
        loose_eligible=loose_eligible,
        loose_accuracy=(loose_passes / loose_eligible) if loose_eligible else None,
    )


__all__ = [
    "GRADER_VERSION",
    "LOOSE_VARIANT_NAMES",
    "OFFICIAL_EVALUATOR_MODULE",
    "OFFICIAL_LOOSE_FILENAME",
    "OFFICIAL_STRICT_FILENAME",
    "IfEvalGradePair",
    "IfEvalSummary",
    "InstructionCheck",
    "dispatch_request",
    "evaluate_loose",
    "evaluate_strict",
    "grade_accepted_sample",
    "grade_generation",
    "is_malformed",
    "loose_variants",
    "summarize_pairs",
]
