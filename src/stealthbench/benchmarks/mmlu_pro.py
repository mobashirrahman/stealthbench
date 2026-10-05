"""MMLU-Pro generation scoring adapter (G06 T06A).

Official source (see ``docs/upstream-inventory.md``):

* Repository ``https://github.com/TIGER-AI-Lab/MMLU-Pro`` at pinned commit
  ``f418b116db00b065c2aea046518d8fcf74d39872`` (no tags exist).
* Dataset ``TIGER-Lab/MMLU-Pro``, split ``test`` (12032 rows), plus a 70-row
  ``validation`` split used as the few-shot source.
* Evaluator module ``compute_accuracy.py`` invoked as
  ``python compute_accuracy.py results/<model>/``.
* Upstream correctness is regex-extracted from generated text. There is no
  likelihood path upstream, so loglikelihood scoring cannot be reproduced
  from a text-only endpoint and this module never fabricates one.
* Reference generation settings: ``max_model_length`` 4096,
  ``max_new_tokens`` 2048, temperature 0, stop ``['Question:']``.

Deliberate divergence (recorded in the inventory): the official extractor
falls back to a uniformly random letter A-J when no answer can be parsed,
which manufactures score. StealthBench records extraction failure as an
invalid response and reports it in the denominator instead of guessing.
This module therefore contains no ``random`` import, no likelihood API,
and no code path that invents a choice.

Frozen extraction protocol (this file is the freeze):

1. ``MAX_RESPONSE_CHARS`` bounds the input; anything larger is invalid
   (``oversized``) without running regexes over unbounded text.
2. Explicit tier: the first match of each of ``PATTERN_ANSWER_IS``,
   ``PATTERN_ANSWER_COLON`` and ``PATTERN_CHOICE_IS`` is collected
   (case-insensitive, any letter A-Z so that ``answer is K`` is detected
   as an invalid choice rather than silently ignored).
3. When the explicit tier yields exactly one distinct letter it wins, even
   if parenthesized letters elsewhere disagree. When it yields two or more
   distinct letters the response is ambiguous (``multiple``).
4. Fallback tier: parenthesized ``(X)`` matches for X in A-J, then a bare
   whole-response single letter. Multiple distinct parenthesized letters
   are ambiguous as well.
5. Anything else (empty, no match, a letter outside A-J, a letter outside
   the item's ``valid_choices``, oversized) is invalid.
6. An invalid response is still a delivered response: ``correctness="fail"``
   and ``format="fail"`` with ``evaluator="pass"``, so it stays in the
   correctness denominator. This mirrors the IFEval malformed-response rule.

Offline by construction: pure functions over local strings and contracts.
No transport, no socket, no subprocess, no credential.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path
from typing import Final

from stealthbench.benchmarks.datasets import (
    DatasetChecksumMismatch,
    DatasetRevisionMismatch,
    sha256_file,
    verify_dataset_revision,
    verify_file_checksum,
)
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

#: Pinned upstream commit for the MMLU-Pro evaluator (no tags exist).
PINNED_MMLU_PRO_COMMIT: Final[str] = "f418b116db00b065c2aea046518d8fcf74d39872"
#: Pinned dataset identity from docs/upstream-inventory.md.
PINNED_MMLU_PRO_DATASET_ID: Final[str] = "TIGER-Lab/MMLU-Pro"
PINNED_MMLU_PRO_DATASET_REVISION: Final[str] = "b189ec765aa7ed75c8acfea42df31fdae71f97be"
PINNED_MMLU_PRO_SPLIT: Final[str] = "test"
PINNED_MMLU_PRO_ITEM_COUNT: Final[int] = 12032

#: Grader identity: the official module at the pinned commit.
GRADER_VERSION: Final[str] = f"mmlu-pro-compute-accuracy@{PINNED_MMLU_PRO_COMMIT}"
OFFICIAL_EVALUATOR_MODULE: Final[str] = "compute_accuracy.py"

#: The ten MMLU-Pro option letters, in canonical order.
VALID_CHOICES: Final[tuple[str, ...]] = (
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
)

#: Responses longer than this are refused without regex evaluation.
MAX_RESPONSE_CHARS: Final[int] = 32768

#: Explicit "answer is X" (any letter, so invalid letters are caught).
PATTERN_ANSWER_IS: Final[re.Pattern[str]] = re.compile(
    r"answer\s+is\s*:?\s*\(?\s*([A-Za-z])\s*\)?", re.IGNORECASE
)
#: Explicit "answer: X" (any letter, so invalid letters are caught).
PATTERN_ANSWER_COLON: Final[re.Pattern[str]] = re.compile(
    r"answer\s*:\s*\(?\s*([A-Za-z])\s*\)?", re.IGNORECASE
)
#: Explicit "choice/option is X" (any letter, so invalid letters are caught).
PATTERN_CHOICE_IS: Final[re.Pattern[str]] = re.compile(
    r"(?:choice|option)\s+is\s*:?\s*\(?\s*([A-Za-z])\s*\)?", re.IGNORECASE
)
#: Fallback: parenthesized single letters A-J only.
PATTERN_PAREN_CHOICE: Final[re.Pattern[str]] = re.compile(r"\(\s*([A-Ja-j])\s*\)")

_EXPLICIT_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (
    PATTERN_ANSWER_IS,
    PATTERN_ANSWER_COLON,
    PATTERN_CHOICE_IS,
)


class ExtractionResult(ResultModel):
    """Outcome of the frozen choice extractor.

    ``choice`` is set only when ``valid`` is true. ``reason`` is one of
    ``single``, ``empty``, ``no_match``, ``multiple``, ``invalid_choice``
    or ``oversized``.
    """

    choice: str | None = None
    valid: bool
    reason: str


class MmluProSummary(ResultModel):
    """Accuracy over the eligible (graded) denominator.

    ``accuracy`` is ``None`` when the denominator is zero: an unknown rate
    stays ``null``, it is never reported as ``0.0``.
    """

    correct: int
    eligible: int
    accuracy: float | None
    invalid_responses: int


def _distinct_upper(letters: Sequence[str]) -> tuple[str, ...]:
    seen: list[str] = []
    for letter in letters:
        upper = letter.upper()
        if upper not in seen:
            seen.append(upper)
    return tuple(seen)


def extract_choice(
    response: str,
    valid_choices: Sequence[str] = VALID_CHOICES,
) -> ExtractionResult:
    """Extract a single option letter under the frozen protocol.

    Deterministic and total: the same input always yields the same output,
    and every input yields an ``ExtractionResult`` (never a raised error,
    never a guessed letter). Ambiguous, invalid and oversized inputs yield
    ``valid=False`` with the reason recorded.
    """
    allowed = tuple(valid_choices)
    if len(response) > MAX_RESPONSE_CHARS:
        return ExtractionResult(choice=None, valid=False, reason="oversized")
    if not response.strip():
        return ExtractionResult(choice=None, valid=False, reason="empty")

    explicit: list[str] = []
    for pattern in _EXPLICIT_PATTERNS:
        explicit.extend(match.group(1) for match in pattern.finditer(response))
    distinct_explicit = _distinct_upper(explicit)
    if distinct_explicit:
        if any(letter not in VALID_CHOICES for letter in distinct_explicit):
            return ExtractionResult(choice=None, valid=False, reason="invalid_choice")
        if len(distinct_explicit) > 1:
            return ExtractionResult(choice=None, valid=False, reason="multiple")
        only = distinct_explicit[0]
        if only not in allowed:
            return ExtractionResult(choice=None, valid=False, reason="invalid_choice")
        return ExtractionResult(choice=only, valid=True, reason="single")

    paren = [match.group(1) for match in PATTERN_PAREN_CHOICE.finditer(response)]
    distinct_paren = _distinct_upper(paren)
    if distinct_paren:
        if len(distinct_paren) > 1:
            return ExtractionResult(choice=None, valid=False, reason="multiple")
        only = distinct_paren[0]
        if only not in allowed:
            return ExtractionResult(choice=None, valid=False, reason="invalid_choice")
        return ExtractionResult(choice=only, valid=True, reason="single")

    stripped = response.strip().upper().rstrip(".,;:)")
    if len(stripped) == 1 and stripped in VALID_CHOICES:
        if stripped not in allowed:
            return ExtractionResult(choice=None, valid=False, reason="invalid_choice")
        return ExtractionResult(choice=stripped, valid=True, reason="single")
    return ExtractionResult(choice=None, valid=False, reason="no_match")


def is_malformed(response: str) -> bool:
    """True when a delivered response carries no gradeable content."""
    return not response.strip()


def dispatch_request(task: TaskSpec) -> ModelRequest:
    """Return the dispatchable request for ``task``, proving gold stays out.

    Re-verifies the recorded prompt hash and asserts evaluator-only data is
    absent; raises instead of returning a request that would leak gold.
    """
    return task.safe_request()


def verify_mmlu_pro_revision(actual: str | None) -> str:
    """Accept only the pinned MMLU-Pro dataset revision, else raise."""
    return verify_dataset_revision(
        actual=actual,
        expected=PINNED_MMLU_PRO_DATASET_REVISION,
        benchmark_id="mmlu_pro",
    )


def verify_mmlu_pro_checksum(path: Path, *, expected_sha256: str) -> str:
    """Return the file digest, or raise on a checksum mismatch."""
    return verify_file_checksum(path, expected_sha256=expected_sha256)


def _normalize_gold(gold_answer: str | None) -> str | None:
    if gold_answer is None:
        return None
    gold = gold_answer.strip().upper()
    if len(gold) == 1 and gold in VALID_CHOICES:
        return gold
    return None


def _graded(
    sample_key: SampleKey,
    *,
    correct: bool,
    extraction_valid: bool,
) -> GradeResult:
    correctness: StatusFlag = "pass" if correct else "fail"
    format_status: StatusFlag = "pass" if extraction_valid else "fail"
    return GradeResult.model_validate(
        {
            "grader_version": GRADER_VERSION,
            "sample_key": sample_key.model_dump(mode="json"),
            "correctness": correctness,
            "format": format_status,
            "transport": "pass",
            "evaluator": "pass",
            "score_components": {
                "accuracy": 1.0 if correct else 0.0,
                "extraction_valid": 1.0 if extraction_valid else 0.0,
            },
            "denominator_eligibility": DenominatorEligibility(
                correctness=True,
                format=True,
                transport_success=True,
                evaluator_ran=True,
            ).model_dump(mode="json"),
        }
    )


def _transport_pair(sample_key: SampleKey, *, transport: StatusFlag) -> GradeResult:
    """A generation that never produced a gradeable response.

    ``correctness`` stays ``"unavailable"``: a delivery failure is not an
    incorrect answer and must not enter a correctness denominator.
    """
    eligibility = DenominatorEligibility(
        correctness=False,
        format=False,
        transport_success=False,
        evaluator_ran=False,
    )
    return GradeResult.model_validate(
        {
            "grader_version": GRADER_VERSION,
            "sample_key": sample_key.model_dump(mode="json"),
            "correctness": "unavailable",
            "format": "unavailable",
            "transport": transport,
            "evaluator": "unavailable",
            "score_components": {},
            "denominator_eligibility": eligibility.model_dump(mode="json"),
        }
    )


def _evaluator_failed_pair(sample_key: SampleKey, *, malformed: bool) -> GradeResult:
    """The grader could not run to a verdict (missing/unusable gold).

    ``correctness`` stays ``"unavailable"`` so an ungraded sample never
    enters an accuracy denominator.
    """
    format_status: StatusFlag = "fail" if malformed else "pass"
    eligibility = DenominatorEligibility(
        correctness=False,
        format=True,
        transport_success=True,
        evaluator_ran=False,
    )
    return GradeResult.model_validate(
        {
            "grader_version": GRADER_VERSION,
            "sample_key": sample_key.model_dump(mode="json"),
            "correctness": "unavailable",
            "format": format_status,
            "transport": "pass",
            "evaluator": "fail",
            "score_components": {},
            "denominator_eligibility": eligibility.model_dump(mode="json"),
        }
    )


def grade_accepted_sample(
    *,
    task: TaskSpec,
    response: str,
    gold_answer: str | None,
    valid_choices: Sequence[str] = VALID_CHOICES,
) -> GradeResult:
    """Grade one delivered response against a gold option letter.

    Calls ``task.safe_request()`` first so gold in the prompt blocks grading
    instead of silently flowing past. A missing or unusable gold answer is
    an evaluator failure, never an incorrect answer. An unparseable,
    ambiguous or invalid choice is ``correctness="fail"`` with
    ``format="fail"`` so it stays in the denominator; no random letter is
    ever substituted.
    """
    task.safe_request()
    sample_key = task.sample_key
    gold = _normalize_gold(gold_answer)
    if gold is None or gold not in tuple(valid_choices):
        return _evaluator_failed_pair(sample_key, malformed=is_malformed(response))
    extracted = extract_choice(response, valid_choices)
    if not extracted.valid or extracted.choice is None:
        return _graded(sample_key, correct=False, extraction_valid=False)
    return _graded(
        sample_key,
        correct=extracted.choice == gold,
        extraction_valid=True,
    )


def grade_generation(
    *,
    task: TaskSpec,
    generation: GenerationResult,
    gold_answer: str | None,
    valid_choices: Sequence[str] = VALID_CHOICES,
) -> GradeResult:
    """Grade one delivery attempt, keeping transport/evaluator failures distinct.

    * The generation's ``sample_key`` must equal the task's; a mismatch
      raises rather than grading one item's response against another's gold.
    * A non-accepted generation (transport failure, cancellation, unresolved
      dispatch) yields ``correctness="unavailable"``.
    * A missing gold answer yields ``evaluator="fail"`` with
      ``correctness="unavailable"``.
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
    return grade_accepted_sample(
        task=task,
        response=generation.response,
        gold_answer=gold_answer,
        valid_choices=valid_choices,
    )


def summarize(grades: Sequence[GradeResult]) -> MmluProSummary:
    """Aggregate accuracy over the eligible denominator.

    Only samples with ``counts_toward_accuracy`` enter the denominator, so
    transport failures and unevaluated samples are excluded while invalid
    responses (``correctness="fail"``) are included.
    """
    eligible = sum(1 for grade in grades if grade.counts_toward_accuracy)
    correct = sum(
        1 for grade in grades if grade.counts_toward_accuracy and grade.correctness == "pass"
    )
    invalid = sum(1 for grade in grades if grade.counts_toward_accuracy and grade.format == "fail")
    return MmluProSummary(
        correct=correct,
        eligible=eligible,
        accuracy=(correct / eligible) if eligible else None,
        invalid_responses=invalid,
    )


__all__ = [
    "GRADER_VERSION",
    "MAX_RESPONSE_CHARS",
    "OFFICIAL_EVALUATOR_MODULE",
    "PINNED_MMLU_PRO_COMMIT",
    "PINNED_MMLU_PRO_DATASET_ID",
    "PINNED_MMLU_PRO_DATASET_REVISION",
    "PINNED_MMLU_PRO_ITEM_COUNT",
    "PINNED_MMLU_PRO_SPLIT",
    "VALID_CHOICES",
    "DatasetChecksumMismatch",
    "DatasetRevisionMismatch",
    "ExtractionResult",
    "MmluProSummary",
    "dispatch_request",
    "extract_choice",
    "grade_accepted_sample",
    "grade_generation",
    "is_malformed",
    "sha256_file",
    "summarize",
    "verify_mmlu_pro_checksum",
    "verify_mmlu_pro_revision",
]
