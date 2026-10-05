"""MATH-500 official scoring adapter (G06 T06B).

Official source (see ``docs/upstream-inventory.md``):

* Dataset ``HuggingFaceH4/MATH-500`` at data revision
  ``6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be`` (500 problems with
  ``problem``, ``solution``, ``answer``, ``subject``, ``level``,
  ``unique_id``).
* Reference harness task ``math_500`` with metric ``pass_at_k_math``,
  backed by ``latex2sympy2_extended==1.0.6`` for symbolic equivalence.
* Reference generation budget: ``generation_size`` 32768 tokens.

Deliberate divergences (recorded in the inventory):

* The harness also registers a non-deterministic auxiliary graded metric.
  That metric is deliberately absent here: this module exposes no such API
  and no score component for it, so a comparable score can never silently
  depend on it.
* Upstream symbolic parsing is unbounded. StealthBench imposes an explicit
  parser wall-time limit (``PARSER_TIMEOUT_SECONDS``) enforced without any
  subprocess, and treats a timeout as a graded failure with the reason
  recorded in the score components. Oversized expressions fail safely
  instead of being parsed.

Frozen protocol (this file is the freeze):

* Extraction: the last balanced ``\\boxed{...}`` wins; else the last
  ``$...$``; else the last non-empty line. Oversized responses raise
  ``OversizedInputError`` instead of being scanned without bound.
* Normalization (ordered): strip; unwrap ``\\boxed``/``\\text``/
  ``\\mathrm``/``\\mathbf``; drop ``$``, ``\\(`` ``\\)``, ``\\[`` ``\\]``,
  ``\\left``/``\\right`` and thin-space commands; remove whitespace and
  thousands commas; strip one trailing period. No case folding.
* Equivalence (ordered): exact normalized-string match, then numeric
  comparison within ``NUMERIC_TOLERANCE`` (recorded in every graded
  verdict), then an optional bounded symbolic check when ``sympy`` is
  importable. When ``sympy`` is absent the verdict degrades gracefully to
  exact-plus-numeric rather than failing open or closed.
* A timeout (``ParserTimeoutError``) or oversized input
  (``OversizedInputError``) is ``correctness="fail"`` with
  ``evaluator="pass"`` so it stays in the denominator; the reason is
  recorded as a 0/1 score component. Only a missing gold answer or a
  grader crash is ``evaluator="fail"``.

Offline by construction: pure functions over local strings and contracts.
No transport, no socket, no subprocess, no credential.
"""

from __future__ import annotations

import math
import re
import signal
import threading
import time
from collections.abc import Callable, Sequence
from fractions import Fraction
from pathlib import Path
from types import FrameType
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

#: Pinned MATH-500 data revision from docs/upstream-inventory.md.
PINNED_MATH500_DATASET_ID: Final[str] = "HuggingFaceH4/MATH-500"
PINNED_MATH500_DATASET_REVISION: Final[str] = "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be"
PINNED_MATH500_SPLIT: Final[str] = "test"
PINNED_MATH500_ITEM_COUNT: Final[int] = 500

#: Grader identity: the reference task at the pinned data revision.
GRADER_VERSION: Final[str] = f"math-500-equiv@{PINNED_MATH500_DATASET_REVISION}"

#: Absolute+relative numeric tolerance for the numeric-equivalence path.
#: Recorded in every graded verdict under ``score_components``.
NUMERIC_TOLERANCE: Final[float] = 1e-6

#: Default wall-time bound for one symbolic-equivalence decision, in seconds.
PARSER_TIMEOUT_SECONDS: Final[float] = 2.0

#: Expressions longer than this are refused without parsing.
MAX_EXPRESSION_CHARS: Final[int] = 4096

#: Responses longer than this are refused without scanning.
MAX_RESPONSE_CHARS: Final[int] = 32768

_FRAC_RE: Final[re.Pattern[str]] = re.compile(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}")
_SQRT_RE: Final[re.Pattern[str]] = re.compile(r"\\sqrt\s*\{([^{}]+)\}")
_TEXT_WRAP_RE: Final[re.Pattern[str]] = re.compile(
    r"\\(?:text|mathrm|mathbf|boldsymbol)\s*\{([^{}]*)\}"
)
_DOLLAR_PAIR_RE: Final[re.Pattern[str]] = re.compile(r"\$(.+?)\$", re.DOTALL)


class ParserTimeoutError(TimeoutError):
    """Bounded parsing exhausted its wall-time budget."""


class OversizedInputError(ValueError):
    """An expression or response exceeded its frozen size bound."""


class MathParseError(ValueError):
    """An expression could not be parsed to a comparable form."""


class Math500Summary(ResultModel):
    """Accuracy over the eligible (graded) denominator.

    ``accuracy`` is ``None`` when the denominator is zero: an unknown rate
    stays ``null``, it is never reported as ``0.0``.
    """

    correct: int
    eligible: int
    accuracy: float | None
    timed_out: int
    oversized: int


def _check_budget(started: float, timeout_seconds: float) -> None:
    if timeout_seconds <= 0:
        raise ParserTimeoutError(f"parser budget exhausted (timeout_seconds={timeout_seconds})")
    if time.monotonic() - started > timeout_seconds:
        raise ParserTimeoutError(f"symbolic parsing exceeded {timeout_seconds}s wall-time bound")


def _signal_available() -> bool:
    return hasattr(signal, "SIGALRM") and threading.current_thread() is threading.main_thread()


def _run_with_wall_timeout(
    func: Callable[[], bool | None],
    *,
    timeout_seconds: float,
) -> bool | None:
    """Run ``func`` with a subprocess-free wall-time bound.

    Uses ``signal.SIGALRM`` on supporting platforms (main thread only);
    elsewhere the caller enforces a step budget via ``_check_budget`` and
    this wrapper still applies a post-hoc elapsed check. No subprocess,
    thread pool or executor is used.
    """
    if timeout_seconds <= 0:
        raise ParserTimeoutError(f"parser budget exhausted (timeout_seconds={timeout_seconds})")
    if not _signal_available():
        started = time.monotonic()
        outcome = func()
        _check_budget(started, timeout_seconds)
        return outcome
    alarm = signal.SIGALRM

    def _handler(_signum: int, _frame: FrameType | None) -> None:
        raise ParserTimeoutError(f"symbolic parsing exceeded {timeout_seconds}s wall-time bound")

    previous = signal.signal(alarm, _handler)
    try:
        signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
        try:
            return func()
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
    finally:
        signal.signal(alarm, previous)


def normalize_answer(text: str) -> str:
    """Normalize a math answer string under the frozen ordered steps."""
    out = text.strip()
    out = out.replace("$", "")
    out = out.replace("\\(", "").replace("\\)", "")
    out = out.replace("\\[", "").replace("\\]", "")
    out = out.replace("\\left", "").replace("\\right", "")
    for thin in ("\\,", "\\:", "\\;", "\\!"):
        out = out.replace(thin, "")
    out = _TEXT_WRAP_RE.sub(r"\1", out)
    out = re.sub(r"\s+", "", out)
    out = out.replace(",", "")
    if out.endswith(".") and len(out) > 1:
        out = out[:-1]
    return out


def _extract_last_boxed(text: str) -> str | None:
    marker = "\\boxed{"
    index = text.rfind(marker)
    if index < 0:
        return None
    depth = 0
    start = index + len(marker)
    for pos in range(start, len(text)):
        char = text[pos]
        if char == "{":
            depth += 1
        elif char == "}":
            if depth == 0:
                return text[start:pos]
            depth -= 1
    return None


def extract_final_answer(response: str) -> str:
    """Extract the candidate answer under the frozen precedence.

    Last balanced ``\\boxed{...}`` wins; else the last ``$...$``; else the
    last non-empty line. Raises ``OversizedInputError`` without scanning
    when the response exceeds ``MAX_RESPONSE_CHARS``.
    """
    if len(response) > MAX_RESPONSE_CHARS:
        raise OversizedInputError(
            f"response has {len(response)} characters, bound is {MAX_RESPONSE_CHARS}"
        )
    boxed = _extract_last_boxed(response)
    if boxed is not None:
        return boxed.strip()
    dollars = _DOLLAR_PAIR_RE.findall(response)
    if dollars:
        last: str = dollars[-1]
        return last.strip()
    lines = [line.strip() for line in response.splitlines() if line.strip()]
    if lines:
        return lines[-1]
    return response.strip()


def _parse_float_value(text: str) -> float | None:
    candidate = text.strip()
    if not candidate:
        return None
    if candidate.endswith("%"):
        inner = _parse_float_value(candidate[:-1])
        return inner / 100.0 if inner is not None else None
    frac = _FRAC_RE.fullmatch(candidate)
    if frac is not None:
        num = _parse_float_value(frac.group(1))
        den = _parse_float_value(frac.group(2))
        if num is not None and den is not None and den != 0.0:
            return num / den
        return None
    sqrt = _SQRT_RE.fullmatch(candidate)
    if sqrt is not None:
        inner = _parse_float_value(sqrt.group(1))
        if inner is not None and inner >= 0.0:
            return math.sqrt(inner)
        return None
    try:
        return float(candidate)
    except ValueError:
        pass
    if "/" in candidate and candidate.count("/") == 1:
        try:
            return float(Fraction(candidate))
        except (ValueError, ZeroDivisionError):
            return None
    return None


def numeric_equals(
    predicted: str,
    gold: str,
    *,
    tolerance: float = NUMERIC_TOLERANCE,
) -> bool:
    """True when both sides parse as numbers within ``tolerance``."""
    pred_value = _parse_float_value(normalize_answer(predicted))
    gold_value = _parse_float_value(normalize_answer(gold))
    if pred_value is None or gold_value is None:
        return False
    if not math.isfinite(pred_value) or not math.isfinite(gold_value):
        return pred_value == gold_value
    return math.isclose(pred_value, gold_value, rel_tol=tolerance, abs_tol=tolerance)


def _try_symbolic_equivalence(
    predicted_norm: str,
    gold_norm: str,
    *,
    timeout_seconds: float,
) -> bool | None:
    """Optional bounded symbolic check; ``None`` when unavailable.

    Returns ``None`` when ``sympy`` cannot be imported or either side does
    not parse. Never raises for parse failures; the wall-time bound still
    applies to the attempt itself.
    """
    try:
        from sympy.parsing.sympy_parser import parse_expr  # type: ignore[import-not-found]
    except ImportError:
        return None

    def _attempt() -> bool | None:
        try:
            pred_expr = parse_expr(predicted_norm)
            gold_expr = parse_expr(gold_norm)
        except Exception:
            return None
        try:
            diff = pred_expr - gold_expr
        except Exception:
            return None
        try:
            simplified = diff.simplify()
            return bool(simplified == 0)
        except Exception:
            return None

    return _run_with_wall_timeout(_attempt, timeout_seconds=timeout_seconds)


def symbolic_equals(
    predicted: str,
    gold: str,
    *,
    timeout_seconds: float = PARSER_TIMEOUT_SECONDS,
) -> tuple[bool, str]:
    """Decide equivalence under the frozen ordered paths.

    Returns ``(equal, method)`` with ``method`` in ``exact``, ``numeric``,
    ``symbolic`` or ``none``. Raises ``ParserTimeoutError`` when the wall
    budget is exhausted and ``OversizedInputError`` for oversized inputs;
    raises ``MathParseError`` when either side is empty after normalization.
    """
    started = time.monotonic()
    _check_budget(started, timeout_seconds)
    if len(predicted) > MAX_EXPRESSION_CHARS or len(gold) > MAX_EXPRESSION_CHARS:
        raise OversizedInputError(
            f"expression exceeds {MAX_EXPRESSION_CHARS} characters; refusing to parse"
        )
    predicted_norm = normalize_answer(predicted)
    _check_budget(started, timeout_seconds)
    gold_norm = normalize_answer(gold)
    _check_budget(started, timeout_seconds)
    if not predicted_norm or not gold_norm:
        raise MathParseError("empty expression after normalization")
    if predicted_norm == gold_norm:
        return True, "exact"
    _check_budget(started, timeout_seconds)
    if numeric_equals(predicted_norm, gold_norm):
        return True, "numeric"
    _check_budget(started, timeout_seconds)
    remaining = timeout_seconds - (time.monotonic() - started)
    symbolic = _try_symbolic_equivalence(
        predicted_norm, gold_norm, timeout_seconds=max(remaining, 0.0)
    )
    _check_budget(started, timeout_seconds)
    if symbolic is True:
        return True, "symbolic"
    if symbolic is False:
        return False, "symbolic"
    return False, "none"


def dispatch_request(task: TaskSpec) -> ModelRequest:
    """Return the dispatchable request for ``task``, proving gold stays out."""
    return task.safe_request()


def verify_math500_revision(actual: str | None) -> str:
    """Accept only the pinned MATH-500 data revision, else raise."""
    return verify_dataset_revision(
        actual=actual,
        expected=PINNED_MATH500_DATASET_REVISION,
        benchmark_id="math500",
    )


def verify_math500_checksum(path: Path, *, expected_sha256: str) -> str:
    """Return the file digest, or raise on a checksum mismatch."""
    return verify_file_checksum(path, expected_sha256=expected_sha256)


def is_malformed(response: str) -> bool:
    """True when a delivered response carries no gradeable content."""
    return not response.strip()


def _graded(
    sample_key: SampleKey,
    *,
    correct: bool,
    format_ok: bool,
    timed_out: bool,
    oversized: bool,
) -> GradeResult:
    correctness: StatusFlag = "pass" if correct else "fail"
    format_status: StatusFlag = "pass" if format_ok else "fail"
    return GradeResult.model_validate(
        {
            "grader_version": GRADER_VERSION,
            "sample_key": sample_key.model_dump(mode="json"),
            "correctness": correctness,
            "format": format_status,
            "transport": "pass",
            "evaluator": "pass",
            "score_components": {
                "math_correct": 1.0 if correct else 0.0,
                "numeric_tolerance": NUMERIC_TOLERANCE,
                "parser_timed_out": 1.0 if timed_out else 0.0,
                "oversized": 1.0 if oversized else 0.0,
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
    """A generation that never produced a gradeable response."""
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
    """The grader could not run to a verdict (missing gold or crashed)."""
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
    timeout_seconds: float = PARSER_TIMEOUT_SECONDS,
) -> GradeResult:
    """Grade one delivered response against the gold answer.

    Calls ``task.safe_request()`` first so gold in the prompt blocks grading.
    A missing gold answer is an evaluator failure. A parser timeout or an
    oversized input is a graded failure (``correctness="fail"`` with the
    reason recorded) that stays in the denominator.
    """
    task.safe_request()
    sample_key = task.sample_key
    if gold_answer is None or not gold_answer.strip():
        return _evaluator_failed_pair(sample_key, malformed=is_malformed(response))
    try:
        candidate = extract_final_answer(response)
    except OversizedInputError:
        return _graded(sample_key, correct=False, format_ok=False, timed_out=False, oversized=True)
    if not candidate.strip():
        return _graded(sample_key, correct=False, format_ok=False, timed_out=False, oversized=False)
    try:
        equal, _method = symbolic_equals(candidate, gold_answer, timeout_seconds=timeout_seconds)
    except ParserTimeoutError:
        return _graded(sample_key, correct=False, format_ok=True, timed_out=True, oversized=False)
    except OversizedInputError:
        return _graded(sample_key, correct=False, format_ok=False, timed_out=False, oversized=True)
    except MathParseError:
        return _graded(sample_key, correct=False, format_ok=False, timed_out=False, oversized=False)
    return _graded(
        sample_key,
        correct=equal,
        format_ok=True,
        timed_out=False,
        oversized=False,
    )


def grade_generation(
    *,
    task: TaskSpec,
    generation: GenerationResult,
    gold_answer: str | None,
    timeout_seconds: float = PARSER_TIMEOUT_SECONDS,
) -> GradeResult:
    """Grade one delivery attempt, keeping transport/evaluator failures distinct."""
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
        timeout_seconds=timeout_seconds,
    )


def summarize(grades: Sequence[GradeResult]) -> Math500Summary:
    """Aggregate accuracy over the eligible denominator."""
    eligible = sum(1 for grade in grades if grade.counts_toward_accuracy)
    correct = sum(
        1 for grade in grades if grade.counts_toward_accuracy and grade.correctness == "pass"
    )
    timed_out = sum(
        1
        for grade in grades
        if grade.counts_toward_accuracy
        and float(grade.score_components.get("parser_timed_out", 0.0)) == 1.0
    )
    oversized = sum(
        1
        for grade in grades
        if grade.counts_toward_accuracy
        and float(grade.score_components.get("oversized", 0.0)) == 1.0
    )
    return Math500Summary(
        correct=correct,
        eligible=eligible,
        accuracy=(correct / eligible) if eligible else None,
        timed_out=timed_out,
        oversized=oversized,
    )


__all__ = [
    "GRADER_VERSION",
    "MAX_EXPRESSION_CHARS",
    "MAX_RESPONSE_CHARS",
    "NUMERIC_TOLERANCE",
    "PARSER_TIMEOUT_SECONDS",
    "PINNED_MATH500_DATASET_ID",
    "PINNED_MATH500_DATASET_REVISION",
    "PINNED_MATH500_ITEM_COUNT",
    "PINNED_MATH500_SPLIT",
    "DatasetChecksumMismatch",
    "DatasetRevisionMismatch",
    "Math500Summary",
    "MathParseError",
    "OversizedInputError",
    "ParserTimeoutError",
    "dispatch_request",
    "extract_final_answer",
    "grade_accepted_sample",
    "grade_generation",
    "is_malformed",
    "normalize_answer",
    "numeric_equals",
    "sha256_file",
    "summarize",
    "symbolic_equals",
    "verify_math500_checksum",
    "verify_math500_revision",
]
