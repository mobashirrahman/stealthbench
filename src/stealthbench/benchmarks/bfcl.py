"""BFCL tool evaluation adapter (G08 T08A).

Official source (see ``docs/upstream-inventory.md``):

* Repo ``https://github.com/ShishirPatil/gorilla`` at pinned commit
  ``f7cf7359b7ac615a0b294831c5ba2bc95ee4a000``, leaderboard path
  ``berkeley-function-call-leaderboard``.
* Dataset ``gorilla-llm/Berkeley-Function-Calling-Leaderboard``, BFCL v4
  per-category JSONL files (5088 scored items plus 5218 non-scoring items).
* Evaluator module ``bfcl_eval/eval_checker/eval_runner.py`` invoked as
  ``bfcl evaluate --model <m> --test-category <c>``.
* Upstream evaluation temperature is fixed at ``0.001``; ``--partial-eval``
  results do not match the leaderboard.
* Two API modes are separate leaderboard rows: ``native_function_calling``
  and ``prompted``. Averaging them into one number would misrepresent the
  API mode.

Deliberate divergences (recorded in the inventory):

* The official summary columns count unevaluated categories as 0 rather than
  excluding them, so a partial run understates the score. StealthBench
  reports an unsupported category as unavailable with its denominator shown,
  never as a zero that shrinks the mean.
* Native and prompted profiles are reported as two separate tracks (G08).
  Mixing them into one summary raises instead of averaging.
* The retired ``Restful``/executable categories are never reported: passing
  one to any entry point raises.
* ``format_sensitivity`` is non-scoring and prompted-only; ``live_relevance``
  is non-scoring. Non-scoring categories grade normally but never enter the
  overall mean.
* The ``web_search`` (agentic) category requires a SerpAPI key and live
  network. Without the credential those items are unavailable, not scored.

Frozen protocol (this file is the freeze):

1. Categories are exact strings. ``classify_category`` maps them to
   ``scoring`` / ``non_scoring`` / ``retired`` / ``unknown``. Retired and
   unknown raise; they are never graded and never summarized.
2. ``is_supported`` gates SerpAPI and profile restrictions. An unsupported
   grade returns ``correctness="unavailable"`` and stays out of every
   denominator; its planned count is still shown by the summary.
3. Tool-call matching is strict: function names are case-sensitive, argument
   keys must match exactly (missing or extra fails), and values compare by
   exact type and value. Strings compare by exact Unicode equality with no
   case folding and no normalization, so ``café`` never equals ``cafe``.
   ``1`` (int) never equals ``1.0`` (float), and ``True`` never equals
   ``1``.
4. Parallel calls compare as multisets (order-insensitive, count-sensitive).
   ``irrelevance`` / ``live_irrelevance`` pass only when the model makes no
   call at all. Any predicted call whose function is outside
   ``offered_tools`` fails as a hallucination.
5. Overall accuracy is a micro-average over graded scoring samples and is
   ``None`` unless every scoring category has at least one eligible sample.
   A partial run never produces a complete-core score, and an unknown rate
   stays ``null``, never ``0.0``.

Offline by construction: pure functions over local values and contracts.
No transport, no socket, no subprocess, no credential.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final, Literal

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

#: Pinned upstream commit for the BFCL leaderboard code (no tags on this path).
PINNED_BFCL_COMMIT: Final[str] = "f7cf7359b7ac615a0b294831c5ba2bc95ee4a000"
#: Pinned dataset identity from docs/upstream-inventory.md.
PINNED_BFCL_DATASET_ID: Final[str] = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"
PINNED_BFCL_DATASET_SPLIT: Final[str] = "BFCL_v4 per-category JSONL files"
PINNED_BFCL_ITEM_COUNT: Final[int] = 5088
PINNED_BFCL_NON_SCORING_ITEM_COUNT: Final[int] = 5218

#: Grader identity: the official eval runner at the pinned commit.
GRADER_VERSION: Final[str] = f"bfcl-eval@{PINNED_BFCL_COMMIT}"
OFFICIAL_EVALUATOR_MODULE: Final[str] = "bfcl_eval/eval_checker/eval_runner.py"

#: Upstream evaluation temperature is fixed; sampling is effectively greedy.
EVALUATION_TEMPERATURE: Final[float] = 0.001

#: The two API modes are distinct tracks, never averaged.
BfclProfile = Literal["native_function_calling", "prompted"]
NATIVE_PROFILE: Final[BfclProfile] = "native_function_calling"
PROMPTED_PROFILE: Final[BfclProfile] = "prompted"

CategoryKind = Literal["scoring", "non_scoring", "retired", "unknown"]
CategoryStatus = Literal["supported", "non_scoring", "unavailable"]

#: Frozen scoring categories for BFCL v4 (this file is the freeze).
SCORING_CATEGORIES: Final[tuple[str, ...]] = (
    "simple_python",
    "simple_java",
    "simple_javascript",
    "multiple",
    "parallel",
    "parallel_multiple",
    "irrelevance",
    "live_irrelevance",
    "live_simple",
    "live_multiple",
    "live_parallel",
    "multi_turn_base",
    "multi_turn_miss_func",
    "multi_turn_miss_param",
    "multi_turn_long_context",
    "agentic_web_search",
)

#: Non-scoring categories: graded when run, never entering the overall mean.
NON_SCORING_CATEGORIES: Final[tuple[str, ...]] = (
    "live_relevance",
    "format_sensitivity",
)

#: Retired leaderboard categories: never reported in any form.
RETIRED_CATEGORIES: Final[tuple[str, ...]] = (
    "restful",
    "executable",
)

#: Categories that need a SerpAPI credential and live network.
SERPAPI_REQUIRED_CATEGORIES: Final[frozenset[str]] = frozenset({"agentic_web_search"})

#: Categories only supported for one profile.
PROMPTED_ONLY_CATEGORIES: Final[frozenset[str]] = frozenset({"format_sensitivity"})

#: Irrelevance categories pass only when the model makes no call.
IRRELEVANCE_CATEGORIES: Final[frozenset[str]] = frozenset({"irrelevance", "live_irrelevance"})


class BfclError(ValueError):
    """Base for BFCL adapter failures."""


class RetiredCategoryError(BfclError):
    """A retired Restful/executable category was requested. Never reported."""


class UnknownCategoryError(BfclError):
    """An unrecognized category was requested. Refusing to invent one."""


class ToolCall(ResultModel):
    """One predicted or expected function call with JSON-like arguments."""

    function: str = ""
    arguments: dict[str, Any] = {}

    @property
    def canonical(self) -> str:
        """Stable identity for multiset matching (sorted keys, unicode kept)."""
        return json.dumps(
            {"function": self.function, "arguments": self.arguments},
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        )


class BfclGrade(ResultModel):
    """One graded BFCL sample tagged with its track.

    The inner ``grade`` is the frozen ``GradeResult`` contract; ``profile``
    and ``category`` exist so native and prompted tracks can never be mixed
    silently. Summaries group by these tags and raise on a mismatch.
    """

    profile: BfclProfile
    category: str
    grade: GradeResult


class BfclCategorySummary(ResultModel):
    """Accuracy for one (category, profile) cell with its planned denominator.

    ``accuracy`` is ``None`` when ``eligible`` is zero: an unavailable cell
    shows its ``planned`` count and stays ``null``, it is never ``0.0``.
    """

    category: str
    profile: BfclProfile
    correct: int
    eligible: int
    planned: int
    accuracy: float | None
    status: CategoryStatus


class BfclSummary(ResultModel):
    """Per-cell accuracies plus the overall scoring mean for one profile.

    ``overall_accuracy`` is a micro-average over graded scoring samples and
    is ``None`` unless every scoring category has at least one eligible
    sample. ``complete`` is false for any partial run.
    """

    profile: BfclProfile
    per_category: tuple[BfclCategorySummary, ...]
    scoring_correct: int
    scoring_eligible: int
    scoring_planned: int
    overall_accuracy: float | None
    complete: bool


def is_retired_category(category: str) -> bool:
    """True for the retired Restful/executable families (case-insensitive)."""
    normalized = category.strip().lower()
    if normalized in RETIRED_CATEGORIES:
        return True
    return normalized.startswith(("restful_", "restful-", "executable_", "executable-"))


def classify_category(category: str) -> CategoryKind:
    """Map an exact category string to its frozen kind."""
    if is_retired_category(category):
        return "retired"
    if category in SCORING_CATEGORIES:
        return "scoring"
    if category in NON_SCORING_CATEGORIES:
        return "non_scoring"
    return "unknown"


def requires_serpapi(category: str) -> bool:
    """True when grading ``category`` needs a SerpAPI credential."""
    return category in SERPAPI_REQUIRED_CATEGORIES


def unsupported_reason(
    category: str,
    profile: BfclProfile,
    *,
    has_serpapi: bool = False,
) -> str | None:
    """Why ``category`` cannot be scored on ``profile``, or None if supported.

    Raises for retired/unknown categories: those are never reported, not
    even as unavailable.
    """
    kind = classify_category(category)
    if kind == "retired":
        raise RetiredCategoryError(
            f"category {category!r} was retired upstream and must never be reported"
        )
    if kind == "unknown":
        raise UnknownCategoryError(
            f"category {category!r} is not a pinned BFCL category; refusing to invent one"
        )
    if requires_serpapi(category) and not has_serpapi:
        return "missing SerpAPI credential and live network"
    if category in PROMPTED_ONLY_CATEGORIES and profile != PROMPTED_PROFILE:
        return f"category {category!r} is supported for prompted models only"
    return None


def is_supported(
    category: str,
    profile: BfclProfile,
    *,
    has_serpapi: bool = False,
) -> bool:
    """True when ``category`` can be scored on ``profile`` with this setup."""
    return unsupported_reason(category, profile, has_serpapi=has_serpapi) is None


def verify_bfcl_revision(actual: str | None) -> str:
    """Accept only the pinned BFCL commit, else raise rather than substitute."""
    return verify_dataset_revision(
        actual=actual,
        expected=PINNED_BFCL_COMMIT,
        benchmark_id="bfcl",
    )


def verify_bfcl_checksum(path: Path, *, expected_sha256: str) -> str:
    """Return the file digest, or raise on a checksum mismatch."""
    return verify_file_checksum(path, expected_sha256=expected_sha256)


def is_malformed(response: str) -> bool:
    """True when a delivered response carries no gradeable content."""
    return not response.strip()


def dispatch_request(task: TaskSpec) -> ModelRequest:
    """Return the dispatchable request for ``task``, proving gold stays out."""
    return task.safe_request()


def _value_equal(expected: Any, actual: Any) -> bool:
    """Strict value equality: exact type and value, unicode-exact strings."""
    if isinstance(expected, bool) or isinstance(actual, bool):
        return type(expected) is type(actual) and expected == actual
    if isinstance(expected, str) and isinstance(actual, str):
        return expected == actual
    if isinstance(expected, int) and isinstance(actual, int):
        return expected == actual
    if isinstance(expected, float) and isinstance(actual, float):
        return expected == actual
    if isinstance(expected, int | float) or isinstance(actual, int | float):
        return False
    if expected is None or actual is None:
        return expected is None and actual is None
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return False
        return all(_value_equal(e, a) for e, a in zip(expected, actual, strict=True))
    if isinstance(expected, dict) and isinstance(actual, dict):
        if set(expected) != set(actual):
            return False
        return all(_value_equal(expected[key], actual[key]) for key in expected)
    return bool(expected == actual)


def args_equal(
    expected: Mapping[str, Any],
    actual: Mapping[str, Any],
) -> bool:
    """True when argument objects match exactly (missing or extra fails)."""
    if set(expected) != set(actual):
        return False
    return all(_value_equal(expected[key], actual[key]) for key in expected)


def calls_match(
    predicted: Sequence[ToolCall],
    expected: Sequence[ToolCall],
) -> bool:
    """Order-insensitive, count-sensitive call-list equality."""
    if len(predicted) != len(expected):
        return False
    remaining = list(predicted)
    for want in expected:
        matched = -1
        for index, got in enumerate(remaining):
            if got.function == want.function and args_equal(want.arguments, got.arguments):
                matched = index
                break
        if matched < 0:
            return False
        remaining.pop(matched)
    return not remaining


def check_tool_calls(
    predicted: Sequence[ToolCall],
    expected: Sequence[ToolCall],
    category: str,
    offered_tools: Sequence[str] | None = None,
) -> bool:
    """Apply the frozen per-category oracle to parsed call lists.

    * Irrelevance categories pass only when ``predicted`` is empty.
    * Otherwise the lists must match via :func:`calls_match`.
    * When ``offered_tools`` is given, any predicted function outside it
      fails as a hallucination, even if the lists would otherwise match.
    """
    kind = classify_category(category)
    if kind == "retired":
        raise RetiredCategoryError(
            f"category {category!r} was retired upstream and must never be reported"
        )
    if kind == "unknown":
        raise UnknownCategoryError(
            f"category {category!r} is not a pinned BFCL category; refusing to invent one"
        )
    predicted_list = list(predicted)
    if offered_tools is not None:
        offered = set(offered_tools)
        for call in predicted_list:
            if call.function not in offered:
                return False
    if category in IRRELEVANCE_CATEGORIES:
        return len(predicted_list) == 0
    return calls_match(predicted_list, list(expected))


def _graded(
    sample_key: SampleKey,
    *,
    correct: bool,
    profile: BfclProfile,
    category: str,
) -> BfclGrade:
    correctness: StatusFlag = "pass" if correct else "fail"
    grade = GradeResult.model_validate(
        {
            "grader_version": GRADER_VERSION,
            "sample_key": sample_key.model_dump(mode="json"),
            "correctness": correctness,
            "format": "pass",
            "transport": "pass",
            "evaluator": "pass",
            "score_components": {
                "bfcl_correct": 1.0 if correct else 0.0,
                "evaluation_temperature": EVALUATION_TEMPERATURE,
            },
            "denominator_eligibility": DenominatorEligibility(
                correctness=True,
                format=True,
                transport_success=True,
                evaluator_ran=True,
            ).model_dump(mode="json"),
        }
    )
    return BfclGrade(profile=profile, category=category, grade=grade)


def _unavailable(
    sample_key: SampleKey,
    *,
    profile: BfclProfile,
    category: str,
    transport: StatusFlag = "unavailable",
    evaluator: StatusFlag = "unavailable",
) -> BfclGrade:
    """An unscored sample: never in a correctness denominator, never zero."""
    eligibility = DenominatorEligibility(
        correctness=False,
        format=False,
        transport_success=False,
        evaluator_ran=False,
    )
    grade = GradeResult.model_validate(
        {
            "grader_version": GRADER_VERSION,
            "sample_key": sample_key.model_dump(mode="json"),
            "correctness": "unavailable",
            "format": "unavailable",
            "transport": transport,
            "evaluator": evaluator,
            "score_components": {},
            "denominator_eligibility": eligibility.model_dump(mode="json"),
        }
    )
    return BfclGrade(profile=profile, category=category, grade=grade)


def _evaluator_failed(
    sample_key: SampleKey,
    *,
    profile: BfclProfile,
    category: str,
) -> BfclGrade:
    eligibility = DenominatorEligibility(
        correctness=False,
        format=True,
        transport_success=True,
        evaluator_ran=False,
    )
    grade = GradeResult.model_validate(
        {
            "grader_version": GRADER_VERSION,
            "sample_key": sample_key.model_dump(mode="json"),
            "correctness": "unavailable",
            "format": "pass",
            "transport": "pass",
            "evaluator": "fail",
            "score_components": {},
            "denominator_eligibility": eligibility.model_dump(mode="json"),
        }
    )
    return BfclGrade(profile=profile, category=category, grade=grade)


def _transport_grade(
    sample_key: SampleKey,
    *,
    profile: BfclProfile,
    category: str,
    transport: StatusFlag,
) -> BfclGrade:
    return _unavailable(sample_key, profile=profile, category=category, transport=transport)


def grade_accepted_sample(
    *,
    task: TaskSpec,
    predicted_calls: Sequence[ToolCall],
    expected_calls: Sequence[ToolCall],
    category: str,
    profile: BfclProfile,
    offered_tools: Sequence[str] | None = None,
    has_serpapi: bool = False,
) -> BfclGrade:
    """Grade one delivered tool-call response on one track.

    Calls ``task.safe_request()`` first so gold in the prompt blocks grading.
    A retired or unknown category raises. An unsupported category (SerpAPI
    gating, prompted-only on native) returns ``unavailable`` with its
    denominator preserved upstream by the summary, never a zero.
    """
    task.safe_request()
    sample_key = task.sample_key
    kind = classify_category(category)
    if kind == "retired":
        raise RetiredCategoryError(
            f"category {category!r} was retired upstream and must never be reported"
        )
    if kind == "unknown":
        raise UnknownCategoryError(
            f"category {category!r} is not a pinned BFCL category; refusing to invent one"
        )
    if profile not in (NATIVE_PROFILE, PROMPTED_PROFILE):
        raise BfclError(f"unknown BFCL profile {profile!r}; refusing to invent one")
    if unsupported_reason(category, profile, has_serpapi=has_serpapi) is not None:
        return _unavailable(sample_key, profile=profile, category=category)
    try:
        correct = check_tool_calls(predicted_calls, expected_calls, category, offered_tools)
    except (RetiredCategoryError, UnknownCategoryError):
        raise
    except Exception:
        return _evaluator_failed(sample_key, profile=profile, category=category)
    return _graded(sample_key, correct=correct, profile=profile, category=category)


def grade_generation(
    *,
    task: TaskSpec,
    generation: GenerationResult,
    predicted_calls: Sequence[ToolCall] | None,
    expected_calls: Sequence[ToolCall],
    category: str,
    profile: BfclProfile,
    offered_tools: Sequence[str] | None = None,
    has_serpapi: bool = False,
) -> BfclGrade:
    """Grade one delivery attempt, keeping transport/evaluator failures distinct.

    * The generation's ``sample_key`` must equal the task's; a mismatch
      raises rather than grading one item's calls against another's tools.
    * A non-accepted generation yields ``correctness="unavailable"``.
    * An accepted generation with ``predicted_calls=None`` (parser/grader
      never ran) yields ``evaluator="fail"`` with
      ``correctness="unavailable"``.
    """
    task.safe_request()
    if generation.sample_key != task.sample_key:
        raise ValueError(
            "generation sample_key "
            f"{generation.sample_key.model_dump()} does not match "
            f"task sample_key {task.sample_key.model_dump()}; refusing to grade"
        )
    kind = classify_category(category)
    if kind == "retired":
        raise RetiredCategoryError(
            f"category {category!r} was retired upstream and must never be reported"
        )
    if kind == "unknown":
        raise UnknownCategoryError(
            f"category {category!r} is not a pinned BFCL category; refusing to invent one"
        )
    if not generation.is_accepted_sample or generation.response is None:
        transport: StatusFlag
        if generation.delivery_status is DeliveryStatus.TRANSPORT_FAILED:
            transport = "fail"
        elif generation.delivery_status is DeliveryStatus.CANCELLED:
            transport = "invalid"
        else:
            transport = "unavailable"
        return _transport_grade(
            generation.sample_key, profile=profile, category=category, transport=transport
        )
    if predicted_calls is None:
        return _evaluator_failed(generation.sample_key, profile=profile, category=category)
    return grade_accepted_sample(
        task=task,
        predicted_calls=predicted_calls,
        expected_calls=expected_calls,
        category=category,
        profile=profile,
        offered_tools=offered_tools,
        has_serpapi=has_serpapi,
    )


def summarize_category(
    grades: Sequence[BfclGrade],
    *,
    category: str,
    profile: BfclProfile,
    planned: int,
) -> BfclCategorySummary:
    """Aggregate one (category, profile) cell over its eligible denominator.

    Every grade must carry the requested ``category`` and ``profile``; a mix
    raises instead of averaging tracks. ``planned`` is the declared item
    count for the cell and is shown even when ``eligible`` is zero, so an
    unavailable category keeps its denominator. ``accuracy`` is ``None``
    when ``eligible`` is zero, never ``0.0``.
    """
    kind = classify_category(category)
    if kind == "retired":
        raise RetiredCategoryError(
            f"category {category!r} was retired upstream and must never be reported"
        )
    if kind == "unknown":
        raise UnknownCategoryError(
            f"category {category!r} is not a pinned BFCL category; refusing to invent one"
        )
    if planned < 0:
        raise ValueError(f"planned must be non-negative, got {planned}")
    for item in grades:
        if item.category != category or item.profile != profile:
            raise ValueError(
                "refusing to mix BFCL tracks: summary asks for "
                f"({profile}, {category}) but a grade carries "
                f"({item.profile}, {item.category})"
            )
    eligible = sum(1 for item in grades if item.grade.counts_toward_accuracy)
    if eligible > planned and planned > 0:
        raise ValueError(f"eligible ({eligible}) exceeds planned ({planned}) for {category}")
    correct = sum(
        1
        for item in grades
        if item.grade.counts_toward_accuracy and item.grade.correctness == "pass"
    )
    if kind == "non_scoring":
        status: CategoryStatus = "non_scoring"
    elif eligible == 0:
        status = "unavailable"
    else:
        status = "supported"
    return BfclCategorySummary(
        category=category,
        profile=profile,
        correct=correct,
        eligible=eligible,
        planned=planned,
        accuracy=(correct / eligible) if eligible else None,
        status=status,
    )


def summarize(
    grades: Sequence[BfclGrade],
    *,
    profile: BfclProfile,
    planned_by_category: Mapping[str, int],
) -> BfclSummary:
    """Aggregate one profile track over its planned categories.

    Raises when any grade carries a different profile (mode separation),
    when a retired/unknown category appears in grades or in
    ``planned_by_category``, or when a planned count is negative.
    Non-scoring categories never enter the overall mean. The overall mean is
    ``None`` unless every scoring category has at least one eligible sample;
    a partial run stays incomplete and never reports a zero-filled average.
    """
    if profile not in (NATIVE_PROFILE, PROMPTED_PROFILE):
        raise BfclError(f"unknown BFCL profile {profile!r}; refusing to invent one")
    for item in grades:
        if item.profile != profile:
            raise ValueError(
                f"refusing to mix BFCL tracks: summary profile is {profile!r} "
                f"but a grade carries {item.profile!r}"
            )
    for category in list(planned_by_category):
        kind = classify_category(category)
        if kind == "retired":
            raise RetiredCategoryError(
                f"category {category!r} was retired upstream and must never be reported"
            )
        if kind == "unknown":
            raise UnknownCategoryError(
                f"category {category!r} is not a pinned BFCL category; refusing to invent one"
            )
    for item in grades:
        kind = classify_category(item.category)
        if kind == "retired":
            raise RetiredCategoryError(
                f"category {item.category!r} was retired upstream and must never be reported"
            )
        if kind == "unknown":
            raise UnknownCategoryError(f"category {item.category!r} is not a pinned BFCL category")
    by_category: dict[str, list[BfclGrade]] = {}
    for item in grades:
        by_category.setdefault(item.category, []).append(item)
    cells: list[BfclCategorySummary] = []
    for category in sorted({*planned_by_category, *by_category}):
        planned = planned_by_category.get(category, 0)
        if category not in planned_by_category:
            raise ValueError(
                f"category {category!r} was graded but has no planned denominator; "
                "an unplanned category cannot enter a published mean"
            )
        cells.append(
            summarize_category(
                by_category.get(category, ()),
                category=category,
                profile=profile,
                planned=planned,
            )
        )
    scoring = [cell for cell in cells if cell.category in SCORING_CATEGORIES]
    scoring_correct = sum(cell.correct for cell in scoring)
    scoring_eligible = sum(cell.eligible for cell in scoring)
    scoring_planned = sum(cell.planned for cell in scoring)
    covers_all = set(SCORING_CATEGORIES) <= set(planned_by_category)
    complete = bool(covers_all) and all(cell.eligible > 0 for cell in scoring)
    overall: float | None = None
    if complete and scoring_eligible:
        overall = scoring_correct / scoring_eligible
    return BfclSummary(
        profile=profile,
        per_category=tuple(cells),
        scoring_correct=scoring_correct,
        scoring_eligible=scoring_eligible,
        scoring_planned=scoring_planned,
        overall_accuracy=overall,
        complete=complete,
    )


__all__ = [
    "EVALUATION_TEMPERATURE",
    "GRADER_VERSION",
    "IRRELEVANCE_CATEGORIES",
    "NATIVE_PROFILE",
    "NON_SCORING_CATEGORIES",
    "OFFICIAL_EVALUATOR_MODULE",
    "PINNED_BFCL_COMMIT",
    "PINNED_BFCL_DATASET_ID",
    "PINNED_BFCL_DATASET_SPLIT",
    "PINNED_BFCL_ITEM_COUNT",
    "PINNED_BFCL_NON_SCORING_ITEM_COUNT",
    "PROMPTED_ONLY_CATEGORIES",
    "PROMPTED_PROFILE",
    "RETIRED_CATEGORIES",
    "SCORING_CATEGORIES",
    "SERPAPI_REQUIRED_CATEGORIES",
    "BfclCategorySummary",
    "BfclError",
    "BfclGrade",
    "BfclSummary",
    "CategoryKind",
    "CategoryStatus",
    "DatasetChecksumMismatch",
    "DatasetRevisionMismatch",
    "RetiredCategoryError",
    "ToolCall",
    "UnknownCategoryError",
    "args_equal",
    "calls_match",
    "check_tool_calls",
    "classify_category",
    "dispatch_request",
    "grade_accepted_sample",
    "grade_generation",
    "is_malformed",
    "is_retired_category",
    "is_supported",
    "requires_serpapi",
    "sha256_file",
    "summarize",
    "summarize_category",
    "unsupported_reason",
    "verify_bfcl_checksum",
    "verify_bfcl_revision",
]
