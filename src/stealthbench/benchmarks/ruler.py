"""RULER long-context generation and grading adapter (G08 T08B).

Official source (see ``docs/upstream-inventory.md``):

* Repo ``https://github.com/NVIDIA/RULER`` on branch ``rulerv1-ns`` at commit
  ``e8bbff677ca2c239640dc90f93310dcf32408c93`` (RULER publishes no release
  tags). The pipeline also lives outside this repo: the ``rulerv1-ns`` and
  ``rulerv2-ns`` branches carry only a README and ``run_example.sh`` that
  drive ``NVIDIA-NeMo/Skills``. The README instructs users to clone the
  NeMo-Skills branch ``chsieh/ruler-remove-prefix``.
* Evaluator module ``nemo_skills/dataset/ruler/ruler_score.py`` invoked as
  ``ns eval --benchmarks=ruler.<model>-<len>``.
* Dataset: 13 tasks, 100 samples per task (1300 items), context lengths
  4096 / 8192 / 16384 / 32768 / 65536 / 131072, generated with seed 42
  (per-chunk offset by the chunk index).
* Upstream sampling is greedy: temperature ``0.0``, top_p ``1.0``.
* Prompts are tokenized per model, so RULER inputs are equal in token length
  but are NOT identical strings across models. Cross-model comparison must
  state this.
* RULERv1 (13 tasks) and RULERv2 (12 tasks) are different pipelines with
  different scorers; exactly one is pinned here (v1).
* Aggregation is an unweighted mean over per-task accuracies, so a task that
  cannot be generated must be unavailable, not zero.
* Model context limits must be raised to the target length or requests
  truncate; a truncated task must never be reported at its intended length.

Deliberate divergences (recorded in the inventory): StealthBench pins
``rulerv1-ns``, records the per-task generation seed and the
template-token overhead separately from the provider's reported token
count, and reports truncation explicitly.

Frozen protocol (this file is the freeze):

1. Only ``RULER_V1_TASKS`` at a ``SUPPORTED_LENGTHS`` entry may be built or
   graded. Anything else raises instead of substituting another pipeline.
2. Generation is greedy (temperature ``0.0``, top_p ``1.0``) with seed
   ``42`` recorded on every ``RulerTaskSettings``. ``deterministic_case_id``
   binds ``(task, length, sample_index, seed)`` so the same seed reproduces
   the same cases and a different seed (overwhelmingly) does not.
3. A truncated or unsupported-length sample is ``correctness="unavailable"``
   with ``truncated=1.0`` in its score components. It never enters a
   denominator, and its length report carries ``effective_length=None`` so
   it can never be mistaken for delivered-at-length.
4. Provider-reported token counts and normalized lengths are separate fields
   on ``RulerLengthReport``. There is no helper that derives one from the
   other, and ``report_length`` refuses a truncated report that still claims
   an effective length.
5. The overall mean is an unweighted mean over per-cell accuracies and is
   ``None`` unless every planned cell has at least one eligible sample. An
   unknown rate stays ``null``, never ``0.0``.

Offline by construction: pure functions over local values and contracts.
No transport, no socket, no subprocess, no credential.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final, Literal

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

#: Pinned RULER branch and commit (no release tags exist upstream).
PINNED_RULER_BRANCH: Final[str] = "rulerv1-ns"
PINNED_RULER_COMMIT: Final[str] = "e8bbff677ca2c239640dc90f93310dcf32408c93"
#: The NeMo-Skills branch the rulerv1-ns README instructs users to clone.
NEMO_SKILLS_BRANCH: Final[str] = "chsieh/ruler-remove-prefix"
NEMO_SKILLS_REPO: Final[str] = "https://github.com/NVIDIA-NeMo/Skills"

#: Grader identity: the NeMo-Skills RULER scorer at the pinned RULER commit.
GRADER_VERSION: Final[str] = f"ruler-score@{PINNED_RULER_COMMIT}"
OFFICIAL_EVALUATOR_MODULE: Final[str] = "nemo_skills/dataset/ruler/ruler_score.py"

#: Frozen generation protocol: greedy sampling with the recorded seed.
GENERATION_SEED: Final[int] = 42
GENERATION_TEMPERATURE: Final[float] = 0.0
GENERATION_TOP_P: Final[float] = 1.0

#: Frozen context lengths (tokens) for the v1 pipeline.
SUPPORTED_LENGTHS: Final[tuple[int, ...]] = (4096, 8192, 16384, 32768, 65536, 131072)

#: Frozen RULER v1 task set (13 tasks). RULERv2 is a different pipeline.
RULER_V1_TASKS: Final[tuple[str, ...]] = (
    "niah_single_1",
    "niah_single_2",
    "niah_single_3",
    "niah_multikey_1",
    "niah_multikey_2",
    "niah_multikey_3",
    "niah_multiquery",
    "niah_multihop",
    "vt",
    "cwe",
    "fwe",
    "qa_1",
    "qa_2",
)

#: Dataset shape from docs/upstream-inventory.md.
RULER_SAMPLES_PER_TASK: Final[int] = 100
RULER_TASK_COUNT: Final[int] = 13
RULER_TOTAL_ITEMS: Final[int] = 1300

#: Per-model tokenization caveat: equal token length is not identical text.
TOKENIZATION_CAVEAT: Final[str] = (
    "RULER prompts are sized by per-model tokenization: inputs are equal in "
    "token length but are NOT identical strings across models. Cross-model "
    "comparison must state this."
)
#: Prompts are never byte-identical across different tokenizers.
PROMPTS_IDENTICAL_ACROSS_MODELS: Final[bool] = False

CellStatus = Literal["supported", "unavailable"]


class RulerError(ValueError):
    """Base for RULER adapter failures."""


class UnknownTaskError(RulerError):
    """An unrecognized RULER task was requested. Refusing to substitute."""


class UnsupportedLengthError(RulerError):
    """An unpinned context length was requested. Refusing to invent one."""


class RulerTaskSettings(ResultModel):
    """Frozen per-task generation settings with the seed recorded."""

    ruler_task: str
    requested_length: int
    seed: int = GENERATION_SEED
    temperature: float = GENERATION_TEMPERATURE
    top_p: float = GENERATION_TOP_P
    samples_per_task: int = RULER_SAMPLES_PER_TASK


class RulerLengthReport(ResultModel):
    """Intended versus measured lengths, kept strictly separate.

    ``provider_reported_tokens`` is what the endpoint counted (may be
    ``None`` when unreported). ``normalized_length`` is the evaluator-side
    normalized measure (may be ``None`` when not computed).
    ``template_overhead_tokens`` is the prompt-template overhead recorded
    separately from either count. A truncated or unsupported report carries
    ``effective_length=None`` so it can never read as delivered-at-length.
    """

    ruler_task: str
    requested_length: int
    effective_length: int | None
    truncated: bool
    provider_reported_tokens: int | None = None
    normalized_length: int | None = None
    template_overhead_tokens: int | None = None


class RulerCellSummary(ResultModel):
    """Accuracy for one (task, length) cell with its planned denominator."""

    ruler_task: str
    requested_length: int
    correct: int
    eligible: int
    planned: int
    accuracy: float | None
    status: CellStatus


class RulerSummary(ResultModel):
    """Per-cell accuracies plus the unweighted overall mean.

    ``overall_accuracy`` is the unweighted mean over per-cell accuracies and
    is ``None`` unless every planned cell has at least one eligible sample.
    """

    per_cell: tuple[RulerCellSummary, ...]
    overall_accuracy: float | None
    complete: bool
    generation_seed: int = GENERATION_SEED
    comparability_note: str = TOKENIZATION_CAVEAT


def is_supported_task(ruler_task: str) -> bool:
    """True when ``ruler_task`` is in the pinned v1 set."""
    return ruler_task in RULER_V1_TASKS


def is_supported_length(requested_length: int) -> bool:
    """True when ``requested_length`` is a pinned v1 context length."""
    return requested_length in SUPPORTED_LENGTHS


def verify_ruler_revision(actual: str | None) -> str:
    """Accept only the pinned rulerv1-ns commit, else raise."""
    return verify_dataset_revision(
        actual=actual,
        expected=PINNED_RULER_COMMIT,
        benchmark_id="ruler",
    )


def verify_ruler_branch(actual: str | None) -> str:
    """Accept only the pinned ``rulerv1-ns`` branch, else raise."""
    if actual != PINNED_RULER_BRANCH:
        raise RulerError(
            f"ruler branch {actual!r} does not match pinned branch "
            f"{PINNED_RULER_BRANCH!r}; RULERv1 and RULERv2 are different pipelines"
        )
    return actual


def verify_ruler_checksum(path: Path, *, expected_sha256: str) -> str:
    """Return the file digest, or raise on a checksum mismatch."""
    return verify_file_checksum(path, expected_sha256=expected_sha256)


def is_malformed(response: str) -> bool:
    """True when a delivered response carries no gradeable content."""
    return not response.strip()


def dispatch_request(task: TaskSpec) -> ModelRequest:
    """Return the dispatchable request for ``task``, proving gold stays out."""
    return task.safe_request()


def generation_settings(*, seed: int = GENERATION_SEED) -> dict[str, float | int]:
    """Frozen greedy generation settings with the seed recorded."""
    return {"temperature": GENERATION_TEMPERATURE, "top_p": GENERATION_TOP_P, "seed": seed}


def build_task_settings(
    ruler_task: str,
    requested_length: int,
    *,
    seed: int = GENERATION_SEED,
) -> RulerTaskSettings:
    """Freeze per-task settings, or raise for an unpinned task or length."""
    if not is_supported_task(ruler_task):
        raise UnknownTaskError(
            f"task {ruler_task!r} is not in the pinned RULER v1 set; "
            "refusing to substitute another pipeline"
        )
    if not is_supported_length(requested_length):
        raise UnsupportedLengthError(
            f"length {requested_length!r} is not a pinned RULER v1 context length; "
            "refusing to invent one"
        )
    return RulerTaskSettings(
        ruler_task=ruler_task,
        requested_length=requested_length,
        seed=seed,
        temperature=GENERATION_TEMPERATURE,
        top_p=GENERATION_TOP_P,
        samples_per_task=RULER_SAMPLES_PER_TASK,
    )


def deterministic_case_id(
    ruler_task: str,
    requested_length: int,
    sample_index: int,
    *,
    seed: int = GENERATION_SEED,
) -> str:
    """Stable case identity binding ``(task, length, index, seed)``.

    The same inputs always yield the same id; a different seed
    (overwhelmingly) yields a different one. Hash ordering keeps the result
    independent of any interpreter RNG.
    """
    if sample_index < 0:
        raise ValueError(f"sample_index must be non-negative, got {sample_index}")
    payload = f"{ruler_task}:{requested_length}:{sample_index}:{seed}".encode()
    return hashlib.sha256(payload).hexdigest()


def _normalize_retrieval(text: str) -> str:
    return text.strip()


def _word_set(text: str) -> frozenset[str]:
    tokens = [token.strip(".,;:()[]{}\"'").lower() for token in text.split()]
    return frozenset(token for token in tokens if token)


def score_response(response: str, expected: str, ruler_task: str) -> bool:
    """Apply the frozen per-task scoring rule.

    Retrieval-style tasks (needle, variable tracking, QA) compare the
    stripped response exactly. Aggregation tasks (``cwe``/``fwe``) compare
    unordered word sets. Raises for an unpinned task.
    """
    if not is_supported_task(ruler_task):
        raise UnknownTaskError(f"task {ruler_task!r} is not in the pinned RULER v1 set")
    if ruler_task in ("cwe", "fwe"):
        return _word_set(response) == _word_set(expected) and bool(_word_set(expected))
    return _normalize_retrieval(response) == _normalize_retrieval(expected)


def _graded(
    sample_key: SampleKey,
    *,
    correct: bool,
    intended_length: int,
    malformed: bool,
) -> GradeResult:
    correctness: StatusFlag = "pass" if correct else "fail"
    format_status: StatusFlag = "fail" if malformed else "pass"
    return GradeResult.model_validate(
        {
            "grader_version": GRADER_VERSION,
            "sample_key": sample_key.model_dump(mode="json"),
            "correctness": correctness,
            "format": format_status,
            "transport": "pass",
            "evaluator": "pass",
            "score_components": {
                "ruler_correct": 1.0 if correct else 0.0,
                "intended_length": float(intended_length),
                "truncated": 0.0,
                "generation_temperature": GENERATION_TEMPERATURE,
            },
            "denominator_eligibility": DenominatorEligibility(
                correctness=True,
                format=True,
                transport_success=True,
                evaluator_ran=True,
            ).model_dump(mode="json"),
        }
    )


def _unavailable_length(
    sample_key: SampleKey,
    *,
    intended_length: int,
    truncated: bool,
) -> GradeResult:
    """A sample that cannot be reported at its intended length.

    ``correctness`` stays ``"unavailable"`` so truncation never enters a
    correctness denominator, and ``truncated=1.0`` records the reason
    explicitly.
    """
    eligibility = DenominatorEligibility(
        correctness=False,
        format=False,
        transport_success=True,
        evaluator_ran=False,
    )
    return GradeResult.model_validate(
        {
            "grader_version": GRADER_VERSION,
            "sample_key": sample_key.model_dump(mode="json"),
            "correctness": "unavailable",
            "format": "unavailable",
            "transport": "pass",
            "evaluator": "unavailable",
            "score_components": {
                "intended_length": float(intended_length),
                "truncated": 1.0 if truncated else 0.0,
                "generation_temperature": GENERATION_TEMPERATURE,
            },
            "denominator_eligibility": eligibility.model_dump(mode="json"),
        }
    )


def _transport_grade(sample_key: SampleKey, *, transport: StatusFlag) -> GradeResult:
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


def _evaluator_failed(sample_key: SampleKey, *, malformed: bool) -> GradeResult:
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
    expected: str | None,
    ruler_task: str,
    intended_length: int,
    truncated: bool = False,
    unsupported_length: bool = False,
) -> GradeResult:
    """Grade one delivered response at one (task, length) cell.

    Calls ``task.safe_request()`` first so gold in the prompt blocks grading.
    An unpinned task raises. A truncated or unsupported-length sample returns
    ``correctness="unavailable"`` with ``truncated`` recorded: it is never
    graded as if delivered at its intended length. A missing expected answer
    is an evaluator failure, never an incorrect retrieval.
    """
    task.safe_request()
    sample_key = task.sample_key
    if not is_supported_task(ruler_task):
        raise UnknownTaskError(f"task {ruler_task!r} is not in the pinned RULER v1 set")
    if unsupported_length or truncated:
        return _unavailable_length(sample_key, intended_length=intended_length, truncated=True)
    if not is_supported_length(intended_length):
        raise UnsupportedLengthError(
            f"length {intended_length!r} is not a pinned RULER v1 context length"
        )
    if expected is None or not expected.strip():
        return _evaluator_failed(sample_key, malformed=is_malformed(response))
    try:
        correct = score_response(response, expected, ruler_task)
    except UnknownTaskError:
        raise
    except Exception:
        return _evaluator_failed(sample_key, malformed=is_malformed(response))
    if is_malformed(response) and correct:
        correct = False
    return _graded(
        sample_key,
        correct=correct,
        intended_length=intended_length,
        malformed=is_malformed(response),
    )


def grade_generation(
    *,
    task: TaskSpec,
    generation: GenerationResult,
    expected: str | None,
    ruler_task: str,
    intended_length: int,
    truncated: bool = False,
    unsupported_length: bool = False,
) -> GradeResult:
    """Grade one delivery attempt, keeping transport/evaluator failures distinct."""
    task.safe_request()
    if generation.sample_key != task.sample_key:
        raise ValueError(
            "generation sample_key "
            f"{generation.sample_key.model_dump()} does not match "
            f"task sample_key {task.sample_key.model_dump()}; refusing to grade"
        )
    if not is_supported_task(ruler_task):
        raise UnknownTaskError(f"task {ruler_task!r} is not in the pinned RULER v1 set")
    if not generation.is_accepted_sample or generation.response is None:
        transport: StatusFlag
        if generation.delivery_status is DeliveryStatus.TRANSPORT_FAILED:
            transport = "fail"
        elif generation.delivery_status is DeliveryStatus.CANCELLED:
            transport = "invalid"
        else:
            transport = "unavailable"
        return _transport_grade(generation.sample_key, transport=transport)
    return grade_accepted_sample(
        task=task,
        response=generation.response,
        expected=expected,
        ruler_task=ruler_task,
        intended_length=intended_length,
        truncated=truncated,
        unsupported_length=unsupported_length,
    )


def report_length(
    *,
    ruler_task: str,
    requested_length: int,
    effective_length: int | None,
    truncated: bool,
    unsupported: bool = False,
    provider_reported_tokens: int | None = None,
    normalized_length: int | None = None,
    template_overhead_tokens: int | None = None,
) -> RulerLengthReport:
    """Report intended versus measured lengths without conflating them.

    Provider counts and normalized lengths are separate nullable fields;
    neither derives from the other. A truncated or unsupported report must
    carry ``effective_length=None``: claiming an effective length equal to
    the request would misreport a cut-down prompt as delivered-at-length
    and therefore raises. A non-truncated report must carry
    ``effective_length == requested_length``.
    """
    if not is_supported_task(ruler_task):
        raise UnknownTaskError(f"task {ruler_task!r} is not in the pinned RULER v1 set")
    if truncated or unsupported:
        if effective_length is not None:
            raise ValueError(
                "a truncated or unsupported RULER task must carry "
                "effective_length=None, never the intended length"
            )
        return RulerLengthReport(
            ruler_task=ruler_task,
            requested_length=requested_length,
            effective_length=None,
            truncated=True,
            provider_reported_tokens=provider_reported_tokens,
            normalized_length=normalized_length,
            template_overhead_tokens=template_overhead_tokens,
        )
    if effective_length != requested_length:
        raise ValueError(
            f"a delivered RULER task must carry effective_length == requested_length "
            f"({requested_length}), got {effective_length!r}"
        )
    return RulerLengthReport(
        ruler_task=ruler_task,
        requested_length=requested_length,
        effective_length=effective_length,
        truncated=False,
        provider_reported_tokens=provider_reported_tokens,
        normalized_length=normalized_length,
        template_overhead_tokens=template_overhead_tokens,
    )


def summarize_cell(
    grades: Sequence[GradeResult],
    *,
    ruler_task: str,
    requested_length: int,
    planned: int,
) -> RulerCellSummary:
    """Aggregate one (task, length) cell over its eligible denominator.

    ``planned`` is the declared sample count and is shown even when
    ``eligible`` is zero, so truncation keeps its denominator. ``accuracy``
    is ``None`` when ``eligible`` is zero, never ``0.0``.
    """
    if not is_supported_task(ruler_task):
        raise UnknownTaskError(f"task {ruler_task!r} is not in the pinned RULER v1 set")
    if not is_supported_length(requested_length):
        raise UnsupportedLengthError(
            f"length {requested_length!r} is not a pinned RULER v1 context length"
        )
    if planned < 0:
        raise ValueError(f"planned must be non-negative, got {planned}")
    eligible = sum(1 for grade in grades if grade.counts_toward_accuracy)
    if eligible > planned and planned > 0:
        raise ValueError(
            f"eligible ({eligible}) exceeds planned ({planned}) for {ruler_task}@{requested_length}"
        )
    correct = sum(
        1 for grade in grades if grade.counts_toward_accuracy and grade.correctness == "pass"
    )
    status: CellStatus = "supported" if eligible else "unavailable"
    return RulerCellSummary(
        ruler_task=ruler_task,
        requested_length=requested_length,
        correct=correct,
        eligible=eligible,
        planned=planned,
        accuracy=(correct / eligible) if eligible else None,
        status=status,
    )


def summarize(
    grades_by_cell: Mapping[tuple[str, int], Sequence[GradeResult]],
    *,
    planned_by_cell: Mapping[tuple[str, int], int],
) -> RulerSummary:
    """Aggregate cells into the unweighted overall mean for the pinned seed.

    Every planned cell appears in the output even when it has no grades, so
    truncation keeps its denominator. The overall mean is ``None`` unless
    every planned cell has at least one eligible sample; a partial run stays
    incomplete and never reports a zero-filled average.
    """
    for key in (*grades_by_cell, *planned_by_cell):
        task_name, length = key
        if not is_supported_task(task_name):
            raise UnknownTaskError(f"task {task_name!r} is not in the pinned RULER v1 set")
        if not is_supported_length(length):
            raise UnsupportedLengthError(
                f"length {length!r} is not a pinned RULER v1 context length"
            )
    for key, planned in planned_by_cell.items():
        if planned < 0:
            raise ValueError(f"planned must be non-negative, got {planned} for {key}")
    graded_keys = set(grades_by_cell)
    planned_keys = set(planned_by_cell)
    unplanned = sorted(graded_keys - planned_keys)
    if unplanned:
        raise ValueError(
            f"cells {unplanned} were graded but have no planned denominator; "
            "an unplanned cell cannot enter a published mean"
        )
    cells: list[RulerCellSummary] = []
    for key in sorted(planned_keys):
        task_name, length = key
        cells.append(
            summarize_cell(
                grades_by_cell.get(key, ()),
                ruler_task=task_name,
                requested_length=length,
                planned=planned_by_cell[key],
            )
        )
    complete = bool(cells) and all(cell.eligible > 0 for cell in cells)
    overall: float | None = None
    if complete:
        accuracies = [cell.accuracy for cell in cells if cell.accuracy is not None]
        if len(accuracies) == len(cells):
            overall = sum(accuracies) / len(accuracies)
    return RulerSummary(
        per_cell=tuple(cells),
        overall_accuracy=overall,
        complete=complete,
        generation_seed=GENERATION_SEED,
        comparability_note=TOKENIZATION_CAVEAT,
    )


__all__ = [
    "GENERATION_SEED",
    "GENERATION_TEMPERATURE",
    "GENERATION_TOP_P",
    "GRADER_VERSION",
    "NEMO_SKILLS_BRANCH",
    "NEMO_SKILLS_REPO",
    "OFFICIAL_EVALUATOR_MODULE",
    "PINNED_RULER_BRANCH",
    "PINNED_RULER_COMMIT",
    "PROMPTS_IDENTICAL_ACROSS_MODELS",
    "RULER_SAMPLES_PER_TASK",
    "RULER_TASK_COUNT",
    "RULER_TOTAL_ITEMS",
    "RULER_V1_TASKS",
    "SUPPORTED_LENGTHS",
    "TOKENIZATION_CAVEAT",
    "CellStatus",
    "DatasetChecksumMismatch",
    "DatasetRevisionMismatch",
    "RulerCellSummary",
    "RulerError",
    "RulerLengthReport",
    "RulerSummary",
    "RulerTaskSettings",
    "UnknownTaskError",
    "UnsupportedLengthError",
    "build_task_settings",
    "deterministic_case_id",
    "dispatch_request",
    "generation_settings",
    "grade_accepted_sample",
    "grade_generation",
    "is_malformed",
    "is_supported_length",
    "is_supported_task",
    "report_length",
    "score_response",
    "sha256_file",
    "summarize",
    "summarize_cell",
    "verify_ruler_branch",
    "verify_ruler_checksum",
    "verify_ruler_revision",
]
