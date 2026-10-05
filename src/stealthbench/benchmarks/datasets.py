"""Pinned dataset loading and deterministic sample manifests (G05 T05A).

Three properties in here carry the gate:

* **Selection is seeded and stable.** Items are sorted by ``item_id`` and then
  ordered by a hash of ``(seed, item_id)``, so the same seed and the same item
  pool always yield the same ordered selection regardless of file order, and a
  different seed (overwhelmingly) yields a different one. Hash ordering is used
  instead of ``random.shuffle`` so the result does not drift with the
  interpreter's RNG implementation.
* **Identity is verified, never assumed.** The dataset revision must equal the
  pinned revision from ``docs/upstream-inventory.md`` and the file's sha256 must
  equal the recorded checksum. A mismatch raises instead of substituting.
* **Gold stays evaluator-only.** Prompts become ``ModelRequest``; gold answers
  become ``EvaluationPayload``. There is no serialization path from one to the
  other, and ``assert_no_gold_leakage`` re-checks that before anything dispatches.

Offline by construction: this module only reads local files. There is no
transport, no socket and no credential in it.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from pydantic import BaseModel, ConfigDict, Field

from stealthbench.schemas.campaign import CampaignManifest
from stealthbench.schemas.hashing import content_digest
from stealthbench.schemas.manifest import PromptRef, prompt_hash
from stealthbench.schemas.results import (
    EVALUATOR_ONLY_FIELDS,
    EvaluationPayload,
    Message,
    ModelRequest,
    TaskSpec,
)

#: IFEval pin from docs/upstream-inventory.md (verified 2026-10-02).
PINNED_IFEVAL_DATASET_ID: Final[str] = "google/IFEval"
PINNED_IFEVAL_DATASET_REVISION: Final[str] = "sha966cd89545d6b6acfd7638bc708b98261ca58e84"
PINNED_IFEVAL_EVALUATOR_ID: Final[str] = "instruction_following_eval"
PINNED_IFEVAL_EVALUATOR_REVISION: Final[str] = "e49bbfe381c9c0e564b937f1c4e163a2273c65cc"
PINNED_IFEVAL_SPLIT: Final[str] = "train"
PINNED_IFEVAL_ITEM_COUNT: Final[int] = 541

#: Record keys that may carry a gold answer in a raw JSONL row. Checked in order;
#: the first present string wins. Everything else (instruction ids, kwargs)
#: stays in ``DatasetItem.metadata`` as declared, evaluator-side context.
_GOLD_KEYS: Final[tuple[str, ...]] = ("gold_answer", "gold", "answer", "solution")


class DatasetError(ValueError):
    """Base for pinned-dataset failures. A ``ValueError`` so schema-style checks catch it."""


class DatasetChecksumMismatch(DatasetError):
    """A dataset file's sha256 does not match the recorded checksum."""


class DatasetRevisionMismatch(DatasetError):
    """A dataset revision does not match the pinned revision. No substitution."""


class DatasetNotMaterialized(DatasetError):
    """Dispatch was attempted before the frozen item selection ran."""


class UnknownDatasetError(DatasetError):
    """No pinned dataset is registered for a benchmark id."""


class PinnedDataset(BaseModel):
    """The frozen identity of one benchmark's dataset and evaluator."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    benchmark_id: str = Field(min_length=1)
    dataset_id: str = Field(min_length=1)
    dataset_revision: str = Field(min_length=1)
    evaluator_id: str = Field(min_length=1)
    evaluator_revision: str = Field(min_length=1)
    split: str | None = None
    item_count: int | None = Field(default=None, gt=0)


PINNED_DATASETS: Final[tuple[PinnedDataset, ...]] = (
    PinnedDataset(
        benchmark_id="ifeval",
        dataset_id=PINNED_IFEVAL_DATASET_ID,
        dataset_revision=PINNED_IFEVAL_DATASET_REVISION,
        evaluator_id=PINNED_IFEVAL_EVALUATOR_ID,
        evaluator_revision=PINNED_IFEVAL_EVALUATOR_REVISION,
        split=PINNED_IFEVAL_SPLIT,
        item_count=PINNED_IFEVAL_ITEM_COUNT,
    ),
)

_PINNED_BY_BENCHMARK: Final[dict[str, PinnedDataset]] = {
    pinned.benchmark_id: pinned for pinned in PINNED_DATASETS
}


class DatasetItem(BaseModel):
    """One dataset row held in memory.

    ``prompt`` is the only field that may enter a ``ModelRequest``.
    ``gold_answer`` and ``metadata`` are evaluator-only and travel in the
    ``EvaluationPayload``. Items themselves are never hashed into a manifest;
    the frozen ``item_ids`` tuple is the manifest identity.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    item_id: str = Field(min_length=1, max_length=256)
    prompt: str = Field(min_length=1)
    gold_answer: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


def pinned_dataset_for(benchmark_id: str) -> PinnedDataset:
    """Return the pin for ``benchmark_id`` or raise instead of guessing."""
    try:
        return _PINNED_BY_BENCHMARK[benchmark_id]
    except KeyError as exc:
        raise UnknownDatasetError(
            f"no pinned dataset for benchmark {benchmark_id!r}; refusing to substitute"
        ) from exc


def sha256_file(path: Path, *, chunk_size: int = 1 << 20) -> str:
    """Hash a file's raw bytes. The file identity, not a parsed representation."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file_checksum(path: Path, *, expected_sha256: str) -> str:
    """Return the file digest, or raise when it differs from ``expected_sha256``."""
    actual = sha256_file(path)
    if not hmac.compare_digest(actual.lower(), expected_sha256.lower()):
        raise DatasetChecksumMismatch(
            f"dataset file {path} hashes to {actual} "
            f"but the manifest records {expected_sha256}; refusing to run"
        )
    return actual


def verify_dataset_revision(
    *, actual: str | None, expected: str, benchmark_id: str = "ifeval"
) -> str:
    """Return ``actual`` when it equals the pin, else raise rather than substitute."""
    if actual != expected:
        raise DatasetRevisionMismatch(
            f"{benchmark_id} dataset revision {actual!r} does not match "
            f"pinned revision {expected!r}; refusing to substitute"
        )
    return actual


def load_ifeval_items(
    path: Path,
    *,
    dataset_revision: str | None = None,
    expected_sha256: str | None = None,
) -> tuple[DatasetItem, ...]:
    """Load IFEval ``{key, prompt, ...}`` JSONL rows from a local file.

    The official ``input_data.jsonl`` carries ``key`` (int or str) and ``prompt``
    (str); instruction ids and kwargs, when present, are kept as declared
    metadata. A gold-bearing key (``gold_answer``/``gold``/``answer``/``solution``)
    is captured as ``gold_answer`` so synthetic fixtures can exercise the
    evaluator-only path; the official dump has none and yields ``None``.

    When ``expected_sha256`` is given the file is hashed before parsing, and when
    ``dataset_revision`` is given it must equal the pin. Both failures raise.
    """
    resolved = Path(path)
    if expected_sha256 is not None:
        verify_file_checksum(resolved, expected_sha256=expected_sha256)
    if dataset_revision is not None:
        verify_dataset_revision(
            actual=dataset_revision,
            expected=PINNED_IFEVAL_DATASET_REVISION,
        )
    try:
        text = resolved.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise
    except OSError as exc:
        raise DatasetError(f"cannot read dataset file {resolved}: {exc}") from exc

    items: list[DatasetItem] = []
    seen: set[str] = set()
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record: Any = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DatasetError(f"{resolved}:{line_number} is not valid JSON: {exc}") from exc
        if not isinstance(record, dict):
            raise DatasetError(f"{resolved}:{line_number} is not a JSON object")
        raw_key = record.get("key")
        raw_prompt = record.get("prompt")
        if raw_key is None or raw_prompt is None:
            raise DatasetError(
                f"{resolved}:{line_number} must carry 'key' and 'prompt' "
                "(IFEval input_data.jsonl fields)"
            )
        item_id = str(raw_key)
        if not item_id:
            raise DatasetError(f"{resolved}:{line_number} has an empty key")
        if not isinstance(raw_prompt, str) or not raw_prompt:
            raise DatasetError(f"{resolved}:{line_number} has a non-string prompt")
        if item_id in seen:
            raise DatasetError(f"duplicate item id {item_id!r} at {resolved}:{line_number}")
        seen.add(item_id)
        gold: str | None = None
        for gold_key in _GOLD_KEYS:
            candidate = record.get(gold_key)
            if candidate is not None:
                if not isinstance(candidate, str):
                    raise DatasetError(
                        f"{resolved}:{line_number} field {gold_key!r} must be a string"
                    )
                gold = candidate
                break
        excluded = {"key", "prompt", *list(_GOLD_KEYS)}
        metadata: dict[str, Any] = {
            key: value for key, value in record.items() if key not in excluded
        }
        items.append(
            DatasetItem(item_id=item_id, prompt=raw_prompt, gold_answer=gold, metadata=metadata)
        )
    if not items:
        raise DatasetError(f"dataset file {resolved} contains no items")
    return tuple(items)


def _selection_digest(seed: int, item_id: str) -> str:
    return content_digest({"seed": seed, "item_id": item_id})


def select_items(
    items: Sequence[DatasetItem], *, seed: int, count: int | None
) -> tuple[DatasetItem, ...]:
    """Deterministically order and truncate ``items``.

    The pool is first sorted by ``item_id`` so file order cannot leak in, then
    ordered by ``sha256(seed, item_id)``. The same seed and pool always give the
    same ordered selection; a different seed (overwhelmingly) gives a different
    one. ``count=None`` selects the whole split in that deterministic order.
    ``count`` must be positive when given: an unknown size is ``None``, never 0.
    """
    if count is not None and count <= 0:
        raise ValueError(
            f"count must be positive or None (unknown stays None, never 0), got {count}"
        )
    if not items:
        raise DatasetError("cannot select from an empty item pool")
    if count is not None and count > len(items):
        raise ValueError(
            f"cannot select {count} items from a pool of {len(items)}; refusing to invent items"
        )
    ordered = sorted(items, key=lambda item: item.item_id)
    ranked = sorted(ordered, key=lambda item: _selection_digest(seed, item.item_id))
    if count is None:
        return tuple(ranked)
    return tuple(ranked[:count])


def select_item_ids(
    items: Sequence[DatasetItem], *, seed: int, count: int | None
) -> tuple[str, ...]:
    """The frozen ``item_ids`` for a selection, in dispatch order."""
    return tuple(item.item_id for item in select_items(items, seed=seed, count=count))


def resolve_planned_count(
    declared_item_count: int | None, expected_item_count: int | None
) -> int | None:
    """Planned items before the frozen selection exists, or ``None`` when unknown.

    ``None`` means the size is genuinely not yet knowable (only an enumeration
    procedure is declared). It is never 0: a zero would silently disable the
    request-cap check that stops a campaign from dying mid-run.
    """
    for name, value in (
        ("declared_item_count", declared_item_count),
        ("expected_item_count", expected_item_count),
    ):
        if value is not None and value <= 0:
            raise ValueError(f"{name} must be positive or None, got {value}")
    if declared_item_count is not None:
        return declared_item_count
    return expected_item_count


def build_task_specs(
    *,
    benchmark_id: str,
    campaign_id: str,
    endpoint_id: str,
    items: Sequence[DatasetItem],
    evaluator_id: str,
    evaluator_revision: str,
    max_output_tokens: int = 512,
    repeats: int = 1,
    temperature: float = 0.0,
    top_p: float = 1.0,
) -> tuple[TaskSpec, ...]:
    """Build one ``TaskSpec`` per item per repeat, gold kept evaluator-only.

    Selection order is preserved: tasks follow ``items`` order with ``repeat_id``
    varying fastest. Each recorded ``prompt_artifact_hash`` is recomputed from
    the held request, so provenance describes the dispatchable bytes.
    """
    if max_output_tokens <= 0:
        raise ValueError(f"max_output_tokens must be positive, got {max_output_tokens}")
    if repeats < 1:
        raise ValueError(f"repeats must be at least 1, got {repeats}")
    if not items:
        raise DatasetError("cannot build tasks from an empty selection")
    if not evaluator_revision:
        raise ValueError("evaluator_revision must be recorded; refusing an unpinned grade")

    specs: list[TaskSpec] = []
    for item in items:
        request = ModelRequest(
            messages=(Message(role="user", content=item.prompt),),
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        evaluation = EvaluationPayload(
            gold_answer=item.gold_answer,
            evaluator_id=evaluator_id,
            evaluator_revision=evaluator_revision,
            declared_metadata=dict(item.metadata),
        )
        # Structural check now, not at dispatch: a gold string that already sits
        # inside the prompt is a fixture bug, and failing here names the item.
        evaluation.assert_absent_from(request)
        prompt_ref = PromptRef.from_request(request)
        for repeat_id in range(1, repeats + 1):
            specs.append(
                TaskSpec(
                    benchmark_id=benchmark_id,
                    item_id=item.item_id,
                    campaign_id=campaign_id,
                    endpoint_id=endpoint_id,
                    repeat_id=repeat_id,
                    prompt_artifact_hash=prompt_hash(prompt_ref),
                    request=request,
                    evaluation=evaluation,
                    declared_metadata=dict(item.metadata),
                )
            )
    return tuple(specs)


def assert_no_gold_leakage(tasks: Sequence[TaskSpec]) -> None:
    """Raise if any task's dispatchable request carries evaluator-only data.

    Checks the closed ``ModelRequest`` schema, re-verifies each recorded prompt
    hash, and asserts no gold value or hidden artifact string appears in the
    serialized request. Call before dispatch and cover with a dedicated test so
    the deliberate "include gold fields in a request" mutation must fail.
    """
    leaked_schema = set(ModelRequest.model_fields) & set(EVALUATOR_ONLY_FIELDS)
    if leaked_schema:
        raise ValueError(f"evaluator-only fields leaked into ModelRequest: {sorted(leaked_schema)}")
    for task in tasks:
        # Re-verifies the prompt hash and asserts evaluator absence; raises either way.
        task.safe_request()
        evaluation = task.evaluation
        serialized = task.request.model_dump_json()
        offenders = [
            value
            for value in (evaluation.gold_answer, *evaluation.hidden_artifacts)
            if value and value in serialized
        ]
        if offenders:
            raise ValueError(
                f"task {task.benchmark_id}::{task.item_id} would send evaluator-only "
                "data to a provider"
            )


def materialize_manifest(
    manifest: CampaignManifest,
    *,
    selections: Mapping[str, Sequence[str]],
    dataset_revisions: Mapping[str, str] | None = None,
    evaluator_revisions: Mapping[str, str] | None = None,
) -> CampaignManifest:
    """Freeze ``item_ids`` (and revisions) into a copy and mark it materialized.

    The input stays unmaterialized; the returned manifest carries the frozen
    selection in the given dispatch order. Unknown benchmarks, empty selections
    and re-materializing an already materialized campaign all raise: an unknown
    size stays unmaterialized (``None``, never 0 items), it is never frozen as
    an empty list and relabelled complete.
    """
    if manifest.materialized:
        raise DatasetNotMaterialized(
            f"campaign {manifest.campaign_id} is already materialized; "
            "a rerun uses a new campaign_id so the original report is preserved"
        )
    known = {spec.benchmark_id for spec in manifest.benchmarks}
    unknown = sorted(set(selections) - known)
    if unknown:
        raise DatasetError(f"selections name unknown benchmarks: {unknown}")

    rebuilt: list[dict[str, Any]] = []
    for spec in manifest.benchmarks:
        if spec.benchmark_id not in selections:
            raise DatasetError(
                f"benchmark {spec.benchmark_id!r} has no selection; "
                "an unknown size stays unmaterialized, it is never frozen as empty"
            )
        item_ids = tuple(selections[spec.benchmark_id])
        if not item_ids:
            raise DatasetError(
                f"benchmark {spec.benchmark_id!r} selected zero items; unknown stays None, never 0"
            )
        if len(set(item_ids)) != len(item_ids):
            duplicates = sorted({i for i in item_ids if item_ids.count(i) > 1})
            raise DatasetError(f"duplicate item ids for {spec.benchmark_id}: {duplicates}")
        dataset_revision = (dataset_revisions or {}).get(spec.benchmark_id)
        if dataset_revision is None:
            dataset_revision = spec.dataset_revision
        if dataset_revision is None and spec.benchmark_id in _PINNED_BY_BENCHMARK:
            dataset_revision = _PINNED_BY_BENCHMARK[spec.benchmark_id].dataset_revision
        evaluator_revision = (evaluator_revisions or {}).get(spec.benchmark_id)
        if evaluator_revision is None:
            evaluator_revision = spec.evaluator_revision
        if evaluator_revision is None and spec.benchmark_id in _PINNED_BY_BENCHMARK:
            evaluator_revision = _PINNED_BY_BENCHMARK[spec.benchmark_id].evaluator_revision
        payload: dict[str, Any] = spec.model_dump(mode="json")
        payload["item_ids"] = list(item_ids)
        payload["dataset_revision"] = dataset_revision
        payload["evaluator_revision"] = evaluator_revision
        recorded = payload.get("declared_item_count") or payload.get("expected_item_count")
        if recorded is not None and int(recorded) != len(item_ids):
            raise DatasetError(
                f"{spec.benchmark_id} declares {recorded} items but selected "
                f"{len(item_ids)}; refusing to freeze a contradictory manifest"
            )
        rebuilt.append(payload)

    manifest_payload: dict[str, Any] = manifest.model_dump(mode="json")
    manifest_payload["benchmarks"] = rebuilt
    manifest_payload["materialized"] = True
    # Re-validate through the frozen contract so duplicate ids, conflicting caps
    # and bad revisions fail here, before anything dispatches.
    return CampaignManifest.model_validate(manifest_payload)


def ensure_dispatchable(manifest: CampaignManifest) -> None:
    """Raise unless the campaign may be dispatched.

    ``materialized=false`` is the normal pre-selection state of a valid profile,
    not malformed input, so it is reported as a dispatch blocker rather than a
    validation error -- and it still blocks. Live campaigns additionally need
    their operator spending cap; see ``CampaignManifest.dispatch_blockers``.
    """
    if not manifest.materialized:
        raise DatasetNotMaterialized(
            f"campaign {manifest.campaign_id}: item selection has not run; "
            "item_ids are empty so there is nothing to dispatch"
        )
    blockers = manifest.dispatch_blockers()
    if blockers:
        raise DatasetNotMaterialized(
            f"campaign {manifest.campaign_id} is not dispatchable: {'; '.join(blockers)}"
        )


def build_dispatch_tasks(
    manifest: CampaignManifest,
    *,
    items_by_benchmark: Mapping[str, Sequence[DatasetItem]],
    endpoint_id: str,
    max_output_tokens: int = 512,
) -> tuple[TaskSpec, ...]:
    """Build dispatchable tasks in frozen ``item_ids`` order, or raise.

    Refuses unmaterialized campaigns first, then binds each frozen id to its
    loaded item. The spec's recorded ``evaluator_revision`` is what grades the
    task, so an unrecorded revision blocks rather than defaulting.
    """
    ensure_dispatchable(manifest)
    tasks: list[TaskSpec] = []
    for spec in manifest.benchmarks:
        if spec.evaluator_revision is None:
            raise DatasetError(
                f"benchmark {spec.benchmark_id!r} records no evaluator_revision; "
                "refusing an unpinned grade"
            )
        if spec.dataset_revision is None:
            raise DatasetError(
                f"benchmark {spec.benchmark_id!r} records no dataset_revision; "
                "refusing an unpinned dataset"
            )
        pool = items_by_benchmark.get(spec.benchmark_id)
        if pool is None:
            raise DatasetError(
                f"no loaded items for benchmark {spec.benchmark_id!r}; "
                "load the pinned revision before dispatching"
            )
        by_id = {item.item_id: item for item in pool}
        ordered: list[DatasetItem] = []
        for item_id in spec.item_ids:
            try:
                ordered.append(by_id[item_id])
            except KeyError as exc:
                raise DatasetError(
                    f"benchmark {spec.benchmark_id!r} selects {item_id!r} "
                    "which is absent from the loaded pinned revision"
                ) from exc
        tasks.extend(
            build_task_specs(
                benchmark_id=spec.benchmark_id,
                campaign_id=manifest.campaign_id,
                endpoint_id=endpoint_id,
                items=ordered,
                evaluator_id=f"{spec.benchmark_id}-evaluator",
                evaluator_revision=spec.evaluator_revision,
                max_output_tokens=max_output_tokens,
                repeats=spec.repeats,
            )
        )
    assert_no_gold_leakage(tasks)
    return tuple(tasks)


__all__ = [
    "PINNED_DATASETS",
    "PINNED_IFEVAL_DATASET_ID",
    "PINNED_IFEVAL_DATASET_REVISION",
    "PINNED_IFEVAL_EVALUATOR_ID",
    "PINNED_IFEVAL_EVALUATOR_REVISION",
    "PINNED_IFEVAL_ITEM_COUNT",
    "PINNED_IFEVAL_SPLIT",
    "DatasetChecksumMismatch",
    "DatasetError",
    "DatasetItem",
    "DatasetNotMaterialized",
    "DatasetRevisionMismatch",
    "PinnedDataset",
    "UnknownDatasetError",
    "assert_no_gold_leakage",
    "build_dispatch_tasks",
    "build_task_specs",
    "ensure_dispatchable",
    "load_ifeval_items",
    "materialize_manifest",
    "pinned_dataset_for",
    "resolve_planned_count",
    "select_item_ids",
    "select_items",
    "sha256_file",
    "verify_dataset_revision",
    "verify_file_checksum",
]
