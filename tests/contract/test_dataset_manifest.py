"""Contract tests for pinned dataset loading and sample manifests (task T05A).

Acceptance for T05A: selected IDs, revisions, sampling seed and checksums are
deterministic; gold data stays evaluator-only.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from stealthbench.benchmarks.datasets import (
    PINNED_IFEVAL_DATASET_REVISION,
    PINNED_IFEVAL_EVALUATOR_REVISION,
    DatasetChecksumMismatch,
    DatasetItem,
    DatasetNotMaterialized,
    DatasetRevisionMismatch,
    assert_no_gold_leakage,
    build_dispatch_tasks,
    build_task_specs,
    ensure_dispatchable,
    load_ifeval_items,
    materialize_manifest,
    resolve_planned_count,
    select_item_ids,
    select_items,
    verify_dataset_revision,
    verify_file_checksum,
)
from stealthbench.schemas.campaign import BenchmarkSpec, CampaignManifest
from stealthbench.schemas.manifest import PromptRef, prompt_hash
from stealthbench.schemas.results import (
    EVALUATOR_ONLY_FIELDS,
    EvaluationPayload,
    ModelRequest,
    TaskSpec,
)

pytestmark = pytest.mark.contract


def _row(index: int) -> dict[str, Any]:
    """One synthetic IFEval-shaped row with a distinctive, non-overlapping gold."""
    return {
        "key": f"item-{index:03d}",
        "prompt": f"Instruction {index}: respond in exactly {index} sentences.",
        "gold_answer": f"canary-answer-{index:03d}-xyz",
        "instruction_id_list": ["word_count"],
        "kwargs": {"index": index},
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def _fixture_items(count: int = 10) -> tuple[DatasetItem, ...]:
    return tuple(
        DatasetItem(
            item_id=f"item-{index:03d}",
            prompt=f"Instruction {index}: respond in exactly {index} sentences.",
            gold_answer=f"canary-answer-{index:03d}-xyz",
            metadata={"instruction_id_list": ["word_count"]},
        )
        for index in range(count)
    )


def _unmaterialized_manifest(expected: int = 4) -> CampaignManifest:
    return CampaignManifest.model_validate(
        {
            "schema_version": "1.0",
            "campaign_id": "t05a-demo",
            "title": "T05A fixture campaign",
            "description": "Unmaterialized fixture used only for dataset selection tests.",
            "mode": "offline",
            "materialized": False,
            "score_version": "stealthbench-core-v1",
            "seed": 20261002,
            "observation_window": {"started_at": None, "ended_at": None, "timezone": "UTC"},
            "endpoints": [
                {
                    "endpoint_id": "fixture-anonymous-alpha",
                    "alias": "fixture-anonymous-alpha",
                    "route": "fixture",
                    "transport": "fixture",
                    "capabilities": {
                        "streaming": True,
                        "tool_calls": False,
                        "reasoning": False,
                        "usage_reporting": True,
                        "logprobs": False,
                    },
                    "pricing": {
                        "currency": "USD",
                        "input_per_mtok": None,
                        "output_per_mtok": None,
                        "cached_input_per_mtok": None,
                        "reasoning_per_mtok": None,
                        "snapshot_id": None,
                    },
                    "credential_ref": None,
                }
            ],
            "benchmarks": [
                {
                    "benchmark_id": "ifeval",
                    "track": "direct",
                    "category": "instruction_following",
                    "dataset_id": "google/IFEval",
                    "dataset_revision": PINNED_IFEVAL_DATASET_REVISION,
                    "evaluator_revision": PINNED_IFEVAL_EVALUATOR_REVISION,
                    "expected_item_count": expected,
                    "item_ids": [],
                    "repeats": 1,
                }
            ],
            "generation": {"max_output_tokens": 512},
            "retry_policy": {
                "max_attempts": 3,
                "initial_backoff_seconds": 1.0,
                "multiplier": 2.0,
                "max_backoff_seconds": 30.0,
                "retryable_statuses": [408, 429, 500, 502, 503, 504],
            },
            "limits": {
                "max_requests": 1000,
                "max_concurrency": 4,
                "max_input_tokens": 200000,
                "max_output_tokens": 200000,
                "max_total_cost_usd": 1.0,
                "max_wall_seconds": 900,
                "require_cost_bounds": True,
            },
        }
    )


# ---------------------------------------------------------------------------
# Deterministic, seeded selection in a stable order
# ---------------------------------------------------------------------------


def test_selection_is_deterministic_for_the_same_seed() -> None:
    items = _fixture_items(10)
    first = select_item_ids(items, seed=42, count=4)
    second = select_item_ids(items, seed=42, count=4)
    assert first == second
    assert len(first) == 4


def test_selection_order_ignores_input_file_order() -> None:
    items = _fixture_items(10)
    forward = select_item_ids(items, seed=42, count=4)
    backward = select_item_ids(tuple(reversed(items)), seed=42, count=4)
    assert forward == backward


def test_a_different_seed_alters_the_selection() -> None:
    items = _fixture_items(10)
    first = select_item_ids(items, seed=1, count=4)
    second = select_item_ids(items, seed=2, count=4)
    assert first != second, "a seed change that selects identically is not a seed"


def test_full_split_selection_is_still_deterministic() -> None:
    items = _fixture_items(6)
    assert select_item_ids(items, seed=7, count=None) == select_item_ids(
        tuple(reversed(items)), seed=7, count=None
    )
    assert len(select_item_ids(items, seed=7, count=None)) == 6


# ---------------------------------------------------------------------------
# Revision and checksum verification
# ---------------------------------------------------------------------------


def test_checksum_mismatch_fails(tmp_path: Path) -> None:
    path = _write_jsonl(tmp_path / "ifeval.jsonl", [_row(index) for index in range(4)])
    with pytest.raises(DatasetChecksumMismatch):
        verify_file_checksum(path, expected_sha256="0" * 64)
    with pytest.raises(DatasetChecksumMismatch):
        load_ifeval_items(path, expected_sha256="0" * 64)


def test_correct_checksum_passes_independently(tmp_path: Path) -> None:
    """The recorded digest must agree with an independent stdlib hash."""
    path = _write_jsonl(tmp_path / "ifeval.jsonl", [_row(index) for index in range(4)])
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    assert verify_file_checksum(path, expected_sha256=expected) == expected
    assert len(load_ifeval_items(path, expected_sha256=expected)) == 4


def test_revision_mismatch_fails(tmp_path: Path) -> None:
    path = _write_jsonl(tmp_path / "ifeval.jsonl", [_row(index) for index in range(2)])
    with pytest.raises(DatasetRevisionMismatch):
        verify_dataset_revision(actual="wrong-revision", expected="pinned-revision")
    with pytest.raises(DatasetRevisionMismatch):
        load_ifeval_items(path, dataset_revision="wrong-revision")
    assert len(load_ifeval_items(path, dataset_revision=PINNED_IFEVAL_DATASET_REVISION)) == 2


def test_duplicate_item_ids_are_rejected(tmp_path: Path) -> None:
    rows = [_row(0), _row(0)]
    path = _write_jsonl(tmp_path / "dup.jsonl", rows)
    with pytest.raises(Exception, match="duplicate item id"):
        load_ifeval_items(path)


# ---------------------------------------------------------------------------
# Gold answers stay evaluator-only
# ---------------------------------------------------------------------------


def test_gold_answers_never_enter_the_dispatchable_request() -> None:
    items = _fixture_items(4)
    tasks = build_task_specs(
        benchmark_id="ifeval",
        campaign_id="c1",
        endpoint_id="e1",
        items=items,
        evaluator_id="instruction_following_eval",
        evaluator_revision=PINNED_IFEVAL_EVALUATOR_REVISION,
    )
    assert len(tasks) == 4
    assert set(ModelRequest.model_fields) & set(EVALUATOR_ONLY_FIELDS) == set()
    for task in tasks:
        assert task.evaluation.gold_answer is not None
        serialized = task.safe_request().model_dump_json()
        assert task.evaluation.gold_answer not in serialized
    assert_no_gold_leakage(tasks)


def test_a_request_smuggling_gold_is_rejected_at_dispatch() -> None:
    gold = "canary-answer-001-xyz"
    request = ModelRequest(
        messages=[{"role": "user", "content": f"Repeat this back: {gold}"}],
        max_output_tokens=64,
    )
    task = TaskSpec(
        benchmark_id="ifeval",
        item_id="item-001",
        campaign_id="c1",
        endpoint_id="e1",
        repeat_id=1,
        prompt_artifact_hash=prompt_hash(PromptRef.from_request(request)),
        request=request,
        evaluation=EvaluationPayload(
            gold_answer=gold,
            evaluator_id="instruction_following_eval",
            evaluator_revision=PINNED_IFEVAL_EVALUATOR_REVISION,
        ),
    )
    with pytest.raises(ValueError, match="would be sent to a provider"):
        task.safe_request()
    with pytest.raises(ValueError, match="would be sent to a provider"):
        assert_no_gold_leakage([task])


def test_model_request_is_closed_to_gold_fields() -> None:
    with pytest.raises(ValidationError):
        ModelRequest(
            messages=[{"role": "user", "content": "hi"}],
            max_output_tokens=8,
            gold_answer="canary-answer-001-xyz",  # type: ignore[call-arg]
        )


# ---------------------------------------------------------------------------
# Unknown size stays None, never 0; unmaterialized blocks dispatch
# ---------------------------------------------------------------------------


def test_unknown_item_count_stays_none_never_zero() -> None:
    spec = BenchmarkSpec.model_validate(
        {
            "benchmark_id": "ifeval",
            "track": "direct",
            "category": "instruction_following",
            "split_enumeration": "enumerate-train-split",
        }
    )
    assert spec.planned_item_count is None
    assert spec.planned_item_count != 0
    assert resolve_planned_count(None, None) is None
    with pytest.raises(ValueError, match="never 0"):
        select_items(_fixture_items(4), seed=1, count=0)


def test_unmaterialized_manifest_blocks_dispatch() -> None:
    manifest = _unmaterialized_manifest()
    assert manifest.materialized is False
    assert manifest.dispatch_blockers(), "an unmaterialized profile must name a blocker"
    assert manifest.is_dispatchable is False
    with pytest.raises(DatasetNotMaterialized, match="selection has not run"):
        ensure_dispatchable(manifest)
    with pytest.raises(DatasetNotMaterialized):
        build_dispatch_tasks(
            manifest,
            items_by_benchmark={"ifeval": _fixture_items(10)},
            endpoint_id="fixture-anonymous-alpha",
        )


def test_materialization_records_ids_and_revisions() -> None:
    manifest = _unmaterialized_manifest(expected=4)
    items = _fixture_items(10)
    selected = select_items(items, seed=42, count=4)
    materialized = materialize_manifest(
        manifest, selections={"ifeval": [item.item_id for item in selected]}
    )
    assert materialized.materialized is True
    assert manifest.materialized is False, "materialization must not mutate its input"
    (spec,) = materialized.benchmarks
    assert tuple(spec.item_ids) == tuple(item.item_id for item in selected)
    assert spec.dataset_revision == PINNED_IFEVAL_DATASET_REVISION
    assert spec.evaluator_revision == PINNED_IFEVAL_EVALUATOR_REVISION
    ensure_dispatchable(materialized)

    tasks = build_dispatch_tasks(
        materialized,
        items_by_benchmark={"ifeval": items},
        endpoint_id="fixture-anonymous-alpha",
    )
    assert [task.item_id for task in tasks] == list(spec.item_ids)
    assert all(
        task.evaluation.evaluator_revision == PINNED_IFEVAL_EVALUATOR_REVISION for task in tasks
    )
    for task in tasks:
        task.check_prompt_hash()
    assert_no_gold_leakage(tasks)


def test_materialization_refuses_an_empty_selection() -> None:
    manifest = _unmaterialized_manifest()
    with pytest.raises(Exception, match="zero items"):
        materialize_manifest(manifest, selections={"ifeval": []})
