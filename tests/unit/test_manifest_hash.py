"""Manifest hashing and artifact identity.

A sample must be traceable to the exact manifest and prompt that produced it, so
the manifest hash has to be stable against reserialization and sensitive to every
field that changes what a run means.

The rule that is easy to get wrong: **object key order is insignificant, sequence
order is significant.** Reordering the keys of a manifest must not invalidate every
stored artifact; reordering its ``item_ids`` must, because that is a different
frozen selection.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from stealthbench.schemas.campaign import CampaignManifest
from stealthbench.schemas.hashing import (
    NonCanonicalValue,
    canonical_json,
    content_digest,
    file_digest,
    is_digest,
    sort_for_hashing,
    stable_id,
)
from stealthbench.schemas.manifest import (
    PromptRef,
    manifest_hash,
    prompt_hash,
    trace_key,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _manifest_payload() -> dict[str, Any]:
    return json.loads((REPO_ROOT / "configs" / "offline-demo.json").read_text(encoding="utf-8"))


def _manifest() -> CampaignManifest:
    return CampaignManifest.model_validate(_manifest_payload())


# ---------------------------------------------------------------------------
# Insignificant ordering
# ---------------------------------------------------------------------------


def test_object_key_order_does_not_change_the_hash() -> None:
    left = {"a": 1, "b": {"c": 2, "d": 3}}
    right = {"b": {"d": 3, "c": 2}, "a": 1}
    assert canonical_json(left) == canonical_json(right)
    assert content_digest(left) == content_digest(right)


def test_deeply_nested_key_order_does_not_change_the_hash() -> None:
    left = {"x": [{"p": 1, "q": 2}], "y": {"z": {"m": 1, "n": 2}}}
    right = {"y": {"z": {"n": 2, "m": 1}}, "x": [{"q": 2, "p": 1}]}
    assert content_digest(left) == content_digest(right)


def test_manifest_key_order_does_not_change_the_manifest_hash() -> None:
    """Reserializing a manifest must not invalidate every artifact bound to it."""
    payload = _manifest_payload()
    reordered = dict(reversed(list(payload.items())))
    for section in ("endpoints", "benchmarks"):
        if isinstance(reordered.get(section), list):
            reordered[section] = [dict(reversed(list(e.items()))) for e in reordered[section]]
    assert list(reordered) == list(reversed(list(payload)))
    assert manifest_hash(_manifest_payload()) == manifest_hash(reordered)


def test_sequence_order_is_significant() -> None:
    """item_ids order is semantic: a different order is a different selection."""
    assert content_digest([1, 2, 3]) != content_digest([3, 2, 1])


def test_sort_for_hashing_matches_the_canonical_digest() -> None:
    payload = {"b": 1, "a": {"d": 2, "c": [3, 1]}}
    assert content_digest(payload) == content_digest(sort_for_hashing(payload))


# ---------------------------------------------------------------------------
# Significant changes
# ---------------------------------------------------------------------------


def _mutate(**changes: Any) -> dict[str, Any]:
    payload = _manifest_payload()
    payload.update(changes)
    return payload


def test_changing_a_prompt_setting_changes_the_hash() -> None:
    baseline = manifest_hash(_manifest_payload())
    assert manifest_hash(_mutate(seed=_manifest().seed + 1)) != baseline


def test_changing_generation_settings_changes_the_hash() -> None:
    payload = _manifest_payload()
    baseline = manifest_hash(payload)
    payload["generation"]["max_output_tokens"] += 1
    assert manifest_hash(payload) != baseline


def test_changing_a_limit_changes_the_hash() -> None:
    payload = _manifest_payload()
    baseline = manifest_hash(payload)
    payload["limits"]["max_concurrency"] += 1
    assert manifest_hash(payload) != baseline


def test_changing_the_retry_policy_changes_the_hash() -> None:
    payload = _manifest_payload()
    baseline = manifest_hash(payload)
    payload["retry_policy"]["max_attempts"] += 1
    assert manifest_hash(payload) != baseline


def test_changing_the_campaign_id_changes_the_hash() -> None:
    baseline = manifest_hash(_manifest_payload())
    assert manifest_hash(_mutate(campaign_id="offline-demo-2")) != baseline


def test_changing_the_score_version_changes_the_hash() -> None:
    baseline = manifest_hash(_manifest_payload())
    assert manifest_hash(_mutate(score_version="stealthbench-core-v2")) != baseline


def test_changing_the_materialization_state_changes_the_hash() -> None:
    """Selection state is part of what a run means, and each side must be valid."""
    unmaterialized = _manifest_payload()
    unmaterialized["materialized"] = False
    for benchmark in unmaterialized["benchmarks"]:
        benchmark["item_ids"] = []
    assert manifest_hash(unmaterialized) != manifest_hash(_manifest_payload())


def test_changing_an_item_selection_changes_the_hash() -> None:
    payload = _manifest_payload()
    payload["benchmarks"][0]["expected_item_count"] = 2
    payload["benchmarks"][0]["item_ids"] = ["syn-if-001", "syn-if-004"]
    changed = manifest_hash(payload)
    assert changed != manifest_hash(_manifest_payload())


def test_changing_the_evaluator_revision_changes_the_hash() -> None:
    payload = _manifest_payload()
    baseline = manifest_hash(payload)
    payload["benchmarks"][0]["evaluator_revision"] = "google-research@0000000"
    assert manifest_hash(payload) != baseline


def test_changing_the_dataset_revision_changes_the_hash() -> None:
    payload = _manifest_payload()
    baseline = manifest_hash(payload)
    payload["benchmarks"][0]["dataset_revision"] = "fixture-2027-01-01"
    assert manifest_hash(payload) != baseline


def test_changing_an_endpoint_capability_changes_the_hash() -> None:
    payload = _manifest_payload()
    baseline = manifest_hash(payload)
    payload["endpoints"][0]["capabilities"]["tool_calls"] = True
    assert manifest_hash(payload) != baseline


def test_changing_a_price_snapshot_changes_the_hash() -> None:
    payload = _manifest_payload()
    baseline = manifest_hash(payload)
    payload["endpoints"][0]["pricing"] = {"input_per_mtok": 1.5, "output_per_mtok": None}
    assert manifest_hash(payload) != baseline


def test_identical_manifests_hash_identically() -> None:
    assert manifest_hash(_manifest_payload()) == manifest_hash(_manifest_payload())


# ---------------------------------------------------------------------------
# Prompt hashing
# ---------------------------------------------------------------------------


def test_prompt_hash_depends_on_content_not_on_key_order() -> None:
    left = PromptRef(messages=[{"role": "user", "content": "hi"}], max_output_tokens=16)
    right = PromptRef(messages=[{"content": "hi", "role": "user"}], max_output_tokens=16)
    assert prompt_hash(left) == prompt_hash(right)


def test_prompt_hash_changes_with_content() -> None:
    short = PromptRef(messages=[{"role": "user", "content": "hi"}], max_output_tokens=16)
    longer = PromptRef(messages=[{"role": "user", "content": "hi there"}], max_output_tokens=16)
    assert prompt_hash(short) != prompt_hash(longer)


def test_prompt_hash_changes_with_output_cap() -> None:
    """The requested cap is part of the prompt's identity."""
    small = PromptRef(messages=[{"role": "user", "content": "hi"}], max_output_tokens=16)
    large = PromptRef(messages=[{"role": "user", "content": "hi"}], max_output_tokens=32)
    assert prompt_hash(small) != prompt_hash(large)


def test_prompt_hash_is_a_sha256_digest() -> None:
    ref = PromptRef(messages=[{"role": "user", "content": "hi"}], max_output_tokens=16)
    assert is_digest(prompt_hash(ref))


# ---------------------------------------------------------------------------
# Traceability
# ---------------------------------------------------------------------------


def test_trace_key_binds_manifest_and_prompt() -> None:
    manifest = _manifest()
    ref = PromptRef(messages=[{"role": "user", "content": "hi"}], max_output_tokens=16)
    key = trace_key(manifest, ref)
    assert is_digest(key)
    assert len(key) == 64


def test_trace_key_changes_with_either_input() -> None:
    manifest = _manifest()
    ref = PromptRef(messages=[{"role": "user", "content": "hi"}], max_output_tokens=16)
    other = PromptRef(messages=[{"role": "user", "content": "different"}], max_output_tokens=16)
    assert trace_key(manifest, ref) != trace_key(manifest, other)
    changed = CampaignManifest.model_validate(
        {**_manifest_payload(), "campaign_id": "offline-demo-x"}
    )
    assert trace_key(manifest, ref) != trace_key(changed, ref)


def test_manifest_hash_is_a_sha256_digest() -> None:
    assert is_digest(manifest_hash(_manifest_payload()))


# ---------------------------------------------------------------------------
# Non-finite and malformed input
# ---------------------------------------------------------------------------


def test_non_finite_floats_are_rejected() -> None:
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(NonCanonicalValue, match="not finite"):
            canonical_json({"value": bad})


def test_non_finite_nested_deep_in_is_rejected_with_a_path() -> None:
    with pytest.raises(NonCanonicalValue, match=r"\$\.a\[1\]\.b is not finite"):
        canonical_json({"a": [0, {"b": float("nan")}]})


def test_unsupported_types_are_rejected() -> None:
    with pytest.raises(NonCanonicalValue, match="unsupported type"):
        canonical_json({"value": object()})


def test_non_string_keys_are_rejected() -> None:
    with pytest.raises(NonCanonicalValue, match="non-string key"):
        canonical_json({1: "a"})  # type: ignore[dict-item]


def test_canonical_json_is_compact_and_sorted() -> None:
    assert canonical_json({"b": 1, "a": 2}) == '{"a":2,"b":1}'


def test_canonical_json_preserves_unicode_rather_than_escaping() -> None:
    assert canonical_json({"k": "café"}) == '{"k":"café"}'


def test_file_digest_matches_content_digest_of_the_bytes(tmp_path: Path) -> None:
    import hashlib

    artifact = tmp_path / "prompt.txt"
    artifact.write_bytes(b"hello")
    assert file_digest(artifact) == hashlib.sha256(b"hello").hexdigest()


def test_file_digest_distinguishes_different_content(tmp_path: Path) -> None:
    first = tmp_path / "a.txt"
    second = tmp_path / "b.txt"
    first.write_bytes(b"a")
    second.write_bytes(b"b")
    assert file_digest(first) != file_digest(second)


# ---------------------------------------------------------------------------
# Stable identifiers
# ---------------------------------------------------------------------------


def test_stable_id_is_prefixed_and_reproducible() -> None:
    first = stable_id("sample", {"a": 1})
    assert first == stable_id("sample", {"a": 1})
    assert first.startswith("sample-")
    assert is_digest("0" * 64)


def test_stable_id_rejects_an_empty_prefix() -> None:
    with pytest.raises(ValueError, match="non-empty prefix"):
        stable_id("", {"a": 1})


def test_stable_id_rejects_a_short_length() -> None:
    with pytest.raises(ValueError, match="between 8"):
        stable_id("sample", {"a": 1}, length=4)


# ---------------------------------------------------------------------------
# Invalid manifests never reach a hash
# ---------------------------------------------------------------------------


def test_an_invalid_manifest_fails_before_it_can_be_hashed() -> None:
    """G01's gate: invalid manifests fail before any request, and certainly before storage."""
    payload = _manifest_payload()
    payload["schema_version"] = "9.9"
    with pytest.raises(ValidationError):
        CampaignManifest.model_validate(payload)
    with pytest.raises(ValidationError):
        manifest_hash(payload)


def test_a_manifest_carrying_a_credential_fails_before_hashing() -> None:
    payload = _manifest_payload()
    payload["endpoints"][0]["credential_ref"] = "sk-canary-should-not-be-hashed"
    with pytest.raises(ValidationError, match="credential_ref"):
        manifest_hash(payload)
