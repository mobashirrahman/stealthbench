"""G11 T11C: ranked reference similarity reports (unit)."""

from __future__ import annotations

import pytest

from stealthbench.fingerprints.features import extract_signature
from stealthbench.fingerprints.similarity import (
    DIVERGENT_BEHAVIOR_NOTE,
    TOKENIZER_ONLY_NOTE,
    ReferenceSignature,
    assert_no_probabilities,
    behavior_similarity,
    build_identity_report,
    combine_signal_similarities,
    exact_match_fraction,
    token_count_similarity,
)
from stealthbench.schemas.results import ProbeValidity, RankedSimilarity

pytestmark = pytest.mark.unit

LIB = "candidate-lib-v1"


def _query(
    counts: list[int | None],
    *,
    baseline: int | None = 8,
    behaviors: dict[str, object] | None = None,
    probe_version: str = "probe-v1",
) -> tuple[object, int | None]:
    ids = [f"p{i}" for i in range(len(counts))]
    observations = {
        probe_id: ([count, count, count] if count is not None else [None, None, None])
        for probe_id, count in zip(ids, counts, strict=True)
    }
    result = extract_signature(
        probe_ids=ids,
        observations=observations,
        baseline_observations=(
            [baseline, baseline, baseline] if baseline is not None else [None, None, None]
        ),
        probe_version=probe_version,
        endpoint_id="ep-anon",
        campaign_id="camp-a",
        behavioral_features=dict(behaviors or {}),
    )
    return result, baseline


def _ref(
    candidate: str,
    counts: list[int | None],
    *,
    baseline: int | None = 8,
    behaviors: dict[str, object] | None = None,
    library: str = LIB,
    probe_version: str = "probe-v1",
) -> ReferenceSignature:
    validity = tuple(
        ProbeValidity.VALID if count is not None else ProbeValidity.MISSING_COUNT
        for count in counts
    )
    return ReferenceSignature(
        candidate_id=candidate,
        candidate_library_revision=library,
        probe_version=probe_version,
        input_counts=tuple(counts),
        validity=validity,
        baseline_count=baseline,
        behavioral_features=dict(behaviors or {}),
    )


def test_exact_match_fraction_counts_only_mutually_valid_probes() -> None:
    similarity, comparable = exact_match_fraction(
        (2, 5, None, 1), (2, 6, 3, 1), (True, True, False, True), (True, True, True, True)
    )
    assert comparable == 3
    assert similarity == pytest.approx(2 / 3)
    similarity, comparable = exact_match_fraction((None,), (None,), (False,), (False,))
    assert (similarity, comparable) == (None, 0)


def test_token_similarity_needs_enough_overlap() -> None:
    query, baseline = _query([10, 11, 12, 13, 14, 15])
    ref = _ref("r1", [10, 11, 12, 13, 14, 15])
    similarity, comparable = token_count_similarity(query, baseline, ref)  # type: ignore[arg-type]
    assert similarity == pytest.approx(1.0)
    assert comparable == 6
    short_query, short_base = _query([10, 11])
    short_ref = _ref("r2", [10, 11])
    similarity, _ = token_count_similarity(
        short_query,
        short_base,
        short_ref,
        min_comparable=5,  # type: ignore[arg-type]
    )
    assert similarity is None


def test_token_similarity_ignores_constant_framing_offsets() -> None:
    query, _ = _query([10, 11, 12, 13, 14, 15], baseline=8)
    ref = _ref("r1", [17, 18, 19, 20, 21, 22], baseline=15)
    # Every raw count shifted by +7 on the reference side; deltas agree.
    similarity, _ = token_count_similarity(query, 8, ref)  # type: ignore[arg-type]
    assert similarity == pytest.approx(1.0)


def test_incomplete_vectors_shrink_comparability_instead_of_imputing() -> None:
    query, baseline = _query([10, None, None, None, None, None])
    ref = _ref("r1", [10, 11, 12, 13, 14, 15])
    similarity, comparable = token_count_similarity(query, baseline, ref)  # type: ignore[arg-type]
    assert similarity is None
    assert comparable == 1


def test_behavior_similarity_handles_shared_keys_only() -> None:
    similarity, n = behavior_similarity({"a": 1.0, "b": "x"}, {"a": 1.0, "b": "x"})
    assert similarity == pytest.approx(1.0)
    assert n == 2
    similarity, _ = behavior_similarity({"a": 1.0}, {"a": 3.0})
    assert similarity == pytest.approx(1 / 3)
    similarity, n = behavior_similarity({"a": 1.0}, {"b": 1.0})
    assert (similarity, n) == (None, 0)
    similarity, _ = behavior_similarity({"k": "yes"}, {"k": "no"})
    assert similarity == pytest.approx(0.0)


def test_combine_averages_available_signals() -> None:
    assert combine_signal_similarities({"a": 1.0, "b": 0.0}) == pytest.approx(0.5)
    assert combine_signal_similarities({"a": None, "b": 0.75}) == pytest.approx(0.75)
    assert combine_signal_similarities({"a": None, "b": None}) is None


def test_shared_tokenizer_does_not_become_an_identity_claim() -> None:
    query, baseline = _query([10, 11, 12, 13, 14, 15])
    refs = [
        _ref("model-a", [10, 11, 12, 13, 14, 15], behaviors={}),
        _ref("model-b", [10, 11, 12, 13, 14, 15], behaviors={}),
    ]
    # Give both references the same tokenizer label but different identities.
    refs[0] = ReferenceSignature(
        candidate_id="model-a",
        candidate_library_revision=LIB,
        probe_version="probe-v1",
        input_counts=refs[0].input_counts,
        validity=refs[0].validity,
        baseline_count=refs[0].baseline_count,
        behavioral_features={},
        label_tokenizer="shared-tok",
        label_exact_version="model-a-v1",
    )
    refs[1] = ReferenceSignature(
        candidate_id="model-b",
        candidate_library_revision=LIB,
        probe_version="probe-v1",
        input_counts=refs[1].input_counts,
        validity=refs[1].validity,
        baseline_count=refs[1].baseline_count,
        behavioral_features={},
        label_tokenizer="shared-tok",
        label_exact_version="model-b-v1",
    )
    report = build_identity_report(
        endpoint_id="ep-anon",
        query=query,  # type: ignore[arg-type]
        query_baseline=baseline,
        references=refs,
        candidate_library_revision=LIB,
    )
    assert report.calibrated_probabilities is None
    assert len(report.ranked_similarities) == 2
    for entry in report.ranked_similarities:
        assert entry.similarity == pytest.approx(1.0)
        assert "does not establish" in (entry.note or "")
        assert TOKENIZER_ONLY_NOTE in (entry.note or "")
    assert_no_probabilities(report)
    assert "probability" not in RankedSimilarity.model_fields


def test_exact_counts_with_divergent_behavior_stay_qualified() -> None:
    query, baseline = _query([10, 11, 12, 13, 14, 15], behaviors={"tone": "terse", "length": 5.0})
    ref = _ref(
        "model-a",
        [10, 11, 12, 13, 14, 15],
        behaviors={"tone": "verbose", "length": 50.0},
    )
    report = build_identity_report(
        endpoint_id="ep-anon",
        query=query,  # type: ignore[arg-type]
        query_baseline=baseline,
        references=[ref],
        candidate_library_revision=LIB,
    )
    assert len(report.ranked_similarities) == 1
    entry = report.ranked_similarities[0]
    assert entry.similarity < 1.0
    assert DIVERGENT_BEHAVIOR_NOTE in (entry.note or "")
    assert report.calibrated_probabilities is None


def test_ranking_orders_by_similarity() -> None:
    query, baseline = _query([10, 11, 12, 13, 14, 15])
    refs = [
        _ref("far", [20, 21, 22, 23, 24, 25]),
        _ref("near", [10, 11, 12, 13, 14, 99]),
    ]
    report = build_identity_report(
        endpoint_id="ep-anon",
        query=query,  # type: ignore[arg-type]
        query_baseline=baseline,
        references=refs,
        candidate_library_revision=LIB,
    )
    assert [entry.candidate_id for entry in report.ranked_similarities] == ["near", "far"]
    scores = [entry.similarity for entry in report.ranked_similarities]
    assert scores == sorted(scores, reverse=True)


def test_weak_evidence_abstains_as_unknown() -> None:
    query, baseline = _query([None, None, None, None, None, None], baseline=None)
    ref = _ref("model-a", [None, None, None, None, None, None], baseline=None)
    report = build_identity_report(
        endpoint_id="ep-anon",
        query=query,  # type: ignore[arg-type]
        query_baseline=baseline,
        references=[ref],
        candidate_library_revision=LIB,
    )
    assert report.ranked_similarities == ()
    assert report.abstention_reason == "insufficient_evidence"
    assert report.is_unknown
    assert report.calibrated_probabilities is None


def test_changed_reference_library_is_incompatible_not_silent() -> None:
    query, baseline = _query([10, 11, 12, 13, 14, 15])
    ref = _ref("model-a", [10, 11, 12, 13, 14, 15], library="candidate-lib-v2")
    report = build_identity_report(
        endpoint_id="ep-anon",
        query=query,  # type: ignore[arg-type]
        query_baseline=baseline,
        references=[ref],
        candidate_library_revision=LIB,
    )
    assert report.ranked_similarities == ()
    assert report.abstention_reason == "incompatible_reference_library"
    stale = _ref("model-a", [10, 11, 12, 13, 14, 15], probe_version="probe-v0")
    report = build_identity_report(
        endpoint_id="ep-anon",
        query=query,  # type: ignore[arg-type]
        query_baseline=baseline,
        references=[stale],
        candidate_library_revision=LIB,
    )
    assert report.is_unknown


def test_similarity_never_produces_probabilities() -> None:
    query, baseline = _query([10, 11, 12, 13, 14, 15])
    ref = _ref("model-a", [10, 11, 12, 13, 14, 15])
    report = build_identity_report(
        endpoint_id="ep-anon",
        query=query,  # type: ignore[arg-type]
        query_baseline=baseline,
        references=[ref],
        candidate_library_revision=LIB,
    )
    assert report.calibrated_probabilities is None
    assert_no_probabilities(report)
    dumped = report.model_dump()
    assert dumped["calibrated_probabilities"] is None
    for entry in report.ranked_similarities:
        assert 0.0 <= entry.similarity <= 1.0
        assert "probability" not in entry.model_dump()
