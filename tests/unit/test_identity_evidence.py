"""G12 T12B: calibration, abstention and the evidence gate (unit)."""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from stealthbench.fingerprints.attribution import (
    AttributionObservation,
    fit_centroid_classifier,
)
from stealthbench.fingerprints.calibration import (
    CONFLICTING_EVIDENCE_REASON,
    LOW_CONFIDENCE_REASON,
    MISSING_FEATURES_REASON,
    UNKNOWN_REASON,
    EvidenceGateSpec,
    FrozenThresholds,
    StaleCalibrationError,
    UncalibratedError,
    UnfrozenThresholdsError,
    build_identity_report,
    decide_from_distances,
    evaluate_evidence_gate,
    evaluate_version_gate,
    expected_calibration_error,
    fit_temperature,
    freeze_thresholds,
    require_calibrated,
    score_holdout,
    similarity_mode_report,
    softmax_from_distances,
)
from stealthbench.schemas.results import (
    IdentityReport,
    RankedSimilarity,
    SignalEvidence,
)

pytestmark = pytest.mark.unit

PRIORS: tuple[tuple[str, float], ...] = (("alpha", 0.5), ("beta", 0.5))


def _train() -> list[AttributionObservation]:
    return [
        AttributionObservation(
            observation_id=f"a{i}",
            source_id="sA",
            family="alpha",
            exact_version="alpha-v1",
            endpoint_id="ep-a",
            route="r1",
            collected_at=1.0,
            features=features,
        )
        for i, features in enumerate([(0.1, 0.0), (-0.1, 0.0), (0.0, 0.1), (0.0, -0.1)])
    ] + [
        AttributionObservation(
            observation_id=f"b{i}",
            source_id="sB",
            family="beta",
            exact_version="beta-v1",
            endpoint_id="ep-b",
            route="r1",
            collected_at=1.0,
            features=features,
        )
        for i, features in enumerate([(10.1, 10.0), (9.9, 10.0), (10.0, 10.1), (10.0, 9.9)])
    ]


def _holdout() -> list[AttributionObservation]:
    return [
        AttributionObservation(
            observation_id="ha1",
            source_id="sHA",
            family="alpha",
            exact_version="alpha-v1",
            endpoint_id="ep-a",
            route="r1",
            collected_at=9.0,
            features=(0.2, 0.1),
        ),
        AttributionObservation(
            observation_id="ha2",
            source_id="sHA2",
            family="alpha",
            exact_version="alpha-v1",
            endpoint_id="ep-a",
            route="r1",
            collected_at=9.0,
            features=(-0.2, 0.0),
        ),
        AttributionObservation(
            observation_id="hb1",
            source_id="sHB",
            family="beta",
            exact_version="beta-v1",
            endpoint_id="ep-b",
            route="r1",
            collected_at=9.0,
            features=(10.0, 10.2),
        ),
        AttributionObservation(
            observation_id="hb2",
            source_id="sHB2",
            family="beta",
            exact_version="beta-v1",
            endpoint_id="ep-b",
            route="r1",
            collected_at=9.0,
            features=(9.8, 10.0),
        ),
        AttributionObservation(
            observation_id="hc1",
            source_id="sHC",
            family="gamma",
            exact_version="gamma-v9",
            endpoint_id="ep-c",
            route="r1",
            collected_at=9.0,
            features=(30.0, 30.0),
        ),
        AttributionObservation(
            observation_id="hc2",
            source_id="sHC2",
            family="gamma",
            exact_version="gamma-v9",
            endpoint_id="ep-c",
            route="r1",
            collected_at=9.0,
            features=(31.0, 29.0),
        ),
    ]


def _thresholds() -> FrozenThresholds:
    return freeze_thresholds(
        max_distance=5.0,
        min_probability=0.6,
        min_margin=0.2,
        temperature=1.0,
        dev_revision="dev-v1",
    )


def _spec(**overrides: Any) -> EvidenceGateSpec:
    fields: dict[str, Any] = {
        "max_known_error": 0.2,
        "max_unknown_false_accept_rate": 0.2,
        "max_calibration_error": 0.3,
        "min_coverage": 0.8,
        "min_holdout_samples": 6,
        "min_per_class_samples": 2,
        "library_revision": "lib-v1",
        "priors": PRIORS,
    }
    fields.update(overrides)
    return EvidenceGateSpec(**fields)


def _base_report() -> IdentityReport:
    return IdentityReport(
        endpoint_id="ep-anon",
        candidate_library_revision="lib-v1",
        signal_evidence=(
            SignalEvidence(
                signal="behavior",
                available=True,
                comparable_observations=4,
                detail="shared behavioral feature keys",
            ),
        ),
        ranked_similarities=(
            RankedSimilarity(
                candidate_id="cand-a",
                candidate_library_revision="lib-v1",
                similarity=0.9,
                evidence_signals=("behavior",),
            ),
            RankedSimilarity(
                candidate_id="cand-b",
                candidate_library_revision="lib-v1",
                similarity=0.6,
                evidence_signals=("behavior",),
            ),
        ),
        abstention_reason=None,
        calibrated_probabilities=None,
        official_reveal_label=None,
    )


def test_unfrozen_thresholds_block_holdout_scoring() -> None:
    model = fit_centroid_classifier(_train(), level="family")
    unfrozen = dataclasses.replace(_thresholds(), frozen=False)
    with pytest.raises(UnfrozenThresholdsError):
        score_holdout(model=model, observations=_holdout(), thresholds=unfrozen)
    with pytest.raises(UnfrozenThresholdsError):
        decide_from_distances({"alpha": 1.0}, unfrozen)


def test_conflicting_evidence_abstains() -> None:
    thresholds = freeze_thresholds(
        max_distance=10.0,
        min_probability=0.4,
        min_margin=0.1,
        temperature=1.0,
        dev_revision="dev-v1",
    )
    decision = decide_from_distances({"alpha": 5.0, "beta": 5.0}, thresholds)
    assert decision.abstained
    assert decision.abstention_reason == CONFLICTING_EVIDENCE_REASON
    assert decision.predicted_label is None


def test_low_confidence_abstains_before_margin() -> None:
    decision = decide_from_distances({"alpha": 5.0, "beta": 5.0}, _thresholds())
    assert decision.abstained
    assert decision.abstention_reason == LOW_CONFIDENCE_REASON


def test_missing_features_abstain() -> None:
    decision = decide_from_distances({}, _thresholds())
    assert decision.abstained
    assert decision.abstention_reason == MISSING_FEATURES_REASON
    assert decision.probabilities == ()


def test_far_observation_is_unknown_not_abstained() -> None:
    decision = decide_from_distances({"alpha": 100.0, "beta": 120.0}, _thresholds())
    assert not decision.abstained
    assert decision.predicted_label is None
    assert decision.abstention_reason == UNKNOWN_REASON


def test_gate_passes_and_report_carries_probabilities() -> None:
    model = fit_centroid_classifier(_train(), level="family")
    cases = score_holdout(model=model, observations=_holdout(), thresholds=_thresholds())
    assert [case.true_label for case in cases].count(None) == 2
    verdict = evaluate_evidence_gate(
        cases, _spec(), claim_level="family", library_revision="lib-v1", priors=dict(PRIORS)
    )
    assert verdict.passed, verdict.reasons
    assert verdict.evidence_id is not None
    evidence_id = require_calibrated(verdict, "family")
    report = build_identity_report(
        _base_report(),
        verdict=verdict,
        claim_level="family",
        probabilities={"cand-a": 0.8, "cand-b": 0.2},
    )
    assert report.calibrated_probabilities == {"cand-a": 0.8, "cand-b": 0.2}
    assert report.calibration_evidence_id == evidence_id
    assert report.ranked_similarities[0].candidate_id == "cand-a"


def test_insufficient_evidence_stays_in_similarity_mode() -> None:
    model = fit_centroid_classifier(_train(), level="family")
    cases = score_holdout(model=model, observations=_holdout()[:2], thresholds=_thresholds())
    verdict = evaluate_evidence_gate(
        cases, _spec(), claim_level="family", library_revision="lib-v1", priors=dict(PRIORS)
    )
    assert not verdict.passed
    assert verdict.evidence_id is None
    with pytest.raises(UncalibratedError):
        require_calibrated(verdict, "family")
    fallback = similarity_mode_report(_base_report(), "evidence_gate_failed: too few samples")
    assert fallback.calibrated_probabilities is None
    assert fallback.abstention_reason is not None
    assert len(fallback.ranked_similarities) == 2


def test_changed_priors_and_library_invalidate_the_gate() -> None:
    model = fit_centroid_classifier(_train(), level="family")
    cases = score_holdout(model=model, observations=_holdout(), thresholds=_thresholds())
    with pytest.raises(StaleCalibrationError):
        evaluate_evidence_gate(
            cases,
            _spec(),
            claim_level="family",
            library_revision="lib-v1",
            priors={"alpha": 0.9, "beta": 0.1},
        )
    with pytest.raises(StaleCalibrationError):
        evaluate_evidence_gate(
            cases,
            _spec(),
            claim_level="family",
            library_revision="lib-v2",
            priors=dict(PRIORS),
        )


def test_version_gate_is_separate_from_family_gate() -> None:
    model = fit_centroid_classifier(_train(), level="family")
    cases = score_holdout(model=model, observations=_holdout(), thresholds=_thresholds())
    family_verdict = evaluate_evidence_gate(
        cases, _spec(), claim_level="family", library_revision="lib-v1", priors=dict(PRIORS)
    )
    assert family_verdict.passed
    with pytest.raises(UncalibratedError):
        require_calibrated(family_verdict, "exact_version")
    version_model = fit_centroid_classifier(_train(), level="exact_version")
    version_holdout = [o for o in _holdout() if o.family in ("alpha", "beta")] + [
        AttributionObservation(
            observation_id="hv2",
            source_id="sHV2",
            family="alpha",
            exact_version="alpha-v2",
            endpoint_id="ep-a",
            route="r1",
            collected_at=9.0,
            features=(0.4, -0.3),
        ),
        AttributionObservation(
            observation_id="hv3",
            source_id="sHV3",
            family="alpha",
            exact_version="alpha-v2",
            endpoint_id="ep-a",
            route="r1",
            collected_at=9.0,
            features=(-0.3, 0.4),
        ),
    ]
    version_cases = score_holdout(
        model=version_model,
        observations=version_holdout,
        thresholds=_thresholds(),
        level="exact_version",
    )
    version_verdict = evaluate_version_gate(
        version_cases, _spec(), library_revision="lib-v1", priors=dict(PRIORS)
    )
    assert not version_verdict.passed
    assert any("false-accept" in reason for reason in version_verdict.reasons)


def test_calibration_error_blocks_despite_perfect_accuracy() -> None:
    soft = freeze_thresholds(
        max_distance=100.0,
        min_probability=0.5,
        min_margin=0.05,
        temperature=5.0,
        dev_revision="dev-v1",
    )
    model = fit_centroid_classifier(_train(), level="family")
    # Mid-boundary samples: always attributed correctly, but only with
    # ~0.75 confidence, so accuracy is perfect while ECE is large.
    boundary = [
        ("ba1", "sBA", "alpha", "alpha-v1", (3.0, 3.0)),
        ("ba2", "sBA2", "alpha", "alpha-v1", (2.0, 4.0)),
        ("bb1", "sBB", "beta", "beta-v1", (7.0, 7.0)),
        ("bb2", "sBB2", "beta", "beta-v1", (8.0, 6.0)),
        ("bu1", "sBU", "gamma", "gamma-v9", (100.0, 100.0)),
        ("bu2", "sBU2", "gamma", "gamma-v9", (101.0, 99.0)),
    ]
    observations = [
        AttributionObservation(
            observation_id=oid,
            source_id=source,
            family=family,
            exact_version=version,
            endpoint_id="ep-x",
            route="r1",
            collected_at=9.0,
            features=features,
        )
        for oid, source, family, version, features in boundary
    ]
    cases = score_holdout(model=model, observations=observations, thresholds=soft)
    assert all(c.predicted_label == c.true_label and not c.abstained for c in cases[:4])
    ece = expected_calibration_error(cases)
    assert ece is not None and ece > 0.1
    verdict = evaluate_evidence_gate(
        cases,
        _spec(max_calibration_error=0.1),
        claim_level="family",
        library_revision="lib-v1",
        priors=dict(PRIORS),
    )
    assert not verdict.passed
    assert any("calibration" in reason for reason in verdict.reasons)


def test_fit_temperature_prefers_confident_when_correct() -> None:
    near = [({"alpha": 0.0, "beta": 4.0}, "alpha")]
    assert fit_temperature(near) == 0.1
    far = [({"alpha": 0.0, "beta": 4.0}, "beta")]
    assert fit_temperature(far) == 5.0


def test_softmax_rejects_uncalibrated_temperature() -> None:
    with pytest.raises(ValueError, match="temperature"):
        softmax_from_distances({"alpha": 1.0}, temperature=0.0)


def test_report_with_mismatched_library_is_rejected() -> None:
    model = fit_centroid_classifier(_train(), level="family")
    cases = score_holdout(model=model, observations=_holdout(), thresholds=_thresholds())
    verdict = evaluate_evidence_gate(
        cases, _spec(), claim_level="family", library_revision="lib-v1", priors=dict(PRIORS)
    )
    assert verdict.passed
    other = _base_report().model_copy(update={"candidate_library_revision": "lib-v2"})
    with pytest.raises(StaleCalibrationError):
        build_identity_report(
            IdentityReport(**other.model_dump()),
            verdict=verdict,
            claim_level="family",
            probabilities={"cand-a": 0.8, "cand-b": 0.2},
        )
