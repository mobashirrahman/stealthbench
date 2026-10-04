"""G12 T12C: canary drift detection and reveal history (unit)."""

from __future__ import annotations

import random

import pytest

from stealthbench.analysis.drift import (
    CanarySchedule,
    DriftSpec,
    RevealEvent,
    RevealHistory,
    apply_reveal,
    canary_due,
    canary_job_key,
    classify_alias_alerts,
    detect_drift,
    due_canaries,
    evaluate_change_detection,
    evaluate_stable_controls,
    record_reveal,
    shift_statistic,
    slot_for,
)
from stealthbench.schemas.results import (
    IdentityReport,
    RankedSimilarity,
    SignalEvidence,
)

pytestmark = pytest.mark.unit

SPEC = DriftSpec(false_alert_target=0.05, detection_target=0.90)
THRESHOLD = 1.0


def _gaussian(random_state: random.Random, mean: float, size: int) -> list[float]:
    return [random_state.gauss(mean, 1.0) for _ in range(size)]


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
        ),
        abstention_reason=None,
        calibrated_probabilities=None,
        official_reveal_label=None,
    )


def test_identical_scores_do_not_alert() -> None:
    decision = detect_drift([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], threshold=THRESHOLD)
    assert not decision.alert
    assert decision.statistic == 0.0


def test_clear_shift_alerts() -> None:
    decision = detect_drift([0.0] * 20, [5.0] * 20, threshold=THRESHOLD)
    assert decision.alert
    assert decision.statistic == 2.0


def test_shift_statistic_rejects_missing_scores() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        shift_statistic([], [1.0])


def test_stable_controls_meet_false_alert_target() -> None:
    random_state = random.Random(20261004)
    alerts = 0
    controls = 200
    for _ in range(controls):
        baseline = _gaussian(random_state, 0.0, 50)
        current = _gaussian(random_state, 0.0, 50)
        if detect_drift(baseline, current, threshold=THRESHOLD).alert:
            alerts += 1
    evaluation = evaluate_stable_controls(alerts, controls, SPEC)
    assert evaluation.false_alert_rate <= SPEC.false_alert_target
    assert evaluation.meets_target


def test_changed_endpoints_meet_detection_target() -> None:
    random_state = random.Random(7)
    detected = 0
    changed = 100
    for _ in range(changed):
        baseline = _gaussian(random_state, 0.0, 50)
        current = _gaussian(random_state, 2.0, 50)
        if detect_drift(baseline, current, threshold=THRESHOLD).alert:
            detected += 1
    evaluation = evaluate_change_detection(detected, changed, SPEC)
    assert evaluation.detection_rate >= SPEC.detection_target
    assert evaluation.meets_target


def test_noisy_detector_fails_its_targets() -> None:
    stable = evaluate_stable_controls(50, 100, SPEC)
    assert not stable.meets_target
    changed = evaluate_change_detection(10, 100, SPEC)
    assert not changed.meets_target


def test_control_wide_shift_suppresses_alias_claims() -> None:
    alerts = {f"alias-{i}": True for i in range(5)}
    classification = classify_alias_alerts(alerts, SPEC)
    assert classification.kind == "control_wide_shift"
    assert classification.changed_aliases == ()
    assert "infrastructure" in classification.detail


def test_isolated_alert_names_its_alias() -> None:
    alerts = {"alias-a": True, "alias-b": False, "alias-c": False, "alias-d": False}
    classification = classify_alias_alerts(alerts, SPEC)
    assert classification.kind == "alias_change"
    assert classification.changed_aliases == ("alias-a",)


def test_quiet_round_is_no_change() -> None:
    classification = classify_alias_alerts({"a": False, "b": False}, SPEC)
    assert classification.kind == "no_change"
    with pytest.raises(ValueError, match="at least one"):
        classify_alias_alerts({}, SPEC)


def test_canary_scheduling_and_idempotent_keys() -> None:
    fresh = CanarySchedule(alias="zen-1", interval_seconds=86400.0, last_run_at=1_000.0)
    stale = CanarySchedule(alias="zen-2", interval_seconds=86400.0, last_run_at=1_000.0)
    never = CanarySchedule(alias="aaa", interval_seconds=10.0, last_run_at=None)
    assert canary_due(never, 1_000.0)
    assert not canary_due(fresh, 1_000.0 + 100.0)
    assert canary_due(stale, 1_000.0 + 86400.0)
    assert due_canaries([fresh, stale, never], 1_000.0 + 100.0) == ("aaa",)
    assert due_canaries([fresh, stale, never], 1_000.0 + 86400.0) == (
        "aaa",
        "zen-1",
        "zen-2",
    )
    slot = slot_for(stale, 1_000.0 + 86400.0)
    assert canary_job_key("zen-2", slot) == canary_job_key("zen-2", slot)
    assert canary_job_key("zen-2", slot) != canary_job_key("zen-2", slot + 1)


def test_retrospective_reveal_preserves_history() -> None:
    report = _base_report()
    history = RevealHistory()
    first = RevealEvent(
        alias="zen-1",
        predicted_label="cand-a",
        predicted_evidence_id="similarity-lib-v1",
        revealed_label="provider-x-model-v3",
        revealed_at=2_000.0,
    )
    after_first = record_reveal(history, first)
    assert len(history.events) == 0
    assert len(after_first.events) == 1
    second = RevealEvent(
        alias="zen-1",
        predicted_label="cand-a",
        predicted_evidence_id="similarity-lib-v1",
        revealed_label="provider-x-model-v3",
        revealed_at=3_000.0,
        note="re-announced with identical label",
    )
    after_second = record_reveal(after_first, second)
    assert len(after_first.events) == 1
    assert len(after_second.events) == 2
    assert after_second.events[0] == first
    revealed = apply_reveal(report, "provider-x-model-v3")
    assert revealed.official_reveal_label == "provider-x-model-v3"
    assert revealed.ranked_similarities == report.ranked_similarities
    assert revealed.signal_evidence == report.signal_evidence
    assert revealed.calibrated_probabilities is None
    with pytest.raises(ValueError, match="revealed_label"):
        apply_reveal(report, "")
