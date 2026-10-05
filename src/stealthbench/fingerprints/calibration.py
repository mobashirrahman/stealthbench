"""Calibration, abstention rules and the G12 evidence gate (G12 T12B).

Contracts (``BENCHMARK_PLAN.md`` "Prediction and validation",
``IMPLEMENTATION_PLAN.md`` G12, ``docs/contracts.md`` ``IdentityReport``):

* Thresholds are frozen on development data before any holdout is
  scored: every scoring entry point requires
  ``FrozenThresholds(frozen=True)`` and raises
  :class:`UnfrozenThresholdsError` otherwise.
* Probabilities are emitted only through a passing :class:`GateVerdict`;
  anything else stays in similarity mode
  (``calibrated_probabilities=None`` plus a reason). Asking for
  probabilities without a passing verdict raises
  :class:`UncalibratedError`, and the report schema independently
  requires a ``calibration_evidence_id`` alongside any probabilities.
* Exact-version claims need their own held-out version-level gate: a
  family pass never implies a version pass. :func:`build_identity_report`
  checks that the verdict's claim level matches the requested one.
* Every verdict records the candidate library revision and the class
  priors it was computed under. Scoring or reporting under changed
  priors or a changed reference set raises :class:`StaleCalibrationError`
  instead of silently reusing the numbers.
* Abstention rules are frozen strings: ``missing_features`` (no
  distances to judge), ``low_confidence`` (top probability below the
  floor), ``conflicting_evidence`` (top-two margin below the floor).
  Distance rejection is the supported ``unknown`` output, not abstention.

Offline by construction: pure functions over stored distances and
labels. No transport, no socket, no subprocess, no credential.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from stealthbench.fingerprints.attribution import (
    AttributionObservation,
    CentroidModel,
    distances_to_centroids,
    observation_label,
    validate_observations,
)
from stealthbench.schemas.hashing import content_digest
from stealthbench.schemas.results import IdentityReport

__all__ = [
    "AttributionDecision",
    "EvidenceGateSpec",
    "FrozenThresholds",
    "GateMetrics",
    "GateVerdict",
    "ScoredCase",
    "StaleCalibrationError",
    "UncalibratedError",
    "UnfrozenThresholdsError",
    "build_identity_report",
    "decide_from_distances",
    "evaluate_evidence_gate",
    "evaluate_version_gate",
    "expected_calibration_error",
    "fit_temperature",
    "freeze_thresholds",
    "require_calibrated",
    "score_holdout",
    "similarity_mode_report",
    "softmax_from_distances",
    "wilson_upper",
]

#: Claim levels. A verdict is valid for exactly one of these.
CLAIM_LEVELS: Final[tuple[str, ...]] = ("family", "exact_version")

#: Abstention reasons. Frozen strings asserted by tests and reports.
MISSING_FEATURES_REASON: Final[str] = "missing_features"
LOW_CONFIDENCE_REASON: Final[str] = "low_confidence"
CONFLICTING_EVIDENCE_REASON: Final[str] = "conflicting_evidence"
UNKNOWN_REASON: Final[str] = "unknown"

#: Grid searched by :func:`fit_temperature`. Covers over- and
#: under-confident validation fits without any optimizer dependency.
TEMPERATURE_GRID: Final[tuple[float, ...]] = (0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0)

#: z for the fixed 95% Wilson upper bound (see :func:`wilson_upper`).
_WILSON_Z_95: Final[float] = 1.96


class UnfrozenThresholdsError(ValueError):
    """Holdout scoring was attempted with experimental (unfrozen) thresholds."""


class UncalibratedError(ValueError):
    """Probabilities were requested without a passing evidence-gate verdict."""


class StaleCalibrationError(ValueError):
    """The priors or reference library changed since the gate evaluation."""


@dataclass(frozen=True, slots=True)
class EvidenceGateSpec:
    """Preregistered evidence gate: thresholds frozen before holdout scoring.

    ``library_revision`` pins the candidate reference set and ``priors``
    the class prior assumption the verdict is conditional on. Both are
    re-checked at evaluation and report time.
    """

    max_known_error: float
    max_unknown_false_accept_rate: float
    max_calibration_error: float
    min_coverage: float
    min_holdout_samples: int
    min_per_class_samples: int
    confidence_level: float = 0.95
    bound_method: str = "wilson"
    library_revision: str = ""
    priors: tuple[tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "max_known_error",
            "max_unknown_false_accept_rate",
            "max_calibration_error",
            "min_coverage",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1], got {value}")
        if self.min_holdout_samples <= 0:
            raise ValueError(
                f"min_holdout_samples must be positive, got {self.min_holdout_samples}"
            )
        if self.min_per_class_samples <= 0:
            raise ValueError(
                f"min_per_class_samples must be positive, got {self.min_per_class_samples}"
            )
        if self.confidence_level != 0.95:
            raise ValueError(
                "only a fixed 95% Wilson bound is supported; "
                f"got confidence_level={self.confidence_level}"
            )
        if self.bound_method != "wilson":
            raise ValueError(f"only bound_method='wilson' is supported, got {self.bound_method!r}")
        if not self.library_revision:
            raise ValueError("library_revision must pin the candidate reference set")
        total = sum(weight for _, weight in self.priors)
        if self.priors and not math.isclose(total, 1.0, abs_tol=1e-6):
            raise ValueError(f"priors must sum to 1, got {total}")
        if any(weight < 0 for _, weight in self.priors):
            raise ValueError("prior weights must be non-negative")

    def prior_map(self) -> dict[str, float]:
        """Priors as a mapping (empty when no prior assumption is registered)."""
        return dict(self.priors)


@dataclass(frozen=True, slots=True)
class FrozenThresholds:
    """Decision thresholds. Only ``frozen=True`` instances may score a holdout.

    Build them with :func:`freeze_thresholds` on development data; the
    ``frozen`` flag is the machine-readable form of "thresholds were
    frozen before holdout scoring".
    """

    max_distance: float
    min_probability: float
    min_margin: float
    temperature: float
    frozen: bool
    dev_revision: str

    def __post_init__(self) -> None:
        if not math.isfinite(self.max_distance) or self.max_distance < 0:
            raise ValueError(f"max_distance must be non-negative finite, got {self.max_distance}")
        if not 0.0 < self.min_probability < 1.0:
            raise ValueError(f"min_probability must lie in (0, 1), got {self.min_probability}")
        if not 0.0 <= self.min_margin < 1.0:
            raise ValueError(f"min_margin must lie in [0, 1), got {self.min_margin}")
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError(f"temperature must be positive finite, got {self.temperature}")
        if not self.dev_revision:
            raise ValueError("dev_revision must name the development data/revision used")


def freeze_thresholds(
    *,
    max_distance: float,
    min_probability: float,
    min_margin: float,
    temperature: float,
    dev_revision: str,
) -> FrozenThresholds:
    """Freeze experimentally chosen thresholds against a dev revision."""
    return FrozenThresholds(
        max_distance=max_distance,
        min_probability=min_probability,
        min_margin=min_margin,
        temperature=temperature,
        frozen=True,
        dev_revision=dev_revision,
    )


def _require_frozen(thresholds: FrozenThresholds) -> None:
    if not thresholds.frozen:
        raise UnfrozenThresholdsError(
            "thresholds are experimental; freeze them on development data with "
            "freeze_thresholds() before scoring any holdout"
        )


@dataclass(frozen=True, slots=True)
class AttributionDecision:
    """One abstaining-or-predicting decision from class distances."""

    predicted_label: str | None
    abstained: bool
    abstention_reason: str | None
    probabilities: tuple[tuple[str, float], ...]


def softmax_from_distances(
    distances: Mapping[str, float], *, temperature: float
) -> dict[str, float]:
    """Calibrated-style probabilities from distances: ``p ∝ exp(-d/T)``.

    Smaller distances mean larger probabilities. ``temperature`` is fitted
    on validation data (:func:`fit_temperature`); ``T=1`` is the
    uncalibrated default and must not back probabilities on its own.
    """
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError(f"temperature must be positive finite, got {temperature}")
    if not distances:
        raise ValueError("softmax needs at least one class distance")
    if any(not math.isfinite(value) or value < 0 for value in distances.values()):
        raise ValueError("distances must be finite and non-negative")
    floor = min(distances.values())
    weights = {
        label: math.exp(-(distance - floor) / temperature) for label, distance in distances.items()
    }
    total = sum(weights.values())
    return {label: weight / total for label, weight in weights.items()}


def _softmax_nll(cases: Sequence[tuple[Mapping[str, float], str]], *, temperature: float) -> float:
    total = 0.0
    for distances, truth in cases:
        probabilities = softmax_from_distances(distances, temperature=temperature)
        if truth not in probabilities:
            raise ValueError(f"truth {truth!r} is not among the scored classes")
        total += -math.log(max(probabilities[truth], 1e-12))
    return total / len(cases)


def fit_temperature(
    cases: Sequence[tuple[Mapping[str, float], str]],
) -> float:
    """Pick the NLL-minimizing temperature over :data:`TEMPERATURE_GRID`.

    ``cases`` are validation ``(distances, true_label)`` pairs: separate
    data from both the centroid fit and the final holdout. Ties keep the
    lowest temperature (least smoothing).
    """
    if not cases:
        raise ValueError("fit_temperature needs at least one validation case")
    best = TEMPERATURE_GRID[0]
    best_nll = math.inf
    for temperature in TEMPERATURE_GRID:
        nll = _softmax_nll(cases, temperature=temperature)
        if nll < best_nll:
            best_nll = nll
            best = temperature
    return best


def decide_from_distances(
    distances: Mapping[str, float],
    thresholds: FrozenThresholds,
    *,
    priors: Mapping[str, float] | None = None,
) -> AttributionDecision:
    """Apply the frozen abstention rules to one set of class distances.

    Order: missing evidence abstains first, then the ``unknown`` distance
    reject, then the low-confidence floor, then the conflicting-evidence
    margin. A decision carries normalized probabilities (descending)
    whenever it does not abstain for lack of evidence.
    """
    _require_frozen(thresholds)
    if not distances:
        return AttributionDecision(
            predicted_label=None,
            abstained=True,
            abstention_reason=MISSING_FEATURES_REASON,
            probabilities=(),
        )
    probabilities = softmax_from_distances(distances, temperature=thresholds.temperature)
    if priors:
        reweighted = {
            label: probabilities[label] * priors.get(label, 0.0) for label in probabilities
        }
        total = sum(reweighted.values())
        if total <= 0:
            return AttributionDecision(
                predicted_label=None,
                abstained=True,
                abstention_reason=LOW_CONFIDENCE_REASON,
                probabilities=(),
            )
        probabilities = {label: weight / total for label, weight in reweighted.items()}
    ordered = sorted(probabilities, key=lambda label: probabilities[label], reverse=True)
    best = ordered[0]
    nearest_distance = min(distances.values())
    if nearest_distance > thresholds.max_distance:
        return AttributionDecision(
            predicted_label=None,
            abstained=False,
            abstention_reason=UNKNOWN_REASON,
            probabilities=tuple((label, probabilities[label]) for label in ordered),
        )
    top_probability = probabilities[best]
    if top_probability < thresholds.min_probability:
        return AttributionDecision(
            predicted_label=None,
            abstained=True,
            abstention_reason=LOW_CONFIDENCE_REASON,
            probabilities=tuple((label, probabilities[label]) for label in ordered),
        )
    runner_up = probabilities[ordered[1]] if len(ordered) > 1 else 0.0
    if top_probability - runner_up < thresholds.min_margin:
        return AttributionDecision(
            predicted_label=None,
            abstained=True,
            abstention_reason=CONFLICTING_EVIDENCE_REASON,
            probabilities=tuple((label, probabilities[label]) for label in ordered),
        )
    return AttributionDecision(
        predicted_label=best,
        abstained=False,
        abstention_reason=None,
        probabilities=tuple((label, probabilities[label]) for label in ordered),
    )


@dataclass(frozen=True, slots=True)
class ScoredCase:
    """One holdout observation after scoring.

    ``true_label=None`` marks an unknown-family (or unknown-version)
    sample. A non-abstained ``predicted_label=None`` with reason
    ``unknown`` is the supported unknown output; any other ``None``
    prediction must be abstained with its rule.
    """

    observation_id: str
    true_label: str | None
    predicted_label: str | None
    abstained: bool
    abstention_reason: str | None
    probabilities: tuple[tuple[str, float], ...]

    def __post_init__(self) -> None:
        if self.abstained and self.predicted_label is not None:
            raise ValueError("an abstained case must not carry a predicted label")
        if self.abstained and not self.abstention_reason:
            raise ValueError("an abstained case must name its abstention rule")
        if (
            not self.abstained
            and self.predicted_label is None
            and self.abstention_reason != UNKNOWN_REASON
        ):
            raise ValueError("a non-abstained case without a label must carry reason 'unknown'")


def score_holdout(
    *,
    model: CentroidModel,
    observations: Sequence[AttributionObservation],
    thresholds: FrozenThresholds,
    level: str = "family",
    priors: Mapping[str, float] | None = None,
    known_labels: Sequence[str] | None = None,
) -> tuple[ScoredCase, ...]:
    """Score untouched holdout observations with frozen thresholds.

    Raises :class:`UnfrozenThresholdsError` unless ``thresholds.frozen``.
    ``level`` must match the model's ``label_level``: family thresholds
    never score a version holdout. Observations whose label is outside
    ``known_labels`` (default: the model's trained labels) are recorded
    with ``true_label=None``: unseen families/versions whose only
    acceptable outputs are ``unknown`` or abstention.
    """
    _require_frozen(thresholds)
    if level != model.label_level:
        raise ValueError(
            f"model was trained at level {model.label_level!r}; refusing to score "
            f"a {level!r} holdout with it"
        )
    known = validate_observations(observations)
    known_set = set(known_labels) if known_labels is not None else set(model.labels)
    scored: list[ScoredCase] = []
    for observation in known:
        distances = distances_to_centroids(model, observation.features)
        decision = decide_from_distances(distances, thresholds, priors=priors)
        label = observation_label(observation, level)
        scored.append(
            ScoredCase(
                observation_id=observation.observation_id,
                true_label=label if label in known_set else None,
                predicted_label=decision.predicted_label,
                abstained=decision.abstained,
                abstention_reason=decision.abstention_reason,
                probabilities=decision.probabilities,
            )
        )
    return tuple(scored)


def expected_calibration_error(cases: Sequence[ScoredCase], *, n_bins: int = 10) -> float | None:
    """Expected calibration error over non-abstained known-truth cases.

    Bins the top probability into ``n_bins`` equal-width bins and returns
    the size-weighted ``|accuracy - confidence|`` sum. ``None`` when no
    case is eligible; the gate treats that as a failure, not a pass.
    """
    if n_bins <= 0:
        raise ValueError(f"n_bins must be positive, got {n_bins}")
    eligible = [
        case
        for case in cases
        if not case.abstained
        and case.true_label is not None
        and case.predicted_label is not None
        and case.probabilities
    ]
    if not eligible:
        return None
    bin_counts = [0] * n_bins
    bin_correct = [0] * n_bins
    bin_confidence = [0.0] * n_bins
    for case in eligible:
        confidence = case.probabilities[0][1]
        index = min(int(confidence * n_bins), n_bins - 1)
        bin_counts[index] += 1
        bin_confidence[index] += confidence
        if case.predicted_label == case.true_label:
            bin_correct[index] += 1
    total = len(eligible)
    error = 0.0
    for count, correct, confidence_sum in zip(bin_counts, bin_correct, bin_confidence, strict=True):
        if not count:
            continue
        error += abs(correct / count - confidence_sum / count) * (count / total)
    return error


def wilson_upper(successes: int, trials: int) -> float:
    """95% Wilson upper bound for a binomial rate (fixed confidence).

    Reported alongside every gated error rate so a small holdout cannot
    hide its uncertainty. Only the preregistered 95% level is supported.
    """
    if trials <= 0:
        raise ValueError(f"trials must be positive, got {trials}")
    if not 0 <= successes <= trials:
        raise ValueError(f"successes must lie in [0, trials], got {successes}")
    rate = successes / trials
    z = _WILSON_Z_95
    denominator = 1.0 + z * z / trials
    center = rate + z * z / (2.0 * trials)
    half = z * math.sqrt(rate * (1.0 - rate) / trials + z * z / (4.0 * trials * trials))
    return min(max((center + half) / denominator, 0.0), 1.0)


@dataclass(frozen=True, slots=True)
class GateMetrics:
    """Measured holdout numbers behind a :class:`GateVerdict`."""

    n_holdout: int
    n_known: int
    n_unknown: int
    n_abstained: int
    coverage: float
    known_error: float | None
    unknown_false_accept_rate: float | None
    calibration_error_ece: float | None
    wilson_upper_known_error: float | None
    min_class_count: int


@dataclass(frozen=True, slots=True)
class GateVerdict:
    """The evidence gate outcome for one claim level.

    ``passed`` is the only path to probabilities. ``evidence_id`` is set
    exactly when the gate passes and is the ``calibration_evidence_id``
    any resulting report must carry.
    """

    claim_level: str
    passed: bool
    reasons: tuple[str, ...]
    metrics: GateMetrics
    evidence_id: str | None
    library_revision: str
    priors: tuple[tuple[str, float], ...]

    def __post_init__(self) -> None:
        if self.claim_level not in CLAIM_LEVELS:
            raise ValueError(f"claim_level must be one of {CLAIM_LEVELS}, got {self.claim_level!r}")
        if self.passed and not self.evidence_id:
            raise ValueError("a passing verdict must carry an evidence_id")
        if not self.passed and self.evidence_id is not None:
            raise ValueError("a failing verdict must not carry an evidence_id")


def _check_gate_context(
    spec: EvidenceGateSpec,
    *,
    library_revision: str,
    priors: Mapping[str, float] | Sequence[tuple[str, float]] | None,
) -> tuple[tuple[str, float], ...]:
    if library_revision != spec.library_revision:
        raise StaleCalibrationError(
            f"reference library changed since the gate was specified "
            f"({spec.library_revision!r} -> {library_revision!r}); recalibrate, "
            "do not reuse the numbers"
        )
    effective: tuple[tuple[str, float], ...]
    if priors is None:
        effective = spec.priors
    elif isinstance(priors, Mapping):
        effective = tuple(sorted(priors.items()))
    else:
        effective = tuple(priors)
    if tuple(sorted(effective)) != tuple(sorted(spec.priors)):
        raise StaleCalibrationError(
            f"priors changed since the gate was specified ({spec.priors!r} -> "
            f"{effective!r}); recalibrate under the new priors"
        )
    return effective


def evaluate_evidence_gate(
    cases: Sequence[ScoredCase],
    spec: EvidenceGateSpec,
    *,
    claim_level: str,
    library_revision: str,
    priors: Mapping[str, float] | Sequence[tuple[str, float]] | None = None,
) -> GateVerdict:
    """Evaluate preregistered gates on untouched holdout scores.

    Gates: known error, unknown false-accept rate, calibration error
    (ECE), coverage, minimum holdout size and minimum per-class size
    (every known class and the unknown group). Unknown samples are
    required: without them the false-accept rate is unmeasurable and the
    gate fails explicitly.
    """
    if claim_level not in CLAIM_LEVELS:
        raise ValueError(f"claim_level must be one of {CLAIM_LEVELS}, got {claim_level!r}")
    effective_priors = _check_gate_context(spec, library_revision=library_revision, priors=priors)
    scored = tuple(cases)
    n = len(scored)
    reasons: list[str] = []
    if n < spec.min_holdout_samples:
        reasons.append(
            f"only {n} holdout samples < minimum {spec.min_holdout_samples}; "
            "the small initial endpoint library may be insufficient for this gate"
        )
    known = [case for case in scored if case.true_label is not None]
    unknown = [case for case in scored if case.true_label is None]
    class_counts: dict[str, int] = {}
    for case in known:
        assert case.true_label is not None
        class_counts[case.true_label] = class_counts.get(case.true_label, 0) + 1
    for label, count in sorted(class_counts.items()):
        if count < spec.min_per_class_samples:
            reasons.append(
                f"class {label!r} has {count} holdout samples "
                f"< minimum {spec.min_per_class_samples}"
            )
    if not unknown:
        reasons.append("no unknown-family samples; false-accept rate is unmeasurable")
    elif len(unknown) < spec.min_per_class_samples:
        reasons.append(
            f"unknown group has {len(unknown)} holdout samples "
            f"< minimum {spec.min_per_class_samples}"
        )
    min_class_count = (
        min([*class_counts.values(), len(unknown)] if class_counts else [len(unknown)])
        if (class_counts or unknown)
        else 0
    )

    decided_known = [case for case in known if not case.abstained]
    known_errors = sum(1 for case in decided_known if case.predicted_label != case.true_label)
    known_error = (known_errors / len(decided_known)) if decided_known else None
    if known_error is None:
        reasons.append("no non-abstained known cases; error is unmeasurable")
    elif known_error > spec.max_known_error:
        reasons.append(f"known error {known_error:.4f} > maximum {spec.max_known_error:.4f}")

    false_accepts = sum(
        1 for case in unknown if not case.abstained and case.predicted_label is not None
    )
    unknown_far = (false_accepts / len(unknown)) if unknown else None
    if unknown_far is None:
        reasons.append("unknown false-accept rate is unmeasurable")
    elif unknown_far > spec.max_unknown_false_accept_rate:
        reasons.append(
            f"unknown false-accept rate {unknown_far:.4f} > maximum "
            f"{spec.max_unknown_false_accept_rate:.4f}"
        )

    ece = expected_calibration_error(scored)
    if ece is None:
        reasons.append("calibration error is unmeasurable (no eligible cases)")
    elif ece > spec.max_calibration_error:
        reasons.append(
            f"calibration error (ECE) {ece:.4f} > maximum {spec.max_calibration_error:.4f}"
        )

    n_abstained = sum(1 for case in scored if case.abstained)
    coverage = (n - n_abstained) / n if n else 0.0
    if coverage < spec.min_coverage:
        reasons.append(
            f"coverage {coverage:.4f} < minimum {spec.min_coverage:.4f} "
            f"({n_abstained}/{n} abstained)"
        )

    wilson = wilson_upper(known_errors, len(decided_known)) if decided_known else None
    metrics = GateMetrics(
        n_holdout=n,
        n_known=len(known),
        n_unknown=len(unknown),
        n_abstained=n_abstained,
        coverage=coverage,
        known_error=known_error,
        unknown_false_accept_rate=unknown_far,
        calibration_error_ece=ece,
        wilson_upper_known_error=wilson,
        min_class_count=min_class_count,
    )
    passed = not reasons
    evidence_id: str | None = None
    if passed:
        evidence_id = (
            f"g12-{claim_level}-"
            + content_digest(
                {
                    "claim_level": claim_level,
                    "library_revision": spec.library_revision,
                    "priors": [list(pair) for pair in effective_priors],
                    "metrics": {
                        "n_holdout": n,
                        "known_error": known_error,
                        "unknown_far": unknown_far,
                        "ece": ece,
                        "coverage": coverage,
                    },
                }
            )[:16]
        )
    return GateVerdict(
        claim_level=claim_level,
        passed=passed,
        reasons=tuple(reasons),
        metrics=metrics,
        evidence_id=evidence_id,
        library_revision=spec.library_revision,
        priors=effective_priors,
    )


def evaluate_version_gate(
    cases: Sequence[ScoredCase],
    spec: EvidenceGateSpec,
    *,
    library_revision: str,
    priors: Mapping[str, float] | Sequence[tuple[str, float]] | None = None,
) -> GateVerdict:
    """The separate held-out version-level gate for exact-version claims.

    Same machinery as the family gate, scored on exact-version labels. A
    family verdict never satisfies this: callers must hold out and score
    versions independently.
    """
    return evaluate_evidence_gate(
        cases,
        spec,
        claim_level="exact_version",
        library_revision=library_revision,
        priors=priors,
    )


def require_calibrated(verdict: GateVerdict, claim_level: str) -> str:
    """Return the evidence id or raise :class:`UncalibratedError`.

    Also rejects claim-level confusion: a family verdict cannot back an
    exact-version report.
    """
    if verdict.claim_level != claim_level:
        raise UncalibratedError(
            f"verdict covers claim level {verdict.claim_level!r}, not {claim_level!r}; "
            "exact-version claims need their own held-out version-level gate"
        )
    if not verdict.passed or verdict.evidence_id is None:
        raise UncalibratedError(
            "evidence gate did not pass: "
            + ("; ".join(verdict.reasons) if verdict.reasons else "no verdict")
            + "; ship similarity mode instead"
        )
    return verdict.evidence_id


def build_identity_report(
    base_report: IdentityReport,
    *,
    verdict: GateVerdict,
    claim_level: str,
    probabilities: Mapping[str, float],
) -> IdentityReport:
    """Attach calibrated probabilities to a similarity report.

    Requires a passing verdict for the same claim level; otherwise raises
    :class:`UncalibratedError`. Probability keys must be ranked
    candidates (checked again by the schema).
    """
    evidence_id = require_calibrated(verdict, claim_level)
    if verdict.library_revision != base_report.candidate_library_revision:
        raise StaleCalibrationError(
            f"verdict was computed for library {verdict.library_revision!r} but the "
            f"report ranks {base_report.candidate_library_revision!r}"
        )
    payload = base_report.model_dump()
    payload["calibrated_probabilities"] = dict(probabilities)
    payload["calibration_evidence_id"] = evidence_id
    return IdentityReport(**payload)


def similarity_mode_report(base_report: IdentityReport, reason: str) -> IdentityReport:
    """Return the similarity-mode report: no probabilities, reason recorded.

    The ranking and signal evidence are preserved; only the abstention
    reason is set to explain why the evidence gate was not cleared.
    """
    if not reason:
        raise ValueError("a similarity-mode fallback must record its reason")
    payload = base_report.model_dump()
    payload["calibrated_probabilities"] = None
    payload["calibration_evidence_id"] = None
    payload["abstention_reason"] = reason
    return IdentityReport(**payload)
