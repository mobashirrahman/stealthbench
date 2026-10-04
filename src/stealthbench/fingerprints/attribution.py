"""Grouped attribution splits and a simple centroid classifier (G12 T12A).

Contracts (``BENCHMARK_PLAN.md`` "Prediction and validation",
``IMPLEMENTATION_PLAN.md`` G12):

* Group splits by model version and endpoint: every observation carries a
  ``source_id`` (one endpoint observation session) and splits are built
  from disjoint source sets, so no source observation can appear on both
  sides of a train/test boundary.
* Later-time, cross-route and unknown-family holdouts are three distinct
  builders with disjoint selection criteria. Each rejects an empty side
  and any source that spans the selection boundary.
* Any duplicate observation id, any source shared between splits, or any
  source whose observations disagree on their identity labels is
  contamination and raises :class:`SplitContaminationError`.
* ``unknown`` is a supported classifier output via an explicit distance
  threshold (:func:`predict_with_reject`). Threshold *values* are
  experimental here; freezing them before holdout scoring is owned by
  :mod:`stealthbench.fingerprints.calibration`.

Offline by construction: pure functions over stored observations. No
transport, no socket, no subprocess, no credential.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

__all__ = [
    "EMPTY_VALIDATION",
    "UNKNOWN_LABEL",
    "AttributionObservation",
    "CentroidModel",
    "EmptyHoldoutError",
    "GroupSplit",
    "SplitContaminationError",
    "ablate_features",
    "check_no_leakage",
    "cross_route_holdout",
    "distances_to_centroids",
    "fit_centroid_classifier",
    "later_time_holdout",
    "observation_label",
    "predict_known",
    "predict_with_reject",
    "split_by_source",
    "split_label_counts",
    "unknown_family_holdout",
    "validate_observations",
]

#: Classifier output when an observation is too far from every known
#: centroid to attribute. A supported output, not an error.
UNKNOWN_LABEL: Final[str] = "unknown"

#: Label levels the classifier can be trained at. Family attribution is
#: evaluated first; exact-version claims need their own held-out gate
#: (see :mod:`stealthbench.fingerprints.calibration`).
SUPPORTED_LEVELS: Final[tuple[str, ...]] = ("family", "exact_version")

#: Shared empty validation tuple so builders without a validation split
#: are explicit about it rather than hiding a missing split.
EMPTY_VALIDATION: Final[tuple[AttributionObservation, ...]] = ()


class SplitContaminationError(ValueError):
    """A split boundary leaks: shared sources, duplicate ids, or a source
    spanning the boundary."""


class EmptyHoldoutError(ValueError):
    """A holdout builder produced an empty train or test side."""


@dataclass(frozen=True, slots=True)
class AttributionObservation:
    """One labeled signature observation for attribution training/eval.

    ``source_id`` identifies the endpoint observation session the sample
    was collected in. Splits are grouped by source: all observations
    from one source stay on one side of every boundary.
    """

    observation_id: str
    source_id: str
    family: str
    exact_version: str
    endpoint_id: str
    route: str
    collected_at: float
    features: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.observation_id:
            raise ValueError("observation_id must be a non-empty string")
        if not self.source_id:
            raise ValueError("source_id must be a non-empty string")
        if not self.family:
            raise ValueError("family must be a non-empty string")
        if not self.exact_version:
            raise ValueError("exact_version must be a non-empty string")
        if not self.endpoint_id:
            raise ValueError("endpoint_id must be a non-empty string")
        if not self.route:
            raise ValueError("route must be a non-empty string")
        if not math.isfinite(self.collected_at):
            raise ValueError(f"collected_at must be finite, got {self.collected_at}")
        if not self.features:
            raise ValueError(f"observation {self.observation_id} has no features")
        for value in self.features:
            if not math.isfinite(value):
                raise ValueError(
                    f"observation {self.observation_id} has a non-finite feature; "
                    "missing features stay missing upstream, never NaN here"
                )


def validate_observations(
    observations: Sequence[AttributionObservation],
) -> tuple[AttributionObservation, ...]:
    """Validate a batch and return it as a tuple.

    Rejects: duplicate observation ids, inconsistent feature dimensions,
    and sources whose observations disagree on family, exact version,
    endpoint or route (a source is one session against one endpoint, so
    disagreement means mislabeled or mixed data).
    """
    if not observations:
        raise ValueError("validate_observations needs at least one observation")
    seen_ids: set[str] = set()
    dim = len(observations[0].features)
    source_labels: dict[str, tuple[str, str, str, str]] = {}
    for observation in observations:
        if observation.observation_id in seen_ids:
            raise SplitContaminationError(
                f"duplicate observation_id {observation.observation_id!r}; "
                "duplicate source observations are rejected, not deduplicated"
            )
        seen_ids.add(observation.observation_id)
        if len(observation.features) != dim:
            raise ValueError(
                f"observation {observation.observation_id} has {len(observation.features)} "
                f"features, expected {dim}; dimensions must agree, never be padded"
            )
        key = (
            observation.family,
            observation.exact_version,
            observation.endpoint_id,
            observation.route,
        )
        previous = source_labels.get(observation.source_id)
        if previous is None:
            source_labels[observation.source_id] = key
        elif previous != key:
            raise SplitContaminationError(
                f"source {observation.source_id!r} spans identity labels "
                f"{previous} and {key}; one source must describe one endpoint"
            )
    return tuple(observations)


def observation_label(observation: AttributionObservation, level: str) -> str:
    """The training label for one observation at ``family``/``exact_version``."""
    if level == "family":
        return observation.family
    if level == "exact_version":
        return observation.exact_version
    raise ValueError(f"level must be one of {SUPPORTED_LEVELS}, got {level!r}")


@dataclass(frozen=True, slots=True)
class GroupSplit:
    """Train/validation/test split with disjoint source sets.

    Invariant: no ``source_id`` and no ``observation_id`` appears in more
    than one side. Builders establish this via :func:`split_by_source`;
    :func:`check_no_leakage` re-verifies it explicitly.
    """

    train: tuple[AttributionObservation, ...]
    validation: tuple[AttributionObservation, ...]
    test: tuple[AttributionObservation, ...]
    description: str


def _sources(
    observations: Sequence[AttributionObservation],
) -> tuple[set[str], set[str]]:
    return (
        {observation.source_id for observation in observations},
        {observation.observation_id for observation in observations},
    )


def check_no_leakage(split: GroupSplit) -> None:
    """Raise :class:`SplitContaminationError` if any source or observation
    appears on more than one side of the split."""
    sides = (
        ("train", split.train),
        ("validation", split.validation),
        ("test", split.test),
    )
    for index, (first_name, first_obs) in enumerate(sides):
        first_sources, first_ids = _sources(first_obs)
        for second_name, second_obs in sides[index + 1 :]:
            second_sources, second_ids = _sources(second_obs)
            shared_sources = sorted(first_sources & second_sources)
            if shared_sources:
                raise SplitContaminationError(
                    f"sources {shared_sources} appear in both {first_name} and "
                    f"{second_name}; splits are grouped by source, never by row"
                )
            shared_ids = sorted(first_ids & second_ids)
            if shared_ids:
                raise SplitContaminationError(
                    f"observations {shared_ids} appear in both {first_name} and {second_name}"
                )


def split_by_source(
    observations: Sequence[AttributionObservation],
    *,
    train_sources: Sequence[str],
    validation_sources: Sequence[str] = (),
    test_sources: Sequence[str] = (),
    description: str = "",
) -> GroupSplit:
    """Build a split from explicit disjoint source sets.

    Rejects overlapping source sets (contamination), unknown source ids,
    and empty train or test sides. Validation may be empty.
    """
    known = validate_observations(observations)
    by_source: dict[str, list[AttributionObservation]] = {}
    for observation in known:
        by_source.setdefault(observation.source_id, []).append(observation)
    train_set = set(train_sources)
    validation_set = set(validation_sources)
    test_set = set(test_sources)
    if (
        len(train_set) != len(list(train_sources))
        or len(validation_set) != len(list(validation_sources))
        or len(test_set) != len(list(test_sources))
    ):
        raise ValueError("source sets must not repeat a source id")
    overlap = (train_set & validation_set) | (train_set & test_set) | (validation_set & test_set)
    if overlap:
        raise SplitContaminationError(
            f"sources {sorted(overlap)} are assigned to more than one split side; "
            "a source observation must train or test, never both"
        )
    requested = train_set | validation_set | test_set
    unknown = sorted(requested - set(by_source))
    if unknown:
        raise ValueError(f"unknown source ids requested: {unknown}")
    train = tuple(o for s in sorted(train_set) for o in by_source[s])
    validation = tuple(o for s in sorted(validation_set) for o in by_source[s])
    test = tuple(o for s in sorted(test_set) for o in by_source[s])
    if not train:
        raise EmptyHoldoutError("train side is empty; a split must train on something")
    if not test:
        raise EmptyHoldoutError("test side is empty; a holdout must hold something out")
    split = GroupSplit(train=train, validation=validation, test=test, description=description)
    check_no_leakage(split)
    return split


def later_time_holdout(
    observations: Sequence[AttributionObservation], *, cutoff: float
) -> GroupSplit:
    """Train on observations before ``cutoff``, test on later ones.

    A source with observations on both sides of the cutoff spans the
    boundary and is contamination: rejected, not assigned arbitrarily.
    """
    if not math.isfinite(cutoff):
        raise ValueError(f"cutoff must be finite, got {cutoff}")
    known = validate_observations(observations)
    before = {o.source_id for o in known if o.collected_at < cutoff}
    after = {o.source_id for o in known if o.collected_at >= cutoff}
    spanning = sorted(before & after)
    if spanning:
        raise SplitContaminationError(
            f"sources {spanning} span the time cutoff {cutoff}; a source cannot "
            "train on its past and test on its future"
        )
    return split_by_source(
        known,
        train_sources=sorted(before),
        test_sources=sorted(after),
        description=f"later-time holdout at collected_at >= {cutoff}",
    )


def cross_route_holdout(
    observations: Sequence[AttributionObservation],
    *,
    held_out_routes: Sequence[str],
) -> GroupSplit:
    """Train on all routes except ``held_out_routes``; test on those routes.

    A source served over routes on both sides spans the boundary and is
    contamination. (Source labels already pin one route per source, so a
    spanning source cannot occur from validated input; the check stays as
    the explicit guard.)
    """
    held_out = set(held_out_routes)
    if not held_out or any(not route for route in held_out):
        raise ValueError("held_out_routes must be a non-empty set of route names")
    known = validate_observations(observations)
    train_sources = sorted({o.source_id for o in known if o.route not in held_out})
    test_sources = sorted({o.source_id for o in known if o.route in held_out})
    if set(train_sources) & set(test_sources):
        raise SplitContaminationError(
            "a source spans the held-out routes; route holdouts need route-pinned sources"
        )
    return split_by_source(
        known,
        train_sources=train_sources,
        test_sources=test_sources,
        description=f"cross-route holdout on routes {sorted(held_out)}",
    )


def unknown_family_holdout(
    observations: Sequence[AttributionObservation],
    *,
    held_out_families: Sequence[str],
) -> GroupSplit:
    """Hold out entire families: the train side never sees them.

    The classifier must answer ``unknown`` on the test side (see
    :func:`predict_with_reject`); a family appearing on both sides would
    make the "unknown" measurement meaningless and is rejected.
    """
    held_out = set(held_out_families)
    if not held_out or any(not family for family in held_out):
        raise ValueError("held_out_families must be a non-empty set of family names")
    known = validate_observations(observations)
    train_sources = sorted({o.source_id for o in known if o.family not in held_out})
    test_sources = sorted({o.source_id for o in known if o.family in held_out})
    if set(train_sources) & set(test_sources):
        raise SplitContaminationError(
            "a source spans the held-out families; family holdouts need family-pinned sources"
        )
    split = split_by_source(
        known,
        train_sources=train_sources,
        test_sources=test_sources,
        description=f"unknown-family holdout on families {sorted(held_out)}",
    )
    train_families = {o.family for o in split.train}
    leaked = sorted(train_families & held_out)
    if leaked:
        raise SplitContaminationError(
            f"held-out families {leaked} appear in train; the unknown outcome "
            "cannot be measured on seen families"
        )
    return split


def ablate_features(
    observations: Sequence[AttributionObservation],
    drop_indices: Sequence[int],
) -> tuple[AttributionObservation, ...]:
    """Return observations with feature positions removed (ablation runs).

    Used for protocol/timing ablations: retrain and rescore without the
    suspect feature group and compare. Dropping every feature raises
    rather than producing an empty-vector model.
    """
    known = validate_observations(observations)
    drop = set(drop_indices)
    dim = len(known[0].features)
    if any(index < 0 or index >= dim for index in drop):
        raise ValueError(f"drop_indices must lie in [0, {dim}), got {sorted(drop)}")
    if len(drop) >= dim:
        raise ValueError("ablation must keep at least one feature")
    ablated: list[AttributionObservation] = []
    for observation in known:
        kept = tuple(value for index, value in enumerate(observation.features) if index not in drop)
        ablated.append(
            AttributionObservation(
                observation_id=observation.observation_id,
                source_id=observation.source_id,
                family=observation.family,
                exact_version=observation.exact_version,
                endpoint_id=observation.endpoint_id,
                route=observation.route,
                collected_at=observation.collected_at,
                features=kept,
            )
        )
    return tuple(ablated)


def split_label_counts(split: GroupSplit, level: str) -> dict[str, dict[str, int]]:
    """Per-side label counts, so tests can assert holdout distinctness."""
    if level not in SUPPORTED_LEVELS:
        raise ValueError(f"level must be one of {SUPPORTED_LEVELS}, got {level!r}")
    counts: dict[str, dict[str, int]] = {}
    for name, side in (
        ("train", split.train),
        ("validation", split.validation),
        ("test", split.test),
    ):
        side_counts: dict[str, int] = {}
        for observation in side:
            label = observation_label(observation, level)
            side_counts[label] = side_counts.get(label, 0) + 1
        counts[name] = side_counts
    return counts


@dataclass(frozen=True, slots=True)
class CentroidModel:
    """Nearest-centroid attribution model: one mean vector per label.

    Deliberately simple (BENCHMARK_PLAN.md: "a simple classifier"): the
    evidence gate judges it on untouched holdouts, so transparency beats
    capacity here. Probabilities and calibration live in
    :mod:`stealthbench.fingerprints.calibration`.
    """

    label_level: str
    labels: tuple[str, ...]
    centroids: tuple[tuple[float, ...], ...]
    feature_dim: int
    trained_sources: tuple[str, ...]
    n_train: int

    def __post_init__(self) -> None:
        if self.label_level not in SUPPORTED_LEVELS:
            raise ValueError(
                f"label_level must be one of {SUPPORTED_LEVELS}, got {self.label_level!r}"
            )
        if not self.labels or len(set(self.labels)) != len(self.labels):
            raise ValueError("labels must be non-empty and unique")
        if len(self.centroids) != len(self.labels):
            raise ValueError("one centroid per label is required")
        if self.feature_dim <= 0:
            raise ValueError(f"feature_dim must be positive, got {self.feature_dim}")
        for label, centroid in zip(self.labels, self.centroids, strict=True):
            if len(centroid) != self.feature_dim:
                raise ValueError(f"centroid for {label!r} has the wrong dimension")
            if not all(math.isfinite(value) for value in centroid):
                raise ValueError(f"centroid for {label!r} is not finite")
        if self.n_train <= 0:
            raise ValueError(f"n_train must be positive, got {self.n_train}")


def fit_centroid_classifier(
    train: Sequence[AttributionObservation], *, level: str = "family"
) -> CentroidModel:
    """Fit one centroid (mean vector) per label on the train side."""
    if level not in SUPPORTED_LEVELS:
        raise ValueError(f"level must be one of {SUPPORTED_LEVELS}, got {level!r}")
    known = validate_observations(train)
    sums: dict[str, list[float]] = {}
    counts: dict[str, int] = {}
    for observation in known:
        label = observation_label(observation, level)
        if label not in sums:
            sums[label] = [0.0] * len(observation.features)
            counts[label] = 0
        for index, value in enumerate(observation.features):
            sums[label][index] += value
        counts[label] += 1
    labels = tuple(sorted(sums))
    dim = len(known[0].features)
    centroids = tuple(tuple(total / counts[label] for total in sums[label]) for label in labels)
    return CentroidModel(
        label_level=level,
        labels=labels,
        centroids=centroids,
        feature_dim=dim,
        trained_sources=tuple(sorted({o.source_id for o in known})),
        n_train=len(known),
    )


def distances_to_centroids(model: CentroidModel, features: Sequence[float]) -> dict[str, float]:
    """Euclidean distance from ``features`` to each class centroid."""
    vector = tuple(features)
    if len(vector) != model.feature_dim:
        raise ValueError(
            f"expected {model.feature_dim} features, got {len(vector)}; "
            "vectors are never padded or truncated"
        )
    if not all(math.isfinite(value) for value in vector):
        raise ValueError("features must be finite")
    return {
        label: math.dist(vector, centroid)
        for label, centroid in zip(model.labels, model.centroids, strict=True)
    }


def predict_known(model: CentroidModel, features: Sequence[float]) -> str:
    """Nearest-centroid label. Never returns ``unknown``; use
    :func:`predict_with_reject` when the unknown outcome is in scope."""
    distances = distances_to_centroids(model, features)
    return min(sorted(distances), key=lambda label: distances[label])


def predict_with_reject(
    model: CentroidModel, features: Sequence[float], *, max_distance: float
) -> str | None:
    """Nearest label, or ``None`` (``unknown``) when the nearest centroid
    is farther than ``max_distance``.

    ``max_distance`` is experimental until frozen on development data
    before holdout scoring; see
    :mod:`stealthbench.fingerprints.calibration`.
    """
    if not math.isfinite(max_distance) or max_distance < 0:
        raise ValueError(f"max_distance must be a non-negative finite value, got {max_distance}")
    distances = distances_to_centroids(model, features)
    best = min(sorted(distances), key=lambda label: distances[label])
    if distances[best] > max_distance:
        return None
    return best
