"""G12 T12A: grouped attribution splits and centroid classifier (unit)."""

from __future__ import annotations

import pytest

from stealthbench.fingerprints.attribution import (
    AttributionObservation,
    EmptyHoldoutError,
    GroupSplit,
    SplitContaminationError,
    ablate_features,
    check_no_leakage,
    cross_route_holdout,
    fit_centroid_classifier,
    later_time_holdout,
    predict_known,
    predict_with_reject,
    split_by_source,
    split_label_counts,
    unknown_family_holdout,
    validate_observations,
)

pytestmark = pytest.mark.unit


def _obs(
    oid: str,
    source: str,
    family: str,
    *,
    version: str = "v1",
    endpoint: str = "ep1",
    route: str = "r1",
    at: float = 1.0,
    features: tuple[float, ...] = (0.0, 0.0),
) -> AttributionObservation:
    return AttributionObservation(
        observation_id=oid,
        source_id=source,
        family=family,
        exact_version=version,
        endpoint_id=endpoint,
        route=route,
        collected_at=at,
        features=features,
    )


def _separable() -> list[AttributionObservation]:
    """Two well-separated families across distinct sources."""
    return [
        _obs("a1", "sA1", "alpha", features=(0.0, 0.0)),
        _obs("a2", "sA1", "alpha", features=(0.2, -0.1)),
        _obs("a3", "sA2", "alpha", features=(-0.1, 0.2)),
        _obs("b1", "sB1", "beta", features=(10.0, 10.0)),
        _obs("b2", "sB1", "beta", features=(10.1, 9.9)),
        _obs("b3", "sB2", "beta", features=(9.8, 10.2)),
    ]


def test_duplicate_observation_ids_are_contamination() -> None:
    observations = [_obs("dup", "s1", "alpha"), _obs("dup", "s2", "beta")]
    with pytest.raises(SplitContaminationError):
        validate_observations(observations)


def test_source_label_disagreement_is_contamination() -> None:
    observations = [_obs("o1", "s1", "alpha"), _obs("o2", "s1", "beta")]
    with pytest.raises(SplitContaminationError):
        validate_observations(observations)


def test_source_observations_stay_together() -> None:
    observations = _separable()
    split = split_by_source(
        observations,
        train_sources=["sA1", "sB1"],
        test_sources=["sA2", "sB2"],
        description="grouped",
    )
    assert {o.observation_id for o in split.train} == {"a1", "a2", "b1", "b2"}
    assert {o.observation_id for o in split.test} == {"a3", "b3"}
    check_no_leakage(split)


def test_shared_source_between_sides_is_rejected() -> None:
    with pytest.raises(SplitContaminationError):
        split_by_source(
            _separable(),
            train_sources=["sA1", "sB1"],
            test_sources=["sB1", "sB2"],
        )


def test_manually_built_leaking_split_is_rejected() -> None:
    first = _obs("o1", "s1", "alpha")
    split = GroupSplit(train=(first,), validation=(), test=(first,), description="leak")
    with pytest.raises(SplitContaminationError):
        check_no_leakage(split)


def test_unknown_source_ids_are_rejected() -> None:
    with pytest.raises(ValueError, match="unknown source"):
        split_by_source(_separable(), train_sources=["sA1"], test_sources=["nope"])


def test_empty_test_side_is_rejected() -> None:
    with pytest.raises(EmptyHoldoutError):
        split_by_source(_separable(), train_sources=["sA1", "sB1"], test_sources=[])


def test_unknown_family_holdout_hides_the_family() -> None:
    observations = [
        *_separable(),
        _obs("c1", "sC1", "gamma", features=(20.0, 0.0)),
        _obs("c2", "sC1", "gamma", features=(20.2, -0.2)),
    ]
    split = unknown_family_holdout(observations, held_out_families=["gamma"])
    counts = split_label_counts(split, "family")
    assert set(counts["train"]) == {"alpha", "beta"}
    assert counts["test"] == {"gamma": 2}
    model = fit_centroid_classifier(split.train, level="family")
    assert set(model.labels) == {"alpha", "beta"}
    for held_out in split.test:
        assert predict_with_reject(model, held_out.features, max_distance=5.0) is None
    for trained in split.train:
        assert predict_with_reject(model, trained.features, max_distance=5.0) is not None


def test_unknown_family_holdout_rejects_missing_family() -> None:
    with pytest.raises(EmptyHoldoutError):
        unknown_family_holdout(_separable(), held_out_families=["gamma"])


def test_later_time_holdout_splits_on_time() -> None:
    observations = [
        _obs("early1", "sE1", "alpha", at=1.0),
        _obs("early2", "sE2", "beta", at=2.0),
        _obs("late1", "sL1", "alpha", at=9.0),
        _obs("late2", "sL2", "beta", at=10.0),
    ]
    split = later_time_holdout(observations, cutoff=5.0)
    assert all(o.collected_at < 5.0 for o in split.train)
    assert all(o.collected_at >= 5.0 for o in split.test)
    assert {o.source_id for o in split.test} == {"sL1", "sL2"}


def test_source_spanning_time_cutoff_is_contamination() -> None:
    observations = [
        _obs("o1", "s1", "alpha", at=1.0),
        _obs("o2", "s1", "alpha", at=9.0),
    ]
    with pytest.raises(SplitContaminationError):
        later_time_holdout(observations, cutoff=5.0)


def test_later_time_holdout_rejects_empty_side() -> None:
    with pytest.raises(EmptyHoldoutError):
        later_time_holdout(_separable(), cutoff=100.0)


def test_cross_route_holdout_splits_on_route() -> None:
    observations = [
        _obs("o1", "s1", "alpha", route="r1"),
        _obs("o2", "s2", "beta", route="r1"),
        _obs("o3", "s3", "alpha", route="r2"),
    ]
    split = cross_route_holdout(observations, held_out_routes=["r2"])
    assert {o.route for o in split.train} == {"r1"}
    assert {o.route for o in split.test} == {"r2"}


def test_holdout_kinds_select_distinct_test_sets() -> None:
    observations = [
        _obs("o1", "s1", "alpha", route="r1", at=1.0, features=(0.0, 0.0)),
        _obs("o2", "s2", "alpha", route="r1", at=2.0, features=(0.1, 0.0)),
        _obs("o3", "s3", "beta", route="r2", at=1.0, features=(10.0, 10.0)),
        _obs("o4", "s4", "gamma", route="r1", at=9.0, features=(20.0, 0.0)),
    ]
    time_split = later_time_holdout(observations, cutoff=5.0)
    route_split = cross_route_holdout(observations, held_out_routes=["r2"])
    family_split = unknown_family_holdout(observations, held_out_families=["gamma"])
    time_test = {o.source_id for o in time_split.test}
    route_test = {o.source_id for o in route_split.test}
    family_test = {o.source_id for o in family_split.test}
    assert time_test == {"s4"}
    assert route_test == {"s3"}
    assert family_test == {"s4"}
    assert time_test != route_test
    assert "later-time" in time_split.description
    assert "cross-route" in route_split.description
    assert "unknown-family" in family_split.description


def test_centroid_classifier_attributes_known_samples() -> None:
    model = fit_centroid_classifier(_separable(), level="family")
    assert predict_known(model, (0.1, 0.1)) == "alpha"
    assert predict_known(model, (9.9, 10.1)) == "beta"


def test_classifier_rejects_bad_feature_dimensions() -> None:
    model = fit_centroid_classifier(_separable(), level="family")
    with pytest.raises(ValueError, match="features"):
        predict_known(model, (1.0, 2.0, 3.0))


def test_ablation_drops_feature_positions() -> None:
    ablated = ablate_features(_separable(), [1])
    assert all(len(o.features) == 1 for o in ablated)
    model = fit_centroid_classifier(ablated, level="family")
    assert model.feature_dim == 1
    with pytest.raises(ValueError, match="ablation"):
        ablate_features(_separable(), [0, 1])
