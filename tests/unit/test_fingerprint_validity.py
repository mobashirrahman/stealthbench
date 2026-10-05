"""G11 T11B: signature extraction and validity masks (unit)."""

from __future__ import annotations

import pytest

from stealthbench.fingerprints.features import (
    aggregate_repeats,
    baseline_aggregate,
    classify_framing_offsets,
    delta_vector,
    delta_vector_for_result,
    extract_signature,
    flag_content_dependent_probes,
    median_of,
    signal_available,
    to_contract_dict,
    validity_mask,
)
from stealthbench.schemas.results import ProbeValidity

pytestmark = pytest.mark.unit


def test_stable_repeats_aggregate_to_the_common_count() -> None:
    count, validity = aggregate_repeats([12, 12, 12])
    assert (count, validity) == (12, ProbeValidity.VALID)
    assert median_of([3, 1, 2]) == 2


def test_missing_repeats_are_explicitly_invalid_without_imputation() -> None:
    for repeats in ([12, None, 12], [None, None, None], []):
        count, validity = aggregate_repeats(repeats)
        assert count is None
        assert validity is ProbeValidity.MISSING_COUNT
        assert count != 0  # missing is None, never a zero that looks measured


def test_unstable_repeats_carry_no_count() -> None:
    count, validity = aggregate_repeats([10, 11, 10])
    assert count is None
    assert validity is ProbeValidity.UNSTABLE_REPEAT
    count, validity = aggregate_repeats([10, 10, 12], tolerance=1)
    assert count is None
    assert validity is ProbeValidity.UNSTABLE_REPEAT
    count, validity = aggregate_repeats([10, 10, 11], tolerance=1)
    assert (count, validity) == (10, ProbeValidity.VALID)


def test_delta_subtracts_the_baseline_and_propagates_missing() -> None:
    assert delta_vector([12, 15, 9], 10) == (2, 5, -1)
    assert delta_vector([12, None, 9], 10) == (2, None, -1)
    assert delta_vector([12, 15], None) == (None, None)
    baseline, validity = baseline_aggregate([20, 20, 20])
    assert (baseline, validity) == (20, ProbeValidity.VALID)


def test_constant_framing_offsets_cancel_out_of_deltas() -> None:
    raw = [12, 15, 9]
    baseline = 10
    framed_raw = [value + 7 for value in raw]
    framed_baseline = baseline + 7
    assert delta_vector(raw, baseline) == delta_vector(framed_raw, framed_baseline)
    assert classify_framing_offsets([7, 7, 7]) == "constant"
    assert flag_content_dependent_probes([7, 7, 7]) == (False, False, False)


def test_content_dependent_offsets_are_flagged() -> None:
    assert classify_framing_offsets([7, 7, 9]) == "content_dependent"
    assert flag_content_dependent_probes([7, 7, 12]) == (False, False, True)
    assert flag_content_dependent_probes([None, 7, 7]) == (False, False, False)


def test_extract_signature_marks_each_failure_mode_explicitly() -> None:
    result = extract_signature(
        probe_ids=["p1", "p2", "p3", "p4"],
        observations={
            "p1": [10, 10, 10],
            "p2": [11, None, 11],
            "p3": [10, 14, 10],
            # p4 absent entirely: missing evidence, not an error.
        },
        baseline_observations=[8, 8, 8],
        probe_version="probe-v1",
        endpoint_id="ep-a",
        campaign_id="camp-a",
    )
    assert result.input_count_vector == (10, None, None, None)
    assert result.validity == (
        ProbeValidity.VALID,
        ProbeValidity.MISSING_COUNT,
        ProbeValidity.UNSTABLE_REPEAT,
        ProbeValidity.MISSING_COUNT,
    )
    assert validity_mask(result) == (True, False, False, False)
    assert result.usable_probes == 1
    # No imputation anywhere: invalid slots are None, never 0.
    for count, validity in zip(result.input_count_vector, result.validity, strict=True):
        if validity is not ProbeValidity.VALID:
            assert count is None


def test_extract_signature_flags_content_dependent_framing() -> None:
    result = extract_signature(
        probe_ids=["p1", "p2", "p3"],
        observations={"p1": [10, 10, 10], "p2": [12, 12, 12], "p3": [14, 14, 14]},
        baseline_observations=[8, 8, 8],
        probe_version="probe-v1",
        endpoint_id="ep-a",
        campaign_id="camp-a",
        aggregated_counts={"p1": 10, "p2": 12, "p3": 14},
        alternate_framing_counts={"p1": 17, "p2": 19, "p3": 30},
    )
    assert result.validity[0] is ProbeValidity.VALID
    assert result.validity[1] is ProbeValidity.VALID
    assert result.validity[2] is ProbeValidity.CONTENT_DEPENDENT_OFFSET
    assert result.input_count_vector[2] is None


def test_extract_signature_needs_both_framings_together() -> None:
    with pytest.raises(ValueError, match="together"):
        extract_signature(
            probe_ids=["p1"],
            observations={"p1": [10, 10, 10]},
            baseline_observations=[8, 8, 8],
            probe_version="probe-v1",
            endpoint_id="ep-a",
            campaign_id="camp-a",
            aggregated_counts={"p1": 10},
        )


def test_signal_availability_and_contract_mapping() -> None:
    full = extract_signature(
        probe_ids=[f"p{i}" for i in range(6)],
        observations={f"p{i}": [10 + i, 10 + i, 10 + i] for i in range(6)},
        baseline_observations=[8, 8, 8],
        probe_version="probe-v1",
        endpoint_id="ep-a",
        campaign_id="camp-a",
    )
    assert signal_available(full) is True
    sparse = extract_signature(
        probe_ids=[f"p{i}" for i in range(6)],
        observations={f"p{i}": [10, 10, 10] if i == 0 else [None, None, None] for i in range(6)},
        baseline_observations=[8, 8, 8],
        probe_version="probe-v1",
        endpoint_id="ep-a",
        campaign_id="camp-a",
    )
    assert signal_available(sparse) is False
    assert delta_vector_for_result(full, 8) == tuple(v - 8 for v in (10, 11, 12, 13, 14, 15))
    contract = to_contract_dict(full)
    assert "validity_mask" in contract
    assert contract["validity_mask"] == ["valid"] * 6
    assert contract["probe_version"] == "probe-v1"


def test_extract_signature_rejects_empty_or_duplicate_ids() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        extract_signature(
            probe_ids=[],
            observations={},
            baseline_observations=[1, 1, 1],
            probe_version="probe-v1",
            endpoint_id="ep-a",
            campaign_id="camp-a",
        )
    with pytest.raises(ValueError, match="unique"):
        extract_signature(
            probe_ids=["p1", "p1"],
            observations={"p1": [1, 1, 1]},
            baseline_observations=[1, 1, 1],
            probe_version="probe-v1",
            endpoint_id="ep-a",
            campaign_id="camp-a",
        )
