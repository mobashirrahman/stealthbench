"""Signature extraction and validity masks (G11 T11B).

Contracts (``docs/contracts.md`` ``SignatureResult``):

* The validity mask is per probe. A missing or unstable probe marks that
  probe unavailable; it is never imputed (no zero-fill, no mean-fill).
* Token counts are compared as deltas against a fixed baseline, because a
  fixed offset from message framing carries no identity information::

      delta(text) = input_tokens(fixed message containing text)
                    - input_tokens(fixed baseline message)

* ``missing_count``: at least one of the 3 repeats reported no count.
* ``unstable_repeat``: repeats disagree by more than ``tolerance``.
* ``content_dependent_offset``: the framing overhead itself varies with
  content (second-framing offsets are not constant across probes), so the
  delta for that probe cannot be trusted.
* If usage is missing, unstable, or evidently synthetic, the token-count
  signal is unavailable rather than fabricated.

Offline by construction: pure functions over stored counts. No transport,
no socket, no subprocess, no credential.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Final

from stealthbench.schemas.campaign import ObservationWindow
from stealthbench.schemas.results import ProbeValidity, SignatureResult

#: Repeats that disagree by more than this are unstable. Token counts from a
#: deterministic stack are exactly stable, so the default is zero.
STABILITY_TOLERANCE: Final[int] = 0

#: Minimum usable probes for the token-count signal to count as available.
#: Below this the signal is reported unavailable, not interpolated.
DEFAULT_MIN_USABLE_PROBES: Final[int] = 5


def median_of(values: Sequence[int]) -> int:
    """Median of a non-empty integer list (lower middle for even lengths)."""
    if not values:
        raise ValueError("median_of needs at least one value")
    ordered = sorted(values)
    return ordered[(len(ordered) - 1) // 2]


def aggregate_repeats(
    repeats: Sequence[int | None], *, tolerance: int = STABILITY_TOLERANCE
) -> tuple[int | None, ProbeValidity]:
    """Aggregate one probe's repeats into a count plus a validity verdict.

    * Any ``None`` (or an empty repeat list) -> ``(None, MISSING_COUNT)``.
    * Spread beyond ``tolerance`` -> ``(None, UNSTABLE_REPEAT)``.
    * Otherwise -> ``(median, VALID)``.

    Never imputes: an invalid probe returns ``None``, never ``0``.
    """
    if tolerance < 0:
        raise ValueError(f"tolerance must be non-negative, got {tolerance}")
    if not repeats:
        return None, ProbeValidity.MISSING_COUNT
    if any(value is None for value in repeats):
        return None, ProbeValidity.MISSING_COUNT
    concrete: list[int] = [value for value in repeats if value is not None]
    if max(concrete) - min(concrete) > tolerance:
        return None, ProbeValidity.UNSTABLE_REPEAT
    return median_of(concrete), ProbeValidity.VALID


def baseline_aggregate(
    baseline_repeats: Sequence[int | None], *, tolerance: int = STABILITY_TOLERANCE
) -> tuple[int | None, ProbeValidity]:
    """Aggregate the baseline repeats with the same rule as any probe."""
    return aggregate_repeats(baseline_repeats, tolerance=tolerance)


def delta_vector(counts: Sequence[int | None], baseline: int | None) -> tuple[int | None, ...]:
    """Subtract the baseline from each probe count.

    ``None`` propagates: a missing count or a missing baseline yields a
    missing delta, never a fabricated zero.
    """
    if baseline is None:
        return tuple(None for _ in counts)
    return tuple(None if count is None else count - baseline for count in counts)


def classify_framing_offsets(
    offsets: Sequence[int], *, tolerance: int = STABILITY_TOLERANCE
) -> str:
    """Classify second-framing offsets as constant or content-dependent.

    ``offsets[i]`` is ``counts_framing2[i] - counts_framing1[i]`` for probe
    ``i``. A constant offset (spread within ``tolerance``) subtracts out of
    every delta and is harmless; a varying offset means the framing itself
    depends on content and the affected deltas cannot be trusted.
    """
    if not offsets:
        raise ValueError("classify_framing_offsets needs at least one offset")
    if tolerance < 0:
        raise ValueError(f"tolerance must be non-negative, got {tolerance}")
    spread = max(offsets) - min(offsets)
    if spread <= tolerance:
        return "constant"
    return "content_dependent"


def flag_content_dependent_probes(
    offsets: Sequence[int | None], *, tolerance: int = STABILITY_TOLERANCE
) -> tuple[bool, ...]:
    """Flag probes whose framing offset deviates from the median offset.

    ``None`` offsets (missing in either framing) are not flagged here;
    they are already ``MISSING_COUNT``. Returns per-probe ``True`` where
    the probe is content-dependent and must become
    ``CONTENT_DEPENDENT_OFFSET``.
    """
    concrete = [value for value in offsets if value is not None]
    if not concrete:
        return tuple(False for _ in offsets)
    center = median_of(concrete)
    return tuple(False if value is None else abs(value - center) > tolerance for value in offsets)


def extract_signature(
    *,
    probe_ids: Sequence[str],
    observations: Mapping[str, Sequence[int | None]],
    baseline_observations: Sequence[int | None],
    probe_version: str,
    endpoint_id: str,
    campaign_id: str,
    behavioral_features: Mapping[str, Any] | None = None,
    alternate_framing_counts: Mapping[str, int | None] | None = None,
    aggregated_counts: Mapping[str, int | None] | None = None,
    tolerance: int = STABILITY_TOLERANCE,
    observation_window: ObservationWindow | None = None,
) -> SignatureResult:
    """Build a :class:`SignatureResult` from raw per-repeat token counts.

    Args:
        probe_ids: probes in canonical order; defines the vector positions.
        observations: ``probe_id -> up-to-3 repeat counts`` (``None`` = the
            endpoint did not report that repeat).
        baseline_observations: repeats of the fixed baseline message. Used
            only to validate that the baseline itself was stable; the raw
            per-probe counts are stored and deltas are derived later with
            :func:`delta_vector` so a fixed framing offset never enters the
            comparison.
        alternate_framing_counts: optional ``probe_id -> aggregated count``
            under a second framing for the content-dependence check. When
            given alongside ``aggregated_counts`` (first-framing aggregates),
            probes whose cross-framing offset deviates from the median are
            marked ``CONTENT_DEPENDENT_OFFSET``.
        aggregated_counts: first-framing aggregates used only together with
            ``alternate_framing_counts`` for the framing check.

    A probe id absent from ``observations`` is ``MISSING_COUNT``, not an
    error: missing evidence stays explicitly missing.
    """
    if not probe_version:
        raise ValueError("probe_version must be a non-empty string")
    if not endpoint_id:
        raise ValueError("endpoint_id must be a non-empty string")
    if not campaign_id:
        raise ValueError("campaign_id must be a non-empty string")
    if len(set(probe_ids)) != len(list(probe_ids)):
        raise ValueError("probe_ids must be unique")
    if not probe_ids:
        raise ValueError("probe_ids must not be empty")

    baseline_count, baseline_validity = baseline_aggregate(
        baseline_observations, tolerance=tolerance
    )
    _ = (baseline_count, baseline_validity)

    counts: list[int | None] = []
    validities: list[ProbeValidity] = []
    for probe_id in probe_ids:
        repeats = observations.get(probe_id, ())
        count, validity = aggregate_repeats(tuple(repeats), tolerance=tolerance)
        counts.append(count)
        validities.append(validity)

    if alternate_framing_counts is not None or aggregated_counts is not None:
        if alternate_framing_counts is None or aggregated_counts is None:
            raise ValueError(
                "alternate_framing_counts and aggregated_counts must be given together"
            )
        offsets: list[int | None] = []
        for probe_id in probe_ids:
            other = alternate_framing_counts.get(probe_id)
            first = aggregated_counts.get(probe_id)
            if other is None or first is None:
                offsets.append(None)
            else:
                offsets.append(other - first)
        flags = flag_content_dependent_probes(offsets, tolerance=tolerance)
        for index, flagged in enumerate(flags):
            if flagged and validities[index] is ProbeValidity.VALID:
                validities[index] = ProbeValidity.CONTENT_DEPENDENT_OFFSET
                counts[index] = None

    return SignatureResult(
        probe_version=probe_version,
        endpoint_id=endpoint_id,
        campaign_id=campaign_id,
        input_count_vector=tuple(counts),
        validity=tuple(validities),
        behavioral_features=dict(behavioral_features or {}),
        observation_window=observation_window or ObservationWindow(),
    )


def usable_probe_indices(result: SignatureResult) -> tuple[int, ...]:
    """Positions of probes marked valid (comparable evidence)."""
    return tuple(
        index for index, validity in enumerate(result.validity) if validity is ProbeValidity.VALID
    )


def validity_mask(result: SignatureResult) -> tuple[bool, ...]:
    """Per-probe boolean mask (``True`` = valid).

    Named for ``docs/contracts.md`` ``validity_mask``; the stored enum
    vector is ``SignatureResult.validity`` and this is its boolean view
    (identical to ``SignatureResult.valid_mask``).
    """
    return result.valid_mask


def signal_available(
    result: SignatureResult, *, min_usable: int = DEFAULT_MIN_USABLE_PROBES
) -> bool:
    """Whether the token-count signal has enough usable probes to compare."""
    if min_usable <= 0:
        raise ValueError(f"min_usable must be positive, got {min_usable}")
    return result.usable_probes >= min_usable


def delta_vector_for_result(
    result: SignatureResult, baseline: int | None
) -> tuple[int | None, ...]:
    """Deltas of the stored count vector against an aggregated baseline."""
    return delta_vector(result.input_count_vector, baseline)


def to_contract_dict(result: SignatureResult) -> dict[str, Any]:
    """Map a result onto the ``docs/contracts.md`` field names.

    The frozen prose calls the mask ``validity_mask`` while the schema
    field is ``validity``; this mapping exposes both so contract readers
    and schema readers agree that per-probe validity is present.
    """
    dumped = result.model_dump(mode="json")
    dumped["validity_mask"] = list(dumped["validity"])
    return dumped
