"""Ranked reference similarity reports (G11 T11C).

Contracts (``BENCHMARK_PLAN.md`` "Prediction and validation",
``docs/contracts.md`` ``IdentityReport``):

* Start with transparent nearest-reference similarities per signal and a
  ranked candidate shortlist.
* Do not turn a similarity of 0.9 into a 90% probability:
  ``calibrated_probabilities`` stays ``None`` until the G12 evidence gate
  passes, and :class:`RankedSimilarity` has no probability field at all.
* Shared-tokenizer evidence is tokenization evidence only. Two different
  models can share a tokenizer (arXiv 2608.29930), so an exact
  count-vector match never becomes an identical-model claim.
* Without sufficient comparable evidence the report abstains with
  ``unknown`` / ``insufficient_evidence`` rather than ranking candidates.
* Comparisons use baseline-subtracted deltas, so a constant framing
  offset cannot inflate a similarity.

Offline by construction: pure functions over stored signatures. No
transport, no socket, no subprocess, no credential.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from stealthbench.fingerprints.features import (
    DEFAULT_MIN_USABLE_PROBES,
    delta_vector,
)
from stealthbench.schemas.results import (
    IdentityReport,
    ProbeValidity,
    RankedSimilarity,
    SignalEvidence,
    SignatureResult,
)

#: Minimum overlapping valid probes for one token-count comparison.
MIN_COMPARABLE_PROBES: Final[int] = 5

#: Abstention reason when no reference offers enough comparable evidence.
INSUFFICIENT_EVIDENCE_REASON: Final[str] = "insufficient_evidence"

#: Reason recorded when every reference is skipped for revision mismatch.
INCOMPATIBLE_LIBRARY_REASON: Final[str] = "incompatible_reference_library"

#: Mandatory note on any tokenizer-only ranking. Repeats the arXiv
#: 2608.29930 caveat so a count match cannot be read as an identity claim.
TOKENIZER_ONLY_NOTE: Final[str] = (
    "tokenization evidence only; does not establish shared weights "
    "(arXiv 2608.29930: shared tokenizer != identical model)"
)

#: Note added when counts match exactly but behavior diverges.
DIVERGENT_BEHAVIOR_NOTE: Final[str] = (
    "exact count-vector match with divergent behavior; " + TOKENIZER_ONLY_NOTE
)

#: Signals this module can score. The remaining benchmark-plan signals
#: (actual token IDs, capability overlap beyond behaviour text,
#: protocol, performance) are recorded as unavailable in similarity mode
#: unless the caller supplies comparable observations for them.
SCORED_SIGNALS: Final[tuple[str, ...]] = ("input_count_vector", "behavior")


@dataclass(frozen=True, slots=True)
class ReferenceSignature:
    """One reference candidate's stored signature plus its identity labels.

    The four labels stay separate: provider, family, exact version,
    tokenizer and route are independent facts. A shared ``label_tokenizer``
    is tokenization evidence only, never an identity claim.
    """

    candidate_id: str
    candidate_library_revision: str
    probe_version: str
    input_counts: tuple[int | None, ...]
    validity: tuple[ProbeValidity, ...]
    baseline_count: int | None
    behavioral_features: Mapping[str, Any] | None = None
    label_family: str | None = None
    label_exact_version: str | None = None
    label_tokenizer: str | None = None
    label_route: str | None = None


def exact_match_fraction(
    first: Sequence[int | None],
    second: Sequence[int | None],
    first_valid: Sequence[bool],
    second_valid: Sequence[bool],
) -> tuple[float | None, int]:
    """Fraction of comparable positions with equal values.

    Comparable = both sides valid and both deltas present. Returns
    ``(None, 0)`` when nothing is comparable; the caller decides the
    sufficiency threshold. Never imputes: invalid positions are excluded,
    not filled.
    """
    if not (len(first) == len(second) == len(first_valid) == len(second_valid)):
        raise ValueError("all inputs must share one length")
    comparable = 0
    equal = 0
    for a_value, b_value, a_ok, b_ok in zip(first, second, first_valid, second_valid, strict=True):
        if not (a_ok and b_ok):
            continue
        if a_value is None or b_value is None:
            continue
        comparable += 1
        if a_value == b_value:
            equal += 1
    if comparable == 0:
        return None, 0
    return equal / comparable, comparable


def token_count_similarity(
    query: SignatureResult,
    query_baseline: int | None,
    reference: ReferenceSignature,
    *,
    min_comparable: int = MIN_COMPARABLE_PROBES,
) -> tuple[float | None, int]:
    """Similarity of baseline-subtracted delta vectors.

    Constant framing offsets cancel in the subtraction, so adding one
    constant to every raw count (including the baseline) leaves the
    result unchanged. Returns ``(None, n)`` when fewer than
    ``min_comparable`` probes overlap.
    """
    if min_comparable <= 0:
        raise ValueError(f"min_comparable must be positive, got {min_comparable}")
    if len(query.input_count_vector) != len(reference.input_counts):
        raise ValueError("query and reference vectors must share one length")
    query_deltas = delta_vector(query.input_count_vector, query_baseline)
    ref_deltas = delta_vector(reference.input_counts, reference.baseline_count)
    query_ok = [validity is ProbeValidity.VALID for validity in query.validity]
    ref_ok = [validity is ProbeValidity.VALID for validity in reference.validity]
    similarity, comparable = exact_match_fraction(query_deltas, ref_deltas, query_ok, ref_ok)
    if similarity is None or comparable < min_comparable:
        return None, comparable
    return similarity, comparable


def _numeric_similarity(first: float, second: float, scale: float = 1.0) -> float:
    """Per-key numeric similarity in ``[0, 1]``; 1.0 means identical."""
    if scale <= 0:
        raise ValueError(f"scale must be positive, got {scale}")
    return 1.0 / (1.0 + abs(first - second) / scale)


def behavior_similarity(
    query_features: Mapping[str, Any],
    reference_features: Mapping[str, Any] | None,
) -> tuple[float | None, int]:
    """Similarity of behavioral feature dicts over shared keys.

    * shared numeric keys -> ``1 / (1 + |a - b|)`` each;
    * shared string keys -> 1.0 when equal, else 0.0;
    * shared equal-length numeric lists -> cosine-style agreement mapped
      to ``[0, 1]`` (1.0 identical, 0.5 orthogonal, 0.0 opposite);
    * anything else (missing keys, mismatched types) is excluded, never
      imputed.

    Returns ``(None, 0)`` when no key is comparable.
    """
    reference = dict(reference_features or {})
    scores: list[float] = []
    for key, query_value in query_features.items():
        if key not in reference:
            continue
        ref_value = reference[key]
        if isinstance(query_value, bool) or isinstance(ref_value, bool):
            continue
        if isinstance(query_value, (int, float)) and isinstance(ref_value, (int, float)):
            scores.append(_numeric_similarity(float(query_value), float(ref_value)))
        elif isinstance(query_value, str) and isinstance(ref_value, str):
            scores.append(1.0 if query_value == ref_value else 0.0)
        elif (
            isinstance(query_value, (list, tuple))
            and isinstance(ref_value, (list, tuple))
            and len(query_value) == len(ref_value)
            and len(query_value) > 0
            and all(isinstance(v, (int, float)) for v in query_value)
            and all(isinstance(v, (int, float)) for v in ref_value)
        ):
            q_list = [float(v) for v in query_value]
            r_list = [float(v) for v in ref_value]
            dot = sum(a * b for a, b in zip(q_list, r_list, strict=True))
            q_norm = sum(a * a for a in q_list) ** 0.5
            r_norm = sum(b * b for b in r_list) ** 0.5
            if q_norm == 0.0 or r_norm == 0.0:
                continue
            cosine = dot / (q_norm * r_norm)
            scores.append((cosine + 1.0) / 2.0)
    if not scores:
        return None, 0
    return sum(scores) / len(scores), len(scores)


def combine_signal_similarities(
    per_signal: Mapping[str, float | None],
) -> float | None:
    """Mean over available per-signal similarities; ``None`` when none."""
    available = [value for value in per_signal.values() if value is not None]
    if not available:
        return None
    return sum(available) / len(available)


def _signal_evidence(
    signal: str,
    available: bool,
    comparable: int,
    detail: str | None = None,
) -> SignalEvidence:
    return SignalEvidence(
        signal=signal,  # type: ignore[arg-type]
        available=available,
        comparable_observations=comparable,
        detail=detail,
    )


def build_identity_report(
    *,
    endpoint_id: str,
    query: SignatureResult,
    query_baseline: int | None,
    references: Sequence[ReferenceSignature],
    candidate_library_revision: str,
    min_comparable: int = MIN_COMPARABLE_PROBES,
    min_usable_probes: int = DEFAULT_MIN_USABLE_PROBES,
) -> IdentityReport:
    """Rank reference candidates against one query signature.

    * References whose ``probe_version`` or ``candidate_library_revision``
      differs from the query/library under test are skipped as
      incomparable (a changed reference library invalidates the
      comparison; it never silently passes).
    * Per-signal similarities use only mutually valid probes. Incomplete
      vectors shrink the comparable count instead of being imputed.
    * An exact token-count match is labelled tokenization evidence only
      (``TOKENIZER_ONLY_NOTE``); it never becomes an identical-model
      claim, and divergent behavior lowers the combined score with an
      explicit note.
    * ``calibrated_probabilities`` is always ``None``: similarity is
      never displayed as probability (G12 owns calibration).
    * With no sufficient comparable evidence the report abstains:
      empty ranking plus ``abstention_reason="insufficient_evidence"``
      (or ``"incompatible_reference_library"`` when revisions alone
      explain the gap).
    """
    if not endpoint_id:
        raise ValueError("endpoint_id must be a non-empty string")
    if not candidate_library_revision:
        raise ValueError("candidate_library_revision must be a non-empty string")
    if min_comparable <= 0:
        raise ValueError(f"min_comparable must be positive, got {min_comparable}")
    if min_usable_probes <= 0:
        raise ValueError(f"min_usable_probes must be positive, got {min_usable_probes}")

    scored: list[RankedSimilarity] = []
    best_token_comparable = 0
    best_behavior_comparable = 0
    skipped_revision = 0
    token_available_any = False
    behavior_available_any = False

    for reference in references:
        if (
            reference.probe_version != query.probe_version
            or reference.candidate_library_revision != candidate_library_revision
        ):
            skipped_revision += 1
            continue
        token_sim, token_n = token_count_similarity(
            query, query_baseline, reference, min_comparable=min_comparable
        )
        beh_sim, beh_n = behavior_similarity(
            query.behavioral_features, reference.behavioral_features
        )
        best_token_comparable = max(best_token_comparable, token_n)
        best_behavior_comparable = max(best_behavior_comparable, beh_n)
        if token_sim is not None:
            token_available_any = True
        if beh_sim is not None:
            behavior_available_any = True
        per_signal: dict[str, float | None] = {
            "input_count_vector": token_sim,
            "behavior": beh_sim,
        }
        overall = combine_signal_similarities(per_signal)
        if overall is None:
            continue
        evidence_signals = tuple(
            sorted(name for name, value in per_signal.items() if value is not None)
        )
        if token_sim == 1.0 and beh_sim is None:
            note: str | None = TOKENIZER_ONLY_NOTE
        elif token_sim == 1.0 and beh_sim is not None and beh_sim < 1.0:
            note = DIVERGENT_BEHAVIOR_NOTE
        elif token_sim is not None and beh_sim is None:
            note = TOKENIZER_ONLY_NOTE if token_sim == 1.0 else None
        else:
            note = None
        scored.append(
            RankedSimilarity(
                candidate_id=reference.candidate_id,
                candidate_library_revision=reference.candidate_library_revision,
                similarity=overall,
                evidence_signals=evidence_signals,
                note=note,
            )
        )

    scored.sort(key=lambda entry: entry.similarity, reverse=True)

    token_evidence = _signal_evidence(
        "input_count_vector",
        token_available_any,
        best_token_comparable,
        detail="delta(text) exact-match fraction over mutually valid probes",
    )
    behavior_evidence = _signal_evidence(
        "behavior",
        behavior_available_any,
        best_behavior_comparable,
        detail="shared behavioral feature keys only; missing keys excluded",
    )
    unavailable = [
        _signal_evidence(
            "actual_tokenization", False, 0, "token IDs unavailable from chat text alone"
        ),
        _signal_evidence(
            "capability_pattern", False, 0, "no benchmark overlap supplied in similarity mode"
        ),
        _signal_evidence(
            "protocol", False, 0, "gateway schema excluded from model-identity scoring"
        ),
        _signal_evidence(
            "performance", False, 0, "timing is weak identity evidence; excluded here"
        ),
    ]
    signal_evidence = (token_evidence, behavior_evidence, *unavailable)

    if not scored:
        if skipped_revision and not references_scored_for_other_reasons(
            token_available_any, behavior_available_any
        ):
            reason = INCOMPATIBLE_LIBRARY_REASON
        else:
            reason = INSUFFICIENT_EVIDENCE_REASON
        return IdentityReport(
            endpoint_id=endpoint_id,
            candidate_library_revision=candidate_library_revision,
            signal_evidence=signal_evidence,
            ranked_similarities=(),
            abstention_reason=reason,
            calibrated_probabilities=None,
            official_reveal_label=None,
        )

    usable = [
        evidence
        for evidence in (token_evidence, behavior_evidence)
        if evidence.available and evidence.comparable_observations > 0
    ]
    abstention: str | None = None
    if not usable:
        abstention = INSUFFICIENT_EVIDENCE_REASON
        return IdentityReport(
            endpoint_id=endpoint_id,
            candidate_library_revision=candidate_library_revision,
            signal_evidence=signal_evidence,
            ranked_similarities=(),
            abstention_reason=abstention,
            calibrated_probabilities=None,
            official_reveal_label=None,
        )
    return IdentityReport(
        endpoint_id=endpoint_id,
        candidate_library_revision=candidate_library_revision,
        signal_evidence=signal_evidence,
        ranked_similarities=tuple(scored),
        abstention_reason=None,
        calibrated_probabilities=None,
        official_reveal_label=None,
    )


def references_scored_for_other_reasons(token_available: bool, behavior_available: bool) -> bool:
    """Whether any signal was comparable despite revision skips."""
    return token_available or behavior_available


def assert_no_probabilities(report: IdentityReport) -> None:
    """Raise if a report carries anything probability-shaped.

    Similarity mode must never display a probability: calibrated
    probabilities stay ``None`` and no ranked entry may grow a
    ``probability`` field (the schema already forbids it; this is the
    explicit runtime guard).
    """
    if report.calibrated_probabilities is not None:
        raise ValueError("similarity reports must not carry calibrated_probabilities")
    for entry in report.ranked_similarities:
        if "probability" in type(entry).model_fields:
            raise ValueError("RankedSimilarity must not expose a probability field")
        dumped = entry.model_dump()
        if "probability" in dumped:
            raise ValueError("ranked entry carries a probability payload")
