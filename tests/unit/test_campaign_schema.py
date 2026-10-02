"""Unit tests for the campaign and endpoint contracts (task T01A).

Oracles are the frozen contract in ``docs/contracts.md`` and
``IMPLEMENTATION_PLAN.md`` sections 3 and 5/G01. Where a test checks a rejection it
asserts the *reason*, so a model that fails for an unrelated reason does not pass.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from stealthbench.schemas.campaign import (
    Authorization,
    BenchmarkSpec,
    CampaignManifest,
    Capabilities,
    EndpointSpec,
    Limits,
    PricingSnapshot,
    RetryPolicy,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_FILES = ("offline-demo", "pilot", "full")

CANARY_SECRET = "sk-canary-0000000000000000000000000000"


@pytest.fixture
def offline_manifest() -> dict[str, Any]:
    """A minimal valid offline manifest as a plain dict."""
    return {
        "campaign_id": "unit-offline",
        "title": "unit fixture",
        "description": "offline manifest for unit tests",
        "mode": "offline",
        "materialized": True,
        "score_version": "stealthbench-core-v1",
        "seed": 7,
        "endpoints": [
            {
                "endpoint_id": "fixture-a",
                "alias": "fixture-a",
                "route": "fixture",
                "transport": "fixture",
                "capabilities": {
                    "streaming": False,
                    "tool_calls": False,
                    "reasoning": False,
                    "usage_reporting": False,
                    "logprobs": False,
                },
            }
        ],
        "benchmarks": [
            {
                "benchmark_id": "ifeval",
                "track": "direct",
                "category": "instruction_following",
                "declared_item_count": 2,
                "item_ids": ["a", "b"],
            }
        ],
        "generation": {"max_output_tokens": 128},
        "retry_policy": {
            "max_attempts": 3,
            "initial_backoff_seconds": 1.0,
            "multiplier": 2.0,
            "max_backoff_seconds": 30.0,
        },
        "limits": {
            "max_requests": 100,
            "max_concurrency": 2,
            "max_input_tokens": 1000,
            "max_output_tokens": 1000,
            "max_total_cost_usd": 1.0,
            "max_wall_seconds": 60.0,
        },
    }


def _build(overrides: dict[str, Any] | None = None) -> CampaignManifest:
    return CampaignManifest.model_validate(overrides or {})


# ---------------------------------------------------------------------------
# The shipped profiles are valid documents
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", CONFIG_FILES)
def test_shipped_profiles_validate(name: str) -> None:
    raw = json.loads((REPO_ROOT / "configs" / f"{name}.json").read_text(encoding="utf-8"))
    if raw["mode"] == "live_authorized":
        raw["authorization"] = {"required": True, "spending_cap_usd": None}
    manifest = CampaignManifest.model_validate(raw)
    assert manifest.schema_version == "1.0"


@pytest.mark.parametrize("name", CONFIG_FILES)
def test_only_the_offline_profile_is_dispatchable(name: str) -> None:
    """Pilot and full are valid documents that are deliberately not dispatchable."""
    raw = json.loads((REPO_ROOT / "configs" / f"{name}.json").read_text(encoding="utf-8"))
    if raw["mode"] == "live_authorized":
        raw["authorization"] = {"required": True, "spending_cap_usd": None}
    manifest = CampaignManifest.model_validate(raw)
    if name == "offline-demo":
        assert manifest.is_dispatchable, manifest.dispatch_blockers()
    else:
        assert not manifest.is_dispatchable
        joined = " ".join(manifest.dispatch_blockers())
        assert "spending cap" in joined
        assert "item selection" in joined


@pytest.mark.parametrize("name", CONFIG_FILES)
def test_no_shipped_profile_plans_more_requests_than_its_cap(name: str) -> None:
    raw = json.loads((REPO_ROOT / "configs" / f"{name}.json").read_text(encoding="utf-8"))
    if raw["mode"] == "live_authorized":
        raw["authorization"] = {"required": True, "spending_cap_usd": None}
    manifest = CampaignManifest.model_validate(raw)
    assert manifest.known_planned_requests <= manifest.limits.max_requests
    if manifest.total_planned_requests is not None:
        assert manifest.total_planned_requests <= manifest.limits.max_requests


# ---------------------------------------------------------------------------
# Versioning
# ---------------------------------------------------------------------------


def test_unknown_schema_version_is_rejected(offline_manifest: dict[str, Any]) -> None:
    offline_manifest["schema_version"] = "2.0"
    with pytest.raises(ValidationError, match="schema_version"):
        _build(offline_manifest)


def test_schema_version_defaults_to_the_supported_version(
    offline_manifest: dict[str, Any],
) -> None:
    assert _build(offline_manifest).schema_version == "1.0"


@pytest.mark.parametrize("field", ["mode", "transport", "track", "budget_scope", "timezone"])
def test_unknown_enum_members_are_rejected(offline_manifest: dict[str, Any], field: str) -> None:
    if field == "mode":
        offline_manifest["mode"] = "turbo"
    elif field == "budget_scope":
        offline_manifest["budget_scope"] = "unlimited"
    elif field == "transport":
        offline_manifest["endpoints"][0]["transport"] = "carrier-pigeon"
    elif field == "track":
        offline_manifest["benchmarks"][0]["track"] = "psychic"
    else:
        offline_manifest["observation_window"] = {"timezone": "CET"}
    with pytest.raises(ValidationError):
        _build(offline_manifest)


def test_unknown_fields_are_rejected_not_ignored(offline_manifest: dict[str, Any]) -> None:
    """A typo in a cap must fail loudly rather than silently use a default."""
    offline_manifest["limits"]["max_reuqests"] = 5
    with pytest.raises(ValidationError, match="max_reuqests"):
        _build(offline_manifest)


def test_manifest_is_immutable(offline_manifest: dict[str, Any]) -> None:
    manifest = _build(offline_manifest)
    with pytest.raises(ValidationError):
        manifest.campaign_id = "mutated"  # type: ignore[misc]


def test_manifest_cannot_drift_after_validation(offline_manifest: dict[str, Any]) -> None:
    """A hashed manifest must not be mutable through a nested container.

    `frozen=True` blocks attribute assignment only. A plain dict field would let the
    manifest change under a hash that was already recorded.
    """
    from stealthbench.schemas.manifest import manifest_hash

    offline_manifest["benchmarks"][0]["scoring_protocol"] = {"report_strict": True}
    manifest = _build(offline_manifest)
    before = manifest_hash(manifest)

    spec = manifest.benchmarks[0]
    assert isinstance(spec.scoring_protocol, tuple), "protocol flags must be stored immutably"
    with pytest.raises((AttributeError, TypeError)):
        spec.scoring_protocol["report_strict"] = False  # type: ignore[index]
    assert manifest_hash(manifest) == before


def test_scoring_protocol_round_trips_through_a_mapping(offline_manifest: dict[str, Any]) -> None:
    offline_manifest["benchmarks"][0]["scoring_protocol"] = {
        "report_loose": True,
        "report_strict": True,
    }
    manifest = _build(offline_manifest)
    serialized = manifest.to_json_dict()
    assert serialized["benchmarks"][0]["scoring_protocol"] == {
        "report_loose": True,
        "report_strict": True,
    }
    assert manifest.benchmarks[0].protocol_flag("report_strict") is True
    assert manifest.benchmarks[0].protocol_flag("nope") is None


def test_scoring_protocol_rejects_non_boolean_flags(offline_manifest: dict[str, Any]) -> None:
    offline_manifest["benchmarks"][0]["scoring_protocol"] = {"report_strict": "yes"}
    with pytest.raises(ValidationError, match="must be a boolean"):
        _build(offline_manifest)


def test_unknown_item_count_stays_unknown_rather_than_zero() -> None:
    """Defect: an enumeration-only benchmark planned 0 requests, disabling the cap check."""
    spec = BenchmarkSpec(
        benchmark_id="bfcl",
        track="direct",
        category="tool_use",
        split_enumeration="every scored category file at the pinned commit",
    )
    assert spec.planned_item_count is None, "an unknown count must not become 0"


def test_total_planned_requests_is_none_when_any_size_is_unknown(
    offline_manifest: dict[str, Any],
) -> None:
    offline_manifest["materialized"] = False
    offline_manifest["benchmarks"][0]["item_ids"] = []
    offline_manifest["benchmarks"][0].pop("declared_item_count")
    offline_manifest["benchmarks"][0]["split_enumeration"] = "full official split"
    manifest = _build(offline_manifest)
    assert manifest.total_planned_requests is None
    assert manifest.known_planned_requests == 0
    assert manifest.benchmarks_with_unknown_size == ("ifeval",)
    assert any("unknown" in blocker for blocker in manifest.dispatch_blockers())


def test_an_unknown_size_is_always_covered_by_a_dispatch_blocker(
    offline_manifest: dict[str, Any],
) -> None:
    """An unknown size is never silently treated as zero.

    Such a campaign cannot be dispatchable, and the blocker must name the cause.
    """
    offline_manifest["materialized"] = False
    offline_manifest["benchmarks"][0]["item_ids"] = []
    offline_manifest["benchmarks"][0].pop("declared_item_count")
    offline_manifest["benchmarks"][0]["split_enumeration"] = "full official split"
    manifest = _build(offline_manifest)
    assert manifest.total_planned_requests is None
    assert not manifest.is_dispatchable
    assert any("unknown" in blocker for blocker in manifest.dispatch_blockers())


# ---------------------------------------------------------------------------
# Credentials must never appear in a manifest
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_ref",
    [
        CANARY_SECRET,
        "Bearer canary.token.value",
        "ghp_canary0123456789",
        "-----BEGIN PRIVATE KEY-----",
        "has spaces",
    ],
)
def test_credential_value_in_credential_ref_is_rejected(bad_ref: str) -> None:
    with pytest.raises(ValidationError, match="credential_ref"):
        EndpointSpec(
            endpoint_id="e",
            alias="a",
            route="zen",
            transport="zen",
            capabilities=Capabilities(
                streaming=False,
                tool_calls=False,
                reasoning=False,
                usage_reporting=False,
                logprobs=False,
            ),
            credential_ref=bad_ref,
        )


def test_credential_ref_accepts_an_environment_variable_name() -> None:
    endpoint = EndpointSpec(
        endpoint_id="e",
        alias="a",
        route="zen",
        transport="zen",
        capabilities=Capabilities(
            streaming=False,
            tool_calls=False,
            reasoning=False,
            usage_reporting=False,
            logprobs=False,
        ),
        credential_ref="STEALTHBENCH_ZEN_API_KEY",
    )
    assert endpoint.credential_ref == "STEALTHBENCH_ZEN_API_KEY"


def test_fixture_endpoint_may_not_name_a_credential() -> None:
    with pytest.raises(ValidationError, match="fixture transport"):
        EndpointSpec(
            endpoint_id="e",
            alias="a",
            route="fixture",
            transport="fixture",
            capabilities=Capabilities(
                streaming=False,
                tool_calls=False,
                reasoning=False,
                usage_reporting=False,
                logprobs=False,
            ),
            credential_ref="STEALTHBENCH_ZEN_API_KEY",
        )


def test_manifest_with_a_smuggled_credential_field_is_rejected(
    offline_manifest: dict[str, Any],
) -> None:
    offline_manifest["endpoints"][0]["api_key"] = CANARY_SECRET
    with pytest.raises(ValidationError, match="api_key"):
        _build(offline_manifest)


def test_serialized_manifest_contains_no_canary(offline_manifest: dict[str, Any]) -> None:
    """A valid manifest cannot carry a credential in any serialized position."""
    manifest = _build(offline_manifest)
    serialized = json.dumps(manifest.to_json_dict())
    assert CANARY_SECRET not in serialized
    assert "sk-" not in serialized

    # A free-text field is stored and hashed, so it must reject a pasted credential
    # rather than carrying it into every artifact.
    for field in ("title", "description", "comparability_rule"):
        offline_manifest[field] = f"run with token {CANARY_SECRET}"
        with pytest.raises(ValidationError, match="credential"):
            _build(offline_manifest)


def test_credential_ref_allowlist_rejects_unlisted_secret_shapes() -> None:
    """A blocklist alone would accept a JWT or a bare hex key as a "name"."""
    for bad in (
        "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.dBjftJeZ4CVP",
        "1234567890abcdef1234567890abcdef12345678",
        "AIzaSyBqKjL9pXf2QwErTyUiOpAsDfGhJkLmNoP",
        "lowercase_name",
        "Name-With-Dashes",
    ):
        with pytest.raises(ValidationError, match="credential_ref"):
            EndpointSpec(
                endpoint_id="e",
                alias="a",
                route="zen",
                transport="zen",
                capabilities=Capabilities(
                    streaming=False,
                    tool_calls=False,
                    reasoning=False,
                    usage_reporting=False,
                    logprobs=False,
                ),
                credential_ref=bad,
            )


# ---------------------------------------------------------------------------
# Duplicate identifiers
# ---------------------------------------------------------------------------


def test_duplicate_endpoint_ids_are_rejected(offline_manifest: dict[str, Any]) -> None:
    endpoint = copy.deepcopy(offline_manifest["endpoints"][0])
    offline_manifest["endpoints"].append(endpoint)
    with pytest.raises(ValidationError, match="duplicate endpoint ids"):
        _build(offline_manifest)


def test_duplicate_benchmark_ids_are_rejected(offline_manifest: dict[str, Any]) -> None:
    benchmark = copy.deepcopy(offline_manifest["benchmarks"][0])
    offline_manifest["benchmarks"].append(benchmark)
    with pytest.raises(ValidationError, match="duplicate benchmark ids"):
        _build(offline_manifest)


def test_duplicate_item_ids_are_rejected() -> None:
    with pytest.raises(ValidationError, match="duplicate item ids"):
        BenchmarkSpec(
            benchmark_id="ifeval",
            track="direct",
            category="instruction_following",
            declared_item_count=3,
            item_ids=("a", "b", "a"),
        )


def test_distinct_item_ids_are_accepted() -> None:
    spec = BenchmarkSpec(
        benchmark_id="ifeval",
        track="direct",
        category="instruction_following",
        declared_item_count=3,
        item_ids=("a", "b", "c"),
    )
    assert spec.item_ids == ("a", "b", "c")


# ---------------------------------------------------------------------------
# Non-finite and invalid numbers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_limits_are_rejected(offline_manifest: dict[str, Any], bad: float) -> None:
    offline_manifest["limits"]["max_wall_seconds"] = bad
    with pytest.raises(ValidationError):
        _build(offline_manifest)


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_non_finite_prices_are_rejected(bad: float) -> None:
    with pytest.raises(ValidationError):
        PricingSnapshot(input_per_mtok=bad)


def test_non_finite_generation_settings_are_rejected(
    offline_manifest: dict[str, Any],
) -> None:
    offline_manifest["generation"]["temperature"] = float("inf")
    with pytest.raises(ValidationError):
        _build(offline_manifest)


def test_negative_price_is_rejected() -> None:
    with pytest.raises(ValidationError, match="must not be negative"):
        PricingSnapshot(output_per_mtok=-1.0)


def test_unknown_price_is_null_not_zero() -> None:
    pricing = PricingSnapshot()
    assert pricing.input_per_mtok is None
    assert pricing.output_per_mtok is None
    assert pricing.model_dump()["input_per_mtok"] is None


def test_zero_price_is_distinguishable_from_unknown() -> None:
    """A published price of zero is a different fact from no price at all."""
    free = PricingSnapshot(input_per_mtok=0.0)
    unknown = PricingSnapshot()
    assert free.input_per_mtok == 0.0
    assert free.input_per_mtok is not None
    assert unknown.input_per_mtok is None
    assert free != unknown


@pytest.mark.parametrize("bad_temperature", [-0.1, 2.1])
def test_temperature_outside_range_is_rejected(
    offline_manifest: dict[str, Any], bad_temperature: float
) -> None:
    offline_manifest["generation"]["temperature"] = bad_temperature
    with pytest.raises(ValidationError, match="temperature"):
        _build(offline_manifest)


def test_top_p_above_one_is_rejected(offline_manifest: dict[str, Any]) -> None:
    offline_manifest["generation"]["top_p"] = 1.5
    with pytest.raises(ValidationError, match="top_p"):
        _build(offline_manifest)


# ---------------------------------------------------------------------------
# Conflicting limits
# ---------------------------------------------------------------------------


def test_concurrency_above_request_cap_is_rejected() -> None:
    with pytest.raises(ValidationError, match="max_concurrency"):
        Limits(
            max_requests=4,
            max_concurrency=8,
            max_input_tokens=10,
            max_output_tokens=10,
            max_total_cost_usd=1.0,
            max_wall_seconds=10.0,
        )


def test_cost_bounds_required_without_a_cap_is_rejected() -> None:
    """An absent cap is not permission to spend an unknown amount."""
    with pytest.raises(ValidationError, match="max_total_cost_usd"):
        Limits(
            max_requests=4,
            max_concurrency=1,
            max_input_tokens=10,
            max_output_tokens=10,
            max_total_cost_usd=None,
            max_wall_seconds=10.0,
            require_cost_bounds=True,
        )


def test_cost_bounds_may_be_relaxed_explicitly() -> None:
    limits = Limits(
        max_requests=4,
        max_concurrency=1,
        max_input_tokens=10,
        max_output_tokens=10,
        max_total_cost_usd=None,
        max_wall_seconds=10.0,
        require_cost_bounds=False,
    )
    assert limits.max_total_cost_usd is None


def test_output_token_cap_below_requested_output_is_rejected(
    offline_manifest: dict[str, Any],
) -> None:
    offline_manifest["limits"]["max_output_tokens"] = 8
    offline_manifest["generation"]["max_output_tokens"] = 128
    with pytest.raises(ValidationError, match="max_output_tokens"):
        _build(offline_manifest)


def test_request_cap_below_planned_work_is_rejected(offline_manifest: dict[str, Any]) -> None:
    """Refuse a campaign that would certainly exceed its own request cap."""
    offline_manifest["benchmarks"][0]["declared_item_count"] = 500
    offline_manifest["benchmarks"][0]["item_ids"] = [f"i{n}" for n in range(500)]
    offline_manifest["limits"]["max_requests"] = 100
    with pytest.raises(ValidationError, match="max_requests"):
        _build(offline_manifest)


def test_spending_cap_without_an_approver_is_rejected() -> None:
    with pytest.raises(ValidationError, match="authorized_by"):
        Authorization(required=True, spending_cap_usd=10.0)


def test_spending_cap_with_an_approver_is_accepted() -> None:
    authorization = Authorization(required=True, spending_cap_usd=10.0, authorized_by="operator")
    assert authorization.spending_cap_usd == 10.0


def test_missingness_threshold_must_be_a_fraction() -> None:
    with pytest.raises(ValidationError, match="missingness_threshold"):
        Limits(
            max_requests=4,
            max_concurrency=1,
            max_input_tokens=10,
            max_output_tokens=10,
            max_total_cost_usd=1.0,
            max_wall_seconds=10.0,
            missingness_threshold=5.0,
        )


# ---------------------------------------------------------------------------
# Retry policy
# ---------------------------------------------------------------------------


def test_success_status_cannot_be_retryable() -> None:
    with pytest.raises(ValidationError, match="not an error status"):
        RetryPolicy(
            max_attempts=3,
            initial_backoff_seconds=1.0,
            multiplier=2.0,
            max_backoff_seconds=10.0,
            retryable_statuses=(200,),
        )


def test_empty_retryable_set_is_rejected() -> None:
    with pytest.raises(ValidationError, match="must not be empty"):
        RetryPolicy(
            max_attempts=3,
            initial_backoff_seconds=1.0,
            multiplier=2.0,
            max_backoff_seconds=10.0,
            retryable_statuses=(),
        )


def test_backoff_ceiling_below_the_floor_is_rejected() -> None:
    with pytest.raises(ValidationError, match="max_backoff_seconds"):
        RetryPolicy(
            max_attempts=3,
            initial_backoff_seconds=10.0,
            multiplier=2.0,
            max_backoff_seconds=1.0,
        )


def test_default_retryable_set_matches_the_frozen_policy() -> None:
    policy = RetryPolicy(
        max_attempts=3, initial_backoff_seconds=1.0, multiplier=2.0, max_backoff_seconds=10.0
    )
    assert policy.retryable_statuses == (408, 429, 500, 502, 503, 504)


# ---------------------------------------------------------------------------
# Mode consistency
# ---------------------------------------------------------------------------


def test_offline_campaign_may_not_declare_authorization(
    offline_manifest: dict[str, Any],
) -> None:
    offline_manifest["authorization"] = {
        "required": True,
        "spending_cap_usd": 5.0,
        "authorized_by": "operator",
    }
    with pytest.raises(ValidationError, match="offline campaign must not declare"):
        _build(offline_manifest)


def test_offline_campaign_may_not_use_a_live_transport(
    offline_manifest: dict[str, Any],
) -> None:
    offline_manifest["endpoints"][0]["transport"] = "zen"
    with pytest.raises(ValidationError, match="non-fixture transport"):
        _build(offline_manifest)


def test_live_campaign_requires_an_authorization_block(
    offline_manifest: dict[str, Any],
) -> None:
    offline_manifest["mode"] = "live_authorized"
    with pytest.raises(ValidationError, match="authorization block"):
        _build(offline_manifest)


def test_live_campaign_without_a_cap_is_valid_but_not_dispatchable(
    offline_manifest: dict[str, Any],
) -> None:
    """Validity and authorization are different questions."""
    offline_manifest["mode"] = "live_authorized"
    offline_manifest["authorization"] = {"required": True, "spending_cap_usd": None}
    offline_manifest["endpoints"][0].update(
        {"transport": "zen", "credential_ref": "STEALTHBENCH_ZEN_API_KEY"}
    )
    manifest = _build(offline_manifest)
    assert manifest.mode == "live_authorized"
    assert not manifest.is_dispatchable
    assert any("spending cap" in blocker for blocker in manifest.dispatch_blockers())


def test_live_campaign_with_a_cap_and_approver_is_dispatchable(
    offline_manifest: dict[str, Any],
) -> None:
    offline_manifest["mode"] = "live_authorized"
    offline_manifest["authorization"] = {
        "required": True,
        "spending_cap_usd": 5.0,
        "authorized_by": "operator",
    }
    offline_manifest["endpoints"][0].update(
        {"transport": "zen", "credential_ref": "STEALTHBENCH_ZEN_API_KEY"}
    )
    assert _build(offline_manifest).is_dispatchable


# ---------------------------------------------------------------------------
# Materialization
# ---------------------------------------------------------------------------


def test_materialized_campaign_needs_item_ids(offline_manifest: dict[str, Any]) -> None:
    offline_manifest["benchmarks"][0]["item_ids"] = []
    with pytest.raises(ValidationError, match="materialized"):
        _build(offline_manifest)


def test_unmaterialized_campaign_may_not_predeclare_items(
    offline_manifest: dict[str, Any],
) -> None:
    offline_manifest["materialized"] = False
    with pytest.raises(ValidationError, match="not materialized"):
        _build(offline_manifest)


def test_recorded_count_must_match_the_listed_selection(
    offline_manifest: dict[str, Any],
) -> None:
    offline_manifest["benchmarks"][0]["declared_item_count"] = 5
    with pytest.raises(ValidationError, match="records 5 items but lists 2"):
        _build(offline_manifest)


def test_benchmark_needs_a_count_or_an_enumeration_procedure() -> None:
    with pytest.raises(ValidationError, match="split_enumeration"):
        BenchmarkSpec(benchmark_id="ifeval", track="direct", category="instruction_following")


# ---------------------------------------------------------------------------
# Observation window
# ---------------------------------------------------------------------------


def test_naive_timestamp_is_rejected() -> None:
    from stealthbench.schemas.campaign import ObservationWindow

    with pytest.raises(ValidationError, match="timezone-aware"):
        ObservationWindow(started_at="2026-10-02T00:00:00")


def test_non_utc_timestamp_is_rejected() -> None:
    from datetime import datetime, timedelta, timezone

    from stealthbench.schemas.campaign import ObservationWindow

    plus_two = timezone(timedelta(hours=2))
    with pytest.raises(ValidationError, match="must be UTC"):
        ObservationWindow(started_at=datetime(2026, 10, 2, tzinfo=plus_two))


def test_reversed_observation_window_is_rejected() -> None:
    from datetime import UTC, datetime

    from stealthbench.schemas.campaign import ObservationWindow

    with pytest.raises(ValidationError, match="must not precede"):
        ObservationWindow(
            started_at=datetime(2026, 10, 2, 12, 0, tzinfo=UTC),
            ended_at=datetime(2026, 10, 1, 12, 0, tzinfo=UTC),
        )


def test_null_observation_window_is_accepted_offline(offline_manifest: dict[str, Any]) -> None:
    manifest = _build(offline_manifest)
    assert manifest.observation_window.started_at is None
    assert manifest.observation_window.ended_at is None


# ---------------------------------------------------------------------------
# Serialization round trip
# ---------------------------------------------------------------------------


def test_round_trip_is_stable(offline_manifest: dict[str, Any]) -> None:
    first = _build(offline_manifest)
    second = CampaignManifest.model_validate(first.to_json_dict())
    assert first == second
    assert first.to_json_dict() == second.to_json_dict()


def test_round_trip_preserves_a_null_and_a_zero_distinctly(
    offline_manifest: dict[str, Any],
) -> None:
    """A null price must not survive a round trip as 0."""
    offline_manifest["endpoints"][0]["pricing"] = {
        "input_per_mtok": None,
        "output_per_mtok": 0.0,
    }
    manifest = _build(offline_manifest)
    round_tripped = CampaignManifest.model_validate(manifest.to_json_dict())
    assert round_tripped.endpoints[0].pricing.input_per_mtok is None
    assert round_tripped.endpoints[0].pricing.output_per_mtok == 0.0
