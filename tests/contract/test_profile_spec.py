"""Contract tests for ``docs/contracts.md`` and ``configs/*.json`` (task T00C).

Acceptance for T00C: interfaces and profiles specify missing values, status semantics,
caps and offline defaults. The oracle here is the frozen spec document plus the example
profiles, checked field by field against each other.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.contract

REPO_ROOT = Path(__file__).resolve().parents[2]
CONTRACTS_PATH = REPO_ROOT / "docs" / "contracts.md"
CONFIG_DIR = REPO_ROOT / "configs"

#: The nine contracts IMPLEMENTATION_PLAN.md section 3 requires to be frozen.
REQUIRED_CONTRACTS = (
    "CampaignManifest",
    "EndpointSpec",
    "TaskSpec",
    "GenerationResult",
    "GradeResult",
    "RunEvent",
    "SignatureResult",
    "IdentityReport",
    "GateEvidence",
)

#: Key names that must never carry a value in a serialized profile.
SECRET_KEY_PATTERN = re.compile(
    r"(?i)\b(api[_-]?key|secret|password|bearer|token_value|access[_-]?key)\b"
)
#: Recognizable canary strings: if any appear in a profile, it leaked a secret.
CANARY_SECRETS = (
    "sk-canary-do-not-log-0000",
    "ghp_canary_0123456789",
    "Bearer canary.token.value",
    "AKIA_CANARYEXAMPLE",
)


@pytest.fixture(scope="module")
def spec() -> dict[str, Any]:
    text = CONTRACTS_PATH.read_text(encoding="utf-8")
    match = re.search(r"^```json contracts\n(?P<body>.*?)^```$", text, re.DOTALL | re.MULTILINE)
    assert match is not None, "contracts.md must contain a fenced '```json contracts' block"
    parsed: dict[str, Any] = json.loads(match.group("body"))
    return parsed


def _profile_files() -> list[Path]:
    return sorted(path for path in CONFIG_DIR.glob("*.json"))


@pytest.fixture(scope="module")
def profiles() -> dict[str, dict[str, Any]]:
    return {path.stem: json.loads(path.read_text(encoding="utf-8")) for path in _profile_files()}


# ---------------------------------------------------------------------------
# The frozen spec itself
# ---------------------------------------------------------------------------


def test_all_nine_contracts_are_frozen(spec: dict[str, Any]) -> None:
    assert set(spec["contracts"]) == set(REQUIRED_CONTRACTS), (
        f"missing contracts: {sorted(set(REQUIRED_CONTRACTS) - set(spec['contracts']))}"
    )


def test_every_contract_declares_its_fields(spec: dict[str, Any]) -> None:
    for name, fields in spec["contracts"].items():
        assert isinstance(fields, list) and fields, f"{name} declares no fields"
        assert len(fields) == len(set(fields)), f"{name} repeats a field"


def test_status_vocabularies_match_the_plan(spec: dict[str, Any]) -> None:
    assert spec["task_states"] == ["todo", "ready", "running", "review", "done", "blocked_external"]
    assert spec["gate_states"] == ["not_run", "passed", "failed", "blocked_external"]
    assert spec["sample_status_axes"] == ["correctness", "format", "transport", "evaluator"]


def test_null_semantics_forbid_zero_substitution(spec: dict[str, Any]) -> None:
    nulls = spec["null_semantics"]
    assert nulls["absent_measurement"] == "null"
    assert nulls["absent_usage_field"] == "null"
    assert nulls["absent_price"] == "null"
    assert nulls["zero_is_never_a_substitute_for_null"] is True
    assert nulls["unknown_identity"] == "unknown"
    assert nulls["unresolved_after_crash"] == "unresolved"


def test_identity_labels_are_four_and_independent(spec: dict[str, Any]) -> None:
    assert spec["identity_labels"] == [
        "label_provider",
        "label_family",
        "label_exact_version",
        "label_tokenizer",
    ]
    assert len(set(spec["identity_labels"])) == 4


def test_retry_policy_excludes_evaluator_verdicts(spec: dict[str, Any]) -> None:
    policy = spec["retryable_status_policy"]
    assert policy["allowed"] == [408, 429, 500, 502, 503, 504]
    assert all(status >= 400 for status in policy["allowed"])
    assert "wrong answer" in policy["never_retry"]
    assert "evaluator verdict" in policy["never_retry"]


def test_offline_and_live_defaults_are_distinct(spec: dict[str, Any]) -> None:
    offline = spec["offline_defaults"]
    live = spec["live_defaults"]
    assert offline["mode"] == "offline"
    assert offline["transport"] == "fixture"
    assert offline["credential_ref"] is None
    assert offline["authorization_required"] is False
    assert offline["network_access"] == "denied"
    assert live["mode"] == "live_authorized"
    assert live["authorization_required"] is True
    assert live["transport"] != offline["transport"]


def test_only_the_current_schema_version_is_supported(spec: dict[str, Any]) -> None:
    assert spec["supported_schema_versions"] == ["1.0"]
    assert spec["schema_version"] == "1.0"


# ---------------------------------------------------------------------------
# Example profiles
# ---------------------------------------------------------------------------


def test_every_example_profile_parses_and_declares_the_required_fields(
    profiles: dict[str, dict[str, Any]], spec: dict[str, Any]
) -> None:
    assert set(profiles) >= {"offline-demo", "pilot", "full"}, (
        "the three planned profiles are missing"
    )
    for name, profile in profiles.items():
        for field in spec["required_profile_fields"]:
            assert field in profile, f"{name} is missing required field {field!r}"


def test_profiles_use_the_supported_schema_version(profiles: dict[str, dict[str, Any]]) -> None:
    for name, profile in profiles.items():
        assert profile["schema_version"] == "1.0", f"{name} declares an unsupported schema version"


def test_campaign_ids_are_unique(profiles: dict[str, dict[str, Any]]) -> None:
    ids = [profile["campaign_id"] for profile in profiles.values()]
    assert len(ids) == len(set(ids)), f"duplicate campaign_id in {ids}"


def test_every_profile_declares_positive_caps(
    profiles: dict[str, dict[str, Any]], spec: dict[str, Any]
) -> None:
    for name, profile in profiles.items():
        limits = profile["limits"]
        for field in spec["required_limit_fields"]:
            assert field in limits, f"{name}.limits is missing {field!r}"
        for field in (
            "max_requests",
            "max_concurrency",
            "max_input_tokens",
            "max_output_tokens",
            "max_total_cost_usd",
            "max_wall_seconds",
        ):
            value = limits[field]
            assert isinstance(value, (int, float)) and not isinstance(value, bool), (
                f"{name}.limits.{field} must be a number, got {value!r}"
            )
            assert value > 0, f"{name}.limits.{field} must be positive, got {value!r}"
        assert limits["require_cost_bounds"] is True, (
            f"{name} must refuse spending-capped execution when cost bounds are unknown"
        )


def test_every_profile_declares_a_retry_policy_that_excludes_verdicts(
    profiles: dict[str, dict[str, Any]], spec: dict[str, Any]
) -> None:
    allowed = set(spec["retryable_status_policy"]["allowed"])
    for name, profile in profiles.items():
        retry = profile["retry_policy"]
        for field in spec["required_retry_policy_fields"]:
            assert field in retry, f"{name}.retry_policy is missing {field!r}"
        assert retry["max_attempts"] >= 1
        assert retry["initial_backoff_seconds"] > 0
        assert retry["multiplier"] >= 1
        assert retry["max_backoff_seconds"] >= retry["initial_backoff_seconds"]
        statuses = set(retry["retryable_statuses"])
        assert statuses == allowed, f"{name} declares a non-standard retryable set {statuses}"
        assert all(status >= 400 for status in statuses), "a 2xx status is never a failure"


def test_offline_profile_matches_the_offline_defaults(
    profiles: dict[str, dict[str, Any]], spec: dict[str, Any]
) -> None:
    profile = profiles["offline-demo"]
    defaults = spec["offline_defaults"]
    assert profile["mode"] == defaults["mode"]
    assert "authorization" not in profile, "an offline profile must not declare a spending cap"
    assert profile["materialized"] is True
    assert profile["endpoints"], "the demonstration profile must have at least one endpoint"
    for endpoint in profile["endpoints"]:
        assert endpoint["transport"] == defaults["transport"]
        assert endpoint["credential_ref"] == defaults["credential_ref"]


def test_live_profiles_require_authorization_and_may_not_pre_approve_a_cap(
    profiles: dict[str, dict[str, Any]], spec: dict[str, Any]
) -> None:
    for name in ("pilot", "full"):
        profile = profiles[name]
        assert profile["mode"] == spec["live_defaults"]["mode"], f"{name} must be live_authorized"
        authorization = profile["authorization"]
        assert authorization["required"] is True
        assert authorization["spending_cap_usd"] is None, (
            f"{name} must not pre-approve a spending cap; an operator sets it"
        )


def test_offline_observation_window_is_null_not_a_placeholder(
    profiles: dict[str, dict[str, Any]],
) -> None:
    """An offline campaign observes no live endpoint, so it has no window."""
    window = profiles["offline-demo"]["observation_window"]
    assert window["started_at"] is None
    assert window["ended_at"] is None
    assert window["timezone"] == "UTC"


def test_materialization_and_item_ids_agree(profiles: dict[str, dict[str, Any]]) -> None:
    """A materialized profile must enumerate its items; an unmaterialized one must not."""
    for name, profile in profiles.items():
        for benchmark in profile["benchmarks"]:
            item_ids = benchmark["item_ids"]
            label = f"{name}/{benchmark['benchmark_id']}"
            assert len(item_ids) == len(set(item_ids)), f"{label} repeats an item id"

            # A pilot records a chosen target; a full profile records the official split size.
            # An explicit null must fall through, so this cannot use dict.get defaults.
            recorded = benchmark.get("declared_item_count")
            if recorded is None:
                recorded = benchmark.get("expected_item_count")
            if recorded is not None:
                assert recorded > 0, f"{label} records a non-positive count"
            else:
                # An unknown split size is legitimate only if the enumeration is declared.
                assert benchmark.get("split_enumeration"), (
                    f"{label} has no item count and no procedure for enumerating the full split"
                )

            if profile["materialized"]:
                assert item_ids, f"{label} is materialized but has no items"
                assert len(item_ids) == recorded, (
                    f"{label} records {recorded} items but lists {len(item_ids)}"
                )
            else:
                assert item_ids == [], (
                    f"{label} is not materialized and must not pre-declare item ids"
                )


def test_benchmark_tracks_cover_direct_and_agent(profiles: dict[str, dict[str, Any]]) -> None:
    pilot = profiles["pilot"]
    tracks = {benchmark["track"] for benchmark in pilot["benchmarks"]}
    assert tracks == {"direct", "agent"}


def test_bfcl_never_merges_native_and_prompted_modes(profiles: dict[str, dict[str, Any]]) -> None:
    for name in ("pilot", "full"):
        bfcl = next(b for b in profiles[name]["benchmarks"] if b["benchmark_id"] == "bfcl")
        assert set(bfcl["api_modes"]) == {"native_function_calling", "prompted"}
        assert bfcl["core_category"] is True, "tool use is a core category and may not be dropped"


def test_full_profile_requires_full_splits(profiles: dict[str, dict[str, Any]]) -> None:
    for benchmark in profiles["full"]["benchmarks"]:
        assert benchmark["require_full_split"] is True, (
            f"full/{benchmark['benchmark_id']} must require the entire declared split"
        )


def test_signature_probes_are_declared_separately_from_benchmarks(
    profiles: dict[str, dict[str, Any]],
) -> None:
    for name in ("pilot", "full"):
        probes = profiles[name]["signature_probes"]
        assert probes["probe_count"] == 60
        assert probes["repeats"] == 3
        assert probes["separate_from_benchmarks"] is True


def test_agent_benchmarks_are_not_part_of_the_core_index(
    profiles: dict[str, dict[str, Any]],
) -> None:
    """The core index is the six direct categories; agent tracks are reported separately."""
    for name in ("pilot", "full"):
        for benchmark in profiles[name]["benchmarks"]:
            if benchmark["track"] == "agent":
                assert benchmark["core_category"] is False, (
                    f"{name}/{benchmark['benchmark_id']} must not enter the core index"
                )


# ---------------------------------------------------------------------------
# Vacuous suites
# ---------------------------------------------------------------------------


def _collect_count(suite: str) -> int:
    env = dict(os.environ)
    env.pop("PYTEST_CURRENT_TEST", None)
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--strict-markers", "--collect-only", "-q", suite],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(REPO_ROOT),
        env=env,
        timeout=300,
    )
    return len([line for line in result.stdout.splitlines() if "::" in line])


def test_required_g00_suites_are_not_vacuous(spec: dict[str, Any]) -> None:
    """A mandatory suite that collects nothing cannot satisfy its gate."""
    for suite in spec["required_suites_per_gate"]["G00"]:
        assert _collect_count(suite) > 0, f"{suite} collects zero tests and is mandatory for G00"


def test_any_empty_suite_is_registered_as_not_yet_started(spec: dict[str, Any]) -> None:
    """An empty suite must be a declared decision, never an accident.

    `IMPLEMENTATION_PLAN.md` section 4: an empty suite cannot satisfy a mandatory
    check. Declaring the suite as not-yet-started keeps it visible and prevents a
    later gate from inheriting a silent pass.
    """

    testpaths = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["tool"][
        "pytest"
    ]["ini_options"]["testpaths"]
    not_started = spec["suites_not_yet_started"]
    for suite in testpaths:
        if _collect_count(suite) == 0:
            assert suite in not_started, (
                f"{suite} collects zero tests but is not registered in "
                f"suites_not_yet_started; either add tests or record why"
            )
            assert not_started[suite], f"{suite} is registered without a reason"


def test_every_registered_suite_reason_names_a_gate() -> None:
    text = CONTRACTS_PATH.read_text(encoding="utf-8")
    match = re.search(r"^```json contracts\n(?P<body>.*?)^```$", text, re.DOTALL | re.MULTILINE)
    assert match is not None
    parsed: dict[str, Any] = json.loads(match.group("body"))
    for suite, reason in parsed["suites_not_yet_started"].items():
        assert re.search(r"\bG\d{2}\b", reason), f"{suite} reason must name the gate that fills it"


# ---------------------------------------------------------------------------
# Secret hygiene
# ---------------------------------------------------------------------------


def _secret_bearing_keys(node: Any, path: str = "") -> list[str]:
    """Return every path whose key name looks like it holds a credential."""
    found: list[str] = []
    if isinstance(node, dict):
        for key, value in node.items():
            where = f"{path}.{key}" if path else key
            if SECRET_KEY_PATTERN.search(key):
                found.append(where)
            found.extend(_secret_bearing_keys(value, where))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            found.extend(_secret_bearing_keys(value, f"{path}[{index}]"))
    return found


def _canaries_in(text: str) -> list[str]:
    return [canary for canary in CANARY_SECRETS if canary in text]


def test_no_profile_carries_a_credential(profiles: dict[str, dict[str, Any]]) -> None:
    for name, profile in profiles.items():
        leaked = _secret_bearing_keys(profile, name)
        assert leaked == [], f"secret-bearing keys found: {leaked}"


def test_no_profile_contains_a_canary_secret() -> None:
    for path in _profile_files():
        found = _canaries_in(path.read_text(encoding="utf-8"))
        assert found == [], f"{path.name} contains canary secrets {found}"


def test_credential_detector_catches_a_planted_leak(tmp_path: Path) -> None:
    """Plant a real canary in a copy of a real profile and prove the predicates fire.

    Without this, the two checks above could be passing because their detectors are
    broken rather than because the profiles are clean.
    """
    source = CONFIG_DIR / "offline-demo.json"
    leaky = json.loads(source.read_text(encoding="utf-8"))
    leaky["endpoints"][0]["api_key"] = CANARY_SECRETS[0]
    leaky["description"] = f"token={CANARY_SECRETS[2]}"
    plant = tmp_path / "leaky.json"
    plant.write_text(json.dumps(leaky), encoding="utf-8")

    assert _secret_bearing_keys(leaky, "leaky") == ["leaky.endpoints[0].api_key"]
    assert _canaries_in(plant.read_text(encoding="utf-8")) == [CANARY_SECRETS[0], CANARY_SECRETS[2]]

    # And the detector must be quiet on the genuine profile.
    clean = json.loads(source.read_text(encoding="utf-8"))
    assert _secret_bearing_keys(clean, "clean") == []
    assert _canaries_in(source.read_text(encoding="utf-8")) == []


def test_offline_endpoints_declare_pricing_as_null_not_zero(
    profiles: dict[str, dict[str, Any]],
) -> None:
    """An unknown price is null. A zero price would silently mean 'free'."""
    for endpoint in profiles["offline-demo"]["endpoints"]:
        pricing = endpoint["pricing"]
        for field in ("input_per_mtok", "output_per_mtok", "cached_input_per_mtok"):
            assert pricing[field] is None, f"fixture endpoint pricing.{field} must be null"


def test_identity_labels_are_present_and_null_for_an_anonymous_alias(
    profiles: dict[str, dict[str, Any]], spec: dict[str, Any]
) -> None:
    """The labels are read from the frozen spec, so the spec cannot drift unnoticed."""
    labels = spec["identity_labels"]
    assert len(labels) == 4
    for endpoint in profiles["offline-demo"]["endpoints"]:
        for label in labels:
            assert label in endpoint, f"endpoint is missing the {label} label"
            assert endpoint[label] is None, (
                f"{label} must be unknown for an anonymous alias, not invented"
            )


def test_endpoint_spec_freezes_the_independent_labels(spec: dict[str, Any]) -> None:
    """The four identity labels must be part of EndpointSpec itself, not only prose."""
    endpoint_fields = spec["contracts"]["EndpointSpec"]
    for label in spec["identity_labels"]:
        assert label in endpoint_fields, (
            f"EndpointSpec must freeze {label}; otherwise G01 can implement it without it"
        )


def test_every_endpoint_matches_the_frozen_endpoint_spec(
    profiles: dict[str, dict[str, Any]], spec: dict[str, Any]
) -> None:
    required = spec["required_endpoint_fields"]
    labels = spec["identity_labels"]
    for name, profile in profiles.items():
        for index, endpoint in enumerate(profile["endpoints"]):
            for field in (*required, *labels):
                assert field in endpoint, f"{name}.endpoints[{index}] is missing {field!r}"


def test_every_profile_mode_comes_from_the_declared_vocabulary(
    profiles: dict[str, dict[str, Any]], spec: dict[str, Any]
) -> None:
    for name, profile in profiles.items():
        assert profile["mode"] in spec["campaign_modes"], (
            f"{name} uses undeclared mode {profile['mode']!r}"
        )
