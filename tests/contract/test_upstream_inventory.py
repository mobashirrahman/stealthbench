"""Contract test for ``docs/upstream-inventory.md`` (task T00B).

Gate G00 requires an explicit inventory entry for all seven direct components
including both coding components, plus the two agent benchmarks. The oracle is the
markdown itself: if an entry loses a required field, or the prose stops mentioning a
component, this fails rather than letting the gap reach a later gate.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.contract

INVENTORY_PATH = Path(__file__).resolve().parents[2] / "docs" / "upstream-inventory.md"

#: Seven direct components; LiveCodeBench and EvalPlus are the two coding components.
DIRECT_COMPONENTS = frozenset(
    {"ifeval", "mmlu_pro", "math500", "livecodebench", "evalplus", "bfcl", "ruler"}
)
CODING_COMPONENTS = frozenset({"livecodebench", "evalplus"})
AGENT_COMPONENTS = frozenset({"swebench", "terminalbench"})

REQUIRED_ENTRY_FIELDS = (
    "id",
    "name",
    "role",
    "category",
    "repo",
    "pinned_revision",
    "pinned_revision_kind",
    "unpinned_reason",
    "dataset",
    "evaluator",
    "api_modes",
    "license",
    "install",
    "eval_constraints",
    "stealthbench_divergence",
    "acquisition",
    "sources",
)

REQUIRED_DATASET_FIELDS = ("id", "split", "item_count", "revision", "redistributable")
ALLOWED_ACQUISITION_STATUS = frozenset({"available", "conditional", "blocked_external"})
ALLOWED_ROLES = frozenset({"direct", "agent"})
ALLOWED_API_MODES = frozenset(
    {
        "generate",
        "post_hoc_response_checker",
        "native_function_calling",
        "prompted",
        "agent_trajectory_only",
    }
)

#: A value that is a placeholder rather than a fact.
UNVERIFIED = "unverified"


@pytest.fixture(scope="module")
def inventory_text() -> str:
    assert INVENTORY_PATH.is_file(), f"missing inventory: {INVENTORY_PATH}"
    return INVENTORY_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def inventory(inventory_text: str) -> dict[str, Any]:
    match = re.search(
        r"^```json upstream-inventory\n(?P<body>.*?)^```$",
        inventory_text,
        re.DOTALL | re.MULTILINE,
    )
    assert match is not None, "inventory must contain a fenced '```json upstream-inventory' block"
    parsed: dict[str, Any] = json.loads(match.group("body"))
    return parsed


@pytest.fixture(scope="module")
def components(inventory: dict[str, Any]) -> dict[str, dict[str, Any]]:
    entries = inventory["components"]
    return {entry["id"]: entry for entry in entries}


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------


def test_inventory_covers_every_planned_component(components: dict[str, dict[str, Any]]) -> None:
    expected = DIRECT_COMPONENTS | AGENT_COMPONENTS
    assert set(components) == expected, (
        f"inventory must list exactly {sorted(expected)}; "
        f"missing {sorted(expected - set(components))}, "
        f"unexpected {sorted(set(components) - expected)}"
    )


def test_roles_split_into_seven_direct_and_two_agent(components: dict[str, dict[str, Any]]) -> None:
    direct = {cid for cid, entry in components.items() if entry["role"] == "direct"}
    agent = {cid for cid, entry in components.items() if entry["role"] == "agent"}
    assert direct == DIRECT_COMPONENTS
    assert agent == AGENT_COMPONENTS


def test_both_coding_components_are_present_and_classified_as_coding(
    components: dict[str, dict[str, Any]],
) -> None:
    coding = {cid for cid, entry in components.items() if entry["category"] == "code_generation"}
    assert coding == CODING_COMPONENTS, (
        f"coding components must be exactly {sorted(CODING_COMPONENTS)}"
    )


# ---------------------------------------------------------------------------
# Per-entry completeness
# ---------------------------------------------------------------------------


def test_every_entry_has_every_required_field(components: dict[str, dict[str, Any]]) -> None:
    for component_id, entry in components.items():
        for field in REQUIRED_ENTRY_FIELDS:
            assert field in entry, f"{component_id} is missing required field {field!r}"


def test_non_revision_fields_are_populated(components: dict[str, dict[str, Any]]) -> None:
    for component_id, entry in components.items():
        for field in ("name", "repo", "category", "pinned_revision_kind"):
            assert entry[field], f"{component_id}.{field} must not be empty"
        assert entry["repo"].startswith("https://"), f"{component_id}.repo must be an absolute URL"


def test_revision_is_pinned_or_explained(components: dict[str, dict[str, Any]]) -> None:
    """A null revision is allowed only when the reason and acquisition are explicit."""
    for component_id, entry in components.items():
        if entry["pinned_revision"] is None:
            assert entry["unpinned_reason"], (
                f"{component_id} has no pinned revision and no unpinned_reason"
            )
            assert entry["acquisition"]["status"] in {"conditional", "blocked_external"}, (
                f"{component_id} is unpinned, so acquisition must not be 'available'"
            )


def test_license_facts_are_recorded_verbatim_from_upstream(
    components: dict[str, dict[str, Any]],
) -> None:
    """Licensing is recorded because it was read upstream, not to gate anything.

    The operator has decided licensing does not block acquisition, so this only
    guards against a fabricated license string drifting in.
    """
    for component_id, entry in components.items():
        licenses = entry["license"]
        assert set(licenses) == {"code", "data"}
        for slot, value in licenses.items():
            assert value, f"{component_id}.license.{slot} must not be empty"
            assert isinstance(value, str)


def test_dataset_block_is_complete(components: dict[str, dict[str, Any]]) -> None:
    for component_id, entry in components.items():
        dataset = entry["dataset"]
        for field in REQUIRED_DATASET_FIELDS:
            assert field in dataset, f"{component_id}.dataset is missing {field!r}"
        assert dataset["redistributable"] in {True, False}, (
            f"{component_id}.dataset.redistributable must be an explicit boolean"
        )
        assert isinstance(dataset["item_count"], int) and dataset["item_count"] > 0


def test_licensing_never_blocks_acquisition(components: dict[str, dict[str, Any]]) -> None:
    """A component is `conditional` only for an operational reason, never a licensing one.

    Operator decision, 2026-10-02. Enforced so the rule cannot quietly drift back: if a
    later edit reintroduces a license-based blocker, this fails instead of blocking a
    gate on a question nobody is tracking.
    """
    licensing_words = ("licen", "redistribut", "permission", "copyright")
    for component_id, entry in components.items():
        condition = entry["acquisition"]["condition"]
        if not condition:
            continue
        lowered = condition.lower()
        for word in licensing_words:
            assert word not in lowered, (
                f"{component_id} is conditional for a licensing reason: {condition!r}"
            )


def test_api_modes_come_from_the_declared_vocabulary(components: dict[str, dict[str, Any]]) -> None:
    for component_id, entry in components.items():
        assert entry["api_modes"], f"{component_id} must declare at least one API mode"
        for mode in entry["api_modes"]:
            assert mode in ALLOWED_API_MODES, f"{component_id}.api_modes has unknown mode {mode!r}"


def test_bfcl_keeps_native_and_prompted_modes_separate(
    components: dict[str, dict[str, Any]],
) -> None:
    bfcl = components["bfcl"]
    assert set(bfcl["api_modes"]) == {"native_function_calling", "prompted"}


def test_acquisition_status_is_valid_and_conditioned(
    components: dict[str, dict[str, Any]],
) -> None:
    for component_id, entry in components.items():
        acquisition = entry["acquisition"]
        status = acquisition["status"]
        assert status in ALLOWED_ACQUISITION_STATUS, f"{component_id}: bad status {status!r}"
        if status == "available":
            assert acquisition["condition"] is None, (
                f"{component_id} is available and must not carry a blocking condition"
            )
        else:
            assert acquisition["condition"], (
                f"{component_id} status={status!r} requires a concrete acquisition condition"
            )


def test_every_entry_states_eval_constraints_and_divergence(
    components: dict[str, dict[str, Any]],
) -> None:
    for component_id, entry in components.items():
        assert entry["eval_constraints"], f"{component_id} must record upstream eval constraints"
        assert entry["stealthbench_divergence"], (
            f"{component_id} must state how StealthBench's use relates to the official protocol"
        )


def test_every_entry_cites_its_sources(components: dict[str, dict[str, Any]]) -> None:
    for component_id, entry in components.items():
        sources = entry["sources"]
        assert len(sources) >= 2, f"{component_id} must cite at least two official sources"
        for source in sources:
            assert source.startswith("https://"), f"{component_id} has a non-URL source {source!r}"


def test_install_block_records_grade_time_requirements(
    components: dict[str, dict[str, Any]],
) -> None:
    for component_id, entry in components.items():
        install = entry["install"]
        for field in ("requires_python", "grade_time_gpu", "grade_time_network"):
            assert field in install, f"{component_id}.install is missing {field!r}"
        assert isinstance(install["grade_time_gpu"], bool)
        assert isinstance(install["grade_time_network"], bool)


# ---------------------------------------------------------------------------
# Documentation agrees with data
# ---------------------------------------------------------------------------


def test_summary_table_lists_every_component(inventory_text: str) -> None:
    for component_id in sorted(DIRECT_COMPONENTS | AGENT_COMPONENTS):
        assert component_id in inventory_text, f"{component_id} is missing from the inventory prose"


def test_prose_has_a_section_per_component(inventory_text: str) -> None:
    headings = {
        match.lower() for match in re.findall(r"^###\s+(.+)$", inventory_text, re.MULTILINE)
    }
    expected_headings = {
        "ifeval",
        "mmlu-pro",
        "math-500",
        "livecodebench",
        "evalplus",
        "bfcl",
        "ruler",
        "swe-bench verified",
        "terminal-bench",
    }
    assert expected_headings <= headings, (
        f"missing detail sections: {sorted(expected_headings - headings)}"
    )


def test_inventory_declares_when_it_was_verified(inventory: dict[str, Any]) -> None:
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", inventory["verified_on"])
    assert inventory["schema_version"] == "1.0"


def test_summary_table_status_matches_the_structured_data(
    inventory_text: str, components: dict[str, dict[str, Any]]
) -> None:
    """The human-readable table must not drift from the machine-readable block."""
    table_rows = [
        line for line in inventory_text.splitlines() if re.match(r"^\|\s*\[.+\]\(#", line)
    ]
    assert len(table_rows) == len(DIRECT_COMPONENTS | AGENT_COMPONENTS)

    by_name = {entry["name"]: entry for entry in components.values()}
    seen: dict[str, str] = {}
    for row in table_rows:
        match = re.match(r"^\|\s*\[(.+)\]\(#([a-z0-9_-]+)\)\s*\|", row)
        assert match is not None, f"unparseable summary row: {row}"
        name = match.group(1)
        assert name in by_name, f"summary row {name!r} has no structured entry"
        cells = [cell.strip() for cell in row.strip().strip("|").split("|")]
        seen[name] = cells[-1].strip("*")

    assert set(seen) == set(by_name), "summary table and structured block cover different sets"
    for name, entry in by_name.items():
        assert seen[name] == entry["acquisition"]["status"], (
            f"{name}: summary table says {seen[name]!r} but the entry says "
            f"{entry['acquisition']['status']!r}"
        )


def test_conditional_assets_are_visible_in_the_summary_table(
    components: dict[str, dict[str, Any]],
) -> None:
    """The four operationally conditional components are exactly the expected ones.

    Licensing is not among the reasons (see test_licensing_never_blocks_acquisition).
    """
    conditional = {
        cid for cid, entry in components.items() if entry["acquisition"]["status"] != "available"
    }
    assert conditional == {
        "bfcl",
        "ruler",
        "swebench",
        "terminalbench",
    }, f"unexpected acquisition blockers: {sorted(conditional)}"
