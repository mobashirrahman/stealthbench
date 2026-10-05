"""Live and paid suite placeholders (gate G00).

These tests exist so the ``live`` and ``paid`` marker vocabulary is real rather than
declared-but-unused, and so that authorization has exactly one place to be checked.

None of them runs in ordinary CI: they are collected, they skip, and the skip reason
is recorded. ``.github/workflows/live.yml`` is the only workflow that executes them,
and it is inert until G14.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

AUTHORIZATION_ENV = "STEALTHBENCH_LIVE_AUTHORIZATION"
SPENDING_CAP_ENV = "STEALTHBENCH_SPENDING_CAP_USD"
CREDENTIAL_ENV = "STEALTHBENCH_ZEN_API_KEY"

#: Operator-approved ceiling for a single dispatch.
MAX_APPROVED_CAP_USD = 5000.0


def _unavailable(reason: str) -> None:
    pytest.skip(f"blocked_external: {reason}")


@pytest.mark.live
def test_live_suite_requires_explicit_authorization() -> None:
    """A live test must refuse to run without an operator-set authorization flag."""
    if os.environ.get(AUTHORIZATION_ENV) != "1":
        _unavailable(f"{AUTHORIZATION_ENV} is not set to 1")
    if not os.environ.get(CREDENTIAL_ENV):
        _unavailable(f"{CREDENTIAL_ENV} is not present in the environment")
    pytest.fail(
        "live dispatch is not implemented yet (gate G03/G04); "
        "an authorized environment must still fail loudly rather than pass"
    )


@pytest.mark.paid
def test_paid_suite_requires_a_numeric_spending_cap() -> None:
    """Money-moving work additionally requires an explicit numeric ceiling."""
    if os.environ.get(AUTHORIZATION_ENV) != "1":
        _unavailable(f"{AUTHORIZATION_ENV} is not set to 1")
    raw_cap = os.environ.get(SPENDING_CAP_ENV)
    if raw_cap is None:
        _unavailable(f"{SPENDING_CAP_ENV} is not set")
    try:
        cap = float(raw_cap)
    except ValueError:
        pytest.fail(f"{SPENDING_CAP_ENV}={raw_cap!r} is not a number")
    if not cap > 0:
        pytest.fail(f"{SPENDING_CAP_ENV}={cap} must be positive")
    if cap > MAX_APPROVED_CAP_USD:
        pytest.fail(f"requested ceiling {cap} exceeds the operator-approved maximum")
    pytest.fail("paid dispatch is not implemented yet (gate G04)")


def test_offline_demo_profile_needs_no_authorization() -> None:
    """The offline demonstration must remain runnable with zero live inputs."""
    profile = json.loads((REPO_ROOT / "configs" / "offline-demo.json").read_text(encoding="utf-8"))
    assert profile["mode"] == "offline", "the demonstration profile must not require authorization"
    assert "authorization" not in profile, "an offline profile must not carry a spending cap"
    for endpoint in profile["endpoints"]:
        assert endpoint["credential_ref"] is None
        assert endpoint["transport"] == "fixture"
