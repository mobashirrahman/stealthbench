"""Toolchain and offline-execution invariants frozen by gate G00.

Oracle here is the *plan*, not the implementation. Each test asserts a rule stated in
``IMPLEMENTATION_PLAN.md`` section 6 and observes it either through pytest's own
behaviour or through a generated test run.
"""

from __future__ import annotations

import argparse
import os
import re
import socket
import subprocess
import sys
import tomllib
from collections.abc import Iterator
from pathlib import Path

import pytest

from stealthbench.cli import build_parser
from tests.conftest import REPO_ROOT, NetworkAccessDenied

pytestmark = pytest.mark.contract

#: Marker vocabulary declared by IMPLEMENTATION_PLAN.md section 6.
REQUIRED_MARKERS = frozenset(
    {"unit", "contract", "integration", "sandbox", "replay", "slow", "live", "paid"}
)


def _pyproject() -> dict:
    return tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Offline tests cannot reach external network
# ---------------------------------------------------------------------------


def test_external_dns_lookup_is_denied() -> None:
    with pytest.raises(NetworkAccessDenied, match="DNS lookup"):
        socket.getaddrinfo("zen.example.invalid", 443)


def test_external_tcp_connection_is_denied() -> None:
    with pytest.raises(NetworkAccessDenied, match="connection"):
        socket.create_connection(("zen.example.invalid", 443), timeout=1)


def test_socket_connect_is_denied_even_when_obtained_directly() -> None:
    with (
        socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock,
        pytest.raises(NetworkAccessDenied),
    ):
        sock.connect(("zen.example.invalid", 443))


def test_external_udp_sendto_is_denied() -> None:
    """Blocking only TCP would leave sendto as a one-line egress path."""
    with (
        socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock,
        pytest.raises(NetworkAccessDenied, match="datagram"),
    ):
        sock.sendto(b"query", ("8.8.8.8", 53))


def test_external_udp_connect_is_denied() -> None:
    with (
        socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock,
        pytest.raises(NetworkAccessDenied, match="connection"),
    ):
        sock.connect(("8.8.8.8", 53))


def test_loopback_udp_is_permitted() -> None:
    """Loopback datagrams must survive so fixture servers can use UDP."""
    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        receiver.bind(("127.0.0.1", 0))
        port = receiver.getsockname()[1]
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender:
            sender.sendto(b"ping", ("127.0.0.1", port))
            receiver.settimeout(5)
            assert receiver.recvfrom(64)[0] == b"ping"
    finally:
        receiver.close()


def test_offline_environment_has_no_credential() -> None:
    """Proves the autouse scrubber ran: no credential-shaped variable may reach here.

    The value arrives from the *parent* process, so this can only pass if the
    autouse fixture removed it.
    """
    leaked = _remaining_credential_env()
    assert leaked == [], f"credential-shaped env vars reached an offline test: {leaked}"


def test_scrubber_removes_an_inherited_credential(repo_root: Path) -> None:
    """Run a nested offline test with a poisoned environment and prove the scrubber fires.

    If the autouse fixture were removed, the nested
    ``test_offline_environment_has_no_credential`` would fail and this would too.
    """
    poisoned = {
        **os.environ,
        "SOME_PROVIDER_TOKEN": "canary-token-value",
        "STEALTHBENCH_ZEN_API_KEY": "sk-canary-value",
    }
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--strict-markers",
            "-q",
            "tests/contract/test_toolchain_config.py::test_offline_environment_has_no_credential",
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(repo_root),
        env=poisoned,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _remaining_credential_env() -> list[str]:
    from tests.conftest import CREDENTIAL_ENV_FRAGMENTS

    return [
        name
        for name in os.environ
        if any(fragment in name.upper() for fragment in CREDENTIAL_ENV_FRAGMENTS)
    ]


def test_refused_loopback_connection_is_a_real_error_not_a_guard_error() -> None:
    """A closed loopback port must fail normally, proving the guard is host-scoped."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    with pytest.raises(ConnectionRefusedError):
        socket.create_connection(("127.0.0.1", port), timeout=5)


@pytest.mark.network
def test_network_marker_allows_a_non_loopback_attempt() -> None:
    """Opting in with @pytest.mark.network removes the guard.

    RFC 5737 TEST-NET-1 is not routable, so this asserts only that the *guard* is
    gone: some ordinary OS error must surface, never NetworkAccessDenied.
    """
    try:
        socket.create_connection(("192.0.2.1", 9), timeout=1)
    except NetworkAccessDenied as exc:  # pragma: no cover - the failure this test prevents
        pytest.fail(f"network-marked test was still guarded: {exc}")
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Marker validation
# ---------------------------------------------------------------------------


def test_every_plan_marker_is_registered() -> None:
    declared = {
        match.group(1)
        for entry in _pyproject()["tool"]["pytest"]["ini_options"]["markers"]
        if (match := re.match(r"([a-z_]+):", entry))
    }
    assert declared >= REQUIRED_MARKERS, f"unregistered markers: {REQUIRED_MARKERS - declared}"


def test_unknown_marker_is_rejected_rather_than_ignored(pytester: pytest.Pytester) -> None:
    """`--strict-markers` turns a typo into a failure, not a silently unmarked test."""
    pytester.makefile(
        ".ini",
        pytest="""
        [pytest]
        addopts = --strict-markers
        markers =
            unit: declared
        """,
    )
    pytester.makepyfile(
        test_typo="""
        import pytest

        @pytest.mark.definitely_not_a_real_marker
        def test_typo():
            assert True
        """,
    )
    result = pytester.runpytest_subprocess()
    assert result.ret != 0, "an unregistered marker must fail, not warn"
    result.stdout.fnmatch_lines(["*definitely_not_a_real_marker*not found in*markers*"])


def test_declared_marker_collects_successfully(pytester: pytest.Pytester) -> None:
    pytester.makefile(
        ".ini",
        pytest="""
        [pytest]
        addopts = --strict-markers
        markers =
            unit: declared
        """,
    )
    pytester.makepyfile(
        test_ok="""
        import pytest

        @pytest.mark.unit
        def test_ok():
            assert True
        """,
    )
    result = pytester.runpytest_subprocess()
    result.assert_outcomes(passed=1)


def test_marker_documentation_is_present_for_each_marker() -> None:
    for entry in _pyproject()["tool"]["pytest"]["ini_options"]["markers"]:
        assert re.match(r"[a-z_]+: .+\S", entry), f"marker lacks a description: {entry!r}"


# ---------------------------------------------------------------------------
# CI selection
# ---------------------------------------------------------------------------


def test_offline_suites_collect_cleanly() -> None:
    """The gate's primary suites must collect with strict markers and no network."""
    env = dict(os.environ)
    env.pop("PYTEST_CURRENT_TEST", None)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--strict-markers",
            "--collect-only",
            "-q",
            "tests/unit",
            "tests/contract",
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(REPO_ROOT),
        env=env,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    collected = [line for line in result.stdout.splitlines() if "::" in line]
    assert collected, "unit and contract suites collected zero tests"


def test_workflow_runs_offline_suites_only() -> None:
    workflow = REPO_ROOT / ".github" / "workflows" / "ci.yml"
    assert workflow.is_file(), "CI workflow is missing"
    text = workflow.read_text(encoding="utf-8")

    pytest_lines = [line for line in text.splitlines() if "pytest" in line]
    assert pytest_lines, "the workflow must actually run tests"
    for line in pytest_lines:
        assert not re.search(r"(?<![\w-])-m\s", line), (
            f"the default CI job must not deselect markers: {line.strip()!r}"
        )

    for forbidden in ("tests/live", "paid", "tests/sandbox"):
        assert forbidden not in text, f"default CI job must not reference {forbidden}"

    assert "pytest --strict-markers tests/unit tests/contract" in text
    assert "pytest --strict-markers tests/integration tests/replay" in text


def test_workflow_contains_no_credentials() -> None:
    for path in sorted((REPO_ROOT / ".github" / "workflows").glob("*.yml")):
        text = path.read_text(encoding="utf-8")
        assert not re.search(r"(?i)(api[_-]?key|secret|token)\s*[:=]\s*['\"][^'\"]+", text), (
            f"{path.name} appears to embed a credential"
        )


def test_workflows_only_invoke_flags_the_cli_defines() -> None:
    """A committed workflow must not call a command that cannot exist yet.

    This caught `stealthbench run --dry-run`, a flag that only lands in G04 (T04D).
    """
    parser = build_parser()
    known: set[str] = {"-h", "--help", "--version"}
    for action in _iter_actions(parser):
        known.update(action.option_strings)

    for path in sorted((REPO_ROOT / ".github" / "workflows").glob("*.yml")):
        text = path.read_text(encoding="utf-8")
        for line in text.splitlines():
            if "stealthbench " not in line:
                continue
            invocation = line.split("stealthbench ", 1)[1]
            for token in re.findall(r"(?<!\w)--?[a-z][a-z-]*", invocation):
                assert token in known, (
                    f"{path.name} invokes `stealthbench {token}` but the parser defines no "
                    f"such option; either implement it or remove the step"
                )


def _iter_actions(parser: argparse.ArgumentParser) -> Iterator[argparse.Action]:
    for action in parser._actions:
        yield action
        if isinstance(action, argparse._SubParsersAction):
            for subparser in action.choices.values():
                yield from _iter_actions(subparser)


def test_live_workflow_is_marked_inert_until_dispatch_lands() -> None:
    """The live workflow is a G14 placeholder and must say so rather than look ready."""
    text = (REPO_ROOT / ".github" / "workflows" / "live.yml").read_text(encoding="utf-8")
    assert "workflow_dispatch" in text
    assert "G14" in text, "live.yml must state which gate makes it real"
    assert "not_implemented" in text or "not implemented" in text.lower()


def test_live_workflow_runs_the_live_suite() -> None:
    """tests/live claims only this workflow executes it; that claim must be true."""
    text = (REPO_ROOT / ".github" / "workflows" / "live.yml").read_text(encoding="utf-8")
    assert "pytest --strict-markers tests/live" in text


def test_live_and_paid_suites_exist_and_are_marked() -> None:
    live_dir = REPO_ROOT / "tests" / "live"
    assert live_dir.is_dir(), "tests/live must exist so live suites have a home"
    env = dict(os.environ)
    env.pop("PYTEST_CURRENT_TEST", None)
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--strict-markers", "--collect-only", "-q", "tests/live"],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(REPO_ROOT),
        env=env,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    for marker in ("live", "paid"):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "--strict-markers",
                "--collect-only",
                "-q",
                "-m",
                marker,
                "tests/live",
            ],
            capture_output=True,
            text=True,
            check=False,
            cwd=str(REPO_ROOT),
            env=env,
            timeout=300,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "::" in result.stdout, f"no test carries @{marker}; it must exist and be labelled"
