"""Installation and CLI shell behaviour (task T00A).

Oracle: a fresh virtual environment is built from the built wheel, the *installed
console script* is exercised as a subprocess, and every assertion is about a declared
behaviour rather than about the implementation's internal state.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from stealthbench.cli import EXIT_ERROR, EXIT_NOT_IMPLEMENTED, EXIT_OK, EXIT_USAGE, main

pytestmark = pytest.mark.integration

#: Substrings that must never appear in CLI output produced by a gate that has
#: not computed them. Guards against fabricated evaluation claims.
FORBIDDEN_CLAIM_TOKENS = (
    '"score"',
    '"accuracy"',
    '"resolved"',
    '"pass_rate"',
    '"identity"',
    '"probability"',
    '"evaluation_complete"',
)


def _run_cli(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    """Invoke the CLI in a subprocess so exit statuses are real."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2] / "src")
    env.pop("PYTEST_CURRENT_TEST", None)
    return subprocess.run(
        [sys.executable, "-m", "stealthbench.cli", *args],
        capture_output=True,
        text=True,
        check=False,
        cwd=None if cwd is None else str(cwd),
        env=env,
        timeout=120,
    )


# ---------------------------------------------------------------------------
# Fresh installation
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_wheel_installs_into_a_fresh_environment_and_console_script_runs(
    repo_root: Path, tmp_path: Path
) -> None:
    """A clean venv from the built wheel can run the installed console script.

    The wheel is installed with ``--no-deps --no-index`` so the check is hermetic and
    offline. That means the declared runtime dependencies are not exercised here;
    they are covered by ``constraints-dev.txt`` and CI's full install.
    """
    wheel_dir = tmp_path / "wheelhouse"
    wheel_dir.mkdir()

    build = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-deps",
            "--no-build-isolation",
            "--no-index",
            "--wheel-dir",
            str(wheel_dir),
            str(repo_root),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=600,
    )
    assert build.returncode == 0, f"wheel build failed:\n{build.stdout}\n{build.stderr}"

    wheels = sorted(wheel_dir.glob("stealthbench-*.whl"))
    assert len(wheels) == 1, f"expected exactly one wheel, found {[w.name for w in wheels]}"

    venv_dir = tmp_path / "fresh"
    subprocess.run(
        [sys.executable, "-m", "venv", str(venv_dir)],
        capture_output=True,
        text=True,
        check=True,
        timeout=600,
    )
    venv_python = _venv_interpreter(venv_dir)

    install = subprocess.run(
        [
            str(venv_python),
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--no-index",
            "--disable-pip-version-check",
            str(wheels[0]),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=600,
    )
    assert install.returncode == 0, f"fresh install failed:\n{install.stdout}\n{install.stderr}"

    # Resolve after install: the entry point only exists once pip has written it.
    console_script = next(
        (
            path
            for path in (
                venv_dir / "bin" / "stealthbench",
                venv_dir / "Scripts" / "stealthbench.exe",
            )
            if path.exists()
        ),
        None,
    )
    assert console_script is not None, (
        "the [project.scripts] entry point was not installed into the fresh environment"
    )

    help_result = subprocess.run(
        [str(console_script), "--help"],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert help_result.returncode == EXIT_OK, help_result.stderr
    assert "stealthbench" in help_result.stdout

    version_result = subprocess.run(
        [str(console_script), "version"],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert version_result.returncode == EXIT_OK, version_result.stderr
    assert json.loads(version_result.stdout)["schema_version"] == "1.0"

    assert venv_python.exists()


def _venv_interpreter(venv_dir: Path) -> Path:
    for candidate in (venv_dir / "bin" / "python", venv_dir / "Scripts" / "python.exe"):
        if candidate.exists():
            return candidate
    raise AssertionError(f"no interpreter inside {venv_dir}")  # pragma: no cover


def test_installed_package_exposes_its_version() -> None:
    """`version` reports the installed distribution version, not a hardcoded string."""
    from importlib.metadata import version as installed_version

    import stealthbench

    assert stealthbench.__version__ == installed_version("stealthbench")
    assert stealthbench.SCHEMA_VERSION == "1.0"


# ---------------------------------------------------------------------------
# CLI help and valid commands
# ---------------------------------------------------------------------------


def test_help_exits_zero_and_lists_every_declared_command() -> None:
    result = _run_cli("--help")
    assert result.returncode == EXIT_OK, result.stderr
    for command in ("manifest", "run", "replay", "report", "signatures", "identify", "doctor"):
        assert command in result.stdout, f"{command} missing from help output"


def test_version_command_prints_machine_readable_status(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["version"]) == EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ok"
    assert payload["schema_version"] == "1.0"


# ---------------------------------------------------------------------------
# Invalid invocations
# ---------------------------------------------------------------------------


def test_unknown_command_fails_without_output() -> None:
    result = _run_cli("definitely-not-a-command")
    assert result.returncode != EXIT_OK
    assert result.returncode == EXIT_USAGE
    assert result.stdout == ""
    assert "definitely-not-a-command" in result.stderr


def test_no_command_fails() -> None:
    result = _run_cli()
    assert result.returncode == EXIT_USAGE


def test_missing_required_argument_fails() -> None:
    result = _run_cli("manifest", "validate")
    assert result.returncode == EXIT_USAGE
    assert result.stdout == ""


def test_unknown_flag_is_rejected() -> None:
    result = _run_cli("run", "configs/offline-demo.json", "--not-a-flag")
    assert result.returncode == EXIT_USAGE


# ---------------------------------------------------------------------------
# Configuration rejection
# ---------------------------------------------------------------------------


def test_nonexistent_manifest_path_is_rejected(tmp_path: Path) -> None:
    missing = tmp_path / "nope.json"
    result = _run_cli("manifest", "validate", str(missing))
    assert result.returncode == EXIT_ERROR
    assert "nope.json" in result.stderr
    assert result.stdout == ""


def test_directory_path_is_not_accepted_as_a_manifest(tmp_path: Path) -> None:
    result = _run_cli("manifest", "validate", str(tmp_path))
    assert result.returncode == EXIT_ERROR
    assert "not a file" in result.stderr


def test_missing_report_output_flag_is_rejected(repo_root: Path) -> None:
    result = _run_cli("report", "configs/offline-demo.json")
    assert result.returncode == EXIT_USAGE


# ---------------------------------------------------------------------------
# No fabricated evaluation claims
# ---------------------------------------------------------------------------


def test_unimplemented_commands_report_pending_and_claim_no_result(repo_root: Path) -> None:
    """Declared-but-unimplemented commands must not look like successes."""
    # `replay` left this set in G05 T05C (now reconstructs offline); `run
    # --offline` is covered by the vertical workflow test; `report` left it
    # in G13 (now builds a static site offline).
    invocations = [
        ("signatures", "configs/offline-demo.json"),
        ("identify", "configs/offline-demo.json"),
    ]
    for args in invocations:
        result = _run_cli(*args)
        combined = result.stdout + result.stderr
        assert result.returncode == EXIT_NOT_IMPLEMENTED, (args, result.returncode, combined)
        assert result.stdout == "", f"{args} wrote a result to stdout: {result.stdout!r}"
        payload = json.loads(result.stderr.splitlines()[-1])
        assert payload["status"] == "not_implemented"
        assert payload["gate"].startswith("G")
        for token in FORBIDDEN_CLAIM_TOKENS:
            assert token not in combined, f"{args} emitted forbidden claim token {token}"


def test_run_offline_exits_not_implemented_and_writes_no_result(
    repo_root: Path,
) -> None:
    """``run`` gained authorization and refusal in G04; the loop lands in G05.

    Until then it must keep the pending-command contract: exit 3, no result on stdout,
    and no forbidden claim token anywhere.
    """
    result = _run_cli("run", "configs/offline-demo.json")
    combined = result.stdout + result.stderr
    assert result.returncode == EXIT_NOT_IMPLEMENTED, (result.returncode, combined)
    assert result.stdout == "", f"run wrote a result to stdout: {result.stdout!r}"
    payload = json.loads(result.stderr.splitlines()[-1])
    assert payload["status"] == "not_implemented"
    assert payload["command"] == "run"
    for token in FORBIDDEN_CLAIM_TOKENS:
        assert token not in combined, f"run emitted forbidden claim token {token}"


def test_run_live_refusal_names_blockers_and_claims_no_dispatch() -> None:
    """An unauthorized live run must refuse with a list, never a dispatch count."""
    result = _run_cli("run", "configs/offline-demo.json", "--live")
    combined = result.stdout + result.stderr
    assert result.returncode == EXIT_USAGE, (result.returncode, combined)
    payload = json.loads(result.stdout)
    assert payload["status"] == "refused"
    assert payload["dispatched"] == 0
    assert payload["blockers"]
    for token in FORBIDDEN_CLAIM_TOKENS:
        assert token not in combined, f"run emitted forbidden claim token {token}"


def test_report_rejects_non_artifact_input_without_writing_report(
    repo_root: Path, tmp_path: Path
) -> None:
    """`report` landed in G13: a manifest is not an artifact directory."""
    target = tmp_path / "report-out"
    result = _run_cli("report", "configs/offline-demo.json", "--output", str(target))
    assert result.returncode == EXIT_ERROR
    assert not (target / "index.html").exists(), "a rejected report must not write a site"


def test_doctor_reports_runtime_and_dataset_availability_offline() -> None:
    """`doctor` (G14) exits 0 with availability, inventing no evaluation claim."""
    result = _run_cli("doctor")
    combined = result.stdout + result.stderr
    assert result.returncode == EXIT_OK, (result.returncode, combined)
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["schema_version"] == "1.0"
    assert payload["configs"]["offline-demo"] == "valid-dispatchable"
    assert payload["container_state"] in ("ready", "blocked_external")
    if payload["container_state"] == "blocked_external":
        assert "blocked_external" in (payload["container_reason"] or "")
    assert payload["datasets"]["offline_demo"] == "available"
    for token in FORBIDDEN_CLAIM_TOKENS:
        assert token not in combined, f"doctor emitted forbidden claim token {token}"
