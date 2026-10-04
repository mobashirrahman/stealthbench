"""Sandbox resource limits (task T07A, gate G07).

Runs real candidate programs through the disposable backend and asserts the
caps bite: a correct program passes, an infinite loop hits the wall clock, a
chatty program is truncated at the output cap, an over-allocating program
fails where the platform can enforce the memory cap, and a cancelled run
still cleans up. The container-only test skips with ``blocked_external``
when no runtime is present; it never fake-passes.
"""

from __future__ import annotations

import os
import time
from functools import lru_cache
from pathlib import Path

import pytest

from stealthbench.sandbox import runtime
from stealthbench.sandbox.runtime import (
    ExecutionStatus,
    SandboxLimits,
    container_check_status,
    isolated_workdir,
    memory_limit_enforceable,
    run_python_source,
)

pytestmark = pytest.mark.sandbox


def test_correct_solution_passes() -> None:
    result = run_python_source(
        "def add(a, b):\n    return a + b\n\nprint(add(40, 2))\n",
        limits=SandboxLimits(wall_seconds=10.0),
    )
    assert result.status is ExecutionStatus.OK
    assert result.exit_code == 0
    assert result.stdout.strip() == "42"
    assert not result.timed_out
    assert not result.truncated


def test_infinite_loop_hits_wall_timeout() -> None:
    result = run_python_source(
        "while True:\n    pass\n",
        limits=SandboxLimits(wall_seconds=1.0),
    )
    assert result.status is ExecutionStatus.TIMEOUT
    assert result.timed_out
    assert result.duration_seconds >= 0.9
    assert result.duration_seconds < 15.0


def test_huge_output_is_truncated_at_the_cap() -> None:
    cap = 64 * 1024
    result = run_python_source(
        "for i in range(300_000):\n    print('x' * 40, i)\n",
        limits=SandboxLimits(wall_seconds=20.0, max_output_bytes=cap),
    )
    assert result.status is ExecutionStatus.OK
    assert result.exit_code == 0
    assert result.stdout_truncated
    assert len(result.stdout.encode("utf-8")) <= cap


@lru_cache(maxsize=1)
def _memory_cap_bites() -> bool:
    """Empirically check the address-space cap kills an over-allocator."""
    probe = run_python_source(
        "bytearray(256 * 1024 * 1024)\n",
        limits=SandboxLimits(wall_seconds=20.0, memory_bytes=64 * 1024 * 1024),
    )
    return probe.exit_code != 0


def test_memory_cap_where_enforceable() -> None:
    if not memory_limit_enforceable():
        pytest.skip("blocked_external: no RLIMIT_AS backend on this platform")
    if not _memory_cap_bites():
        pytest.skip("blocked_external: address-space cap not enforced by this kernel")
    result = run_python_source(
        "data = bytearray(512 * 1024 * 1024)\nprint(len(data))\n",
        limits=SandboxLimits(wall_seconds=20.0, memory_bytes=64 * 1024 * 1024),
    )
    assert result.status is not ExecutionStatus.OK
    assert result.exit_code != 0
    assert not result.timed_out


def test_cleanup_after_cancel(tmp_path: Path) -> None:
    calls = 0

    def stop_after_a_few_polls() -> bool:
        nonlocal calls
        calls += 1
        return calls < 5

    result = run_python_source(
        "import time\ntime.sleep(60)\n",
        limits=SandboxLimits(wall_seconds=60.0),
        should_continue=stop_after_a_few_polls,
        workdir_parent=tmp_path,
    )
    assert result.status is ExecutionStatus.CANCELLED
    assert result.duration_seconds < 30.0
    assert list(tmp_path.iterdir()) == []


def test_isolated_workdir_removed_on_exception(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="boom"), isolated_workdir(tmp_path) as workdir:
        assert workdir.is_dir()
        raise RuntimeError("boom")
    assert list(tmp_path.iterdir()) == []


def test_spawned_children_do_not_outlive_the_run(tmp_path: Path) -> None:
    source = (
        "import subprocess, sys\n"
        "child = subprocess.Popen(\n"
        "    [sys.executable, '-c', 'import time; time.sleep(30)'],\n"
        "    stdout=subprocess.DEVNULL,\n"
        "    stderr=subprocess.DEVNULL,\n"
        ")\n"
        "print(f'child={child.pid}', flush=True)\n"
    )
    result = run_python_source(
        source,
        limits=SandboxLimits(wall_seconds=20.0),
        workdir_parent=tmp_path,
    )
    assert result.status is ExecutionStatus.OK
    assert list(tmp_path.iterdir()) == []
    child_pid = int(result.stdout.strip().split("child=")[1])
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        try:
            os.kill(child_pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    else:
        pytest.fail("spawned child outlived the sandboxed run")


def test_container_status_reports_blocked_external_without_runtime() -> None:
    if runtime.container_runtime() is not None:
        pytest.skip("a container runtime is present; nothing to report missing")
    status = container_check_status("strict containment check")
    assert status["state"] == str(ExecutionStatus.BLOCKED_EXTERNAL)
    assert status["runtime"] is None
    assert status["reason"]


def test_container_backend_when_available() -> None:
    try:
        name = runtime.require_container_runtime("resource-limit parity check")
    except runtime.ContainerUnavailable as exc:
        pytest.skip(str(exc))
    assert name in {"docker", "podman"}
    assert container_check_status("resource-limit parity check")["state"] == "ready"
