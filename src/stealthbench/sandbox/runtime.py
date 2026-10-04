"""Disposable execution backend for untrusted candidate code (T07A, G07).

Each run gets a fresh temporary directory as its working directory, a
scrubbed environment (see :mod:`stealthbench.sandbox.policy`), POSIX resource
limits where the platform offers them, a wall-clock timeout, and an output
cap. Nothing outlives the run: on normal exit stray children are reaped via
the process group, and on timeout or cancellation the group is killed and the
directory removed.

Default limits mirror the pinned upstream checkers recorded in
``docs/upstream-inventory.md``: the 6-second LiveCodeBench ``--timeout``
default for the wall clock, and the EvalPlus memory policy
(``min(4GB, system maximum)``, overridable via ``EVALPLUS_MAX_MEMORY_BYTES``)
available through :func:`evalplus_limits`-style explicit limits. Timeouts and
out-of-memory surface as failures, never as missing data.

What this backend cannot do is also declared, not silently absorbed:

* True packet-level network denial and host-filesystem hiding need a
  container. :func:`container_runtime` detects docker/podman;
  :func:`require_container_runtime` raises :class:`ContainerUnavailable` when
  absent so container-only checks report ``blocked_external`` via
  :func:`container_check_status` instead of fake-passing.
* :func:`memory_limit_enforceable` reports whether the ``RLIMIT_AS`` cap can
  actually be applied on this platform; callers skip the memory test when it
  cannot, rather than asserting a limit that is not there. Each requested
  rlimit is applied best-effort in the child (kernels differ: XNU rejects
  address-space caps outright), so the wall clock, output cap, tmpdir
  isolation, env scrub and process-group cleanup are the portable
  guarantees, while address-space and process-count caps hold where the
  kernel honours them (Linux) and via the container flags otherwise.
"""

from __future__ import annotations

import contextlib
import os
import platform
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Final

from stealthbench.sandbox.policy import candidate_env, write_candidate_file

__all__ = [
    "EVALPLUS_MEMORY_BYTES",
    "EVALPLUS_WALL_BASE_SECONDS",
    "LIVECODEBENCH_WALL_SECONDS",
    "ContainerUnavailable",
    "ExecutionResult",
    "ExecutionStatus",
    "SandboxLimits",
    "container_check_status",
    "container_exec_args",
    "container_runtime",
    "evalplus_limits",
    "isolated_workdir",
    "livecodebench_limits",
    "memory_limit_enforceable",
    "posix_rlimit_available",
    "require_container_runtime",
    "run_in_sandbox",
    "run_python_source",
]

#: Wall clock matching the LiveCodeBench ``--timeout`` default (6s). Upstream
#: warns this moves pass@1 by >0.5 points, so it is the default, recorded per
#: run, and never tuned after seeing scores.
LIVECODEBENCH_WALL_SECONDS: Final[float] = 6.0

#: EvalPlus per-task base timeout ``T_base`` (4s); the full rule is
#: ``T = max(T_base, T_gt * k)`` with ``k = 4``.
EVALPLUS_WALL_BASE_SECONDS: Final[float] = 4.0

#: EvalPlus memory policy: ``min(4GB, system maximum)`` per process.
EVALPLUS_MEMORY_BYTES: Final[int] = 4 * 1024 * 1024 * 1024

#: How often a running child is polled for exit, timeout and cancellation.
_POLL_SECONDS: Final[float] = 0.02

#: Grace period to reap output after killing a timed-out process group.
_REAP_SECONDS: Final[float] = 10.0

_resource: Any
try:
    import resource as _resource_module

    _resource = _resource_module
except ImportError:  # Non-POSIX platforms have no rlimit backend.
    _resource = None


class ExecutionStatus(StrEnum):
    """How a sandboxed run ended. Timeouts and OOM are failures, not gaps."""

    OK = "ok"
    TIMEOUT = "timeout"
    FAILED = "failed"
    CANCELLED = "cancelled"
    BLOCKED_EXTERNAL = "blocked_external"


@dataclass(frozen=True, slots=True)
class SandboxLimits:
    """Caps for one disposable run. All sizes are in bytes."""

    wall_seconds: float = LIVECODEBENCH_WALL_SECONDS
    cpu_seconds: int | None = None
    memory_bytes: int | None = 256 * 1024 * 1024
    max_output_bytes: int = 256 * 1024
    max_file_bytes: int | None = 64 * 1024 * 1024
    max_processes: int | None = None

    def __post_init__(self) -> None:
        if self.wall_seconds <= 0:
            raise ValueError(f"wall_seconds must be positive, got {self.wall_seconds}")
        for name in ("memory_bytes", "max_output_bytes", "max_file_bytes", "max_processes"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.cpu_seconds is not None and self.cpu_seconds <= 0:
            raise ValueError(f"cpu_seconds must be positive, got {self.cpu_seconds}")


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    """What a disposable run did. Output is already capped; flags say so."""

    status: ExecutionStatus
    exit_code: int | None
    stdout: str
    stderr: str
    stdout_truncated: bool
    stderr_truncated: bool
    timed_out: bool
    duration_seconds: float
    disk_bytes: int
    reason: str | None = None

    @property
    def truncated(self) -> bool:
        """Whether either stream hit the output cap."""
        return self.stdout_truncated or self.stderr_truncated

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": str(self.status),
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "stdout_truncated": self.stdout_truncated,
            "stderr_truncated": self.stderr_truncated,
            "timed_out": self.timed_out,
            "duration_seconds": self.duration_seconds,
            "disk_bytes": self.disk_bytes,
            "reason": self.reason,
        }


class ContainerUnavailable(Exception):
    """A container-only check was requested without a container runtime.

    The ``reason`` doubles as the ``blocked_external`` evidence string: it
    names what is missing rather than letting the check fake-pass on the
    best-effort subprocess backend.
    """

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def posix_rlimit_available() -> bool:
    """Whether the ``resource`` module can apply limits on this platform."""
    return _resource is not None and hasattr(_resource, "setrlimit")


def memory_limit_enforceable() -> bool:
    """Whether the address-space cap can actually be applied here.

    XNU (macOS) rejects ``setrlimit`` on ``RLIMIT_AS``/``RLIMIT_DATA``/
    ``RLIMIT_RSS`` outright, so the answer is unconditionally false there even
    though the constants exist. Callers asserting a memory cap must
    additionally probe empirically (an over-allocator must die); the static
    answer alone is not proof on any kernel.
    """
    if os.name != "posix" or not posix_rlimit_available():
        return False
    if not hasattr(_resource, "RLIMIT_AS"):
        return False
    # XNU rejects the address-space cap even though the constant exists.
    return platform.system() != "Darwin"


def container_runtime() -> str | None:
    """Name of a usable container executable, or ``None`` when absent."""
    for name in ("docker", "podman"):
        if shutil.which(name) is not None:
            return name
    return None


def require_container_runtime(purpose: str) -> str:
    """Return the container executable or raise with a ``blocked_external`` reason."""
    runtime = container_runtime()
    if runtime is None:
        raise ContainerUnavailable(
            f"blocked_external: no container runtime (docker/podman) on PATH for {purpose}"
        )
    return runtime


def container_check_status(purpose: str) -> dict[str, str | None]:
    """Report readiness for a container-only check without raising."""
    runtime = container_runtime()
    if runtime is None:
        return {
            "state": str(ExecutionStatus.BLOCKED_EXTERNAL),
            "runtime": None,
            "reason": f"no container runtime (docker/podman) on PATH for {purpose}",
        }
    return {"state": "ready", "runtime": runtime, "reason": None}


def container_exec_args(
    *,
    runtime: str,
    image: str,
    workdir: Path,
    command: Sequence[str],
    limits: SandboxLimits,
) -> list[str]:
    """Build a locked-down container invocation for candidate code.

    The container gets no network, a read-only bind of the candidate work
    directory, and the recorded memory/process caps. The verifier directory
    is deliberately not a parameter: hidden tests run in a separate
    evaluator environment and are never mounted here.
    """
    args = [
        runtime,
        "run",
        "--rm",
        "--network",
        "none",
        "--workdir",
        "/work",
        "--volume",
        f"{workdir}:/work:ro",
    ]
    if limits.memory_bytes is not None:
        args += ["--memory", f"{limits.memory_bytes}b"]
    if limits.max_processes is not None:
        args += ["--pids-limit", str(limits.max_processes)]
    args.append(image)
    args.extend(command)
    return args


def evalplus_limits(
    *,
    wall_seconds: float = EVALPLUS_WALL_BASE_SECONDS,
    memory_bytes: int | None = EVALPLUS_MEMORY_BYTES,
) -> SandboxLimits:
    """Limits matching the EvalPlus execution policy (timeout + 4GB cap)."""
    return SandboxLimits(wall_seconds=wall_seconds, memory_bytes=memory_bytes)


def livecodebench_limits(
    *,
    wall_seconds: float = LIVECODEBENCH_WALL_SECONDS,
    memory_bytes: int | None = 256 * 1024 * 1024,
) -> SandboxLimits:
    """Limits matching the LiveCodeBench 6-second checker timeout."""
    return SandboxLimits(wall_seconds=wall_seconds, memory_bytes=memory_bytes)


@contextmanager
def isolated_workdir(parent: Path | str | None = None) -> Iterator[Path]:
    """Yield a fresh disposable directory, removed on exit for any reason."""
    directory = parent if parent is None else str(parent)
    with tempfile.TemporaryDirectory(prefix="stealthbench-sandbox-", dir=directory) as tmp:
        yield Path(tmp)


def _apply_limits(limits: SandboxLimits) -> None:
    """Apply rlimits in the forked child before exec. Runs without imports.

    Best-effort per limit: kernels honour different subsets (XNU rejects
    address-space caps; only some kernels allow lowering ``RLIMIT_NPROC``),
    and a rejected cap must not abort caps that did apply. Honesty about
    what held comes from :func:`memory_limit_enforceable` plus empirical
    probes in the test suite, never from assuming ``setrlimit`` succeeded.
    """
    if _resource is None:
        return
    requested: tuple[tuple[str, int | None], ...] = (
        ("RLIMIT_AS", limits.memory_bytes),
        ("RLIMIT_CPU", limits.cpu_seconds),
        ("RLIMIT_FSIZE", limits.max_file_bytes),
        ("RLIMIT_NPROC", limits.max_processes),
    )
    for attr, value in requested:
        if value is None or not hasattr(_resource, attr):
            continue
        with contextlib.suppress(OSError, ValueError):
            _resource.setrlimit(getattr(_resource, attr), (value, value))


def _child_setup(limits: SandboxLimits) -> Callable[[], None]:
    """Detach into a new session, then apply limits best-effort.

    Linux-only path: ``preexec_fn`` is unsafe on Darwin (threads + fork),
    so Darwin callers use ``start_new_session=True`` instead and skip this.
    """

    def _setup() -> None:
        os.setsid()
        _apply_limits(limits)

    return _setup


def _kill_tree(proc: subprocess.Popen[bytes], pgid: int | None) -> None:
    """SIGKILL the child's process group; fall back to the child alone.

    The group id is captured at spawn: once the group leader is reaped,
    ``getpgid`` can no longer resolve it, which would otherwise let detached
    grandchildren outlive the run.
    """
    killpg = getattr(os, "killpg", None)
    if pgid is not None and killpg is not None:
        with contextlib.suppress(OSError):
            killpg(pgid, signal.SIGKILL)
    with contextlib.suppress(OSError):
        proc.kill()


def _cap_output(data: bytes, limit: int) -> tuple[str, bool]:
    """Decode up to ``limit`` bytes; report whether the rest was dropped."""
    if len(data) > limit:
        return (data[:limit].decode("utf-8", errors="replace"), True)
    return (data.decode("utf-8", errors="replace"), False)


def _workdir_bytes(root: Path) -> int:
    """Total bytes of regular files left in the work directory."""
    total = 0
    for path in root.rglob("*"):
        with contextlib.suppress(OSError):
            if path.is_file() and not path.is_symlink():
                total += path.stat().st_size
    return total


def run_in_sandbox(
    command: Sequence[str],
    *,
    files: Mapping[str, str | bytes] | None = None,
    limits: SandboxLimits | None = None,
    extra_env: Mapping[str, str] | None = None,
    should_continue: Callable[[], bool] | None = None,
    workdir_parent: Path | str | None = None,
) -> ExecutionResult:
    """Run ``command`` once in a disposable directory under ``limits``.

    ``files`` are staged through the policy checks, so traversal and symlink
    escapes raise :class:`PolicyDenied` before anything spawns. ``None`` for
    a limit disables that cap; the wall clock and output cap always apply.
    ``should_continue`` returning ``False`` cancels the run: the process
    group is killed, the directory is still removed, and the result reports
    ``cancelled``.
    """
    if not command:
        raise ValueError("run_in_sandbox requires a non-empty command")
    active = limits or SandboxLimits()
    started = time.monotonic()
    with isolated_workdir(workdir_parent) as workdir:
        for name, content in (files or {}).items():
            write_candidate_file(workdir, name, content)
        env = candidate_env(workdir, extra_env)
        # Portable spawn: preexec_fn (fork-time setsid + rlimits) is Linux
        # only. On Darwin it deadlocks with threads and XNU rejects the
        # address-space cap anyway, so detach via start_new_session and rely
        # on the wall clock, output cap, tmpdir isolation, env scrub and
        # process-group cleanup there.
        use_preexec = os.name == "posix" and platform.system() != "Darwin"
        use_new_session = os.name == "posix" and not use_preexec
        proc = subprocess.Popen(
            list(command),
            cwd=workdir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=_child_setup(active) if use_preexec else None,
            start_new_session=use_new_session,
        )
        getpgid = getattr(os, "getpgid", None)
        try:
            child_pgid: int | None = getpgid(proc.pid) if getpgid is not None else None
        except OSError:
            child_pgid = None
        deadline = started + active.wall_seconds
        timed_out = False
        cancelled = False
        stdout_data = b""
        stderr_data = b""
        while True:
            if should_continue is not None and not should_continue():
                cancelled = True
                _kill_tree(proc, child_pgid)
                with contextlib.suppress(subprocess.TimeoutExpired):
                    stdout_data, stderr_data = proc.communicate(timeout=_REAP_SECONDS)
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
                _kill_tree(proc, child_pgid)
                with contextlib.suppress(subprocess.TimeoutExpired):
                    stdout_data, stderr_data = proc.communicate(timeout=_REAP_SECONDS)
                break
            try:
                stdout_data, stderr_data = proc.communicate(timeout=min(remaining, _POLL_SECONDS))
                break
            except subprocess.TimeoutExpired:
                continue
        # Nothing outlives the run: reap strays the child detached before exit.
        _kill_tree(proc, child_pgid)
        with contextlib.suppress(OSError, subprocess.TimeoutExpired):
            proc.wait(timeout=_REAP_SECONDS)
        disk_bytes = _workdir_bytes(workdir)
    duration = time.monotonic() - started
    stdout, stdout_truncated = _cap_output(stdout_data, active.max_output_bytes)
    stderr, stderr_truncated = _cap_output(stderr_data, active.max_output_bytes)
    if cancelled:
        return ExecutionResult(
            status=ExecutionStatus.CANCELLED,
            exit_code=proc.returncode,
            stdout=stdout,
            stderr=stderr,
            stdout_truncated=stdout_truncated,
            stderr_truncated=stderr_truncated,
            timed_out=False,
            duration_seconds=duration,
            disk_bytes=disk_bytes,
            reason="cancelled by the caller; process group killed",
        )
    if timed_out:
        return ExecutionResult(
            status=ExecutionStatus.TIMEOUT,
            exit_code=proc.returncode,
            stdout=stdout,
            stderr=stderr,
            stdout_truncated=stdout_truncated,
            stderr_truncated=stderr_truncated,
            timed_out=True,
            duration_seconds=duration,
            disk_bytes=disk_bytes,
            reason=f"wall clock {active.wall_seconds}s exceeded; process group killed",
        )
    if proc.returncode == 0:
        return ExecutionResult(
            status=ExecutionStatus.OK,
            exit_code=0,
            stdout=stdout,
            stderr=stderr,
            stdout_truncated=stdout_truncated,
            stderr_truncated=stderr_truncated,
            timed_out=False,
            duration_seconds=duration,
            disk_bytes=disk_bytes,
            reason=None,
        )
    return ExecutionResult(
        status=ExecutionStatus.FAILED,
        exit_code=proc.returncode,
        stdout=stdout,
        stderr=stderr,
        stdout_truncated=stdout_truncated,
        stderr_truncated=stderr_truncated,
        timed_out=False,
        duration_seconds=duration,
        disk_bytes=disk_bytes,
        reason=f"process exited with code {proc.returncode}",
    )


def run_python_source(
    source: str,
    *,
    filename: str = "candidate.py",
    args: Sequence[str] | None = None,
    limits: SandboxLimits | None = None,
    extra_env: Mapping[str, str] | None = None,
    should_continue: Callable[[], bool] | None = None,
    workdir_parent: Path | str | None = None,
) -> ExecutionResult:
    """Stage ``source`` as ``filename`` and run it with the current interpreter."""
    command = [sys.executable, filename, *(args or [])]
    return run_in_sandbox(
        command,
        files={filename: source},
        limits=limits,
        extra_env=extra_env,
        should_continue=should_continue,
        workdir_parent=workdir_parent,
    )
