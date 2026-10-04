"""Isolated execution and evaluator separation (gate G07).

Two layers with one rule between them: **candidate code never shares an
environment with hidden tests or secrets.**

* :mod:`stealthbench.sandbox.runtime` runs untrusted code in a disposable
  directory with POSIX resource limits, a wall-clock timeout, an output cap,
  a scrubbed environment and cleanup even on cancellation.
* :mod:`stealthbench.sandbox.policy` decides what may enter that directory:
  no path traversal, no symlink escape, no secret mounts, and the verifier
  lives in a separate directory that is never staged for the candidate.

Strict network/host containment needs a container runtime. When none is
present, :func:`require_container_runtime` raises
:class:`ContainerUnavailable` so the check reports ``blocked_external``
instead of fake-passing on the best-effort subprocess backend.
"""

from __future__ import annotations

from stealthbench.sandbox.policy import (
    CREDENTIAL_ENV_FRAGMENTS,
    EvaluatorLayout,
    PolicyDenied,
    assert_no_secret_in_env,
    assert_verifier_separation,
    candidate_env,
    candidate_visible_paths,
    is_secret_env_name,
    network_policy,
    resolve_candidate_path,
    scrub_env,
    write_candidate_file,
)
from stealthbench.sandbox.runtime import (
    ContainerUnavailable,
    ExecutionResult,
    ExecutionStatus,
    SandboxLimits,
    container_check_status,
    container_exec_args,
    container_runtime,
    isolated_workdir,
    memory_limit_enforceable,
    posix_rlimit_available,
    require_container_runtime,
    run_in_sandbox,
    run_python_source,
)

__all__ = [
    "CREDENTIAL_ENV_FRAGMENTS",
    "ContainerUnavailable",
    "EvaluatorLayout",
    "ExecutionResult",
    "ExecutionStatus",
    "PolicyDenied",
    "SandboxLimits",
    "assert_no_secret_in_env",
    "assert_verifier_separation",
    "candidate_env",
    "candidate_visible_paths",
    "container_check_status",
    "container_exec_args",
    "container_runtime",
    "is_secret_env_name",
    "isolated_workdir",
    "memory_limit_enforceable",
    "network_policy",
    "posix_rlimit_available",
    "require_container_runtime",
    "resolve_candidate_path",
    "run_in_sandbox",
    "run_python_source",
    "scrub_env",
    "write_candidate_file",
]
