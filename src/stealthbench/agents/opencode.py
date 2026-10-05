"""Pinned OpenCode runner and model broker (G10 T10A).

Official posture (see ``docs/upstream-inventory.md`` SWE-bench Verified and
Terminal-Bench sections): the agent trajectory never touches credentials or
hidden evaluator assets. Model access goes through a broker whose secrets live
outside the task container; the task container gets a scrubbed environment
(see :mod:`stealthbench.sandbox.policy`) and a fresh disposable directory per
task (see :mod:`stealthbench.sandbox.runtime`). The verifier lives in a
sibling directory that is never mounted into the task container.

What this module does and does not do:

* It does **not** reimplement OpenCode. :func:`pinned_invocation` builds the
  single pinned CLI argv; live execution requires that binary at exactly
  :data:`PINNED_OPENCODE_VERSION`. Offline tests inject ``step_fn`` /
  ``tool_fn`` fixtures and never spawn the binary.
* One complete trajectory is **one** attempt (``docs/contracts.md``). The
  runner returns one :class:`AgentResult` per call; retries or regrades are
  separate attempts with separate ids.
* Every cap binds the full trajectory: steps, tool calls, per-tool time,
  wall time, input/output tokens and cost. Prompt selection, compaction
  events and auxiliary model calls are recorded in the result and folded
  into token/cost totals and attribution metadata.
* Secrets and evaluator paths are inaccessible by construction. A tool that
  names a secret, a proxy, an evaluator pointer or a verifier path raises
  :class:`PolicyDenied` (or :class:`BrokerDenied`) instead of leaking it.
* Cancellation is clean: ``should_continue`` returning ``False`` stops the
  loop, the temporary task root is still removed, and the result reports
  ``cancelled``.

Offline by construction unless the caller explicitly requests live
execution: fixture stepping is pure local logic over strings and contracts.
No socket, no credential in the task environment.
"""

from __future__ import annotations

import shutil
import tempfile
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Final

from stealthbench.sandbox.policy import (
    EvaluatorLayout,
    PolicyDenied,
    assert_no_secret_in_env,
    assert_verifier_separation,
    candidate_env,
)
from stealthbench.sandbox.runtime import (
    ContainerUnavailable,
    SandboxLimits,
    container_exec_args,
    container_runtime,
)

__all__ = [
    "OPENCODE_BINARY_NAME",
    "OPENCODE_PINNED_ARGS",
    "PINNED_OPENCODE_VERSION",
    "AgentActionKind",
    "AgentAttribution",
    "AgentLimits",
    "AgentResult",
    "AgentStatus",
    "AgentTraceEvent",
    "AuxCall",
    "BrokerDenied",
    "CompactionEvent",
    "ModelBroker",
    "StepContext",
    "StepOutcome",
    "agent_container_args",
    "agent_cost_usd",
    "live_check_status",
    "opencode_binary",
    "pinned_invocation",
    "require_opencode_binary",
    "run_agent_task",
    "verify_pinned_version",
]

#: StealthBench-pinned OpenCode CLI version. Live runs refuse any other
#: version so prompts, compaction behaviour and trace shape cannot drift
#: under a moving binary.
PINNED_OPENCODE_VERSION: Final[str] = "0.14.0"

#: Binary name resolved via ``shutil.which``; never a hardcoded absolute path.
OPENCODE_BINARY_NAME: Final[str] = "opencode"

#: Pinned subcommand shape after the binary name. The model, workdir and
#: prompt file are appended by :func:`pinned_invocation`.
OPENCODE_PINNED_ARGS: Final[tuple[str, ...]] = ("run", "--format", "json")


class AgentStatus(StrEnum):
    """How one agent trajectory ended. Budget ends are failures, not gaps."""

    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    STEP_EXHAUSTED = "step_exhausted"
    BUDGET_EXHAUSTED = "budget_exhausted"
    CANCELLED = "cancelled"


class AgentActionKind(StrEnum):
    """The single decision a fixture step function may return."""

    TOOL_CALL = "tool_call"
    FINISH = "finish"
    FAIL = "fail"
    COMPACTION = "compaction"
    AUX_CALL = "aux_call"
    SECRET_READ_ATTEMPT = "secret_read_attempt"


class BrokerDenied(Exception):
    """The broker refused an agent-side access (credential or verifier)."""

    def __init__(self, kind: str, detail: str) -> None:
        super().__init__(f"{kind}: {detail}")
        self.kind = kind
        self.detail = detail


def opencode_binary() -> str | None:
    """Absolute path of the pinned OpenCode binary, or ``None`` when absent."""
    found = shutil.which(OPENCODE_BINARY_NAME)
    return found


def require_opencode_binary(purpose: str) -> str:
    """Return the binary path or raise with a ``blocked_external`` reason."""
    found = opencode_binary()
    if found is None:
        raise ContainerUnavailable(
            f"blocked_external: no {OPENCODE_BINARY_NAME!r} binary on PATH for {purpose}"
        )
    return found


def verify_pinned_version(actual: str) -> str:
    """Return ``actual`` when it equals the pin, else raise rather than drift."""
    if actual != PINNED_OPENCODE_VERSION:
        raise ValueError(
            f"opencode version {actual!r} does not match pinned "
            f"{PINNED_OPENCODE_VERSION!r}; refusing to run"
        )
    return actual


def pinned_invocation(
    *,
    workdir: Path | str,
    prompt_file: Path | str,
    model: str,
    opencode_version: str = PINNED_OPENCODE_VERSION,
) -> list[str]:
    """Build the single pinned OpenCode argv for one task.

    The version is verified before the argv is built so a drifting binary
    cannot produce a trace that claims the pinned shape.
    """
    verify_pinned_version(opencode_version)
    if not model:
        raise ValueError("model must be a non-empty OpenCode model id")
    binary = opencode_binary() or OPENCODE_BINARY_NAME
    return [
        binary,
        *OPENCODE_PINNED_ARGS,
        "--model",
        model,
        "--working-dir",
        str(workdir),
        "--prompt-file",
        str(prompt_file),
    ]


def live_check_status(purpose: str) -> dict[str, str | None]:
    """Report readiness for a live agent run without raising."""
    binary = opencode_binary()
    runtime = container_runtime()
    if binary is None:
        return {
            "state": "blocked_external",
            "runtime": None,
            "reason": f"no {OPENCODE_BINARY_NAME!r} binary on PATH for {purpose}",
        }
    if runtime is None:
        return {
            "state": "blocked_external",
            "runtime": None,
            "reason": f"no container runtime (docker/podman) on PATH for {purpose}",
        }
    return {"state": "ready", "runtime": runtime, "reason": None}


def agent_container_args(
    *,
    runtime: str,
    image: str,
    workdir: Path,
    command: Sequence[str],
    limits: SandboxLimits,
) -> list[str]:
    """Build the locked-down task-container argv for agent execution.

    The task workdir is mounted read-only; the verifier directory is
    deliberately not a parameter so hidden tests can never be mounted here.
    Network is denied at the container boundary; secrets never enter the
    argv. Callers needing strict containment must first call
    :func:`require_container_runtime`.
    """
    return container_exec_args(
        runtime=runtime,
        image=image,
        workdir=workdir,
        command=command,
        limits=limits,
    )


def agent_cost_usd(
    *,
    input_tokens: int,
    output_tokens: int,
    input_per_mtok_usd: float | None,
    output_per_mtok_usd: float | None,
) -> float | None:
    """Full-trajectory cost, or ``None`` when any needed price is unknown.

    ``None`` is not zero: an unpriced trajectory has unknown cost, which is
    a different statement from free.
    """
    if input_tokens < 0 or output_tokens < 0:
        raise ValueError("token counts must be non-negative")
    if input_per_mtok_usd is None or output_per_mtok_usd is None:
        return None
    if input_per_mtok_usd < 0 or output_per_mtok_usd < 0:
        raise ValueError("prices must not be negative")
    return (input_tokens / 1_000_000) * input_per_mtok_usd + (
        output_tokens / 1_000_000
    ) * output_per_mtok_usd


class ModelBroker:
    """Credential holder that keeps secrets outside the task container.

    The secret value lives only on this broker object (broker side). Task
    environments are built through :func:`candidate_env` and then asserted
    secret-free, so even a complete dump of the task env cannot leak the
    credential. The broker never logs the secret: ``repr`` is redacted.
    """

    def __init__(
        self,
        *,
        credential_ref: str | None = None,
        secret_value: str | None = None,
        input_per_mtok_usd: float | None = None,
        output_per_mtok_usd: float | None = None,
    ) -> None:
        if credential_ref is not None and not credential_ref:
            raise ValueError("credential_ref must be non-empty or None")
        if secret_value is not None and not secret_value:
            raise ValueError("secret_value must be non-empty or None")
        if (input_per_mtok_usd is not None and input_per_mtok_usd < 0) or (
            output_per_mtok_usd is not None and output_per_mtok_usd < 0
        ):
            raise ValueError("broker prices must not be negative")
        self._credential_ref: str | None = credential_ref
        self._secret_value: str | None = secret_value
        self._input_per_mtok_usd: float | None = input_per_mtok_usd
        self._output_per_mtok_usd: float | None = output_per_mtok_usd

    def __repr__(self) -> str:
        return (
            f"ModelBroker(credential_ref={self._credential_ref!r}, "
            "secret_value=<redacted>, "
            f"input_per_mtok_usd={self._input_per_mtok_usd!r}, "
            f"output_per_mtok_usd={self._output_per_mtok_usd!r})"
        )

    @property
    def credential_ref(self) -> str | None:
        return self._credential_ref

    @property
    def input_per_mtok_usd(self) -> float | None:
        return self._input_per_mtok_usd

    @property
    def output_per_mtok_usd(self) -> float | None:
        return self._output_per_mtok_usd

    def known_secret_values(self) -> tuple[str, ...]:
        """Secret values the task env must never contain (broker side only)."""
        if self._secret_value is None:
            return ()
        return (self._secret_value,)

    def task_env(self, workdir: Path, extra: Mapping[str, str] | None = None) -> dict[str, str]:
        """Build the scrubbed task-container environment.

        Raises :class:`PolicyDenied` if ``extra`` tries to mount a secret,
        and asserts the built env carries no known secret value.
        """
        env = candidate_env(workdir, extra)
        assert_no_secret_in_env(env, list(self.known_secret_values()))
        for value in self.known_secret_values():
            for key, entry in env.items():
                if value in entry:
                    raise PolicyDenied(
                        "secret-mounted",
                        f"broker secret present in task variable {key!r}",
                    )
        return env

    def check_tool_allowed(
        self,
        tool_name: str,
        tool_args: Mapping[str, Any],
        *,
        known_secrets: Sequence[str] = (),
    ) -> None:
        """Refuse tools that reach for credentials or the verifier.

        Tool names/args naming a secret, a proxy, an evaluator pointer or a
        verifier path are broker restrictions, not payloads: they raise
        instead of executing.
        """
        haystack = f"{tool_name} {dict(tool_args)}"
        lowered = haystack.lower()
        if "verifier" in lowered or "evaluator" in lowered or "gold" in lowered:
            raise BrokerDenied(
                "verifier-access",
                f"tool {tool_name!r} targets evaluator assets; denied",
            )
        secrets = (*self.known_secret_values(), *known_secrets)
        for secret in secrets:
            if secret and secret in haystack:
                raise BrokerDenied(
                    "secret-access",
                    f"tool {tool_name!r} carries a credential value; denied",
                )
        for key in tool_args:
            upper = str(key).upper()
            if any(
                fragment in upper
                for fragment in (
                    "API_KEY",
                    "TOKEN",
                    "SECRET",
                    "PASSWORD",
                    "CREDENTIAL",
                )
            ):
                raise BrokerDenied(
                    "secret-access",
                    f"tool {tool_name!r} names credential-bearing key {key!r}; denied",
                )


@dataclass(frozen=True, slots=True)
class AgentLimits:
    """Caps for one complete agent trajectory."""

    max_steps: int = 20
    max_tool_calls: int = 40
    tool_timeout_seconds: float = 30.0
    max_wall_seconds: float = 600.0
    max_input_tokens: int = 100_000
    max_output_tokens: int = 20_000
    max_total_cost_usd: float | None = None

    def __post_init__(self) -> None:
        if self.max_steps <= 0:
            raise ValueError(f"max_steps must be positive, got {self.max_steps}")
        if self.max_tool_calls <= 0:
            raise ValueError(f"max_tool_calls must be positive, got {self.max_tool_calls}")
        if self.tool_timeout_seconds <= 0:
            raise ValueError(
                f"tool_timeout_seconds must be positive, got {self.tool_timeout_seconds}"
            )
        if self.max_wall_seconds <= 0:
            raise ValueError(f"max_wall_seconds must be positive, got {self.max_wall_seconds}")
        if self.max_input_tokens <= 0 or self.max_output_tokens <= 0:
            raise ValueError("token caps must be positive")
        if self.max_total_cost_usd is not None and self.max_total_cost_usd <= 0:
            raise ValueError("max_total_cost_usd must be positive or None")


@dataclass(frozen=True, slots=True)
class CompactionEvent:
    """One context-compaction turn, recorded for cost and attribution."""

    step_index: int
    tokens_before: int
    tokens_after: int
    reason: str = "context window pressure"


@dataclass(frozen=True, slots=True)
class AuxCall:
    """One auxiliary model call inside the trajectory (judge, reranker, ...)."""

    step_index: int
    model: str
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True, slots=True)
class AgentTraceEvent:
    """One recorded step for replay and attribution."""

    step_index: int
    kind: str
    detail: str


@dataclass(frozen=True, slots=True)
class AgentAttribution:
    """Who/what produced the trajectory: prompt, compaction and aux usage."""

    prompt_id: str
    model: str
    opencode_version: str
    compaction_count: int
    aux_calls: tuple[AuxCall, ...]
    input_tokens: int
    output_tokens: int
    cost_usd: float | None


@dataclass(frozen=True, slots=True)
class AgentResult:
    """One complete trajectory: exactly one attempt."""

    attempt_id: str
    task_id: str
    status: AgentStatus
    steps_taken: int
    tool_calls: int
    tool_timeouts: int
    blocked_access_attempts: int
    input_tokens: int
    output_tokens: int
    cost_usd: float | None
    prompt_id: str
    model: str
    opencode_version: str
    compaction_events: tuple[CompactionEvent, ...]
    aux_calls: tuple[AuxCall, ...]
    patch: str | None
    reason: str | None
    duration_seconds: float
    workdir_id: str
    invocation: tuple[str, ...]

    @property
    def is_success(self) -> bool:
        return self.status is AgentStatus.SUCCESS

    def attribution(self) -> AgentAttribution:
        """Attribution metadata folding prompt, compaction and aux usage."""
        return AgentAttribution(
            prompt_id=self.prompt_id,
            model=self.model,
            opencode_version=self.opencode_version,
            compaction_count=len(self.compaction_events),
            aux_calls=self.aux_calls,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            cost_usd=self.cost_usd,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempt_id": self.attempt_id,
            "task_id": self.task_id,
            "status": str(self.status),
            "steps_taken": self.steps_taken,
            "tool_calls": self.tool_calls,
            "tool_timeouts": self.tool_timeouts,
            "blocked_access_attempts": self.blocked_access_attempts,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": self.cost_usd,
            "prompt_id": self.prompt_id,
            "model": self.model,
            "opencode_version": self.opencode_version,
            "compaction_events": [
                {
                    "step_index": event.step_index,
                    "tokens_before": event.tokens_before,
                    "tokens_after": event.tokens_after,
                    "reason": event.reason,
                }
                for event in self.compaction_events
            ],
            "aux_calls": [
                {
                    "step_index": call.step_index,
                    "model": call.model,
                    "input_tokens": call.input_tokens,
                    "output_tokens": call.output_tokens,
                }
                for call in self.aux_calls
            ],
            "patch": self.patch,
            "reason": self.reason,
            "duration_seconds": self.duration_seconds,
            "workdir_id": self.workdir_id,
            "invocation": list(self.invocation),
        }


@dataclass(frozen=True, slots=True)
class StepContext:
    """Read-only view one fixture step function observes."""

    step_index: int
    attempt_id: str
    task_id: str
    workdir: Path
    task_env_keys: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class StepOutcome:
    """One fixture step decision. Validated per kind."""

    kind: AgentActionKind
    tool_name: str | None = None
    tool_args: dict[str, Any] | None = None
    simulated_tool_duration_seconds: float | None = None
    patch: str | None = None
    fail_reason: str | None = None
    aux_model: str | None = None
    aux_input_tokens: int | None = None
    aux_output_tokens: int | None = None
    compaction_tokens_before: int | None = None
    compaction_tokens_after: int | None = None
    secret_name: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0

    def __post_init__(self) -> None:
        if self.input_tokens < 0 or self.output_tokens < 0:
            raise ValueError("step token counts must be non-negative")
        if self.simulated_tool_duration_seconds is not None and (
            self.simulated_tool_duration_seconds < 0
        ):
            raise ValueError("simulated tool duration must be non-negative")
        if self.kind is AgentActionKind.TOOL_CALL and not self.tool_name:
            raise ValueError("TOOL_CALL requires tool_name")
        if self.kind is AgentActionKind.FINISH and not self.patch:
            raise ValueError("FINISH requires a non-empty patch")
        if self.kind is AgentActionKind.FAIL and not self.fail_reason:
            raise ValueError("FAIL requires fail_reason")
        if self.kind is AgentActionKind.AUX_CALL:
            if not self.aux_model:
                raise ValueError("AUX_CALL requires aux_model")
            if self.aux_input_tokens is None or self.aux_output_tokens is None:
                raise ValueError("AUX_CALL requires aux token counts")
            if self.aux_input_tokens < 0 or self.aux_output_tokens < 0:
                raise ValueError("aux token counts must be non-negative")
        if self.kind is AgentActionKind.COMPACTION and (
            self.compaction_tokens_before is None or self.compaction_tokens_after is None
        ):
            raise ValueError("COMPACTION requires tokens_before/after")
        if self.kind is AgentActionKind.SECRET_READ_ATTEMPT and not self.secret_name:
            raise ValueError("SECRET_READ_ATTEMPT requires secret_name")


StepFn = Callable[[StepContext], StepOutcome]
ToolFn = Callable[[str, Mapping[str, Any]], str]


def _execute_tool(
    tool_fn: ToolFn | None,
    tool_name: str,
    tool_args: Mapping[str, Any],
    *,
    simulated_duration: float | None,
    timeout_seconds: float,
) -> tuple[str | None, float, bool]:
    """Execute one tool call, returning (output, duration, timed_out).

    Fixture steps may declare ``simulated_duration`` so timeout tests stay
    fast and deterministic: the duration is taken as measured without
    executing anything. Otherwise the injected ``tool_fn`` runs under a
    wall-clock measurement with a thread-pool timeout; a timeout is a tool
    failure, never a crash.
    """
    if simulated_duration is not None:
        return None, simulated_duration, simulated_duration > timeout_seconds
    if tool_fn is None:
        raise ValueError(f"no tool_fn for tool {tool_name!r}; refusing to invent output")
    executor = ThreadPoolExecutor(max_workers=1)
    started = time.monotonic()
    future: Future[str] = executor.submit(tool_fn, tool_name, tool_args)
    try:
        output = future.result(timeout=timeout_seconds)
        duration = time.monotonic() - started
        return output, duration, False
    except TimeoutError:
        duration = time.monotonic() - started
        return None, duration, True
    except Exception as exc:
        duration = time.monotonic() - started
        raise RuntimeError(f"tool {tool_name!r} failed: {exc}") from exc
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def run_agent_task(
    *,
    task_id: str,
    attempt_id: str,
    prompt_id: str,
    model: str,
    limits: AgentLimits | None = None,
    broker: ModelBroker | None = None,
    step_fn: StepFn,
    tool_fn: ToolFn | None = None,
    initial_files: Mapping[str, str | bytes] | None = None,
    verifier_files: Mapping[str, str | bytes] | None = None,
    known_secret_values: Sequence[str] = (),
    should_continue: Callable[[], bool] | None = None,
    workdir_parent: Path | str | None = None,
    opencode_version: str = PINNED_OPENCODE_VERSION,
) -> AgentResult:
    """Run one complete agent trajectory in a fresh disposable task root.

    Each call creates its own temporary root with sibling ``work/`` and
    ``verifier/`` directories, stages ``initial_files`` into ``work/`` and
    ``verifier_files`` into ``verifier/``, builds a scrubbed task env via
    the broker, then steps until finish, failure, a cap, cancellation or
    step exhaustion. The root is always removed, including on cancellation
    or step-function errors.
    """
    if not task_id:
        raise ValueError("task_id must be non-empty")
    if not attempt_id:
        raise ValueError("attempt_id must be non-empty")
    if not prompt_id:
        raise ValueError("prompt_id must be non-empty")
    if not model:
        raise ValueError("model must be non-empty")
    verify_pinned_version(opencode_version)
    active = limits or AgentLimits()
    active_broker = broker or ModelBroker()
    secrets: list[str] = [
        *active_broker.known_secret_values(),
        *known_secret_values,
    ]
    started = time.monotonic()
    workdir_id = uuid.uuid4().hex
    steps_taken = 0
    tool_calls = 0
    tool_timeouts = 0
    blocked_access_attempts = 0
    total_input = 0
    total_output = 0
    compactions: list[CompactionEvent] = []
    aux_calls: list[AuxCall] = []
    trace: list[AgentTraceEvent] = []  # Kept for debugging; summarized in reason.
    _ = trace

    parent = str(workdir_parent) if workdir_parent is not None else None
    with tempfile.TemporaryDirectory(prefix="stealthbench-agent-", dir=parent) as tmp:
        root = Path(tmp) / f"task-{workdir_id}"
        layout = EvaluatorLayout.create(root)
        assert_verifier_separation(layout.workdir, layout.verifier_dir)
        from stealthbench.sandbox.policy import write_candidate_file as _stage

        for name, content in (initial_files or {}).items():
            _stage(layout.workdir, name, content)
        for name, content in (verifier_files or {}).items():
            target = layout.verifier_dir / name
            if ".." in Path(name).parts or Path(name).is_absolute():
                raise PolicyDenied("traversal", f"verifier path escapes: {name!r}")
            target.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(content, bytes):
                target.write_bytes(content)
            else:
                target.write_text(content, encoding="utf-8")
        task_env = active_broker.task_env(layout.workdir)
        assert_no_secret_in_env(task_env, secrets)
        invocation = pinned_invocation(
            workdir=layout.workdir,
            prompt_file=layout.workdir / "prompt.md",
            model=model,
            opencode_version=opencode_version,
        )

        def _cost() -> float | None:
            return agent_cost_usd(
                input_tokens=total_input,
                output_tokens=total_output,
                input_per_mtok_usd=active_broker.input_per_mtok_usd,
                output_per_mtok_usd=active_broker.output_per_mtok_usd,
            )

        def _budget_hit() -> str | None:
            if total_input > active.max_input_tokens:
                return f"input tokens {total_input} exceed cap {active.max_input_tokens}"
            if total_output > active.max_output_tokens:
                return f"output tokens {total_output} exceed cap {active.max_output_tokens}"
            cost = _cost()
            if (
                active.max_total_cost_usd is not None
                and cost is not None
                and cost > active.max_total_cost_usd
            ):
                return f"cost {cost} exceeds cap {active.max_total_cost_usd}"
            return None

        def _finish(
            status: AgentStatus,
            *,
            patch: str | None,
            reason: str | None,
        ) -> AgentResult:
            duration = time.monotonic() - started
            return AgentResult(
                attempt_id=attempt_id,
                task_id=task_id,
                status=status,
                steps_taken=steps_taken,
                tool_calls=tool_calls,
                tool_timeouts=tool_timeouts,
                blocked_access_attempts=blocked_access_attempts,
                input_tokens=total_input,
                output_tokens=total_output,
                cost_usd=_cost(),
                prompt_id=prompt_id,
                model=model,
                opencode_version=opencode_version,
                compaction_events=tuple(compactions),
                aux_calls=tuple(aux_calls),
                patch=patch,
                reason=reason,
                duration_seconds=duration,
                workdir_id=workdir_id,
                invocation=tuple(invocation),
            )

        while True:
            if should_continue is not None and not should_continue():
                return _finish(AgentStatus.CANCELLED, patch=None, reason="cancelled")
            elapsed = time.monotonic() - started
            if elapsed > active.max_wall_seconds:
                return _finish(
                    AgentStatus.TIMEOUT,
                    patch=None,
                    reason=f"wall clock {active.max_wall_seconds}s exceeded",
                )
            hit = _budget_hit()
            if hit is not None:
                return _finish(AgentStatus.BUDGET_EXHAUSTED, patch=None, reason=hit)
            if steps_taken >= active.max_steps:
                return _finish(
                    AgentStatus.STEP_EXHAUSTED,
                    patch=None,
                    reason=f"step budget {active.max_steps} exhausted",
                )
            context = StepContext(
                step_index=steps_taken,
                attempt_id=attempt_id,
                task_id=task_id,
                workdir=layout.workdir,
                task_env_keys=tuple(sorted(task_env)),
            )
            try:
                outcome = step_fn(context)
            except (PolicyDenied, BrokerDenied) as exc:
                blocked_access_attempts += 1
                return _finish(
                    AgentStatus.FAILED,
                    patch=None,
                    reason=f"blocked access denied: {exc}",
                )
            total_input += outcome.input_tokens
            total_output += outcome.output_tokens
            current_index = steps_taken
            steps_taken += 1

            if outcome.kind is AgentActionKind.FINISH:
                return _finish(AgentStatus.SUCCESS, patch=outcome.patch, reason="finished")
            if outcome.kind is AgentActionKind.FAIL:
                return _finish(AgentStatus.FAILED, patch=None, reason=outcome.fail_reason)
            if outcome.kind is AgentActionKind.COMPACTION:
                assert outcome.compaction_tokens_before is not None
                assert outcome.compaction_tokens_after is not None
                compactions.append(
                    CompactionEvent(
                        step_index=current_index,
                        tokens_before=outcome.compaction_tokens_before,
                        tokens_after=outcome.compaction_tokens_after,
                    )
                )
                continue
            if outcome.kind is AgentActionKind.AUX_CALL:
                assert outcome.aux_model is not None
                assert outcome.aux_input_tokens is not None
                assert outcome.aux_output_tokens is not None
                aux_calls.append(
                    AuxCall(
                        step_index=current_index,
                        model=outcome.aux_model,
                        input_tokens=outcome.aux_input_tokens,
                        output_tokens=outcome.aux_output_tokens,
                    )
                )
                total_input += outcome.aux_input_tokens
                total_output += outcome.aux_output_tokens
                continue
            if outcome.kind is AgentActionKind.SECRET_READ_ATTEMPT:
                blocked_access_attempts += 1
                name = outcome.secret_name or "<unknown>"
                if name in task_env:
                    return _finish(
                        AgentStatus.FAILED,
                        patch=None,
                        reason=f"secret {name!r} visible in task env",
                    )
                return _finish(
                    AgentStatus.FAILED,
                    patch=None,
                    reason=f"secret access blocked: {name!r} not in task env",
                )
            if outcome.kind is not AgentActionKind.TOOL_CALL:
                return _finish(  # type: ignore[unreachable]
                    AgentStatus.FAILED, patch=None, reason=f"unknown action {outcome.kind}"
                )
            assert outcome.tool_name is not None
            tool_args: Mapping[str, Any] = outcome.tool_args or {}
            try:
                active_broker.check_tool_allowed(
                    outcome.tool_name, tool_args, known_secrets=secrets
                )
            except (PolicyDenied, BrokerDenied) as exc:
                blocked_access_attempts += 1
                return _finish(
                    AgentStatus.FAILED,
                    patch=None,
                    reason=f"blocked tool denied: {exc}",
                )
            if tool_calls >= active.max_tool_calls:
                return _finish(
                    AgentStatus.BUDGET_EXHAUSTED,
                    patch=None,
                    reason=(f"tool budget {active.max_tool_calls} exhausted"),
                )
            tool_calls += 1
            try:
                _, duration_s, timed_out = _execute_tool(
                    tool_fn,
                    outcome.tool_name,
                    tool_args,
                    simulated_duration=outcome.simulated_tool_duration_seconds,
                    timeout_seconds=active.tool_timeout_seconds,
                )
            except RuntimeError as exc:
                return _finish(AgentStatus.FAILED, patch=None, reason=str(exc))
            if timed_out:
                tool_timeouts += 1
                continue
            _ = duration_s
            continue
