"""Sandbox tests for the pinned OpenCode runner and model broker (T10A, G10).

Acceptance for T10A: full-trajectory budgets, prompts, compaction,
auxiliary usage and cancellation are captured; task containers cannot read
credentials.

All stepping is fixture-driven (injected ``step_fn`` / ``tool_fn``); the
real ``opencode`` binary is never spawned. Tests needing a container
runtime skip as ``blocked_external`` when absent instead of fake-passing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from stealthbench.agents.opencode import (
    OPENCODE_BINARY_NAME,
    PINNED_OPENCODE_VERSION,
    AgentActionKind,
    AgentLimits,
    AgentStatus,
    ModelBroker,
    StepContext,
    StepOutcome,
    agent_container_args,
    live_check_status,
    pinned_invocation,
    run_agent_task,
    verify_pinned_version,
)
from stealthbench.sandbox.policy import PolicyDenied
from stealthbench.sandbox.runtime import (
    SandboxLimits,
    container_runtime,
    require_container_runtime,
)

pytestmark = pytest.mark.sandbox

BROKER_SECRET = "broker-secret-canary-t10a-4f21"
HOST_SECRET = "host-secret-canary-t10a-9c33"


def _broker(**overrides: Any) -> ModelBroker:
    params: dict[str, Any] = {
        "credential_ref": "STEALTHBENCH_BROKER_TOKEN",
        "secret_value": BROKER_SECRET,
        "input_per_mtok_usd": 2.0,
        "output_per_mtok_usd": 8.0,
    }
    params.update(overrides)
    return ModelBroker(**params)


def _finish_patch(patch: str = "diff --git a/f b/f\n+fix\n") -> StepOutcome:
    return StepOutcome(kind=AgentActionKind.FINISH, patch=patch, input_tokens=10, output_tokens=5)


def test_fixture_success_captures_prompt_and_attribution() -> None:
    broker = _broker()
    calls = {"n": 0}

    def step_fn(_ctx: StepContext) -> StepOutcome:
        calls["n"] += 1
        return _finish_patch()

    result = run_agent_task(
        task_id="swe-task-001",
        attempt_id="attempt-1",
        prompt_id="swe-prompt-v1",
        model="zen-test-model",
        limits=AgentLimits(max_steps=5, max_wall_seconds=30.0),
        broker=broker,
        step_fn=step_fn,
    )
    assert result.status is AgentStatus.SUCCESS
    assert result.patch is not None and "fix" in result.patch
    assert result.prompt_id == "swe-prompt-v1"
    assert result.model == "zen-test-model"
    assert result.opencode_version == PINNED_OPENCODE_VERSION
    assert result.steps_taken == 1
    assert result.input_tokens == 10
    assert result.output_tokens == 5
    # Cost folds the full trajectory at the broker prices.
    assert result.cost_usd == pytest.approx((10 / 1e6) * 2.0 + (5 / 1e6) * 8.0)
    attribution = result.attribution()
    assert attribution.prompt_id == "swe-prompt-v1"
    assert attribution.opencode_version == PINNED_OPENCODE_VERSION
    assert attribution.compaction_count == 0
    assert "opencode" in result.invocation[0]
    assert "--format" in result.invocation


def test_fixture_failure_reports_reason() -> None:
    def step_fn(_ctx: StepContext) -> StepOutcome:
        return StepOutcome(kind=AgentActionKind.FAIL, fail_reason="tests still red")

    result = run_agent_task(
        task_id="t-fail",
        attempt_id="a-1",
        prompt_id="p-v1",
        model="m",
        broker=_broker(),
        step_fn=step_fn,
    )
    assert result.status is AgentStatus.FAILED
    assert result.reason == "tests still red"
    assert result.patch is None
    assert not result.is_success


def test_tool_timeout_is_a_tool_failure_not_a_crash() -> None:
    def step_fn(ctx: StepContext) -> StepOutcome:
        if ctx.step_index == 0:
            return StepOutcome(
                kind=AgentActionKind.TOOL_CALL,
                tool_name="bash",
                tool_args={"cmd": "sleep 60"},
                simulated_tool_duration_seconds=60.0,
                input_tokens=5,
                output_tokens=5,
            )
        return _finish_patch()

    result = run_agent_task(
        task_id="t-timeout",
        attempt_id="a-1",
        prompt_id="p-v1",
        model="m",
        limits=AgentLimits(max_steps=5, tool_timeout_seconds=1.0, max_wall_seconds=30.0),
        broker=_broker(),
        step_fn=step_fn,
    )
    assert result.status is AgentStatus.SUCCESS
    assert result.tool_calls == 1
    assert result.tool_timeouts == 1


def test_step_exhaustion_reports_budget() -> None:
    def step_fn(_ctx: StepContext) -> StepOutcome:
        return StepOutcome(
            kind=AgentActionKind.TOOL_CALL,
            tool_name="bash",
            tool_args={"cmd": "echo hi"},
            simulated_tool_duration_seconds=0.01,
        )

    result = run_agent_task(
        task_id="t-steps",
        attempt_id="a-1",
        prompt_id="p-v1",
        model="m",
        limits=AgentLimits(max_steps=2, max_wall_seconds=30.0),
        broker=_broker(),
        step_fn=step_fn,
        tool_fn=lambda _name, _args: "hi",
    )
    assert result.status is AgentStatus.STEP_EXHAUSTED
    assert result.steps_taken == 2
    assert "step budget" in (result.reason or "")


def test_tool_budget_exhaustion_is_distinct_from_steps() -> None:
    def step_fn(_ctx: StepContext) -> StepOutcome:
        return StepOutcome(
            kind=AgentActionKind.TOOL_CALL,
            tool_name="bash",
            tool_args={"cmd": "echo hi"},
            simulated_tool_duration_seconds=0.01,
        )

    result = run_agent_task(
        task_id="t-tools",
        attempt_id="a-1",
        prompt_id="p-v1",
        model="m",
        limits=AgentLimits(max_steps=10, max_tool_calls=1, max_wall_seconds=30.0),
        broker=_broker(),
        step_fn=step_fn,
    )
    assert result.status is AgentStatus.BUDGET_EXHAUSTED
    assert "tool budget" in (result.reason or "")


def test_compaction_is_captured_in_attribution() -> None:
    def step_fn(ctx: StepContext) -> StepOutcome:
        if ctx.step_index == 0:
            return StepOutcome(
                kind=AgentActionKind.COMPACTION,
                compaction_tokens_before=90000,
                compaction_tokens_after=12000,
                input_tokens=100,
                output_tokens=50,
            )
        return _finish_patch()

    result = run_agent_task(
        task_id="t-compact",
        attempt_id="a-1",
        prompt_id="prompt-with-compaction-v3",
        model="m",
        broker=_broker(),
        step_fn=step_fn,
    )
    assert result.status is AgentStatus.SUCCESS
    assert len(result.compaction_events) == 1
    event = result.compaction_events[0]
    assert (event.tokens_before, event.tokens_after) == (90000, 12000)
    assert result.attribution().compaction_count == 1
    assert result.attribution().prompt_id == "prompt-with-compaction-v3"


def test_aux_usage_folds_into_cost_and_attribution() -> None:
    def step_fn(ctx: StepContext) -> StepOutcome:
        if ctx.step_index == 0:
            return StepOutcome(
                kind=AgentActionKind.AUX_CALL,
                aux_model="aux-judge-v1",
                aux_input_tokens=1000,
                aux_output_tokens=200,
            )
        return _finish_patch()

    broker = _broker(input_per_mtok_usd=1.0, output_per_mtok_usd=4.0)
    result = run_agent_task(
        task_id="t-aux",
        attempt_id="a-1",
        prompt_id="p-v1",
        model="m",
        broker=broker,
        step_fn=step_fn,
    )
    assert result.status is AgentStatus.SUCCESS
    assert len(result.aux_calls) == 1
    assert result.aux_calls[0].model == "aux-judge-v1"
    # Step tokens (10 in / 5 out from the finish) plus aux tokens.
    assert result.input_tokens == 1010
    assert result.output_tokens == 205
    assert result.cost_usd == pytest.approx((1010 / 1e6) * 1.0 + (205 / 1e6) * 4.0)
    assert result.attribution().aux_calls == result.aux_calls


def test_unknown_price_stays_null_never_zero() -> None:
    broker = _broker(input_per_mtok_usd=None, output_per_mtok_usd=None)

    def step_fn(_ctx: StepContext) -> StepOutcome:
        return _finish_patch()

    result = run_agent_task(
        task_id="t-noprice",
        attempt_id="a-1",
        prompt_id="p-v1",
        model="m",
        broker=broker,
        step_fn=step_fn,
    )
    assert result.cost_usd is None
    assert result.cost_usd != 0.0  # None is not a measured zero


def test_broker_keeps_credentials_outside_the_task_container(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("STEALTHBENCH_HOST_SECRET_CANARY", HOST_SECRET)
    broker = _broker()
    seen: dict[str, Any] = {}

    def step_fn(ctx: StepContext) -> StepOutcome:
        seen["keys"] = ctx.task_env_keys
        assert BROKER_SECRET not in str(ctx.task_env_keys)
        return _finish_patch()

    result = run_agent_task(
        task_id="t-broker",
        attempt_id="a-1",
        prompt_id="p-v1",
        model="m",
        broker=broker,
        step_fn=step_fn,
    )
    assert result.status is AgentStatus.SUCCESS
    assert "STEALTHBENCH_BROKER_TOKEN" not in (seen["keys"] or ())
    assert "PROVIDER_API_KEY" not in (seen["keys"] or ())
    # The broker object never renders its secret.
    assert BROKER_SECRET not in repr(broker)


def test_extra_env_refusing_secret_mount() -> None:
    broker = _broker()
    with pytest.raises(PolicyDenied):
        broker.task_env(Path("/tmp"), {"MY_API_TOKEN": "nope"})


def test_tool_naming_a_credential_is_denied() -> None:
    broker = _broker()

    def step_fn(_ctx: StepContext) -> StepOutcome:
        return StepOutcome(
            kind=AgentActionKind.TOOL_CALL,
            tool_name="read_env",
            tool_args={"key": "STEALTHBENCH_BROKER_TOKEN", "value": BROKER_SECRET},
        )

    result = run_agent_task(
        task_id="t-deny",
        attempt_id="a-1",
        prompt_id="p-v1",
        model="m",
        broker=broker,
        step_fn=step_fn,
    )
    assert result.status is AgentStatus.FAILED
    assert result.blocked_access_attempts == 1
    assert "denied" in (result.reason or "").lower()


def test_verifier_targeting_tool_is_denied() -> None:
    broker = _broker()

    def step_fn(_ctx: StepContext) -> StepOutcome:
        return StepOutcome(
            kind=AgentActionKind.TOOL_CALL,
            tool_name="read_file",
            tool_args={"path": "../verifier/hidden_tests.json"},
        )

    result = run_agent_task(
        task_id="t-verifier-deny",
        attempt_id="a-1",
        prompt_id="p-v1",
        model="m",
        broker=broker,
        step_fn=step_fn,
        verifier_files={"hidden_tests.json": "secret-tests"},
    )
    assert result.status is AgentStatus.FAILED
    assert result.blocked_access_attempts == 1


def test_secret_read_attempt_is_blocked_without_leak() -> None:
    broker = _broker()

    def step_fn(_ctx: StepContext) -> StepOutcome:
        return StepOutcome(kind=AgentActionKind.SECRET_READ_ATTEMPT, secret_name="BROKER_TOKEN")

    result = run_agent_task(
        task_id="t-secret",
        attempt_id="a-1",
        prompt_id="p-v1",
        model="m",
        broker=broker,
        step_fn=step_fn,
        known_secret_values=[BROKER_SECRET],
    )
    assert result.status is AgentStatus.FAILED
    assert result.blocked_access_attempts == 1
    assert BROKER_SECRET not in (result.reason or "")


def test_cancellation_cleans_up(tmp_path: Path) -> None:
    calls = {"n": 0}

    def should_continue() -> bool:
        calls["n"] += 1
        return calls["n"] < 3

    def step_fn(_ctx: StepContext) -> StepOutcome:
        return StepOutcome(
            kind=AgentActionKind.TOOL_CALL,
            tool_name="bash",
            tool_args={"cmd": "echo hi"},
            simulated_tool_duration_seconds=0.01,
        )

    result = run_agent_task(
        task_id="t-cancel",
        attempt_id="a-1",
        prompt_id="p-v1",
        model="m",
        broker=_broker(),
        step_fn=step_fn,
        should_continue=should_continue,
        workdir_parent=tmp_path,
    )
    assert result.status is AgentStatus.CANCELLED
    assert list(tmp_path.iterdir()) == []


def test_fresh_env_per_task_has_no_leakage() -> None:
    seen_workdirs: list[Path] = []

    def first_step(ctx: StepContext) -> StepOutcome:
        seen_workdirs.append(ctx.workdir)
        (ctx.workdir / "marker-from-first-task.txt").write_text("first", encoding="utf-8")
        return _finish_patch()

    def second_step(ctx: StepContext) -> StepOutcome:
        seen_workdirs.append(ctx.workdir)
        assert not (ctx.workdir / "marker-from-first-task.txt").exists()
        return _finish_patch()

    first = run_agent_task(
        task_id="t-fresh-1",
        attempt_id="a-1",
        prompt_id="p-v1",
        model="m",
        broker=_broker(),
        step_fn=first_step,
    )
    second = run_agent_task(
        task_id="t-fresh-2",
        attempt_id="a-2",
        prompt_id="p-v1",
        model="m",
        broker=_broker(),
        step_fn=second_step,
    )
    assert first.status is AgentStatus.SUCCESS
    assert second.status is AgentStatus.SUCCESS
    assert first.workdir_id != second.workdir_id
    assert seen_workdirs[0] != seen_workdirs[1]


def test_pinned_invocation_shape_and_version_gate(tmp_path: Path) -> None:
    argv = pinned_invocation(
        workdir=tmp_path, prompt_file=tmp_path / "prompt.md", model="zen-model"
    )
    assert argv[0].endswith(OPENCODE_BINARY_NAME) or argv[0] == OPENCODE_BINARY_NAME
    assert "--model" in argv and "zen-model" in argv
    assert "--format" in argv and "json" in argv
    with pytest.raises(ValueError, match="pinned"):
        verify_pinned_version("9.99.99")
    assert verify_pinned_version(PINNED_OPENCODE_VERSION) == PINNED_OPENCODE_VERSION
    with pytest.raises(ValueError, match="pinned"):
        pinned_invocation(
            workdir=tmp_path,
            prompt_file=tmp_path / "prompt.md",
            model="m",
            opencode_version="9.99.99",
        )


def test_agent_container_args_never_mount_the_verifier(tmp_path: Path) -> None:
    args = agent_container_args(
        runtime="docker",
        image="python:3.12-slim",
        workdir=tmp_path,
        command=["opencode", "run"],
        limits=SandboxLimits(wall_seconds=60.0),
    )
    assert "--network" in args and "none" in args
    assert f"{tmp_path}:/work:ro" in args
    assert "verifier" not in " ".join(args).lower()


def test_live_agent_run_reports_blocked_external_without_toolchain() -> None:
    if container_runtime() is not None and live_check_status("x")["state"] == "ready":
        pytest.skip("container runtime and opencode binary present; nothing missing")
    status = live_check_status("fixture agent run")
    assert status["state"] in {"ready", "blocked_external"}
    if container_runtime() is None:
        assert status["state"] == "blocked_external"
        assert status["reason"]


def test_container_only_agent_check_skips_without_runtime() -> None:
    try:
        name = require_container_runtime("agent strict containment")
    except Exception as exc:
        pytest.skip(str(exc))
    assert name in {"docker", "podman"}
