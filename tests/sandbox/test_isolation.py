"""Sandbox isolation and evaluator separation (task T07B, gate G07).

Policy checks (traversal, symlink escape, secret scrub, verifier overlap)
run everywhere because they are pure path/env logic. Strict containment of
host files and the network needs a real container, so those tests skip with
``blocked_external`` when no runtime or image is available; they never
fake-pass by asserting on the best-effort subprocess backend.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from stealthbench.sandbox import policy, runtime
from stealthbench.sandbox.policy import (
    EvaluatorLayout,
    PolicyDenied,
    assert_no_secret_in_env,
    candidate_env,
    candidate_visible_paths,
    network_policy,
    resolve_candidate_path,
    scrub_env,
    write_candidate_file,
)
from stealthbench.sandbox.runtime import (
    SandboxLimits,
    container_exec_args,
    isolated_workdir,
    run_in_sandbox,
    run_python_source,
)

pytestmark = pytest.mark.sandbox

HOST_CANARY = "stealthbench-host-secret-canary-7b3f"
EVALUATOR_CANARY = "stealthbench-evaluator-secret-canary-9d21"


def test_path_traversal_denied() -> None:
    with isolated_workdir() as workdir:
        for name in ("../evil.py", "a/../../evil.py", "/absolute.py", ".."):
            with pytest.raises(PolicyDenied):
                resolve_candidate_path(workdir, name)
            with pytest.raises(PolicyDenied):
                write_candidate_file(workdir, name, "print('evil')\n")
        with pytest.raises(PolicyDenied):
            run_in_sandbox(
                [sys.executable, "candidate.py"],
                files={"../evil.py": "print('evil')\n"},
                limits=SandboxLimits(wall_seconds=5.0),
            )


def test_symlink_escape_denied(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text(HOST_CANARY, encoding="utf-8")
    with isolated_workdir() as workdir:
        (workdir / "link").symlink_to(outside, target_is_directory=True)
        with pytest.raises(PolicyDenied):
            resolve_candidate_path(workdir, "link/secret.txt")
        with pytest.raises(PolicyDenied):
            write_candidate_file(workdir, "link/payload.py", "print('evil')\n")


def test_host_secret_not_visible_to_candidate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STEALTHBENCH_HOST_SECRET_CANARY", HOST_CANARY)
    monkeypatch.setenv("PROVIDER_API_KEY", "super-secret-key")
    with isolated_workdir() as workdir:
        env = candidate_env(workdir)
        assert "STEALTHBENCH_HOST_SECRET_CANARY" not in env
        assert "PROVIDER_API_KEY" not in env
        assert_no_secret_in_env(env, [HOST_CANARY, "super-secret-key"])
        result = run_python_source(
            "import os\nprint(sorted(os.environ))\n",
            limits=SandboxLimits(wall_seconds=10.0),
        )
        assert result.stdout and HOST_CANARY not in result.stdout
        assert "super-secret-key" not in result.stdout
        assert "PROVIDER_API_KEY" not in result.stdout


def test_extra_env_refuses_secret_mount() -> None:
    with isolated_workdir() as workdir:
        with pytest.raises(PolicyDenied):
            candidate_env(workdir, {"EVAL_API_TOKEN": "nope"})
        # Proxies and evaluator pointers are posture, not payloads: dropped,
        # never forwarded, and never an error.
        assert "HTTP_PROXY" not in candidate_env(workdir, {"HTTP_PROXY": "http://x"})
        assert "VERIFIER_DIR" not in candidate_env(workdir, {"VERIFIER_DIR": "/v"})


def test_verifier_lives_in_a_separate_directory(tmp_path: Path) -> None:
    layout = EvaluatorLayout.create(tmp_path / "eval")
    assert layout.verifier_dir != layout.workdir
    assert layout.verifier_dir.parent == layout.workdir.parent
    verifier_secret = layout.verifier_dir / "hidden_tests.json"
    verifier_secret.write_text(EVALUATOR_CANARY, encoding="utf-8")
    write_candidate_file(layout.workdir, "solution.py", "print('solved')\n")

    visible = candidate_visible_paths(layout.workdir)
    assert "solution.py" in visible
    assert not any("hidden_tests" in name for name in visible)
    assert not any("verifier" in name for name in visible)

    with isolated_workdir() as _sandbox_dir:
        env = candidate_env(layout.workdir)
        assert str(layout.verifier_dir) not in env.values()
        assert not [key for key in env if key.startswith(("VERIFIER_", "EVALUATOR_"))]
        assert EVALUATOR_CANARY not in " ".join(env.values())

    result = run_in_sandbox(
        [sys.executable, "solution.py"],
        files={"solution.py": "import os\nprint(os.listdir('.'))\n"},
        limits=SandboxLimits(wall_seconds=10.0),
    )
    assert "hidden_tests" not in result.stdout
    assert EVALUATOR_CANARY not in result.stdout


def test_verifier_overlap_rejected(tmp_path: Path) -> None:
    with pytest.raises(PolicyDenied):
        policy.assert_verifier_separation(tmp_path, tmp_path)
    nested = tmp_path / "work" / "verifier"
    nested.mkdir(parents=True)
    with pytest.raises(PolicyDenied):
        policy.assert_verifier_separation(tmp_path / "work", nested)


def test_network_posture_denies_egress() -> None:
    posture = network_policy()
    assert posture["egress"] == "deny"
    with isolated_workdir() as workdir:
        env = candidate_env(workdir, {"HTTP_PROXY": "http://proxy.invalid:8080"})
        assert "HTTP_PROXY" not in env
        scrubbed = scrub_env(
            {"HTTP_PROXY": "x", "http_proxy": "x", "ALL_PROXY": "x", "PATH": "/usr/bin"}
        )
        assert "HTTP_PROXY" not in scrubbed
        assert "http_proxy" not in scrubbed
        assert "ALL_PROXY" not in scrubbed


def test_container_builder_never_mounts_the_verifier(tmp_path: Path) -> None:
    args = container_exec_args(
        runtime="docker",
        image="python:3.12-slim",
        workdir=tmp_path,
        command=["python", "/work/solution.py"],
        limits=SandboxLimits(wall_seconds=6.0, memory_bytes=256 * 1024 * 1024),
    )
    assert "--network" in args and "none" in args
    assert f"{tmp_path}:/work:ro" in args
    assert "verifier" not in " ".join(args).lower()


_CONTAINER_IMAGE = "python:3.12-slim"


def test_container_denies_network_and_host_files(tmp_path: Path) -> None:
    """Strict containment on the real runtime; skip (blocked_external) without one."""
    try:
        name = runtime.require_container_runtime("strict network/host containment")
    except runtime.ContainerUnavailable as exc:
        pytest.skip(str(exc))
    inspect = subprocess.run(
        [name, "image", "inspect", _CONTAINER_IMAGE],
        capture_output=True,
        timeout=60,
    )
    if inspect.returncode != 0:
        pytest.skip(
            f"blocked_external: container image {_CONTAINER_IMAGE} unavailable "
            "for the strict containment check"
        )
    with isolated_workdir(tmp_path) as workdir:
        write_candidate_file(
            workdir,
            "probe.py",
            "import socket\n"
            "socket.create_connection(('8.8.8.8', 53), timeout=5)\n"
            "print('egressed')\n",
        )
        limits = SandboxLimits(wall_seconds=30.0)
        args = container_exec_args(
            runtime=name,
            image=_CONTAINER_IMAGE,
            workdir=workdir,
            command=["python", "/work/probe.py"],
            limits=limits,
        )
        try:
            proc = subprocess.run(args, capture_output=True, timeout=60)
        except (OSError, subprocess.SubprocessError) as exc:
            pytest.skip(f"blocked_external: container daemon unavailable: {exc}")
        assert proc.returncode != 0
        assert b"egressed" not in proc.stdout
