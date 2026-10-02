"""Build the G03 gate evidence record from real command output.

Every recorded exit status, count and tail comes from actually running the command;
nothing here is asserted without being measured. Writes to a new revision directory
and never modifies an existing one.

Run: .venv/bin/python implementation/mutation/build_g03_evidence.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = str(ROOT / ".venv" / "bin" / "python")
RUFF = str(ROOT / ".venv" / "bin" / "ruff")
MYPY = str(ROOT / ".venv" / "bin" / "mypy")

RECOMPUTE = (
    "find . -type f \\( -name '*.py' -o -name '*.toml' -o -name '*.md' -o -name '*.json' "
    "-o -name '*.yml' -o -name 'py.typed' -o -name 'constraints-dev.txt' \\) "
    "-not -path './.venv/*' -not -path '*/__pycache__/*' "
    "-not -path './implementation/evidence/*' -not -path './implementation/tasks.json' "
    "| sort | while read -r f; do printf '%s  %s\\n' \"$(sha256sum \"$f\" | cut -d' ' -f1)\" "
    '"${f#./}"; done | sha256sum | cut -d" " -f1'
)


def emit(line: str) -> None:
    sys.stderr.write(f"{line}\n")
    sys.stderr.flush()


def run(argv: list[str]) -> dict[str, object]:
    proc = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True, check=False)
    lines = [ln for ln in (proc.stdout + proc.stderr).splitlines() if ln.strip()]
    emit(f"  exit {proc.returncode}: {' '.join(argv[1:3])} {lines[-1] if lines else ''}")
    return {
        "command": " ".join(a.replace(f"{ROOT}/", "") for a in argv),
        "exit_status": proc.returncode,
        "stdout_tail": lines[-3:],
        "stderr_tail": [ln for ln in lines if ln.startswith(("ERROR", "error:"))][:3],
    }


def pytest_counts(argv: list[str]) -> dict[str, object]:
    record = run(argv)
    tail = "\n".join(str(x) for x in record["stdout_tail"])
    passed = failed = skipped = 0
    # Parse the final summary line, e.g. "688 passed in 17.77s".
    summary = tail.splitlines()[-1] if tail else ""
    parts = summary.replace(",", " ").split()
    for index, word in enumerate(parts):
        if word == "passed" and index:
            passed = int(parts[index - 1])
        if word == "failed" and index:
            failed = int(parts[index - 1])
        if word == "skipped" and index:
            skipped = int(parts[index - 1])
    record["stdout_tail"] = [summary] if summary else []
    return {
        **record,
        "_passed": passed,
        "_failed": failed,
        "_skipped": skipped,
    }


def collect_suite(path: str) -> dict[str, int]:
    proc = subprocess.run(
        [PY, "-m", "pytest", "--strict-markers", "--collect-only", "-q", path],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    for line in proc.stdout.splitlines():
        if "tests collected" in line or "test collected" in line:
            digits = "".join(c for c in line.split("test")[0] if c.isdigit())
            if digits:
                return {"collected": int(digits)}
    return {"collected": 0}


def mutation_results() -> tuple[list[dict[str, object]], int]:
    proc = subprocess.run(
        [PY, "implementation/mutation/g03_verify.py"], cwd=ROOT, capture_output=True, text=True
    )
    checks: list[dict[str, object]] = []
    for line in proc.stderr.splitlines():
        if " PASS  caught (" in line:
            ident, _, rest = line.partition(" PASS  caught (")
            checks.append({"mutation": ident.strip(), "defect": rest.rstrip(")"), "detected": True})
        elif "SURVIVED" in line:
            ident = line.split()[0]
            checks.append({"mutation": ident.strip(), "defect": "", "detected": False})
    if "restore PASS" not in proc.stderr:
        raise SystemExit("mutation run did not restore a green suite; refusing to record")
    return checks, sum(1 for c in checks if not c["detected"])


def main() -> int:
    lint = run([RUFF, "check", "."])
    fmt = run([RUFF, "format", "--check", "."])
    types = run([MYPY, "src/stealthbench"])
    unit_contract = pytest_counts(
        [PY, "-m", "pytest", "--strict-markers", "-q", "tests/unit", "tests/contract"]
    )
    integ_replay = pytest_counts(
        [PY, "-m", "pytest", "--strict-markers", "-q", "tests/integration", "tests/replay"]
    )
    full = pytest_counts(
        [
            PY,
            "-m",
            "pytest",
            "--strict-markers",
            "-q",
            "tests/unit",
            "tests/contract",
            "tests/integration",
            "tests/replay",
            "tests/sandbox",
            "tests/live",
        ]
    )
    cli_validate = run(
        [
            str(ROOT / ".venv" / "bin" / "stealthbench"),
            "manifest",
            "validate",
            "configs/offline-demo.json",
        ]
    )
    cli_run = run(
        [str(ROOT / ".venv" / "bin" / "stealthbench"), "run", "configs/offline-demo.json"]
    )

    mutations, undetected = mutation_results()
    revision_proc = subprocess.run(
        ["bash", "-c", RECOMPUTE], cwd=ROOT, capture_output=True, text=True, check=True
    )
    revision = revision_proc.stdout.strip()

    suites = {
        name: collect_suite(f"tests/{name}")
        for name in ("unit", "contract", "integration", "replay", "sandbox", "live")
    }
    total_collected = sum(v["collected"] for v in suites.values())

    record: dict[str, object] = {
        "schema_version": "1.0",
        "gate_id": "G03",
        "recorded_on": "2026-10-02",
        "source_revision": {
            "kind": "source_tree_sha256",
            "digest": revision,
            "recompute_command": RECOMPUTE,
        },
        "required_tasks": ["T03A", "T03B", "T03C"],
        "depends_on_gates": ["G00", "G01", "G02"],
        "prerequisite_evidence": [
            "implementation/evidence/G00/3491f1f114880f14db8979c82bd57a8c2c3814cd9f1e70bd46f4f1bbcd043594/gate.json",
            "implementation/evidence/G01/790d70f64a12e87b4d26d123c78c794c8e99816d897122616451d2f00281f006/gate.json",
            "implementation/evidence/G02/4079454a4cbd7149531c197dc3b171ec8401cc995d75498444c0261993eefd35/gate.json",
        ],
        "artifacts": [
            "src/stealthbench/adapters/base.py",
            "src/stealthbench/adapters/zen.py",
            "src/stealthbench/adapters/streaming.py",
            "tests/fixtures/zen/chat.completions.json",
            "tests/fixtures/zen/chat.completions.stream.sse",
            "tests/fixtures/zen/chat.completions.truncated.sse",
            "tests/fixtures/zen/chat.completions.server_error.sse",
        ],
        "commands": [lint, fmt, types, unit_contract, integ_replay, full, cli_validate, cli_run],
        "exit_statuses": {
            "lint": lint["exit_status"],
            "format": fmt["exit_status"],
            "typecheck": types["exit_status"],
            "unit_contract": unit_contract["exit_status"],
            "integration_replay": integ_replay["exit_status"],
            "full_offline_collection": full["exit_status"],
            "cli_validate": cli_validate["exit_status"],
            "cli_run": cli_run["exit_status"],
        },
        "counted_suites": [
            "tests/unit",
            "tests/contract",
            "tests/integration",
            "tests/replay",
            "tests/live",
        ],
        "vacuous_suites": {
            "tests/sandbox": "registered in docs/contracts.md suites_not_yet_started (T07A / G07)"
        },
        "suite_counts": {f"tests/{k}": v["collected"] for k, v in suites.items()},
        "collected": total_collected,
        "passed": full["_passed"],
        "failed": full["_failed"],
        "skipped": full["_skipped"],
        "gate_suite_totals": {
            "unit_contract": {
                "collected": unit_contract["_passed"]
                + unit_contract["_failed"]
                + unit_contract["_skipped"],
                "passed": unit_contract["_passed"],
                "failed": unit_contract["_failed"],
                "skipped": unit_contract["_skipped"],
            },
            "integration_replay": {
                "collected": integ_replay["_passed"]
                + integ_replay["_failed"]
                + integ_replay["_skipped"],
                "passed": integ_replay["_passed"],
                "failed": integ_replay["_failed"],
                "skipped": integ_replay["_skipped"],
            },
        },
        "skip_reasons": {
            "tests/live/test_live_authorization.py": (
                "blocked_external: STEALTHBENCH_LIVE_AUTHORIZATION is not set to 1"
            )
        },
        "mutation_checks": mutations,
        "live_readiness": (
            "blocked_external: no credentials, no spending authorization, "
            "no container runtime probed"
        ),
        "passed_gate": full["exit_status"] == 0 and undetected == 0,
    }

    out_dir = ROOT / "implementation" / "evidence" / "G03" / revision
    if out_dir.exists():
        raise SystemExit(f"{out_dir} already exists; evidence is append-only")
    out_dir.mkdir(parents=True)
    (out_dir / "gate.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    emit(f"wrote {out_dir.relative_to(ROOT)}/gate.json")
    emit(
        f"revision {revision}  passed={full['_passed']} "
        f"failed={full['_failed']} skipped={full['_skipped']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
