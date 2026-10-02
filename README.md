# StealthBench

Reproducible benchmarking and evidence-based identity estimation for anonymous model
endpoints served through OpenCode Zen.

## Status

This project is in early construction. Nothing here has produced a benchmark result.
Capabilities land gate by gate under [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md);
the machine-readable backlog and gate states are in
[implementation/tasks.json](implementation/tasks.json).

| Area | State |
| --- | --- |
| Software implementation | in progress, gate G00 |
| Offline validation | not run |
| Real campaign execution | blocked — no credentials or spending authorization configured |
| Attribution evidence | not started; similarity only, never probability, until the G12 evidence gate passes |

## Layout

```text
src/stealthbench/   package source
tests/              unit, contract, integration, sandbox, replay, live suites
configs/            example campaign profiles (no credentials)
docs/               frozen contracts, upstream inventory, agent handoffs
implementation/     task states and gate evidence
```

## Install and test

```bash
python3 -m venv .venv
.venv/bin/pip install -c constraints-dev.txt -e '.[dev]'
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/mypy src/stealthbench
.venv/bin/pytest --strict-markers tests/unit tests/contract
.venv/bin/pytest --strict-markers tests/integration tests/replay
```

`pyproject.toml` pins direct dependencies with `==`; `constraints-dev.txt` pins the
full transitive dev/test closure so an offline suite cannot drift with an upstream
release.

Default test fixtures deny all non-loopback network access (TCP and UDP) and scrub
credential-shaped environment variables. Suites that contact a provider are marked
`live` and `paid` and run only in a designated workflow with explicit authorization.

## Design rules

These are constraints, not aspirations:

- Unknown measurements stay missing. An absent token count is `null`, never `0`.
- A command that cannot do its work exits non-zero and says why. No score, identity
  or success claim is ever printed without the artifact that supports it.
- Gold answers and hidden evaluator files never enter a model request.
- A retry is not a sample. One complete agent trajectory is one attempt.
- Similarity is never displayed as probability.

## Reading order

1. [BENCHMARK_PLAN.md](BENCHMARK_PLAN.md) — design source
2. [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) — gates, contracts, test commands
3. [docs/contracts.md](docs/contracts.md) — frozen interfaces
4. [docs/upstream-inventory.md](docs/upstream-inventory.md) — pinned upstream revisions
5. [docs/AGENT_TASK_TEMPLATE.md](docs/AGENT_TASK_TEMPLATE.md) — per-task handoff format
