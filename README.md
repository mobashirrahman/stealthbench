# StealthBench

Reproducible benchmarking and evidence-based identity estimation for anonymous model
endpoints served through OpenCode Zen.

## Status

All implementation gates G00–G14 are built in this tree. No benchmark result
exists yet: nothing here has contacted a provider or spent money.
Capabilities land gate by gate under [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md);
the machine-readable backlog and gate states are in
[implementation/tasks.json](implementation/tasks.json).

| Area | State |
| --- | --- |
| Software implementation | complete in working tree (G00–G14); formal per-gate `gate.json` recording pending |
| Offline validation | green: `unit+contract 1406 passed`, `ruff` + `mypy --strict` clean; 2 `test_install` wheel failures are environment-only (no pip build env) |
| Real campaign execution | blocked — no credentials or spending authorization configured |
| Attribution evidence | similarity mode only, never probability, until the G12 evidence gate passes |

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

## Use

```bash
stealthbench manifest validate configs/offline-demo.json
stealthbench run configs/offline-demo.json --offline --output artifacts/offline-demo
stealthbench replay artifacts/offline-demo
stealthbench report artifacts/offline-demo --output reports/offline-demo
stealthbench doctor
```

`run` without `--offline` refuses (exit 3, `G05` pending semantics preserved for
bare runs); `--offline` dispatches fixture-only generations under declared caps.
`doctor` reports container/dataset availability; live suites stay
`blocked_external` without `STEALTHBENCH_LIVE_AUTHORIZATION=1` plus credentials
and a spending cap. See [docs/operations.md](docs/operations.md) for the full
operator guide.

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
