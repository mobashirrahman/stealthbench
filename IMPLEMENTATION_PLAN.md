# StealthBench: gated implementation plan for LLM agents

Date: 2 October 2026. Design source: [BENCHMARK_PLAN.md](BENCHMARK_PLAN.md).

This is an executable work specification for a smaller coding LLM. The application and the commands described below still need to be implemented. Creating this plan does not mean any implementation gate has passed.

## 1. Deliverable

Ship an installable Python CLI that discovers Zen aliases, runs the six-category benchmark, executes OpenCode agent tasks, measures endpoint performance, extracts model signatures, produces evidence-based identity estimates, monitors alias changes, and builds a browsable static leaderboard with JSON/CSV exports.

Support offline fixtures and replay throughout. Real evaluations use explicit campaign configuration, API credentials and spending limits. Preserve pilot/full-suite distinctions and uncertainty. A functioning similarity estimator is a valid first release; calibrated identity probabilities require a separate evidence gate.

The initial scope includes complete public benchmark adapters, the 600-item pilot profile, full-set profiles, 60 signature probes with three repetitions, an 80-item direct setup profile, a 20-task agent setup profile and a 50-task agent pilot. Availability and dataset licensing are checked at implementation time. Do not silently substitute another dataset when a dependency is unavailable.

## 2. Agent operating model

Use one smaller coding LLM for implementation. Give it one task packet at a time. Use a separate review pass with the same model in fresh context, or a designated reviewer when available. The controller may be a person or an agent; it selects tasks and records evidence. Concurrent agents are unnecessary for this plan.

| Role | Responsibility | Output |
| --- | --- | --- |
| Controller | Select the next ready task; enforce dependencies and scope | Task packet and gate decision |
| Implementer | Implement one contract and its tests | Small patch, test evidence and handoff |
| Reviewer | Read the diff, acceptance criteria and failure tests independently | Specific defects or a justified pass |

### Rules for a smaller model

1. Read the task packet, relevant interfaces and named fixtures. Do not load the entire codebase by default.
2. Normally change at most three production modules plus their tests. Split a task if it requires a wider change.
3. Work on one contract at a time. Reuse official evaluators; avoid implementing an alternative grader.
4. Write tests against expected outcomes, reference fixtures or invariants, not copies of implementation logic.
5. Use the specified dependency versions and interfaces. Resolve versions and upstream commits in G00; do not guess APIs from memory.
6. Preserve existing passing tests. A regression needs a specific documented explanation, not deletion of the assertion.
7. Run targeted tests after edits and the relevant complete gate before advancing. Include actual commands and exit statuses.
8. After two unsuccessful fix attempts on the same failure, reduce it to a minimal reproducer and escalate the concrete blocker to the controller. Continue independent ready tasks where possible.
9. Missing credentials, unavailable container runtimes or missing reference data block the affected gate, not unrelated offline work.
10. Preserve prior authorization. Local edits, tests and review proceed automatically. Live runs require configured authorization and limits; publishing externally requires authorization if it has not already been given.

Use [docs/AGENT_TASK_TEMPLATE.md](docs/AGENT_TASK_TEMPLATE.md) for every task. [implementation/tasks.json](implementation/tasks.json) contains the initial ordered backlog.

## 3. Architecture and contracts

Use Python, a package under `src/stealthbench`, a standard HTTP client, pytest, a formatter/linter, a type checker and property tests for numerical and scheduler invariants. Choose and lock exact supported versions in G00. Begin with SQLite and JSONL; add another storage format only when a concrete export needs it.

```text
src/stealthbench/
  schemas/        versioned configuration and result contracts
  storage/        event logs, SQLite index, redaction and artifact hashes
  adapters/       Zen, reference providers and fixture transport
  runner/         scheduling, retries, budgeting and crash recovery
  benchmarks/     pinned official evaluator wrappers
  sandbox/        isolated execution and evaluator separation
  agents/         pinned OpenCode execution and trace capture
  fingerprints/   probes, feature extraction, reference matching and attribution
  analysis/       scoring, intervals, performance and drift
  reporting/      static pages and JSON/CSV export
  cli.py
tests/
  unit/ contract/ integration/ sandbox/ replay/ live/
configs/          example campaign profiles without credentials
implementation/   task status and gate evidence
docs/             protocol, operation and agent handoffs
```

Freeze these contracts before implementing dependent modules:

- `CampaignManifest`: schema version, observation window, endpoint IDs, dataset/evaluator revisions, item IDs, generation settings, retry policy, limits, sampling seed and score version.
- `EndpointSpec`: alias, route, capabilities, token accounting, price snapshot and credential reference. Secret values stay outside serialized configuration.
- `TaskSpec`: benchmark ID, item ID, prompt artifact hash, evaluator revision, declared metadata and repeat ID. Gold answers and hidden evaluator files remain outside model requests.
- `GenerationResult`: attempt ID, task/repeat ID, response, usage as nullable values, supported effective settings, streaming measurements, finish status and redacted provider metadata.
- `GradeResult`: grader version, score components, denominator eligibility and distinct correctness/format/transport/evaluator statuses.
- `RunEvent`: stable IDs, event type, UTC wall time, monotonic durations and payload hash. Delivery attempts and accepted samples are different objects.
- `SignatureResult`: probe version, input-count vector, validity mask, behavioral features and observation window.
- `IdentityReport`: candidate library revision, signal evidence, ranked similarities, abstention reason, optional calibrated probabilities and separate official reveal label.
- `GateEvidence`: gate ID, code revision or source-tree digest, manifest hash, commands, exit statuses, collected/passed/failed/skipped counts, artifacts and review decision.

Schema errors must be explicit. Unknown usage is `null`, never zero. Unsupported features remain unsupported. A benchmark result is keyed by campaign, endpoint observation, task and repetition; retries do not add samples.

## 4. Gate policy

Task states are `todo`, `ready`, `running`, `review`, `done` and `blocked_external`. Gate states are `not_run`, `passed`, `failed` and `blocked_external`.

Mark a task done only after its acceptance tests and review pass. Mark a gate passed only when every required task is done and its checks pass on the same source revision. A relevant code, evaluator, schema, manifest or fixture change invalidates affected gates and their dependents. Gate history is append-only; passing results are never overwritten.

Test discovery must collect the expected suites. A skip, expected failure, empty suite or successful test-run exit code alone cannot satisfy a mandatory check. An unavailable mandatory environment produces `blocked_external`. Optional provider tests must be labeled as optional rather than silently standing in for required coverage.

Record evidence under `implementation/evidence/<gate>/<revision>/`. Gate JSON identifies prerequisite evidence, environment/tool versions, artifacts and reviewer findings. Never record a fictional command result.

The controller advances automatically through local gates. It asks only for missing execution inputs or actions outside existing authorization.

## 5. Build sequence

Each gate below is a stage, not one large agent assignment. Execute its tasks separately in the order specified by the backlog.

### G00 — Repository, dependency and protocol freeze

Tasks: `T00A–T00C`.

Create packaging, a fixture-only CLI shell, test markers, tooling, dependency lock and CI. Select compatible Python/tool versions. Inspect official upstream benchmark sources and write an inventory of dataset revisions, evaluator commits, supported API modes, licenses and install requirements. Freeze interface specifications and example manifests.

Tests: installation into a fresh environment, CLI help and invalid-command behavior, configuration rejection, marker validation and CI fixture selection.

Gate: package installs; empty CLI commands cannot claim successful evaluation; offline tests cannot reach external network; all seven benchmark components have an explicit upstream inventory entry, including both coding components. Unavailable assets have a concrete acquisition condition.

### G01 — Schemas and provenance

Tasks: `T01A–T01C`; depends on G00.

Implement configuration/results, canonical hashing, manifests and version checks.

Tests: serialization round trips; omitted versus zero usage; unknown enum/schema versions; duplicate item IDs; non-finite values; conflicting limits; item/evaluator changes altering hashes; insignificant serialization ordering preserving canonical hashes.

Gate: invalid manifests fail before any model request. A request cannot contain gold-answer or evaluator-only fields. Every sample can be traced to its exact manifest and prompt.

### G02 — Artifact storage and replay

Tasks: `T02A–T02C`; depends on G01.

Implement append-only events, artifact storage, SQLite indexing, export redaction and replay. Preserve restricted raw data only when explicitly configured; default exports and logs are redacted. Use atomic artifact writes and durable event completion records.

Tests: process interruption during writes; partial final JSONL records; duplicate ingestion; mismatched hashes; event ordering; credential-bearing headers/errors; recognizable canary secrets; replay producing identical deterministic grades.

Gate: interrupted writes cannot invent completed samples. Redacted exports contain no canary secrets. Replaying saved generations does not contact a provider.

### G03 — Endpoint adapters and streaming

Tasks: `T03A–T03C`; depends on G02.

Implement fixture transport, Zen catalog discovery, supported chat protocol and a reference-provider interface. Add one reference adapter at a time; additional wire protocols need separate tested modules. Snapshot the observed catalog rather than hardcoding stealth aliases.

Tests: normal/empty/malformed catalog; authentication failures; 429/5xx; usage absence; tool responses; fragmented Unicode; multiple SSE events in one read; split SSE records; reasoning-only chunks; usage-only final chunks; stream interruption; timeout and unsupported settings.

Gate: adapter output matches the result contract; chunks are not counted as tokens; error bodies are redacted; unsupported parameters are reported; production adapter behavior is demonstrated against fixture protocol transcripts.

### G04 — Scheduler, costs and recovery

Tasks: `T04A–T04D`; depends on G03.

Implement deterministic scheduling, bounded retries, a spending ledger, token/time caps and resume behavior. Reserve worst-case cost before dispatch and account for concurrent outstanding requests. Keep billed usage and conservative estimates separate. Persist reservations and unfinished requests.

If reliable upper bounds are unavailable, refuse spending-capped execution until explicit bounds exist. On crash, a dispatched request without a durable response is ambiguous: it may have been billed. Default recovery records it unresolved rather than resubmitting it automatically. An explicit retry retains the reservation/accounting history and duplicate-risk disclosure.

Tests: fake-clock retries; non-retryable 4xx; cancelled requests; rate limits; concurrent budget races; price changes; cached/reasoning usage; absent billing data; crashes before/after dispatch; resumed item de-duplication; aliases with zero current price still respecting token/request limits.

Gate: controlled fake-provider runs cannot exceed declared request, token, concurrency or reserved-cost limits. Task failures are not retried as if they were transport errors. Resume cannot silently create a second accepted sample.

### G05 — First benchmark and vertical workflow

Tasks: `T05A–T05C`; depends on G04.

Implement the IFEval adapter using its official scorer first. Add fixed item selection and run → grade → replay → export CLI commands. Use a small legal/synthetic fixture collection for routine CI.

Tests: known compliant/noncompliant answers; strict/loose distinctions; selected item order; forbidden answer leakage; malformed responses; transport failures versus incorrect answers; complete offline workflow.

Gate: wrapper scores agree with pinned official outputs on the golden fixture set; request logs exclude solutions; saved responses can be regraded with no model calls. This gate proves the architecture before expanding adapters.

### G06 — Knowledge and mathematics adapters

Tasks: `T06A–T06B`; depends on G05.

Implement MMLU-Pro generated-answer extraction and MATH-500 scoring under frozen protocols. Each adapter is its own task. Keep upstream grader isolation requirements explicit, including symbolic parsing time limits.

Tests: known correct/incorrect gold fixtures; ambiguous/multiple answers; invalid choices; official numerical/symbolic edge cases; malicious or oversized expression inputs; parser timeouts; dataset checksum mismatch and unsupported revisions.

Gate: official fixture agreement is exact within explicitly declared numeric tolerances. No likelihood-based score is fabricated from a text-only endpoint.

### G07 — Execution sandbox and coding benchmarks

Tasks: `T07A–T07D`; depends on G05.

Implement the execution backend before LiveCodeBench and EvalPlus adapters. Run generated programs in disposable containers or an equivalently validated runtime. Use CPU/memory/process/disk/output/wall-time caps; no privileged mode, container-engine socket, host credentials or writable host mounts. Network is denied for candidate execution. Grade hidden tests in a separate evaluator environment.

Tests: a known correct solution, a subtly wrong solution, syntax error, infinite loop, excessive memory, process spawning, huge output, path traversal, symlink traversal, attempts to read host/evaluator files, network requests and cleanup after cancellation.

Gate: containment tests demonstrate enforced restrictions on the actual chosen runtime. Missing enforcement blocks code execution. Both coding wrappers match official grader fixtures and preserve the requested date window and item selection.

### G08 — Tool use and long context

Tasks: `T08A–T08B`; depends on G05; use G07 where tool execution requires it.

Implement BFCL and RULER adapters separately. Keep native function calling and prompted calling as distinct profiles. Pin RULER generation seeds, task settings and context lengths; report provider counts and normalized lengths separately.

Tests: valid/invalid arguments, irrelevant tools, missing parameters, multiple/parallel calls, Unicode arguments, correct/incorrect retrieval, length mismatch, truncation and unsupported context windows.

Gate: adapters agree with the chosen official fixtures. Unsupported categories cannot silently reduce the denominator or produce a complete-core score. Long-context tasks that were truncated cannot be reported as delivered at their intended length.

### G09 — Scoring, statistics and endpoint performance

Tasks: `T09A–T09C`; depends on G06, G07 and G08.

Implement category aggregation, declared core index, repeated-trial accounting, paired task-level intervals, reliability metrics, latency and normalized output rates.

Tests: hand-computed score examples; incorrect versus missing responses; uneven categories; incomplete coverage; repeated tasks; identical paired outcomes; order invariance; seeded resampling; zero/one success cases; too few tasks; first content versus first answer timing; Unicode lengths; absent usage and exact synthetic time traces.

Gate: index matches a hand-computed oracle and remains absent for incomplete coverage. Repetitions are clustered by task. Return unavailable intervals where inference is undefined. Endpoint failures and response-conditional scores have explicit denominators. Benchmarks are compared on shared declared items rather than cherry-picked responses.

### G10 — OpenCode agent evaluation

Tasks: `T10A–T10C`; depends on G04 and G07.

Implement a pinned OpenCode invocation and traces, followed by SWE-bench and Terminal-Bench wrappers as separate tasks. Route model access through a broker with credentials outside the task container. Limit steps, tool calls, time, token usage and cost for the full trajectory. Record prompt selection, compaction and auxiliary model calls; all are included in cost and attribution metadata.

Tests: fixture agent success/failure, tool timeout, step exhaustion, compaction, auxiliary usage, broker restrictions, secret/evaluator access attempts, cancellation and fresh task environments. Use known resolving/non-resolving patches for SWE-bench and known pass/fail task artifacts for Terminal-Bench.

Gate: a complete trajectory is one attempt; official resolution grades match fixtures; effective configuration and prompts are captured; hidden evaluator assets and credentials are inaccessible to the agent. Required dependency versions and task images are pinned.

### G11 — Fingerprint collection and similarity

Tasks: `T11A–T11C`; depends on G04 and G09.

Version the 60-probe bank, baseline framing, repeat protocol and reference tokenizer assets. Implement validity checks, delta vectors, per-signal similarities and a ranked reference report. Separate family, exact version, tokenizer and route labels.

Tests: constant framing offsets; content-dependent offsets; missing counts; unstable repeats; two different models sharing a tokenizer; exact count matches with divergent behavior; incomplete vectors; changing reference libraries; tokenizer downloads with remote executable code disabled.

Gate: missing signals stay unavailable; count matches are labeled tokenization evidence. Similarity is never displayed as probability. Without sufficient comparable evidence, report unknown/insufficient evidence.

### G12 — Attribution validation and alias monitoring

Tasks: `T12A–T12C`; depends on G11.

Implement grouped dataset splitting, a simple classifier with optional calibration, abstention rules, later-time/cross-route holdouts, and protocol/timing ablations. Add drift detection with stable controls and append-only reveal history.

Tests: deliberate train/validation/test leakage; duplicate source observations; held-out unknown families; conflicting evidence; missing features; uncalibrated outputs; changed priors/reference sets; simulated stable/changed endpoints; control-wide infrastructure shifts and retrospective reveal updates.

Software gate: split contamination is rejected, experimental thresholds are frozen before holdout scoring, and unknown is a supported output. Stable control simulations meet the preregistered false-alert target; change simulations meet the declared detection target.

Evidence gate: before displaying probabilities, declare the acceptable known-family error, unknown false-accept rate, calibration error and coverage, plus confidence-bound methods and minimum independent sample sizes. Freeze numerical thresholds using development data. Pass them on untouched holdout data; otherwise ship similarity mode. Exact-version claims require a separate held-out version-level gate. The small initial endpoint library may be insufficient to pass these evidence gates.

### G13 — Leaderboard, exports and operations

Tasks: `T13A–T13C`; depends on G09, G10 and G12.

Build static model pages, comparisons, history, JSON/CSV exports, canary scheduling and operational commands. Show observation dates, sample sizes, confidence intervals, costs, missing coverage, supported identity resolution and the distinction between predicted and revealed identities. Serve the site locally for review; deployment is a separate authorized action.

Tests: complete/partial/failed campaigns; tied or uncertain rankings; unknown identity; stale evidence; prediction versus reveal display; script-bearing response/alias content; CSV formula injection; artifact-link traversal; redacted downloads; offline scheduled canary idempotency; export/import consistency.

Gate: pages render with escaped content, counts match source results, unsupported claims are absent, and every displayed score links to reproducible provenance. Canary jobs enforce the same budget and authorization rules as ordinary campaigns.

### G14 — Integrated release and real campaign gates

Tasks: `T14A–T14D`; depends on G13.

Run fresh-install, offline integration, replay, sandbox and full regression checks. Exercise recovery under fault injection. Provide operator instructions, example configurations, package artifacts and a complete fixture-based demonstration report.

Software release gate: all mandatory offline/runtime suites pass; a fresh installation performs discovery from fixtures, benchmark generation, grading, signature matching, agent fixture execution, resume, replay and report generation. Full-set configurations enumerate the entire declared official split rather than a pilot selection. No public scores are invented.

Live readiness gate: current aliases/reference IDs are verified, credentials and spending authorization are configured, dataset access is valid, runtime isolation passes, price snapshots and caps are present, and production evidence records the exact configuration. A planning request alone does not authorize paid calls.

Live setup gate: run the 80-item direct setup, 60 repeated signature probes and 20-task agent setup under caps. Compare measured telemetry and cost with reservations; resolve unsupported modes and protocol errors before expanding. Do not require a specific model accuracy to pass an infrastructure check.

Pilot gate: run the 600-item core and 50-task agent profiles on the declared endpoints. Freeze missingness thresholds before dispatch. Publish complete scores or qualified partial reports as appropriate. Family probabilities remain disabled unless the G12 evidence gate passes.

Full benchmark gate: execute full declared splits under a new immutable campaign. Compare with external scores only where dataset, prompts, settings, scaffold and grading are equivalent. Preserve the pilot report. Runtime interruptions or access gaps are explicit external blockers, never a reason to relabel a subset as full.

## 6. Test execution and CI

Proposed commands become valid after G00 and subsequent CLI tasks implement them:

```bash
ruff check .
ruff format --check .
mypy src/stealthbench
pytest --strict-markers tests/unit tests/contract
pytest --strict-markers tests/integration tests/replay
pytest --strict-markers tests/sandbox
stealthbench manifest validate configs/offline-demo.json
stealthbench run configs/offline-demo.json --offline
stealthbench replay artifacts/offline-demo
stealthbench report artifacts/offline-demo --output reports/offline-demo
```

Markers distinguish `unit`, `contract`, `integration`, `sandbox`, `replay`, `slow`, `live` and `paid`. Markers alone are not a security boundary: default test fixtures deny external network, and the live runner checks explicit configuration in addition to markers. CI contains no production provider credentials. Live/paid suites run only in the designated execution workflow.

Run lint, type checks and relevant tests for every task. Run the full offline regression suite at gate boundaries; run container suites when execution paths change and before release. Verify a fixture-sized workflow deterministically, then test property invariants on scoring and scheduler behavior. Overall line coverage is a diagnostic; mandatory failure scenarios and independent oracles are the release requirements.

Useful mutation checks: turn missing usage into zero, count retries as samples, bypass a reservation, include gold fields in a request, treat a tokenizer match as exact identity, and calculate a core index with a missing category. Each deliberate mutation must cause at least one required test to fail. This can be a targeted review check rather than a new general-purpose mutation framework.

Tool references: [pytest markers](https://docs.pytest.org/en/stable/how-to/mark.html), [Hypothesis property tests](https://hypothesis.readthedocs.io/en/latest/), [Ruff](https://docs.astral.sh/ruff/), [Docker resource constraints](https://docs.docker.com/engine/containers/resource_constraints/). Resolve actual supported versions and runtime restrictions in G00/G07 rather than relying on illustrative commands alone.

## 7. Evidence and completion

Every task handoff contains changed files, the completed contract, actual test commands/results, artifact locations, unresolved issues and the next ready task. Every review checks the scope, independent oracle, required failure paths and claims in the output.

Project status has separate fields for software implementation, offline validation, real campaign execution and attribution evidence. An offline demonstration can pass while live inputs are unavailable. A completed application can report similarity while probability calibration remains unvalidated. Report these conditions precisely.

The controller starts at T00A. A smaller agent should not attempt this whole document in one turn. Complete the first vertical workflow through G05, then add one benchmark adapter at a time, followed by agent execution, signatures, attribution, reporting and release evidence.
