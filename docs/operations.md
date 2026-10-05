# StealthBench operations

Date: 2026-10-04. Normative source: `IMPLEMENTATION_PLAN.md` G14.
This document is the operator manual for the integrated release: how to
install from scratch, how to run the offline demonstration, what the
artifacts mean, and what must be true before any live (paid) dispatch.
No public benchmark outcome is reported here: every number below is either
a file count, a declared profile count, or a synthetic fixture result
labelled as such.

## 1. Fresh install

Prerequisites: Python 3.12, `pip`, no provider credential, no container
runtime required for the offline path.

```bash
python3 -m venv .venv
.venv/bin/pip install -c constraints-dev.txt -e '.[dev]'
.venv/bin/stealthbench --help
.venv/bin/stealthbench version
.venv/bin/stealthbench doctor
```

What each step proves:

| Command | Proves |
| --- | --- |
| `--help` | the console script installed and lists every declared command |
| `version` | machine-readable `{"status": "ok", "version", "schema_version"}` |
| `doctor` | exit 0 with runtime/dataset availability; container-only checks report `blocked_external` when docker/podman is absent instead of fake-passing |

A hermetic wheel check (no dependencies, no index) is exercised by
`tests/integration/test_install.py::test_wheel_installs_into_a_fresh_environment_and_console_script_runs`
(marked `slow`).

## 2. Offline demonstration (no credentials, no network, no spend)

The default posture is offline (`docs/contracts.md` section 4): mode
`offline`, fixture transport, `credential_ref: null`, network denied by the
test suite. Nothing about the default configuration can dispatch a request.

```bash
stealthbench manifest validate configs/offline-demo.json
stealthbench run configs/offline-demo.json --offline --output artifacts/offline-demo
stealthbench replay artifacts/offline-demo
```

`manifest validate` fails an invalid profile before any dispatch and reports
`dispatch_blockers` for a valid-but-not-yet-dispatchable one. `run --offline`
dispatches through recorded fixture responses only, enforces caps, grades
with the IFEval wrapper, and writes redacted exports. `replay` reconstructs
grades from `events.jsonl` without contacting any provider; its
`grade_digest` must equal the run's digest.

Reference fixture result (synthetic prompts, compliant fixture answers —
*not* a public outcome):

```json
{
  "accepted_samples": 3,
  "campaign_id": "offline-demo",
  "dispatched": 3,
  "planned": 3,
  "strict_correct": 3,
  "strict_eligible": 3,
  "transport_failed": 0
}
```

Reproduce it with the commands above; the digests in `summary.json` are the
evidence, not this copy. `report` (static site, G13) is still pending and
exits 3; release evidence uses `summary.json` plus `replay` until it lands.

## 3. Operator instructions

### 3.1 Offline runs

No authorization is needed or accepted: an offline profile must not declare
an `authorization` block and must use fixture transport only. Caps still
apply, including to zero-price fixtures.

### 3.2 Live runs (setup, pilot, full)

Live execution is never implicit. All of the following are required, and the
preflight module (`src/stealthbench/runner/preflight.py`) checks them without
dispatching anything:

1. `mode: live_authorized` with an `authorization` block naming
   `authorized_by` and a numeric `spending_cap_usd`.
2. Explicit operator approval outside the manifest:
   `STEALTHBENCH_LIVE_AUTHORIZATION=1`. A manifest on disk — a planning
   request — alone never authorizes paid calls.
3. A configured credential for every live endpoint: the `credential_ref`
   environment variable must be present. Secret values never appear in any
   serialized file.
4. A materialized selection: `materialized: true` with frozen `item_ids` for
   every benchmark and no unknown sizes.
5. Pinned revisions: `dataset_revision` (and `evaluator_revision` where an
   evaluator is declared; `image_digest`/`item_digest_manifest` for agent
   tasks). `null` means unverified, never "latest".
6. Price snapshots and caps: every live endpoint carries non-null pricing
   with a `snapshot_id`; `limits.require_cost_bounds` is true;
   `limits.max_total_cost_usd` is set and covered by the operator cap; the
   `missingness_threshold` is frozen before dispatch.
7. Runtime isolation: a container runtime (docker/podman) on `PATH` for any
   code/agent execution. When absent the check reports `blocked_external`.

Run preflight first and read every blocker:

```bash
STEALTHBENCH_LIVE_AUTHORIZATION=1 stealthbench run configs/pilot.json --live
```

refuses with a `blockers` list until all of the above hold. The setup
profile (80 direct items, 60 repeated signature probes, 20 agent tasks) runs
first under caps to establish measured cost and protocol support; the pilot
runs only after setup telemetry matches reservations. No specific model
result is required to pass an infrastructure check. Partial coverage is a
qualified partial report, never relabelled as complete; family probabilities
stay disabled unless the G12 evidence gate passes.

## 4. Example configurations

`configs/offline-demo.json` — small offline fixture campaign. One
fixture endpoint, one 3-item IFEval selection, materialized, dispatchable
with zero live inputs. Used by routine CI.

`configs/pilot.json` — declared pilot profile, `materialized: false`,
undispatchable until the frozen selection runs. Declared targets (not
measured results):

| Benchmark | Track | Declared items |
| --- | --- | --- |
| livecodebench | direct | 100 |
| evalplus | direct | 50 |
| ifeval | direct | 150 |
| mmlu_pro | direct | 100 |
| math500 | direct | 100 |
| bfcl (native + prompted) | direct | 50 |
| ruler | direct | 50 |
| **direct subtotal** | | **600** |
| swebench | agent | 25 |
| terminalbench | agent | 25 |
| **agent subtotal** | | **50** |
| signature probes (separate) | | 60 probes x 3 repeats |

`configs/full.json` — full-suite profile, `materialized: false`,
`require_full_split: true` on every benchmark. It enumerates the entire
declared official split item by item once materialized; a subset is never
relabelled as full. Expected split sizes follow `docs/upstream-inventory.md`
(livecodebench 1055, evalplus 164, ifeval 541, mmlu_pro 12032, math500 500,
swebench 500, terminalbench 66); BFCL and RULER carry a `split_enumeration`
procedure with a genuinely unknown count until G08 freezes the category
list, reported as `null` rather than guessed. Comparability with external
published outcomes requires identical dataset revision, prompt, extraction
rule, generation settings, scaffold, and grading protocol.

Validate any profile without side effects:

```bash
stealthbench manifest validate configs/pilot.json
stealthbench manifest validate configs/full.json
```

## 5. Artifact layout

One directory per campaign (`artifacts/<campaign_id>` by default):

| File | Content |
| --- | --- |
| `manifest.json` | canonical frozen manifest this run executed |
| `events.jsonl` | append-only durable record (`sample.accepted`, `request.failed`, `grade.recorded`, `campaign.completed`); the only source replay reads |
| `grades.jsonl` | redacted per-sample grades with explicit transport/evaluator statuses |
| `export.jsonl` | redacted generations (no canary secret survives; see redaction suites) |
| `summary.json` | counts, denominators, grade digest, ledger totals |
| `ledger.json` | spending/token ledger snapshot |
| `index.sqlite3` | queryable index over accepted samples |

Resume appends to the same store: previously accepted samples are skipped,
never duplicated (one accepted sample per campaign/endpoint/task/repeat).
Interrupted writes never look like completed samples. Restricted raw data is
preserved only when a profile explicitly asks; default exports are redacted.

## 6. Demo report

The complete fixture-based demonstration is `configs/offline-demo.json`
through the Section 2 commands. Its evidence is:

* `manifest validate` exit 0 with `dispatchable: true`, `planned_requests: 3`.
* `run --offline` exit 0; `summary.json` records 3 planned, 3 accepted,
  0 transport failures, with strict/loose denominators over graded responses
  only.
* `replay` exit 0 with identical `accepted_samples` and `grade_digest`.
* A second `run --offline` into the same directory skips accepted samples
  and creates no second accepted sample for any key.
* `doctor` exit 0 reporting fixture availability and any `blocked_external`
  runtime honestly.

Coverage and attribution stay qualified: the demo exercises the fixture
path only. Official revisions, live endpoints, container enforcement, and
calibrated identity probabilities are reported as unavailable until their
gates pass, with the exact missing input named.
