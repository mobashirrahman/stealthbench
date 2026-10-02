# Task packet for a StealthBench coding agent

Use this packet for one task from `implementation/tasks.json`. Fill in actual interfaces, fixtures and commands before assigning it. These instructions apply to implementation work; this file does not establish that the application already exists.

## Assignment

```yaml
task_id: T00A
title: Replace with the backlog task title
gate: G00
depends_on: []
source_revision: Replace with commit or source-tree digest
objective: One concrete observable behavior
read_first:
  - Relevant plan gate and interface specification
allowed_changes:
  - Exact production modules and test files
inputs:
  - Fixture names and required configuration
outputs:
  - Interface or artifact expected by the next task
acceptance:
  - Observable behavior with an independent expected result
required_failure_cases:
  - Invalid input or dependency failure with expected handling
checks:
  - Exact command that collects the required tests
authorization:
  mode: offline
  live_campaign: null
  spending_cap: null
```

## Implementer prompt

You are implementing one StealthBench task using the frozen project contracts. Read the assignment, the named files and the relevant gate in `IMPLEMENTATION_PLAN.md`. Confirm dependencies from their evidence records. Implement the smallest complete change that satisfies the contract and its failure cases.

Use at most three production modules unless the controller explicitly scopes a larger change. Split the work when needed. Add meaningful tests using reference fixtures, hand-computed outcomes or documented invariants. Keep the official benchmark grader unchanged. Run targeted tests, then the task checks. Record real outputs and exit statuses.

Default work is offline. Existing API credentials do not independently authorize requests. Live execution follows the already authorized campaign and its caps. Continue local work without repeated permission requests. Do not change scoring rules, erase failures, invent telemetry or mark a skipped mandatory test as passed.

If two repair attempts fail on the same issue, provide a minimal reproducer, the observed failure and the exact missing decision or dependency. Continue any independently authorized work that can progress.

Return a handoff with:

1. Completed behavior and changed files.
2. Tests run, collected counts, pass/fail/skip counts and exit statuses.
3. Evidence and artifact paths.
4. Any unmet acceptance criterion.
5. Suggested next task from the dependency graph.

## Reviewer prompt

Review this task in fresh context. Read the packet, diff, public contracts and test evidence. Check that the test oracle is independent of the implementation, required failure cases are exercised and no claims exceed evidence. Check missing values, retry accounting, provenance and secret handling where relevant.

Return `PASS` only when every acceptance criterion is supported. Otherwise return `FAIL` with file locations, a concrete reproducer and the smallest necessary correction. Return `BLOCKED_EXTERNAL` when a required environment or dataset is absent. Do not turn an unavailable check into a pass.

## Example first task

Task `T00A`: create the installable package and CLI shell, a development dependency lock and a test layout. The CLI must show help and reject invalid commands; it must not print fabricated scores. Acceptance includes installation into a clean environment and tests for both behaviors. No provider access is needed. Leave dataset selection and contract implementation to their own tasks.

---

## Publishing after each gate

Every gate that passes is committed and pushed. Concretely, for each gate:

1. Re-run the gate checks and record `implementation/evidence/<gate>/<revision>/gate.json`
   with the real commands and exit statuses.
2. Verify the working tree is what the evidence claims:
   `ruff check . && ruff format --check . && mypy src/stealthbench` and the gate suites.
3. Stage and review: `git status --short`, then confirm nothing ignored-but-needed is
   missing and that no `.venv`, cache, artifact or report path is staged.
4. Commit with a message that states which gates passed, which are still open, and
   explicitly that no benchmark result exists. Never describe a gate as passing when
   its review has outstanding findings.
5. Push: `git push origin main`.

Rules that apply to publishing:

- **Never fabricate or tidy the evidence history.** Gate history is append-only. A
  superseded record stays on disk and says what superseded it and why.
- **A public push is a real publication.** Do not push content the operator has not
  seen, and do not publish a milestone whose review findings are unfixed without
  saying so in the commit message.
- **Secrets are checked, not assumed.** A local `pre-push` hook refuses any
  credential-shaped string that is not a deliberate canary (constants named with
  `canary`, or literals listed in `.git/allowed-canaries`). It is intentionally not
  committed.
- **Known open findings are stated in the commit message**, not buried. A commit that
  says "G03 in review, 9 defects unfixed" is honest and useful; one that implies the
  gate passed is not.
