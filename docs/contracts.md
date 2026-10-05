# Frozen contracts

Date frozen: **2026-10-02**. Normative source: [IMPLEMENTATION_PLAN.md](../IMPLEMENTATION_PLAN.md)
sections 3 and 4. This file fixes the interface surface that every later task builds
against, so that G01 can implement schemas without renegotiating field names.

The machine-readable block near the end is the authoritative structural spec;
`tests/contract/test_profile_spec.py` validates both this file and the example
campaign profiles in `configs/` against it.

## 1. Missing values

One rule governs the whole system.

| Situation | Representation | Never |
| --- | --- | --- |
| Measurement not reported by the endpoint | `null` | `0` |
| Token count absent, so cost unknown | `null`, and cost `null` | cost `0` |
| Benchmark category not run | item absent from the result set | a zero that enters a mean |
| Unsupported generation setting | `unsupported_settings` entry naming the setting | silently dropping the setting |
| Feature not offered by an endpoint | `capabilities.<feature> == false` | assuming support because other endpoints have it |
| Endpoint has no published price | price fields `null` | treating the endpoint as free |
| Attempt dispatched but outcome unknown after a crash | `unresolved`, explicitly | counting it as a failure or a success |
| Identity evidence insufficient | `unknown` / `insufficient_evidence` | a probability |

A missing measurement must be distinguishable from a measured zero after
serialization. Any test that substitutes `0` for an absent value is a defect, and the
deliberate mutation "turn missing usage into zero" is expected to break a mandatory
test.

## 2. Status semantics

Statuses are distinct sets. Collapsing them destroys the ability to explain a result.

### Task and gate state

| State | Meaning |
| --- | --- |
| `todo` | not started |
| `ready` | dependencies done; may be worked |
| `running` | in progress |
| `review` | implementation complete, awaiting independent review |
| `done` | acceptance tests **and** review passed |
| `blocked_external` | a required environment, credential, container or dataset is absent |
| `not_run` / `passed` / `failed` / `blocked_external` | gate states |

`done` requires review. `passed` requires every required task `done` **and** its checks
passing on the same source revision. A relevant code, schema, evaluator, manifest or
fixture change invalidates the affected gates and their dependents. Gate history is
append-only: a passing result is never overwritten.

### Per-sample outcome

Four orthogonal statuses are recorded separately and are never merged:

| Status | Meaning | Denominator effect |
| --- | --- | --- |
| `correctness` | did the answer satisfy the task rule | in the correctness denominator |
| `format` | was the response well formed / parseable | reported separately |
| `transport` | did the request complete at all | excluded from response-conditional scores |
| `evaluator` | did the official grader run to a verdict | excluded from graded scores |

This separation is what makes "endpoint success rate", "response-conditional accuracy"
and "end-to-end task success" three different numbers with three different
denominators.

### Attempts versus samples

- One accepted sample per (campaign, endpoint observation, task, repeat).
- A **retry** is a delivery attempt. Retries are retained and counted as attempts.
- An agent trajectory — however many API calls it makes — is **one** attempt.
- Best-of-N is never labelled `pass@1`. `pass@1` comes from accepted samples only.

## 3. Caps and budgeting

Every run, including an offline one, declares caps.

| Cap | Enforces |
| --- | --- |
| `max_requests` | total accepted-plus-attempted request count |
| `max_concurrency` | in-flight requests |
| `max_input_tokens` / `max_output_tokens` | token totals |
| `max_total_cost_usd` | reserved worst-case spend |
| `max_wall_seconds` | campaign duration |

Rules:

1. **Reserve before dispatch.** Worst-case cost is reserved before a request leaves,
   and the reservation is settled or released against the measured or estimated cost.
2. **Outstanding requests count.** A concurrency race must not let the campaign exceed
   a cap, so in-flight reservations are visible to the budget check.
3. **Billed and estimated costs are separate.** Measured billed usage is never mixed
   with a conservative pre-dispatch estimate.
4. **`require_cost_bounds: true` everywhere.** If a reliable upper bound on the cost of
   a request cannot be computed — unknown price, unbounded output — spending-capped
   execution is **refused**. An unknown price is not permission to spend an unknown
   amount.
5. **Free endpoints still obey limits.** An alias priced at zero is still subject to
   request, token and concurrency caps.
6. **A cap is not an approval.** `max_total_cost_usd` is a ceiling recorded in a
   profile. Live execution additionally requires explicit operator authorization and a
   separate cap.

## 4. Offline defaults

The default posture is offline, and offline is a structural property rather than a
flag that could be forgotten.

| Aspect | Offline default |
| --- | --- |
| `mode` | `offline` |
| Transport | `fixture` — recorded protocol transcripts, never a live socket |
| Credentials | none; `credential_ref` is `null` |
| Network | denied by the test suite; fixture transport cannot dial out |
| Determinism | same seed and same fixtures produce identical events and grades |
| Budget | caps still declared and still enforced |
| Provenance | every sample traceable to manifest, prompt hash and fixture digest |

A live campaign must opt in explicitly: `mode: live_authorized`, a non-null
`authorization.spending_cap_usd`, and configured credentials. Nothing about the
default configuration can dispatch a request.

## 5. Secrets

- Secret **values** never appear in a serialized manifest, log, event, artifact or
  export. `credential_ref` is a name resolved to an environment variable at run time.
- Provider error bodies and response headers are redacted before storage, because
  gateways routinely echo credentials back inside error payloads.
- Default exports are redacted. Restricted raw data is preserved only when a profile
  explicitly asks for it.
- The redaction tests use recognizable canary secrets and assert the canary appears
  nowhere in an export.

## 6. The nine contracts

Field lists below are normative minimums. Additional fields are allowed; removing one
requires a schema version bump.

### `CampaignManifest`

`schema_version`, `campaign_id`, `mode`, `materialized`, `score_version`, `seed`,
`observation_window`, `endpoints[]`, `benchmarks[]`, `generation`, `retry_policy`,
`limits`.

- `schema_version` is exact-matched; an unknown version is rejected, never coerced.
- `campaign_id` is unique per campaign. A rerun uses a new id so the original report is
  preserved.
- `materialized: false` means the frozen item selection has not run, and the profile
  cannot be dispatched.
- `observation_window.started_at` / `ended_at` are UTC instants or `null`. A fixture
  campaign observes no live endpoint, so both stay `null` rather than carrying a
  placeholder timestamp.

### `EndpointSpec`

`endpoint_id`, `alias`, `route`, `transport`, `capabilities`, `pricing`, `credential_ref`,
and four **independent** identity labels: `label_provider`, `label_family`,
`label_exact_version`, `label_tokenizer`.

- An alias is not an immutable model version. The catalog is snapshotted per campaign.
- The four labels stay separate. A shared tokenizer is tokenization evidence only;
  two different models ship the same tokenizer, so it is never sufficient for an
  identity claim.
- `pricing` fields are `null` when no price snapshot exists.

### `TaskSpec`

`benchmark_id`, `item_id`, `prompt_artifact_hash`, `evaluator_revision`, `declared_metadata`,
`repeat_id`.

- Gold answers and hidden evaluator files never enter a model request. This is
  enforced structurally, not by convention.
- Changing an item, a prompt or the evaluator changes the manifest hash, which
  invalidates every cached sample for that campaign.

### `GenerationResult`

`attempt_id`, `task_id`, `repeat_id`, `response`, `usage`, `effective_settings`,
`streaming`, `finish_status`, `redacted_provider_metadata`.

- `usage` fields are individually nullable: an endpoint may report input tokens and
  nothing else.
- `effective_settings` records what the endpoint says it actually did, separately from
  what was requested.
- `streaming` measurements distinguish first content from first visible answer.

### `GradeResult`

`grader_version`, `score_components`, `denominator_eligibility`, and the four statuses
from section 2: `correctness`, `format`, `transport`, `evaluator`.

- `denominator_eligibility` decides which published numbers may include the sample;
  it is a first-class field, not a filtering step applied at render time.

### `RunEvent`

`event_id`, `event_type`, `wall_time_utc`, `monotonic_duration`, `payload_hash`.

- Delivery attempts and accepted samples are different event types.
- Wall time is UTC for humans; durations use a monotonic clock so they cannot go
  backwards under clock adjustment.
- An event is only durable once its completion record is written. An interrupted write
  must never look like a completed sample.

### `SignatureResult`

`probe_version`, `input_count_vector`, `validity_mask`, `behavioral_features`,
`observation_window`.

- The validity mask is per-probe. A missing or unstable probe marks that probe
  unavailable; it is not imputed.
- Token counts are compared as deltas against a fixed baseline, because a fixed offset
  from message framing carries no identity information.

### `IdentityReport`

`candidate_library_revision`, `signal_evidence`, `ranked_similarities`,
`abstention_reason`, `calibrated_probabilities`, `official_reveal_label`.

- `calibrated_probabilities` is `null` unless the G12 evidence gate passed. Similarity
  is never displayed as probability.
- `official_reveal_label` is recorded separately from any earlier prediction, so a later
  reveal does not rewrite history.

### `GateEvidence`

`gate_id`, `source_revision`, `manifest_hash`, `commands`, `exit_statuses`,
`collected`, `passed`, `failed`, `skipped`, `artifacts`, `review_decision`.

- A successful test-run exit status alone cannot satisfy a check. A skip, an expected
  failure, an empty suite and a zero-exit run are all recorded, and none of them
  satisfies a mandatory check on its own.
- Evidence records real commands and real statuses. A fictional result is worse than
  no result.

## 7. Machine-readable contract spec

```json contracts
{
  "schema_version": "1.0",
  "frozen_on": "2026-10-02",
  "supported_schema_versions": ["1.0"],
  "campaign_modes": ["offline", "live_authorized"],
  "task_states": ["todo", "ready", "running", "review", "done", "blocked_external"],
  "gate_states": ["not_run", "passed", "failed", "blocked_external"],
  "sample_status_axes": ["correctness", "format", "transport", "evaluator"],
  "identity_labels": ["label_provider", "label_family", "label_exact_version", "label_tokenizer"],
  "null_semantics": {
    "absent_measurement": "null",
    "absent_usage_field": "null",
    "absent_price": "null",
    "zero_is_never_a_substitute_for_null": true,
    "unknown_identity": "unknown",
    "unresolved_after_crash": "unresolved"
  },
  "required_profile_fields": [
    "schema_version",
    "campaign_id",
    "mode",
    "materialized",
    "score_version",
    "seed",
    "observation_window",
    "endpoints",
    "benchmarks",
    "generation",
    "retry_policy",
    "limits"
  ],
  "required_limit_fields": [
    "max_requests",
    "max_concurrency",
    "max_input_tokens",
    "max_output_tokens",
    "max_total_cost_usd",
    "max_wall_seconds",
    "require_cost_bounds"
  ],
  "required_retry_policy_fields": [
    "max_attempts",
    "initial_backoff_seconds",
    "multiplier",
    "max_backoff_seconds",
    "retryable_statuses"
  ],
  "required_endpoint_fields": [
    "endpoint_id",
    "alias",
    "route",
    "transport",
    "capabilities",
    "pricing",
    "credential_ref"
  ],
  "offline_defaults": {
    "mode": "offline",
    "transport": "fixture",
    "credential_ref": null,
    "authorization_required": false,
    "network_access": "denied"
  },
  "live_defaults": {
    "mode": "live_authorized",
    "transport": "zen",
    "authorization_required": true,
    "network_access": "explicit"
  },
  "retryable_status_policy": {
    "allowed": [408, 429, 500, 502, 503, 504],
    "never_retry": "a wrong answer, a malformed answer, or any evaluator verdict"
  },
  "suites_not_yet_started": {
    "tests/sandbox": "T07A (G07). Containment tests need the execution runtime, which does not exist yet."
  },
  "required_suites_per_gate": {
    "G00": ["tests/unit", "tests/contract", "tests/integration"],
    "G02": ["tests/unit", "tests/contract", "tests/integration", "tests/replay"]
  },
  "contracts": {
    "CampaignManifest": ["schema_version", "campaign_id", "mode", "materialized", "score_version", "seed", "observation_window", "endpoints", "benchmarks", "generation", "retry_policy", "limits"],
    "EndpointSpec": ["endpoint_id", "alias", "route", "transport", "capabilities", "pricing", "credential_ref", "label_provider", "label_family", "label_exact_version", "label_tokenizer"],
    "TaskSpec": ["benchmark_id", "item_id", "prompt_artifact_hash", "evaluator_revision", "declared_metadata", "repeat_id"],
    "GenerationResult": ["attempt_id", "task_id", "repeat_id", "response", "usage", "effective_settings", "streaming", "finish_status", "redacted_provider_metadata"],
    "GradeResult": ["grader_version", "score_components", "denominator_eligibility", "correctness", "format", "transport", "evaluator"],
    "RunEvent": ["event_id", "event_type", "wall_time_utc", "monotonic_duration", "payload_hash"],
    "SignatureResult": ["probe_version", "input_count_vector", "validity_mask", "behavioral_features", "observation_window"],
    "IdentityReport": ["candidate_library_revision", "signal_evidence", "ranked_similarities", "abstention_reason", "calibrated_probabilities", "official_reveal_label"],
    "GateEvidence": ["gate_id", "source_revision", "manifest_hash", "commands", "exit_statuses", "collected", "passed", "failed", "skipped", "artifacts", "review_decision"]
  }
}
```
