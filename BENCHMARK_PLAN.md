# StealthBench: benchmark and identity estimation plan

Research date: 2 October 2026. Status: proposed design; no model evaluations or paid API calls have been run.

## Objective and scope

Build a reproducible benchmark for anonymous models served through OpenCode Zen, alongside an evidence-based estimator of their likely model family or candidate identity. Track every anonymous alias over time, including after a provider reveals its identity.

The benchmark should answer three questions: how capable is this endpoint, how well does it work as an OpenCode coding agent, and which known models have similar observable signatures? Identity estimates must remain separate from capability scores.

OpenCode's documentation currently labels Big Pickle (`big-pickle`) and Space Bunny Free (`space-bunny-free`) as stealth models. It documents the catalog at `https://opencode.ai/zen/v1/models` and chat completions for these aliases at `https://opencode.ai/zen/v1/chat/completions`. Recheck the catalog before each evaluation campaign; an alias is not an immutable model version. OpenCode provides the gateway, and an anonymous model may come from another developer. [OpenCode Zen documentation](https://opencode.ai/docs/zen/)

## What existing benchmark sites do

| Reference | Evaluation approach | What StealthBench should adopt |
| --- | --- | --- |
| Artificial Analysis | Versioned capability index with benchmark-specific protocols; independent API speed measurements; some private evaluations | Versioned protocols, category scores, cost accounting and separate quality/speed reporting |
| Arena, formerly Chatbot Arena/LMArena | Blind comparisons rated by people; statistical preference rankings | Optional blind coding comparisons, randomized presentation, confidence intervals |
| Stanford HELM | Controlled adaptation across scenarios with multiple metrics | Shared prompts and controlled settings, explicit coverage and limitations |
| LiveCodeBench | Code tasks with release dates and several coding scenarios | Executable grading, pinned releases and explicit problem date windows |
| EvalPlus | More extensive tests for HumanEval and MBPP; pass@1 | A compact correctness baseline graded by tests |
| SWE-bench | Repository issue resolution; Verified includes 500 human-filtered tasks; a shared agent view is available | Real repository work, isolated environments, resolved-task scoring and fixed agent configuration |
| Terminal-Bench | Terminal-based agent tasks; current site presents version 4.0, resolution rate, costs and confidence intervals | Fixed task version and execution environment, task success, cost and time |
| Berkeley Function Calling Leaderboard | Function calling and agentic tasks; distinguishes native function calls from prompting alternatives | Valid tool arguments, correct execution and separate reporting of API modes |

Sources: [Artificial Analysis intelligence methodology](https://artificialanalysis.ai/methodology/intelligence-benchmarking), [API performance methodology](https://artificialanalysis.ai/methodology/performance-benchmarking), [Arena paper](https://arxiv.org/abs/2403.04132), [HELM approach](https://crfm.stanford.edu/2022/11/17/helm.html), [LiveCodeBench](https://livecodebench.github.io/), [EvalPlus leaderboard](https://evalplus.github.io/leaderboard.html), [SWE-bench](https://www.swebench.com/), [Terminal-Bench](https://www.tbench.ai/), [BFCL](https://gorilla.cs.berkeley.edu/leaderboard).

Replicate available benchmark protocols when claiming comparability. Build a separately named StealthBench index for our own mixture of tasks. Artificial Analysis currently documents index v4.3.2, including private components, so an exact reproduction of its headline index cannot be assumed possible. Versions and score formulas must be recorded. [Artificial Analysis methodology](https://artificialanalysis.ai/methodology/intelligence-benchmarking)

## Evaluation tracks

### 1. Direct model endpoint

Send benchmark prompts directly to Zen. Use a fixed, minimal system prompt where the benchmark permits one, no tools unless the task requires them, and a new conversation for each item. This measures the observable model endpoint, including any gateway behavior that cannot be removed.

### 2. OpenCode agent

Run repository and terminal tasks using one pinned OpenCode version and configuration. Fix tools, permissions, system prompts, runtime, step limits, context compaction and task budgets. Capture effective requests and any model-dependent prompt selection; a shared configuration file alone does not guarantee identical agent behavior.

Use this track to measure the experience developers obtain with OpenCode. If we reproduce a published mini-SWE-agent or other scaffold, publish that as another explicitly named track. OpenCode exposes programmatic CLI commands and configurable model variants. [CLI documentation](https://opencode.ai/docs/cli/), [model configuration](https://opencode.ai/docs/models/)

### 3. Endpoint performance

Measure first streamed content, first visible answer, total completion time, normalized output rate, failure rate and cost. Reasoning may delay the answer and may not be streamed. An SSE chunk is not necessarily one token.

Run fixed short and medium workloads, with longer workloads conditional on context support. Interleave reference and anonymous endpoints from the same location and across different times. Label concurrency and cache status. Report a common reference-token rate or characters per second for speed comparisons, and use provider token accounting for billing. Artificial Analysis also distinguishes these measurement purposes. [Performance methodology](https://artificialanalysis.ai/methodology/performance-benchmarking)

## Initial benchmark set

Start with a small setup check, then a 600-item pilot. These counts are proposed cost controls, not full benchmark scores.

| Category | Proposed pilot | Scoring and expansion |
| --- | --- | --- |
| Code generation | 100 LiveCodeBench problems + 50 EvalPlus problems | pass@1 from executable tests; expand to complete pinned evaluation splits |
| Mathematical reasoning | 100 MATH-500 problems | Official answer normalization/equivalence procedure; expand to full MATH-500 and report harder math separately if scores saturate |
| Instruction following | 150 IFEval prompts | Official strict and loose metrics; expand to the full official set |
| General reasoning/knowledge | 100 stratified MMLU-Pro questions | Generated-answer accuracy with frozen extraction; expand to the full set |
| Tool use | 50 BFCL cases from declared categories | Official category grading; expand supported categories and preserve native/prompt distinctions |
| Long context | 50 pinned RULER cases across supported lengths | Retrieval/reasoning accuracy by length; report generated task configuration |

Use a frozen sample manifest containing item IDs, revisions and sampling seed. Record difficulty, subject, language and release-date strata where available. Report each component separately. Additional multilingual, JSON schema and private coding tasks are extensions with their own names and scores.

Official resources: [LiveCodeBench code](https://github.com/livecodebench/LiveCodeBench), [EvalPlus](https://evalplus.github.io/), [IFEval](https://github.com/google-research/google-research/tree/master/instruction_following_eval), [MMLU-Pro](https://github.com/TIGER-AI-Lab/MMLU-Pro), [BFCL](https://gorilla.cs.berkeley.edu/leaderboard), [RULER](https://github.com/NVIDIA/RULER). For MATH-500, select and pin the canonical dataset revision and evaluator before starting.

For agent evaluation, add a 20-task setup run followed by 50 preselected repository/terminal tasks. Publish those as pilot subsets. Full SWE-bench Verified and the selected Terminal-Bench version are later release milestones. Code execution must occur in disposable environments with evaluator tests inaccessible to the agent and without host credentials.

Do not assume a benchmark named “Live” still supplies recent problems. Inspect actual task dates. Because anonymous model training cutoffs are unknown, recent problems reduce some contamination risks without proving absence of training exposure.

## Fairness and reproducibility contract

Before collecting scored results, freeze a run manifest with:

- Endpoint, alias, UTC start/end times, catalog snapshot and an internal observation-period ID.
- Dataset revision, selected item IDs, exact prompts, few-shot examples, extraction rules and evaluator commit.
- Requested and observable effective generation settings, output cap, reasoning settings and supported features.
- Agent/scaffold version, tool schemas, container image digest, runtime limits and network policy.
- Retry policy, timeouts, concurrency, cache treatment, scheduling seed and spend limit.

Use benchmark-prescribed settings for faithful reproduction. For a separate controlled comparison, fix supported settings and publish exceptions. A setting called “high” need not represent the same compute across providers. Show both fixed-budget comparisons and clearly labeled recommended configurations if needed.

Begin with one sample per task and repeat a fixed representative subset three times. Repeat more tasks where uncertainty changes a conclusion. Never choose the best of several attempts and label it pass@1. Agent pass@1 refers to one complete trajectory under the declared budget, which can contain many API calls.

Use executable tests and deterministic verifiers wherever available. For subjective tasks, use blind human review; if automated judges are used, disclose their model, prompts, order randomization and agreement with human review.

Transport failures receive bounded retries with all attempts retained. Wrong answers and valid-format failures are scored under task rules. Missing measurements remain missing. Publish response-conditional accuracy, endpoint success rate and end-to-end task success with their denominators. Abort or qualify a run when missingness exceeds its preregistered threshold.

Show 95% intervals and task counts. Use paired comparisons on shared items, and bootstrap at the task level so repetitions of one item are not treated as independent problems. Correct multiple comparisons where making many winner claims. Small pilot subsets support screening; they do not justify precise global rankings.

Build on existing evaluators rather than rewriting them. The EleutherAI harness supports API integrations, but endpoints without suitable log probabilities need generation-based tasks; likelihood-based multiple-choice scores are not automatically reproducible from text completions. [Harness documentation](https://github.com/EleutherAI/lm-evaluation-harness)

## Model identity estimator

### Reference library

Start with 6–10 known model endpoints spanning plausible families, including related versions and at least two models sharing a tokenizer where feasible. Select actual available IDs at campaign start. Expand based on evidence; do not assume an anonymous alias belongs to an already released model.

Run references through Zen when available to control gateway differences. For a smaller subset, collect direct-provider observations as well to estimate route effects. Version public tokenizer assets by revision and chat template. Keep provider identity, family, exact model version, tokenizer and serving route as separate labels.

### Observable signatures

| Signal | Measurement | Interpretation |
| --- | --- | --- |
| Input token counts | Fixed texts containing code, whitespace, punctuation, Unicode, emoji, multilingual text and numeric strings | Evidence about tokenizer and message framing |
| Actual tokenization | Compare token IDs or boundaries only where an endpoint exposes them | More direct tokenizer evidence; generally unavailable from chat text alone |
| Behavior | Fixed harmless tasks, formatting, tool choices, recurring errors and response distributions | Candidate similarity affected by prompting and sampling |
| Capability pattern | Per-item correctness/error overlap, beyond overall benchmark score | Supporting evidence affected by training, tuning and task contamination |
| Protocol | Public response schema, finish reasons, usage fields, tool formatting and observable metadata | Often identifies gateway behavior rather than model developer |
| Performance | Repeated timing and output measurements | Weak identity evidence because hardware, load and routing vary |

Model self-identification is not reliable evidence. Do not build the estimator around “what model are you?” responses.

Begin with 60 signature prompts, repeated three times. Keep them separate from the benchmark and classifier holdout. Use short outputs for token-count probes to control cost. Check whether streaming and non-streaming responses supply usable usage fields.

For tokenizer comparisons, keep message framing constant and measure differences:

`delta(text) = input_tokens(fixed message containing text) - input_tokens(fixed baseline message)`

Compare vectors of deltas against reference endpoints and local candidate tokenizers. Include paired strings and concatenations to expose token boundary effects. Check repeat stability and whether hidden prompt overhead changes. Fixed offsets do not resolve content-dependent templates, tool schema accounting or middleware transformations. If usage is missing, unstable or evidently synthetic, mark this signal unavailable.

An exact count-vector match supports a shared tokenization stack. It cannot establish identical weights: different models can share a tokenizer, and related models can use different serving templates. A recent small holdout study explicitly found this limitation, so its findings support caution rather than a universal identification threshold. [Token Counts Are Not Model Lineage](https://arxiv.org/abs/2608.29930)

### Prediction and validation

Start with transparent nearest-reference similarities per signal and a ranked candidate shortlist. Do not turn a similarity of 0.9 into a 90% probability.

After gathering enough labeled endpoint observations, train a simple classifier and calibrate its outputs on separate data. Evaluate family attribution first. Attempt exact version attribution only if held-out results support it. Group splits by model version and endpoint; reserve different prompt templates, later observation times and alternate routes. Hold out entire families to evaluate the “unknown” outcome.

Avoid learning one provider's gateway schema as if it were model identity. Run ablations with protocol and timing features removed. Evaluate tokenizer-sharing confusions, prompt variation, reasoning settings and missing telemetry. Report macro F1, top-k recall, unknown false-accept rate, calibration and abstention coverage on untouched data.

Set confidence/abstention thresholds using validation data and freeze them before testing. The public output should show:

- Supported tokenizer family or stack, with evidence quality.
- Ranked model-family candidates; calibrated probabilities only after calibration is demonstrated.
- Exact-version estimate only when validation supports that resolution.
- Contradictory evidence, unavailable signals and a clear unknown/insufficient-evidence result.

Confidence is always conditional on the candidate library, experimental conditions and validation coverage. An official reveal is the strongest available identity label, and should be recorded separately from earlier predictions.

## Detecting alias changes

Treat observations as time-bounded snapshots. Save a small fixed canary suite and rerun it daily while an alias is active, using matched reference controls. Track token-count vectors, behavior distributions and repeated task scores.

Preregister change thresholds using stable labeled controls. A change triggers a fresh benchmark snapshot and an alert. It does not automatically prove a weight change: gateway updates, routing, prompt changes, inference settings and load can also shift observations. If results suggest mixed routing, publish uncertainty and observation clusters rather than assuming one stable backend.

After a reveal, preserve the original alias, prediction timestamp and evidence; add the announced model identity and measure whether earlier predictions were accurate.

## Scoreboard and artifacts

The first release should show category scores, per-benchmark metrics, intervals, counts, agent resolution, latency, reliability and actual measured cost. Use separate columns for inferred identity and revealed identity.

If a summary score is useful, name it `StealthBench Core v1`: normalize the six pilot categories to 0–100 and take an equal-weight category mean. Average declared component scores within a category, so additional items do not silently increase its weight. Freeze the formula before runs. Label it a project-defined index; pilot and full-suite versions need different identifiers. Do not calculate the headline score for incomplete core coverage.

Provide item-level results where licenses allow, raw redacted request/response artifacts, evaluator logs, manifests and downloadable JSON/CSV. A model page should include its observation window, capability profile, signature evidence, configuration, coverage and history of changes.

Start with a Python runner, official benchmark adapters, YAML/JSON manifests, append-only JSONL logs and SQLite/Parquet analysis. Generate a static report before building a public web service. Proposed repository layout:

```text
configs/          campaign manifests and endpoint configuration
adapters/         Zen and reference-provider clients
benchmarks/       wrappers around pinned official evaluators
fingerprints/     probe definitions, feature extraction and estimator
analysis/         scoring, intervals, calibration and drift detection
reports/          generated model cards and comparison tables
artifacts/        catalog snapshots, redacted logs and run manifests
```

Only share or redistribute task content when permitted. For non-public holdouts, account for endpoint retention: OpenCode documents possible use of Big Pickle inputs for model improvement. Once sent to such an endpoint, a task should not be assumed permanently unseen. Rotate later holdouts and retain disclosure dates. [Zen data policy](https://opencode.ai/docs/zen/)

## Delivery sequence and estimated effort

These are planning estimates for one developer with working API access and sufficient compute.

| Stage | Approximate effort | Deliverable and completion condition |
| --- | --- | --- |
| 1. Freeze design and discover endpoints | 1–2 days | Available aliases, candidate IDs, supported features, source revisions and manifest agreed |
| 2. Capture endpoint telemetry and signatures | 2–3 days | An 80-item setup check, 60 probes, replayable logs and measured pilot costs |
| 3. Run the core benchmark | 3–5 days plus model runtime | 600-item pilot per endpoint, component metrics, intervals and reliability report |
| 4. Add OpenCode agent evaluation | 3–5 days plus model runtime | Fixed environments and 50-task agent pilot with complete traces |
| 5. Validate identity estimates and publish report | 3–5 days | Held-out attribution results, frozen abstention rules, model pages and downloadable data |
| 6. Expand and monitor | Ongoing | Full benchmark runs, fresh tasks, new candidates, daily canaries and reveal tracking |

An initial useful pilot is roughly a 2–4 week effort. Credible exact-version attribution may need substantially more labeled observations; the estimator should remain a similarity report until those exist.

Budget from measured usage, not assumed free access. For eight endpoints, the 600-item direct pilot needs 4,800 initial generations. The 60-probe suite with three repeats adds 1,440 short calls. Setup, repeated benchmark items, direct-provider controls, speed measurements and retries add more. Fifty agent tasks per endpoint add 400 trajectories, each potentially containing many API calls.

Estimate total cost as the sum of billed input, cached input, output and reasoning tokens under each endpoint's pricing, plus compute/storage and any judging costs. Derive token distributions from Stage 2 and impose per-run caps. Long-context requests and agent trajectories should receive separate budgets.

## Recommended first campaign

Target one currently available anonymous alias and seven labeled reference endpoints. Freeze the manifest; collect the 60 repeated signatures and 80-item setup check; establish actual costs; then run the 600-item core and 50-task agent pilots. Publish category results and an evidence-based candidate shortlist. Advance to full official benchmark runs and calibrated predictions only after the pilot's measurement and validation gates pass.

Decisions needed before execution are the alias, available reference endpoints, API credentials, spending cap and runtime/compute capacity. These do not block the design in this document, but they determine the executable campaign configuration.
