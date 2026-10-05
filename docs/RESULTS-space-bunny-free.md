# space-bunny-free — live benchmark results (finished evidence only)

Campaign date: 2026-10-04. No live requests were run to produce this report.
Every number below is copied from the finished local evidence files listed under
"Evidence sources". Nothing is invented and no value is rounded beyond 3 decimals.

## Evidence sources

- `implementation/evidence/live-ifeval-20261004.json`
  (campaign `live-ifeval-space-bunny-free-20261004`)
- `implementation/evidence/live-suite-20261004.json`
  (campaign `live-suite-space-bunny-free-20261004`)
- `implementation/evidence/live-setup-20261004.json`
  (campaign `live-setup-telemetry-20261004`)
- `implementation/evidence/live-math-full.json`
  (campaign `live-math-full-space-bunny-free`)
- `implementation/evidence/live-mmlu-500.json`
  (campaign `live-mmlu-500-space-bunny-free`)
- `implementation/evidence/live-bfcl-20261004.json`
  (campaign `live-bfcl-space-bunny-free-20261004`)

## Model

- Alias: `space-bunny-free`
- Identity status: unknown stealth alias. No identity mapping is recorded in the
  evidence files. The setup evidence lists `space-bunny-free` in
  `stealth_aliases_present` alongside `big-pickle`, and records
  `catalog_alias_count` of 86.

## Per-benchmark results (finished runs only)

| Benchmark | Score | n | Selection | Generation (temp 0, bare prompt) | Scorer revision | Transport failures | Cost |
| --- | --- | --- | --- | --- | --- | --- | --- |
| IFEval subsample | strict 0.547, loose 0.56 | 150 | sha256(seed:key) rank-order, seed 20261004; dataset string: google/IFEval rev 966cd895, split train (see selection note) | temperature 0.0, system_prompt null, one sample per task, max_tokens 1024 | google-research@e49bbfe3 (official evaluation_main.py) | failed 0 (requested 150, ok 150) | "0" (distinct_cost_values ["0"]) |
| IFEval full set | strict 0.573, loose 0.584 | 541 | Entire official google/IFEval train split (541 prompts); same generation protocol as n=150 run | Same generation protocol as n=150 run (per full-set note) | google-research@e49bbfe3 (official evaluation_main.py) | Not separately recorded in evidence | Not separately recorded in evidence (subsample run: "0") |
| MATH-500 subsample | accuracy 0.57 (correct 57) | 100 | seed_rank true, seed 20261004; dataset HuggingFaceH4/MATH-500 rev 6e4ed1a2 | temperature 0.0, system_prompt null, one sample per task (max_tokens not recorded in evidence) | Frozen exact->numeric->symbolic protocol (per correction note); methods: exact 57, numeric_only 0, symbolic unavailable (no sympy) | transport_failed 0 | spend_usd 0.0 (suite campaign) |
| MMLU-Pro subsample | accuracy_content_only 0.36 (correct 36, invalid_or_wrong 64) | 100 | stratified_14_categories true, seed 20261004; dataset TIGER-Lab/MMLU-Pro test rev b189ec76 | temperature 0.0, system_prompt null, one sample per task, output budget 64 (per rerun note "64->1024") | Content-only extraction; no external scorer revision recorded in evidence | transport_failed 0 | spend_usd 0.0 (suite campaign) |
| MMLU-Pro rerun, larger budget | accuracy_content_only 0.73 (correct 73, empty_content 15) | 100 | Same 100 items/prompts as the 0.36 run (same_selection true) | temperature 0.0, system_prompt null, one sample per task, max_tokens 1024 | Content-only extraction; no external scorer revision recorded in evidence | transport_failed 0 | spend_usd 0.0 (suite campaign) |
| MATH-500 full-500 | accuracy 0.61 (correct 305) | 500 | seed_rank true, seed 20261004; dataset HuggingFaceH4/MATH-500 rev 6e4ed1a2 | temperature 0.0, system_prompt null, one sample per task, max_tokens 1024, boxed_instruction true | Frozen exact->numeric protocol, symbolic unavailable (no sympy); methods: exact 302, numeric_only 3, symbolic unavailable | transport_failed 0 | spend_usd 0.0, credentials_used false |
| MMLU-Pro 500 | accuracy_content_only 0.712 (correct 356) | 500 | stratified_14_categories true, seed 20261004; dataset TIGER-Lab/MMLU-Pro test rev b189ec76 | temperature 0.0, system_prompt null, one sample per task, max_tokens 1024 | Content-only extraction; no external scorer revision recorded in evidence | transport_failed 0 | spend_usd 0.0, credentials_used false |
| BFCL-50 prompted slice | scoring 30/50 eligible, overall_accuracy null by rule | 50 (planned 50, eligible 50) | sha256(seed:item_id) rank-order per category, seed 20261004; dataset gorilla-llm/Berkeley-Function-Calling-Leaderboard; quotas simple_python 9, simple_java 8, simple_javascript 8, multiple 9, parallel 8, irrelevance 8 | system_prompt null, bare user message, temperature 0.0, max_tokens 512, one sample per task | grader bfcl-eval@f7cf7359b7ac615a0b294831c5ba2bc95ee4a000 via stealthbench.benchmarks.bfcl prompted mode; content-only python-format parse via stdlib ast | failed 0 (requested 50, ok 50) | spend_usd 0.0 (distinct_cost_values ["0"]) |

Additional recorded detail:

- IFEval subsample 95% interval for strict: 0.47-0.63 (SE 0.041), as recorded
  in `ci_95_strict`.
- MATH-500 correction note (verbatim): "an earlier 0.80 used symbolic_equals()
  truthiness (it returns a tuple); recomputed per frozen exact->numeric->symbolic
  protocol gives 0.57".
- MMLU-Pro note (verbatim): "58 responses had empty message.content (answer
  confined to reasoning_content); reasoning-inclusive extraction also 0.36".
- MMLU-Pro 1024 note (verbatim): "Same 100 items/prompts as the 0.36 run; only
  the output budget changed (64->1024). Reasoning traces complete and the answer
  letter lands in content."
- Selection note: the IFEval evidence file's `dataset` string reads "n=50 of 541"
  and its `selection` string reads "50 items", while its `scores` (n 150),
  `transport` (requested 150, ok 150), and `disclaimer` ("150-of-541 seeded
  subsample") record 150. All strings are reproduced verbatim above; the scored
  n used in the table is 150.
- MATH-500 full-500: SE 0.022. Disclaimer (verbatim): "Full official split with
  prompted boxed-answer format; grader is repo frozen protocol (exact->numeric,
  symbolic unavailable)." Note (verbatim): "Consistent with the 100-item run
  (0.57) within noise."
- MMLU-Pro 500: SE 0.02. Disclaimer (verbatim): "Stratified subsample with
  zero-shot letter-only prompts; not comparable to published 5-shot CoT
  full-set numbers." Note (verbatim): "Consistent with the 100-item 1024-token
  rerun (0.73); law weakest at 0.39." by_category (verbatim): biology 31/36,
  business 28/36, chemistry 29/36, computer science 28/36, economics 23/36,
  engineering 23/36, health 25/36, history 20/32, law 14/36, math 32/36,
  other 22/36, philosophy 25/36, physics 29/36, psychology 27/36.
- BFCL-50 prompted slice: scoring_correct 30, scoring_eligible 50,
  scoring_planned 50, overall_accuracy null, complete false. Disclaimer
  (verbatim): "50-case prompted-track slice; not comparable to published
  full-set leaderboard numbers; overall_accuracy is null by construction for a
  partial run." per_category: irrelevance correct 5 eligible 8 accuracy 0.625;
  multiple correct 4 eligible 9 accuracy 0.4444444444444444; parallel correct 7
  eligible 8 accuracy 0.875; simple_java correct 6 eligible 8 accuracy 0.75;
  simple_javascript correct 4 eligible 8 accuracy 0.5; simple_python correct 4
  eligible 9 accuracy 0.4444444444444444. All six categories status supported.

## Protocol caveats

- Subsamples vs full sets. The IFEval subsample disclaimer reads: "150-of-541
  seeded subsample, temperature 0; not comparable to published full-set numbers;
  infrastructure-grade check, not a global ranking." The suite disclaimer reads:
  "100-item seeded subsamples; not comparable to published full-set numbers."
  The IFEval full set (541 prompts) covers its official split and the MATH-500
  full-500 covers the full official MATH-500 split per its disclaimer. The
  MMLU-Pro 500 remains a stratified subsample per its disclaimer above, and the
  BFCL-50 remains a 50-case prompted-track partial slice with overall null by
  rule.
- Zero-shot, letter-only MMLU. Generation records system_prompt null with one
  sample per task (bare prompt), and MMLU accuracy is content-only letter
  extraction (`accuracy_content_only`). There is no external scorer revision for
  MMLU in the evidence.
- No sympy for the MATH symbolic path. MATH-500 full methods record exact 302,
  numeric_only 3, and symbolic "unavailable (no sympy)", so the symbolic branch
  contributed nothing to the 0.61. The 100-item subsample likewise records exact
  57, numeric_only 0, symbolic unavailable.
- reasoning_content vs content distinction. Per setup telemetry, reasoning is
  exposed as `reasoning_content` with separate `reasoning_tokens`. In the
  MMLU-Pro 0.36 run, 58 responses had empty `message.content` with the answer
  confined to `reasoning_content`, and reasoning-inclusive extraction also gave
  0.36. The 1024-budget rerun on the same 100 items gave 0.73 with complete
  reasoning traces and the answer letter in content (empty_content 15).
- BFCL strict lower bound. Per BFCL evidence caveats: strict repo-adapter oracle
  (exact Unicode, exact int/float/bool) vs lenient upstream AST checker; expected
  uses the first concrete surface form, so reported accuracies are a lower bound,
  not leaderboard-comparable. Content-only grading ignores reasoning_content; 2
  irrelevance cases returned empty content with finish_reason=length (512-token
  budget exhausted by 511 reasoning tokens) and graded as no calls (pass).
  Agentic/web_search, live, multi-turn, format_sensitivity, memory, and retired
  restful/executable categories were excluded by design, never zeroed.
- Credentials and spend. Suite, math-full, mmlu-500, and setup evidence record
  credentials_used false and spend_usd 0.0. The IFEval run records distinct cost
  values ["0"]. BFCL transport records distinct cost values ["0"] with
  spend_usd 0.0.

## Comparison against published frontier figures (scale context only, not a ranking)

Published figures used here are ONLY the ones listed below (provided for this
report, not measured here). Our numbers alongside are finished-evidence scores
under mismatched protocols, so this is scale context, not a leaderboard claim.

- IFEval strict frontier ~0.93-0.97 (GPT-5.4 Pro 0.97, Opus 4.6 0.95).
  Ours: IFEval full-541 strict 0.573 / loose 0.584; subsample n=150 strict
  0.547 / loose 0.56.
- MMLU-Pro frontier ~0.92-0.94 (GPT-5.5 0.942, Fable 5.1 0.924; DeepSeek V4
  Flash 0.852). Ours: MMLU-Pro 500 accuracy_content_only 0.712 (correct 356).
- MATH-500 frontier ~0.96-0.99 (GPT-5 0.994, Gemini 3 Pro 0.964).
  Ours: MATH-500 full-500 accuracy 0.61 (correct 305).

Protocol caveats for this comparison: our runs are zero-shot bare-prompt
temperature 0.0 one-sample; MMLU is content-only letter extraction with no
external scorer revision; MATH uses the frozen exact->numeric protocol with
symbolic unavailable (no sympy); MMLU-Pro 500 carries the disclaimer "not
comparable to published 5-shot CoT full-set numbers"; the IFEval subsample and
suite subsamples carry "not comparable to published full-set numbers"
disclaimers; BFCL-50 is a partial 50-case prompted-track slice with overall
null by rule and strict repo-adapter lower-bound grading. Published frontier
figures are full-set / few-shot / CoT as published and are therefore not
protocol-comparable to ours.

## PENDING / blocked (no scores)

- Fingerprints comparison — status pending, no score recorded.
- LiveCodeBench — not run, no score recorded, no evidence file. Blocked note:
  code/agent tracks need a container runtime; setup evidence records
  "code/agent tracks: no container runtime on this host".
- EvalPlus — not run, no score recorded, no evidence file. Blocked note: same
  container-runtime block as LiveCodeBench (see setup evidence verbatim above).
- RULER — not run, no score recorded, no evidence file.

## Appendix: free-model census (20261004 probe)

Source: `free_model_probe_20261004` in the IFEval evidence file. Working 1 of 13
probed (8 opencode-console-only 403 + 4 upstream unavailable/500 + 1 working).

- Working: `space-bunny-free`
- opencode_console_only_403: `mimo-v2.6-flash-free`, `longcat-2.5-preview-free`,
  `mimo-v2.5-free`, `ling-3.0-flash-fin-free`, `nemotron-3-ultra-free`,
  `nemotron-3.5-lightning-free`, `fledge-alpha-free`, `ling-3.1-flash-free`
- upstream_unavailable_or_500: `deepseek-v4-flash-free`, `jev-1.13-free`,
  `muse-spark-1.3-contributor-free`, `muse-spark-1.2-contributor-free`

Setup-telemetry notes (same date, verbatim protocol notes):

- Usage reports prompt/completion/total + cached_tokens (128 constant:
  system-prompt cache).
- Reasoning exposed as reasoning_content with separate reasoning_tokens.
- Cost field always "0" on free tier.
- Latencies 1.8-17.6s single-sequential.
- Setup probe of `space-bunny-free`: 11 requests, 11 succeeded,
  exact_checks_passed 2, spend_usd 0.0.
- Setup blocked list (verbatim): "big-pickle: 403 FreeTierError (usable only
  from within OpenCode)", "muse-spark-1.3-contributor-free: 500 internal
  error", "code/agent tracks: no container runtime on this host".
- Setup disclaimer (verbatim): "Setup telemetry only, not a benchmark score.
  Prompts are synthetic; checks use local exact-match rules, not official
  evaluators."
