# Upstream inventory

Every benchmark StealthBench wraps is listed here with the exact revision, license,
supported evaluation mode and acquisition condition observed on **2026-10-02**. This
file is the input to `tests/contract/test_upstream_inventory.py`, which fails if an
entry is incomplete or if the prose and the machine-readable block disagree.

Three rules govern this inventory:

1. **Verified or absent.** A fact is recorded only if it was read from an official
   source. Anything else is the literal string `unverified`, never a plausible guess.
   `pinned_revision: null` is allowed but must carry an `unpinned_reason`.
2. **No silent substitution.** If an asset cannot be obtained, the acquisition status
   says so and names the operational condition that would unblock it. Another dataset
   is never substituted to keep a gate moving.
3. **The official grader is the oracle.** Where an official protocol contains a
   behaviour StealthBench refuses to reproduce, the difference is recorded in
   `stealthbench_divergence` and implemented deliberately in the adapter task.

Licensing is recorded because it was read from upstream, not as a gate: the operator
decided on 2026-10-02 that it does not block acquisition. Every component is fetched
and evaluated locally, and whether item-level content is republished later is a
publication-time decision, not a build condition. `acquisition.status` therefore
reflects only whether the assets can actually be obtained and run.

## Summary

| Component | Track | Pinned revision | Code license | Acquisition |
| --- | --- | --- | --- | --- |
| [IFEval](#ifeval) | direct | commit `e49bbfe3` (no tags exist) | Apache-2.0 | available |
| [MMLU-Pro](#mmlu-pro) | direct | commit `f418b116` (no tags exist) | Apache-2.0 | available |
| [MATH-500](#math-500) | direct | data rev `6e4ed1a2` | n/a (data only) | available |
| [LiveCodeBench](#livecodebench) | direct | commit `28fef95e` (no tags exist) | MIT | available |
| [EvalPlus](#evalplus) | direct | data revs `v0.1.10` / `v0.2.0` | Apache-2.0 | available |
| [BFCL](#bfcl) | direct | commit `f7cf7359` | Apache-2.0 | conditional |
| [RULER](#ruler) | direct | branch `rulerv1-ns` @ `e8bbff67` | Apache-2.0 | conditional |
| [SWE-bench Verified](#swe-bench-verified) | agent | harness `02e7a74f` | MIT | conditional |
| [Terminal-Bench](#terminal-bench) | agent | tag `v4.0.0` @ `452bf305` | Apache-2.0 | conditional |

Every component can be fetched and run today. Four remain `conditional` on
operational grounds only — a container runtime, Harbor Hub egress, git-lfs, or a
SerpAPI key — each naming the specific thing that is missing. Licensing is recorded
per component below because it was read from upstream, but it is **not** treated as an
acquisition blocker: all nine are evaluated locally, and whether any task content is
later republished is the operator's call at publication time, not a gate condition.

## Machine-readable inventory

The block below is the authoritative structured form; the prose sections add detail.

```json upstream-inventory
{
  "schema_version": "1.0",
  "verified_on": "2026-10-02",
  "components": [
    {
      "id": "ifeval",
      "name": "IFEval",
      "role": "direct",
      "category": "instruction_following",
      "repo": "https://github.com/google-research/google-research/tree/master/instruction_following_eval",
      "pinned_revision": "e49bbfe381c9c0e564b937f1c4e163a2273c65cc",
      "pinned_revision_kind": "commit",
      "unpinned_reason": null,
      "dataset": {
        "id": "google/IFEval",
        "split": "train",
        "item_count": 541,
        "revision": "sha966cd89545d6b6acfd7638bc708b98261ca58e84",
        "in_repo_path": "instruction_following_eval/data/input_data.jsonl",
        "redistributable": true
      },
      "evaluator": {
        "module": "instruction_following_eval/evaluation_main.py",
        "command": "python3 -m instruction_following_eval.evaluation_main --input_data=<f> --input_response_data=<f> --output_dir=<d>",
        "emits": [
          "eval_results_strict.jsonl",
          "eval_results_loose.jsonl"
        ],
        "pinned_golden_fixture": null
      },
      "api_modes": [
        "post_hoc_response_checker"
      ],
      "license": {
        "code": "Apache-2.0",
        "data": "Apache-2.0"
      },
      "install": {
        "requires_python": "unverified",
        "key_packages": [
          "absl-py",
          "langdetect",
          "nltk",
          "immutabledict"
        ],
        "grade_time_gpu": false,
        "grade_time_network": false,
        "notes": "requirements.txt is entirely unpinned, so the grader must be run in a separately pinned environment."
      },
      "eval_constraints": [
        "No subprocess sandbox, wall-clock limit or dataset checksum verification exists in the official tooling.",
        "Response language is detected with langdetect, so grading is not purely lexical.",
        "Loose scoring matches any of 7 response variants (first/last/both line removed, each with '*' stripped)."
      ],
      "stealthbench_divergence": [
        "The official checker is reused unchanged. StealthBench only supplies the request/response pairing and reports strict and loose metrics separately."
      ],
      "acquisition": {
        "status": "available",
        "condition": null
      },
      "sources": [
        "https://github.com/google-research/google-research/tree/master/instruction_following_eval",
        "https://huggingface.co/api/datasets/google/IFEval"
      ]
    },
    {
      "id": "mmlu_pro",
      "name": "MMLU-Pro",
      "role": "direct",
      "category": "general_knowledge",
      "repo": "https://github.com/TIGER-AI-Lab/MMLU-Pro",
      "pinned_revision": "f418b116db00b065c2aea046518d8fcf74d39872",
      "pinned_revision_kind": "commit",
      "unpinned_reason": null,
      "dataset": {
        "id": "TIGER-Lab/MMLU-Pro",
        "split": "test",
        "item_count": 12032,
        "revision": "b189ec765aa7ed75c8acfea42df31fdae71f97be",
        "validation_split": "validation (70 rows, few-shot source)",
        "redistributable": true
      },
      "evaluator": {
        "module": "compute_accuracy.py",
        "command": "python compute_accuracy.py results/<model>/",
        "pinned_golden_fixture": null
      },
      "api_modes": [
        "generate"
      ],
      "license": {
        "code": "Apache-2.0",
        "data": "MIT"
      },
      "install": {
        "requires_python": "unverified",
        "key_packages": [
          "datasets",
          "torch",
          "vllm",
          "transformers"
        ],
        "grade_time_gpu": false,
        "grade_time_network": false,
        "notes": "requirements.txt hard-pins torch/vllm for the local-generation script. compute_accuracy.py itself is CPU only."
      },
      "eval_constraints": [
        "No likelihood path exists upstream: correctness is regex-extracted from generated text, so loglikelihood scoring cannot be reproduced from a text-only endpoint.",
        "Reference generation settings are max_model_length 4096, max_new_tokens 2048, temperature 0, stop ['Question:'].",
        "Output-token truncation yields unparsable answers, which the official scripts partially compensate for with --rerun-unknown."
      ],
      "stealthbench_divergence": [
        "The official extractor falls back to a uniformly random letter A-J when no answer can be parsed, which manufactures score. StealthBench records extraction failure as an invalid response and reports it in the denominator instead of guessing."
      ],
      "acquisition": {
        "status": "available",
        "condition": null
      },
      "sources": [
        "https://github.com/TIGER-AI-Lab/MMLU-Pro",
        "https://huggingface.co/api/datasets/TIGER-Lab/MMLU-Pro"
      ]
    },
    {
      "id": "math500",
      "name": "MATH-500",
      "role": "direct",
      "category": "mathematical_reasoning",
      "repo": "https://huggingface.co/datasets/HuggingFaceH4/MATH-500",
      "pinned_revision": "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be",
      "pinned_revision_kind": "dataset_revision",
      "unpinned_reason": null,
      "dataset": {
        "id": "HuggingFaceH4/MATH-500",
        "split": "test",
        "item_count": 500,
        "revision": "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be",
        "provenance": "500 MATH problems selected by OpenAI for 'Let's Verify Step by Step' (prm800k)",
        "redistributable": false
      },
      "evaluator": {
        "module": "unverified (no dedicated scorer module ships with the dataset)",
        "command": "lighteval task math_500 --metrics pass_at_k_math",
        "pinned_golden_fixture": null
      },
      "api_modes": [
        "generate"
      ],
      "license": {
        "code": "MIT (lighteval)",
        "data": "unverified"
      },
      "install": {
        "requires_python": ">=3.10",
        "key_packages": [
          "lighteval",
          "inspect-ai",
          "latex2sympy2_extended==1.0.6",
          "nltk"
        ],
        "grade_time_gpu": false,
        "grade_time_network": false,
        "notes": "lighteval's math_500 task also declares an LLM-judge metric (model_graded_fact) which must not be used for a comparable score."
      },
      "eval_constraints": [
        "latex2sympy2_extended symbolic equivalence parsing is unbounded upstream; a malicious or oversized expression must be bounded by StealthBench with an explicit parser time limit (G06).",
        "Reference generation budget is generation_size 32768 tokens.",
        "Answer extraction is not standardized across the dataset; normalization and extraction must be frozen before dispatch."
      ],
      "stealthbench_divergence": [
        "The LLM-judge metric is deliberately not used. Grading uses the official answer-normalization/equivalence procedure with a bounded parser timeout and a recorded numeric tolerance."
      ],
      "acquisition": {
        "status": "available",
        "condition": null
      },
      "sources": [
        "https://huggingface.co/datasets/HuggingFaceH4/MATH-500",
        "https://github.com/huggingface/lighteval/blob/main/src/lighteval/tasks/tasks/math_500.py"
      ]
    },
    {
      "id": "livecodebench",
      "name": "LiveCodeBench",
      "role": "direct",
      "category": "code_generation",
      "repo": "https://github.com/LiveCodeBench/LiveCodeBench",
      "pinned_revision": "28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24",
      "pinned_revision_kind": "commit",
      "unpinned_reason": null,
      "dataset": {
        "id": "livecodebench/code_generation_lite",
        "split": "release_v6",
        "item_count": 1055,
        "revision": null,
        "date_window": "May 2023 to Apr 2025 per README; latest contest_date verified in test6.jsonl is 2025-03-29",
        "full_test_variant": "livecodebench/code_generation (pass --not_fast)",
        "redistributable": false
      },
      "evaluator": {
        "module": "lcb_runner/evaluation/compute_scores.py",
        "command": "python -m lcb_runner.evaluation.compute_scores --eval_all_file <f> --start_date 2023-09-01",
        "pinned_golden_fixture": null
      },
      "api_modes": [
        "generate"
      ],
      "license": {
        "code": "MIT",
        "data": "unverified (conflicting declarations)"
      },
      "install": {
        "requires_python": ">=3.10",
        "key_packages": [
          "torch>=2.3.0",
          "vllm>=0.5.0.post1",
          "datasets>=3.2.0",
          "openai>=1.59.6"
        ],
        "grade_time_gpu": false,
        "grade_time_network": false,
        "notes": "Default local serving path needs a GPU; use --multiprocess for closed APIs. Evaluation itself is CPU subprocesses."
      },
      "eval_constraints": [
        "Default --timeout is 6 seconds. Upstream warns that time limits alone move pass@1 by more than 0.5 points, so the timeout is recorded per run and never tuned after seeing scores.",
        "--openai_timeout defaults to 90s for API requests; --num_process_evaluate defaults to 12.",
        "Default dataset is code_generation_lite with pruned tests; the leaderboard is mid-migration between lite and full test sets.",
        "ERRATA.md documents erroneous tests and problems not amenable to autograding; those item IDs must be declared, not silently dropped.",
        "No dataset checksum verification exists upstream, so StealthBench records its own checksum over the selected files."
      ],
      "stealthbench_divergence": [
        "Generation defaults upstream are n=10, temperature=0.2. pass@1 is reported from accepted samples only; StealthBench never selects the best of several attempts and labels it pass@1."
      ],
      "acquisition": {
        "status": "available",
        "condition": null
      },
      "sources": [
        "https://github.com/LiveCodeBench/LiveCodeBench",
        "https://huggingface.co/datasets/livecodebench/code_generation_lite"
      ]
    },
    {
      "id": "evalplus",
      "name": "EvalPlus",
      "role": "direct",
      "category": "code_generation",
      "repo": "https://github.com/evalplus/evalplus",
      "pinned_revision": "v0.1.10 (HumanEval+) / v0.2.0 (MBPP+) dataset releases",
      "pinned_revision_kind": "dataset_release",
      "unpinned_reason": "Dataset identity is pinned by data release rather than by harness commit: HUMANEVAL_PLUS_VERSION and MBPP_PLUS_VERSION are the values the official loader checksums and caches on.",
      "dataset": {
        "id": "evalplus/humanevalplus",
        "split": "test",
        "item_count": 164,
        "revision": "v0.1.10",
        "companion": {
          "id": "evalplus/mbppplus",
          "item_count": 378,
          "revision": "v0.2.0"
        },
        "redistributable": true
      },
      "evaluator": {
        "module": "evalplus/eval/__init__.py (untrusted_check, estimate_pass_at_k)",
        "command": "evalplus.evaluate --dataset humaneval --samples <f>",
        "pinned_golden_fixture": null
      },
      "api_modes": [
        "generate"
      ],
      "license": {
        "code": "Apache-2.0",
        "data": "Apache-2.0"
      },
      "install": {
        "requires_python": ">=3.9",
        "key_packages": [
          "evalplus",
          "tree_sitter",
          "datasets",
          "psutil"
        ],
        "grade_time_gpu": false,
        "grade_time_network": true,
        "notes": "First grading run downloads the dataset tarball from GitHub releases; the md5 then keys the cached ground-truth outputs."
      },
      "eval_constraints": [
        "Per-task timeout T = max(T_base, T_gt * k) with T_base 4s and k 4 by default, where T_gt is the profiled ground-truth runtime.",
        "Memory is capped per process at min(4GB, system maximum); EVALPLUS_MAX_MEMORY_BYTES overrides, -1 means unlimited.",
        "Timeouts and out-of-memory are graded as failures, not as missing data.",
        "No container is enforced by default. docs/execution.md recommends Docker and calls native execution unsafe.",
        "MD5 checksum verification of the dataset is present upstream, so dataset identity is verifiable."
      ],
      "stealthbench_divergence": [
        "Execution runs inside the disposable sandbox backend (G07) instead of the host process, with the same timeout and memory policy recorded explicitly."
      ],
      "acquisition": {
        "status": "available",
        "condition": null
      },
      "sources": [
        "https://github.com/evalplus/evalplus/blob/master/docs/execution.md",
        "https://huggingface.co/api/datasets/evalplus/humanevalplus"
      ]
    },
    {
      "id": "bfcl",
      "name": "BFCL",
      "role": "direct",
      "category": "tool_use",
      "repo": "https://github.com/ShishirPatil/gorilla/tree/f7cf7359b7ac615a0b294831c5ba2bc95ee4a000/berkeley-function-call-leaderboard",
      "pinned_revision": "f7cf7359b7ac615a0b294831c5ba2bc95ee4a000",
      "pinned_revision_kind": "commit",
      "unpinned_reason": null,
      "dataset": {
        "id": "gorilla-llm/Berkeley-Function-Calling-Leaderboard",
        "split": "BFCL_v4 per-category JSONL files",
        "item_count": 5088,
        "revision": null,
        "scored_composition": "Agentic 665 (40%), Multi-Turn 800 (30%), Live 1351 (10%), Non-Live 1150 (10%), Hallucination 1122 (10%)",
        "non_scoring_item_count": 5218,
        "redistributable": true
      },
      "evaluator": {
        "module": "bfcl_eval/eval_checker/eval_runner.py",
        "command": "bfcl evaluate --model <m> --test-category <c>",
        "pinned_golden_fixture": null
      },
      "api_modes": [
        "native_function_calling",
        "prompted"
      ],
      "license": {
        "code": "Apache-2.0",
        "data": "Apache-2.0"
      },
      "install": {
        "requires_python": ">=3.10",
        "key_packages": [
          "bfcl-eval",
          "tree_sitter",
          "pandas",
          "pydantic"
        ],
        "grade_time_gpu": false,
        "grade_time_network": true,
        "notes": "BFCL_PROJECT_ROOT is required for PyPI installs. Local-model serving needs a GPU, but scoring a saved response file does not."
      },
      "eval_constraints": [
        "Native function calling and prompted calling are separate leaderboard rows. Averaging them into one number would misrepresent the API mode.",
        "In the official summary columns, unevaluated categories are counted as 0 rather than excluded, so a partial run understates the score.",
        "The web_search category requires a SerpAPI key and real network access, making it nondeterministic; the V4 blog describes a DuckDuckGo backend while the pinned code imports serpapi.",
        "format_sensitivity is non-scoring and supported for prompted models only.",
        "The 'Restful'/executable categories were retired from the leaderboard and must not be reported.",
        "Evaluation temperature is fixed at 0.001 upstream; --partial-eval results do not match the leaderboard."
      ],
      "stealthbench_divergence": [
        "An unsupported category is reported as unavailable with its denominator shown, never as a zero that shrinks the mean. Native and prompted profiles are reported as two separate tracks (G08)."
      ],
      "acquisition": {
        "status": "conditional",
        "condition": "Scoring a saved response file needs BFCL_PROJECT_ROOT and the pinned commit but no GPU. The agentic web_search categories additionally need a SerpAPI credential and live network; without them those categories are unavailable rather than scored."
      },
      "sources": [
        "https://github.com/ShishirPatil/gorilla/blob/f7cf7359b7ac615a0b294831c5ba2bc95ee4a000/berkeley-function-call-leaderboard/TEST_CATEGORIES.md",
        "https://gorilla.cs.berkeley.edu/leaderboard.html"
      ]
    },
    {
      "id": "ruler",
      "name": "RULER",
      "role": "direct",
      "category": "long_context",
      "repo": "https://github.com/NVIDIA/RULER/tree/rulerv1-ns",
      "pinned_revision": "e8bbff677ca2c239640dc90f93310dcf32408c93",
      "pinned_revision_kind": "commit",
      "unpinned_reason": "RULER publishes no release tags. The pipeline also lives outside this repo: the rulerv1-ns and rulerv2-ns branches carry only a README and run_example.sh that drive NVIDIA-NeMo/Skills, and the RULER main-branch evaluation path is marked deprecated upstream. The README instructs users to clone the NeMo-Skills branch chsieh/ruler-remove-prefix.",
      "dataset": {
        "id": null,
        "split": "<task>/test.jsonl",
        "item_count": 1300,
        "revision": null,
        "generation_seed": 42,
        "sample_count_per_task": 100,
        "task_count": 13,
        "context_lengths": [
          4096,
          8192,
          16384,
          32768,
          65536,
          131072
        ],
        "redistributable": false
      },
      "evaluator": {
        "module": "nemo_skills/dataset/ruler/ruler_score.py",
        "command": "ns eval --benchmarks=ruler.<model>-<len>",
        "pinned_golden_fixture": null
      },
      "api_modes": [
        "generate"
      ],
      "license": {
        "code": "Apache-2.0",
        "data": "unverified (inherited from embedded corpora)"
      },
      "install": {
        "requires_python": "unverified",
        "key_packages": [
          "nemo-skills",
          "wonderwords",
          "html2text",
          "tenacity",
          "nltk"
        ],
        "grade_time_gpu": false,
        "grade_time_network": true,
        "notes": "git-lfs is mandatory for the cwe task. Preparing data clones the RULER repo and downloads PaulGrahamEssays, SQuAD, HotpotQA and word lists."
      },
      "eval_constraints": [
        "Prompts are tokenized per model, so RULER inputs are equal in token length but are NOT identical strings across models. Cross-model comparison must state this.",
        "RULERv1 (13 tasks) and RULERv2 (12 tasks) are different pipelines with different scorers; exactly one must be pinned.",
        "Aggregation is an unweighted mean over per-task accuracies, so a task that cannot be generated must be unavailable, not zero.",
        "Model context limits must be raised to the target length or requests truncate; a truncated task must never be reported at its intended length.",
        "Upstream sampling is greedy: temperature 0.0, top_p 1.0."
      ],
      "stealthbench_divergence": [
        "StealthBench pins rulerv1-ns, records the per-task generation seed and the template-token overhead separately from the provider's reported token count, and reports truncation explicitly."
      ],
      "acquisition": {
        "status": "conditional",
        "condition": "Data is generated rather than hosted, so preparing it needs a git clone of NVIDIA/RULER, the corpus downloads in the NeMo-Skills prepare step, and git-lfs for the cwe task."
      },
      "sources": [
        "https://github.com/NVIDIA/RULER/blob/rulerv1-ns/README.md",
        "https://raw.githubusercontent.com/NVIDIA-NeMo/Skills/main/nemo_skills/dataset/ruler/prepare.py"
      ]
    },
    {
      "id": "swebench",
      "name": "SWE-bench Verified",
      "role": "agent",
      "category": "agent_repository",
      "repo": "https://github.com/SWE-bench/SWE-bench",
      "pinned_revision": "02e7a74ffd0b707aab73d203fe87bdc7c76afc8e",
      "pinned_revision_kind": "commit",
      "unpinned_reason": null,
      "dataset": {
        "id": "SWE-bench/SWE-bench_Verified",
        "split": "test",
        "item_count": 500,
        "revision": null,
        "released": "2024-08-13",
        "supersedes": [
          "SWE-bench test (2294)",
          "SWE-bench Lite (300)"
        ],
        "image_pattern": "swebench/sweb.eval.<arch>.<repo>_<version>_<instance_id>:latest",
        "image_pin_warning": "The ':latest' registry tag is mutable, so image identity must be recorded as a digest at run time.",
        "redistributable": true
      },
      "evaluator": {
        "module": "swebench/harness/grading.py via swebench.harness.run_evaluation",
        "command": "swebench eval verified -p <predictions> --run-id <id> -j <n>",
        "pinned_golden_fixture": null
      },
      "api_modes": [
        "agent_trajectory_only"
      ],
      "license": {
        "code": "MIT",
        "data": "MIT (project statement; the dataset repos declare no license tag)"
      },
      "install": {
        "requires_python": ">=3.10",
        "key_packages": [
          "swebench==5.0.2",
          "datasets",
          "docker",
          "GitPython",
          "unidiff"
        ],
        "grade_time_gpu": false,
        "grade_time_network": true,
        "notes": "Docker is mandatory for evaluation. Upstream recommends an x86_64 host with 120GB free storage, 16GB RAM and 8 cores."
      },
      "eval_constraints": [
        "RESOLVED_FULL requires every FAIL_TO_PASS and every PASS_TO_PASS test to pass.",
        "Six JavaScript repositories are hardcoded to fail-only grading because their reporter only records failing tests.",
        "The harness caches results by run_id and instance_id only. Re-grading a changed patch requires a new run_id, otherwise a stale verdict is silently reused.",
        "A failed evaluation is not a crash: zero failures alongside a nonzero error count means instances could not be graded and must be read individually.",
        "arm64 is experimental and requires the task-repo plus Docker Buildx path.",
        "Worker count should stay below min(0.75 * cpu_count, 24)."
      ],
      "stealthbench_divergence": [
        "The harness never calls a model API; it grades patches produced elsewhere. StealthBench runs one complete agent trajectory per instance and treats that trajectory as a single attempt."
      ],
      "acquisition": {
        "status": "conditional",
        "condition": "Requires a container runtime plus roughly 120GB of image storage and network access to the image registry. Pinned instance images are recorded by digest because the ':latest' tag is mutable."
      },
      "sources": [
        "https://huggingface.co/api/datasets/SWE-bench/SWE-bench_Verified",
        "https://raw.githubusercontent.com/SWE-bench/SWE-bench/main/swebench/harness/grading.py"
      ]
    },
    {
      "id": "terminalbench",
      "name": "Terminal-Bench",
      "role": "agent",
      "category": "agent_terminal",
      "repo": "https://github.com/harbor-framework/terminal-bench/tree/v4.0.0",
      "pinned_revision": "452bf305c6daa62fc59061d22133a7cbc7c1572e",
      "pinned_revision_kind": "tag",
      "unpinned_reason": null,
      "dataset": {
        "id": "terminal-bench/terminal-bench",
        "split": null,
        "item_count": 66,
        "revision": "4.0.0",
        "hosting": "Harbor Hub dataset package; tasks/dataset.toml carries a sha256 digest per task",
        "redistributable": false
      },
      "evaluator": {
        "module": "tests/test_scoring.py in a separate verifier container",
        "command": "harbor run -d terminal-bench/terminal-bench@4.0.0",
        "pinned_golden_fixture": null
      },
      "api_modes": [
        "agent_trajectory_only"
      ],
      "license": {
        "code": "Apache-2.0",
        "data": "unverified (no license grant; explicit no-training restriction)"
      },
      "install": {
        "requires_python": ">=3.12",
        "key_packages": [
          "harbor==0.23.0"
        ],
        "grade_time_gpu": false,
        "grade_time_network": true,
        "notes": "The current harness is the harbor CLI. The retired PyPI package 'terminal-bench' (0.2.18) is the old 1.x harness and must not be used. Docker is mandatory; Modal and Daytona are alternatives."
      },
      "eval_constraints": [
        "Every task in 4.0 has a flat 8-hour agent timeout; the verifier timeout is 300s for the inspected task. These vary per task and changed in 4.0, so 3.0 results are not comparable.",
        "Reward is binary per task: reward 1.0 only when every claim is correct, written to /logs/verifier/reward.txt and reward.json.",
        "The verifier runs in a separate container, so the agent cannot reach verifier code or the reward channel.",
        "Some tasks do use an LLM judge. Whether a given task does must be read from task.toml [metadata].verification_explanation; a full survey is unverified.",
        "Verifier tampering is an acknowledged upstream risk with a 4.1 remediation milestone; verifier behaviour is version-specific.",
        "Task data carries a canary GUID and must never appear in training corpora."
      ],
      "stealthbench_divergence": [
        "Tasks are identified by their per-task sha256 digest so the task set cannot drift under a moving version tag."
      ],
      "acquisition": {
        "status": "conditional",
        "condition": "Requires a container runtime and network access to Harbor Hub to resolve terminal-bench/terminal-bench@4.0.0 and build or pull the task images."
      },
      "sources": [
        "https://raw.githubusercontent.com/harbor-framework/terminal-bench/v4.0.0/tasks/dataset.toml",
        "https://hub.harborframework.com/datasets/terminal-bench/terminal-bench/4"
      ]
    }
  ]
}
```

## Detail

### IFEval

Post-hoc response checker, not a harness. Prompts and responses are supplied by the
caller as `{key, prompt, response}` JSONL; the official `evaluation_main.py` writes
strict and loose result files and prints both accuracies. The registry holds 29
verifiable instruction types. Loose scoring accepts any of 7 response variants, so
strict and loose must be reported as separate numbers.

There is no release tag at all, so the repository is pinned by commit. Grading runs
offline with no GPU.

### MMLU-Pro

12,032 test questions over 14 categories with up to 10 options, plus a 70-row
validation split used for few-shot prompting. The upstream answer extractor is a
seeded regex cascade; on total extraction failure it substitutes a uniformly random
letter, which turns an unparsable response into a 10% expected score. StealthBench
treats that case as an invalid response instead. Generation is text-only, so no
likelihood-based score is reproducible.

### MATH-500

500 problems with `problem`, `solution`, `answer`, `subject`, `level`, `unique_id`.
No license is declared anywhere on the dataset repository. That does not affect local
evaluation. The referenced harness task also registers an LLM-judge metric; that judge
is not used here because it is neither deterministic nor free, and its use would make
the score incomparable.

The symbolic equivalence path (`latex2sympy2_extended`) parses arbitrary LaTeX. It is
unbounded upstream, so StealthBench imposes a parser time limit and treats a timeout
as a graded failure with the reason recorded.

### LiveCodeBench

Release windows v1–v6 accumulate problems with `contest_date` tags; `release_v6` is
the latest window. Scores are sliced by date with `--start_date` / `--end_date`, and
the paper convention is to report problems released after 2023-09-01. The benchmark
name does not imply recent problems: the newest contest date verified inside
`test6.jsonl` is 2025-03-29.

Correctness comes from a modified copy of the `apps` checker vendored inside
`lcb_runner`. The 6-second default timeout is documented upstream as worth more than
0.5 points of pass@1, so it is recorded per run and never tuned after seeing results.
The repository has been inactive since 2025-07-16 and publishes no tags.

### EvalPlus

HumanEval+ (164 tasks) and MBPP+ (378 tasks), roughly 80x and 35x the original test
counts. The dataset is not fetched from Hugging Face by the harness; it downloads
release tarballs from GitHub and caches ground-truth outputs under an md5 of the
dataset, which is the only dataset checksum verification found in any of these
components.

`pass@1` is the unbiased estimator `1 - C(n-c, k) / C(n, k)`, and with the default
single greedy sample it reduces to single-sample accuracy. Timeouts and out-of-memory
count as failures. `docs/execution.md` states that execution is unsafe without a
container, which is why G07 builds the sandbox backend before the wrapper.

### BFCL

Lives inside the `gorilla` monorepo, not a standalone repository; the leaderboard
states models are evaluated at commit `f7cf735`. Two things about the current version
surprise implementers:

- Category names moved. `simple` became `simple_python` / `simple_java` /
  `simple_javascript`; `Relevance` became the scored `irrelevance` /
  `live_irrelevance` pair plus a non-scoring `live_relevance`; the old Restful and
  executable categories were retired outright.
- `agentic` is 40% of the score and requires live network access plus a SerpAPI key.

The dangerous default for us is the summary-column behaviour: unevaluated categories
count as zero rather than being excluded. Copying that convention would let a missing
category silently deflate or distort an endpoint, so G08 reports unavailable
categories with their denominators instead.

### RULER

Prompts are generated, not hosted. `niah`, `variable_tracking`,
`common_words_extraction`, `freq_words_extraction` and `qa` tasks are synthesised
programmatically — no LLM generates the haystacks — from Paul Graham essays, SQuAD,
HotpotQA and word lists. Seed 42 by default, per-chunk offset by the chunk index.

The single most important caveat: because haystack size is chosen by binary search
against each model's tokenizer, the same RULER task produces *different strings* for
different models that are only equal in token length. RULERv1 and RULERv2 are separate
pipelines with separate scorers, and the `main` branch pipeline is deprecated upstream.

The `rulerv1-ns` branch itself holds no evaluation code: `.gitattributes`,
`.gitignore`, `LICENSE`, a README and `run_example.sh`. The README is a NeMo-Skills
instruction page that tells you to clone the NeMo-Skills branch
`chsieh/ruler-remove-prefix`, install it, then call `ns prepare_data ruler` and
`ns eval`. Pinning RULER therefore means pinning two repositories, not one.

### SWE-bench Verified

500 human-verified instances released 2024-08-13, superseding both the original 2,294
-item test set and Lite. Grading requires every FAIL_TO_PASS and every PASS_TO_PASS
test to pass, and six JavaScript repositories use fail-only grading because their test
reporter only records failures.

Two operational traps: the harness caches verdicts by `run_id` + `instance_id`, so a
changed patch under the same run id silently reuses the old verdict; and image tags
end in `:latest`, so image identity is not pinned by the reference itself.

Tests are never shown to the agent. The container needs roughly 120GB, and the
harness never contacts a model API — it grades patches produced elsewhere.

### Terminal-Bench

Current release is 4.0.0, hosted as a Harbor dataset package and driven by
`harbor run -d terminal-bench/terminal-bench@4.0.0`. Each of the 66 tasks carries a
sha256 digest in `tasks/dataset.toml`, which is how the task set stays stable.

The verifier is a separate container that runs `tests/test_scoring.py` against a
verifier-owned `expected.json` and writes a binary reward to
`/logs/verifier/reward.txt`. Not every task is judge-free; the `task.toml` metadata
says which are. 4.0 applied a flat 8-hour agent timeout to every task and changed
per-task resources, both of which upstream calls breaking changes requiring re-runs.

No data license grant was found, and every task file states that the benchmark data
must never appear in training corpora, backed by a canary GUID. Worth knowing before
any task content is republished; irrelevant to running the tasks locally.

## Consequences for the build sequence

- **G06** must implement MMLU-Pro extraction *without* the random-letter fallback, and
  must bound MATH-500 symbolic parsing.
- **G07** must land the sandbox backend before EvalPlus and LiveCodeBench wrappers,
  because EvalPlus explicitly declines to sandbox by default and LiveCodeBench's
  checker executes candidate code.
- **G08** must keep BFCL native and prompted modes as separate tracks and must never
  copy the unevaluated-category-counts-as-zero convention.
- **G08** must record RULER truncation explicitly and pin exactly one of v1/v2.
- **G10** must pin Terminal-Bench task digests, run a new SWE-bench `run_id` for any
  re-grade, and record image digests rather than tags.
- **G14** is unblocked on data acquisition: all nine components can be fetched and
  run. If item-level content is ever published rather than just aggregate scores, that
  is the point at which the four unverified data licenses become the operator's
  problem to weigh.
