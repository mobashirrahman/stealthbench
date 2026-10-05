# StealthBench

[![ci](https://github.com/mobashirrahman/stealthbench/actions/workflows/ci.yml/badge.svg)](https://github.com/mobashirrahman/stealthbench/actions/workflows/ci.yml)
![python](https://img.shields.io/badge/python-3.12-blue)
![license](https://img.shields.io/badge/license-Apache--2.0-green)

**A reproducible benchmark harness for anonymous ("stealth") LLM endpoints, built so
that every published number can be traced to a pinned dataset, a seeded selection and
a stored evidence file.**

Model gateways such as OpenCode Zen regularly expose unnamed models under aliases like
`space-bunny-free`. Nobody publishes scores for them, and casual "I tried five prompts"
evaluations are not reproducible. StealthBench measures such an endpoint with official
benchmarks and official graders, and it refuses to report anything it cannot back with
an artifact.

## What was done

- Built a ~24k-line typed Python package: campaign manifests, an endpoint adapter with
  streaming, a deterministic scheduler, a bounded spending ledger, crash recovery,
  append-only event storage with replay, grader wrappers for nine benchmarks,
  fingerprint and similarity analysis, and a static report builder.
- Developed it through 15 gated milestones (G00 to G14), each with recorded evidence
  under [`implementation/evidence/`](implementation/evidence/).
- Ran the first live campaign on 2026-10-04 against the stealth alias
  `space-bunny-free`: 1,591 scored items across four benchmarks, for **$0**, with no
  credentials.

## Results: `space-bunny-free`

All runs use a bare user message, no system prompt, temperature 0 and one sample per
task. Item selection is a `sha256(seed:item_id)` rank order with seed `20261004`.

| Benchmark | Coverage | Score | Scorer |
| --- | --- | --- | --- |
| **IFEval** | full official set, n=541 | **0.573** strict, 0.584 loose | official `google-research@e49bbfe3` |
| **MATH-500** | full official set, n=500 | **0.610** (SE 0.022) | frozen exact, then numeric match |
| **MMLU-Pro** | stratified over 14 categories, n=500 | **0.712** (SE 0.020) | letter extraction from `content` only |
| **BFCL** | prompted slice, n=50 | **30/50** | `bfcl-eval@f7cf7359`, strict adapter |

No transport failures were recorded in any run, and every recorded cost field was `"0"`.
(The full IFEval run did not log transport separately from its 150-item pilot, which
recorded 150 of 150.)

MMLU-Pro ranges from 32/36 in math and 31/36 in biology down to 14/36 in law. BFCL
ranges from 7/8 on parallel calls to 4/9 on simple Python and multiple-function calls.
The BFCL figure is a lower bound: the adapter requires exact types and exact Unicode
where the upstream checker is lenient, and a 50-case slice has no overall accuracy by
rule.

These scores are not leaderboard-comparable. They are zero-shot, single-sample runs,
while published MMLU-Pro figures are typically 5-shot chain-of-thought. The full
per-run detail and caveats are in
[`docs/RESULTS-space-bunny-free.md`](docs/RESULTS-space-bunny-free.md), with the
machine-readable version in
[`results/space-bunny-free/summary.json`](results/space-bunny-free/summary.json).

### Findings beyond the scores

- **Token budget can halve a reasoning model's score.** With a 64-token output budget,
  MMLU-Pro scored 0.36: 58 of 100 responses had empty `content` because the budget was
  spent on `reasoning_content`. The same 100 items at 1,024 tokens scored 0.73. A
  harness that ignores this reports a measurement artifact as a capability gap.
- **The evidence trail caught a grading bug.** An early MATH-500 figure of 0.80 came
  from treating a tuple return value as truthy. Recomputing under the frozen protocol
  gave 0.57, and the correction is recorded in the evidence file instead of being
  silently overwritten.
- **Most "free" aliases are not reachable.** Of 13 free aliases probed, 1 worked from
  outside the OpenCode client, 8 returned 403 and 4 returned upstream errors.
- **Pilots predicted the full runs.** The 100-item pilots (MATH 0.57, MMLU-Pro 0.73)
  landed within noise of the 500-item runs (0.61, 0.712).

### Not yet run

LiveCodeBench, EvalPlus, SWE-bench and Terminal-Bench need a container runtime that
the campaign host did not have. RULER and the identity fingerprint comparison have
adapters and tests but no live run. The identity of `space-bunny-free` remains unknown,
and nothing here claims otherwise.

## Quick start

Requires Python 3.12. The offline path needs no credentials, network or container
runtime.

```bash
git clone https://github.com/mobashirrahman/stealthbench.git && cd stealthbench
python3 -m venv .venv
.venv/bin/pip install -c constraints-dev.txt -e '.[dev]'
source .venv/bin/activate

stealthbench doctor                                    # what this host can run
stealthbench manifest validate configs/offline-demo.json
stealthbench run configs/offline-demo.json --offline --output artifacts/offline-demo
stealthbench replay artifacts/offline-demo             # regrade from stored events
stealthbench report artifacts/offline-demo --output reports/offline-demo
```

Run the checks that CI runs:

```bash
ruff check . && ruff format --check . && mypy src/stealthbench
pytest --strict-markers tests/unit tests/contract
pytest --strict-markers tests/integration tests/replay
```

### Reproduce the live results

The live scripts call the keyless free tier with the standard library only. Each takes
a locally downloaded copy of the pinned dataset (revisions are listed in
[`docs/upstream-inventory.md`](docs/upstream-inventory.md)) and writes raw responses
to `artifacts/`, which is gitignored.

```bash
python3 scripts/live_probe.py --output artifacts/live-setup          # catalog and telemetry

python3 scripts/live_bench_ifeval.py --input ifeval_input.jsonl \
    --model space-bunny-free --n 541 --seed 20261004 --output artifacts/live-ifeval

PYTHONPATH=src python3 scripts/live_bench_suite.py --mmlu mmlu_test.jsonl --math math500.bin \
    --model space-bunny-free --seed 20261004 --output artifacts/live-suite

PYTHONPATH=src python3 scripts/live_bfcl_50.py \
    --model space-bunny-free --seed 20261004 --output artifacts/live-bfcl-50
```

IFEval responses are then scored with the official `evaluation_main.py`. A stealth
alias can change or disappear without notice, so a rerun measures the endpoint as it
is that day.

Paid or credentialed campaigns go through `stealthbench run` with a manifest that
declares a spending cap. `python3 scripts/live_setup.py` writes one interactively, and
[`docs/operations.md`](docs/operations.md) is the operator guide.

## How it stays honest

- **Missing is not zero.** An unknown token count or a failed request stays `null`. It
  is never folded into a score.
- **No implicit spend.** Live dispatch needs an explicit manifest, a credential
  reference and a spending cap. An unknown price blocks the run instead of counting
  as free.
- **Retries are not samples.** Only 429 and 5xx responses are retried. A wrong answer
  is scored once.
- **Everything is pinned.** Dataset revisions, grader commits and the full dependency
  closure are fixed, and the manifest hash binds a run to its exact item list.
- **Tests cannot reach a provider.** The suite of 1,400+ offline tests denies
  non-loopback network access and scrubs credential-shaped environment variables.
  `mypy --strict` and `ruff` run on every push.
- **Similarity is never shown as probability.** Identity estimation reports evidence,
  and stays in similarity mode until calibration evidence exists.

## Repository layout

```text
src/stealthbench/   adapters, scheduler, storage, benchmarks, fingerprints, reporting, CLI
tests/              unit, contract, integration, replay, sandbox and live suites
scripts/            live campaign scripts used for the published results
configs/            example campaign profiles (no credentials)
results/, docs/     published results, frozen contracts, operations guide
implementation/     per-gate evidence and live-run evidence files
```

Design background: [`BENCHMARK_PLAN.md`](BENCHMARK_PLAN.md),
[`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) and
[`docs/contracts.md`](docs/contracts.md).

## Conclusion

`space-bunny-free` is a usable mid-tier reasoning model at no cost: solid on broad
knowledge (0.71 on MMLU-Pro), moderate on competition math (0.61) and weak at following
precise formatting instructions (0.57 on IFEval), with uneven function calling. It sits
well below current frontier models on all three, so it suits low-stakes drafting and
question answering more than work that needs strict output compliance.

The more durable result is the method. The largest errors in this campaign came from
the measurement, not the model: a token budget that hid the answers, and a grader bug
that inflated a score by 23 points. Both were caught because every number had to be
backed by a stored, replayable artifact. Evaluating an anonymous endpoint credibly is
mostly an engineering discipline problem, and it can be done for $0.

## Author

Built by [Md Mobashir Rahman](https://github.com/mobashirrahman).

## License

Apache-2.0. See [`LICENSE`](LICENSE).
