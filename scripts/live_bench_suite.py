"""Live MMLU-Pro (100 stratified) + MATH-500 (100) on a keyless $0 Zen alias.

Selection: sha256(seed:item_id) rank; MMLU-Pro stratified ~7/category over the
14 test categories (+2 by global rank); MATH-500 first 100 by rank.
Generation: bare user message, temperature 0, one sample per task.
Grading: repo adapters (stealthbench.benchmarks.mmlu_pro / math500) —
frozen extraction, no random fallback, bounded symbolic check.
Retries: 429/5xx only, max 3 attempts. Transport failures stay missing.

Usage:
    PYTHONPATH=src python3 scripts/live_bench_suite.py \\
        --mmlu /tmp/mmlu_test.jsonl --math /tmp/math500.bin \\
        --model space-bunny-free --seed 20261004 --output artifacts/live-suite-20261004
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

from stealthbench.benchmarks.math500 import (
    extract_final_answer,
    normalize_answer,
    numeric_equals,
    symbolic_equals,
)
from stealthbench.benchmarks.mmlu_pro import VALID_CHOICES, extract_choice

ZEN_URL = "https://opencode.ai/zen/v1/chat/completions"
USER_AGENT = "stealthbench-live-bench/0.1"
RETRYABLE = (429, 500, 502, 503, 504)
LETTERS = "ABCDEFGHIJ"


def post(model: str, prompt: str, max_tokens: int) -> dict:
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
        }
    ).encode()
    attempts: list[dict] = []
    for attempt in range(3):
        started = time.monotonic()
        try:
            req = urllib.request.Request(
                ZEN_URL,
                data=body,
                headers={"Content-Type": "application/json", "User-Agent": USER_AGENT},
            )
            with urllib.request.urlopen(req, timeout=180) as r:
                payload = json.loads(r.read())
            return {
                "ok": True,
                "attempts": attempt + 1,
                "attempt_log": attempts,
                "latency_ms": int((time.monotonic() - started) * 1000),
                "payload": payload,
            }
        except urllib.error.HTTPError as e:
            try:
                detail = e.read()[:300].decode("utf-8", "replace")
            except Exception:
                detail = ""
            attempts.append({"attempt": attempt, "http": e.code, "detail": detail})
            if e.code not in RETRYABLE or attempt == 2:
                return {"ok": False, "attempts": attempt + 1, "attempt_log": attempts}
            time.sleep(4 * (attempt + 1))
        except Exception as e:  # transport failure is data
            attempts.append({"attempt": attempt, "error": type(e).__name__})
            if attempt == 2:
                return {"ok": False, "attempts": attempt + 1, "attempt_log": attempts}
            time.sleep(4 * (attempt + 1))
    return {"ok": False, "attempts": 3, "attempt_log": attempts}  # pragma: no cover


def rank(key: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()


def content_of(res: dict) -> str:
    try:
        return res["payload"]["choices"][0]["message"].get("content", "") or ""
    except (KeyError, IndexError, TypeError):
        return ""


def mmlu_prompt(item: dict) -> str:
    lines = [
        "Answer the following multiple choice question. Reply with only the option letter.",
        "",
        f"Question: {item['question']}",
        "Options:",
    ]
    for letter, opt in zip(LETTERS, item["options"], strict=False):
        lines.append(f"{letter}. {opt}")
    lines += ["", "Answer:"]
    return "\n".join(lines)


def math_prompt(item: dict) -> str:
    return (
        f"{item['problem']}\n\nSolve the problem step by step. "
        "Put your final answer in \\boxed{}."
    )


def run_mmlu(
    items: list[dict],
    model: str,
    seed: int,
    out: Path,
    max_tokens: int = 64,
    per_cat: int = 7,
    limit: int | None = 100,
) -> dict:
    by_cat: dict[str, list[dict]] = {}
    for it in items:
        by_cat.setdefault(it["category"], []).append(it)
    chosen: list[dict] = []
    for cat in sorted(by_cat):
        chosen += sorted(by_cat[cat], key=lambda it: rank(str(it["question_id"]), seed))[:per_cat]
    if limit is not None and len(chosen) > limit:
        order = {str(it["question_id"]): rank(str(it["question_id"]), seed) for it in chosen}
        chosen = sorted(chosen, key=lambda it: order[str(it["question_id"])])[:limit]
    elif limit is not None:
        rest = sorted(
            (it for it in items if it not in chosen),
            key=lambda it: rank(str(it["question_id"]), seed),
        )
        chosen += rest[: max(0, limit - len(chosen))]
    (out / "mmlu_selection.json").write_text(
        json.dumps(
            {
                "seed": seed,
                "n": len(chosen),
                "keys": [c["question_id"] for c in chosen],
                "categories": sorted({c["category"] for c in chosen}),
            },
            indent=2,
        )
        + "\n"
    )
    correct = invalid = failed = 0
    gen_path = out / "mmlu_generations.jsonl"
    done: set[int] = set()
    if gen_path.exists():
        for line in gen_path.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["question_id"])
        if done:
            print(f"resuming mmlu: {len(done)} done")
    with gen_path.open("a") as f:
        for i, item in enumerate(chosen):
            if item["question_id"] in done:
                continue
            res = post(model, mmlu_prompt(item), max_tokens=max_tokens)
            entry: dict = {
                "question_id": item["question_id"],
                "category": item["category"],
                "gold": item["answer"],
                "generation": res,
            }
            if res["ok"]:
                text = content_of(res)
                ext = extract_choice(text, VALID_CHOICES)
                entry["extracted"] = {
                    "choice": ext.choice,
                    "valid": ext.valid,
                    "reason": ext.reason,
                }
                if ext.valid and ext.choice == item["answer"]:
                    correct += 1
                    entry["correct"] = True
                else:
                    invalid += 1
                    entry["correct"] = False
            else:
                failed += 1
                entry["correct"] = None
            f.write(json.dumps(entry, sort_keys=True) + "\n")
            print(
                f"[mmlu {i + 1}/{len(chosen)}] q={item['question_id']} ok={res['ok']}", flush=True
            )
            time.sleep(1)
    correct = invalid = failed = 0
    for line in gen_path.read_text().splitlines():
        if not line.strip():
            continue
        c = json.loads(line)["correct"]
        if c is True:
            correct += 1
        elif c is False:
            invalid += 1
        else:
            failed += 1
    graded = correct + invalid
    return {
        "requested": len(chosen),
        "transport_ok": graded,
        "transport_failed": failed,
        "accuracy": (correct / graded) if graded else None,
        "correct": correct,
        "invalid_or_wrong": invalid,
    }


def run_math(items: list[dict], model: str, seed: int, out: Path, n: int = 100) -> dict:
    chosen = sorted(items, key=lambda it: rank(it["unique_id"], seed))[:n]
    (out / "math_selection.json").write_text(
        json.dumps(
            {"seed": seed, "n": len(chosen), "keys": [c["unique_id"] for c in chosen]}, indent=2
        )
        + "\n"
    )
    correct = wrong = failed = 0
    math_path = out / "math_generations.jsonl"
    done_ids: set[str] = set()
    if math_path.exists():
        for line in math_path.read_text().splitlines():
            if line.strip():
                done_ids.add(json.loads(line)["unique_id"])
        if done_ids:
            print(f"resuming math: {len(done_ids)} done")
    with math_path.open("a") as f:
        for i, item in enumerate(chosen):
            if item["unique_id"] in done_ids:
                continue
            res = post(model, math_prompt(item), max_tokens=1024)
            entry: dict = {
                "unique_id": item["unique_id"],
                "subject": item.get("subject"),
                "gold": item["answer"],
                "generation": res,
            }
            if res["ok"]:
                text = content_of(res)
                try:
                    pred = extract_final_answer(text)
                    gold = normalize_answer(item["answer"])
                    pred_n = normalize_answer(pred)
                    try:
                        eq, _method = symbolic_equals(pred, item["answer"])
                    except Exception:
                        eq = False
                    hit = pred_n == gold or numeric_equals(pred_n, gold) or eq
                except Exception as e:  # bounded-parser failures are wrong, not missing
                    hit = False
                    entry["grade_error"] = type(e).__name__
                entry["correct"] = bool(hit)
                if hit:
                    correct += 1
                else:
                    wrong += 1
            else:
                failed += 1
                entry["correct"] = None
            f.write(json.dumps(entry, sort_keys=True) + "\n")
            print(f"[math {i + 1}/{len(chosen)}] id={item['unique_id']} ok={res['ok']}", flush=True)
            time.sleep(1)
    correct = wrong = failed = 0
    for line in math_path.read_text().splitlines():
        if not line.strip():
            continue
        c = json.loads(line)["correct"]
        if c is True:
            correct += 1
        elif c is False:
            wrong += 1
        else:
            failed += 1
    graded = correct + wrong
    return {
        "requested": len(chosen),
        "transport_ok": graded,
        "transport_failed": failed,
        "accuracy": (correct / graded) if graded else None,
        "correct": correct,
        "wrong": wrong,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mmlu", required=True)
    ap.add_argument("--math", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--seed", type=int, default=20261004)
    ap.add_argument("--output", required=True)
    ap.add_argument("--only", choices=["mmlu", "math"], default=None)
    ap.add_argument("--max-tokens-mmlu", type=int, default=64)
    ap.add_argument("--n-math", type=int, default=100)
    ap.add_argument("--n-mmlu-per-cat", type=int, default=7)
    ap.add_argument("--mmlu-limit", type=int, default=100)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(UTC)
    summary: dict = {
        "model": args.model,
        "seed": args.seed,
        "started_at": started_at.isoformat(),
        "generation": {"system_prompt": None, "temperature": 0.0, "one_sample_per_task": True},
    }
    if args.only in (None, "mmlu"):
        mmlu_items = [
            json.loads(line) for line in Path(args.mmlu).read_text().splitlines() if line.strip()
        ]
        summary["mmlu_pro"] = run_mmlu(
            mmlu_items,
            args.model,
            args.seed,
            out,
            args.max_tokens_mmlu,
            args.n_mmlu_per_cat,
            args.mmlu_limit,
        )
        summary["mmlu_pro"]["max_tokens"] = args.max_tokens_mmlu
    if args.only in (None, "math"):
        math_items = [
            json.loads(line) for line in Path(args.math).read_text().splitlines() if line.strip()
        ]
        summary["math500"] = run_math(math_items, args.model, args.seed, out, args.n_math)
    if args.tag:
        summary["tag"] = args.tag
    summary["ended_at"] = datetime.now(UTC).isoformat()
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
