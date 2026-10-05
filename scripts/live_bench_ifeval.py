"""Live IFEval benchmark run on a keyless $0 Zen alias.

1. Selects N prompts from the pinned IFEval input file by seeded hash order.
2. Sends each as a bare user message (no system prompt) with temperature 0.
3. Writes {prompt, response} JSONL for the OFFICIAL scorer
   (instruction_following_eval@e49bbfe3, run separately).
4. Retries only 429/5xx (max 3 attempts); wrong answers are scored, never
   retried. Transport failures stay missing, never zero.

Usage:
    python3 scripts/live_bench_ifeval.py --input /tmp/ifeval_input.jsonl \\
        --model space-bunny-free --n 50 --seed 20261004 \\
        --output artifacts/live-ifeval-20261004
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

ZEN_URL = "https://opencode.ai/zen/v1/chat/completions"
USER_AGENT = "stealthbench-live-bench/0.1"
RETRYABLE = (429, 500, 502, 503, 504)


def post(model: str, prompt: str, max_tokens: int, temperature: float) -> dict:
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
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
            with urllib.request.urlopen(req, timeout=120) as r:
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


def select(items: list[dict], n: int, seed: int) -> list[dict]:
    def rank(item: dict) -> str:
        return hashlib.sha256(f"{seed}:{item['key']}".encode()).hexdigest()

    return sorted(items, key=rank)[:n]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=20261004)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    items = [json.loads(line) for line in Path(args.input).read_text().splitlines() if line.strip()]
    chosen = select(items, args.n, args.seed)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    started_at = datetime.now(UTC)
    (out / "selection.json").write_text(
        json.dumps(
            {
                "source_file": args.input,
                "source_items": len(items),
                "seed": args.seed,
                "selected_keys": [c["key"] for c in chosen],
            },
            indent=2,
        )
        + "\n"
    )

    done_prompts: set[str] = set()
    responses_path = out / "responses.jsonl"
    if responses_path.exists():
        done_prompts = {
            json.loads(line)["prompt"]
            for line in responses_path.read_text().splitlines()
            if line.strip()
        }
        if done_prompts:
            print(f"resuming: {len(done_prompts)} done, {len(chosen) - len(done_prompts)} left")

    results: list[dict] = []
    with responses_path.open("a") as f:
        for i, item in enumerate(chosen):
            if item["prompt"] in done_prompts:
                continue
            res = post(args.model, item["prompt"], args.max_tokens, temperature=0.0)
            entry = {
                "key": item["key"],
                "instruction_id_list": item.get("instruction_id_list"),
                "generation": res,
            }
            results.append(entry)
            if res["ok"]:
                try:
                    content = res["payload"]["choices"][0]["message"].get("content", "")
                except (KeyError, IndexError, TypeError):
                    content = ""
                f.write(json.dumps({"prompt": item["prompt"], "response": content}) + "\n")
            print(
                f"[{i + 1}/{len(chosen)}] key={item['key']} "
                f"ok={res['ok']} attempts={res['attempts']}",
                flush=True,
            )
            time.sleep(1)

    ok = 0
    if responses_path.exists():
        ok = sum(1 for line in responses_path.read_text().splitlines() if line.strip())
    print(f"done: {ok} responses written")
    (out / "run.json").write_text(
        json.dumps(
            {
                "model": args.model,
                "started_at": started_at.isoformat(),
                "ended_at": datetime.now(UTC).isoformat(),
                "generation": {
                    "system_prompt": None,
                    "temperature": 0.0,
                    "max_tokens": args.max_tokens,
                    "one_sample_per_task": True,
                },
                "requested": len(chosen),
                "transport_ok": ok,
                "transport_failed": len(chosen) - ok,
                "distinct_cost_values": ["0"],
                "score_with": (
                    "python3 -m instruction_following_eval.evaluation_main "
                    f"--input_data={args.input} "
                    f"--input_response_data={out}/responses.jsonl "
                    f"--output_dir={out}/scores"
                ),
            },
            indent=2,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
