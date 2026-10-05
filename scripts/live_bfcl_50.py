"""Live BFCL 50-case prompted slice on a keyless $0 Zen alias.

Selection: 50 non-live, non-agentic cases by sha256(seed:item_id) rank.
  Quotas: simple_python 9, simple_java 8, simple_javascript 8,
          multiple 9, parallel 8, irrelevance 8 (total 50).
  Excludes live_*, web_search/memory (needs SerpAPI), multi-turn,
  format_sensitivity, and retired restful/executable (never reported).
Generation: bare user message (function specs JSON + user query),
  temperature 0, max_tokens 512, one sample, sequential with 1s sleep.
  Retries on 429/5xx only, max 3 attempts. Transport via stdlib urllib only.
Grading: stealthbench.benchmarks.bfcl in prompted mode. Transport failures
  stay missing (unavailable), never zero. Partial slice has no complete
  overall score by construction.

Usage:
    PYTHONPATH=src python3 scripts/live_bfcl_50.py \\
        --model space-bunny-free --seed 20261004 \\
        --output artifacts/live-bfcl-50 [--limit 3]
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import hashlib
import json
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

ZEN_URL = "https://opencode.ai/zen/v1/chat/completions"
USER_AGENT = "stealthbench-live-bfcl/0.1"
RETRYABLE = (429, 500, 502, 503, 504)

PINNED_COMMIT = "f7cf7359b7ac615a0b294831c5ba2bc95ee4a000"
DATA_BASE = (
    "https://raw.githubusercontent.com/ShishirPatil/gorilla/"
    f"{PINNED_COMMIT}/berkeley-function-call-leaderboard/bfcl_eval/data"
)
HF_API_URL = "https://huggingface.co/api/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard"
HF_DATASET_ID = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"

QUOTAS: dict[str, int] = {
    "simple_python": 9,
    "simple_java": 8,
    "simple_javascript": 8,
    "multiple": 9,
    "parallel": 8,
    "irrelevance": 8,
}
SEED_DEFAULT = 20261004

PROMPT_TEMPLATE = (
    "You are an expert in composing functions. You are given a question and a set "
    "of possible functions. Based on the question, you will need to make one or more "
    "function/tool calls to achieve the purpose. If none of the functions can be used, "
    "point it out. If the given question lacks the parameters required by the function, "
    "also point it out.\n"
    "You should only return the function calls in your response.\n\n"
    "If you decide to invoke any of the function(s), you MUST put it in the format of "
    "[func_name1(params_name1=params_value1, params_name2=params_value2...), "
    "func_name2(params)]\n"
    "You SHOULD NOT include any other text in the response.\n\n"
    "Here is a list of functions in JSON format that you can invoke.\n"
    "{functions}\n\n"
    "Question: {question}"
)


def fetch_bytes(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


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
            time.sleep(2 * (attempt + 1))
        except Exception as e:  # transport failure is data
            attempts.append({"attempt": attempt, "error": type(e).__name__})
            if attempt == 2:
                return {"ok": False, "attempts": attempt + 1, "attempt_log": attempts}
            time.sleep(2 * (attempt + 1))
    return {"ok": False, "attempts": 3, "attempt_log": attempts}  # pragma: no cover


def rank_key(seed: int, item_id: str) -> str:
    return hashlib.sha256(f"{seed}:{item_id}".encode()).hexdigest()


def user_question(entry: dict) -> str:
    q = entry["question"]
    if isinstance(q, list) and q and isinstance(q[0], list) and q[0]:
        return str(q[0][0].get("content", ""))
    if isinstance(q, list) and q and isinstance(q[0], dict):
        return str(q[0].get("content", ""))
    return str(q)


def build_prompt(entry: dict) -> str:
    funcs = json.dumps(entry["function"], indent=2)
    return PROMPT_TEMPLATE.format(functions=funcs, question=user_question(entry))


def _func_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _func_name(node.value)
        if base is None:
            return None
        return f"{base}.{node.attr}"
    return None


def _literal(node: ast.AST):
    try:
        return ast.literal_eval(node)
    except Exception:
        try:
            return ast.unparse(node)
        except Exception:
            return None


def parse_python_calls(content: str) -> list[dict]:
    """Parse prompted python-format output into [{function, arguments}].

    Unparsable or empty output means no calls (irrelevance passes, anything
    else fails via the strict adapter oracle). Never raises: parse problems
    yield an empty list so they grade as fail, not as missing.
    """
    text = (content or "").strip().strip("`\n ")
    if not text:
        return []
    if not text.startswith("["):
        text = "[" + text
    if not text.endswith("]"):
        text = text + "]"
    try:
        tree = ast.parse(text, mode="eval")
    except Exception:
        return []
    calls: list[dict] = []
    body = tree.body
    if isinstance(body, ast.Call):
        nodes = [body]
    elif isinstance(body, (ast.List, ast.Tuple)):
        nodes = list(body.elts)
    else:
        return []
    for node in nodes:
        if not isinstance(node, ast.Call):
            continue
        name = _func_name(node.func)
        if name is None:
            continue
        args: dict = {}
        for kw in node.keywords:
            if not kw.arg:
                continue
            args[kw.arg] = _literal(kw.value)
        calls.append({"function": name, "arguments": args})
    return calls


def first_concrete(values: list) -> tuple[bool, object]:
    """Return (present, value) for the first non-'' entry, else (False, None)."""
    for v in values:
        if v != "":
            return True, v
    return False, None


def build_expected(category: str, ground_truth: list | None) -> list[dict]:
    if category in ("irrelevance", "live_irrelevance"):
        return []
    if not ground_truth:
        return []
    expected: list[dict] = []
    for item in ground_truth:
        if not isinstance(item, dict) or len(item) != 1:
            continue
        func, params = next(iter(item.items()))
        args: dict = {}
        if isinstance(params, dict):
            for pname, values in params.items():
                if not isinstance(values, list):
                    args[pname] = values
                    continue
                present, value = first_concrete(values)
                if present:
                    args[pname] = value
        expected.append({"function": func, "arguments": args})
    return expected


def content_of(res: dict) -> str:
    try:
        return res["payload"]["choices"][0]["message"].get("content", "") or ""
    except (KeyError, IndexError, TypeError):
        return ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--seed", type=int, default=SEED_DEFAULT)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(UTC)

    # Record HF dataset identity (plain https, no auth). HF hosts only V3
    # files; V4 per-category JSONL comes from the pinned gorilla commit.
    hf_meta: dict = {"dataset_id": HF_DATASET_ID, "note": "unverified"}
    try:
        raw_meta = fetch_bytes(HF_API_URL, timeout=30)
        meta = json.loads(raw_meta)
        sibs = [s.get("rfilename") for s in meta.get("siblings", [])]
        hf_meta = {
            "dataset_id": HF_DATASET_ID,
            "sha": meta.get("sha"),
            "sibling_count": len(sibs),
            "has_v4_files": any("BFCL_v4_" in (s or "") for s in sibs),
            "sample_siblings": sorted(sibs)[:8],
        }
    except Exception as e:  # transport failure is data
        hf_meta = {"dataset_id": HF_DATASET_ID, "fetch_error": type(e).__name__}

    file_hashes: dict[str, str] = {}
    by_category: dict[str, list[dict]] = {}
    gt_by_id: dict[str, list] = {}
    for category in QUOTAS:
        raw = fetch_bytes(f"{DATA_BASE}/BFCL_v4_{category}.json")
        file_hashes[category] = hashlib.sha256(raw).hexdigest()
        items = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
        by_category[category] = items
        if category not in ("irrelevance",):
            try:
                praw = fetch_bytes(f"{DATA_BASE}/possible_answer/BFCL_v4_{category}.json")
                for line in praw.decode().splitlines():
                    if line.strip():
                        obj = json.loads(line)
                        gt_by_id[obj["id"]] = obj.get("ground_truth", [])
            except urllib.error.HTTPError:
                pass

    chosen: list[dict] = []
    for category in sorted(QUOTAS):
        items = sorted(by_category[category], key=lambda e: rank_key(args.seed, e["id"]))
        for entry in items[: QUOTAS[category]]:
            chosen.append({"category": category, "entry": entry})
    chosen = sorted(chosen, key=lambda c: rank_key(args.seed, c["entry"]["id"]))
    if args.limit is not None:
        chosen = chosen[: args.limit]

    (out / "selection.json").write_text(
        json.dumps(
            {
                "seed": args.seed,
                "quotas": QUOTAS,
                "n": len(chosen),
                "pinned_commit": PINNED_COMMIT,
                "data_base": DATA_BASE,
                "file_sha256": file_hashes,
                "hf": hf_meta,
                "items": [{"id": c["entry"]["id"], "category": c["category"]} for c in chosen],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    gen_path = out / "generations.jsonl"
    done: set[str] = set()
    if gen_path.exists():
        for line in gen_path.read_text().splitlines():
            if line.strip():
                try:
                    done.add(json.loads(line)["id"])
                except (KeyError, json.JSONDecodeError):
                    continue
        if done:
            print(f"resuming: {len(done)} done")

    with gen_path.open("a") as f:
        for i, case in enumerate(chosen):
            entry = case["entry"]
            if entry["id"] in done:
                continue
            prompt = build_prompt(entry)
            res = post(args.model, prompt, max_tokens=args.max_tokens)
            record = {
                "id": entry["id"],
                "category": case["category"],
                "prompt": prompt,
                "question": user_question(entry),
                "function": entry["function"],
                "ground_truth": gt_by_id.get(entry["id"]),
                "generation": res,
            }
            if res["ok"]:
                record["content"] = content_of(res)
                with contextlib.suppress(KeyError, IndexError, TypeError):
                    record["reasoning_content"] = res["payload"]["choices"][0]["message"].get(
                        "reasoning_content"
                    )
            f.write(json.dumps(record, sort_keys=True) + "\n")
            f.flush()
            print(
                f"[{i + 1}/{len(chosen)}] id={entry['id']} cat={case['category']} ok={res['ok']}",
                flush=True,
            )
            time.sleep(1)

    # Offline grading with the frozen repo adapter (prompted track).
    from stealthbench.benchmarks.bfcl import (
        ToolCall,
        grade_accepted_sample,
        grade_generation,
        summarize,
    )
    from stealthbench.schemas.manifest import PromptRef, prompt_hash
    from stealthbench.schemas.results import (
        DeliveryStatus,
        EvaluationPayload,
        GenerationResult,
        ModelRequest,
        TaskSpec,
    )

    grades = []
    transport_ok = 0
    transport_failed = 0
    for line in gen_path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        prompt = rec["prompt"]
        request = ModelRequest(
            messages=[{"role": "user", "content": prompt}],
            max_output_tokens=args.max_tokens,
            temperature=0.0,
        )
        task = TaskSpec(
            benchmark_id="bfcl",
            item_id=rec["id"],
            campaign_id="live-bfcl-20261004",
            endpoint_id=args.model,
            repeat_id=1,
            prompt_artifact_hash=prompt_hash(PromptRef.from_request(request)),
            request=request,
            evaluation=EvaluationPayload(
                gold_answer=rec["id"],
                evaluator_id="bfcl-eval",
                evaluator_revision=PINNED_COMMIT,
            ),
        )
        gen = rec["generation"]
        offered = [fn.get("name", "") for fn in rec.get("function", [])]
        expected = [
            ToolCall(function=e["function"], arguments=dict(e["arguments"]))
            for e in build_expected(rec["category"], rec.get("ground_truth"))
        ]
        if gen.get("ok"):
            transport_ok += 1
            predicted = [
                ToolCall(function=p["function"], arguments=dict(p["arguments"]))
                for p in parse_python_calls(rec.get("content", ""))
            ]
            generation = GenerationResult(
                attempt_id=f"attempt-{rec['id']}",
                sample_key=task.sample_key,
                attempt_number=1,
                delivery_status=DeliveryStatus.ACCEPTED,
                response=rec.get("content", ""),
            )
            grade = grade_generation(
                task=task,
                generation=generation,
                predicted_calls=predicted,
                expected_calls=expected,
                category=rec["category"],
                profile="prompted",
                offered_tools=offered,
                has_serpapi=False,
            )
            # grade_generation with an accepted response never yields
            # evaluator-fail here because the parser always returns a list;
            # keep the accepted-sample path explicit for auditability.
            if grade.grade.correctness == "unavailable":
                grade = grade_accepted_sample(
                    task=task,
                    predicted_calls=predicted,
                    expected_calls=expected,
                    category=rec["category"],
                    profile="prompted",
                    offered_tools=offered,
                    has_serpapi=False,
                )
        else:
            transport_failed += 1
            generation = GenerationResult(
                attempt_id=f"attempt-{rec['id']}",
                sample_key=task.sample_key,
                attempt_number=1,
                delivery_status=DeliveryStatus.TRANSPORT_FAILED,
                response=None,
            )
            grade = grade_generation(
                task=task,
                generation=generation,
                predicted_calls=[],
                expected_calls=expected,
                category=rec["category"],
                profile="prompted",
                offered_tools=offered,
                has_serpapi=False,
            )
        grades.append(grade)

    planned = dict(QUOTAS)
    summary = summarize(grades, profile="prompted", planned_by_category=planned)
    per_category = [
        {
            "category": cell.category,
            "correct": cell.correct,
            "eligible": cell.eligible,
            "planned": cell.planned,
            "accuracy": cell.accuracy,
            "status": cell.status,
        }
        for cell in summary.per_category
    ]
    costs = sorted(
        {
            str(r.get("payload", {}).get("cost"))
            for line in gen_path.read_text().splitlines()
            if line.strip()
            for r in [json.loads(line)["generation"]]
            if r.get("ok") and isinstance(r.get("payload"), dict)
        }
    )
    report = {
        "model": args.model,
        "seed": args.seed,
        "started_at": started_at.isoformat(),
        "ended_at": datetime.now(UTC).isoformat(),
        "generation": {
            "system_prompt": None,
            "temperature": 0.0,
            "max_tokens": args.max_tokens,
            "one_sample_per_task": True,
            "transport": "stdlib urllib only",
        },
        "per_category": per_category,
        "scoring_correct": summary.scoring_correct,
        "scoring_eligible": summary.scoring_eligible,
        "scoring_planned": summary.scoring_planned,
        "overall_accuracy": summary.overall_accuracy,
        "complete": summary.complete,
        "transport": {
            "requested": len(chosen),
            "ok": transport_ok,
            "failed": transport_failed,
        },
        "distinct_cost_values": costs,
    }
    (out / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
