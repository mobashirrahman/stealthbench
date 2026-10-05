"""Live setup-telemetry campaign (keyless $0 free tier).

Sends a small fixed set of requests to keyless free Zen aliases and records
telemetry: catalog snapshot, per-probe usage vectors, latency, finish reasons,
cost fields, and hand-graded synthetic instruction checks.

This is NOT a benchmark score: prompts are synthetic setup probes, grading is
exact local rules defined below, and coverage is far below any pilot profile.
Anything it reports is measured telemetry only.

Usage:
    python3 scripts/live_probe.py [--output artifacts/live-setup-<date>]
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

ZEN_BASE = "https://opencode.ai/zen/v1"
USER_AGENT = "stealthbench-live-probe/0.1"

BASELINE = "The quick brown fox jumps over the lazy dog."

# (probe_id, text appended after the fixed baseline message)
PROBES = [
    ("baseline", ""),
    ("code", "def fib(n):\n    return n if n < 2 else fib(n-1) + fib(n-2)"),
    ("whitespace", "a   b\t\tc\n\n\nd    e"),
    ("punct", "Hello, world! (yes?) [no] {maybe} — end."),
    ("unicode", "Zürich naïve façade déjà vu Σίσυφος"),
    ("emoji", "🚀🔥💧起こる 🎉"),
    ("multilingual", "العربية हिन्दी 中文 русский português"),
    ("numeric", "3.14159 0xFF 1e-6 42 1000000"),
]

# (check_id, prompt, exact expected response)
CHECKS = [
    ("exact-ok", "Reply with exactly this and nothing else: OK", "OK"),
    ("exact-word", "Reply with exactly this and nothing else: blueberry", "blueberry"),
    ("keyword", "Answer in one short sentence: what color is the sky on a clear day?", None),
]

RETRYABLE = (429, 500, 502, 503, 504)


def post(model: str, prompt: str, max_tokens: int, timeout: int = 30) -> dict:
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
        }
    ).encode()
    attempts: list[dict] = []
    for attempt in range(3):
        started = time.monotonic()
        try:
            req = urllib.request.Request(
                ZEN_BASE + "/chat/completions",
                data=body,
                headers={"Content-Type": "application/json", "User-Agent": USER_AGENT},
            )
            with urllib.request.urlopen(req, timeout=timeout) as r:
                raw = r.read()
            latency_ms = int((time.monotonic() - started) * 1000)
            payload = json.loads(raw)
            return {
                "ok": True,
                "attempts": attempt + 1,
                "attempt_log": attempts,
                "latency_ms": latency_ms,
                "payload": payload,
            }
        except urllib.error.HTTPError as e:
            latency_ms = int((time.monotonic() - started) * 1000)
            try:
                detail = e.read()[:500].decode("utf-8", "replace")
            except Exception:
                detail = ""
            attempts.append(
                {"attempt": attempt, "http": e.code, "latency_ms": latency_ms, "detail": detail}
            )
            if e.code not in RETRYABLE or attempt == 2:
                return {"ok": False, "attempts": attempt + 1, "attempt_log": attempts}
            time.sleep(2 * (attempt + 1))
        except Exception as e:  # transport failure is data
            latency_ms = int((time.monotonic() - started) * 1000)
            attempts.append(
                {"attempt": attempt, "error": type(e).__name__, "latency_ms": latency_ms}
            )
            if attempt == 2:
                return {"ok": False, "attempts": attempt + 1, "attempt_log": attempts}
            time.sleep(2 * (attempt + 1))
    return {"ok": False, "attempts": 3, "attempt_log": attempts}  # pragma: no cover


def fetch_catalog(timeout: int = 25) -> dict:
    req = urllib.request.Request(ZEN_BASE + "/models", headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="space-bunny-free", help="comma-separated Zen aliases")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    started_at = datetime.now(UTC)
    out_dir = args.output or (f"artifacts/live-setup-{started_at:%Y%m%d}")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    catalog = fetch_catalog()
    (out / "catalog.json").write_text(
        json.dumps({"fetched_at": started_at.isoformat(), "catalog": catalog}, indent=2) + "\n"
    )
    alias_ids = [m.get("id") for m in catalog.get("data", [])]

    generations: list[dict] = []
    for model in models:
        if model not in alias_ids:
            generations.append(
                {
                    "model": model,
                    "kind": "alias_check",
                    "ok": False,
                    "reason": "alias absent from live catalog snapshot",
                }
            )
            continue
        for probe_id, extra in PROBES:
            prompt = BASELINE if not extra else BASELINE + "\n" + extra
            res = post(model, prompt, max_tokens=8)
            generations.append(
                {"model": model, "kind": "probe", "probe_id": probe_id, "prompt": prompt, **res}
            )
            time.sleep(1)
        for check_id, prompt, expected in CHECKS:
            res = post(model, prompt, max_tokens=32)
            entry: dict = {
                "model": model,
                "kind": "check",
                "check_id": check_id,
                "prompt": prompt,
                "expected": expected,
                **res,
            }
            if res["ok"] and expected is not None:
                try:
                    content = res["payload"]["choices"][0]["message"].get("content", "")
                except (KeyError, IndexError, TypeError):
                    content = None
                entry["grade"] = {
                    "rule": "exact_match_after_strip",
                    "pass": content is not None and content.strip() == expected,
                    "observed": content,
                }
            generations.append(entry)
            time.sleep(1)

    ended_at = datetime.now(UTC)
    ok_gens = [g for g in generations if g.get("ok")]
    costs = {str(g.get("payload", {}).get("cost")) for g in ok_gens if "payload" in g}
    prompt_tokens = [
        g["payload"]["usage"]["prompt_tokens"]
        for g in ok_gens
        if isinstance(g.get("payload"), dict)
        and isinstance(g["payload"].get("usage"), dict)
        and g["payload"]["usage"].get("prompt_tokens") is not None
    ]
    # Delta of each probe vs the baseline probe, per model (tokenizer signal).
    deltas: dict[str, dict[str, int | None]] = {}
    by_model: dict[str, dict[str, dict]] = {}
    for g in ok_gens:
        if g.get("kind") == "probe":
            by_model.setdefault(g["model"], {})[g["probe_id"]] = g
    for model, probes in by_model.items():
        base = probes.get("baseline", {}).get("payload", {}).get("usage", {})
        base_n = base.get("prompt_tokens")
        row: dict[str, int | None] = {}
        for probe_id, _extra in PROBES:
            if probe_id == "baseline":
                continue
            n = probes.get(probe_id, {}).get("payload", {}).get("usage", {}).get("prompt_tokens")
            row[probe_id] = (n - base_n) if (n is not None and base_n is not None) else None
        deltas[model] = row

    prompt_digest = hashlib.sha256(
        json.dumps(
            [(p[0], BASELINE if not p[1] else BASELINE + "\n" + p[1]) for p in PROBES]
            + [(c[0], c[1]) for c in CHECKS],
            sort_keys=True,
        ).encode()
    ).hexdigest()

    summary = {
        "campaign": "live-setup-telemetry",
        "models": models,
        "window": {"started_at": started_at.isoformat(), "ended_at": ended_at.isoformat()},
        "catalog_alias_count": len(alias_ids),
        "requests": len(generations),
        "succeeded": len(ok_gens),
        "failed": len(generations) - len(ok_gens),
        "distinct_cost_values": sorted(costs),
        "prompt_token_values": prompt_tokens,
        "token_deltas_vs_baseline": deltas,
        "prompt_digest": prompt_digest,
        "checks": [
            {
                "model": g["model"],
                "check_id": g["check_id"],
                "grade": g.get("grade", {"rule": "ungraded_human_read"}),
            }
            for g in generations
            if g.get("kind") == "check"
        ],
        "disclaimer": (
            "Setup telemetry only, not a benchmark score. Prompts are synthetic; "
            "checks use local exact-match rules, not official evaluators."
        ),
    }
    (out / "generations.jsonl").write_text(
        "".join(json.dumps(g, sort_keys=True) + "\n" for g in generations)
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
