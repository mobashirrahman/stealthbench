"""Interactive live-campaign setup for StealthBench.

Walks you through credentials + spending authorization and writes a live
manifest (e.g. ``configs/live-setup.json``). Secret *values* never touch disk:
they live only in your shell environment (and this process, for preflight).
The manifest records only ``credential_ref`` names.

Usage:
    python3 scripts/live_setup.py
"""

from __future__ import annotations

import getpass
import json
import os
import sys
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from stealthbench.runner.preflight import AUTHORIZATION_ENV, run_preflight  # noqa: E402
from stealthbench.schemas.campaign import CampaignManifest  # noqa: E402
from stealthbench.schemas.manifest import manifest_hash  # noqa: E402


def ask(prompt: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default is not None else ""
    answer = input(f"{prompt}{suffix}: ").strip()
    return answer or (default or "")


def ask_yn(prompt: str, default: bool) -> bool:
    hint = "Y/n" if default else "y/N"
    answer = input(f"{prompt} [{hint}]: ").strip().lower()
    if not answer:
        return default
    return answer in ("y", "yes")


def ask_secret(prompt: str) -> str:
    try:
        return getpass.getpass(f"{prompt}: ").strip()
    except (EOFError, OSError, getpass.GetPassWarning):
        # No tty (piped stdin): fall back to visible input.
        return input(f"{prompt} (visible, no tty): ").strip()


def ask_float(prompt: str, default: float | None) -> float | None:
    raw = ask(prompt, None if default is None else str(default))
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        print(f"  not a number {raw!r}; leaving unset (null).")
        return None
    if value <= 0:
        print("  must be > 0; leaving unset (null).")
        return None
    return value


def main() -> int:
    print("StealthBench live setup — secrets stay in your shell, never in files.\n")

    template_path = REPO_ROOT / "configs" / "pilot.json"
    template = json.loads(template_path.read_text(encoding="utf-8"))

    authorized_by = ask("Operator name (authorized_by)", os.environ.get("USER", "operator"))
    while not authorized_by:
        print("  an operator name is required.")
        authorized_by = ask("Operator name (authorized_by)")

    cap = ask_float("Operator spending cap in USD (spending_cap_usd)", 25.0)
    while cap is None:
        print("  a numeric cap is required for live dispatch.")
        cap = ask_float("Operator spending cap in USD (spending_cap_usd)", 25.0)

    campaign_id = ask("Campaign id", f"live-setup-{date.today():%Y%m%d}").lower()
    endpoints: list[dict] = []
    cred_exports: dict[str, str] = {}

    while True:
        print(f"\n--- endpoint #{len(endpoints) + 1} ---")
        alias = ask("Alias (lowercase id, e.g. big-pickle)").lower()
        while not alias:
            alias = ask("Alias (required)").lower()
        transport = ask("Transport (zen/reference)", "zen").lower()
        while transport not in ("zen", "reference"):
            transport = ask("Transport (zen/reference)", "zen").lower()
        cred_name = ask("Credential env-var name (credential_ref)", "ZEN_API_KEY").upper()
        secret = ""
        if os.environ.get(cred_name):
            print(f"  {cred_name} is already set in this shell; keeping it.")
        else:
            secret = ask_secret(f"Value for {cred_name} (kept in env only, never written)")
            if secret:
                os.environ[cred_name] = secret
                cred_exports[cred_name] = secret
        print("  capabilities:")
        caps = {
            "streaming": ask_yn("    streaming?", True),
            "tool_calls": ask_yn("    tool_calls?", False),
            "reasoning": ask_yn("    reasoning?", False),
            "usage_reporting": ask_yn("    usage_reporting?", True),
            "logprobs": ask_yn("    logprobs?", False),
        }
        print("  price snapshot in USD per 1M tokens (blank = unknown/null):")
        price_in = ask_float("    input_per_mtok", None)
        price_out = ask_float("    output_per_mtok", None)
        snapshot_id = ask("    snapshot_id", f"manual-{date.today():%Y%m%d}") or None
        endpoints.append(
            {
                "endpoint_id": alias,
                "alias": alias,
                "route": transport,
                "transport": transport,
                "capabilities": caps,
                "pricing": {
                    "currency": "USD",
                    "input_per_mtok": price_in,
                    "output_per_mtok": price_out,
                    "cached_input_per_mtok": None,
                    "reasoning_per_mtok": None,
                    "snapshot_id": snapshot_id,
                },
                "credential_ref": cred_name,
                "label_provider": None,
                "label_family": None,
                "label_exact_version": None,
                "label_tokenizer": None,
            }
        )
        if not ask_yn("Add another endpoint?", False):
            break

    ceiling = ask_float("Campaign cost ceiling USD (limits.max_total_cost_usd)", min(250.0, cap))
    if ceiling is None or ceiling > cap:
        print(f"  ceiling must be numeric and covered by your cap ({cap}); using {cap}.")
        ceiling = cap
    template["limits"]["max_total_cost_usd"] = ceiling

    template.update(
        {
            "campaign_id": campaign_id,
            "title": f"Live setup for {campaign_id}",
            "description": (
                "Operator-configured live setup. Benchmarks are unmaterialized until "
                "the frozen item selection runs; this profile cannot dispatch before that."
            ),
            "mode": "live_authorized",
            "materialized": False,
            "authorization": {
                "required": True,
                "spending_cap_usd": cap,
                "authorized_by": authorized_by,
            },
            "endpoints": endpoints,
        }
    )
    for bench in template.get("benchmarks", []):
        bench["item_ids"] = []

    try:
        manifest = CampaignManifest.model_validate(template)
    except Exception as exc:  # pydantic ValidationError
        print(f"\nManifest invalid, nothing written:\n{exc}")
        return 1

    out_name = ask("\nOutput file", f"configs/live-{campaign_id}.json")
    out_path = REPO_ROOT / out_name
    out_path.write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"\nWrote {out_path.relative_to(REPO_ROOT)} (credential_refs only, no secrets).")
    print(f"manifest_hash: {manifest_hash(manifest)}")

    # Preflight in-process (env already carries any secrets entered above).
    os.environ[AUTHORIZATION_ENV] = "1"
    report = run_preflight(manifest)
    print("\nPreflight:")
    for check in report.checks:
        mark = "ok  " if check.passed else "FAIL"
        print(f"  [{mark}] {check.name}" + (f" — {check.detail}" if check.detail else ""))
    if report.passed:
        print("\nPreflight passed: this manifest is dispatchable.")
    else:
        print("\nStill blocked (expected before item selection is frozen):")
        for blocker in report.blockers:
            print(f"  - {blocker}")

    print("\nFor future shells, export (values never touch the repo):")
    for name in dict.fromkeys([e["credential_ref"] for e in endpoints]):
        print(f'  export {name}="..."')
    print(f"  export {AUTHORIZATION_ENV}=1")
    print(f"\nThen: stealthbench manifest validate {out_name}")
    print(f"      STEALTHBENCH_LIVE_AUTHORIZATION=1 stealthbench run {out_name} --live")
    print("\nNext real step: freeze the item selection (materialize benchmarks with")
    print("pinned dataset revisions) before any live dispatch can proceed.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except EOFError:
        print("\nAborted: input ended before setup finished; nothing was written.")
        raise SystemExit(1) from None
