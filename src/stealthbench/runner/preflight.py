"""Live readiness preflight (G14 T14B).

Every check here is a refusal with a named reason, never a warning that can be
scrolled past. A live dispatch requires *all* of them to pass on the same
manifest that will be dispatched:

* **authorization** — the campaign opts into ``live_authorized`` mode, names an
  operator (``authorized_by``) and a numeric spending cap, *and* the operator
  signals approval outside the manifest (``STEALTHBENCH_LIVE_AUTHORIZATION=1``).
  A planning request — a manifest file sitting on disk — alone never authorizes
  paid calls.
* **materialization** — the frozen item selection exists (``materialized`` with
  non-empty ``item_ids`` and no unknown sizes). An unmaterialized profile is a
  valid document, not a dispatchable one.
* **credentials** — every non-fixture endpoint names a ``credential_ref`` whose
  environment variable is actually present. Secret values never appear in the
  manifest; the reference is resolved here, at preflight time.
* **endpoint_modes** — a live campaign uses live transports (``zen`` or
  ``reference``). Fixture transport cannot reach a provider and is refused.
* **revisions** — dataset and evaluator revisions are pinned. ``null`` means
  unverified, never "latest".
* **pricing** — every live endpoint carries a price snapshot and the campaign
  declares caps with ``require_cost_bounds``. An unknown price is not permission
  to spend an unknown amount. The operator cap must cover the campaign ceiling,
  and the missingness threshold is frozen before dispatch.
* **runtime** — the actual execution backend is probed (container runtime on
  ``PATH`` for code/agent execution). A missing runtime reports
  ``blocked_external`` instead of fake-passing.

Offline by construction: no socket, no credential use beyond presence checks,
no dispatch. The report's ``passed`` flag is the only dispatch signal.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

from stealthbench.schemas.campaign import CampaignManifest
from stealthbench.schemas.manifest import manifest_hash

__all__ = [
    "AUTHORIZATION_ENV",
    "PreflightBlocked",
    "PreflightCheck",
    "PreflightReport",
    "live_authorization_granted",
    "run_preflight",
]

#: Outside-the-manifest operator approval. The manifest's spending cap is a
#: ceiling recorded in a profile; this flag is the approval to spend against it.
AUTHORIZATION_ENV: Final[str] = "STEALTHBENCH_LIVE_AUTHORIZATION"


class PreflightBlocked(Exception):
    """A live dispatch was refused. ``blockers`` names every missing requirement."""

    def __init__(self, blockers: tuple[str, ...] | list[str]) -> None:
        super().__init__("live preflight blocked: " + "; ".join(blockers))
        self.blockers = tuple(blockers)


@dataclass(frozen=True, slots=True)
class PreflightCheck:
    """One mandatory gate. A failed check contributes its blocker, if any."""

    name: str
    passed: bool
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class PreflightReport:
    """What preflight verified, and what still blocks a live dispatch."""

    campaign_id: str
    manifest_hash: str
    passed: bool
    blockers: tuple[str, ...]
    checks: tuple[PreflightCheck, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "blockers": list(self.blockers),
            "campaign_id": self.campaign_id,
            "checks": [
                {"detail": check.detail, "name": check.name, "passed": check.passed}
                for check in self.checks
            ],
            "manifest_hash": self.manifest_hash,
            "passed": self.passed,
        }


def live_authorization_granted(env: Mapping[str, str] | None = None) -> bool:
    """Whether the operator approved live execution outside the manifest."""
    source = os.environ if env is None else env
    return source.get(AUTHORIZATION_ENV) == "1"


def _check_authorization(
    manifest: CampaignManifest, live_authorization: bool
) -> tuple[list[str], PreflightCheck]:
    blockers: list[str] = []
    if manifest.mode != "live_authorized":
        blockers.append(
            "authorization: campaign mode is "
            f"{manifest.mode!r}; live dispatch requires mode 'live_authorized'"
        )
    authorization = manifest.authorization
    if authorization is None:
        blockers.append("authorization: live campaign declares no authorization block")
    else:
        if not authorization.required:
            blockers.append("authorization: a live campaign must require authorization")
        if authorization.spending_cap_usd is None:
            blockers.append(
                "authorization: live campaign has no operator spending cap; "
                "authorization.spending_cap_usd must be a number"
            )
        if authorization.spending_cap_usd is not None and not authorization.authorized_by:
            blockers.append(
                "authorization: spending cap names no authorized_by operator; "
                "a cap without a named approver is not authorization"
            )
    if not live_authorization:
        blockers.append(
            "authorization: planning request alone does not authorize live dispatch; "
            f"set {AUTHORIZATION_ENV}=1 with explicit operator approval"
        )
    detail = "operator authorization and spending cap configured" if not blockers else None
    return blockers, PreflightCheck(name="authorization", passed=not blockers, detail=detail)


def _check_materialization(manifest: CampaignManifest) -> tuple[list[str], PreflightCheck]:
    blockers: list[str] = []
    if not manifest.materialized:
        blockers.append(
            "materialization: item selection has not run; "
            "item_ids are empty so there is nothing to dispatch"
        )
    if not manifest.endpoints:
        blockers.append("materialization: no endpoints declared for live observation")
    for spec in manifest.benchmarks:
        if not spec.item_ids:
            blockers.append(
                f"materialization: benchmark {spec.benchmark_id!r} has no frozen item_ids"
            )
    unknown = manifest.benchmarks_with_unknown_size
    if unknown:
        blockers.append(
            f"materialization: item count is unknown for {list(unknown)}; "
            "their split must be enumerated before the request cap can be checked"
        )
    detail = "frozen selection present for every benchmark" if not blockers else None
    return blockers, PreflightCheck(name="materialization", passed=not blockers, detail=detail)


def _check_credentials(
    manifest: CampaignManifest, env: Mapping[str, str]
) -> tuple[list[str], PreflightCheck]:
    blockers: list[str] = []
    for endpoint in manifest.endpoints:
        if endpoint.transport == "fixture":
            continue
        if endpoint.credential_ref is None:
            blockers.append(
                f"credentials: endpoint {endpoint.endpoint_id!r} "
                "names no credential_ref and no authorized_by is recorded"
            )
        elif not env.get(endpoint.credential_ref):
            blockers.append(
                f"credentials: {endpoint.credential_ref!r} "
                f"for endpoint {endpoint.endpoint_id!r} is not present in the environment"
            )
    detail = "every live endpoint resolves a configured credential" if not blockers else None
    return blockers, PreflightCheck(name="credentials", passed=not blockers, detail=detail)


def _check_endpoint_modes(manifest: CampaignManifest) -> tuple[list[str], PreflightCheck]:
    blockers: list[str] = []
    for endpoint in manifest.endpoints:
        if endpoint.transport == "fixture":
            blockers.append(
                f"endpoint_modes: endpoint {endpoint.endpoint_id!r} uses fixture transport; "
                "a live campaign requires transport 'zen' or 'reference'"
            )
        elif endpoint.transport not in ("zen", "reference"):
            blockers.append(
                f"endpoint_modes: endpoint {endpoint.endpoint_id!r} uses "
                f"unsupported transport {endpoint.transport!r}"
            )
    detail = "all endpoints use live transports" if not blockers else None
    return blockers, PreflightCheck(name="endpoint_modes", passed=not blockers, detail=detail)


def _check_revisions(manifest: CampaignManifest) -> tuple[list[str], PreflightCheck]:
    blockers: list[str] = []
    for spec in manifest.benchmarks:
        if spec.dataset_id is not None and spec.dataset_revision is None:
            blockers.append(
                f"revisions: benchmark {spec.benchmark_id!r} declares "
                f"dataset {spec.dataset_id!r} but no dataset_revision pin"
            )
        if spec.evaluator_id is not None and spec.evaluator_revision is None:
            blockers.append(
                f"revisions: benchmark {spec.benchmark_id!r} declares "
                f"evaluator {spec.evaluator_id!r} but no evaluator_revision pin"
            )
        if (
            spec.track == "agent"
            and spec.dataset_revision is None
            and spec.image_digest is None
            and spec.item_digest_manifest is None
        ):
            blockers.append(
                f"revisions: agent benchmark {spec.benchmark_id!r} pins no "
                "dataset_revision, image_digest or item_digest_manifest"
            )
    detail = "dataset and evaluator revisions pinned" if not blockers else None
    return blockers, PreflightCheck(name="revisions", passed=not blockers, detail=detail)


def _check_pricing(manifest: CampaignManifest) -> tuple[list[str], PreflightCheck]:
    blockers: list[str] = []
    limits = manifest.limits
    if limits.max_total_cost_usd is None:
        blockers.append(
            "pricing: limits.max_total_cost_usd must be a number for live dispatch; "
            "an absent cap is not permission to spend an unknown amount"
        )
    if not limits.require_cost_bounds:
        blockers.append("pricing: limits.require_cost_bounds must be true for live dispatch")
    if limits.missingness_threshold is None:
        blockers.append("pricing: limits.missingness_threshold must be frozen before dispatch")
    authorization = manifest.authorization
    if (
        authorization is not None
        and authorization.spending_cap_usd is not None
        and limits.max_total_cost_usd is not None
        and authorization.spending_cap_usd < limits.max_total_cost_usd
    ):
        blockers.append(
            f"pricing: operator spending cap {authorization.spending_cap_usd} is below "
            f"limits.max_total_cost_usd {limits.max_total_cost_usd}"
        )
    for endpoint in manifest.endpoints:
        if endpoint.transport == "fixture":
            continue
        pricing = endpoint.pricing
        if pricing.input_per_mtok is None or pricing.output_per_mtok is None:
            blockers.append(
                f"pricing: endpoint {endpoint.endpoint_id!r} has no price snapshot; "
                "unknown price blocks spending-capped execution"
            )
        if pricing.snapshot_id is None:
            blockers.append(
                f"pricing: endpoint {endpoint.endpoint_id!r} names no pricing.snapshot_id"
            )
    detail = "price snapshots and caps present" if not blockers else None
    return blockers, PreflightCheck(name="pricing", passed=not blockers, detail=detail)


def _check_runtime(require_container: bool) -> tuple[list[str], PreflightCheck]:
    from stealthbench.sandbox.runtime import (
        container_runtime,
        memory_limit_enforceable,
        posix_rlimit_available,
    )

    blockers: list[str] = []
    runtime = container_runtime()
    if require_container and runtime is None:
        blockers.append(
            "runtime: blocked_external: no container runtime (docker/podman) on PATH "
            "for live code/agent execution"
        )
        detail: str | None = None
    else:
        detail = (
            f"container={runtime or 'not-required'}, "
            f"posix_rlimit={posix_rlimit_available()}, "
            f"memory_limit_enforceable={memory_limit_enforceable()}"
        )
    return blockers, PreflightCheck(name="runtime", passed=not blockers, detail=detail)


def run_preflight(
    manifest: CampaignManifest,
    *,
    env: Mapping[str, str] | None = None,
    live_authorization: bool | None = None,
    require_container: bool = True,
) -> PreflightReport:
    """Verify every mandatory live requirement without dispatching anything.

    ``live_authorization`` defaults to the ``STEALTHBENCH_LIVE_AUTHORIZATION``
    environment flag so that a manifest on disk — a planning request — can never
    read as approval. Pass ``live_authorization=True`` only from a code path that
    already established explicit operator approval.
    """
    source: Mapping[str, str] = os.environ if env is None else env
    granted = (
        live_authorization if live_authorization is not None else live_authorization_granted(source)
    )

    blockers: list[str] = []
    checks: list[PreflightCheck] = []

    auth_blockers, auth_check = _check_authorization(manifest, granted)
    blockers.extend(auth_blockers)
    checks.append(auth_check)

    mat_blockers, mat_check = _check_materialization(manifest)
    blockers.extend(mat_blockers)
    checks.append(mat_check)

    cred_blockers, cred_check = _check_credentials(manifest, source)
    blockers.extend(cred_blockers)
    checks.append(cred_check)

    mode_blockers, mode_check = _check_endpoint_modes(manifest)
    blockers.extend(mode_blockers)
    checks.append(mode_check)

    rev_blockers, rev_check = _check_revisions(manifest)
    blockers.extend(rev_blockers)
    checks.append(rev_check)

    price_blockers, price_check = _check_pricing(manifest)
    blockers.extend(price_blockers)
    checks.append(price_check)

    runtime_blockers, runtime_check = _check_runtime(require_container)
    blockers.extend(runtime_blockers)
    checks.append(runtime_check)

    return PreflightReport(
        campaign_id=manifest.campaign_id,
        manifest_hash=manifest_hash(manifest),
        passed=not blockers,
        blockers=tuple(blockers),
        checks=tuple(checks),
    )
