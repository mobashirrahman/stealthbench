"""Command line interface.

Design rules enforced here (see ``IMPLEMENTATION_PLAN.md`` sections 3 and 5):

* A command that cannot actually do its work exits non-zero and says why. No
  command may print a score, an identity estimate or a success claim it has not
  computed from artifacts.
* Invalid usage and invalid inputs are distinguishable from "this gate has not
  landed yet", so an unimplemented capability is never mistaken for a pass.
* Nothing in this module contacts a provider. Live dispatch is added in G04/G14
  behind explicit campaign configuration and spending caps.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Final, TextIO

from stealthbench import SCHEMA_VERSION, __version__

PROG: Final[str] = "stealthbench"

EXIT_OK: Final[int] = 0
EXIT_ERROR: Final[int] = 1
EXIT_USAGE: Final[int] = 2
EXIT_NOT_IMPLEMENTED: Final[int] = 3

Handler = Callable[["argparse.Namespace", TextIO, TextIO], int]


class UserError(Exception):
    """Invalid input supplied by the operator. Maps to :data:`EXIT_ERROR`."""


class GatePending(Exception):
    """A declared command whose implementation has not landed yet.

    This exists so that an incomplete gate is reported as incomplete. It must
    never be raised in a way that could be read as a successful evaluation.
    """

    def __init__(self, command: str, gate: str) -> None:
        super().__init__(f"'{command}' is not implemented yet; scheduled for gate {gate}")
        self.command = command
        self.gate = gate


def _existing_file(value: str) -> Path:
    path = Path(value)
    if not path.is_file():
        raise UserError(f"not a file: {value}")
    return path


def _existing_path(value: str) -> Path:
    path = Path(value)
    if not path.exists():
        raise UserError(f"no such file or directory: {value}")
    return path


# ---------------------------------------------------------------------------
# Implemented commands
# ---------------------------------------------------------------------------


def cmd_version(args: argparse.Namespace, stdout: TextIO, stderr: TextIO) -> int:
    """Print the installed package version and schema contract version."""
    del args, stderr
    payload = {
        "status": "ok",
        "version": __version__,
        "schema_version": SCHEMA_VERSION,
    }
    print(json.dumps(payload, sort_keys=True), file=stdout)
    return EXIT_OK


# ---------------------------------------------------------------------------
# Declared commands that land in later gates
# ---------------------------------------------------------------------------


def cmd_manifest_validate(args: argparse.Namespace, stdout: TextIO, stderr: TextIO) -> int:
    """Validate a campaign manifest against the frozen schema.

    Validation is a precondition for dispatch, not a dry run: an invalid manifest
    fails here, before any provider is contacted. A valid manifest that is not yet
    authorized to run is reported as such and still exits 0, because a valid
    document that awaits an operator decision is not an error.
    """
    # Imported here, not at module scope: the CLI shell must stay importable with no
    # third-party dependency so a bare install can still show help and report errors.
    from pydantic import ValidationError

    from stealthbench.schemas.campaign import CampaignManifest
    from stealthbench.schemas.manifest import manifest_hash

    try:
        manifest = CampaignManifest.model_validate_json(Path(args.path).read_text(encoding="utf-8"))
    except ValidationError as exc:
        raise UserError(f"{args.path} is not a valid campaign manifest:\n{exc}") from exc
    except OSError as exc:
        raise UserError(f"cannot read {args.path}: {exc.strerror or exc}") from exc

    digest = manifest_hash(manifest)
    blockers = manifest.dispatch_blockers()
    payload: dict[str, Any] = {
        "status": "valid",
        "campaign_id": manifest.campaign_id,
        "mode": manifest.mode,
        "materialized": manifest.materialized,
        "manifest_hash": digest,
        "endpoints": len(manifest.endpoints),
        "benchmarks": len(manifest.benchmarks),
        "planned_requests": manifest.total_planned_requests,
        "known_planned_requests": manifest.known_planned_requests,
        "benchmarks_with_unknown_size": list(manifest.benchmarks_with_unknown_size),
        "dispatchable": not blockers,
        "dispatch_blockers": list(blockers),
    }
    print(json.dumps(payload, indent=2, sort_keys=True), file=stdout)
    return EXIT_OK


def cmd_run(args: argparse.Namespace, stdout: TextIO, stderr: TextIO) -> int:
    """Execute a campaign, or report exactly why it cannot run.

    ``--offline`` dispatches through the fixture-only vertical workflow (G05):
    no provider socket, caps enforced, grades via the IFEval wrapper, redacted
    exports written. Without ``--offline`` (and without ``--live``) the command
    remains pending so a bare invocation can never read as a completed campaign.
    A live run additionally requires a configured credential and an operator
    spending cap, and is refused naming what is missing.
    """
    from pydantic import ValidationError

    from stealthbench.runner import RunMode, resolve_mode
    from stealthbench.schemas.campaign import Authorization, CampaignManifest

    try:
        manifest = CampaignManifest.model_validate_json(Path(args.path).read_text(encoding="utf-8"))
    except ValidationError as exc:
        raise UserError(f"{args.path} is not a valid campaign manifest:\n{exc}") from exc
    except OSError as exc:
        raise UserError(f"cannot read {args.path}: {exc.strerror or exc}") from exc

    mode = RunMode.LIVE if args.live else RunMode.OFFLINE
    # A manifest with no authorization block has not been approved by an operator. That
    # is the same state as an approval whose cap was never set: asserted, uncapped.
    authorization = manifest.authorization or Authorization()
    missing = resolve_mode(
        requested=mode,
        authorization=authorization,
        credentials_configured=False,
        spending_cap=authorization.spending_cap_usd,
    )
    if missing:
        # Refusing with the list is more useful than a permission error from deeper in.
        print(
            json.dumps(
                {
                    "blockers": [f"live execution is missing: {item}" for item in missing],
                    "campaign_id": manifest.campaign_id,
                    "dispatched": 0,
                    "mode": str(mode),
                    "status": "refused",
                },
                indent=2,
                sort_keys=True,
            ),
            file=stdout,
        )
        return EXIT_USAGE
    if not bool(getattr(args, "offline", False)):
        # Bare `run` without an explicit mode flag stays pending: exit 3, not 0.
        # A caller that sees success would conclude the campaign ran, and an
        # exit code is the only part of this output a script cannot misread.
        print(f"{PROG}: error: 'run' needs --offline (fixture) or --live", file=stderr)
        raise GatePending("run", "G05")

    from stealthbench.benchmarks.workflow import load_fixture_bundle, run_offline

    fixture_bundle = None
    fixture_arg = getattr(args, "fixture", None)
    if fixture_arg is not None:
        try:
            fixture_bundle = load_fixture_bundle(Path(fixture_arg))
        except (OSError, ValueError) as exc:
            raise UserError(f"cannot load fixture bundle {fixture_arg}: {exc}") from exc
    output_arg = getattr(args, "output", None)
    artifacts_dir = (
        Path(output_arg) if output_arg is not None else Path("artifacts") / manifest.campaign_id
    )
    try:
        report = run_offline(
            manifest,
            artifacts_dir=artifacts_dir,
            fixture_bundle=fixture_bundle,
        )
    except ValueError as exc:
        raise UserError(str(exc)) from exc
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True), file=stdout)
    return EXIT_OK


def cmd_replay(args: argparse.Namespace, stdout: TextIO, stderr: TextIO) -> int:
    """Reconstruct stored samples and grades without contacting a provider."""
    from stealthbench.benchmarks.workflow import replay_offline

    root = Path(args.path)
    if not root.exists():
        raise UserError(f"no such file or directory: {args.path}")
    if root.is_file() or not (root / "events.jsonl").exists():
        raise UserError(
            f"{args.path} is not an artifact directory: expected events.jsonl inside it"
        )
    try:
        summary = replay_offline(root)
    except ValueError as exc:
        raise UserError(str(exc)) from exc
    except Exception as exc:
        raise UserError(f"cannot replay {args.path}: {exc}") from exc
    del stderr
    print(json.dumps(summary.to_dict(), indent=2, sort_keys=True), file=stdout)
    return EXIT_OK


def cmd_report(args: argparse.Namespace, stdout: TextIO, stderr: TextIO) -> int:
    """Build a static report from stored artifacts.

    Offline by construction: reads the artifact directory only and writes a
    browsable static site plus redacted JSON/CSV exports. Never contacts a
    provider and never deploys anywhere; review serving is loopback-only.
    """
    from stealthbench.reporting.site import build_report

    del stderr
    root = Path(args.path)
    if not root.exists():
        raise UserError(f"no such file or directory: {args.path}")
    if root.is_file() or not (root / "manifest.json").exists():
        raise UserError(
            f"{args.path} is not an artifact directory: expected manifest.json inside it"
        )
    output = Path(args.output)
    try:
        summary = build_report(root, output)
    except ValueError as exc:
        raise UserError(str(exc)) from exc
    print(json.dumps(summary, indent=2, sort_keys=True), file=stdout)
    return EXIT_OK


def cmd_signatures(args: argparse.Namespace, stdout: TextIO, stderr: TextIO) -> int:
    """Collect and compare endpoint signature observations."""
    del args, stdout, stderr
    raise GatePending("signatures", "G11")


def cmd_identify(args: argparse.Namespace, stdout: TextIO, stderr: TextIO) -> int:
    """Produce an identity report for an endpoint observation."""
    del args, stdout, stderr
    raise GatePending("identify", "G12")


def cmd_doctor(args: argparse.Namespace, stdout: TextIO, stderr: TextIO) -> int:
    """Report which optional runtimes and datasets are actually available.

    Offline and dependency-free by construction: no provider socket, no
    credential, no dataset download. Container-only capabilities report
    ``blocked_external`` when the runtime is absent instead of fake-passing,
    and official datasets report ``blocked_external`` until their revisions
    are pinned and materialized. Never prints a benchmark outcome.
    """
    import platform

    del args, stderr
    from stealthbench.sandbox.runtime import (
        container_runtime,
        memory_limit_enforceable,
        posix_rlimit_available,
    )

    runtime_name = container_runtime()
    if runtime_name is None:
        container_state = "blocked_external"
        container_reason = "blocked_external: no container runtime (docker/podman) on PATH"
    else:
        container_state = "ready"
        container_reason = None

    configs: dict[str, str] = {}
    try:
        from stealthbench.schemas.campaign import CampaignManifest as _Manifest

        manifest_layer: Any = _Manifest
    except ImportError:
        manifest_layer = None
    for profile in ("offline-demo", "pilot", "full"):
        candidate = Path(f"configs/{profile}.json")
        if not candidate.is_file():
            configs[profile] = "missing"
            continue
        if manifest_layer is None:
            configs[profile] = "unknown-dependency-unavailable"
            continue
        try:
            manifest = manifest_layer.model_validate_json(candidate.read_text(encoding="utf-8"))
            blockers = manifest.dispatch_blockers()
            if not blockers:
                configs[profile] = "valid-dispatchable"
            elif not manifest.materialized:
                configs[profile] = "valid-unmaterialized"
            else:
                configs[profile] = "valid-blocked"
        except Exception:
            configs[profile] = "invalid"

    inventory = Path("docs/upstream-inventory.md")
    payload: dict[str, Any] = {
        "configs": configs,
        "container_reason": container_reason,
        "container_runtime": runtime_name,
        "container_state": container_state,
        "datasets": {
            "inventory": "available" if inventory.is_file() else "missing",
            "official": (
                "blocked_external: revisions unpinned and selection "
                "unmaterialized until G05/G08 freeze them"
            ),
            "offline_demo": (
                "available" if configs.get("offline-demo") == "valid-dispatchable" else "missing"
            ),
        },
        "memory_limit_enforceable": memory_limit_enforceable(),
        "platform": platform.system(),
        "posix_rlimit": posix_rlimit_available(),
        "python": platform.python_version(),
        "sandbox_backend": "subprocess-disposable",
        "schema_version": SCHEMA_VERSION,
        "status": "ok",
        "version": __version__,
    }
    print(json.dumps(payload, indent=2, sort_keys=True), file=stdout)
    return EXIT_OK


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Construct the full argument parser.

    Every subcommand validates its own inputs through argparse so that a bad
    invocation can never reach a code path that could report a result.
    """
    parser = argparse.ArgumentParser(
        prog=PROG,
        description=(
            "Reproducible benchmarking and identity estimation for anonymous model endpoints."
        ),
        epilog=(
            "Live execution is never implicit. A run that contacts a provider requires an "
            "explicit campaign manifest, configured credentials and spending caps."
        ),
    )
    parser.add_argument("--version", action="version", version=f"{PROG} {__version__}")
    subparsers = parser.add_subparsers(dest="command", metavar="<command>", required=True)

    subparsers.add_parser("version", help="print package and schema versions").set_defaults(
        handler=cmd_version
    )

    manifest = subparsers.add_parser("manifest", help="campaign manifest operations")
    manifest_sub = manifest.add_subparsers(
        dest="manifest_command", metavar="<subcommand>", required=True
    )
    manifest_validate = manifest_sub.add_parser(
        "validate", help="validate a campaign manifest without contacting a provider"
    )
    manifest_validate.add_argument("path", type=_existing_file, help="path to the manifest JSON")
    manifest_validate.set_defaults(handler=cmd_manifest_validate)

    run = subparsers.add_parser("run", help="execute a campaign")
    run.add_argument("path", type=_existing_file, help="path to the campaign manifest JSON")
    run.add_argument(
        "--offline",
        action="store_true",
        help="forbid all provider network access; use recorded fixture responses only",
    )
    run.add_argument(
        "--live",
        action="store_true",
        help="dispatch to a provider; requires authorization, credentials and a spending cap",
    )
    run.add_argument(
        "--output",
        type=Path,
        required=False,
        default=None,
        help="artifact directory (default: artifacts/<campaign_id>)",
    )
    run.add_argument(
        "--fixture",
        type=Path,
        required=False,
        default=None,
        help="recorded fixture bundle JSON for offline dispatch",
    )
    run.set_defaults(handler=cmd_run)

    replay = subparsers.add_parser("replay", help="rebuild stored results without a provider")
    replay.add_argument("path", type=_existing_path, help="path to the artifact directory")
    replay.set_defaults(handler=cmd_replay)

    report = subparsers.add_parser("report", help="build a static report from artifacts")
    report.add_argument("path", type=_existing_path, help="path to the artifact directory")
    report.add_argument(
        "--output", type=Path, required=True, help="directory to write the report into"
    )
    report.set_defaults(handler=cmd_report)

    signatures = subparsers.add_parser("signatures", help="collect or compare endpoint signatures")
    signatures.add_argument("path", type=_existing_file, help="path to the signature manifest JSON")
    signatures.set_defaults(handler=cmd_signatures)

    identify = subparsers.add_parser("identify", help="estimate endpoint identity from signatures")
    identify.add_argument("path", type=_existing_file, help="path to the artifact directory")
    identify.add_argument("--output", type=Path, help="directory to write the identity report into")
    identify.set_defaults(handler=cmd_identify)

    subparsers.add_parser(
        "doctor", help="report optional runtime and dataset availability"
    ).set_defaults(handler=cmd_doctor)

    return parser


def main(argv: Sequence[str] | None = None, stdout: TextIO | None = None) -> int:
    """Run the CLI and return a process exit status."""
    out = sys.stdout if stdout is None else stdout
    parser = build_parser()

    try:
        args = parser.parse_args(sys.argv[1:] if argv is None else list(argv))
    except SystemExit as exc:
        # argparse exits 0 for --help/--version and 2 for usage errors.
        return int(exc.code) if exc.code is not None else EXIT_USAGE
    except UserError as exc:
        print(f"{PROG}: error: {exc}", file=sys.stderr)
        return EXIT_ERROR

    handler: Handler | None = getattr(args, "handler", None)
    if handler is None:  # pragma: no cover - subparsers are required, so argparse fails first
        print(f"{PROG}: error: a <command> is required", file=sys.stderr)
        return EXIT_USAGE

    try:
        result = handler(args, out, sys.stderr)
    except GatePending as exc:
        payload: dict[str, str] = {
            "status": "not_implemented",
            "command": exc.command,
            "gate": exc.gate,
        }
        print(json.dumps(payload, sort_keys=True), file=sys.stderr)
        return EXIT_NOT_IMPLEMENTED
    except UserError as exc:
        print(f"{PROG}: error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    except KeyboardInterrupt:
        print(f"{PROG}: interrupted", file=sys.stderr)
        return EXIT_ERROR

    return int(result) if isinstance(result, int) else EXIT_OK


if __name__ == "__main__":  # pragma: no cover - exercised via the console script
    sys.exit(main())
