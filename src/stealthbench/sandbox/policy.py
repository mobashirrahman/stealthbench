"""Evaluator separation and containment policy (task T07B, gate G07).

This module decides what may enter a candidate execution environment. It
never executes anything itself; enforcement at write time lives here, and
enforcement at run time lives in :mod:`stealthbench.sandbox.runtime`, which
calls back into these checks before staging files or building the child
environment.

The guarantees:

* **Path traversal denied.** Declared file names must be relative and must
  not contain ``..``. Absolute paths are rejected outright.
* **Symlink escape denied.** A name that resolves (through existing
  symlinks) outside the work directory is rejected, so ``link -> /elsewhere``
  cannot be used to write or address host files.
* **Verifier separation.** Hidden tests live in a ``verifier/`` directory
  that is a sibling of the candidate ``work/`` directory, never nested
  inside it. The two must never overlap.
* **Secrets never mounted.** Secret-bearing environment variables are
  stripped from the candidate environment, and explicitly passing a
  secret-named variable raises instead of leaking it.
* **Network denied by posture.** Proxies and credentials are removed from
  the candidate environment. That removes proxy-mediated egress; true
  packet-level denial needs the container path (``--network none``), which
  is why strict network checks require a container runtime.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

__all__ = [
    "CREDENTIAL_ENV_FRAGMENTS",
    "EvaluatorLayout",
    "PolicyDenied",
    "assert_no_secret_in_env",
    "assert_verifier_separation",
    "candidate_env",
    "candidate_visible_paths",
    "is_secret_env_name",
    "network_policy",
    "resolve_candidate_path",
    "scrub_env",
    "write_candidate_file",
]

#: Substrings that mark an environment variable as credential-bearing. Mirrors
#: the offline-test scrub in ``tests/conftest.py`` so both layers agree.
CREDENTIAL_ENV_FRAGMENTS: Final[tuple[str, ...]] = (
    "API_KEY",
    "TOKEN",
    "SECRET",
    "PASSWORD",
    "CREDENTIAL",
)

#: Proxy variables removed from the candidate environment. A sandbox that
#: inherits the operator's egress proxy is not network-denied.
_PROXY_VARIABLES: Final[frozenset[str]] = frozenset(
    {
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "NO_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "no_proxy",
    }
)

#: Variable prefixes reserved for the evaluator side. The candidate must
#: never learn where the verifier lives, so these never enter its env.
_VERIFIER_VAR_PREFIXES: Final[tuple[str, ...]] = (
    "VERIFIER_",
    "EVALUATOR_",
    "GOLD_",
    "STEALTHBENCH_GOLD",
)


class PolicyDenied(Exception):
    """A staging request violated evaluator separation.

    Raised rather than defaulted: a traversal, symlink escape, secret mount
    or verifier overlap must stop the run, never be coerced into something
    that looks allowed.
    """

    def __init__(self, kind: str, detail: str) -> None:
        super().__init__(f"{kind}: {detail}")
        self.kind = kind
        self.detail = detail


def is_secret_env_name(name: str) -> bool:
    """Whether an environment variable name is credential-bearing."""
    upper = name.upper()
    return any(fragment in upper for fragment in CREDENTIAL_ENV_FRAGMENTS)


def scrub_env(env: Mapping[str, str]) -> dict[str, str]:
    """Remove secrets, proxies and evaluator pointers from an environment."""
    clean: dict[str, str] = {}
    for key, value in env.items():
        if is_secret_env_name(key):
            continue
        if key in _PROXY_VARIABLES:
            continue
        if key.startswith(_VERIFIER_VAR_PREFIXES):
            continue
        clean[key] = value
    return clean


def candidate_env(workdir: Path, extra: Mapping[str, str] | None = None) -> dict[str, str]:
    """Build the minimal environment a candidate process may see.

    Only ``PATH`` survives from the operator environment (needed to find the
    interpreter); everything else is fixed and pointed inside the work
    directory. ``extra`` entries with secret-bearing names raise
    :class:`PolicyDenied` instead of mounting the secret; proxy and
    evaluator-pointer entries are dropped, since the candidate is never
    allowed proxied egress or knowledge of the verifier location.
    """
    parent = scrub_env(os.environ)
    env = {
        "PATH": parent.get("PATH", "/usr/bin:/bin:/usr/local/bin"),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "HOME": str(workdir),
        "TMPDIR": str(workdir),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
    }
    if extra:
        for key, value in extra.items():
            if is_secret_env_name(key):
                raise PolicyDenied(
                    "secret-mount",
                    f"refusing to mount secret-bearing variable {key!r} "
                    "into the candidate environment",
                )
            if key in _PROXY_VARIABLES or key.startswith(_VERIFIER_VAR_PREFIXES):
                continue
            env[key] = value
    return env


def resolve_candidate_path(workdir: Path, name: str) -> Path:
    """Resolve a declared file name strictly inside the work directory.

    Rejects empty names, absolute paths and ``..`` components before touching
    the filesystem, then resolves existing symlinks and rejects anything
    that escapes the work directory.
    """
    try:
        candidate = Path(name)
    except (TypeError, ValueError) as exc:
        raise PolicyDenied("invalid-name", f"unusable file name {name!r}: {exc}") from exc
    if not name or name.strip() in {"", "."}:
        raise PolicyDenied("invalid-name", f"empty file name {name!r}")
    if candidate.is_absolute():
        raise PolicyDenied("absolute-path", f"candidate path must be relative: {name!r}")
    if ".." in candidate.parts:
        raise PolicyDenied("traversal", f"candidate path escapes its directory: {name!r}")
    base = workdir.resolve()
    target = (base / candidate).resolve()
    if target != base and base not in target.parents:
        raise PolicyDenied(
            "symlink-escape",
            f"candidate path {name!r} resolves outside its directory",
        )
    return target


def write_candidate_file(workdir: Path, name: str, content: str | bytes) -> Path:
    """Stage one candidate file after the separation checks pass."""
    target = resolve_candidate_path(workdir, name)
    target.parent.mkdir(parents=True, exist_ok=True)
    # Re-check after creating parents: a pre-existing symlink on the new
    # parent chain could otherwise redirect the write after validation.
    base = workdir.resolve()
    if target.resolve() != base and base not in target.resolve().parents:
        raise PolicyDenied(
            "symlink-escape",
            f"candidate path {name!r} resolves outside its directory",
        )
    if isinstance(content, bytes):
        target.write_bytes(content)
    else:
        target.write_text(content, encoding="utf-8")
    return target


def assert_verifier_separation(workdir: Path, verifier_dir: Path) -> None:
    """Require the verifier directory to sit apart from candidate space."""
    work = workdir.resolve()
    verifier = verifier_dir.resolve()
    if work == verifier:
        raise PolicyDenied(
            "verifier-overlap", "candidate workdir and verifier directory are identical"
        )
    if verifier in work.parents or work in verifier.parents:
        raise PolicyDenied(
            "verifier-overlap",
            "candidate workdir and verifier directory must be siblings, never nested",
        )


def assert_no_secret_in_env(env: Mapping[str, str], known_values: Sequence[str]) -> None:
    """Assert no known secret value appears in a candidate environment."""
    for value in known_values:
        if not value:
            continue
        for key, entry in env.items():
            if value in entry:
                raise PolicyDenied(
                    "secret-mounted",
                    f"secret value present in candidate variable {key!r}",
                )


def candidate_visible_paths(workdir: Path) -> list[str]:
    """File names visible from inside the candidate directory, relative."""
    visible: list[str] = []
    for path in sorted(workdir.rglob("*")):
        try:
            if path.is_symlink() or path.is_file() or path.is_dir():
                visible.append(str(path.relative_to(workdir)))
        except OSError:
            continue
    return visible


@dataclass(frozen=True, slots=True)
class EvaluatorLayout:
    """Sibling directories for candidate work and hidden verifier assets."""

    root: Path
    workdir: Path
    verifier_dir: Path

    @classmethod
    def create(cls, root: Path | str) -> EvaluatorLayout:
        """Create ``work/`` and ``verifier/`` side by side under ``root``."""
        base = Path(root)
        layout = cls(root=base, workdir=base / "work", verifier_dir=base / "verifier")
        layout.workdir.mkdir(parents=True, exist_ok=False)
        layout.verifier_dir.mkdir(parents=True, exist_ok=False)
        layout.assert_separation()
        return layout

    def assert_separation(self) -> None:
        """Fail if the verifier could be reached from candidate space."""
        assert_verifier_separation(self.workdir, self.verifier_dir)


def network_policy() -> dict[str, str]:
    """The declared network posture for candidate execution."""
    return {
        "egress": "deny",
        "ingress": "deny",
        "proxy": "removed from the candidate environment",
        "credentials": "removed from the candidate environment",
        "enforcement": (
            "container runs use --network none; the subprocess backend scrubs "
            "proxies and credentials but cannot filter loopback, so strict "
            "network checks require a container runtime"
        ),
    }
