"""Canonical serialization and content hashing.

Two properties matter and they are easy to confuse:

* **Insignificant ordering is insignificant.** JSON object key order must never
  change a digest, or every reserialization would invalidate stored artifacts.
* **Significant ordering is significant.** List order carries meaning here: a
  reordered ``item_ids`` is a different frozen selection and must hash
  differently.

Non-finite floats are rejected outright. ``NaN`` is not valid JSON, and allowing
it would let two different payloads canonicalize to the same text.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

DIGEST_ALGORITHM: Final[str] = "sha256"
DIGEST_LENGTH: Final[int] = 64

_CANONICAL_SEPARATORS: Final[tuple[str, str]] = (",", ":")


class NonCanonicalValue(ValueError):
    """A value that cannot be canonically serialized."""


def _reject_non_finite(value: Any, path: str = "$") -> None:
    """Raise for NaN or infinity anywhere in the structure.

    ``json.dumps(allow_nan=False)`` also rejects these, but only at the leaves it
    happens to reach. Walking first produces a path that names the offender.
    """
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            raise NonCanonicalValue(f"{path} is not finite: {value!r}")
    elif isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise NonCanonicalValue(f"{path} has a non-string key {key!r}")
            _reject_non_finite(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_non_finite(child, f"{path}[{index}]")
    elif value is None or isinstance(value, (str, bool, int)):
        return
    else:
        raise NonCanonicalValue(f"{path} has unsupported type {type(value).__name__}")


def canonical_json(value: Any) -> str:
    """Serialize to canonical JSON: sorted keys, no insignificant whitespace.

    Sequence order is preserved because it is semantic in this project.
    """
    _reject_non_finite(value)
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=_CANONICAL_SEPARATORS,
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:  # pragma: no cover - guarded by _reject_non_finite
        raise NonCanonicalValue(str(exc)) from exc


def canonical_bytes(value: Any) -> bytes:
    """Canonical UTF-8 bytes, the actual input to the digest."""
    return canonical_json(value).encode("utf-8")


def content_digest(value: Any) -> str:
    """Return the hex sha256 of the canonical form of ``value``."""
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_digest(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """Hash a file's raw bytes.

    Used for prompt artifacts and fixture transcripts, where the identity of the
    bytes on disk is what matters rather than a parsed representation.
    """
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def stable_id(prefix: str, value: Any, length: int = 16) -> str:
    """A short, stable identifier derived from content.

    The prefix makes the namespace readable in logs; the digest keeps the value
    reproducible across processes and machines.
    """
    if not prefix:
        raise ValueError("stable_id requires a non-empty prefix")
    if length < 8 or length > DIGEST_LENGTH:
        raise ValueError(f"length must be between 8 and {DIGEST_LENGTH}, got {length}")
    return f"{prefix}-{content_digest(value)[:length]}"


def is_digest(value: str) -> bool:
    """Whether a string looks like a hex sha256 digest produced by this module."""
    return len(value) == DIGEST_LENGTH and all(char in "0123456789abcdef" for char in value)


def sort_for_hashing(value: Any) -> Any:
    """Recursively sort mappings into key order, leaving sequence order alone.

    Provided for callers that need a human-readable dump which is guaranteed to
    produce the same digest as the canonical form.
    """
    if isinstance(value, Mapping):
        return {key: sort_for_hashing(value[key]) for key in sorted(value)}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [sort_for_hashing(item) for item in value]
    return value
