"""StealthBench: reproducible benchmarking and identity estimation for anonymous endpoints.

This package is under active development. Importing it makes no network call and
loads no benchmark data. Individual capabilities land gate by gate; see
``IMPLEMENTATION_PLAN.md`` for the gate map and ``docs/contracts.md`` for the
frozen interfaces.
"""

from __future__ import annotations

from typing import Final

__version__: Final[str] = "0.1.0"

#: Identifier of the frozen configuration/result contract version implemented so far.
SCHEMA_VERSION: Final[str] = "1.0"

__all__ = ["SCHEMA_VERSION", "__version__"]
