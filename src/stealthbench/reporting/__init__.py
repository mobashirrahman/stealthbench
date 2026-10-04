"""Static leaderboard, redacted exports and local review (G13).

Offline by construction: pure file reads plus atomic writes. No transport,
no socket (except loopback review serving), no subprocess, no credential.
"""

from __future__ import annotations

__all__: tuple[str, ...] = ()
