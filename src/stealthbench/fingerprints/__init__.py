"""Observable model signatures (G11).

Offline fingerprint collection and similarity reporting. No transport, no
socket, no subprocess, no credential.

Submodules:

* :mod:`stealthbench.fingerprints.probes` -- the 60-probe bank (T11A).
* :mod:`stealthbench.fingerprints.features` -- validity masks and deltas (T11B).
* :mod:`stealthbench.fingerprints.similarity` -- ranked reference reports (T11C).

Identity rule for the whole package: a similarity of 0.9 is not a 90%
probability. ``calibrated_probabilities`` stays ``None`` until the G12
evidence gate passes.
"""

from __future__ import annotations

__all__: list[str] = []
