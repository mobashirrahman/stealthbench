"""Shared pytest fixtures.

The single most important rule enforced here: **offline suites cannot reach the
network**. Markers are not a security boundary (``IMPLEMENTATION_PLAN.md``
section 6), so the default fixture set denies real sockets outright and any
suite that genuinely needs egress must say so explicitly with
``@pytest.mark.network``.
"""

from __future__ import annotations

import os
import socket
from collections.abc import Iterator
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Marker that must be declared for a test to be allowed to open a socket.
NETWORK_MARKER = "network"

# Enables the `pytester` fixture so configuration invariants (marker strictness,
# suite selection) can be checked by running pytest against generated tests.
pytest_plugins = ("pytester",)


class NetworkAccessDenied(RuntimeError):
    """Raised when an offline test attempts an external network connection."""


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        f"{NETWORK_MARKER}: explicitly opt in to external network access (offline suites deny it)",
    )


def _is_local(address: object) -> bool:
    if not isinstance(address, tuple) or not address:
        return False
    host = address[0]
    if isinstance(host, bytes):
        host = host.decode("ascii", "replace")
    return host in {"127.0.0.1", "::1", "localhost", "", "0.0.0.0"}


def _build_guarded_socket_class(base: type[socket.socket]) -> type[socket.socket]:
    """Return a socket subclass that refuses non-loopback sends of any kind.

    ``socket.socket.connect`` is a read-only C slot, so the guard has to be a
    subclass installed over ``socket.socket`` rather than an instance attribute.

    UDP is guarded as well as TCP: blocking only ``connect`` would leave
    ``sendto``/``sendmsg`` as a one-line egress path out of an offline suite.
    """

    def _refuse(address: object, operation: str) -> NetworkAccessDenied:
        return NetworkAccessDenied(
            f"offline test attempted {operation} to {address!r}; "
            f"mark it with @{NETWORK_MARKER} if that is genuinely required"
        )

    class GuardedSocket(base):  # type: ignore[valid-type,misc]
        def connect(self, address: object) -> None:
            if not _is_local(address):
                raise _refuse(address, "an external connection")
            super().connect(address)  # type: ignore[arg-type]

        def connect_ex(self, address: object) -> int:
            if not _is_local(address):
                raise _refuse(address, "an external connection")
            return int(super().connect_ex(address))  # type: ignore[arg-type]

        def sendto(self, data: object, address: object, *args: object) -> int:
            if not _is_local(address):
                raise _refuse(address, "an external datagram")
            return int(super().sendto(data, address, *args))  # type: ignore[arg-type]

        def sendmsg(
            self, buffers: object, ancdata: object = (), flags: int = 0, address: object = None
        ) -> int:
            if address is not None and not _is_local(address):
                raise _refuse(address, "an external datagram")
            return int(super().sendmsg(buffers, ancdata, flags, address))  # type: ignore[arg-type]

    return GuardedSocket


@pytest.fixture(autouse=True)
def deny_external_network(request: pytest.FixtureRequest) -> Iterator[None]:
    """Fail any test that tries to reach a non-loopback address.

    Loopback is allowed so that tests may start local fixture servers; anything
    else is a bug in an offline suite.
    """
    if NETWORK_MARKER in request.keywords:
        yield
        return

    real_socket_class = socket.socket
    real_create_connection = socket.create_connection
    real_getaddrinfo = socket.getaddrinfo
    guarded_class = _build_guarded_socket_class(real_socket_class)

    def guarded_create_connection(
        address: object, *args: object, **kwargs: object
    ) -> socket.socket:
        if not _is_local(address):
            raise NetworkAccessDenied(f"offline test attempted connection to {address!r}")
        return real_create_connection(address, *args, **kwargs)  # type: ignore[arg-type]

    def guarded_getaddrinfo(
        host: object, port: object, *args: object, **kwargs: object
    ) -> list[tuple]:
        probe = host.decode("ascii", "replace") if isinstance(host, bytes) else host
        if isinstance(probe, str) and probe not in {"127.0.0.1", "::1", "localhost", ""}:
            raise NetworkAccessDenied(f"offline test attempted DNS lookup for {probe!r}")
        return real_getaddrinfo(host, port, *args, **kwargs)  # type: ignore[arg-type]

    socket.socket = guarded_class  # type: ignore[misc,assignment]
    socket.create_connection = guarded_create_connection  # type: ignore[assignment]
    socket.getaddrinfo = guarded_getaddrinfo  # type: ignore[assignment]
    try:
        yield
    finally:
        socket.socket = real_socket_class  # type: ignore[misc]
        socket.create_connection = real_create_connection
        socket.getaddrinfo = real_getaddrinfo


@pytest.fixture
def repo_root() -> Path:
    """Absolute path to the repository root."""
    return REPO_ROOT


#: Substrings that mark an environment variable as credential-bearing.
CREDENTIAL_ENV_FRAGMENTS = ("API_KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")


@pytest.fixture(autouse=True)
def scrub_credentials_from_offline_tests(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Strip provider credentials for every offline test.

    Autouse on purpose: a fixture nothing requests protects nothing. Live and paid
    tests are exempt, because checking the authorization path is their whole job.
    """
    if request.node.get_closest_marker("live") or request.node.get_closest_marker("paid"):
        return
    for name in list(os.environ):
        if any(fragment in name.upper() for fragment in CREDENTIAL_ENV_FRAGMENTS):
            monkeypatch.delenv(name, raising=False)
