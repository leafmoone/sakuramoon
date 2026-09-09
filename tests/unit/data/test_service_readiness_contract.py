"""Readiness contract: the service socket must not exist before warmup.

Regression for 0U FAIL #2 (corrected 100U rerun, 09-08): the server
bound the Unix socket before ``wait_until_ready()``, while
``training_stack.sh`` treats socket existence as "data ready" and launches
the trainer. The trainer's health requests then sat in an empty backlog
until the 16-shard warmup barrier passed, past the 30s client timeout,
and the whole run died in preflight with zero successful updates.

The fix moves the socket creation to after the barrier. These tests
construct a blocking ``wait_until_ready()`` with a threading.Event — no
shard, cache, or download is involved, so a warm production cache cannot
make them pass.
"""

from __future__ import annotations

# pyright: reportPrivateUsage=false
import socket
import stat
import threading
import time
from collections.abc import Callable
from pathlib import Path

import pytest

from sakuramoon.data.client import DataServiceClient
from sakuramoon.data.service import (
    DataServiceLimits,
    DataServiceServer,
    DataSupplyService,
)
from sakuramoon.data.service_protocol import DataServiceSessionIdentity

_DATASET_ID = "readiness-contract-dataset"


def _gated_service(barrier: threading.Event, stop_event: threading.Event) -> DataSupplyService:
    """DataSupplyService surface with an Event-gated warmup barrier.

    Only the attributes ``DataServiceServer.serve``/``_dispatch`` touch are
    populated; ``wait_until_ready`` blocks until the barrier is released
    (returns False if stop is requested first, mirroring the real
    ``_ServiceStopping`` path).
    """
    service = object.__new__(DataSupplyService)
    service.identity = DataServiceSessionIdentity(
        dataset_id=_DATASET_ID,
        worker_count=2,
    )
    service.limits = DataServiceLimits(
        download_concurrency=2,
        verified_shard_lookahead=2,
        lease_channel_capacity=2,
        ack_channel_capacity=2,
    )

    def _start(stop: threading.Event) -> None:
        del stop

    def _wait_until_ready() -> bool:
        while not barrier.wait(0.05):
            if stop_event.is_set():
                return False
        return True

    def _close() -> None:
        return None

    service.start = _start  # type: ignore[method-assign]
    service.wait_until_ready = _wait_until_ready  # type: ignore[method-assign]
    service.close = _close  # type: ignore[method-assign]
    return service


def _wait_until(predicate: Callable[[], bool], timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


def test_socket_hidden_until_warmup_barrier_released(tmp_path: Path) -> None:
    barrier = threading.Event()
    stop_event = threading.Event()
    socket_path = tmp_path / "data-service.sock"
    server = DataServiceServer(
        _gated_service(barrier, stop_event),
        socket_path,
        request_timeout_seconds=5.0,
    )
    thread = threading.Thread(target=server.serve, args=(stop_event,), daemon=True)
    thread.start()
    try:
        # While the barrier is held the socket must stay absent and the
        # endpoint unreachable — poll for a stable window so a fast bind
        # that lingers briefly cannot hide behind scheduling luck.
        deadline = time.monotonic() + 1.5
        while time.monotonic() < deadline:
            assert not socket_path.exists(), (
                "socket advertised before the warmup barrier passed"
            )
            time.sleep(0.05)

        probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        probe.settimeout(0.2)
        with pytest.raises(OSError):
            probe.connect(str(socket_path))
        probe.close()

        # Barrier release: the socket appears, is a proper 0600 socket, and
        # a real protocol health round-trip succeeds.
        barrier.set()
        assert _wait_until(lambda: socket_path.exists(), 5.0), (
            "socket never appeared after the warmup barrier was released"
        )
        mode = socket_path.stat().st_mode
        assert stat.S_ISSOCK(mode), f"not a unix socket: {oct(mode)}"
        assert stat.S_IMODE(mode) == 0o600, f"socket mode {oct(stat.S_IMODE(mode))} != 0600"

        client = DataServiceClient(
            socket_path,
            worker_count=2,
            request_timeout_seconds=5.0,
        )
        assert client.identity.dataset_id == _DATASET_ID
        assert client.identity.worker_count == 2
    finally:
        stop_event.set()
        thread.join(timeout=5.0)
        assert not thread.is_alive(), "serve() did not exit after stop"
    assert not socket_path.exists(), "socket not unlinked on clean shutdown"


def test_stop_during_warmup_returns_without_socket(tmp_path: Path) -> None:
    """The 0U FAIL #2 teardown shape: stack stop while still warming up.

    The service must exit, and it must not leave a socket file behind that
    a later start could mistake for a live service.
    """
    barrier = threading.Event()  # never released
    stop_event = threading.Event()
    socket_path = tmp_path / "data-service.sock"
    server = DataServiceServer(
        _gated_service(barrier, stop_event),
        socket_path,
        request_timeout_seconds=5.0,
    )
    thread = threading.Thread(target=server.serve, args=(stop_event,), daemon=True)
    thread.start()
    try:
        time.sleep(0.3)  # serve() is now parked inside wait_until_ready
        assert thread.is_alive(), "serve() exited before reaching the barrier"
        assert not socket_path.exists()

        stop_event.set()
        thread.join(timeout=5.0)
        assert not thread.is_alive(), "serve() ignored stop during the warmup barrier"
    finally:
        stop_event.set()
        thread.join(timeout=5.0)
    assert not socket_path.exists(), "stop during warmup left a socket file behind"


def test_wait_until_ready_false_means_no_socket_created(tmp_path: Path) -> None:
    """wait_until_ready() == False (service stopping) must skip the socket entirely."""
    stop_event = threading.Event()
    stop_event.set()  # already stopped: the gated barrier returns False immediately
    barrier = threading.Event()
    service = _gated_service(barrier, stop_event)
    socket_path = tmp_path / "data-service.sock"
    server = DataServiceServer(service, socket_path, request_timeout_seconds=5.0)

    server.serve(stop_event)  # must return promptly, not block

    assert not socket_path.exists(), "socket created although wait_until_ready returned False"
