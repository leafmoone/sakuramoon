from __future__ import annotations

# pyright: reportPrivateUsage=false
import errno
import socket
import threading
from concurrent.futures import Future
from pathlib import Path
from typing import cast

import pytest

from sakuramoon.data.cache import CachedShard
from sakuramoon.data.modelscope import DatasetTransientError, DatasetTransportError
from sakuramoon.data.service import (
    DataServiceError,
    DataServiceServer,
    DataSupplyService,
)


class _ResponseServer(DataServiceServer):
    def __init__(self) -> None:
        self.request_timeout_seconds = 1.0

    def _dispatch(self, request: dict[str, object]) -> dict[str, object]:
        return {"ok": True, "request": request}


class _FakeConnection:
    def __init__(self, send_error: OSError | None = None) -> None:
        self.send_error = send_error
        self.timeout: float | None = None
        self.sent = b""

    def settimeout(self, value: float) -> None:
        self.timeout = value

    def recv(self, size: int) -> bytes:
        del size
        return b'{"op":"health"}\n'

    def sendall(self, payload: bytes) -> None:
        if self.send_error is not None:
            raise self.send_error
        self.sent += payload


def _service_with_failed_future(error: Exception) -> DataSupplyService:
    service = object.__new__(DataSupplyService)
    future: Future[CachedShard] = Future()
    future.set_exception(error)
    service._futures = {"data/shard.tar": future}
    service._ready = {}
    service._shutdown = threading.Event()
    service._external_stop = None
    return service


def test_transient_download_failure_is_left_pending_for_retry() -> None:
    service = _service_with_failed_future(DatasetTransientError("transfer ended early"))

    service._collect_completed()

    assert service._futures == {}
    assert service._ready == {}


def test_permanent_download_failure_keeps_exact_reason() -> None:
    service = _service_with_failed_future(DatasetTransportError("shard is unavailable"))

    with pytest.raises(
        DataServiceError,
        match=r"data/shard\.tar: shard is unavailable",
    ):
        service._collect_completed()


@pytest.mark.parametrize(
    "error",
    [
        BrokenPipeError(errno.EPIPE, "client closed"),
        ConnectionResetError(errno.ECONNRESET, "client reset"),
        TimeoutError(errno.ETIMEDOUT, "client timed out"),
    ],
)
def test_client_disconnect_does_not_prevent_next_request(error: OSError) -> None:
    server = _ResponseServer()
    disconnected = _FakeConnection(error)
    healthy = _FakeConnection()

    server._handle_connection(cast(socket.socket, disconnected))
    server._handle_connection(cast(socket.socket, healthy))

    assert disconnected.timeout == 1.0
    assert healthy.timeout == 1.0
    assert b'"ok":true' in healthy.sent


def test_unexpected_socket_error_is_not_hidden() -> None:
    server = _ResponseServer()
    connection = _FakeConnection(OSError(errno.EIO, "unexpected I/O error"))

    with pytest.raises(OSError, match="unexpected I/O error"):
        server._handle_connection(cast(socket.socket, connection))


class _StubService:
    """Just enough surface for DataServiceServer.serve() without a real cache."""

    def __init__(self) -> None:
        from sakuramoon.data.service import DataServiceLimits

        self.limits = DataServiceLimits(
            download_concurrency=2,
            verified_shard_lookahead=2,
            lease_channel_capacity=4,
            ack_channel_capacity=4,
        )
        self.started = threading.Event()
        self.closed = threading.Event()

    def start(self, stop_event: threading.Event) -> None:
        del stop_event
        self.started.set()

    def wait_until_ready(self) -> bool:
        return True

    def close(self) -> None:
        self.closed.set()


class _BlockingLeaseServer(DataServiceServer):
    """Lease dispatch blocks (shard not ready); health stays responsive."""

    def __init__(self, socket_path: Path, stop: threading.Event) -> None:
        from sakuramoon.data.service import DataServiceLimits

        stub = _StubService()
        super().__init__(
            cast(DataSupplyService, stub),
            socket_path,
            request_timeout_seconds=5.0,
        )
        del stub
        self._stop = stop
        self.limits = DataServiceLimits(
            download_concurrency=2,
            verified_shard_lookahead=2,
            lease_channel_capacity=4,
            ack_channel_capacity=4,
        )

    def _dispatch(self, request: dict[str, object]) -> dict[str, object]:
        if request.get("op") == "lease":
            self._stop.wait(timeout=10.0)
            return {"done": True, "lease": None, "ok": True}
        return {"ok": True, "request": request}


def test_blocking_lease_does_not_starve_other_connections(
    tmp_path: Path,
) -> None:
    """A lease waiting for a ready shard must not block other clients.

    Regression: the accept loop used to handle connections inline, so one
    lease blocked on an empty ready queue during a degraded download window
    starved every other worker's ACK/lease until their timeouts expired.
    """
    import json as _json

    stop = threading.Event()
    server = _BlockingLeaseServer(tmp_path / "data-service.sock", stop)
    thread = threading.Thread(target=server.serve, args=(stop,), daemon=True)
    thread.start()
    for _ in range(100):
        if server.socket_path.exists():
            break
        threading.Event().wait(0.05)
    else:
        pytest.fail("service socket did not appear")

    import time

    lease_conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    lease_conn.connect(str(server.socket_path))
    lease_conn.sendall(
        (_json.dumps({"op": "lease", "session_id": "x", "worker_id": 0}) + "\n").encode()
    )
    time.sleep(0.5)  # let the server accept and enter the blocked lease

    health_conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    health_conn.settimeout(5.0)
    health_conn.connect(str(server.socket_path))
    health_conn.sendall((_json.dumps({"op": "health"}) + "\n").encode())
    response = b""
    while not response.endswith(b"\n"):
        block = health_conn.recv(4096)
        if not block:
            break
        response += block
    health_conn.close()
    lease_conn.close()
    stop.set()

    assert b'"ok":true' in response
