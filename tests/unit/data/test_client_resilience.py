from __future__ import annotations

# pyright: reportPrivateUsage=false
import socket
import threading
import time
from pathlib import Path
from typing import cast

import pytest

from sakuramoon.data.client import (
    DataServiceClient,
    DataServiceUnavailable,
)
from sakuramoon.data.service_protocol import (
    DataServiceSessionIdentity,
    canonical_json_bytes,
)

# AF_UNIX only exists on Unix; these tests exercise a real unix-socket server.
_AF_UNIX = getattr(socket, "AF_UNIX", None)
pytestmark = pytest.mark.skipif(
    _AF_UNIX is None, reason="AF_UNIX is not available on this platform"
)


def _client(path: Path, timeout_seconds: float = 1.0) -> DataServiceClient:
    client = object.__new__(DataServiceClient)
    client.socket_path = path
    client.worker_count = 1
    client.request_timeout_seconds = timeout_seconds
    client.identity = DataServiceSessionIdentity(dataset_id="dataset", worker_count=1)
    return client


class _UnixResponder:
    """AF_UNIX responder that scripts per-connection behaviors in order.

    Steps: "ok" reads one frame and replies, "refuse" closes without
    replying, "silence" holds the connection until the client times out.
    """

    def __init__(
        self, path: Path, steps: list[str], response: dict[str, object]
    ) -> None:
        self.path = path
        self.steps = list(steps)
        self.response = response
        self.accepted = 0
        self._listener = socket.socket(cast(int, _AF_UNIX), socket.SOCK_STREAM)
        self._listener.bind(str(path))
        self._listener.listen(16)
        self._listener.settimeout(0.2)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while True:
            try:
                connection, _ = self._listener.accept()
            except TimeoutError:
                continue
            except OSError:
                return
            step = self.steps[min(self.accepted, len(self.steps) - 1)]
            self.accepted += 1
            if step == "refuse":
                connection.close()
            elif step == "silence":
                time.sleep(1.0)
                connection.close()
            else:
                connection.recv(65536)
                connection.sendall(canonical_json_bytes(self.response) + b"\n")

    def stop(self) -> None:
        self._listener.close()


def test_transient_transport_failure_is_retried_until_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("sakuramoon.data.client._TRANSPORT_RETRY_BASE_SECONDS", 0.05)
    path = tmp_path / "service.sock"
    responder = _UnixResponder(
        path, ["refuse", "silence", "ok"], {"done": False, "lease": None, "ok": True}
    )
    try:
        client = _client(path, timeout_seconds=1.0)
        response = client._request(
            {"op": "lease", "session_id": client.identity.session_id, "worker_id": 0}
        )
        assert response["ok"] is True
        assert responder.accepted == 3
    finally:
        responder.stop()


def test_sustained_outage_fails_after_bounded_retries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("sakuramoon.data.client._TRANSPORT_RETRY_BASE_SECONDS", 0.05)
    client = _client(tmp_path / "missing.sock", timeout_seconds=0.2)
    started = time.monotonic()
    with pytest.raises(DataServiceUnavailable, match="data service is unavailable"):
        client._request({"op": "health"})
    assert time.monotonic() - started < 10.0


def test_protocol_rejection_is_not_retried(tmp_path: Path) -> None:
    path = tmp_path / "service.sock"
    responder = _UnixResponder(path, ["ok"], {"error": "lease is invalid", "ok": False})
    try:
        client = _client(path)
        with pytest.raises(
            DataServiceUnavailable, match="data service rejected the request"
        ):
            client._request(
                {
                    "op": "lease",
                    "session_id": client.identity.session_id,
                    "worker_id": 0,
                }
            )
        assert responder.accepted == 1
    finally:
        responder.stop()
