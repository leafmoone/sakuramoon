from __future__ import annotations

# pyright: reportPrivateUsage=false
import errno
import socket
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import cast
from unittest.mock import Mock

import pytest

from sakuramoon.data.cache import CachedShard, ShardCache, ShardCacheError
from sakuramoon.data.modelscope import DatasetTransientError, DatasetTransportError
from sakuramoon.data.service import (
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
    service._prepare_attempts = {}
    service._outstanding = {}
    service.cache = Mock(spec=ShardCache)
    service._executor = Mock(spec=ThreadPoolExecutor)
    service._shutdown = threading.Event()
    service._external_stop = None
    return service


@pytest.mark.parametrize(
    "error",
    [
        DatasetTransientError("transfer ended early"),
        DatasetTransportError("shard is unavailable"),
        OSError("disk read failed"),
        ShardCacheError("cache entry is invalid"),
    ],
)
def test_download_failure_is_retried_with_exact_reason(
    error: Exception, capsys: pytest.CaptureFixture[str]
) -> None:
    service = _service_with_failed_future(error)

    service._collect_completed()

    assert "data/shard.tar" in service._futures
    assert service._prepare_attempts == {"data/shard.tar": 1}
    assert service._ready == {}
    assert str(error) in capsys.readouterr().out


def test_exhausted_shard_is_skipped_using_legacy_completed_status(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    service = _service_with_failed_future(DatasetTransportError("shard is unavailable"))
    service._prepare_attempts["data/shard.tar"] = 3
    replace_status = Mock()
    monkeypatch.setattr(service, "_replace_status", replace_status)

    service._collect_completed()

    replace_status.assert_called_once_with("data/shard.tar", "completed")
    assert service._prepare_attempts["data/shard.tar"] == 4
    assert service._futures == {}
    assert "shard is unavailable" in capsys.readouterr().out


def test_unexpected_preparation_error_is_not_hidden() -> None:
    service = _service_with_failed_future(ValueError("invalid internal state"))

    with pytest.raises(ValueError, match="invalid internal state"):
        service._collect_completed()


def test_successful_retry_clears_attempt_count() -> None:
    service = _service_with_failed_future(DatasetTransientError("first failure"))
    service._prepare_attempts["data/shard.tar"] = 1
    ready: Future[CachedShard] = Future()
    ready.set_result(Mock(spec=CachedShard, fetched=Mock(cache_hit=True)))
    service._futures["data/shard.tar"] = ready

    service._collect_completed()

    assert service._prepare_attempts == {}
    assert "data/shard.tar" in service._ready


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
