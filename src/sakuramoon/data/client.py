"""Lightweight trainer-side client for an already-running data service."""

from __future__ import annotations

import socket
from pathlib import Path
from typing import Any

from sakuramoon.data.service_protocol import (
    MAX_SERVICE_FRAME_BYTES,
    SERVICE_PROTOCOL_VERSION,
    DataServiceProtocolError,
    DataServiceSessionIdentity,
    ShardLeaseDescriptor,
    canonical_json_bytes,
    parse_frame,
)


class DataServiceUnavailable(RuntimeError):
    """The configured service is absent or rejects the exact session identity."""


def _recv(connection: socket.socket) -> dict[str, Any]:
    body = bytearray()
    while len(body) <= MAX_SERVICE_FRAME_BYTES:
        block = connection.recv(min(4096, MAX_SERVICE_FRAME_BYTES + 1 - len(body)))
        if not block:
            break
        body.extend(block)
        if body.endswith(b"\n"):
            break
    try:
        return parse_frame(bytes(body))
    except DataServiceProtocolError:
        raise DataServiceUnavailable("data service returned an invalid frame") from None


class DataServiceClient:
    """Transmit bounded lease/ACK messages without owning cache or state."""

    def __init__(
        self,
        socket_path: Path,
        identity: DataServiceSessionIdentity,
        *,
        request_timeout_seconds: float,
    ) -> None:
        if (
            not socket_path.is_absolute()
            or type(request_timeout_seconds) is not float
            or request_timeout_seconds <= 0.0
        ):
            raise ValueError("data service client settings are invalid")
        self.socket_path = socket_path
        self.identity = identity
        self.request_timeout_seconds = request_timeout_seconds
        self.health()

    def _request(self, payload: dict[str, object]) -> dict[str, Any]:
        frame = canonical_json_bytes(payload)
        if len(frame) > MAX_SERVICE_FRAME_BYTES:
            raise DataServiceUnavailable("data service request exceeds the frame bound")
        connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        connection.settimeout(self.request_timeout_seconds)
        try:
            connection.connect(str(self.socket_path))
            connection.sendall(frame)
            response = _recv(connection)
        except (OSError, TimeoutError):
            raise DataServiceUnavailable("data service is unavailable") from None
        finally:
            connection.close()
        if response.get("ok") is not True:
            raise DataServiceUnavailable("data service rejected the request")
        return response

    def health(self) -> bool:
        response = self._request(
            {
                "op": "health",
                "protocol_version": SERVICE_PROTOCOL_VERSION,
                "session_identity": self.identity.as_dict(),
            }
        )
        if set(response) != {
            "done",
            "ok",
            "protocol_version",
            "session_sha256",
        } or (
            response["protocol_version"] != SERVICE_PROTOCOL_VERSION
            or response["session_sha256"] != self.identity.sha256
            or type(response["done"]) is not bool
        ):
            raise DataServiceUnavailable("data service health identity is invalid")
        return response["done"]

    def lease(self, worker_id: int) -> ShardLeaseDescriptor | None:
        response = self._request({"op": "lease", "worker_id": worker_id})
        if (
            set(response) != {"done", "lease", "ok"}
            or type(response["done"]) is not bool
        ):
            raise DataServiceUnavailable("data service lease response is invalid")
        if response["done"]:
            if response["lease"] is not None:
                raise DataServiceUnavailable("completed service returned a lease")
            return None
        try:
            descriptor = ShardLeaseDescriptor.from_dict(response["lease"])
        except DataServiceProtocolError:
            raise DataServiceUnavailable("data service lease is invalid") from None
        if descriptor.worker_id != worker_id or not descriptor.local_path.is_file():
            raise DataServiceUnavailable("data service lease identity is invalid")
        return descriptor

    def acknowledge(self, descriptor: ShardLeaseDescriptor) -> None:
        response = self._request(
            {
                "lease_id": descriptor.lease_id,
                "op": "ack",
                "state_identity": descriptor.state_identity,
                "worker_id": descriptor.worker_id,
            }
        )
        if set(response) != {"ok"}:
            raise DataServiceUnavailable("data service ACK response is invalid")

__all__ = ["DataServiceClient", "DataServiceUnavailable"]
