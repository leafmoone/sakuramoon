"""Strict local IPC identities for the process-isolated data service."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from sakuramoon.data.manifest import ShardRecord

SERVICE_PROTOCOL_VERSION = 3
MAX_SERVICE_FRAME_BYTES = 16 * 1024 * 1024
_HEX64 = re.compile(r"[0-9a-f]{64}")


class DataServiceProtocolError(RuntimeError):
    """An IPC document or service identity is invalid."""


def canonical_json_bytes(payload: dict[str, object]) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


def _sha256(value: str, name: str) -> None:
    if _HEX64.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")


@dataclass(frozen=True, slots=True)
class DataServiceSessionIdentity:
    manifest_id: str
    worker_count: int

    def __post_init__(self) -> None:
        _sha256(self.manifest_id, "manifest_id")
        if type(self.worker_count) is not int or self.worker_count <= 0:
            raise ValueError("data service session identity is invalid")

    def as_dict(self) -> dict[str, object]:
        return {
            "manifest_id": self.manifest_id,
            "worker_count": self.worker_count,
        }

    @property
    def sha256(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.as_dict())).hexdigest()

    @classmethod
    def from_dict(cls, value: object) -> DataServiceSessionIdentity:
        document = _mapping(value, "session identity")
        _exact_keys(
            document,
            {"manifest_id", "worker_count"},
            "session identity",
        )
        try:
            return cls(
                manifest_id=cast(str, document["manifest_id"]),
                worker_count=cast(int, document["worker_count"]),
            )
        except (TypeError, ValueError):
            raise DataServiceProtocolError("session identity is invalid") from None


@dataclass(frozen=True, slots=True)
class ShardLeaseDescriptor:
    lease_id: str
    worker_id: int
    cycle_index: int
    state_identity: str
    record: ShardRecord
    local_path: Path

    def __post_init__(self) -> None:
        if (
            _HEX64.fullmatch(self.lease_id) is None
            or type(self.worker_id) is not int
            or self.worker_id < 0
            or type(self.cycle_index) is not int
            or self.cycle_index < 0
            or _HEX64.fullmatch(self.state_identity) is None
            or not self.local_path.is_absolute()
        ):
            raise ValueError("shard lease descriptor is invalid")

    def as_dict(self) -> dict[str, object]:
        return {
            "cycle_index": self.cycle_index,
            "lease_id": self.lease_id,
            "local_path": str(self.local_path),
            "record": {
                "bytes": self.record.bytes,
                "path": self.record.path,
                "upstream_sha256": self.record.upstream_sha256,
            },
            "state_identity": self.state_identity,
            "worker_id": self.worker_id,
        }

    @classmethod
    def from_dict(cls, value: object) -> ShardLeaseDescriptor:
        document = _mapping(value, "lease")
        _exact_keys(
            document,
            {
                "cycle_index",
                "lease_id",
                "local_path",
                "record",
                "state_identity",
                "worker_id",
            },
            "lease",
        )
        record = _mapping(document["record"], "lease record")
        _exact_keys(
            record,
            {"bytes", "path", "upstream_sha256"},
            "lease record",
        )
        try:
            return cls(
                lease_id=cast(str, document["lease_id"]),
                worker_id=cast(int, document["worker_id"]),
                cycle_index=cast(int, document["cycle_index"]),
                state_identity=cast(str, document["state_identity"]),
                local_path=Path(cast(str, document["local_path"])),
                record=ShardRecord(
                    path=cast(str, record["path"]),
                    bytes=cast(int, record["bytes"]),
                    upstream_sha256=cast(str, record["upstream_sha256"]),
                ),
            )
        except (TypeError, ValueError):
            raise DataServiceProtocolError("lease is invalid") from None


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise DataServiceProtocolError(f"{name} must be an object")
    return cast(dict[str, Any], value)


def _exact_keys(document: dict[str, Any], expected: set[str], name: str) -> None:
    if set(document) != expected:
        raise DataServiceProtocolError(f"{name} has unknown or missing fields")


def parse_frame(payload: bytes) -> dict[str, Any]:
    if (
        not payload
        or len(payload) > MAX_SERVICE_FRAME_BYTES
        or not payload.endswith(b"\n")
    ):
        raise DataServiceProtocolError("service frame size or terminator is invalid")
    try:
        document = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise DataServiceProtocolError("service frame is invalid JSON") from None
    return _mapping(document, "service frame")


__all__ = [
    "MAX_SERVICE_FRAME_BYTES",
    "SERVICE_PROTOCOL_VERSION",
    "DataServiceProtocolError",
    "DataServiceSessionIdentity",
    "ShardLeaseDescriptor",
    "canonical_json_bytes",
    "parse_frame",
]
