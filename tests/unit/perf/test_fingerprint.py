"""Unit tests for the P0 runtime fingerprint."""

from __future__ import annotations

import platform
import socket
from dataclasses import FrozenInstanceError

import pytest

from sakuramoon.perf.fingerprint import (
    SCHEMA_VERSION,
    capture_runtime_fingerprint,
)


def test_capture_records_machine_identity() -> None:
    fingerprint = capture_runtime_fingerprint(
        git_sha="a" * 40,
        distributed_backend="native",
        world_size=1,
    )
    assert fingerprint.schema_version == SCHEMA_VERSION
    assert fingerprint.git_sha == "a" * 40
    assert fingerprint.hostname == socket.gethostname()
    assert fingerprint.python_version == platform.python_version()
    assert fingerprint.torch_version  # nonempty
    assert fingerprint.dtk_identity  # nonempty (file or env discovered)
    assert fingerprint.distributed_backend == "native"
    assert fingerprint.world_size == 1
    assert type(fingerprint.device_count) is int
    assert fingerprint.device_count >= 0
    if fingerprint.device_count > 0:
        assert fingerprint.device_total_memory_bytes is not None
        assert len(fingerprint.device_total_memory_bytes) == fingerprint.device_count
        assert all(value > 0 for value in fingerprint.device_total_memory_bytes)
        assert fingerprint.device_name and fingerprint.device_name != "unavailable"
    else:
        assert fingerprint.device_total_memory_bytes is None
        assert fingerprint.device_name == "unavailable"


def test_unavailable_fields_are_never_fabricated() -> None:
    fingerprint = capture_runtime_fingerprint()
    assert fingerprint.git_sha == "unavailable"
    assert fingerprint.distributed_backend is None
    assert fingerprint.world_size is None
    payload = fingerprint.to_dict()
    assert payload["git_sha"] == "unavailable"
    assert payload["world_size"] is None


def test_git_sha_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="git_sha"):
        capture_runtime_fingerprint(git_sha="")
    with pytest.raises(ValueError, match="git_sha"):
        capture_runtime_fingerprint(git_sha=123)  # type: ignore[arg-type]


def test_fingerprint_is_frozen() -> None:
    fingerprint = capture_runtime_fingerprint()
    with pytest.raises(FrozenInstanceError):
        fingerprint.hostname = "forged"  # type: ignore[misc]


def test_write_json_round_trips(tmp_path) -> None:
    fingerprint = capture_runtime_fingerprint(git_sha="b" * 40)
    destination = tmp_path / "nested" / "fingerprint.json"
    written = fingerprint.write_json(destination)
    assert written == destination
    assert destination.is_file()
    text = destination.read_text(encoding="utf-8")
    assert '"git_sha": "bbb' in text
