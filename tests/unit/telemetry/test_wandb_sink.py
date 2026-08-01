from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import pytest
from wandb.errors import AuthenticationError

from sakuramoon.telemetry.metrics import DROPOUT_KEYS, TIMING_PHASES, TrainingMetric
from sakuramoon.telemetry.wandb_sink import AsyncWandbSink, replay_retry_queue


def _metric() -> TrainingMetric:
    return TrainingMetric(
        successful_update=7,
        recorded_at_unix_ns=10,
        total_loss=1.0,
        high_noise_loss=1.2,
        low_noise_loss=0.8,
        high_noise_sample_count=3,
        low_noise_sample_count=1,
        pre_clip_grad_norm=2.0,
        post_clip_grad_norm=1.0,
        clip_fraction=0.5,
        learning_rate=0.00002,
        timestep_min=0.1,
        timestep_max=0.9,
        timestep_mean=0.5,
        timestep_std=0.2,
        effective_batch=4,
        image_tokens=1024,
        text_tokens=128,
        dit_flops=1000,
        samples_per_second=12.5,
        gpu_memory_allocated_bytes=1024,
        gpu_memory_reserved_bytes=2048,
        ready_queue_depth=2,
        ready_queue_wait_seconds=0.01,
        nonfinite_count=0,
        dropout_hits=dict.fromkeys(DROPOUT_KEYS, 0),
        phase_seconds={**dict.fromkeys(TIMING_PHASES, 0.0), "data": 0.1},
    )


class _FailingRun:
    def log(self, data: Mapping[str, int | float], *, step: int) -> None:
        raise ConnectionError("secret-shaped network diagnostic")


class _ExceptionalRun:
    def __init__(self, error: Exception) -> None:
        self.error = error

    def log(self, data: Mapping[str, int | float], *, step: int) -> None:
        del data, step
        raise self.error


class _RecordingRun:
    def __init__(self) -> None:
        self.records: list[tuple[int, dict[str, int | float]]] = []

    def log(self, data: Mapping[str, int | float], *, step: int) -> None:
        self.records.append((step, dict(data)))


def test_network_failure_enters_durable_redacted_queue_and_replays(
    tmp_path: Path,
) -> None:
    retry = tmp_path / "wandb-retry.jsonl"
    sink = AsyncWandbSink(_FailingRun(), retry_path=retry, queue_capacity=2)

    sink.submit(_metric())
    sink.close()

    text = retry.read_text()
    assert "secret-shaped" not in text
    payload = json.loads(text)
    assert payload["error_type"] == "ConnectionError"
    assert payload["successful_update"] == 7

    recovered = _RecordingRun()
    assert replay_retry_queue(recovered, retry) == 1
    assert recovered.records[0][0] == 7
    assert recovered.records[0][1]["total_loss"] == 1.0
    assert not retry.exists()


@pytest.mark.parametrize(
    "error",
    [
        pytest.param(AuthenticationError("invalid credentials"), id="authentication"),
        pytest.param(ValueError("remote protocol drift"), id="non_communication"),
    ],
)
def test_nonretryable_remote_failure_surfaces_without_spill(
    tmp_path: Path, error: Exception
) -> None:
    retry = tmp_path / "wandb-retry.jsonl"
    sink = AsyncWandbSink(
        _ExceptionalRun(error), retry_path=retry, queue_capacity=1
    )
    sink.submit(_metric())

    with pytest.raises(RuntimeError, match="remote sink failed") as captured:
        sink.close()

    assert captured.value.__cause__ is error
    assert retry.read_bytes() == b""


def test_failed_replay_retains_complete_queue(tmp_path: Path) -> None:
    retry = tmp_path / "wandb-retry.jsonl"
    sink = AsyncWandbSink(_FailingRun(), retry_path=retry, queue_capacity=1)
    sink.submit(_metric())
    sink.close()
    original = retry.read_bytes()

    with pytest.raises(ConnectionError):
        replay_retry_queue(_FailingRun(), retry)

    assert retry.read_bytes() == original


def test_retry_reader_rejects_free_form_or_nonnumeric_payload(tmp_path: Path) -> None:
    retry = tmp_path / "wandb-retry.jsonl"
    retry.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "error_type": "ConnectionError",
                "successful_update": 1,
                "metrics": {"token": "must-not-pass"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    retry.chmod(0o600)

    with pytest.raises(ValueError, match="metric field"):
        replay_retry_queue(_RecordingRun(), retry)


def test_retry_reader_rejects_nonprivate_file_before_parsing(tmp_path: Path) -> None:
    retry = tmp_path / "wandb-retry.jsonl"
    retry.write_text("not-json\n", encoding="utf-8")
    retry.chmod(0o644)

    with pytest.raises(PermissionError, match="mode 0600"):
        replay_retry_queue(_RecordingRun(), retry)


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("total_loss", float("nan"), "metric field"),
        ("successful_update", 8.0, "metric identity"),
    ],
)
def test_retry_reader_rejects_nonfinite_or_mismatched_numeric_identity(
    tmp_path: Path,
    field: str,
    value: float,
    expected: str,
) -> None:
    retry = tmp_path / "wandb-retry.jsonl"
    sink = AsyncWandbSink(_FailingRun(), retry_path=retry, queue_capacity=1)
    sink.submit(_metric())
    sink.close()
    payload = json.loads(retry.read_text())
    payload["metrics"][field] = value
    retry.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match=expected):
        replay_retry_queue(_RecordingRun(), retry)
