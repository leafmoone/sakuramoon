"""Asynchronous W&B submission with a durable local retry queue."""

from __future__ import annotations

import json
import math
import os
import queue
import stat
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Protocol, Self, cast

from sakuramoon.telemetry.metrics import DurableJsonlSink, TrainingMetric


class RemoteRun(Protocol):
    def log(self, data: Mapping[str, int | float], *, step: int) -> None: ...


def is_retryable_remote_communication_error(error: BaseException) -> bool:
    """Classify only communication outages as durable-retry candidates."""

    try:
        from wandb.errors import AuthenticationError, CommError
    except ImportError:
        return isinstance(error, ConnectionError)
    if isinstance(error, AuthenticationError):
        return False
    return isinstance(error, (ConnectionError, CommError))


def _retry_payload(metric: TrainingMetric, error: Exception) -> dict[str, object]:
    return {
        "schema_version": 1,
        "error_type": type(error).__name__,
        "successful_update": metric.successful_update,
        "metrics": metric.as_wandb_mapping(),
    }


def _validate_retry_payload(payload: object) -> tuple[int, dict[str, int | float]]:
    if not isinstance(payload, dict):
        raise TypeError("retry record schema is invalid")
    mapping = cast(dict[object, object], payload)
    if set(mapping) != {
        "schema_version",
        "error_type",
        "successful_update",
        "metrics",
    }:
        raise ValueError("retry record schema is invalid")
    successful_update = mapping["successful_update"]
    if (
        mapping["schema_version"] != 1
        or type(successful_update) is not int
        or successful_update <= 0
    ):
        raise ValueError("retry record identity is invalid")
    error_type = mapping["error_type"]
    if not isinstance(error_type, str) or not error_type.isidentifier():
        raise ValueError("retry error type is invalid")
    metrics_value = mapping["metrics"]
    if not isinstance(metrics_value, dict):
        raise TypeError("retry metrics are invalid")
    metrics = cast(dict[object, object], metrics_value)
    numeric: dict[str, int | float] = {}
    for key, value in metrics.items():
        if (
            not isinstance(key, str)
            or type(value) not in {int, float}
            or (type(value) is float and not math.isfinite(value))
        ):
            raise ValueError("retry metric field is invalid")
        numeric[key] = cast(int | float, value)
    if numeric.get("successful_update") != successful_update:
        raise ValueError("retry metric identity is invalid")
    return successful_update, numeric


def replay_retry_queue(run: RemoteRun, path: Path) -> int:
    """Replay a complete queue; retain it unchanged if any upload fails."""

    if path.is_symlink():
        raise ValueError("retry queue must be a regular file")
    if not path.exists():
        return 0
    metadata = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError("retry queue must be a regular file")
    if stat.S_IMODE(metadata.st_mode) != 0o600:
        raise PermissionError("retry queue must use mode 0600")
    records = [
        _validate_retry_payload(cast(object, json.loads(line)))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    for step, metrics in records:
        run.log(metrics, step=step)
    path.unlink()
    directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return len(records)


class AsyncWandbSink:
    """Bounded worker queue that spills network failures to durable JSONL."""

    _STOP = object()

    def __init__(
        self,
        run: RemoteRun,
        *,
        retry_path: Path,
        queue_capacity: int,
    ) -> None:
        if type(queue_capacity) is not int or queue_capacity <= 0:
            raise ValueError("W&B queue capacity must be a positive integer")
        self.run = run
        self.retry = DurableJsonlSink(retry_path, fsync_every_records=1)
        self._queue: queue.Queue[TrainingMetric | object] = queue.Queue(queue_capacity)
        self._background_error: Exception | None = None
        self._closed = False
        self._worker = threading.Thread(
            target=self._run,
            name="sakuramoon-wandb",
            daemon=True,
        )
        self._worker.start()

    def _set_background_error(self, error: Exception) -> None:
        if self._background_error is None:
            self._background_error = error

    def _spill(self, metric: TrainingMetric, error: Exception) -> None:
        try:
            self.retry.write(_retry_payload(metric, error))
        except Exception as exc:  # noqa: BLE001 - background durability boundary
            self._set_background_error(exc)

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is self._STOP:
                    return
                metric = cast(TrainingMetric, item)
                try:
                    self.run.log(
                        metric.as_wandb_mapping(),
                        step=metric.successful_update,
                    )
                except Exception as exc:  # noqa: BLE001 - network boundary
                    if is_retryable_remote_communication_error(exc):
                        self._spill(metric, exc)
                    else:
                        self._set_background_error(exc)
            finally:
                self._queue.task_done()

    def _check_health(self) -> None:
        if self._background_error is not None:
            raise RuntimeError("W&B remote sink failed") from self._background_error

    def submit(self, metric: TrainingMetric) -> None:
        if self._closed:
            raise RuntimeError("W&B sink is closed")
        self._check_health()
        try:
            self._queue.put_nowait(metric)
        except queue.Full:
            self._spill(metric, RuntimeError("remote queue full"))
            self._check_health()

    def drain(self) -> None:
        self._queue.join()
        self._check_health()

    def close(self) -> None:
        if self._closed:
            return
        self._queue.put(self._STOP)
        self._queue.join()
        self._worker.join()
        self.retry.close()
        self._closed = True
        self._check_health()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


__all__ = [
    "AsyncWandbSink",
    "RemoteRun",
    "is_retryable_remote_communication_error",
    "replay_retry_queue",
]
