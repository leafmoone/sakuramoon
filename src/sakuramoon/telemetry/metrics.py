"""Strict training metrics and durable local-first JSONL publication."""

from __future__ import annotations

import json
import math
import os
import stat
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Protocol, Self

from sakuramoon.data.caption import (
    CAPTION_DROPOUT_KEYS,
    CONDITION_ROUTE_KEYS,
)

CORE_TIMING_PHASES = (
    "data",
    "qwen",
    "vae",
    "conditioning",
    "dit_forward",
    "loss",
    "backward",
    "ddp",
    "clip",
    "optimizer",
    "checkpoint",
    "evaluation",
)
DETAILED_TIMING_PHASES = (
    "cache",
    "tar",
    "json",
    "caption",
    "tokenize",
    "decode",
    "exif",
    "crop",
    "bucket",
    "h2d",
    "condition",
    "zero_grad",
    "sample",
)
TIMING_PHASES = frozenset((*CORE_TIMING_PHASES, *DETAILED_TIMING_PHASES))
NOISE_T_BIN_COUNT = 20
NOISE_T_BIN_LABELS = tuple(
    f"bin_{index:02d}_t{index * 5:03d}_{(index + 1) * 5:03d}"
    for index in range(NOISE_T_BIN_COUNT)
)
TRAINING_METRIC_SCHEMA_VERSION = 7
DROPOUT_KEYS = CAPTION_DROPOUT_KEYS


def _finite_float(name: str, value: float, *, minimum: float | None = None) -> None:
    if type(value) is not float or not math.isfinite(value):
        raise TypeError(f"{name} must be a finite float")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} is below its minimum")


def _nonnegative_int(name: str, value: int, *, positive: bool = False) -> None:
    if type(value) is not int or value < (1 if positive else 0):
        boundary = "positive" if positive else "nonnegative"
        raise TypeError(f"{name} must be a {boundary} integer")


@dataclass(frozen=True, slots=True)
class TrainingMetric:
    """One complete successful-update metric record with no free-form fields."""

    successful_update: int
    recorded_at_unix_ns: int
    total_loss: float
    high_noise_loss: float
    low_noise_loss: float
    high_noise_sample_count: int
    low_noise_sample_count: int
    t_bin_losses: tuple[float, ...]
    t_bin_sample_counts: tuple[int, ...]
    pre_clip_grad_norm: float
    post_clip_grad_norm: float
    condition_encoder_grad_norm: float
    condition_global_projection_grad_norm: float
    clip_fraction: float
    learning_rate: float
    timestep_min: float
    timestep_max: float
    timestep_mean: float
    timestep_std: float
    effective_batch: int
    image_tokens: int
    text_tokens: int
    dit_flops: int
    samples_per_second: float
    gpu_memory_allocated_bytes: int
    gpu_memory_reserved_bytes: int
    ready_queue_depth: int
    ready_queue_wait_seconds: float
    nonfinite_count: int
    dropout_hits: Mapping[str, int]
    condition_routes: Mapping[str, int]
    phase_seconds: Mapping[str, float]

    def __post_init__(self) -> None:
        _nonnegative_int("successful_update", self.successful_update, positive=True)
        _nonnegative_int("recorded_at_unix_ns", self.recorded_at_unix_ns, positive=True)
        for name in ("total_loss", "high_noise_loss", "low_noise_loss"):
            _finite_float(name, getattr(self, name), minimum=0.0)
        for name in ("high_noise_sample_count", "low_noise_sample_count"):
            _nonnegative_int(name, getattr(self, name))
        if (
            type(self.t_bin_losses) is not tuple
            or type(self.t_bin_sample_counts) is not tuple
            or len(self.t_bin_losses) != NOISE_T_BIN_COUNT
            or len(self.t_bin_sample_counts) != NOISE_T_BIN_COUNT
        ):
            raise ValueError("t-bin metrics must contain exactly 20 bins")
        for index, value in enumerate(self.t_bin_losses):
            _finite_float(f"t_bin_losses[{index}]", value, minimum=0.0)
        for index, value in enumerate(self.t_bin_sample_counts):
            _nonnegative_int(f"t_bin_sample_counts[{index}]", value)
            if value == 0 and self.t_bin_losses[index] != 0.0:
                raise ValueError("empty t-bin loss must be zero")
        for name in (
            "pre_clip_grad_norm",
            "post_clip_grad_norm",
            "condition_encoder_grad_norm",
            "condition_global_projection_grad_norm",
            "learning_rate",
            "timestep_min",
            "timestep_max",
            "timestep_mean",
            "timestep_std",
            "samples_per_second",
            "ready_queue_wait_seconds",
        ):
            _finite_float(name, getattr(self, name), minimum=0.0)
        _finite_float("clip_fraction", self.clip_fraction, minimum=0.0)
        if self.clip_fraction > 1.0:
            raise ValueError("clip_fraction must not exceed one")
        if self.post_clip_grad_norm > self.pre_clip_grad_norm:
            raise ValueError("post-clip gradient norm must not exceed pre-clip norm")
        if not (
            self.timestep_min <= self.timestep_mean <= self.timestep_max <= 1.0
        ):
            raise ValueError("timestep summary is inconsistent")
        for name in (
            "effective_batch",
            "image_tokens",
            "text_tokens",
            "dit_flops",
            "gpu_memory_allocated_bytes",
            "gpu_memory_reserved_bytes",
            "ready_queue_depth",
            "nonfinite_count",
        ):
            _nonnegative_int(
                name,
                getattr(self, name),
                positive=name
                in {"effective_batch", "image_tokens", "text_tokens", "dit_flops"},
            )
        if (
            self.high_noise_sample_count + self.low_noise_sample_count
            != self.effective_batch
        ):
            raise ValueError("noise bucket sample counts must equal effective batch")
        if sum(self.t_bin_sample_counts) != self.effective_batch:
            raise ValueError("t-bin sample counts must equal effective batch")
        if self.high_noise_sample_count == 0 and self.high_noise_loss != 0.0:
            raise ValueError("empty high-noise bucket loss must be zero")
        if self.low_noise_sample_count == 0 and self.low_noise_loss != 0.0:
            raise ValueError("empty low-noise bucket loss must be zero")
        if self.gpu_memory_reserved_bytes < self.gpu_memory_allocated_bytes:
            raise ValueError("reserved GPU memory must cover allocated GPU memory")
        if set(self.dropout_hits) != set(DROPOUT_KEYS):
            raise ValueError("dropout_hits must contain every fixed dropout key")
        for key, value in self.dropout_hits.items():
            _nonnegative_int(f"dropout_hits.{key}", value)
            if value > self.effective_batch:
                raise ValueError("dropout hit count exceeds effective batch")
        if set(self.condition_routes) != set(CONDITION_ROUTE_KEYS):
            raise ValueError(
                "condition_routes must contain every fixed route key"
            )
        for key, value in self.condition_routes.items():
            _nonnegative_int(f"condition_routes.{key}", value)
        if sum(self.condition_routes.values()) != self.effective_batch:
            raise ValueError("condition route counts must equal effective batch")
        required_phases: set[str] = set(TIMING_PHASES)
        actual_phases = set(self.phase_seconds)
        if actual_phases != required_phases:
            missing = sorted(required_phases - actual_phases)
            unknown = sorted(actual_phases - required_phases)
            raise ValueError(
                "phase_seconds must contain every fixed timing phase; "
                f"missing={missing}, unknown={unknown}"
            )
        for phase, duration in self.phase_seconds.items():
            _finite_float(f"phase_seconds.{phase}", duration, minimum=0.0)
        object.__setattr__(
            self, "dropout_hits", MappingProxyType(dict(self.dropout_hits))
        )
        object.__setattr__(
            self,
            "condition_routes",
            MappingProxyType(dict(self.condition_routes)),
        )
        object.__setattr__(
            self, "phase_seconds", MappingProxyType(dict(self.phase_seconds))
        )

    def as_json_mapping(self) -> dict[str, object]:
        return {
            "schema_version": TRAINING_METRIC_SCHEMA_VERSION,
            "successful_update": self.successful_update,
            "recorded_at_unix_ns": self.recorded_at_unix_ns,
            "total_loss": self.total_loss,
            "high_noise_loss": self.high_noise_loss,
            "low_noise_loss": self.low_noise_loss,
            "high_noise_sample_count": self.high_noise_sample_count,
            "low_noise_sample_count": self.low_noise_sample_count,
            "train_loss_by_t": dict(
                zip(NOISE_T_BIN_LABELS, self.t_bin_losses, strict=True)
            ),
            "train_count_by_t": dict(
                zip(NOISE_T_BIN_LABELS, self.t_bin_sample_counts, strict=True)
            ),
            "pre_clip_grad_norm": self.pre_clip_grad_norm,
            "post_clip_grad_norm": self.post_clip_grad_norm,
            "condition_encoder_grad_norm": self.condition_encoder_grad_norm,
            "condition_global_projection_grad_norm": (
                self.condition_global_projection_grad_norm
            ),
            "clip_fraction": self.clip_fraction,
            "learning_rate": self.learning_rate,
            "timestep_min": self.timestep_min,
            "timestep_max": self.timestep_max,
            "timestep_mean": self.timestep_mean,
            "timestep_std": self.timestep_std,
            "effective_batch": self.effective_batch,
            "image_tokens": self.image_tokens,
            "text_tokens": self.text_tokens,
            "dit_flops": self.dit_flops,
            "samples_per_second": self.samples_per_second,
            "gpu_memory_allocated_bytes": self.gpu_memory_allocated_bytes,
            "gpu_memory_reserved_bytes": self.gpu_memory_reserved_bytes,
            "ready_queue_depth": self.ready_queue_depth,
            "ready_queue_wait_seconds": self.ready_queue_wait_seconds,
            "nonfinite_count": self.nonfinite_count,
            "dropout_hits": dict(self.dropout_hits),
            "condition_routes": dict(self.condition_routes),
            "phase_seconds": dict(self.phase_seconds),
        }

    def as_wandb_mapping(self) -> dict[str, int | float]:
        payload: dict[str, int | float] = {}
        for key, value in self.as_json_mapping().items():
            if type(value) is int or type(value) is float:
                payload[key] = value
        payload.update(
            {f"dropout_hits/{key}": value for key, value in self.dropout_hits.items()}
        )
        payload.update(
            {
                f"condition_routes/{key}": value
                for key, value in self.condition_routes.items()
            }
        )
        payload.update(
            {f"phase_seconds/{key}": value for key, value in self.phase_seconds.items()}
        )
        # Do not publish an empty bin as a numeric zero.  A zero would be
        # interpreted by W&B as an observed loss and would pull sparse
        # high-timestep curves down artificially.  Keep the count metric
        # below so an empty bin remains visible and auditable.
        payload.update(
            {
                f"train_loss_by_t/{label}": loss
                for label, loss, count in zip(
                    NOISE_T_BIN_LABELS,
                    self.t_bin_losses,
                    self.t_bin_sample_counts,
                    strict=True,
                )
                if count > 0
            }
        )
        payload.update(
            {
                f"train_count_by_t/{label}": value
                for label, value in zip(
                    NOISE_T_BIN_LABELS, self.t_bin_sample_counts, strict=True
                )
            }
        )
        return payload


class DurableJsonlSink:
    """Thread-safe append-only JSONL with explicit fsync cadence."""

    def __init__(self, path: Path, *, fsync_every_records: int) -> None:
        _nonnegative_int("fsync_every_records", fsync_every_records, positive=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_symlink():
            raise ValueError("JSONL destination may not be a symlink")
        existed = path.exists()
        flags = (
            os.O_APPEND | os.O_CREAT | os.O_WRONLY | os.O_CLOEXEC | os.O_NOFOLLOW
        )
        self._fd = os.open(path, flags, 0o600)
        metadata = os.fstat(self._fd)
        if not stat.S_ISREG(metadata.st_mode):
            os.close(self._fd)
            raise ValueError("JSONL destination must be a regular file")
        if stat.S_IMODE(metadata.st_mode) != 0o600:
            os.close(self._fd)
            raise PermissionError("JSONL destination must use mode 0600")
        if not existed:
            directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        self.path = path
        self.fsync_every_records = fsync_every_records
        self._pending = 0
        self._closed = False
        self._lock = threading.Lock()

    def write(self, payload: Mapping[str, object]) -> None:
        line = (
            json.dumps(
                dict(payload),
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode()
        with self._lock:
            if self._closed:
                raise RuntimeError("JSONL sink is closed")
            remaining = memoryview(line)
            while remaining:
                written = os.write(self._fd, remaining)
                if written <= 0:
                    raise OSError("JSONL append made no progress")
                remaining = remaining[written:]
            self._pending += 1
            if self._pending >= self.fsync_every_records:
                os.fsync(self._fd)
                self._pending = 0

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            os.fsync(self._fd)
            os.close(self._fd)
            self._closed = True
            self._pending = 0

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class AsyncMetricSink(Protocol):
    def submit(self, metric: TrainingMetric) -> None: ...


class MetricsPublisher:
    """Persist locally before exposing a record to the asynchronous sink."""

    def __init__(self, local: DurableJsonlSink, remote: AsyncMetricSink) -> None:
        self.local = local
        self.remote = remote

    def publish(self, metric: TrainingMetric) -> None:
        self.local.write(metric.as_json_mapping())
        self.remote.submit(metric)


__all__ = [
    "CORE_TIMING_PHASES",
    "DETAILED_TIMING_PHASES",
    "DROPOUT_KEYS",
    "NOISE_T_BIN_COUNT",
    "NOISE_T_BIN_LABELS",
    "TIMING_PHASES",
    "TRAINING_METRIC_SCHEMA_VERSION",
    "DurableJsonlSink",
    "MetricsPublisher",
    "TrainingMetric",
]
