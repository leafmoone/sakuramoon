"""Asynchronous conversion of successful training updates into fixed metrics."""

from __future__ import annotations

import math
import queue
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Self

import torch

from sakuramoon.telemetry.metrics import (
    DROPOUT_KEYS,
    NOISE_T_BIN_COUNT,
    TIMING_PHASES,
    MetricsPublisher,
    TrainingMetric,
)

if TYPE_CHECKING:
    from sakuramoon.train.runtime import SuccessfulTrainingObservation


def _finite_float(name: str, value: float, *, positive: bool = False) -> None:
    if type(value) is not float or not math.isfinite(value):
        raise TypeError(f"{name} must be a finite float")
    if value < 0.0 or (positive and value == 0.0):
        boundary = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be {boundary}")


def _scalar_float(name: str, value: torch.Tensor) -> float:
    if value.numel() != 1:
        raise TypeError(f"{name} must be a scalar tensor")
    result = float(value.detach().to(device="cpu", dtype=torch.float32).item())
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _scalar_count(name: str, value: torch.Tensor) -> int:
    result = _scalar_float(name, value)
    integer = int(result)
    if result != integer or integer < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return integer


@dataclass(frozen=True, slots=True)
class UpdateMetricContext:
    """Explicit non-model facts not owned by the T050 observation contract."""

    dit_flops: int
    samples_per_second: float
    ready_queue_depth: int
    supplemental_phase_seconds: Mapping[str, float]

    def __post_init__(self) -> None:
        if type(self.dit_flops) is not int or self.dit_flops <= 0:
            raise TypeError("dit_flops must be a positive integer")
        _finite_float("samples_per_second", self.samples_per_second, positive=True)
        if type(self.ready_queue_depth) is not int or self.ready_queue_depth < 0:
            raise TypeError("ready_queue_depth must be a nonnegative integer")
        unknown = sorted(set(self.supplemental_phase_seconds) - TIMING_PHASES)
        if unknown:
            raise ValueError(f"unknown supplemental timing phases: {unknown}")
        for phase, seconds in self.supplemental_phase_seconds.items():
            _finite_float(f"supplemental_phase_seconds.{phase}", seconds)
        object.__setattr__(
            self,
            "supplemental_phase_seconds",
            MappingProxyType(dict(self.supplemental_phase_seconds)),
        )


def _noise_bucket(
    observation: SuccessfulTrainingObservation,
    *,
    loss_attribute: str,
    count_attribute: str,
) -> tuple[float, int]:
    loss_sum = 0.0
    sample_count = 0
    for index, measurement in enumerate(observation.microbatches):
        loss_sum += _scalar_float(
            f"microbatches[{index}].{loss_attribute}",
            getattr(measurement, loss_attribute),
        )
        sample_count += _scalar_count(
            f"microbatches[{index}].{count_attribute}",
            getattr(measurement, count_attribute),
        )
    if loss_sum < 0.0:
        raise ValueError(f"{loss_attribute} must be nonnegative")
    return (loss_sum / sample_count if sample_count else 0.0), sample_count


def _timestep_bin_stats(
    observation: SuccessfulTrainingObservation,
) -> tuple[tuple[float, ...], tuple[int, ...]]:
    """Aggregate the existing per-sample losses into fixed 0.05-wide t bins.

    This is telemetry-only: it consumes the detached loss vector already
    produced for the update and never runs another model operation.
    """

    loss_sums = torch.zeros(NOISE_T_BIN_COUNT, dtype=torch.float64)
    sample_counts = torch.zeros(NOISE_T_BIN_COUNT, dtype=torch.int64)
    for index, measurement in enumerate(observation.microbatches):
        losses = measurement.per_sample_loss.detach().to(
            device="cpu", dtype=torch.float64
        )
        timesteps = measurement.timesteps.detach().to(
            device="cpu", dtype=torch.float32
        )
        if (
            losses.ndim != 1
            or timesteps.ndim != 1
            or losses.numel() == 0
            or losses.numel() != timesteps.numel()
        ):
            raise ValueError(
                f"microbatches[{index}] loss/timestep vectors are inconsistent"
            )
        if not bool(torch.isfinite(losses).all().item()):
            raise ValueError(f"microbatches[{index}].per_sample_loss must be finite")
        if bool((losses < 0.0).any().item()):
            raise ValueError(f"microbatches[{index}].per_sample_loss must be nonnegative")
        if not bool(torch.isfinite(timesteps).all().item()):
            raise ValueError(f"microbatches[{index}].timesteps must be finite")
        if bool(((timesteps < 0.0) | (timesteps > 1.0)).any().item()):
            raise ValueError(f"microbatches[{index}].timesteps must be in [0,1]")

        bin_indices = torch.floor(timesteps * NOISE_T_BIN_COUNT).to(torch.int64)
        bin_indices.clamp_(min=0, max=NOISE_T_BIN_COUNT - 1)
        loss_sums.scatter_add_(0, bin_indices, losses)
        sample_counts += torch.bincount(
            bin_indices, minlength=NOISE_T_BIN_COUNT
        )

    if int(sample_counts.sum().item()) != observation.loop.update.effective_samples:
        raise ValueError("t-bin sample counts differ from effective batch")
    losses = tuple(
        float(loss_sums[index].item() / sample_counts[index].item())
        if int(sample_counts[index].item())
        else 0.0
        for index in range(NOISE_T_BIN_COUNT)
    )
    counts = tuple(int(value.item()) for value in sample_counts)
    return losses, counts


def _dropout_counts(
    observation: SuccessfulTrainingObservation,
) -> dict[str, int]:
    totals: dict[str, int] = {key: 0 for key in DROPOUT_KEYS}
    for index, measurement in enumerate(observation.microbatches):
        values = measurement.dropout_hits.as_mapping()
        if set(values) != set(DROPOUT_KEYS):
            raise ValueError(
                f"microbatches[{index}].dropout_hits has an invalid key set"
            )
        for key, value in values.items():
            if type(value) is not int or value < 0:
                raise TypeError(f"dropout hit {key} must be a nonnegative integer")
            totals[key] += value
    return totals


def _condition_route_counts(
    observation: SuccessfulTrainingObservation,
) -> dict[str, int]:
    totals = {"artist_text": 0, "character_text": 0, "null": 0}
    for index, measurement in enumerate(observation.microbatches):
        values = measurement.condition_routes.as_mapping()
        if set(values) != set(totals):
            raise ValueError(
                f"microbatches[{index}].condition_routes has an invalid key set"
            )
        for key, value in values.items():
            if type(value) is not int or value < 0:
                raise TypeError(
                    f"condition route {key} must be a nonnegative integer"
                )
            totals[key] += value
        if sum(values.values()) != measurement.per_sample_loss.numel():
            raise ValueError(
                f"microbatches[{index}].condition routes differ from samples"
            )
    if sum(totals.values()) != observation.loop.update.effective_samples:
        raise ValueError("condition route counts differ from effective batch")
    return totals


def _timestep_summary(
    observation: SuccessfulTrainingObservation,
) -> tuple[float, float, float, float]:
    values = tuple(
        measurement.timesteps.detach().to(device="cpu", dtype=torch.float32)
        for measurement in observation.microbatches
    )
    if not values or any(value.ndim != 1 or value.numel() == 0 for value in values):
        raise ValueError("each microbatch must contain a timestep vector")
    timesteps = torch.cat(values)
    if timesteps.numel() != observation.loop.update.effective_samples:
        raise ValueError("timestep count differs from effective batch")
    if not bool(torch.isfinite(timesteps).all().item()):
        raise ValueError("timesteps must be finite")
    return (
        float(timesteps.min().item()),
        float(timesteps.max().item()),
        float(timesteps.mean().item()),
        float(timesteps.std(unbiased=False).item()),
    )


def _phase_seconds(
    observation: SuccessfulTrainingObservation,
    context: UpdateMetricContext,
) -> dict[str, float]:
    timer = observation.phase_timer
    observed = timer.collect_ready()
    if timer.pending_cuda_pairs:
        raise RuntimeError("CUDA timing events are not ready")
    duplicates = sorted(set(context.supplemental_phase_seconds) & set(observed))
    if duplicates:
        raise ValueError(f"supplemental timing duplicates observed phases: {duplicates}")
    reserved = set(context.supplemental_phase_seconds) & {"data", "checkpoint"}
    if reserved:
        raise ValueError("data and checkpoint timings are owned by the T050 loop")

    phases: dict[str, float] = {phase: 0.0 for phase in TIMING_PHASES}
    phases.update(observed)
    phases.update(context.supplemental_phase_seconds)
    phases["data"] = observation.loop.data_wait_seconds
    phases["checkpoint"] = observation.loop.checkpoint_seconds
    return phases


def build_training_metric(
    observation: SuccessfulTrainingObservation,
    *,
    context: UpdateMetricContext,
    recorded_at_unix_ns: int,
) -> TrainingMetric:
    """Build one complete metric after all update CUDA events become queryable."""

    if type(recorded_at_unix_ns) is not int or recorded_at_unix_ns <= 0:
        raise TypeError("recorded_at_unix_ns must be a positive integer")
    update = observation.loop.update
    high_noise_loss, high_noise_count = _noise_bucket(
        observation,
        loss_attribute="high_noise_loss_sum",
        count_attribute="high_noise_sample_count",
    )
    low_noise_loss, low_noise_count = _noise_bucket(
        observation,
        loss_attribute="low_noise_loss_sum",
        count_attribute="low_noise_sample_count",
    )
    t_bin_losses, t_bin_sample_counts = _timestep_bin_stats(observation)
    timestep_min, timestep_max, timestep_mean, timestep_std = _timestep_summary(
        observation
    )
    coefficient = _scalar_float("clip.coefficient", update.clip.coefficient)
    if coefficient < 0.0 or coefficient > 1.0:
        raise ValueError("clip coefficient must be in [0,1]")

    return TrainingMetric(
        successful_update=update.state.successful_updates,
        recorded_at_unix_ns=recorded_at_unix_ns,
        total_loss=_scalar_float("mean_loss", update.mean_loss),
        high_noise_loss=high_noise_loss,
        low_noise_loss=low_noise_loss,
        high_noise_sample_count=high_noise_count,
        low_noise_sample_count=low_noise_count,
        t_bin_losses=t_bin_losses,
        t_bin_sample_counts=t_bin_sample_counts,
        pre_clip_grad_norm=_scalar_float(
            "clip.pre_clip_norm", update.clip.pre_clip_norm
        ),
        post_clip_grad_norm=_scalar_float(
            "clip.post_clip_norm", update.clip.post_clip_norm
        ),
        condition_encoder_grad_norm=_scalar_float(
            "condition_encoder_grad_norm", update.condition_encoder_grad_norm
        ),
        clip_fraction=float(coefficient < 1.0),
        learning_rate=observation.learning_rate,
        timestep_min=timestep_min,
        timestep_max=timestep_max,
        timestep_mean=timestep_mean,
        timestep_std=timestep_std,
        effective_batch=update.effective_samples,
        image_tokens=sum(item.image_tokens for item in observation.microbatches),
        text_tokens=sum(item.text_tokens for item in observation.microbatches),
        dit_flops=context.dit_flops,
        samples_per_second=context.samples_per_second,
        gpu_memory_allocated_bytes=observation.gpu_memory_allocated_bytes,
        gpu_memory_reserved_bytes=observation.gpu_memory_reserved_bytes,
        ready_queue_depth=context.ready_queue_depth,
        ready_queue_wait_seconds=observation.loop.data_wait_seconds,
        nonfinite_count=0,
        dropout_hits=_dropout_counts(observation),
        condition_routes=_condition_route_counts(observation),
        phase_seconds=_phase_seconds(observation, context),
    )


@dataclass(frozen=True, slots=True)
class _QueuedObservation:
    observation: SuccessfulTrainingObservation
    context: UpdateMetricContext
    recorded_at_unix_ns: int


class AsyncTrainingMetricObserver:
    """Bounded observer that waits by event query and never synchronizes CUDA."""

    _STOP = object()

    def __init__(
        self,
        publisher: MetricsPublisher,
        *,
        context_provider: Callable[
            [SuccessfulTrainingObservation], UpdateMetricContext
        ],
        queue_capacity: int,
        event_timeout_seconds: float,
        clock_ns: Callable[[], int] = time.time_ns,
    ) -> None:
        if not callable(context_provider):
            raise TypeError("context_provider must be callable")
        if type(queue_capacity) is not int or queue_capacity <= 0:
            raise TypeError("observer queue capacity must be a positive integer")
        _finite_float(
            "event_timeout_seconds", event_timeout_seconds, positive=True
        )
        if not callable(clock_ns):
            raise TypeError("clock_ns must be callable")
        self.publisher = publisher
        self.context_provider = context_provider
        self.event_timeout_seconds = event_timeout_seconds
        self.clock_ns = clock_ns
        self._queue: queue.Queue[_QueuedObservation | object] = queue.Queue(
            queue_capacity
        )
        self._state_lock = threading.Lock()
        self._background_error: BaseException | None = None
        self._closed = False
        self._last_submitted_update: int | None = None
        self._worker = threading.Thread(
            target=self._run,
            name="sakuramoon-training-metrics",
            daemon=True,
        )
        self._worker.start()

    def _set_background_error(self, error: BaseException) -> None:
        with self._state_lock:
            if self._background_error is None:
                self._background_error = error

    def _check_health_locked(self) -> None:
        if self._background_error is not None:
            raise RuntimeError("training metric observer failed") from self._background_error

    def _check_health(self) -> None:
        with self._state_lock:
            self._check_health_locked()

    def _await_metric(self, item: _QueuedObservation) -> TrainingMetric:
        deadline = time.monotonic() + self.event_timeout_seconds
        while True:
            item.observation.phase_timer.collect_ready()
            if not item.observation.phase_timer.pending_cuda_pairs:
                return build_training_metric(
                    item.observation,
                    context=item.context,
                    recorded_at_unix_ns=item.recorded_at_unix_ns,
                )
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                raise TimeoutError("CUDA timing events did not become queryable")
            time.sleep(min(0.001, remaining))

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is self._STOP:
                    return
                assert isinstance(item, _QueuedObservation)
                try:
                    self.publisher.publish(self._await_metric(item))
                except BaseException as error:  # noqa: BLE001
                    self._set_background_error(error)
            finally:
                self._queue.task_done()

    def submit(self, observation: SuccessfulTrainingObservation) -> None:
        with self._state_lock:
            if self._closed:
                raise RuntimeError("training metric observer is closed")
            self._check_health_locked()
        context = self.context_provider(observation)
        if type(context) is not UpdateMetricContext:
            raise TypeError("context_provider must return UpdateMetricContext")
        recorded_at_unix_ns = self.clock_ns()
        if type(recorded_at_unix_ns) is not int or recorded_at_unix_ns <= 0:
            raise TypeError("clock_ns must return a positive integer")
        queued = _QueuedObservation(observation, context, recorded_at_unix_ns)
        with self._state_lock:
            if self._closed:
                raise RuntimeError("training metric observer is closed")
            self._check_health_locked()
            successful_update = observation.loop.update.state.successful_updates
            if type(successful_update) is not int or successful_update <= 0:
                raise ValueError("observed successful update must be positive")
            if (
                self._last_submitted_update is not None
                and successful_update != self._last_submitted_update + 1
            ):
                raise ValueError("metric observations must be consecutive")
            try:
                self._queue.put_nowait(queued)
            except queue.Full as error:
                raise RuntimeError("training metric observer queue is full") from error
            self._last_submitted_update = successful_update

    def __call__(self, observation: SuccessfulTrainingObservation) -> None:
        self.submit(observation)

    def drain(self) -> None:
        self._queue.join()
        self._check_health()

    def close(self) -> None:
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
        self._queue.put(self._STOP)
        self._queue.join()
        self._worker.join()
        self._check_health()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        _error_type: type[BaseException] | None,
        error: BaseException | None,
        _traceback: object,
    ) -> None:
        try:
            self.close()
        except BaseException as close_error:
            if error is not None:
                raise BaseExceptionGroup(
                    "training and metric observer close both failed",
                    [error, close_error],
                ) from None
            raise


__all__ = [
    "AsyncTrainingMetricObserver",
    "UpdateMetricContext",
    "build_training_metric",
]
