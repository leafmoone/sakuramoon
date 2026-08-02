from __future__ import annotations

import json
import threading
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import NoReturn

import pytest
import torch

from sakuramoon.data.caption import CaptionDropoutCounts
from sakuramoon.optim.clip import ClipResult
from sakuramoon.telemetry.metrics import (
    TIMING_PHASES,
    DurableJsonlSink,
    MetricsPublisher,
    TrainingMetric,
)
from sakuramoon.telemetry.observer import (
    AsyncTrainingMetricObserver,
    UpdateMetricContext,
    build_training_metric,
)
from sakuramoon.telemetry.timers import PhaseTimer
from sakuramoon.train.loop import SuccessfulLoopObservation
from sakuramoon.train.runtime import RuntimeMeasurement, SuccessfulTrainingObservation
from sakuramoon.train.step import SingleGpuUpdateResult, SingleGpuUpdateState


def _observation(
    successful_update: int = 7,
) -> SuccessfulTrainingObservation:
    timer = PhaseTimer(device=torch.device("cpu"))
    with timer.record("qwen"):
        pass
    with timer.record("loss"):
        pass
    update = SingleGpuUpdateResult(
        mean_loss=torch.tensor(2.0),
        clip=ClipResult(
            pre_clip_norm=torch.tensor(2.0),
            post_clip_norm=torch.tensor(1.0),
            coefficient=torch.tensor(0.5),
        ),
        microbatches=1,
        effective_samples=2,
        state=SingleGpuUpdateState(
            successful_update,
            successful_update,
            successful_update * 2,
        ),
    )
    measurement = RuntimeMeasurement(
        per_sample_loss=torch.tensor([1.0, 3.0]),
        image_tokens=32,
        text_tokens=12,
        sample_ids=("sample-a", "sample-b"),
        shape_keys=("256x256x98", "256x256x98"),
        high_noise_loss_sum=torch.tensor(1.0),
        high_noise_sample_count=torch.tensor(1),
        low_noise_loss_sum=torch.tensor(3.0),
        low_noise_sample_count=torch.tensor(1),
        timesteps=torch.tensor([0.25, 0.99], dtype=torch.float32),
        dropout_hits=CaptionDropoutCounts(
            all_condition=0,
            nsfw=0,
            character=0,
            copyright=0,
            general=1,
            artist=0,
            candidate_source=0,
            long_names=0,
            long_no_names=0,
            short_vibes=0,
            nl2=0,
            nl3=0,
        ),
    )
    return SuccessfulTrainingObservation(
        loop=SuccessfulLoopObservation(
            update=update,
            checkpoint_reason=None,
            data_wait_seconds=0.2,
            checkpoint_seconds=0.3,
            update_wall_seconds=0.5,
            phase_timer=timer,
        ),
        microbatches=(measurement,),
        phase_timer=timer,
        learning_rate=0.00002,
        gpu_memory_allocated_bytes=1024,
        gpu_memory_reserved_bytes=2048,
    )


def _context(**changes: object) -> UpdateMetricContext:
    values: dict[str, object] = {
        "dit_flops": 10_000,
        "samples_per_second": 4.0,
        "ready_queue_depth": 1,
        "supplemental_phase_seconds": {"cache": 0.01},
    }
    values.update(changes)
    return UpdateMetricContext(**values)  # pyright: ignore[reportArgumentType]


def test_build_training_metric_aggregates_exact_t050_observation() -> None:
    metric = build_training_metric(
        _observation(), context=_context(), recorded_at_unix_ns=123
    )

    assert metric.successful_update == 7
    assert metric.total_loss == 2.0
    assert metric.high_noise_loss == 1.0
    assert metric.low_noise_loss == 3.0
    assert metric.high_noise_sample_count == 1
    assert metric.low_noise_sample_count == 1
    assert metric.clip_fraction == 1.0
    assert metric.dropout_hits["general"] == 1
    assert metric.timestep_min == pytest.approx(0.25)
    assert metric.timestep_max == pytest.approx(0.99)
    assert metric.effective_batch == 2
    assert metric.image_tokens == 32
    assert metric.text_tokens == 12
    assert metric.ready_queue_wait_seconds == 0.2
    assert set(metric.phase_seconds) == TIMING_PHASES
    assert metric.phase_seconds["data"] == 0.2
    assert metric.phase_seconds["checkpoint"] == 0.3
    assert metric.phase_seconds["cache"] == 0.01
    assert metric.phase_seconds["qwen"] >= 0.0


def test_build_training_metric_distinguishes_empty_noise_bucket() -> None:
    observation = _observation()
    measurement = replace(
        observation.microbatches[0],
        high_noise_loss_sum=torch.tensor(4.0),
        high_noise_sample_count=torch.tensor(2),
        low_noise_loss_sum=torch.tensor(0.0),
        low_noise_sample_count=torch.tensor(0),
        timesteps=torch.tensor([0.25, 0.5], dtype=torch.float32),
    )
    observation = replace(observation, microbatches=(measurement,))

    metric = build_training_metric(
        observation, context=_context(), recorded_at_unix_ns=123
    )

    assert metric.high_noise_loss == 2.0
    assert metric.high_noise_sample_count == 2
    assert metric.low_noise_loss == 0.0
    assert metric.low_noise_sample_count == 0


def test_build_training_metric_rejects_duplicate_or_pending_phase_facts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="duplicates observed"):
        build_training_metric(
            _observation(),
            context=_context(supplemental_phase_seconds={"qwen": 0.1}),
            recorded_at_unix_ns=123,
        )

    monkeypatch.setattr(
        PhaseTimer,
        "pending_cuda_pairs",
        property(lambda _self: 1),
    )
    with pytest.raises(RuntimeError, match="not ready"):
        build_training_metric(
            _observation(), context=_context(), recorded_at_unix_ns=123
        )


class _RecordingRemote:
    def __init__(self) -> None:
        self.metrics: list[TrainingMetric] = []

    def submit(self, metric: TrainingMetric) -> None:
        self.metrics.append(metric)


class _FailOncePublisher:
    def __init__(self) -> None:
        self.calls = 0
        self.published: list[TrainingMetric] = []
        self.first_entered = threading.Event()
        self.release_first = threading.Event()

    def publish(self, metric: TrainingMetric) -> None:
        self.calls += 1
        if self.calls == 1:
            self.first_entered.set()
            if not self.release_first.wait(timeout=1.0):
                raise TimeoutError("test did not release first publication")
            raise OSError("synthetic local publication failure")
        self.published.append(metric)


class _FailingRemote:
    def submit(self, metric: TrainingMetric) -> None:
        del metric
        raise OSError("remote submit failed")


def test_async_observer_publishes_local_first_complete_record(tmp_path: Path) -> None:
    local = DurableJsonlSink(tmp_path / "metrics.jsonl", fsync_every_records=1)
    remote = _RecordingRemote()
    observer = AsyncTrainingMetricObserver(
        MetricsPublisher(local, remote),
        context_provider=lambda _observation: _context(),
        queue_capacity=2,
        event_timeout_seconds=0.1,
        clock_ns=lambda: 123,
    )

    observer.submit(_observation())
    observer.close()
    local.close()

    payload = json.loads((tmp_path / "metrics.jsonl").read_text())
    assert payload["successful_update"] == 7
    assert payload["high_noise_sample_count"] == 1
    assert remote.metrics[0].successful_update == 7


def test_async_observer_times_out_without_synchronizing_cuda_events(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        PhaseTimer,
        "pending_cuda_pairs",
        property(lambda _self: 1),
    )

    def fail_synchronize(*_args: object, **_kwargs: object) -> NoReturn:
        pytest.fail("observer must not synchronize CUDA")

    monkeypatch.setattr(
        torch.cuda,
        "synchronize",
        fail_synchronize,
    )
    local = DurableJsonlSink(tmp_path / "metrics.jsonl", fsync_every_records=1)
    observer = AsyncTrainingMetricObserver(
        MetricsPublisher(local, _RecordingRemote()),
        context_provider=lambda _observation: _context(),
        queue_capacity=1,
        event_timeout_seconds=0.005,
        clock_ns=lambda: 123,
    )
    observer.submit(_observation())

    with pytest.raises(RuntimeError, match="observer failed") as captured:
        observer.close()
    local.close()

    assert isinstance(captured.value.__cause__, TimeoutError)
    assert not (tmp_path / "metrics.jsonl").read_text()


def test_async_observer_attempts_already_queued_records_after_first_failure() -> None:
    publisher = _FailOncePublisher()
    observer = AsyncTrainingMetricObserver(
        publisher,  # type: ignore[arg-type]
        context_provider=lambda _observation: _context(),
        queue_capacity=2,
        event_timeout_seconds=0.1,
        clock_ns=lambda: 123,
    )
    observer.submit(_observation(7))
    assert publisher.first_entered.wait(timeout=1.0)
    observer.submit(_observation(8))
    publisher.release_first.set()

    with pytest.raises(RuntimeError, match="observer failed") as captured:
        observer.close()

    assert isinstance(captured.value.__cause__, OSError)
    assert publisher.calls == 2
    assert [metric.successful_update for metric in publisher.published] == [8]


def test_observer_context_preserves_training_and_close_failures(
    tmp_path: Path,
) -> None:
    local = DurableJsonlSink(tmp_path / "metrics.jsonl", fsync_every_records=1)
    observer = AsyncTrainingMetricObserver(
        MetricsPublisher(local, _FailingRemote()),
        context_provider=lambda _observation: _context(),
        queue_capacity=1,
        event_timeout_seconds=0.1,
        clock_ns=lambda: 123,
    )

    with pytest.raises(BaseExceptionGroup) as captured, observer:
        observer.submit(_observation())
        raise ValueError("training failed")
    local.close()

    assert [type(error) for error in captured.value.exceptions] == [
        ValueError,
        RuntimeError,
    ]


def test_observer_rejects_duplicate_successful_update(tmp_path: Path) -> None:
    local = DurableJsonlSink(tmp_path / "metrics.jsonl", fsync_every_records=1)
    observer = AsyncTrainingMetricObserver(
        MetricsPublisher(local, _RecordingRemote()),
        context_provider=lambda _observation: _context(),
        queue_capacity=2,
        event_timeout_seconds=0.1,
        clock_ns=lambda: 123,
    )
    observation = _observation()
    observer.submit(observation)

    with pytest.raises(ValueError, match="consecutive"):
        observer.submit(observation)
    observer.close()
    local.close()

    assert len((tmp_path / "metrics.jsonl").read_text().splitlines()) == 1


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({"dit_flops": 0}, "positive integer"),
        ({"samples_per_second": 0.0}, "positive"),
        ({"ready_queue_depth": -1}, "nonnegative integer"),
        ({"supplemental_phase_seconds": {"unknown": 0.1}}, "unknown"),
    ],
)
def test_update_metric_context_is_strict(
    changes: Mapping[str, object], expected: str
) -> None:
    with pytest.raises((TypeError, ValueError), match=expected):
        _context(**changes)
