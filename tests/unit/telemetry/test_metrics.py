from __future__ import annotations

import json
from pathlib import Path

import pytest

from sakuramoon.telemetry.metrics import (
    DROPOUT_KEYS,
    TIMING_PHASES,
    TRAINING_METRIC_SCHEMA_VERSION,
    DurableJsonlSink,
    MetricsPublisher,
    TrainingMetric,
)


def _metric(**changes: object) -> TrainingMetric:
    values: dict[str, object] = {
        "successful_update": 3,
        "recorded_at_unix_ns": 10,
        "total_loss": 1.0,
        "high_noise_loss": 1.2,
        "low_noise_loss": 0.8,
        "high_noise_sample_count": 3,
        "low_noise_sample_count": 1,
        "pre_clip_grad_norm": 2.0,
        "post_clip_grad_norm": 1.0,
        "clip_fraction": 0.5,
        "learning_rate": 0.00002,
        "timestep_min": 0.1,
        "timestep_max": 0.9,
        "timestep_mean": 0.5,
        "timestep_std": 0.2,
        "effective_batch": 4,
        "image_tokens": 1024,
        "text_tokens": 128,
        "dit_flops": 1000,
        "samples_per_second": 12.5,
        "gpu_memory_allocated_bytes": 1024,
        "gpu_memory_reserved_bytes": 2048,
        "ready_queue_depth": 2,
        "ready_queue_wait_seconds": 0.01,
        "nonfinite_count": 0,
        "dropout_hits": dict.fromkeys(DROPOUT_KEYS, 0),
        "phase_seconds": {
            **dict.fromkeys(TIMING_PHASES, 0.0),
            "data": 0.1,
            "dit_forward": 0.2,
        },
    }
    values.update(changes)
    return TrainingMetric(**values)  # pyright: ignore[reportArgumentType]


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({"total_loss": float("nan")}, "finite"),
        ({"clip_fraction": 1.1}, "exceed"),
        ({"dropout_hits": {"general": 1}}, "every fixed dropout"),
        ({"phase_seconds": {"unknown": 0.1}}, "every fixed timing"),
        ({"effective_batch": True}, "integer"),
        ({"high_noise_sample_count": 4}, "must equal effective batch"),
        ({"gpu_memory_reserved_bytes": 512}, "reserved GPU memory"),
        ({"post_clip_grad_norm": 3.0}, "post-clip gradient norm"),
        ({"timestep_min": -0.1}, "below its minimum"),
    ],
)
def test_metric_schema_rejects_invalid_or_incomplete_records(
    changes: dict[str, object], expected: str
) -> None:
    with pytest.raises((TypeError, ValueError), match=expected):
        _metric(**changes)


def test_metric_payload_is_fixed_numeric_and_contains_detailed_counts() -> None:
    metric = _metric(
        dropout_hits={key: index % 2 for index, key in enumerate(DROPOUT_KEYS)},
        phase_seconds={
            **dict.fromkeys(TIMING_PHASES, 0.0),
            "cache": 0.1,
            "qwen": 0.2,
            "zero_grad": 0.01,
        },
    )

    local = metric.as_json_mapping()
    remote = metric.as_wandb_mapping()

    assert local["schema_version"] == TRAINING_METRIC_SCHEMA_VERSION
    assert local["high_noise_sample_count"] == 3
    assert set(local["dropout_hits"]) == set(DROPOUT_KEYS)  # type: ignore[arg-type]
    assert remote["phase_seconds/cache"] == 0.1
    assert remote["dropout_hits/artist"] in {0, 1}
    encoded = json.dumps(local)
    assert "api_key" not in encoded
    assert "password" not in encoded
    assert "token=" not in encoded


def test_publisher_writes_local_jsonl_before_remote_submission(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    local = DurableJsonlSink(path, fsync_every_records=1)

    class AssertLocalFirst:
        def submit(self, metric: TrainingMetric) -> None:
            payload = json.loads(path.read_text())
            assert payload["successful_update"] == metric.successful_update

    publisher = MetricsPublisher(local, AssertLocalFirst())
    publisher.publish(_metric())
    local.close()

    assert len(path.read_text().splitlines()) == 1
    assert path.stat().st_mode & 0o077 == 0


def test_jsonl_sink_rejects_symlink_and_fsyncs_tail_on_close(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.write_text("untouched", encoding="utf-8")
    link = tmp_path / "metrics.jsonl"
    link.symlink_to(target)

    with pytest.raises((OSError, ValueError)):
        DurableJsonlSink(link, fsync_every_records=2)

    assert target.read_text() == "untouched"

    path = tmp_path / "tail.jsonl"
    with DurableJsonlSink(path, fsync_every_records=2) as sink:
        sink.write({"successful_update": 1})
    assert json.loads(path.read_text()) == {"successful_update": 1}


def test_jsonl_sink_rejects_insecure_existing_file(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    path.write_text("", encoding="utf-8")
    path.chmod(0o644)

    with pytest.raises(PermissionError, match="0600"):
        DurableJsonlSink(path, fsync_every_records=1)
