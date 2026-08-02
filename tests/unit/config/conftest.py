from __future__ import annotations

import copy
import tomllib
from pathlib import Path
from typing import Any, cast

import pytest

REPOSITORY_ROOT = Path(__file__).parents[3]
EXAMPLE_PATH = REPOSITORY_ROOT / "config/examples/all_options.example.toml"

SYNTHETIC_VALUES: dict[str, object] = {
    "storage.shared_mount_source": "server.example:/governed/export",
    "storage.minimum_free_gib": 8,
    "data.source.revision": "master",
    "data.manifest.path": "synthetic/train-manifest.json",
    "data.manifest.initialize_if_missing": True,
    "data.manifest.refresh_existing": False,
    "data.cache.low_watermark_gib": 8,
    "data.cache.high_watermark_gib": 16,
    "data.cache.download_concurrency": 2,
    "data.cache.verified_shard_lookahead": 2,
    "data.cache.persistent_workers_per_rank": 2,
    "data.cache.ready_batches_per_rank": 2,
    "data.service.request_timeout_seconds": 30.0,
    "data.service.lease_channel_capacity": 2,
    "data.service.ack_channel_capacity": 2,
    "data.transport.connect_timeout_seconds": 10.0,
    "data.transport.read_timeout_seconds": 30.0,
    "data.transport.max_retries": 2,
    "data.transport.retry_backoff_seconds": 0.0,
    "data.transport.stream_chunk_bytes": 1048576,
    "data.validation.selection_path": "synthetic/validation-selection.json",
    "data.validation.shard_root": "synthetic/validation-shards",
    "checkpoint.slots": 2,
    "stage.local_batch": 1,
    "stage.accumulation": 1,
    "stage.global_batch": 1,
    "stage.activation_checkpoint_mode": "none",
    "stage.planned_updates": 10,
    "profiling.schedule_updates": 10,
    "benchmark.profile_trace_updates": 5,
    "logging.flush_every_updates": 1,
    "logging.observer_queue_capacity": 2,
    "logging.observer_event_timeout_seconds": 30.0,
    "wandb.entity": "synthetic-entity",
    "wandb.queue_capacity": 2,
    "evaluation.enabled": True,
    "evaluation.extractor.enabled": True,
    "evaluation.extractor.feature_extractor": "synthetic-locked-extractor",
    "evaluation.extractor.feature_extractor_version": "synthetic-version",
    "evaluation.extractor.feature_extractor_path": "synthetic/extractor.pt",
    "evaluation.extractor.preprocess_path": "synthetic/preprocess.pt",
    "evaluation.fid.every_successful_updates": 10,
    "evaluation.fid.trend_samples": 100,
    "evaluation.fid.real_stats_path": "synthetic/real-stats.npz",
    "evaluation.is.every_successful_updates": 10,
    "evaluation.is.trend_samples": 100,
    "evaluation.is.splits": 10,
    "evaluation.gpu_index": 0,
    "evaluation.training_paused": True,
    "evaluation.batch_size": 10,
    "evaluation.output_reserve_gib": 8,
    "evaluation.manual_quality.enabled": True,
    "evaluation.manual_quality.samples": 100,
}


def _set_path(payload: dict[str, Any], path: str, value: object) -> None:
    parts = path.split(".")
    current = payload
    for part in parts[:-1]:
        child = current.setdefault(part, {})
        assert isinstance(child, dict)
        current = cast(dict[str, Any], child)
    current[parts[-1]] = value


@pytest.fixture
def valid_payload() -> dict[str, Any]:
    payload = tomllib.loads(EXAMPLE_PATH.read_text(encoding="utf-8"))
    for path, value in SYNTHETIC_VALUES.items():
        _set_path(payload, path, value)
    return copy.deepcopy(payload)


@pytest.fixture
def secret_environment() -> dict[str, str]:
    return {
        "MODELSCOPE_API_TOKEN": "synthetic-modelscope-secret",
        "WANDB_API_KEY": "synthetic-wandb-secret",
    }
