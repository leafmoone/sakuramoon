from __future__ import annotations

import copy
import tomllib
from pathlib import Path
from typing import Any, cast

import pytest

REPOSITORY_ROOT = Path(__file__).parents[3]
EXAMPLE_PATH = REPOSITORY_ROOT / "config/examples/all_options.example.toml"

SYNTHETIC_VALUES: dict[str, object] = {
    "data.source.revision": "c" * 40,
    "data.manifest.path": "synthetic/train-manifest.jsonl",
    "data.manifest.sha256": "3" * 64,
    "data.cache.low_watermark_gib": 300,
    "data.cache.high_watermark_gib": 400,
    "data.cache.download_concurrency": 2,
    "data.cache.range_workers": 2,
    "data.cache.persistent_workers_per_rank": 2,
    "data.cache.ready_batches_per_rank": 2,
    "data.transport.connect_timeout_seconds": 10.0,
    "data.transport.read_timeout_seconds": 30.0,
    "data.transport.max_retries": 2,
    "data.transport.retry_backoff_seconds": 0.0,
    "data.transport.stream_chunk_bytes": 1048576,
    "data.validation.manifest_path": "synthetic/validation-manifest.jsonl",
    "data.validation.manifest_sha256": "4" * 64,
    # Synthetic values exercise schema behavior only. They are not defaults,
    # recommendations, or approved production dropout probabilities.
    "caption.dropout.general": 0.01,
    "caption.dropout.artist": 0.01,
    "caption.dropout.character": 0.01,
    "caption.dropout.copyright": 0.01,
    "caption.dropout.nsfw": 0.01,
    "caption.dropout.candidate_source": 0.01,
    "caption.dropout.nl.long_names": 0.02,
    "caption.dropout.nl.long_no_names": 0.02,
    "caption.dropout.nl.short_vibes": 0.02,
    "caption.dropout.nl.nl2": 0.02,
    "caption.dropout.nl.nl3": 0.02,
    "checkpoint.slots": 3,
    "stage.local_batch": 1,
    "stage.accumulation": 1,
    "stage.planned_updates": 10,
    "profiling.schedule_updates": 10,
    "logging.flush_every_updates": 1,
    "wandb.entity": "synthetic-entity",
    "evaluation.fid.every_successful_updates": 10,
    "evaluation.fid.trend_samples": 100,
    "evaluation.fid.feature_extractor": "synthetic-locked-extractor",
    "evaluation.fid.feature_extractor_version": "synthetic-version",
    "evaluation.fid.preprocess_sha256": "6" * 64,
    "evaluation.fid.real_stats_sha256": "5" * 64,
    "evaluation.is.every_successful_updates": 10,
    "evaluation.is.trend_samples": 100,
    "evaluation.is.splits": 10,
    "evaluation.prompt_manifest_path": "synthetic/prompts.json",
    "evaluation.prompt_manifest_sha256": "7" * 64,
    "evaluation.gpu_index": 0,
    "evaluation.training_paused": True,
}


def _set_path(payload: dict[str, Any], path: str, value: object) -> None:
    parts = path.split(".")
    current = payload
    for part in parts[:-1]:
        child = current[part]
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
