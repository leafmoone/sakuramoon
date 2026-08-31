"""Config-routed SR data-service instance: SR_v2 source, isolated runtime
paths, train-only validation (shard_count = 0), corpus-sized cache window."""

from __future__ import annotations

from pathlib import Path

import pytest

from sakuramoon.config import load_config
from sakuramoon.config.schema import (
    DataServiceConfig,
    DataValidationConfig,
    RuntimeConfig,
)
from sakuramoon.data.manifest import (
    DatasetManifest,
    DatasetSourceIdentity,
    ShardRecord,
    canonical_manifest_bytes,
    parse_dataset_manifest_bytes,
)

REPOSITORY_ROOT = Path(__file__).parents[3]


@pytest.fixture
def secret_environment() -> dict[str, str]:
    return {
        "MODELSCOPE_API_TOKEN": "synthetic-modelscope-secret",
        "WANDB_API_KEY": "synthetic-wandb-secret",
    }


def test_sr_data_service_config_loads_and_routes(
    secret_environment: dict[str, str],
) -> None:
    loaded = load_config(
        Path("sr_data_service.toml"),
        config_root=REPOSITORY_ROOT / "config",
        environment=secret_environment,
    )
    config = loaded.config
    assert config.data.source.repo_id == "leafmoone/SR_v2"
    assert config.data.source.revision == "master"
    assert config.data.manifest.path == "data/dataset-manifest-sr-v2.json"
    # Isolated from a co-located G1 instance (data-service.sock/.lock).
    assert config.data.service.socket_path == "/run/sakuramoon/sr-data-service.sock"
    assert (
        config.data.service.ownership_lock_path
        == "/run/sakuramoon/sr-data-service.lock"
    )
    # Train-only corpus: no held-out validation split.
    assert config.data.validation.shard_count == 0
    # Corpus-sized window (whole SR shard set held without LRU eviction).
    assert config.data.cache.low_watermark_gib == 40
    assert config.data.cache.high_watermark_gib == 48
    assert "BENCHMARK" not in loaded.resolved_toml


def test_g1_default_runtime_paths_still_validate() -> None:
    service = DataServiceConfig(
        socket_path="/run/sakuramoon/data-service.sock",
        ownership_lock_path="/run/sakuramoon/data-service.lock",
        mainset_path="/root/private_data/sakuramoon/config/data-service-mainset.json",
        request_timeout_seconds=30.0,
        lease_channel_capacity=24,
        ack_channel_capacity=24,
    )
    assert service.socket_path == "/run/sakuramoon/data-service.sock"


@pytest.mark.parametrize(
    ("socket_path", "lock_path"),
    [
        ("data-service.sock", "/run/sakuramoon/data-service.lock"),  # relative
        ("/run/other/data-service.sock", "/run/other/data-service.lock"),
        ("/run/sakuramoon/data-service.lock", "/run/sakuramoon/data-service.lock"),
        ("/run/sakuramoon/data-service.sock", "/run/elsewhere/data-service.lock"),
        ("/run/sakuramoon/..", "/run/sakuramoon/data-service.lock"),
        (
            "/run/sakuramoon/data-service.sock",
            "/run/sakuramoon/./data-service.lock",
        ),
        ("/run/sakuramoon/.sock", "/run/sakuramoon/data-service.lock"),
    ],
)
def test_runtime_path_validation_rejects_bad_paths(
    socket_path: str,
    lock_path: str,
) -> None:
    with pytest.raises(ValueError):
        DataServiceConfig(
            socket_path=socket_path,
            ownership_lock_path=lock_path,
            mainset_path="/root/private_data/sr/state/data-service-mainset.json",
            request_timeout_seconds=30.0,
            lease_channel_capacity=24,
            ack_channel_capacity=24,
        )


@pytest.mark.parametrize(
    "shard_count",
    [0, 7],
)
def test_validation_shard_count_bounds(shard_count: int) -> None:
    validation = DataValidationConfig(
        selection_path="data/validation/selection.json",
        shard_root="data/validation/shards",
        shard_count=shard_count,
    )
    assert validation.shard_count == shard_count


def test_validation_shard_count_negative_rejected() -> None:
    with pytest.raises(ValueError):
        DataValidationConfig(
            selection_path="data/validation/selection.json",
            shard_root="data/validation/shards",
            shard_count=-1,
        )


def test_dataset_manifest_accepts_sr_source_identity() -> None:
    source = DatasetSourceIdentity(repo_id="leafmoone/SR_v2", revision="master")
    shards = (
        ShardRecord(path="data/1_2024/shard-000000-p2-00.tar", bytes=100),
        ShardRecord(path="data/1_2024/shard-000001-p2-00.tar", bytes=200),
    )
    manifest = DatasetManifest.from_shards(source, shards)
    assert manifest.dataset_id == "leafmoone/SR_v2@master"
    assert manifest.aggregates.shards == 2
    assert manifest.aggregates.bytes == 300
    roundtrip = parse_dataset_manifest_bytes(canonical_manifest_bytes(manifest))
    assert roundtrip == manifest


def test_unknown_dataset_source_rejected() -> None:
    with pytest.raises(ValueError):
        DatasetSourceIdentity(repo_id="leafmoone/other-corpus", revision="master")  # type: ignore[arg-type]


def test_runtime_config_accepts_sr_config_object() -> None:
    loaded = load_config(
        Path("sr_data_service.toml"),
        config_root=REPOSITORY_ROOT / "config",
        environment={
            "MODELSCOPE_API_TOKEN": "synthetic-modelscope-secret",
            "WANDB_API_KEY": "synthetic-wandb-secret",
        },
    )
    assert isinstance(loaded.config, RuntimeConfig)
