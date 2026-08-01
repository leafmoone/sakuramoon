from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

import sakuramoon.data.production as production_module
from sakuramoon.config.schema import RuntimeConfig
from sakuramoon.data.collate import DataLeaseClient, TrainingBatch
from sakuramoon.data.manifest import ShardRecord
from sakuramoon.data.production import (
    ConfiguredDataLoader,
    ProductionDataError,
    ProductionPipelineFactory,
)
from sakuramoon.data.serialize import FramingContract
from sakuramoon.data.service_protocol import (
    DataServiceSessionIdentity,
    ShardLeaseDescriptor,
)
from sakuramoon.data.validation import VALIDATION_SAMPLE_COUNT


class _Tokenizer:
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        del text, add_special_tokens
        return []


class _Client:
    def __init__(self, worker_count: int) -> None:
        self.identity = DataServiceSessionIdentity("1" * 64, worker_count)

    def health(self) -> bool:
        return False

    def lease(self, worker_id: int) -> ShardLeaseDescriptor | None:
        del worker_id
        return None

    def acknowledge(self, descriptor: ShardLeaseDescriptor) -> None:
        del descriptor


class _LeaseClient(_Client):
    def __init__(self, descriptor: ShardLeaseDescriptor) -> None:
        super().__init__(worker_count=2)
        self.descriptor: ShardLeaseDescriptor | None = descriptor
        self.requested_workers: list[int] = []

    def lease(self, worker_id: int) -> ShardLeaseDescriptor | None:
        self.requested_workers.append(worker_id)
        descriptor, self.descriptor = self.descriptor, None
        return descriptor


def _validation_payload() -> bytes:
    lines = (
        json.dumps(
            {
                "aspect_bucket": "square",
                "caption_available": False,
                "id": sample_id,
                "release": "1_2024",
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        for sample_id in range(1, VALIDATION_SAMPLE_COUNT + 1)
    )
    return ("\n".join(lines) + "\n").encode()


def test_loader_controls_are_required_resolved_toml_fields(
    valid_payload: dict[str, Any],
) -> None:
    for field in ("pin_memory", "drop_last"):
        missing = copy.deepcopy(valid_payload)
        missing["data"]["loader"].pop(field)
        with pytest.raises(ValidationError, match=rf"(?s){field}.*missing"):
            RuntimeConfig.model_validate(missing)

    wrong_type = copy.deepcopy(valid_payload)
    wrong_type["data"]["loader"]["pin_memory"] = 1
    with pytest.raises(ValidationError, match="bool_type"):
        RuntimeConfig.model_validate(wrong_type)


def test_configured_loader_passes_only_exact_resolved_values(
    valid_payload: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    valid_payload["stage"]["local_batch"] = 7
    valid_payload["data"]["cache"]["ready_batches_per_rank"] = 4
    valid_payload["data"]["loader"]["pin_memory"] = False
    valid_payload["data"]["loader"]["drop_last"] = False
    config = RuntimeConfig.model_validate(valid_payload)
    observed: dict[str, object] = {}

    def fake_iter(
        pipeline: object,
        client: object,
        **values: object,
    ) -> Iterator[TrainingBatch]:
        observed.update(values)
        return iter(())

    monkeypatch.setattr(production_module, "iter_service_batches", fake_iter)
    loader = ConfiguredDataLoader.from_config(config)
    list(loader.batches(object(), _Client(worker_count=2)))  # type: ignore[arg-type]

    assert observed == {
        "batch_size": 7,
        "worker_count": 2,
        "ready_batches": 4,
        "pin_memory": False,
        "drop_last": False,
    }
    with pytest.raises(ProductionDataError, match="worker_count"):
        loader.batches(object(), _Client(worker_count=1))  # type: ignore[arg-type]


def test_factory_loads_validation_and_freezes_production_pipeline_contract(
    valid_payload: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = tmp_path / "validation_manifest.jsonl"
    payload = _validation_payload()
    manifest_path.write_bytes(payload)
    valid_payload["data"]["validation"]["manifest_path"] = str(manifest_path)
    valid_payload["data"]["validation"]["manifest_sha256"] = hashlib.sha256(
        payload
    ).hexdigest()
    config = RuntimeConfig.model_validate(valid_payload)
    factory = ProductionPipelineFactory.from_config(
        config,
        repository_root=tmp_path,
        tokenizer=_Tokenizer(),
        framing=FramingContract(34, 5, 0),
        rejection_observer=lambda _reason: None,
        pass_index=0,
    )
    shard_path = (tmp_path / "sample.tar").absolute()
    shard_path.write_bytes(b"placeholder")
    record = ShardRecord(
        path=shard_path.name,
        release="1_2024",
        bytes=shard_path.stat().st_size,
        sha256="2" * 64,
        samples=1,
    )
    descriptor = ShardLeaseDescriptor(
        lease_id="3" * 64,
        worker_id=0,
        state_identity="4" * 64,
        record=record,
        local_path=shard_path,
    )

    pipeline = factory.pipeline_for_lease(descriptor)

    assert len(factory.validation_ids) == VALIDATION_SAMPLE_COUNT
    assert pipeline.validation_ids == factory.validation_ids
    assert pipeline.metadata_adapter is production_module.adapt_modelscope_metadata
    assert pipeline.metadata_fields is production_module.PRODUCTION_METADATA_FIELDS
    assert (
        pipeline.caption_fields_parser
        is production_module.parse_modelscope_caption_fields
    )
    assert len(pipeline.buckets) == 17
    assert max(shape.height for shape in pipeline.buckets) <= 512
    assert pipeline.base_seed == config.run.seed
    assert pipeline.stage == config.stage.name
    assert pipeline.pass_index == 0

    observed: dict[str, object] = {}

    def fake_iter(
        pipeline: object,
        client: DataLeaseClient,
        **values: object,
    ) -> Iterator[TrainingBatch]:
        assert pipeline is not None
        assert not client.health()
        assert client.lease(0) == descriptor
        observed.update(values)
        return iter(())

    monkeypatch.setattr(production_module, "iter_service_batches", fake_iter)
    client = _LeaseClient(descriptor)
    list(factory.batches(client))

    assert client.requested_workers == [0]
    assert observed == {
        "batch_size": config.stage.local_batch,
        "worker_count": config.data.cache.persistent_workers_per_rank,
        "ready_batches": config.data.cache.ready_batches_per_rank,
        "pin_memory": config.data.loader.pin_memory,
        "drop_last": config.data.loader.drop_last,
    }
