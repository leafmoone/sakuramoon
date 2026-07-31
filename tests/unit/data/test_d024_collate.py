from __future__ import annotations

import hashlib
import io
import json
import os
import tarfile
from collections.abc import Callable, Generator, Mapping
from pathlib import Path
from typing import cast

import pytest
from PIL import Image

from sakuramoon.data.buckets import BucketShape
from sakuramoon.data.caption import (
    CaptionDropoutProbabilities,
    CaptionFields,
    NlCandidates,
    NlDropoutProbabilities,
)
from sakuramoon.data.collate import iter_service_batches
from sakuramoon.data.manifest import ShardRecord
from sakuramoon.data.metadata import MetadataFieldMapping
from sakuramoon.data.pipeline import WebDatasetPipeline
from sakuramoon.data.serialize import MAIN_SUFFIX, SYSTEM_PREFIX, FramingContract
from sakuramoon.data.service_protocol import (
    DataServiceSessionIdentity,
    ShardLeaseDescriptor,
)


class _Tokenizer:
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert not add_special_tokens
        if text == SYSTEM_PREFIX:
            return list(range(100, 134))
        if text == MAIN_SUFFIX:
            return list(range(200, 205))
        return [ord(character) + 300 for character in text]


def _metadata(value: Mapping[str, object]) -> Mapping[str, object]:
    return value


def _fields() -> MetadataFieldMapping:
    return MetadataFieldMapping(
        id_field="id",
        width_field="width",
        height_field="height",
        caption_available_field="caption_available",
    )


def _caption(_value: Mapping[str, object]) -> CaptionFields:
    return CaptionFields(
        nsfw=(),
        character=(),
        copyright=(),
        general=(),
        artists=(),
        candidate_tags=frozenset(),
        nl=NlCandidates(None, None, None, None, None),
    )


def _probabilities() -> CaptionDropoutProbabilities:
    nl = NlDropoutProbabilities(0.0, 0.0, 0.0, 0.0, 0.0)
    return CaptionDropoutProbabilities(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, nl)


def _jpeg() -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (640, 512), color=(10, 20, 30)).save(output, format="JPEG")
    return output.getvalue()


def _write_tar(path: Path, sample_id: int) -> None:
    with tarfile.open(path, "w") as archive:
        for extension, payload in (
            (
                "json",
                json.dumps(
                    {
                        "id": sample_id,
                        "width": 640,
                        "height": 512,
                        "caption_available": False,
                    }
                ).encode(),
            ),
            ("jpg", _jpeg()),
        ):
            info = tarfile.TarInfo(f"{sample_id:06d}.{extension}")
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))


def _pipeline(
    path: Path,
    record: ShardRecord,
    *,
    metadata_adapter: Callable[
        [Mapping[str, object]], Mapping[str, object]
    ] = _metadata,
) -> WebDatasetPipeline:
    return WebDatasetPipeline(
        shard_paths=(path,),
        shard_records=(record,),
        metadata_adapter=metadata_adapter,
        metadata_fields=_fields(),
        validation_ids=frozenset(),
        buckets=(BucketShape(512, 512),),
        min_crop_retention=0.8,
        probabilities=_probabilities(),
        tokenizer=_Tokenizer(),
        framing=FramingContract(34, 5, 0),
        caption_fields_parser=_caption,
        rejection_observer=lambda _reason: None,
        base_seed=9,
        stage="S0",
        pass_index=0,
    )


class _Client:
    def __init__(self, descriptors: tuple[ShardLeaseDescriptor, ...]) -> None:
        self.identity = DataServiceSessionIdentity(
            manifest_sha256="2" * 64,
            worker_count=2,
        )
        self.pending = list(descriptors)
        self.active: dict[int, ShardLeaseDescriptor] = {}
        self.acknowledged: list[str] = []

    def health(self) -> bool:
        return not self.pending and not self.active

    def lease(self, worker_id: int) -> ShardLeaseDescriptor | None:
        if not self.pending:
            return None
        source = self.pending.pop(0)
        descriptor = ShardLeaseDescriptor(
            lease_id=source.lease_id,
            worker_id=worker_id,
            state_identity=source.state_identity,
            record=source.record,
            local_path=source.local_path,
        )
        self.active[worker_id] = descriptor
        return descriptor

    def acknowledge(self, descriptor: ShardLeaseDescriptor) -> None:
        assert self.active.pop(descriptor.worker_id) == descriptor
        self.acknowledged.append(descriptor.record.path)


def _descriptors(tmp_path: Path) -> tuple[ShardLeaseDescriptor, ...]:
    values: list[ShardLeaseDescriptor] = []
    for index in range(4):
        path = (tmp_path / f"{index}.tar").absolute()
        _write_tar(path, index + 1)
        record = ShardRecord(
            path=path.name,
            release="trusted",
            bytes=path.stat().st_size,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            samples=1,
        )
        values.append(
            ShardLeaseDescriptor(
                lease_id=hashlib.sha256(f"lease-{index}".encode()).hexdigest(),
                worker_id=index % 2,
                state_identity=hashlib.sha256(f"state-{index}".encode()).hexdigest(),
                record=record,
                local_path=path,
            )
        )
    return tuple(values)


def test_service_descriptors_drive_two_persistent_workers_and_ordered_acks(
    tmp_path: Path,
) -> None:
    descriptors = _descriptors(tmp_path)
    client = _Client(descriptors)
    batches = tuple(
        iter_service_batches(
            _pipeline(descriptors[0].local_path, descriptors[0].record),
            client,
            batch_size=1,
            worker_count=2,
            ready_batches=2,
            pin_memory=False,
            drop_last=True,
        )
    )

    assert sorted(int(batch.sample_ids[0]) for batch in batches) == [1, 2, 3, 4]
    assert sorted(client.acknowledged) == sorted(
        descriptor.record.path for descriptor in descriptors
    )
    assert client.active == {}


def test_parent_close_never_acks_partially_consumed_service_leases(
    tmp_path: Path,
) -> None:
    descriptors = _descriptors(tmp_path)
    client = _Client(descriptors)
    batches = iter_service_batches(
        _pipeline(descriptors[0].local_path, descriptors[0].record),
        client,
        batch_size=1,
        worker_count=2,
        ready_batches=2,
        pin_memory=False,
        drop_last=True,
    )
    next(batches)
    cast_batches = cast(Generator[object, None, None], batches)
    cast_batches.close()

    assert client.acknowledged == []
    assert len(client.active) == 2


def _crash_worker(value: Mapping[str, object]) -> Mapping[str, object]:
    del value
    os._exit(23)


def test_worker_exit_never_acks_service_lease(tmp_path: Path) -> None:
    descriptors = _descriptors(tmp_path)
    client = _Client(descriptors)
    with pytest.raises(RuntimeError, match="worker.*exited"):
        tuple(
            iter_service_batches(
                _pipeline(
                    descriptors[0].local_path,
                    descriptors[0].record,
                    metadata_adapter=_crash_worker,
                ),
                client,
                batch_size=1,
                worker_count=2,
                ready_batches=2,
                pin_memory=False,
                drop_last=True,
            )
        )
    assert client.acknowledged == []
    assert len(client.active) == 2
