from __future__ import annotations

import hashlib
import io
import json
import multiprocessing as mp
import os
import tarfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol, cast

import torch
from PIL import Image

from sakuramoon.data.buckets import BucketShape
from sakuramoon.data.cache import CacheQuota, ShardCache
from sakuramoon.data.caption import (
    CaptionDropoutProbabilities,
    CaptionFields,
    NlCandidates,
    NlDropoutProbabilities,
)
from sakuramoon.data.client import DataServiceClient
from sakuramoon.data.collate import iter_service_batches
from sakuramoon.data.manifest import (
    DatasetManifest,
    DatasetSourceIdentity,
    ShardRecord,
    manifest_sha256,
)
from sakuramoon.data.metadata import MetadataFieldMapping
from sakuramoon.data.modelscope import ModelScopeDatasetTransport
from sakuramoon.data.pipeline import WebDatasetPipeline
from sakuramoon.data.serialize import MAIN_SUFFIX, SYSTEM_PREFIX, FramingContract
from sakuramoon.data.service import (
    DataServiceLimits,
    DataServiceServer,
    DataSupplyService,
)
from sakuramoon.data.service_protocol import (
    DataServiceSessionIdentity,
    ShardLeaseDescriptor,
)


class _Writer(Protocol):
    def write(self, payload: bytes, /) -> int: ...


class _Tokenizer:
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert not add_special_tokens
        if text == SYSTEM_PREFIX:
            return list(range(100, 134))
        if text == MAIN_SUFFIX:
            return list(range(200, 205))
        return [ord(character) + 300 for character in text]


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


def _pipeline(path: Path, record: ShardRecord) -> WebDatasetPipeline:
    nl = NlDropoutProbabilities(0.0, 0.0, 0.0, 0.0, 0.0)
    probabilities = CaptionDropoutProbabilities(
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, nl
    )
    fields = MetadataFieldMapping(
        id_field="id",
        width_field="width",
        height_field="height",
        caption_available_field="caption_available",
    )
    return WebDatasetPipeline(
        shard_paths=(path,),
        shard_records=(record,),
        metadata_adapter=lambda value: value,
        metadata_fields=fields,
        validation_ids=frozenset(),
        buckets=(BucketShape(512, 512),),
        min_crop_retention=0.8,
        probabilities=probabilities,
        tokenizer=_Tokenizer(),
        framing=FramingContract(34, 5, 0),
        caption_fields_parser=_caption,
        rejection_observer=lambda _reason: None,
        base_seed=9,
        stage="S0",
        pass_index=0,
    )


def _write_tar(path: Path, sample_id: int) -> None:
    image = io.BytesIO()
    Image.new("RGB", (640, 512), color=(10, 20, 30)).save(image, format="JPEG")
    metadata = json.dumps(
        {
            "id": sample_id,
            "width": 640,
            "height": 512,
            "caption_available": False,
        }
    ).encode()
    with tarfile.open(path, "w") as archive:
        for extension, payload in (("json", metadata), ("jpg", image.getvalue())):
            info = tarfile.TarInfo(f"{sample_id:06d}.{extension}")
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))


def _descriptors(tmp_path: Path) -> tuple[ShardLeaseDescriptor, ...]:
    descriptors: list[ShardLeaseDescriptor] = []
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
        descriptors.append(
            ShardLeaseDescriptor(
                lease_id=hashlib.sha256(f"lease-{index}".encode()).hexdigest(),
                worker_id=index % 2,
                state_identity=hashlib.sha256(
                    f"state-{index}".encode()
                ).hexdigest(),
                record=record,
                local_path=path,
            )
        )
    return tuple(descriptors)


class _Client:
    def __init__(self, descriptors: tuple[ShardLeaseDescriptor, ...]) -> None:
        self.identity = DataServiceSessionIdentity("2" * 64, 2)
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


class _LocalTransport:
    stream_chunk_bytes = 4096

    def __init__(self, bodies: dict[str, bytes]) -> None:
        self.bodies = bodies

    def download(
        self,
        manifest: DatasetManifest,
        shard: ShardRecord,
        output: _Writer,
    ) -> None:
        assert manifest.shard(shard.path) == shard
        output.write(self.bodies[shard.path])


class _BoundedRealClient:
    def __init__(self, client: DataServiceClient, lease_budget: int) -> None:
        self.client = client
        self.identity = client.identity
        self.remaining = lease_budget
        self.acknowledged: list[str] = []

    def health(self) -> bool:
        return self.client.health()

    def lease(self, worker_id: int) -> ShardLeaseDescriptor | None:
        if self.remaining == 0:
            return None
        descriptor = self.client.lease(worker_id)
        if descriptor is not None:
            self.remaining -= 1
        return descriptor

    def acknowledge(self, descriptor: ShardLeaseDescriptor) -> None:
        self.client.acknowledge(descriptor)
        self.acknowledged.append(descriptor.record.path)


def _run_service_process(
    root: Path,
    socket_path: Path,
    manifest: DatasetManifest,
    bodies: dict[str, bytes],
    identity: DataServiceSessionIdentity,
    stop: Any,
    ready: Any,
) -> None:
    cache = ShardCache(
        (root / "cache").absolute(),
        manifest,
        cast(ModelScopeDatasetTransport, _LocalTransport(bodies)),
        CacheQuota(1024, 1024 * 1024),
    )
    service = DataSupplyService(
        manifest,
        cache,
        root / "mainset.json",
        (root / "runtime" / "data-service.lock").absolute(),
        identity,
        DataServiceLimits(2, 3, 2, 2),
    )
    DataServiceServer(
        service,
        socket_path,
        request_timeout_seconds=5.0,
    ).serve(stop, ready_callback=ready.set)


def test_service_batch_consumer_runs_on_real_cuda(tmp_path: Path) -> None:
    assert torch.cuda.is_available()
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
    observed: list[int] = []
    for batch in batches:
        images = batch.images.to(device="cuda", dtype=torch.float32)
        observed.append(int(batch.sample_ids[0]))
        assert torch.isfinite(images.mean())
    torch.cuda.synchronize()

    assert sorted(observed) == [1, 2, 3, 4]
    assert len(client.acknowledged) == 4


def test_real_service_af_unix_client_reaches_real_cuda_consumer(tmp_path: Path) -> None:
    assert torch.cuda.is_available()
    descriptors = _descriptors(tmp_path)
    source = DatasetSourceIdentity(
        repo_id="leafmoone/webdataset_danbooru",
        revision="a" * 40,
        license_id="test-license",
        access_terms="test-terms",
    )
    records = tuple(descriptor.record for descriptor in descriptors)
    manifest = DatasetManifest.from_shards(source, records)
    bodies = {
        descriptor.record.path: descriptor.local_path.read_bytes()
        for descriptor in descriptors
    }
    identity = DataServiceSessionIdentity(manifest_sha256(manifest), 2)
    socket_path = (
        Path(__file__).parents[3] / f".d024-gpu-service-{os.getpid()}.sock"
    ).absolute()
    socket_path.unlink(missing_ok=True)
    context = cast(Any, mp.get_context("spawn"))
    stop = context.Event()
    ready = context.Event()
    process = context.Process(
        target=_run_service_process,
        args=(tmp_path, socket_path, manifest, bodies, identity, stop, ready),
    )
    process.start()
    observed: list[int] = []
    try:
        assert ready.wait(timeout=15.0)
        client = _BoundedRealClient(
            DataServiceClient(
                socket_path, identity, request_timeout_seconds=5.0
            ),
            lease_budget=len(records),
        )
        batches = iter_service_batches(
            _pipeline(descriptors[0].local_path, descriptors[0].record),
            client,
            batch_size=1,
            worker_count=2,
            ready_batches=2,
            pin_memory=False,
            drop_last=True,
        )
        for batch in batches:
            images = batch.images.to(device="cuda", dtype=torch.float32)
            observed.append(int(batch.sample_ids[0]))
            assert torch.isfinite(images.mean())
        torch.cuda.synchronize()
        assert sorted(client.acknowledged) == sorted(bodies)
    finally:
        stop.set()
        process.join(timeout=15.0)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5.0)
    assert process.exitcode == 0
    assert sorted(observed) == [1, 2, 3, 4]
    state = json.loads((tmp_path / "mainset.json").read_bytes())
    assert {row["status"] for row in state["rows"]} == {"pending"}
