from __future__ import annotations

import copy
import hashlib
import io
import json
import multiprocessing as mp
import multiprocessing.reduction
import os
import pickle
import tarfile
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Protocol, cast

import pytest
from PIL import Image
from pydantic import ValidationError

import sakuramoon.data.production as production_module
from sakuramoon.config.schema import RuntimeConfig
from sakuramoon.data.cache import CacheQuota, ShardCache
from sakuramoon.data.client import DataServiceClient
from sakuramoon.data.collate import DataLeaseClient, TrainingBatch
from sakuramoon.data.manifest import (
    DatasetManifest,
    DatasetSourceIdentity,
    ShardRecord,
    manifest_sha256,
)
from sakuramoon.data.modelscope import ModelScopeDatasetTransport
from sakuramoon.data.pipeline import PipelineSampleError
from sakuramoon.data.production import (
    AcceptedProductionBatchStream,
    ConfiguredDataLoader,
    ProductionDataError,
    ProductionPipelineFactory,
    require_accepted_production_batch_stream,
)
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
from sakuramoon.data.validation import VALIDATION_SAMPLE_COUNT


class _Tokenizer:
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert not add_special_tokens
        if text == SYSTEM_PREFIX:
            return [os.getpid(), *range(101, 134)]
        if text == MAIN_SUFFIX:
            return list(range(200, 205))
        return [] if not text else [os.getpid()]


def _observe_rejection(_reason: str) -> None:
    return None


class _Client:
    def __init__(self, worker_count: int, manifest_sha256: str = "1" * 64) -> None:
        self.identity = DataServiceSessionIdentity(manifest_sha256, worker_count)

    def health(self) -> bool:
        return False

    def lease(self, worker_id: int) -> ShardLeaseDescriptor | None:
        del worker_id
        return None

    def acknowledge(self, descriptor: ShardLeaseDescriptor) -> None:
        del descriptor


class _LeaseClient(_Client):
    def __init__(
        self, descriptor: ShardLeaseDescriptor, *, manifest_sha256: str
    ) -> None:
        super().__init__(worker_count=2, manifest_sha256=manifest_sha256)
        self.descriptor: ShardLeaseDescriptor | None = descriptor
        self.requested_workers: list[int] = []

    def lease(self, worker_id: int) -> ShardLeaseDescriptor | None:
        self.requested_workers.append(worker_id)
        descriptor, self.descriptor = self.descriptor, None
        return descriptor


class _Writer(Protocol):
    def write(self, payload: bytes, /) -> int: ...


class _TarTransport:
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


class _BoundedClient:
    def __init__(self, client: DataServiceClient, lease_budget: int) -> None:
        self.client = client
        self.identity = client.identity
        self.remaining = lease_budget
        self.leased: list[ShardLeaseDescriptor] = []
        self.acknowledged: list[ShardLeaseDescriptor] = []

    def health(self) -> bool:
        return self.client.health()

    def lease(self, worker_id: int) -> ShardLeaseDescriptor | None:
        if self.remaining == 0:
            return None
        descriptor = self.client.lease(worker_id)
        if descriptor is not None:
            self.remaining -= 1
            self.leased.append(descriptor)
        return descriptor

    def acknowledge(self, descriptor: ShardLeaseDescriptor) -> None:
        self.client.acknowledge(descriptor)
        self.acknowledged.append(descriptor)


def _tar_bytes(sample_id: int, *, valid_image: bool = True) -> bytes:
    metadata = json.dumps(
        {
            "captions": {"nl2": "", "nl3": ""},
            "dropout": {"candidate_tags": []},
            "id": sample_id,
            "image": {"height": 512, "width": 640},
            "multicaptions": {"vibes": ""},
            "nsfw": "safe",
            "tags": {
                "artist": [],
                "character": [],
                "copyright": [],
                "general": [],
            },
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    if valid_image:
        image = io.BytesIO()
        Image.new("RGB", (640, 512), color=(10, 20, 30)).save(
            image, format="JPEG"
        )
        image_payload = image.getvalue()
    else:
        image_payload = b"not-a-jpeg"
    body = io.BytesIO()
    with tarfile.open(fileobj=body, mode="w") as archive:
        for extension, payload in (("json", metadata), ("jpg", image_payload)):
            info = tarfile.TarInfo(f"{sample_id:06d}.{extension}")
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    return body.getvalue()


def _dataset(*, valid_images: bool = True) -> tuple[DatasetManifest, dict[str, bytes]]:
    source = DatasetSourceIdentity(
        repo_id="leafmoone/webdataset_danbooru",
        revision="a" * 40,
        license_id="test-license",
        access_terms="test-terms",
    )
    bodies = {
        f"release/{index:06d}.tar": _tar_bytes(
            3000 + index,
            valid_image=valid_images,
        )
        for index in range(4)
    }
    records = tuple(
        ShardRecord(
            path=path,
            release="1_2024",
            bytes=len(body),
            sha256=hashlib.sha256(body).hexdigest(),
            samples=1,
        )
        for path, body in bodies.items()
    )
    return DatasetManifest.from_shards(source, records), bodies


def _run_data_service(
    root: Path,
    socket_path: Path,
    ownership_lock_path: Path,
    manifest: DatasetManifest,
    bodies: dict[str, bytes],
    identity: DataServiceSessionIdentity,
    stop: Any,
    ready: Any,
) -> None:
    total_bytes = sum(len(body) for body in bodies.values())
    cache = ShardCache(
        (root / "cache").absolute(),
        manifest,
        cast(ModelScopeDatasetTransport, _TarTransport(bodies)),
        CacheQuota(total_bytes, total_bytes * 2),
    )
    service = DataSupplyService(
        manifest,
        cache,
        root / "mainset.json",
        ownership_lock_path,
        identity,
        DataServiceLimits(2, 3, 2, 2),
    )
    DataServiceServer(
        service,
        socket_path,
        request_timeout_seconds=5.0,
    ).serve(cast(threading.Event, stop), ready_callback=ready.set)


def _start_data_service(
    root: Path,
    manifest: DatasetManifest,
    bodies: dict[str, bytes],
) -> tuple[Any, Any, Path, DataServiceSessionIdentity]:
    context = cast(Any, mp.get_context("spawn"))
    runtime_root = Path("/run/sakuramoon")
    test_identity = hashlib.sha256(str(root).encode()).hexdigest()[:12]
    socket_path = runtime_root / f"d025-{test_identity}.sock"
    ownership_lock_path = runtime_root / f"d025-{test_identity}.lock"
    identity = DataServiceSessionIdentity(manifest_sha256(manifest), 2)
    stop = context.Event()
    ready = context.Event()
    process = context.Process(
        target=_run_data_service,
        args=(
            root,
            socket_path,
            ownership_lock_path,
            manifest,
            bodies,
            identity,
            stop,
            ready,
        ),
    )
    process.start()
    if not ready.wait(timeout=15.0):
        process.join(timeout=5.0)
        ownership_lock_path.unlink(missing_ok=True)
        raise AssertionError("data service did not reach its readiness barrier")
    return process, stop, socket_path, identity


def _stop_data_service(process: Any, stop: Any, socket_path: Path) -> None:
    stop.set()
    process.join(timeout=15.0)
    if process.is_alive():
        process.terminate()
        process.join(timeout=5.0)
    assert process.exitcode == 0
    assert not socket_path.exists()
    socket_path.with_suffix(".lock").unlink(missing_ok=True)


def _production_factory(
    valid_payload: dict[str, Any],
    root: Path,
    manifest: DatasetManifest,
) -> ProductionPipelineFactory:
    validation_path = root / "validation_manifest.jsonl"
    validation_payload = _validation_payload()
    validation_path.write_bytes(validation_payload)
    valid_payload["data"]["manifest"]["sha256"] = manifest_sha256(manifest)
    valid_payload["data"]["validation"]["manifest_path"] = str(validation_path)
    valid_payload["data"]["validation"]["manifest_sha256"] = hashlib.sha256(
        validation_payload
    ).hexdigest()
    valid_payload["stage"]["local_batch"] = 1
    valid_payload["data"]["cache"]["persistent_workers_per_rank"] = 2
    valid_payload["data"]["cache"]["ready_batches_per_rank"] = 2
    valid_payload["data"]["loader"]["pin_memory"] = False
    valid_payload["data"]["loader"]["drop_last"] = True
    config = RuntimeConfig.model_validate(valid_payload)
    return ProductionPipelineFactory.from_config(
        config,
        repository_root=root,
        tokenizer=_Tokenizer(),
        framing=FramingContract(34, 5, 0),
        rejection_observer=_observe_rejection,
        pass_index=0,
    )


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
    valid_payload["stage"]["global_batch"] = 7
    valid_payload["stage"]["planned_valid_samples"] = (
        7 * valid_payload["stage"]["planned_updates"]
    )
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
        rejection_observer=_observe_rejection,
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
    client = _LeaseClient(
        descriptor,
        manifest_sha256=config.data.manifest.sha256,
    )
    stream = factory.batches(client)
    assert isinstance(stream, AcceptedProductionBatchStream)
    assert require_accepted_production_batch_stream(stream) is stream
    assert stream.identity.loader == factory.loader
    assert stream.identity.manifest_sha256 == config.data.manifest.sha256
    assert stream.identity.service_session_sha256 == client.identity.sha256
    assert stream.identity.factory_identity == factory.factory_identity
    list(stream)

    assert client.requested_workers == [0]
    assert observed == {
        "batch_size": config.stage.local_batch,
        "worker_count": config.data.cache.persistent_workers_per_rank,
        "ready_batches": config.data.cache.ready_batches_per_rank,
        "pin_memory": config.data.loader.pin_memory,
        "drop_last": config.data.loader.drop_last,
    }


def test_factory_hard_fails_manifest_drift_unpickleable_fields_and_plain_streams(
    valid_payload: dict[str, Any],
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "validation_manifest.jsonl"
    payload = _validation_payload()
    manifest_path.write_bytes(payload)
    valid_payload["data"]["validation"]["manifest_path"] = str(manifest_path)
    valid_payload["data"]["validation"]["manifest_sha256"] = hashlib.sha256(
        payload
    ).hexdigest()
    config = RuntimeConfig.model_validate(valid_payload)

    with pytest.raises(ProductionDataError, match="spawn context"):
        ProductionPipelineFactory.from_config(
            config,
            repository_root=tmp_path,
            tokenizer=_Tokenizer(),
            framing=FramingContract(34, 5, 0),
            rejection_observer=lambda _reason: None,
            pass_index=0,
        )

    factory = ProductionPipelineFactory.from_config(
        config,
        repository_root=tmp_path,
        tokenizer=_Tokenizer(),
        framing=FramingContract(34, 5, 0),
        rejection_observer=_observe_rejection,
        pass_index=0,
    )
    with pytest.raises(ProductionDataError, match="manifest identity"):
        factory.batches(_Client(worker_count=2, manifest_sha256="0" * 64))
    with pytest.raises(ProductionDataError, match="factory-issued"):
        require_accepted_production_batch_stream(iter(()))


def test_factory_rejects_direct_construction_and_object_new_forgery(
    valid_payload: dict[str, Any],
    tmp_path: Path,
) -> None:
    ungoverned_payload = copy.deepcopy(valid_payload)
    missing_manifest = tmp_path / "missing-validation.jsonl"
    ungoverned_payload["data"]["validation"]["manifest_path"] = str(
        missing_manifest
    )
    ungoverned_payload["data"]["validation"]["manifest_sha256"] = "f" * 64
    ungoverned_config = RuntimeConfig.model_validate(ungoverned_payload)
    assert not missing_manifest.exists()
    with pytest.raises(ProductionDataError, match="issued by from_config"):
        ProductionPipelineFactory(
            config=ungoverned_config,
            validation_ids=frozenset(range(9_000_001, 9_002_001)),
            tokenizer=_Tokenizer(),
            framing=FramingContract(34, 5, 0),
            rejection_observer=_observe_rejection,
            pass_index=0,
            factory_identity="f" * 64,
        )

    manifest, _bodies = _dataset()
    factory = _production_factory(valid_payload, tmp_path, manifest)
    client = _Client(
        worker_count=2,
        manifest_sha256=factory.config.data.manifest.sha256,
    )

    forged = object.__new__(ProductionPipelineFactory)
    for field in (
        "config",
        "validation_ids",
        "tokenizer",
        "framing",
        "rejection_observer",
        "pass_index",
        "factory_identity",
        "_owner_pid",
    ):
        object.__setattr__(forged, field, getattr(factory, field))
    with pytest.raises(ProductionDataError, match="issued by from_config"):
        forged.batches(client)

    deserialized = pickle.loads(
        multiprocessing.reduction.ForkingPickler.dumps(factory)
    )
    with pytest.raises(ProductionDataError, match="issued by from_config"):
        deserialized.batches(client)


def test_accepted_stream_is_factory_issued_process_local_and_not_pickleable(
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
        rejection_observer=_observe_rejection,
        pass_index=0,
    )
    other_factory = ProductionPipelineFactory.from_config(
        config,
        repository_root=tmp_path,
        tokenizer=_Tokenizer(),
        framing=FramingContract(34, 5, 0),
        rejection_observer=_observe_rejection,
        pass_index=0,
    )
    assert factory.factory_identity != other_factory.factory_identity
    client = _Client(
        worker_count=2,
        manifest_sha256=config.data.manifest.sha256,
    )
    stream = factory.batches(client)
    other_stream = other_factory.batches(client)
    assert stream.identity.factory_identity == factory.factory_identity
    assert other_stream.identity.factory_identity == other_factory.factory_identity
    other_stream.close()

    with pytest.raises(ProductionDataError, match="issued by the production factory"):
        AcceptedProductionBatchStream(
            iter(()),
            stream.identity,
            token="0" * 64,
            authority=object(),
        )
    with pytest.raises(ProductionDataError, match="cannot be serialized"):
        multiprocessing.reduction.ForkingPickler.dumps(stream)
    other_pid = os.getpid() + 1
    monkeypatch.setattr(production_module.os, "getpid", lambda: other_pid)
    with pytest.raises(ProductionDataError, match="process boundary"):
        require_accepted_production_batch_stream(stream)


def test_real_service_factory_runs_two_spawned_workers_and_acks_each_lease(
    valid_payload: dict[str, Any],
    tmp_path: Path,
) -> None:
    manifest, bodies = _dataset()
    factory = _production_factory(valid_payload, tmp_path, manifest)
    process, stop, socket_path, identity = _start_data_service(
        tmp_path, manifest, bodies
    )
    stream: AcceptedProductionBatchStream | None = None
    try:
        client = _BoundedClient(
            DataServiceClient(socket_path, identity, request_timeout_seconds=5.0),
            lease_budget=len(manifest.shards),
        )
        stream = factory.batches(client)
        batches = tuple(stream)

        worker_pids = {
            int(
                batch.input_ids[0, 0].item()  # pyright: ignore[reportUnknownMemberType]
            )
            for batch in batches
        }
        assert len(worker_pids) == 2
        assert os.getpid() not in worker_pids
        assert sorted(int(batch.sample_ids[0]) for batch in batches) == [
            3000,
            3001,
            3002,
            3003,
        ]
        assert {descriptor.worker_id for descriptor in client.leased} == {0, 1}
        assert sorted(
            descriptor.record.path for descriptor in client.acknowledged
        ) == sorted(bodies)
        assert len(client.acknowledged) == len(manifest.shards)
    finally:
        if stream is not None:
            stream.close()
        _stop_data_service(process, stop, socket_path)

    state = json.loads((tmp_path / "mainset.json").read_bytes())
    assert {row["status"] for row in state["rows"]} == {"pending"}


def test_factory_early_close_keeps_leases_active_and_restart_replays_from_start(
    valid_payload: dict[str, Any],
    tmp_path: Path,
) -> None:
    manifest, bodies = _dataset()
    factory = _production_factory(valid_payload, tmp_path, manifest)
    process, stop, socket_path, identity = _start_data_service(
        tmp_path, manifest, bodies
    )
    client = _BoundedClient(
        DataServiceClient(socket_path, identity, request_timeout_seconds=5.0),
        lease_budget=len(manifest.shards),
    )
    stream = factory.batches(client)
    try:
        first_batch = next(stream)
        assert int(first_batch.sample_ids[0]) in {3000, 3001, 3002, 3003}
        stream.close()
        state = json.loads((tmp_path / "mainset.json").read_bytes())
        active_paths = tuple(
            row["path"] for row in state["rows"] if row["status"] == "active"
        )
        assert len(active_paths) == 2
        assert client.acknowledged == []
    finally:
        stream.close()
        _stop_data_service(process, stop, socket_path)

    restarted, restarted_stop, socket_path, identity = _start_data_service(
        tmp_path, manifest, bodies
    )
    replay_stream: AcceptedProductionBatchStream | None = None
    try:
        replay_client = _BoundedClient(
            DataServiceClient(socket_path, identity, request_timeout_seconds=5.0),
            lease_budget=len(active_paths),
        )
        replay_stream = factory.batches(replay_client)
        replayed_batches = tuple(replay_stream)
        assert sorted(
            descriptor.record.path for descriptor in replay_client.leased
        ) == sorted(active_paths)
        assert sorted(
            int(batch.sample_ids[0]) for batch in replayed_batches
        ) == sorted(
            3000 + int(Path(path).stem) for path in active_paths
        )
        assert sorted(
            descriptor.record.path for descriptor in replay_client.acknowledged
        ) == sorted(active_paths)
    finally:
        if replay_stream is not None:
            replay_stream.close()
        _stop_data_service(restarted, restarted_stop, socket_path)

    replay_state = json.loads((tmp_path / "mainset.json").read_bytes())
    assert replay_state["replayed_shards"] == 2
    assert replay_state["replayed_samples"] == 2


def test_factory_worker_failure_never_acks_and_restart_replays_active_leases(
    valid_payload: dict[str, Any],
    tmp_path: Path,
) -> None:
    manifest, bodies = _dataset(valid_images=False)
    factory = _production_factory(valid_payload, tmp_path, manifest)
    process, stop, socket_path, identity = _start_data_service(
        tmp_path, manifest, bodies
    )
    client = _BoundedClient(
        DataServiceClient(socket_path, identity, request_timeout_seconds=5.0),
        lease_budget=2,
    )
    stream = factory.batches(client)
    try:
        with pytest.raises(PipelineSampleError, match="worker"):
            tuple(stream)
        state = json.loads((tmp_path / "mainset.json").read_bytes())
        active_paths = tuple(
            row["path"] for row in state["rows"] if row["status"] == "active"
        )
        assert len(active_paths) == 2
        assert client.acknowledged == []
    finally:
        stream.close()
        _stop_data_service(process, stop, socket_path)

    restarted, restarted_stop, socket_path, identity = _start_data_service(
        tmp_path, manifest, bodies
    )
    replay_stream: AcceptedProductionBatchStream | None = None
    try:
        replay_client = _BoundedClient(
            DataServiceClient(socket_path, identity, request_timeout_seconds=5.0),
            lease_budget=2,
        )
        replay_stream = factory.batches(replay_client)
        with pytest.raises(PipelineSampleError, match="worker"):
            tuple(replay_stream)
        assert sorted(
            descriptor.record.path for descriptor in replay_client.leased
        ) == sorted(active_paths)
        assert replay_client.acknowledged == []
    finally:
        if replay_stream is not None:
            replay_stream.close()
        _stop_data_service(restarted, restarted_stop, socket_path)

    replay_state = json.loads((tmp_path / "mainset.json").read_bytes())
    assert replay_state["replayed_shards"] == 2
    assert replay_state["replayed_samples"] == 2
