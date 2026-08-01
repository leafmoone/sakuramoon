from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
import threading
import time
from pathlib import Path
from typing import Any, Protocol, cast

from sakuramoon.data.cache import CacheQuota, ShardCache
from sakuramoon.data.client import DataServiceClient
from sakuramoon.data.manifest import (
    DatasetManifest,
    DatasetSourceIdentity,
    ShardRecord,
    manifest_sha256,
)
from sakuramoon.data.modelscope import ModelScopeDatasetTransport
from sakuramoon.data.service import (
    DataServiceLimits,
    DataServiceServer,
    DataSupplyService,
)
from sakuramoon.data.service_protocol import DataServiceSessionIdentity


class _Writer(Protocol):
    def write(self, payload: bytes, /) -> int: ...


class _ProcessTransport:
    stream_chunk_bytes = 64

    def download(
        self, manifest: DatasetManifest, shard: ShardRecord, output: _Writer
    ) -> None:
        del manifest
        time.sleep(0.02)
        index = int(Path(shard.path).stem)
        output.write(bytes([index]) * shard.bytes)


def _manifest() -> DatasetManifest:
    source = DatasetSourceIdentity(
        repo_id="leafmoone/webdataset_danbooru",
        revision="a" * 40,
        license_id="test-license",
        access_terms="test-terms",
    )
    records = tuple(
        ShardRecord(
            path=f"release/{index:06d}.tar",
            release="release",
            bytes=512,
            sha256=hashlib.sha256(bytes([index]) * 512).hexdigest(),
            samples=index + 1,
        )
        for index in range(4)
    )
    return DatasetManifest.from_shards(source, records)


def _identity(manifest: DatasetManifest) -> DataServiceSessionIdentity:
    return DataServiceSessionIdentity(
        manifest_sha256=manifest_sha256(manifest),
        worker_count=2,
    )


def _run_service(
    root: Path,
    socket_path: Path,
    manifest: DatasetManifest,
    identity: DataServiceSessionIdentity,
    stop_event: Any,
    ready_queue: Any,
) -> None:
    cache = ShardCache(
        (root / "cache").absolute(),
        manifest,
        cast(ModelScopeDatasetTransport, _ProcessTransport()),
        CacheQuota(2048, 4096),
    )
    service = DataSupplyService(
        manifest,
        cache,
        root / "mainset.json",
        (root / "runtime" / "data-service.lock").absolute(),
        identity,
        DataServiceLimits(2, 3, 2, 2),
    )
    server = DataServiceServer(service, socket_path, request_timeout_seconds=5.0)
    server.serve(
        cast(threading.Event, stop_event),
        ready_callback=lambda: ready_queue.put((os.getpid(), "ready")),
    )


def _start(
    context: Any,
    root: Path,
    socket_path: Path,
    manifest: DatasetManifest,
    identity: DataServiceSessionIdentity,
) -> tuple[Any, Any]:
    stop = context.Event()
    ready = context.Queue(maxsize=1)
    process = context.Process(
        target=_run_service,
        args=(root, socket_path, manifest, identity, stop, ready),
    )
    process.start()
    pid, status = ready.get(timeout=15.0)
    assert status == "ready"
    assert pid == process.pid
    return process, stop


def test_service_kill_and_client_disconnect_replay_all_active_shards(
    tmp_path: Path,
) -> None:
    context = cast(Any, mp.get_context("spawn"))
    manifest = _manifest()
    identity = _identity(manifest)
    socket_path = (Path(__file__).parents[3] / f".d024-fault-{os.getpid()}.sock").absolute()
    socket_path.unlink(missing_ok=True)
    process, _stop = _start(context, tmp_path, socket_path, manifest, identity)
    assert process.pid != os.getpid()
    client = DataServiceClient(socket_path, identity, request_timeout_seconds=5.0)
    first = client.lease(0)
    second = client.lease(1)
    assert first is not None and second is not None
    initial_document = json.loads((tmp_path / "mainset.json").read_bytes())
    initial_order = tuple(row["path"] for row in initial_document["rows"])
    assert (first.record.path, second.record.path) == initial_order[:2]

    process.terminate()
    process.join(timeout=10.0)
    assert process.exitcode is not None and process.exitcode != 0

    restarted, stop = _start(context, tmp_path, socket_path, manifest, identity)
    restarted_client = DataServiceClient(
        socket_path, identity, request_timeout_seconds=5.0
    )
    replayed_first = restarted_client.lease(0)
    replayed_second = restarted_client.lease(1)
    assert replayed_first is not None and replayed_second is not None
    assert (replayed_first.record.path, replayed_second.record.path) == (
        first.record.path,
        second.record.path,
    )
    document = json.loads((tmp_path / "mainset.json").read_bytes())
    assert document["mainset_id"] == initial_document["mainset_id"]
    assert tuple(row["path"] for row in document["rows"]) == initial_order
    assert document["replayed_shards"] == 2
    assert document["replayed_samples"] == sum(
        descriptor.record.samples for descriptor in (first, second)
    )
    restarted_client.acknowledge(replayed_first)
    restarted_client.acknowledge(replayed_second)

    stop.set()
    restarted.join(timeout=10.0)
    assert restarted.exitcode == 0
    assert not socket_path.exists()
