from __future__ import annotations

import hashlib
import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Protocol, cast

import pytest

from sakuramoon.data.cache import CacheQuota, ShardCache
from sakuramoon.data.client import DataServiceClient
from sakuramoon.data.manifest import (
    DatasetManifest,
    DatasetSourceIdentity,
    ShardRecord,
    manifest_sha256,
)
from sakuramoon.data.modelscope import DatasetTransportError, ModelScopeDatasetTransport
from sakuramoon.data.service import (
    DataServiceError,
    DataServiceLimits,
    DataServiceServer,
    DataServiceStateCommittedError,
    DataSupplyService,
)
from sakuramoon.data.service_protocol import DataServiceSessionIdentity
from sakuramoon.data.validation import (
    VALIDATION_SHARD_PATHS,
    ValidationSelection,
    select_validation_shards,
)


class _Writer(Protocol):
    def write(self, payload: bytes, /) -> int: ...


class _SlowTransport:
    stream_chunk_bytes = 64

    def __init__(self, bodies: dict[str, bytes], delay: float = 0.02) -> None:
        self.bodies = bodies
        self.delay = delay
        self._lock = threading.Lock()
        self.active = 0
        self.max_active = 0
        self.downloaded: list[str] = []

    def download(
        self, manifest: DatasetManifest, shard: ShardRecord, output: _Writer
    ) -> None:
        assert manifest.shard(shard.path) == shard
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.downloaded.append(shard.path)
        try:
            time.sleep(self.delay)
            output.write(self.bodies[shard.path])
        finally:
            with self._lock:
                self.active -= 1


class _GatedTransport(_SlowTransport):
    def __init__(self, bodies: dict[str, bytes]) -> None:
        super().__init__(bodies, delay=0.0)
        self.release = threading.Event()
        self.first_wave_started = threading.Event()
        self._started = 0

    def download(
        self, manifest: DatasetManifest, shard: ShardRecord, output: _Writer
    ) -> None:
        with self._lock:
            self._started += 1
            if self._started == 2:
                self.first_wave_started.set()
        if not self.release.wait(timeout=5.0):
            raise AssertionError("test did not release cold-cache downloads")
        super().download(manifest, shard, output)


class _FailingTransport(_SlowTransport):
    def download(
        self, manifest: DatasetManifest, shard: ShardRecord, output: _Writer
    ) -> None:
        del manifest, shard, output
        raise DatasetTransportError("injected cold-cache failure")


def _manifest(count: int = 5) -> tuple[DatasetManifest, dict[str, bytes]]:
    source = DatasetSourceIdentity(
        repo_id="leafmoone/webdataset_danbooru",
        revision="master",
    )
    paths = (
        *VALIDATION_SHARD_PATHS,
        *(f"release/{index:06d}.tar" for index in range(max(0, count - 2))),
    )
    bodies = {path: bytes([index]) * 512 for index, path in enumerate(paths)}
    records = tuple(
        ShardRecord(
            path=path,
            bytes=len(body),
            upstream_sha256=hashlib.sha256(body).hexdigest(),
        )
        for path, body in bodies.items()
    )
    return DatasetManifest.from_shards(source, records), bodies


def _identity(
    manifest: DatasetManifest, *, worker_count: int = 2
) -> DataServiceSessionIdentity:
    return DataServiceSessionIdentity(
        manifest_id=manifest_sha256(manifest),
        worker_count=worker_count,
    )


def _selection(manifest: DatasetManifest) -> ValidationSelection:
    return select_validation_shards(manifest)


def _service(
    root: Path,
    manifest: DatasetManifest,
    transport: _SlowTransport,
    *,
    mainset_name: str = "mainset.json",
    identity: DataServiceSessionIdentity | None = None,
    limits: DataServiceLimits | None = None,
    selection: ValidationSelection | None = None,
) -> DataSupplyService:
    cache = ShardCache(
        (root / "cache").absolute(),
        manifest,
        cast(ModelScopeDatasetTransport, transport),
        CacheQuota(2048, 4096),
    )
    return DataSupplyService(
        manifest,
        selection or _selection(manifest),
        cache,
        root / mainset_name,
        (root / "runtime" / "data-service.lock").absolute(),
        identity or _identity(manifest),
        limits or DataServiceLimits(2, 3, 2, 2),
    )


def test_persistent_full_manifest_order_is_bounded_and_rotates_atomically(
    tmp_path: Path,
) -> None:
    manifest, bodies = _manifest()
    transport = _SlowTransport(bodies)
    service = _service(tmp_path, manifest, transport)
    service.start()
    try:
        initial = service.mainset
        initial_order = tuple(row.path for row in initial.rows)
        excluded = set(_selection(manifest).shard_paths)
        training = {record.path for record in manifest.shards} - excluded
        assert tuple(row.ordinal for row in initial.rows) == tuple(range(3))
        assert set(initial_order) == training
        assert len(initial_order) == len(set(initial_order)) == 3
        assert initial.validation_selection_id == _selection(manifest).selection_id
        assert initial.excluded_shard_paths == _selection(manifest).shard_paths
        assert initial.cycle_index == 0
        assert {row.status for row in initial.rows} == {"pending"}

        first = service.lease(0)
        second = service.lease(1)
        assert first is not None and second is not None
        assert first.cycle_index == second.cycle_index == initial.cycle_index
        assert (first.record.path, second.record.path) == initial_order[:2]
        assert transport.max_active == 2
        assert service.stats.outstanding_leases == 2
        assert (
            service.stats.in_flight_downloads + service.stats.verified_ready_shards <= 3
        )
        assert service.mainset.active_paths == initial_order[:2]
        with pytest.raises(DataServiceError, match="identity"):
            service.acknowledge(first.lease_id, first.worker_id, "0" * 64)

        service.acknowledge(first.lease_id, first.worker_id, first.state_identity)
        service.acknowledge(second.lease_id, second.worker_id, second.state_identity)
        observed = [first.record.path, second.record.path]
        worker = 0
        while len(observed) < len(initial.rows):
            descriptor = service.lease(worker)
            assert descriptor is not None
            assert descriptor.cycle_index == initial.cycle_index
            observed.append(descriptor.record.path)
            service.acknowledge(
                descriptor.lease_id,
                descriptor.worker_id,
                descriptor.state_identity,
            )
            worker = 1 - worker

        rotated = service.mainset
        assert tuple(observed) == initial_order
        assert rotated.mainset_id != initial.mainset_id
        assert rotated.cycle_index == initial.cycle_index + 1
        assert rotated.shuffle_identity != initial.shuffle_identity
        assert {row.path for row in rotated.rows} == set(initial_order)
        assert {row.status for row in rotated.rows} == {"pending"}
        document = json.loads((tmp_path / "mainset.json").read_bytes())
        assert document["mainset_id"] == rotated.mainset_id
        assert document["cycle_index"] == rotated.cycle_index
        assert document["validation_selection_id"] == _selection(manifest).selection_id
        assert document["excluded_shard_paths"] == list(_selection(manifest).shard_paths)
        assert [row["ordinal"] for row in document["rows"]] == list(range(3))
        assert not tuple(tmp_path.glob(".mainset.json.*.tmp"))
        assert not tuple(tmp_path.glob(".mainset.json.*.rollback"))
    finally:
        service.close()

    reopened = _service(tmp_path, manifest, _SlowTransport(bodies))
    reopened.start()
    try:
        assert reopened.mainset.mainset_id == rotated.mainset_id
        assert reopened.mainset.cycle_index == rotated.cycle_index
        assert reopened.mainset.shuffle_identity == rotated.shuffle_identity
        assert reopened.mainset.rows == rotated.rows
        assert reopened.mainset.replayed_shards == 0
    finally:
        reopened.close()


def test_restart_replays_all_active_before_preparing_new_rows(tmp_path: Path) -> None:
    manifest, bodies = _manifest()
    first_service = _service(tmp_path, manifest, _SlowTransport(bodies))
    first_service.start()
    initial_id = first_service.mainset.mainset_id
    initial_order = tuple(row.path for row in first_service.mainset.rows)
    first = first_service.lease(0)
    second = first_service.lease(1)
    assert first is not None and second is not None
    assert first.cycle_index == second.cycle_index == 0
    first_service.close()

    for descriptor in (first, second):
        descriptor.local_path.unlink()

    restarted_transport = _SlowTransport(bodies)
    restarted = _service(tmp_path, manifest, restarted_transport)
    restarted.start()
    try:
        assert restarted.mainset.mainset_id == initial_id
        assert tuple(row.path for row in restarted.mainset.rows) == initial_order
        assert restarted.mainset.replayed_shards == 2
        assert restarted.recovery_pending == frozenset()
        assert restarted_transport.downloaded[:2] == [
            first.record.path,
            second.record.path,
        ]
        replayed_first = restarted.lease(0)
        replayed_second = restarted.lease(1)
        assert replayed_first is not None and replayed_second is not None
        assert replayed_first.cycle_index == replayed_second.cycle_index == 0
        assert (replayed_first.record.path, replayed_second.record.path) == (
            first.record.path,
            second.record.path,
        )
        replayed_shards = restarted.mainset.replayed_shards
        restarted.acknowledge(
            replayed_first.lease_id,
            replayed_first.worker_id,
            replayed_first.state_identity,
        )
        restarted.acknowledge(
            replayed_second.lease_id,
            replayed_second.worker_id,
            replayed_second.state_identity,
        )
        worker = 0
        while restarted.mainset.mainset_id == initial_id:
            descriptor = restarted.lease(worker)
            assert descriptor is not None
            assert descriptor.cycle_index == 0
            restarted.acknowledge(
                descriptor.lease_id,
                descriptor.worker_id,
                descriptor.state_identity,
            )
            worker = 1 - worker
        assert restarted.mainset.replayed_shards == replayed_shards
        assert restarted.mainset.cycle_index == 1
    finally:
        restarted.close()


def test_failed_final_rotation_keeps_last_row_active_for_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, bodies = _manifest(4)
    service = _service(tmp_path, manifest, _SlowTransport(bodies))
    service.start()
    first = service.lease(0)
    assert first is not None
    service.acknowledge(first.lease_id, first.worker_id, first.state_identity)
    last = service.lease(1)
    assert last is not None
    initial_id = service.mainset.mainset_id

    def fail_save(_state: object) -> None:
        raise DataServiceError("injected publication failure")

    monkeypatch.setattr(service.store, "save", fail_save)
    with pytest.raises(DataServiceError, match="injected"):
        service.acknowledge(last.lease_id, last.worker_id, last.state_identity)
    assert service.mainset.mainset_id == initial_id
    assert service.mainset.active_paths == (last.record.path,)
    service.close()

    restarted = _service(tmp_path, manifest, _SlowTransport(bodies))
    restarted.start()
    try:
        assert restarted.mainset.mainset_id == initial_id
        replayed = restarted.lease(0)
        assert replayed is not None
        assert replayed.record.path == last.record.path
    finally:
        restarted.close()


def test_committed_cleanup_failure_hard_fails_and_restart_cleans_residue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, bodies = _manifest(4)
    service = _service(tmp_path, manifest, _SlowTransport(bodies))
    service.start()
    first_path = service.mainset.rows[0].path
    original_unlink = Path.unlink
    failed = False

    def fail_first_rollback_unlink(
        path: Path, missing_ok: bool = False
    ) -> None:
        nonlocal failed
        if path.name.endswith(".rollback") and not failed:
            failed = True
            raise OSError("injected cleanup failure")
        original_unlink(path, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", fail_first_rollback_unlink)
    with pytest.raises(DataServiceStateCommittedError, match="committed"):
        service.lease(0)
    service.close()
    assert tuple(tmp_path.glob(".mainset.json.*.rollback"))

    restarted = _service(tmp_path, manifest, _SlowTransport(bodies))
    restarted.start()
    try:
        assert not tuple(tmp_path.glob(".mainset.json.*.rollback"))
        assert restarted.mainset.active_paths == (first_path,)
        descriptor = restarted.lease(0)
        assert descriptor is not None
        assert descriptor.record.path == first_path
    finally:
        restarted.close()


def test_worker_count_drift_fails_without_reordering_mainset(tmp_path: Path) -> None:
    manifest, bodies = _manifest(5)
    original = _service(tmp_path, manifest, _SlowTransport(bodies))
    original.start()
    mainset_id = original.mainset.mainset_id
    order = tuple(row.path for row in original.mainset.rows)
    original.close()

    drifted = _service(
        tmp_path,
        manifest,
        _SlowTransport(bodies),
        identity=_identity(manifest, worker_count=1),
        limits=DataServiceLimits(1, 1, 1, 1),
    )
    with pytest.raises(DataServiceError, match="identity"):
        drifted.start()
    drifted.close()

    restored = _service(tmp_path, manifest, _SlowTransport(bodies))
    restored.start()
    try:
        assert restored.mainset.mainset_id == mainset_id
        assert tuple(row.path for row in restored.mainset.rows) == order
    finally:
        restored.close()


@pytest.mark.parametrize(
    "mutation",
    ["selection_id", "excluded_paths", "cycle_index", "old_schema"],
)
def test_mainset_rejects_validation_exclusion_identity_drift(
    tmp_path: Path,
    mutation: str,
) -> None:
    manifest, bodies = _manifest(5)
    original = _service(tmp_path, manifest, _SlowTransport(bodies))
    original.start()
    original.close()

    mainset_path = tmp_path / "mainset.json"
    document = json.loads(mainset_path.read_bytes())
    if mutation == "selection_id":
        document["validation_selection_id"] = "0" * 64
    elif mutation == "excluded_paths":
        document["excluded_shard_paths"] = list(
            reversed(document["excluded_shard_paths"])
        )
    elif mutation == "cycle_index":
        document["cycle_index"] = -1
    else:
        document["schema_version"] = 3
        del document["cycle_index"]
    mainset_path.write_text(json.dumps(document), encoding="utf-8")

    invalid = _service(tmp_path, manifest, _SlowTransport(bodies))
    with pytest.raises(DataServiceError, match="persistent mainset"):
        invalid.start()
    invalid.close()


def test_mainset_rejects_nonpending_row_after_pending_ordinal(
    tmp_path: Path,
) -> None:
    manifest, bodies = _manifest(5)
    service = _service(tmp_path, manifest, _SlowTransport(bodies))
    service.start()
    service.close()

    mainset_path = tmp_path / "mainset.json"
    document = json.loads(mainset_path.read_bytes())
    document["rows"][1]["status"] = "completed"
    mainset_path.write_text(json.dumps(document), encoding="utf-8")

    invalid = _service(tmp_path, manifest, _SlowTransport(bodies))
    with pytest.raises(DataServiceError, match="identity or rows"):
        invalid.start()
    invalid.close()


def test_trainer_lifecycle_fields_cannot_enter_service_identity_or_cli() -> None:
    manifest, _ = _manifest(2)
    identity = _identity(manifest)
    assert set(identity.as_dict()) == {"manifest_id", "worker_count"}

    from sakuramoon.cli.data_service import build_parser

    parser = build_parser()
    with pytest.raises(ValueError, match="invalid arguments"):
        parser.parse_args(
            [
                "--config",
                "config.toml",
                "--pass-index",
                "3",
                "--checkpoint-parent-id",
                "checkpoint",
                "--resume-snapshot",
                "snapshot.json",
            ]
        )


def test_resumed_client_leases_current_service_position_without_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, bodies = _manifest(5)
    first_service = _service(tmp_path, manifest, _SlowTransport(bodies))
    first_service.start()
    order = tuple(row.path for row in first_service.mainset.rows)
    first = first_service.lease(0)
    assert first is not None and first.record.path == order[0]
    first_service.acknowledge(
        first.lease_id, first.worker_id, first.state_identity
    )
    first_service.close()

    resumed_service = _service(tmp_path, manifest, _SlowTransport(bodies))
    socket_path = (
        Path(__file__).parents[3] / f".t044-resume-{os.getpid()}.sock"
    ).absolute()
    socket_path.unlink(missing_ok=True)
    server = DataServiceServer(
        resumed_service, socket_path, request_timeout_seconds=5.0
    )
    requests: list[frozenset[str]] = []
    dispatch = server._dispatch  # pyright: ignore[reportPrivateUsage]

    def capture(request: dict[str, object]) -> dict[str, object]:
        requests.append(frozenset(request))
        return dispatch(request)

    monkeypatch.setattr(server, "_dispatch", capture)
    stop = threading.Event()
    ready = threading.Event()
    errors: list[BaseException] = []

    def serve() -> None:
        try:
            server.serve(stop, ready_callback=ready.set)
        except BaseException as exc:  # noqa: BLE001 - surface server failure
            errors.append(exc)

    thread = threading.Thread(target=serve)
    thread.start()
    assert ready.wait(timeout=5.0)
    try:
        client = DataServiceClient(
            socket_path,
            worker_count=2,
            request_timeout_seconds=5.0,
        )
        resumed = client.lease(0)
        assert resumed is not None
        assert resumed.record.path == order[1]
        assert requests == [
            frozenset({"op", "protocol_version", "worker_count"}),
            frozenset({"op", "session_sha256", "worker_id"}),
        ]
    finally:
        stop.set()
        thread.join(timeout=5.0)
    assert not thread.is_alive()
    assert errors == []
    assert not socket_path.exists()


def test_server_ready_waits_for_cold_cache_worker_wave(tmp_path: Path) -> None:
    manifest, bodies = _manifest(5)
    transport = _GatedTransport(bodies)
    service = _service(tmp_path, manifest, transport)
    socket_path = (
        Path(__file__).parents[3] / f".d024-cold-ready-{os.getpid()}.sock"
    ).absolute()
    socket_path.unlink(missing_ok=True)
    server = DataServiceServer(
        service, socket_path, request_timeout_seconds=0.5
    )
    stop = threading.Event()
    ready = threading.Event()
    readiness: list[int] = []
    errors: list[BaseException] = []

    def mark_ready() -> None:
        readiness.append(service.stats.verified_ready_shards)
        ready.set()

    def serve() -> None:
        try:
            server.serve(stop, ready_callback=mark_ready)
        except BaseException as exc:  # noqa: BLE001 - surface server failure
            errors.append(exc)

    thread = threading.Thread(target=serve)
    thread.start()
    try:
        assert transport.first_wave_started.wait(timeout=5.0)
        deadline = time.monotonic() + 5.0
        while not socket_path.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert socket_path.exists()
        assert not ready.wait(timeout=0.05)

        transport.release.set()
        assert ready.wait(timeout=5.0)
        assert readiness[0] >= 2
        client = DataServiceClient(
            socket_path,
            worker_count=2,
            request_timeout_seconds=0.5,
        )
        first = client.lease(0)
        second = client.lease(1)
        assert first is not None and second is not None
    finally:
        stop.set()
        transport.release.set()
        thread.join(timeout=5.0)
        socket_path.unlink(missing_ok=True)
    assert not thread.is_alive()
    assert errors == []


def test_server_cold_cache_failure_never_signals_ready(tmp_path: Path) -> None:
    manifest, bodies = _manifest(5)
    service = _service(tmp_path, manifest, _FailingTransport(bodies))
    socket_path = (
        Path(__file__).parents[3] / f".d024-cold-failure-{os.getpid()}.sock"
    ).absolute()
    socket_path.unlink(missing_ok=True)
    ready = threading.Event()

    with pytest.raises(DataServiceError, match="verified lookahead failed"):
        DataServiceServer(
            service, socket_path, request_timeout_seconds=0.5
        ).serve(threading.Event(), ready_callback=ready.set)

    assert not ready.is_set()
    assert not socket_path.exists()


def test_mainset_has_one_process_owner(tmp_path: Path) -> None:
    manifest, bodies = _manifest(4)
    first = _service(tmp_path, manifest, _SlowTransport(bodies))
    second = _service(tmp_path, manifest, _SlowTransport(bodies))
    first.start()
    try:
        with pytest.raises(DataServiceError, match="another data service"):
            second.start()
    finally:
        second.close()
        first.close()


def test_explicit_runtime_lock_is_shared_across_mainset_paths(
    tmp_path: Path,
) -> None:
    manifest, bodies = _manifest(4)
    first = _service(
        tmp_path, manifest, _SlowTransport(bodies), mainset_name="first.json"
    )
    second = _service(
        tmp_path, manifest, _SlowTransport(bodies), mainset_name="second.json"
    )
    first.start()
    try:
        with pytest.raises(DataServiceError, match="another data service"):
            second.start()
    finally:
        second.close()
        first.close()


def test_concurrent_servers_leave_winner_socket_connected(tmp_path: Path) -> None:
    manifest, bodies = _manifest(4)
    first = _service(
        tmp_path, manifest, _SlowTransport(bodies), mainset_name="first.json"
    )
    second = _service(
        tmp_path, manifest, _SlowTransport(bodies), mainset_name="second.json"
    )
    socket_path = (
        Path(__file__).parents[3] / f".d024-ownership-{os.getpid()}.sock"
    ).absolute()
    socket_path.unlink(missing_ok=True)
    stop = threading.Event()
    barrier = threading.Barrier(2)
    ready: queue.Queue[str] = queue.Queue()
    errors: list[BaseException] = []

    def run(service: DataSupplyService) -> None:
        try:
            barrier.wait(timeout=5.0)
            DataServiceServer(
                service, socket_path, request_timeout_seconds=5.0
            ).serve(stop, ready_callback=lambda: ready.put("ready"))
        except BaseException as exc:  # noqa: BLE001 - assert the losing owner
            errors.append(exc)

    threads = [
        threading.Thread(target=run, args=(first,)),
        threading.Thread(target=run, args=(second,)),
    ]
    for thread in threads:
        thread.start()
    assert ready.get(timeout=5.0) == "ready"
    assert socket_path.exists()
    stop.set()
    for thread in threads:
        thread.join(timeout=5.0)
    assert all(not thread.is_alive() for thread in threads)
    assert len(errors) == 1
    assert isinstance(errors[0], DataServiceError)
    assert not socket_path.exists()


def test_service_start_removes_only_manifest_owned_partial(tmp_path: Path) -> None:
    manifest, bodies = _manifest(5)
    cache_root = tmp_path / "cache"
    owned = cache_root / manifest.shards[-1].path
    owned.parent.mkdir(parents=True)
    owned.with_name(f"{owned.name}.partial").write_bytes(b"partial")
    unrelated = cache_root / "unrelated.partial"
    unrelated.write_bytes(b"keep")
    service = _service(tmp_path, manifest, _SlowTransport(bodies))
    service.start()
    try:
        assert not owned.with_name(f"{owned.name}.partial").exists()
        assert unrelated.read_bytes() == b"keep"
    finally:
        service.close()


def test_client_module_has_no_transport_cache_state_or_snapshot_imports() -> None:
    source = (Path(__file__).parents[3] / "src/sakuramoon/data/client.py").read_text(
        encoding="utf-8"
    )
    assert "sakuramoon.data.modelscope" not in source
    assert "sakuramoon.data.cache" not in source
    assert "sakuramoon.data.state" not in source
    assert "fetch_dataset_shard" not in source
    assert "snapshot" not in source
    assert "hashlib" not in source

    script = """
import json
import sys
import sakuramoon.data.client
import sakuramoon.data.collate
import sakuramoon.data.pipeline
forbidden = [
    name for name in (
        'sakuramoon.data.cache',
        'sakuramoon.data.modelscope',
        'sakuramoon.data.state',
    ) if name in sys.modules
]
print(json.dumps(forbidden))
"""
    result = subprocess.run(
        (sys.executable, "-c", script),
        cwd=Path(__file__).parents[3],
        env={"PYTHONPATH": "src"},
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout) == []
