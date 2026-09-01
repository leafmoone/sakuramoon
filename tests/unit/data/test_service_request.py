"""Demand-directed shard request (streaming window driver, 2026-09-01).

The SR window driver pins shards in the trainer's slot-stream order, not
the queue order, so the service exposes request(worker_id, path): pin a
SPECIFIC shard (download it first when not ready), same lease/ACK
semantics as the queue-ordered lease.

Covered:
  * a not-ready shard is downloaded (executor submit) and leased once ready
    (row -> active, descriptor identity, ACK releases it);
  * an already-ready shard is leased immediately;
  * a completed-in-cycle shard is refused (the driver tracks cycles);
  * an unknown path is refused;
  * a worker holding a lease for another path is refused (serialization);
  * the same (worker, path) request is idempotent (returns the held lease);
  * a request deadline raises instead of hanging forever.
"""

from __future__ import annotations

# pyright: reportPrivateUsage=false
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from sakuramoon.data.cache import CachedShard
from sakuramoon.data.manifest import (
    DatasetManifest,
    DatasetSourceIdentity,
    ShardRecord,
)
from sakuramoon.data.modelscope import FetchedShard
from sakuramoon.data.service import (
    DataServiceError,
    DataSupplyService,
    _QueueStore,
)

_EXECUTORS: list[ThreadPoolExecutor] = []


@pytest.fixture(autouse=True)
def _shutdown_executors():
    yield
    for executor in _EXECUTORS:
        executor.shutdown(wait=False, cancel_futures=True)
    _EXECUTORS.clear()


def _cached(path: Path, rel: str, n_bytes: int) -> CachedShard:
    return CachedShard(
        fetched=FetchedShard(
            path=path, relative_path=rel, bytes=n_bytes, cache_hit=True
        ),
        evicted_paths=(),
        usage_bytes=n_bytes,
    )


def _service(tmp_path: Path) -> DataSupplyService:
    """A minimal running service over a 3-shard queue (SR_v2-shaped paths)."""
    shards = tuple(
        ShardRecord(path=f"data/1_2024/shard-00000{i}-p2-00.tar", bytes=800)
        for i in range(3)
    )
    manifest = DatasetManifest.from_shards(
        DatasetSourceIdentity(repo_id="leafmoone/SR_v2", revision="master"),
        shards,
    )
    store = _QueueStore(tmp_path / "mainset.json", manifest, None)
    state = store.new(0)
    service = object.__new__(DataSupplyService)
    service.manifest = manifest
    service.identity = SimpleNamespace(worker_count=8, dataset_id=manifest.dataset_id)
    service.limits = SimpleNamespace(
        lease_channel_capacity=8,
        ack_channel_capacity=16,
        verified_shard_lookahead=4,
    )
    service.store = store
    service.cache = SimpleNamespace()
    service._state = state
    service._started = True
    service._closed = False
    service._lock = threading.RLock()
    service._outstanding = {}
    service._worker_leases = {}
    service._ready = {}
    service._futures = {}
    service._revision = 0
    service._shutdown = threading.Event()
    service._external_stop = None
    service._auth_failures = {}
    service._auth_failures_since_success = 0
    service._recent_failures = []
    service._cooldown_until = 0.0
    service._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="test")
    _EXECUTORS.append(service._executor)
    return service


def _submit_fetch(service: DataSupplyService, tmp_path: Path) -> None:
    """Stand-in for the real cache.fetch: the executor submit resolves the
    path to a ready cached shard (written to a tmp file)."""

    def fetch(path: str, **_kwargs: object) -> CachedShard:
        p = tmp_path / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x" * 800)
        return _cached(p, path, 800)

    service._futures.clear()
    service.cache.fetch = fetch


def test_request_downloads_then_leases(tmp_path: Path) -> None:
    service = _service(tmp_path)
    _submit_fetch(service, tmp_path)

    rel = "data/1_2024/shard-000000-p2-00.tar"
    descriptor = service.request(worker_id=0, path=rel, timeout_seconds=30.0)

    assert descriptor.record.path == rel
    assert descriptor.worker_id == 0
    assert descriptor.local_path.is_file()

    def _row(p: str):
        return next(r for r in service.state.rows if r.path == p)

    assert _row(rel).status == "active"
    assert service._worker_leases[0] == descriptor.lease_id

    # ACK releases the lease and completes the row
    service.acknowledge(descriptor.lease_id, 0, descriptor.state_revision, rel)
    assert _row(rel).status == "completed"
    assert service._worker_leases == {}
    service._executor.shutdown(wait=False, cancel_futures=True)


def test_request_duplicate_ack_after_lost_roundtrip_is_noop(tmp_path: Path) -> None:
    """Protocol v5: a transport-retried ack whose first copy already
    landed (lease gone, row completed) is accepted as a no-op — the
    2026-09-01 salt9 index-pass incident (NFS state write exceeded the
    client socket timeout; the blind retry was rejected and killed the
    index worker thread)."""
    service = _service(tmp_path)
    _submit_fetch(service, tmp_path)

    rel = "data/1_2024/shard-000000-p2-00.tar"
    descriptor = service.request(worker_id=0, path=rel, timeout_seconds=30.0)
    service.acknowledge(descriptor.lease_id, 0, descriptor.state_revision, rel)

    # the duplicate: no lease to match, row already completed
    service.acknowledge(descriptor.lease_id, 0, descriptor.state_revision, rel)
    assert next(r.status for r in service.state.rows if r.path == rel) == "completed"
    service._executor.shutdown(wait=False, cancel_futures=True)


def test_ack_unknown_lease_and_pending_row_is_rejected(tmp_path: Path) -> None:
    """The no-op path is fail-closed: an unmatched ack for a row that is
    NOT completed still raises (a genuine protocol error)."""
    service = _service(tmp_path)
    _submit_fetch(service, tmp_path)
    rel = "data/1_2024/shard-000000-p2-00.tar"
    with pytest.raises(DataServiceError, match="does not match an active lease"):
        service.acknowledge("ghost-lease", 0, 0, rel)
    service._executor.shutdown(wait=False, cancel_futures=True)


def test_request_uses_ready_shard_without_download(tmp_path: Path) -> None:
    service = _service(tmp_path)
    rel = "data/1_2024/shard-000001-p2-00.tar"
    cached = _cached(tmp_path / rel, rel, 800)
    cached.fetched.path.parent.mkdir(parents=True, exist_ok=True)
    cached.fetched.path.write_bytes(b"x" * 800)
    service._ready[rel] = cached
    service._schedule_lookahead = lambda: None  # type: ignore[method-assign]

    descriptor = service.request(worker_id=1, path=rel, timeout_seconds=5.0)
    assert descriptor.record.path == rel
    assert service._futures == {}  # no download was scheduled
    assert next(r for r in service.state.rows if r.path == rel).status == "active"


def test_request_refuses_completed_shard(tmp_path: Path) -> None:
    service = _service(tmp_path)
    from sakuramoon.data.service import _Row

    rel = "data/1_2024/shard-000000-p2-00.tar"
    service._state = type(service._state)(
        service._state.cycle,
        (
            _Row(rel, "completed"),
            *service._state.rows[1:],
        ),
    )

    with pytest.raises(DataServiceError, match="completed in this cycle"):
        service.request(worker_id=0, path=rel, timeout_seconds=5.0)


def test_request_refuses_unknown_path(tmp_path: Path) -> None:
    service = _service(tmp_path)

    with pytest.raises(DataServiceError, match="absent from the queue"):
        service.request(worker_id=0, path="data/1_2024/shard-999999-p2-00.tar", timeout_seconds=5.0)


def test_request_refuses_worker_holding_another_lease(tmp_path: Path) -> None:
    service = _service(tmp_path)
    _submit_fetch(service, tmp_path)
    held = service.request(worker_id=2, path="data/1_2024/shard-000000-p2-00.tar", timeout_seconds=30.0)

    with pytest.raises(DataServiceError, match="already holds a lease"):
        service.request(worker_id=2, path="data/1_2024/shard-000001-p2-00.tar", timeout_seconds=5.0)
    # the first lease is untouched
    assert service._worker_leases[2] == held.lease_id


def test_request_same_worker_same_path_is_idempotent(tmp_path: Path) -> None:
    service = _service(tmp_path)
    _submit_fetch(service, tmp_path)
    rel = "data/1_2024/shard-000000-p2-00.tar"
    first = service.request(worker_id=3, path=rel, timeout_seconds=30.0)
    second = service.request(worker_id=3, path=rel, timeout_seconds=5.0)
    assert second.lease_id == first.lease_id


def test_request_deadline_raises(tmp_path: Path) -> None:
    service = _service(tmp_path)
    stop = threading.Event()

    def never_ready(_path: str, **_kwargs: object) -> CachedShard:
        # hang until the test releases it (a hard 60s wait would block the
        # non-daemon executor thread at interpreter shutdown)
        while not stop.wait(0.05):
            continue
        raise AssertionError("fetch must not complete before the deadline")

    service.cache.fetch = never_ready  # type: ignore[assignment]
    try:
        with pytest.raises(DataServiceError, match="shard request timed out"):
            service.request(
                worker_id=4,
                path="data/1_2024/shard-000000-p2-00.tar",
                timeout_seconds=0.75,
            )
    finally:
        stop.set()
        service._executor.shutdown(wait=True, cancel_futures=True)
