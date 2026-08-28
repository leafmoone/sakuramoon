from __future__ import annotations

# pyright: reportPrivateUsage=false
import json
import threading
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import pytest

from sakuramoon.data.manifest import (
    DatasetManifest,
    DatasetSourceIdentity,
    ShardRecord,
)
from sakuramoon.data.service import (
    DataServiceError,
    DataServiceLimits,
    DataSupplyService,
    _QueueState,
    _QueueStore,
    _Row,
)
from sakuramoon.data.service_protocol import ShardLeaseDescriptor, canonical_json_bytes
from sakuramoon.data.validation import ValidationSelection


def _queue_store(tmp_path: Path):
    validation = ShardRecord(path="data/validation.tar", bytes=1)
    first = ShardRecord(path="data/first.tar", bytes=1)
    second = ShardRecord(path="data/second.tar", bytes=1)
    manifest = DatasetManifest.from_shards(
        DatasetSourceIdentity(
            repo_id="leafmoone/webdataset_danbooru_v2",
            revision="master",
        ),
        (validation, first, second),
    )
    selection = ValidationSelection(
        selection_id="validation",
        dataset_id=manifest.dataset_id,
        seed=44,
        shards=(validation,),
    )
    return (
        _QueueStore(tmp_path / "mainset.json", manifest, selection),
        manifest,
        first,
        second,
    )


def _service_at_boundary(
    tmp_path: Path,
    store: _QueueStore,
    manifest: DatasetManifest,
    first: ShardRecord,
    second: ShardRecord,
    monkeypatch: pytest.MonkeyPatch,
):
    """A service at the epoch boundary: one shard completed, one shard
    still leased (active), zero pending — the deadlock signature."""
    state = _QueueState(
        0, (_Row(first.path, "completed"), _Row(second.path, "active"))
    )
    store.save(state)
    active_descriptor = ShardLeaseDescriptor(
        lease_id="lease-active",
        worker_id=0,
        cycle_index=0,
        state_revision=0,
        record=second,
        local_path=(tmp_path / "second.tar").absolute(),
    )
    service = object.__new__(DataSupplyService)
    service.store = store
    service.manifest = manifest
    service.identity = SimpleNamespace(worker_count=16)
    service.limits = DataServiceLimits(
        download_concurrency=1,
        verified_shard_lookahead=2,
        lease_channel_capacity=4,
        ack_channel_capacity=4,
    )
    service._state = state
    service._started = True
    service._closed = False
    service._lock = threading.RLock()
    service._revision = 0
    service._outstanding = {active_descriptor.lease_id: active_descriptor}
    service._worker_leases = {0: active_descriptor.lease_id}
    service._retired_leases = OrderedDict()

    def no_schedule(_service: DataSupplyService) -> None:
        pass

    monkeypatch.setattr(DataSupplyService, "_schedule_lookahead", no_schedule)
    monkeypatch.setattr(
        service,
        "_take_ready",
        lambda: (
            first.path,
            SimpleNamespace(
                fetched=SimpleNamespace(path=(tmp_path / "first.tar").absolute())
            ),
        ),
    )
    return service, active_descriptor


def _rows_by_path(document: dict) -> dict:
    return {row["path"]: row["status"] for row in document["rows"]}


def test_lease_forces_rollover_at_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    store, manifest, first, second = _queue_store(tmp_path)
    service, active = _service_at_boundary(
        tmp_path, store, manifest, first, second, monkeypatch
    )

    descriptor = service.lease(1)

    assert descriptor is not None
    assert descriptor.worker_id == 1
    assert descriptor.cycle_index == 1
    document = json.loads(store.path.read_bytes())
    assert document["cycle"] == 1
    assert document["completed_epochs"] == 1
    assert document["current_epoch"] == 2
    rows = _rows_by_path(document)
    assert rows[first.path] == "active"
    assert rows[second.path] == "pending"
    assert service._outstanding == {descriptor.lease_id: descriptor}
    assert service._worker_leases == {1: descriptor.lease_id}
    assert service._retired_leases[active.lease_id].path == second.path
    output = capsys.readouterr().out
    assert "epoch 边界强制完成: epoch=1, 退休未完成租约=1" in output
    assert "已重建 epoch=2 队列" in output


def test_late_ack_after_force_rollover_is_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    store, manifest, first, second = _queue_store(tmp_path)
    service, active = _service_at_boundary(
        tmp_path, store, manifest, first, second, monkeypatch
    )
    service.lease(1)
    capsys.readouterr()

    # The rank's worker eventually finishes the retired shard and ACKs it
    # with the original (grant-time) identity.
    service.acknowledge(active.lease_id, active.worker_id, active.state_revision)

    document = json.loads(store.path.read_bytes())
    assert document["cycle"] == 1
    assert _rows_by_path(document)[second.path] == "pending"
    output = capsys.readouterr().out
    assert "完成分片(幂等): data/second.tar" in output


def test_late_ack_with_wrong_identity_still_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, manifest, first, second = _queue_store(tmp_path)
    service, active = _service_at_boundary(
        tmp_path, store, manifest, first, second, monkeypatch
    )
    service.lease(1)

    with pytest.raises(DataServiceError, match="ACK does not match"):
        service.acknowledge(active.lease_id, active.worker_id, 5)
    with pytest.raises(DataServiceError, match="ACK does not match"):
        service.acknowledge(active.lease_id, 7, active.state_revision)


def test_ack_of_unknown_lease_still_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, manifest, first, second = _queue_store(tmp_path)
    service, _ = _service_at_boundary(
        tmp_path, store, manifest, first, second, monkeypatch
    )

    with pytest.raises(DataServiceError, match="ACK does not match"):
        service.acknowledge("missing-lease", 0, 0)


def test_lease_mid_epoch_does_not_force_rollover(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    store, manifest, first, second = _queue_store(tmp_path)
    service, _ = _service_at_boundary(
        tmp_path, store, manifest, first, second, monkeypatch
    )
    # Mid-epoch: the completed shard is still pending, so the queue is not
    # drained and the rollover must not be forced.
    service._state = _QueueState(
        0, (_Row(first.path, "pending"), _Row(second.path, "active"))
    )
    store.save(service._state)

    descriptor = service.lease(1)

    assert descriptor is not None
    assert descriptor.cycle_index == 0
    document = json.loads(store.path.read_bytes())
    assert document["cycle"] == 0
    rows = _rows_by_path(document)
    assert rows[first.path] == "active"
    assert rows[second.path] == "active"
    assert service._retired_leases == OrderedDict()
    assert "epoch 边界强制完成" not in capsys.readouterr().out


def test_wait_until_ready_forces_rollover_at_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, manifest, first, second = _queue_store(tmp_path)
    service, active = _service_at_boundary(
        tmp_path, store, manifest, first, second, monkeypatch
    )
    requested: list[int] = []
    monkeypatch.setattr(service, "_wait_for_ready_count", requested.append)

    assert service.wait_until_ready() is True

    assert requested == [2]
    document = json.loads(store.path.read_bytes())
    assert document["cycle"] == 1
    assert service._retired_leases[active.lease_id].path == second.path


def test_wait_until_ready_still_raises_when_truly_empty(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, manifest, first, second = _queue_store(tmp_path)
    service, _ = _service_at_boundary(
        tmp_path, store, manifest, first, second, monkeypatch
    )
    service._outstanding = {}
    service._worker_leases = {}
    service._state = _QueueState(
        0, (_Row(first.path, "completed"), _Row(second.path, "completed"))
    )
    store.save(service._state)

    with pytest.raises(DataServiceError, match="no remaining training shard"):
        service.wait_until_ready()


def test_v2_state_load_reclaims_active_rows_to_pending(
    tmp_path: Path,
) -> None:
    store, _manifest, first, second = _queue_store(tmp_path)
    store.path.write_bytes(
        canonical_json_bytes(
            {
                "completed_epochs": 0,
                "current_epoch": 1,
                "cycle": 0,
                "rows": [
                    {"path": first.path, "status": "completed"},
                    {"path": second.path, "status": "active"},
                ],
                "schema_version": 2,
            }
        )
    )

    state = store.load()

    assert state is not None
    assert state.rows == (
        _Row(first.path, "completed"),
        _Row(second.path, "pending"),
    )
