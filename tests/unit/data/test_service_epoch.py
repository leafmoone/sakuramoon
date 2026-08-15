from __future__ import annotations

# pyright: reportPrivateUsage=false
import json
import threading
from pathlib import Path

import pytest

from sakuramoon.data.manifest import (
    DatasetManifest,
    DatasetSourceIdentity,
    ShardRecord,
)
from sakuramoon.data.service import (
    DataServiceError,
    DataSupplyService,
    _QueueState,
    _QueueStore,
    _Row,
)
from sakuramoon.data.service_protocol import ShardLeaseDescriptor, canonical_json_bytes
from sakuramoon.data.validation import ValidationSelection


def _queue_store(tmp_path: Path) -> tuple[_QueueStore, ShardRecord]:
    validation = ShardRecord(path="data/validation.tar", bytes=1)
    training = ShardRecord(path="data/training.tar", bytes=1)
    manifest = DatasetManifest.from_shards(
        DatasetSourceIdentity(
            repo_id="leafmoone/webdataset_danbooru",
            revision="master",
        ),
        (validation, training),
    )
    selection = ValidationSelection(
        selection_id="validation",
        dataset_id=manifest.dataset_id,
        seed=44,
        shards=(validation,),
    )
    return _QueueStore(tmp_path / "mainset.json", manifest, selection), training


def test_legacy_queue_state_is_upgraded_without_resetting_epoch(
    tmp_path: Path,
) -> None:
    store, training = _queue_store(tmp_path)
    store.path.write_bytes(
        canonical_json_bytes(
            {
                "cycle": 3,
                "rows": [{"path": training.path, "status": "active"}],
                "schema_version": 1,
            }
        )
    )

    state = store.load()

    assert state is not None
    assert state == _QueueState(3, (_Row(training.path, "pending"),))
    store.save(state)
    document = json.loads(store.path.read_bytes())
    assert document == {
        "completed_epochs": 3,
        "current_epoch": 4,
        "cycle": 3,
        "rows": [{"path": training.path, "status": "pending"}],
        "schema_version": 2,
    }


def test_queue_state_rejects_inconsistent_epoch_metadata(tmp_path: Path) -> None:
    store, training = _queue_store(tmp_path)
    store.path.write_bytes(
        canonical_json_bytes(
            {
                "completed_epochs": 1,
                "current_epoch": 3,
                "cycle": 2,
                "rows": [{"path": training.path, "status": "pending"}],
                "schema_version": 2,
            }
        )
    )

    with pytest.raises(DataServiceError, match="could not be loaded"):
        store.load()


def test_last_ack_persists_and_logs_completed_epoch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    store, training = _queue_store(tmp_path)
    state = _QueueState(0, (_Row(training.path, "active"),))
    store.save(state)
    descriptor = ShardLeaseDescriptor(
        lease_id="lease-1",
        worker_id=0,
        cycle_index=0,
        state_revision=0,
        record=training,
        local_path=(tmp_path / "training.tar").absolute(),
    )
    service = object.__new__(DataSupplyService)
    service.store = store
    service._state = state
    service._started = True
    service._closed = False
    service._lock = threading.RLock()
    service._revision = 0
    service._outstanding = {descriptor.lease_id: descriptor}
    service._worker_leases = {descriptor.worker_id: descriptor.lease_id}

    def no_schedule(_service: DataSupplyService) -> None:
        pass

    monkeypatch.setattr(DataSupplyService, "_schedule_lookahead", no_schedule)

    service.acknowledge(
        descriptor.lease_id,
        descriptor.worker_id,
        descriptor.state_revision,
    )

    document = json.loads(store.path.read_bytes())
    assert document == {
        "completed_epochs": 1,
        "current_epoch": 2,
        "cycle": 1,
        "rows": [{"path": training.path, "status": "pending"}],
        "schema_version": 2,
    }
    assert service.state.completed_epochs == 1
    assert service.state.current_epoch == 2
    output = capsys.readouterr().out
    assert "数据 epoch 完成: epoch=1, completed_epochs=1, shards=1" in output
    assert "已重建 epoch=2 队列" in output


def test_epoch_log_is_not_emitted_when_metadata_commit_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    store, training = _queue_store(tmp_path)
    state = _QueueState(0, (_Row(training.path, "active"),))
    store.save(state)
    descriptor = ShardLeaseDescriptor(
        lease_id="lease-1",
        worker_id=0,
        cycle_index=0,
        state_revision=0,
        record=training,
        local_path=(tmp_path / "training.tar").absolute(),
    )
    service = object.__new__(DataSupplyService)
    service.store = store
    service._state = state
    service._started = True
    service._closed = False
    service._lock = threading.RLock()
    service._revision = 0
    service._outstanding = {descriptor.lease_id: descriptor}
    service._worker_leases = {descriptor.worker_id: descriptor.lease_id}

    def no_schedule(_service: DataSupplyService) -> None:
        pass

    monkeypatch.setattr(DataSupplyService, "_schedule_lookahead", no_schedule)

    def fail_rollover(_candidate: _QueueState) -> None:
        raise DataServiceError("forced rollover failure")

    monkeypatch.setattr(store, "save", fail_rollover)

    with pytest.raises(DataServiceError, match="forced rollover failure"):
        service.acknowledge(
            descriptor.lease_id,
            descriptor.worker_id,
            descriptor.state_revision,
        )

    assert service.state == state
    assert service._outstanding == {descriptor.lease_id: descriptor}
    assert service._worker_leases == {descriptor.worker_id: descriptor.lease_id}
    document = json.loads(store.path.read_bytes())
    assert document["cycle"] == 0
    assert document["rows"] == [{"path": training.path, "status": "active"}]
    output = capsys.readouterr().out
    assert "完成分片" not in output
    assert "数据 epoch 完成" not in output
