from __future__ import annotations

# pyright: reportPrivateUsage=false
import threading
from concurrent.futures import Future

import pytest

from sakuramoon.data.cache import CachedShard
from sakuramoon.data.modelscope import DatasetTransientError, DatasetTransportError
from sakuramoon.data.service import DataServiceError, DataSupplyService


def _service_with_failed_future(error: Exception) -> DataSupplyService:
    service = object.__new__(DataSupplyService)
    future: Future[CachedShard] = Future()
    future.set_exception(error)
    service._futures = {"data/shard.tar": future}
    service._ready = {}
    service._shutdown = threading.Event()
    service._external_stop = None
    return service


def test_transient_download_failure_is_left_pending_for_retry() -> None:
    service = _service_with_failed_future(DatasetTransientError("transfer ended early"))

    service._collect_completed()

    assert service._futures == {}
    assert service._ready == {}


def test_permanent_download_failure_keeps_exact_reason() -> None:
    service = _service_with_failed_future(DatasetTransportError("shard is unavailable"))

    with pytest.raises(
        DataServiceError,
        match=r"data/shard\.tar: shard is unavailable",
    ):
        service._collect_completed()
