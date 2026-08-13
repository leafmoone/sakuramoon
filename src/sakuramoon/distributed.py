"""Fail-fast control-plane coordination for distributed ranks."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections.abc import Callable, Sequence
from datetime import timedelta
from typing import Protocol, TypeVar, cast

from torch import distributed


class DistributedProgressError(RuntimeError):
    """A distributed rank failed or stopped making observable progress."""


class _Store(Protocol):
    def set(self, key: str, value: str) -> None: ...

    def check(self, keys: Sequence[str]) -> bool: ...

    def get(self, key: str) -> bytes: ...


_T = TypeVar("_T")
_STATUS_RUNNING = "running"
_STATUS_COMPLETE = "complete"
_STATUS_FAILED = "failed"
_VALID_STATUSES = frozenset(
    {_STATUS_RUNNING, _STATUS_COMPLETE, _STATUS_FAILED}
)


class DistributedProgress:
    """Coordinate long rank-local work without occupying an NCCL collective."""

    def __init__(
        self,
        store: _Store | None,
        *,
        namespace: str,
        rank: int,
        world_size: int,
        stall_timeout: timedelta = timedelta(minutes=5),
        poll_interval_seconds: float = 0.1,
        heartbeat_interval_seconds: float = 30.0,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if type(namespace) is not str or not namespace:
            raise ValueError("distributed progress namespace must be nonempty")
        if (
            type(rank) is not int
            or type(world_size) is not int
            or world_size <= 0
            or not 0 <= rank < world_size
        ):
            raise ValueError("distributed progress topology is invalid")
        timeout_seconds = stall_timeout.total_seconds()
        if timeout_seconds <= 0.0:
            raise ValueError("distributed progress timeout must be positive")
        if (
            type(poll_interval_seconds) is not float
            or poll_interval_seconds <= 0.0
            or poll_interval_seconds > timeout_seconds
            or type(heartbeat_interval_seconds) is not float
            or heartbeat_interval_seconds <= 0.0
            or heartbeat_interval_seconds >= timeout_seconds
        ):
            raise ValueError("distributed progress intervals are invalid")
        if not callable(monotonic) or not callable(sleep):
            raise TypeError("distributed progress clock controls must be callable")
        if world_size > 1 and store is None:
            raise ValueError("distributed progress store is required")
        self._store = store
        self.namespace = namespace
        self.rank = rank
        self.world_size = world_size
        self.stall_timeout_seconds = timeout_seconds
        self.poll_interval_seconds = poll_interval_seconds
        self.heartbeat_interval_seconds = heartbeat_interval_seconds
        self._monotonic = monotonic
        self._sleep = sleep
        self._prefix = (
            "sakuramoon-progress/"
            + hashlib.sha256(namespace.encode("utf-8")).hexdigest()
        )
        self._sequences: dict[str, int] = {}
        self._publish_lock = threading.Lock()

    @classmethod
    def from_default_store(
        cls,
        *,
        namespace: str,
        rank: int,
        world_size: int,
        stall_timeout: timedelta = timedelta(minutes=5),
    ) -> DistributedProgress:
        if world_size == 1:
            return cls(
                None,
                namespace=namespace,
                rank=rank,
                world_size=world_size,
                stall_timeout=stall_timeout,
            )
        if not distributed.is_available() or not distributed.is_initialized():
            raise DistributedProgressError(
                "distributed progress requires an initialized process group"
            )
        from torch.distributed.distributed_c10d import (  # pyright: ignore[reportPrivateUsage]
            _get_default_store,
        )

        return cls(
            cast(_Store, _get_default_store()),
            namespace=namespace,
            rank=rank,
            world_size=world_size,
            stall_timeout=stall_timeout,
        )

    @staticmethod
    def _require_stage(stage: str) -> str:
        if type(stage) is not str or not stage:
            raise ValueError("distributed progress stage must be nonempty")
        return stage

    def _key(self, stage: str, rank: int) -> str:
        digest = hashlib.sha256(stage.encode("utf-8")).hexdigest()
        return f"{self._prefix}/{digest}/rank-{rank}"

    def _publish(self, stage: str, status: str, detail: str) -> None:
        stage = self._require_stage(stage)
        if status not in _VALID_STATUSES:
            raise ValueError("distributed progress status is invalid")
        if type(detail) is not str:
            raise TypeError("distributed progress detail must be a string")
        if self._store is None:
            return
        with self._publish_lock:
            sequence = self._sequences.get(stage, 0) + 1
            self._sequences[stage] = sequence
            self._store.set(
                self._key(stage, self.rank),
                json.dumps(
                    {
                        "stage": stage,
                        "rank": self.rank,
                        "sequence": sequence,
                        "status": status,
                        "detail": detail[:2000],
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ),
            )

    def heartbeat(self, stage: str, detail: str) -> None:
        self._publish(stage, _STATUS_RUNNING, detail)

    def complete(self, stage: str, detail: str = "complete") -> None:
        self._publish(stage, _STATUS_COMPLETE, detail)

    def fail(self, stage: str, error: Exception) -> None:
        self._publish(
            stage,
            _STATUS_FAILED,
            f"{type(error).__name__}: {error}",
        )

    def _run_with_heartbeat(self, stage: str, operation: Callable[[], _T]) -> _T:
        stage = self._require_stage(stage)
        if not callable(operation):
            raise TypeError("distributed progress operation must be callable")
        stop = threading.Event()
        heartbeat_errors: list[Exception] = []
        started = self._monotonic()
        self.heartbeat(stage, "started")

        def publish_heartbeats() -> None:
            while not stop.wait(self.heartbeat_interval_seconds):
                try:
                    self.heartbeat(
                        stage,
                        f"running elapsed={self._monotonic() - started:.1f}s",
                    )
                except Exception as error:  # noqa: BLE001 - surfaced after operation
                    heartbeat_errors.append(error)
                    return

        thread = threading.Thread(
            target=publish_heartbeats,
            name=f"sakuramoon-progress-rank-{self.rank}",
            daemon=True,
        )
        thread.start()
        try:
            result = operation()
        except Exception as error:
            stop.set()
            thread.join()
            try:
                self.fail(stage, error)
            except Exception as publish_error:  # noqa: BLE001 - preserve both failures
                raise BaseExceptionGroup(
                    "distributed operation and failure publication failed",
                    [error, publish_error],
                ) from None
            raise
        stop.set()
        thread.join()
        if heartbeat_errors:
            error = DistributedProgressError(
                f"distributed stage {stage!r} heartbeat publication failed"
            )
            try:
                self.fail(stage, error)
            except Exception as publish_error:  # noqa: BLE001 - preserve both failures
                raise BaseExceptionGroup(
                    "heartbeat and failure publication failed",
                    [*heartbeat_errors, publish_error],
                ) from None
            raise error from heartbeat_errors[0]
        self.complete(stage)
        return result

    def _wait(self, stage: str, ranks: tuple[int, ...]) -> None:
        stage = self._require_stage(stage)
        if self._store is None:
            return
        if (
            not ranks
            or len(set(ranks)) != len(ranks)
            or any(
                type(rank) is not int or not 0 <= rank < self.world_size
                for rank in ranks
            )
        ):
            raise ValueError("distributed progress wait ranks are invalid")
        observed: dict[int, bytes] = {}
        sequences: dict[int, int] = {}
        complete: set[int] = set()
        started = self._monotonic()
        last_progress = {rank: started for rank in ranks}
        while True:
            for rank in ranks:
                key = self._key(stage, rank)
                if not self._store.check([key]):
                    continue
                raw = self._store.get(key)
                if raw == observed.get(rank):
                    continue
                observed[rank] = raw
                last_progress[rank] = self._monotonic()
                try:
                    raw_document: object = json.loads(raw.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise DistributedProgressError(
                        f"distributed stage {stage!r} rank {rank} published invalid JSON"
                    ) from error
                if type(raw_document) is not dict:
                    raise DistributedProgressError(
                        f"distributed stage {stage!r} rank {rank} status is invalid"
                    )
                document = cast(dict[str, object], raw_document)
                status = document.get("status")
                sequence = document.get("sequence")
                detail = document.get("detail")
                if (
                    document.get("stage") != stage
                    or document.get("rank") != rank
                    or status not in _VALID_STATUSES
                    or type(sequence) is not int
                    or sequence <= sequences.get(rank, 0)
                    or type(detail) is not str
                ):
                    raise DistributedProgressError(
                        f"distributed stage {stage!r} rank {rank} status is invalid"
                    )
                sequences[rank] = sequence
                if status == _STATUS_FAILED:
                    raise DistributedProgressError(
                        f"distributed stage {stage!r} failed on rank {rank}: {detail}"
                    )
                if status == _STATUS_COMPLETE:
                    complete.add(rank)
                else:
                    complete.discard(rank)
            if len(complete) == len(ranks):
                return
            now = self._monotonic()
            stalled = tuple(
                rank
                for rank in ranks
                if rank not in complete
                and now - last_progress[rank] >= self.stall_timeout_seconds
            )
            if stalled:
                detail = ", ".join(
                    f"rank {rank}"
                    + (
                        " has not published status"
                        if rank not in observed
                        else " stopped publishing progress"
                    )
                    for rank in stalled
                )
                raise DistributedProgressError(
                    f"distributed stage {stage!r} made no rank progress for "
                    f"{self.stall_timeout_seconds:.1f}s: {detail}"
                )
            remaining = min(
                self.stall_timeout_seconds - (now - last_progress[rank])
                for rank in ranks
                if rank not in complete
            )
            self._sleep(min(self.poll_interval_seconds, remaining))

    def run_all(self, stage: str, operation: Callable[[], _T]) -> _T:
        """Run local work on every rank, then wait for all ranks to finish."""

        result = self._run_with_heartbeat(stage, operation)
        self._wait(stage, tuple(range(self.world_size)))
        return result

    def run_on_rank(
        self,
        stage: str,
        owner_rank: int,
        operation: Callable[[], _T],
    ) -> _T | None:
        """Run work on one rank while peers observe its Store heartbeat."""

        if type(owner_rank) is not int or not 0 <= owner_rank < self.world_size:
            raise ValueError("distributed progress owner rank is invalid")
        if not callable(operation):
            raise TypeError("distributed progress operation must be callable")
        result: _T | None = None
        if self.rank == owner_rank:
            result = self._run_with_heartbeat(stage, operation)
        self._wait(stage, (owner_rank,))
        return result

    def synchronize(self, stage: str) -> None:
        """Synchronize ranks through the control-plane Store, never NCCL/RCCL."""

        stage = self._require_stage(stage)
        self.complete(stage, "rank reached synchronization point")
        self._wait(stage, tuple(range(self.world_size)))


__all__ = ["DistributedProgress", "DistributedProgressError"]
