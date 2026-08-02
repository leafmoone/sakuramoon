"""Bucketed dense-Qwen batches with bounded DataLoader prefetch."""

from __future__ import annotations

import multiprocessing as mp
import os
import queue
import sys
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, replace
from multiprocessing.queues import Queue as MultiprocessingQueue
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast

import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from sakuramoon.data.caption import (
    CAPTION_DROPOUT_KEYS,
    CaptionDropoutCounts,
)
from sakuramoon.data.manifest import ShardRecord
from sakuramoon.data.pipeline import (
    ImageAudit,
    PipelineSample,
    PipelineSampleError,
    RngIdentity,
    WebDatasetPipeline,
)
from sakuramoon.data.service_protocol import (
    DataServiceSessionIdentity,
    ShardLeaseDescriptor,
)

if TYPE_CHECKING:
    from sakuramoon.data.state import SingleProcessShardCoordinator


_WORKER_CONTEXT = mp.get_context("spawn")
_MAX_TORCH_SEED = 2**64 - 1


class CollateError(ValueError):
    """Samples cannot form one homogeneous training batch."""


@dataclass(frozen=True)
class TrainingBatch:
    images: torch.Tensor
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    main_token_indices: torch.Tensor
    main_mask: torch.Tensor
    main_token_lengths: tuple[int, ...]
    artist_token_indices: torch.Tensor
    artist_mask: torch.Tensor
    active_style_sample_indices: torch.Tensor
    sample_ids: torch.Tensor
    target_height: int
    target_width: int
    dense_length: int
    use_null_style: torch.Tensor
    all_condition_dropped: torch.Tensor
    dropout_hits: CaptionDropoutCounts
    source_shards: tuple[str, ...]
    audits: tuple[ImageAudit, ...]
    rng_identities: tuple[RngIdentity, ...]

    def pin_memory(self) -> TrainingBatch:
        return replace(
            self,
            images=self.images.pin_memory(),
            input_ids=self.input_ids.pin_memory(),
            attention_mask=self.attention_mask.pin_memory(),
            main_token_indices=self.main_token_indices.pin_memory(),
            main_mask=self.main_mask.pin_memory(),
            artist_token_indices=self.artist_token_indices.pin_memory(),
            artist_mask=self.artist_mask.pin_memory(),
            active_style_sample_indices=self.active_style_sample_indices.pin_memory(),
            sample_ids=self.sample_ids.pin_memory(),
            use_null_style=self.use_null_style.pin_memory(),
            all_condition_dropped=self.all_condition_dropped.pin_memory(),
        )


@dataclass(frozen=True)
class _ShardWork:
    """A parent-prepared shard handed to one persistent DataLoader worker."""

    shard_path: str
    local_path: Path
    record: ShardRecord
    cycle_index: int
    stop: bool = False

    def __post_init__(self) -> None:
        if type(self.cycle_index) is not int or self.cycle_index < 0:
            raise CollateError("shard work cycle_index is invalid")


@dataclass(frozen=True)
class _WorkerBatch:
    worker_id: int
    worker_pid: int
    shard_path: str
    batch: TrainingBatch

    def pin_memory(self) -> _WorkerBatch:
        return _WorkerBatch(
            self.worker_id,
            self.worker_pid,
            self.shard_path,
            self.batch.pin_memory(),
        )


@dataclass(frozen=True)
class _WorkerDone:
    worker_id: int
    worker_pid: int
    shard_path: str


@dataclass(frozen=True)
class _WorkerCompletion:
    worker_id: int
    worker_pid: int
    shard_path: str
    normal: bool
    error: str = ""


class _PersistentShardDataset(IterableDataset[_WorkerBatch | _WorkerDone]):
    """Command-driven dataset used by exactly the configured worker processes.

    The dataset deliberately receives only immutable, parent-prepared shard
    records and local paths.  It has no coordinator or cache reference, so a
    worker cannot publish state or evict a shard.  The parent controls command
    admission through the bounded input queue.
    """

    def __init__(
        self,
        pipeline: WebDatasetPipeline,
        *,
        batch_size: int,
        drop_last: bool,
        worker_count: int,
    ) -> None:
        super().__init__()
        if (
            type(batch_size) is not int
            or batch_size <= 0
            or type(drop_last) is not bool
            or type(worker_count) is not int
            or worker_count <= 0
        ):
            raise CollateError("persistent shard dataset fields are invalid")
        self.pipeline = pipeline
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.worker_count = worker_count
        # A command slot per worker prevents unbounded shard prefetch.  The
        # completion queue has the same bound because each active shard emits
        # exactly one terminal message.
        self.multiprocessing_context = _WORKER_CONTEXT
        self.input_queues = tuple(
            cast(
                MultiprocessingQueue[object],
                self.multiprocessing_context.Queue(maxsize=1),
            )
            for _ in range(worker_count)
        )
        self.completion_queue = cast(
            MultiprocessingQueue[object],
            self.multiprocessing_context.Queue(maxsize=worker_count),
        )
        self.input_queue_capacity = worker_count
        self.input_queue_capacity_per_worker = 1
        self.completion_queue_capacity = worker_count

    def submit(self, worker_id: int, command: _ShardWork) -> None:
        if (
            type(worker_id) is not int
            or not 0 <= worker_id < self.worker_count
            or command.stop
        ):
            raise CollateError("persistent worker submission is invalid")
        self.input_queues[worker_id].put(command)

    def stop(self, worker_id: int, record: ShardRecord) -> None:
        if type(worker_id) is not int or not 0 <= worker_id < self.worker_count:
            raise CollateError("persistent worker stop target is invalid")
        self.input_queues[worker_id].put_nowait(
            _ShardWork("", Path("."), record, 0, stop=True)
        )

    def __iter__(self) -> Iterator[_WorkerBatch | _WorkerDone]:
        info = get_worker_info()
        if info is None:
            raise CollateError("persistent shard dataset requires DataLoader workers")
        worker_id = info.id
        worker_pid = os.getpid()
        while True:
            command = self.input_queues[worker_id].get()
            if not isinstance(command, _ShardWork):
                raise CollateError("persistent worker received an invalid command")
            if command.stop:
                return
            try:
                shard_pipeline = self.pipeline._with_local_shards(  # pyright: ignore[reportPrivateUsage]
                    (command.local_path,),
                    (command.record,),
                    cycle_index=command.cycle_index,
                )
                for batch in bucketed_batches(
                    shard_pipeline._iter_paths(  # pyright: ignore[reportPrivateUsage]
                        (command.local_path,), (command.record,)
                    ),
                    batch_size=self.batch_size,
                    drop_last=self.drop_last,
                ):
                    yield _WorkerBatch(worker_id, worker_pid, command.shard_path, batch)
            except BaseException as error:
                # Never let worker failures publish durable state.  The
                # completion message is bounded and only carries diagnostics.
                self.completion_queue.put(
                    _WorkerCompletion(
                        worker_id=worker_id,
                        worker_pid=worker_pid,
                        shard_path=command.shard_path,
                        normal=False,
                        error=f"{type(error).__name__}: {error}",
                    )
                )
                raise
            self.completion_queue.put(
                _WorkerCompletion(worker_id, worker_pid, command.shard_path, True)
            )
            # This marker is ordered after all batches in this worker's
            # DataLoader output stream.  The parent waits for both this marker
            # and the completion-channel message before marking the shard.
            yield _WorkerDone(worker_id, worker_pid, command.shard_path)


def _shutdown_loader(
    loader: DataLoader[Any], *, suppress_worker_failure: bool = False
) -> None:
    """Stop persistent workers deterministically when a lease is interrupted."""

    iterator = getattr(loader, "_iterator", None)
    shutdown = getattr(iterator, "_shutdown_workers", None)
    if callable(shutdown):
        try:
            shutdown()
        except RuntimeError:
            if not suppress_worker_failure:
                raise


def _completion_for(
    completion_queue: MultiprocessingQueue[object],
    pending: dict[str, _WorkerCompletion],
    shard_path: str,
) -> _WorkerCompletion:
    message = pending.pop(shard_path, None)
    while message is None:
        try:
            candidate = completion_queue.get(timeout=10.0)
        except queue.Empty:
            raise CollateError(
                f"worker completion missing for shard {shard_path}"
            ) from None
        if not isinstance(candidate, _WorkerCompletion):
            raise CollateError("persistent worker completion channel is invalid")
        if candidate.shard_path == shard_path:
            message = candidate
        else:
            pending[candidate.shard_path] = candidate
    return message


def _index_tensor(
    values: tuple[tuple[int, ...], ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    width = max((len(item) for item in values), default=0)
    indices = torch.full((len(values), width), -1, dtype=torch.long)
    mask = torch.zeros((len(values), width), dtype=torch.bool)
    for row, item in enumerate(values):
        if item:
            indices[row, : len(item)] = torch.tensor(item, dtype=torch.long)
            mask[row, : len(item)] = True
    return indices, mask


def _validate_main_indices(samples: tuple[PipelineSample, ...]) -> None:
    for sample in samples:
        indices = sample.caption.main_token_indices
        mask = sample.caption.main_mask
        if (
            len(indices) != len(mask)
            or any(type(active) is not bool or not active for active in mask)
            or any(
                type(index) is not int
                or index < 0
                or index >= len(sample.caption.input_ids)
                for index in indices
            )
        ):
            raise CollateError(
                "main token indices must be active positions in the serialized Qwen input"
            )


def _active_style_sample_indices(samples: tuple[PipelineSample, ...]) -> torch.Tensor:
    active_samples: list[int] = []
    for row, sample in enumerate(samples):
        indices = sample.caption.artist_token_indices
        mask = sample.caption.artist_mask
        use_null = sample.caption.use_null_style
        if (
            type(use_null) is not bool
            or len(indices) != len(mask)
            or any(type(active) is not bool or not active for active in mask)
            or any(
                type(index) is not int
                or index < 0
                or index >= len(sample.caption.input_ids)
                for index in indices
            )
        ):
            raise CollateError(
                "Artist token indices must be active positions in the serialized Qwen input"
            )
        if bool(indices) == use_null:
            raise CollateError(
                "Artist token presence and null-style routing must be complementary"
            )
        if indices:
            active_samples.append(row)
    return torch.tensor(active_samples, dtype=torch.long)


def collate_samples(samples: tuple[PipelineSample, ...]) -> TrainingBatch:
    if not samples:
        raise CollateError("collate requires samples")
    first = samples[0]
    padding_token_id = first.padding_token_id
    if (
        type(padding_token_id) is not int
        or padding_token_id < 0
        or any(sample.padding_token_id != padding_token_id for sample in samples)
    ):
        raise CollateError("batch samples must share the framing padding token")
    key = (first.target_height, first.target_width, first.caption.dense_length)
    if any(
        (sample.target_height, sample.target_width, sample.caption.dense_length) != key
        for sample in samples
    ):
        raise CollateError("batch samples must share image and Qwen dense buckets")
    if any(
        sample.image.shape != (3, first.target_height, first.target_width)
        or sample.image.dtype != torch.uint8
        for sample in samples
    ):
        raise CollateError("batch images must be RGB uint8 tensors at the target size")
    _validate_main_indices(samples)
    active_style_sample_indices = _active_style_sample_indices(samples)

    dense_length = first.caption.dense_length
    input_ids = torch.full(
        (len(samples), dense_length), padding_token_id, dtype=torch.long
    )
    attention_mask = torch.zeros((len(samples), dense_length), dtype=torch.bool)
    for row, sample in enumerate(samples):
        length = len(sample.caption.input_ids)
        if length > dense_length:
            raise CollateError("serialized caption exceeds its dense bucket")
        input_ids[row, :length] = torch.tensor(
            sample.caption.input_ids, dtype=torch.long
        )
        attention_mask[row, :length] = True
    main_indices, main_mask = _index_tensor(
        tuple(sample.caption.main_token_indices for sample in samples)
    )
    main_token_lengths = tuple(
        len(sample.caption.main_token_indices) for sample in samples
    )
    artist_indices, artist_mask = _index_tensor(
        tuple(sample.caption.artist_token_indices for sample in samples)
    )
    dropout_hits = dict.fromkeys(CAPTION_DROPOUT_KEYS, 0)
    for sample in samples:
        for key, hit in sample.caption.dropout_hits.as_mapping().items():
            dropout_hits[key] += int(hit)
    return TrainingBatch(
        images=torch.stack(tuple(sample.image for sample in samples)),
        input_ids=input_ids,
        attention_mask=attention_mask,
        main_token_indices=main_indices,
        main_mask=main_mask,
        main_token_lengths=main_token_lengths,
        artist_token_indices=artist_indices,
        artist_mask=artist_mask,
        active_style_sample_indices=active_style_sample_indices,
        sample_ids=torch.tensor(
            tuple(sample.sample_id for sample in samples), dtype=torch.long
        ),
        target_height=first.target_height,
        target_width=first.target_width,
        dense_length=dense_length,
        use_null_style=torch.tensor(
            tuple(sample.caption.use_null_style for sample in samples), dtype=torch.bool
        ),
        all_condition_dropped=torch.tensor(
            tuple(sample.caption.all_condition_dropped for sample in samples),
            dtype=torch.bool,
        ),
        dropout_hits=CaptionDropoutCounts(**dropout_hits),
        source_shards=tuple(sample.source_shard for sample in samples),
        audits=tuple(sample.audit for sample in samples),
        rng_identities=tuple(sample.rng for sample in samples),
    )


def bucketed_batches(
    samples: Iterable[PipelineSample],
    *,
    batch_size: int,
    drop_last: bool,
) -> Iterator[TrainingBatch]:
    if type(batch_size) is not int or batch_size <= 0 or type(drop_last) is not bool:
        raise CollateError("batch_size and drop_last are invalid")
    pending: dict[tuple[int, int, int], list[PipelineSample]] = {}
    for sample in samples:
        key = (sample.target_height, sample.target_width, sample.caption.dense_length)
        bucket = pending.setdefault(key, [])
        bucket.append(sample)
        if len(bucket) == batch_size:
            yield collate_samples(tuple(bucket))
            del pending[key]
    if not drop_last:
        for key in sorted(pending):
            yield collate_samples(tuple(pending[key]))


class BucketedBatchDataset(IterableDataset[TrainingBatch]):
    def __init__(
        self,
        samples: IterableDataset[PipelineSample],
        *,
        batch_size: int,
        drop_last: bool,
    ) -> None:
        super().__init__()
        if (
            not isinstance(  # pyright: ignore[reportUnnecessaryIsInstance]
                samples, IterableDataset
            )
            or type(batch_size) is not int
            or batch_size <= 0
            or type(drop_last) is not bool
        ):
            raise CollateError("bucketed batch dataset fields are invalid")
        self.samples = samples
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self) -> Iterator[TrainingBatch]:
        return bucketed_batches(
            self.samples,
            batch_size=self.batch_size,
            drop_last=self.drop_last,
        )


def _build_batch_loader[BatchItem](
    dataset: IterableDataset[BatchItem],
    *,
    worker_count: int,
    ready_batches: int,
    pin_memory: bool,
    worker_seed: int,
    in_order: bool = True,
) -> DataLoader[BatchItem]:
    """Build persistent workers with an exact divisible prefetch budget."""

    if (
        type(worker_seed) is not int
        or worker_seed < 0
        or worker_seed > _MAX_TORCH_SEED
    ):
        raise CollateError("worker_seed must be an unsigned 64-bit integer")
    if (
        type(worker_count) is not int
        or worker_count <= 0
        or type(ready_batches) is not int
        or ready_batches < worker_count
        or ready_batches % worker_count
        or type(pin_memory) is not bool
        or type(in_order) is not bool
    ):
        raise CollateError(
            "ready_batches must be a positive multiple of persistent worker_count"
        )
    worker_generator = torch.Generator(device="cpu")
    worker_generator.manual_seed(worker_seed)
    return DataLoader(
        dataset,
        batch_size=None,
        num_workers=worker_count,
        persistent_workers=True,
        prefetch_factor=ready_batches // worker_count,
        pin_memory=pin_memory,
        in_order=in_order,
        multiprocessing_context=_WORKER_CONTEXT,
        generator=worker_generator,
    )


def iter_leased_batches(
    pipeline: WebDatasetPipeline,
    coordinator: SingleProcessShardCoordinator,
    shard_paths: tuple[str, ...],
    *,
    batch_size: int,
    worker_count: int,
    ready_batches: int,
    pin_memory: bool,
    drop_last: bool,
) -> Iterator[TrainingBatch]:
    """Drain durable shard leases through the configured persistent workers.

    The coordinator remains exclusively in the parent process.  Workers only
    consume parent-prepared local files and return batches plus terminal
    markers.  A shard is completed after its ordered done marker and bounded
    completion-channel record have both arrived.
    """

    if worker_count != coordinator.store.worker_count:
        raise CollateError(
            "durable shard iteration worker_count must exactly match the state "
            "worker_count; exactly one worker is only valid when schema v3 "
            "worker_count is one (no fallback)"
        )
    if (
        type(shard_paths) is not tuple
        or not shard_paths
        or any(type(path) is not str or not path for path in shard_paths)
        or len(set(shard_paths)) != len(shard_paths)
    ):
        raise CollateError("durable shard iteration requires unique shard paths")
    if type(batch_size) is not int or batch_size <= 0 or type(drop_last) is not bool:
        raise CollateError("batch_size and drop_last are invalid")

    requested = list(shard_paths)
    recovered = tuple(coordinator.state.active_shards)
    if any(path not in requested for path in recovered):
        raise CollateError(
            "all recovered active shards must be included for replay before new shards"
        )
    ordered_paths = list(dict.fromkeys((*recovered, *requested)))
    dataset = _PersistentShardDataset(
        pipeline,
        batch_size=batch_size,
        drop_last=drop_last,
        worker_count=worker_count,
    )
    loader = _build_batch_loader(
        dataset,
        worker_count=worker_count,
        ready_batches=ready_batches,
        pin_memory=pin_memory,
        worker_seed=pipeline.base_seed,
        in_order=False,
    )
    iterator = iter(loader)
    queued: dict[str, int] = {}
    available_workers = list(range(worker_count))
    completion_messages: dict[str, _WorkerCompletion] = {}
    next_index = 0

    def submit_available() -> None:
        nonlocal next_index
        while available_workers and next_index < len(ordered_paths):
            worker_id = available_workers.pop(0)
            shard_path = ordered_paths[next_index]
            next_index += 1
            cached = coordinator.prepare(shard_path)
            if cached is None:
                available_workers.append(worker_id)
                continue
            if cached.fetched.relative_path != shard_path:
                raise PipelineSampleError("cache returned a different leased shard")
            dataset.submit(
                worker_id,
                _ShardWork(
                    shard_path=shard_path,
                    local_path=cached.fetched.path,
                    record=coordinator.store.manifest.shard(shard_path),
                    cycle_index=pipeline.cycle_index,
                ),
            )
            queued[shard_path] = worker_id

    def finish_shard(shard_path: str) -> None:
        completion = _completion_for(
            dataset.completion_queue, completion_messages, shard_path
        )
        if not completion.normal:
            raise CollateError(
                f"persistent worker failed for shard {shard_path}: {completion.error}"
            )
        if shard_path not in queued:
            raise CollateError("persistent worker completed an unknown shard")
        worker_id = queued[shard_path]
        if completion.worker_id != worker_id:
            raise CollateError("persistent worker completion identity drifted")
        coordinator.mark_completed(shard_path)
        del queued[shard_path]
        available_workers.append(worker_id)
        submit_available()

    try:
        submit_available()
        while queued:
            item = next(iterator)
            if isinstance(item, _WorkerBatch):
                if queued.get(item.shard_path) != item.worker_id:
                    raise CollateError("persistent worker returned an unknown shard")
                yield item.batch
            elif isinstance(item, _WorkerDone):
                if queued.get(item.shard_path) != item.worker_id:
                    raise CollateError("persistent worker done identity drifted")
                finish_shard(item.shard_path)
            else:
                raise CollateError("persistent worker output channel is invalid")
    finally:
        # A stop command is best-effort.  _shutdown_workers also handles an
        # interrupted or failed worker, while active state remains replayable.
        for worker_id in range(worker_count):
            try:
                dataset.stop(worker_id, coordinator.store.manifest.shards[0])
            except (queue.Full, IndexError):
                continue
        _shutdown_loader(loader, suppress_worker_failure=sys.exception() is not None)


class DataLeaseClient(Protocol):
    identity: DataServiceSessionIdentity

    def health(self) -> bool: ...

    def lease(self, worker_id: int) -> ShardLeaseDescriptor | None: ...

    def acknowledge(self, descriptor: ShardLeaseDescriptor) -> None: ...


class ServiceBatchIterator(Iterator[TrainingBatch]):
    """Own a service DataLoader and expose its live result-queue depth."""

    def __init__(
        self,
        batches: Iterator[TrainingBatch],
        loader_iterator: object,
    ) -> None:
        self._batches = batches
        self._loader_iterator = loader_iterator
        self._closed = False

    def __iter__(self) -> ServiceBatchIterator:
        return self

    def __next__(self) -> TrainingBatch:
        if self._closed:
            raise StopIteration
        try:
            return next(self._batches)
        except StopIteration:
            self._closed = True
            raise

    def ready_batch_depth_snapshot(self) -> int:
        if self._closed:
            raise CollateError("live DataLoader ready-batch depth is unavailable")
        data_queue = getattr(self._loader_iterator, "_data_queue", None)
        qsize = getattr(data_queue, "qsize", None)
        if not callable(qsize):
            raise CollateError("live DataLoader ready-batch depth is unsupported")
        try:
            depth = qsize()
        except (NotImplementedError, OSError):
            raise CollateError(
                "live DataLoader ready-batch depth is unsupported"
            ) from None
        if type(depth) is not int or depth < 0:
            raise CollateError("live DataLoader ready-batch depth is invalid")
        return depth

    def close(self) -> None:
        if self._closed:
            return
        try:
            close = getattr(self._batches, "close", None)
            if callable(close):
                close()
        finally:
            self._closed = True


def iter_service_batches(
    pipeline: WebDatasetPipeline,
    client: DataLeaseClient,
    *,
    batch_size: int,
    worker_count: int,
    ready_batches: int,
    pin_memory: bool,
    drop_last: bool,
) -> ServiceBatchIterator:
    """Drain service-issued leases without owning service/cache/state in trainer."""

    if worker_count != client.identity.worker_count:
        raise CollateError(
            "data service worker_count must exactly match the configured topology"
        )
    if type(batch_size) is not int or batch_size <= 0 or type(drop_last) is not bool:
        raise CollateError("batch_size and drop_last are invalid")
    if client.health():
        raise CollateError("data service has no training lease available")

    dataset = _PersistentShardDataset(
        pipeline,
        batch_size=batch_size,
        drop_last=drop_last,
        worker_count=worker_count,
    )
    loader = _build_batch_loader(
        dataset,
        worker_count=worker_count,
        ready_batches=ready_batches,
        pin_memory=pin_memory,
        worker_seed=pipeline.base_seed,
        in_order=False,
    )
    iterator = iter(loader)
    queued: dict[str, ShardLeaseDescriptor] = {}
    available_workers = list(range(worker_count))
    completion_messages: dict[str, _WorkerCompletion] = {}
    stop_record: ShardRecord | None = None

    def submit_available() -> None:
        nonlocal stop_record
        remaining: list[int] = []
        for worker_id in available_workers:
            descriptor = client.lease(worker_id)
            if descriptor is None:
                remaining.append(worker_id)
                continue
            if descriptor.record.path in queued:
                raise CollateError("data service returned a duplicate active shard")
            dataset.submit(
                worker_id,
                _ShardWork(
                    shard_path=descriptor.record.path,
                    local_path=descriptor.local_path,
                    record=descriptor.record,
                    cycle_index=descriptor.cycle_index,
                ),
            )
            queued[descriptor.record.path] = descriptor
            stop_record = descriptor.record
        available_workers[:] = remaining

    def finish_shard(shard_path: str) -> None:
        completion = _completion_for(
            dataset.completion_queue, completion_messages, shard_path
        )
        descriptor = queued.get(shard_path)
        if descriptor is None:
            raise CollateError("persistent worker completed an unknown service lease")
        if not completion.normal:
            raise CollateError(
                f"persistent worker failed for shard {shard_path}: {completion.error}"
            )
        if completion.worker_id != descriptor.worker_id:
            raise CollateError("persistent worker service identity drifted")
        client.acknowledge(descriptor)
        del queued[shard_path]
        available_workers.append(descriptor.worker_id)
        submit_available()

    def drain() -> Iterator[TrainingBatch]:
        try:
            submit_available()
            while queued:
                item = next(iterator)
                if isinstance(item, _WorkerBatch):
                    descriptor = queued.get(item.shard_path)
                    if descriptor is None or descriptor.worker_id != item.worker_id:
                        raise CollateError(
                            "persistent worker returned an unknown service lease"
                        )
                    yield item.batch
                elif isinstance(item, _WorkerDone):
                    descriptor = queued.get(item.shard_path)
                    if descriptor is None or descriptor.worker_id != item.worker_id:
                        raise CollateError(
                            "persistent worker done service identity drifted"
                        )
                    finish_shard(item.shard_path)
                else:
                    raise CollateError("persistent worker output channel is invalid")
        finally:
            if stop_record is not None:
                for worker_id in range(worker_count):
                    try:
                        dataset.stop(worker_id, stop_record)
                    except queue.Full:
                        continue
            _shutdown_loader(loader, suppress_worker_failure=sys.exception() is not None)

    return ServiceBatchIterator(drain(), iterator)


__all__ = [
    "BucketedBatchDataset",
    "CollateError",
    "ServiceBatchIterator",
    "TrainingBatch",
    "bucketed_batches",
    "collate_samples",
    "iter_leased_batches",
    "iter_service_batches",
]
