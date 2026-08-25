"""Image-bucketed batches with dynamic right-padding and bounded prefetch."""

from __future__ import annotations

import multiprocessing as mp
import os
import queue
import sys
import time
from collections.abc import Iterable, Iterator, MutableMapping
from dataclasses import dataclass, field, replace
from multiprocessing.queues import Queue as MultiprocessingQueue
from pathlib import Path
from typing import Any, Protocol, TypeVar, cast

import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from sakuramoon.data.caption import (
    CAPTION_DROPOUT_KEYS,
    CaptionDropoutCounts,
    ConditionRole,
    ConditionRouteCounts,
    ConditionSource,
    count_condition_routes,
)
from sakuramoon.data.manifest import ShardRecord
from sakuramoon.data.pipeline import (
    ImageAudit,
    PipelineSample,
    RngIdentity,
    WebDatasetPipeline,
)
from sakuramoon.data.serialize import SerializedCaption
from sakuramoon.data.service_protocol import (
    DataServiceSessionIdentity,
    ShardLeaseDescriptor,
)
from sakuramoon.data.spatial_crop import SpatialCropCounts, aggregate_spatial_crop
from sakuramoon.data.transparent_white import (
    TRANSPARENT_REJECTION_KEYS,
    TransparentWhiteCounts,
    aggregate_transparent_white,
)

_WORKER_CONTEXT = mp.get_context("spawn")
_MAX_TORCH_SEED = 2**64 - 1
_LEASE_RETRY_SECONDS = 0.25


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
    condition_token_indices: torch.Tensor
    condition_mask: torch.Tensor
    active_condition_sample_indices: torch.Tensor
    condition_sources: tuple[ConditionSource | None, ...]
    condition_roles: tuple[ConditionRole | None, ...]
    condition_routes: ConditionRouteCounts
    sample_ids: torch.Tensor
    target_height: int
    target_width: int
    dense_length: int
    use_null_condition: torch.Tensor
    all_condition_dropped: torch.Tensor
    dropout_hits: CaptionDropoutCounts
    source_shards: tuple[str, ...]
    audits: tuple[ImageAudit, ...]
    rng_identities: tuple[RngIdentity, ...]
    spatial_crop: SpatialCropCounts
    # Fixed-key transparent-white counters for the retained samples of this
    # batch.  Rejects are structurally zero here (rejected samples produce
    # no PipelineSample) and audit through the per-shard completion channel.
    transparent: TransparentWhiteCounts
    # Exact post-dropout/post-assembly captions are retained for periodic samples.
    captions: tuple[SerializedCaption, ...] = ()

    def pin_memory(self) -> TrainingBatch:
        return replace(
            self,
            images=self.images.pin_memory(),
            input_ids=self.input_ids.pin_memory(),
            attention_mask=self.attention_mask.pin_memory(),
            main_token_indices=self.main_token_indices.pin_memory(),
            main_mask=self.main_mask.pin_memory(),
            condition_token_indices=self.condition_token_indices.pin_memory(),
            condition_mask=self.condition_mask.pin_memory(),
            active_condition_sample_indices=(
                self.active_condition_sample_indices.pin_memory()
            ),
            sample_ids=self.sample_ids.pin_memory(),
            use_null_condition=self.use_null_condition.pin_memory(),
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


def _zero_transparent_rejections() -> dict[str, int]:
    """Zero-filled fixed-key reject table for a shard that rejected nothing."""

    return dict.fromkeys(TRANSPARENT_REJECTION_KEYS, 0)


@dataclass(frozen=True)
class _WorkerCompletion:
    worker_id: int
    worker_pid: int
    shard_path: str
    normal: bool
    error: str = ""
    # Fixed-key transparent-white reject counters of the finished shard.
    # Rejected samples never produce a PipelineSample, so these counters are
    # the reliable channel that carries the rejects from worker to parent.
    transparent_rejections: dict[str, int] = field(
        default_factory=_zero_transparent_rejections
    )

    def __post_init__(self) -> None:
        for key, value in self.transparent_rejections.items():
            if key not in TRANSPARENT_REJECTION_KEYS:
                raise CollateError(
                    f"unknown transparent rejection key: {key}"
                )
            if type(value) is not int or value < 0:
                raise CollateError(
                    f"transparent rejection count for {key} must be a "
                    "nonnegative integer"
                )
        if len(self.transparent_rejections) != len(TRANSPARENT_REJECTION_KEYS):
            raise CollateError(
                "transparent_rejections must carry exactly "
                f"{TRANSPARENT_REJECTION_KEYS}"
            )


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
        length_sort_window_batches: int,
        worker_count: int,
    ) -> None:
        super().__init__()
        if (
            type(batch_size) is not int
            or batch_size <= 0
            or type(drop_last) is not bool
            or type(length_sort_window_batches) is not int
            or not 1 <= length_sort_window_batches <= 8
            or type(worker_count) is not int
            or worker_count <= 0
        ):
            raise CollateError("persistent shard dataset fields are invalid")
        self.pipeline = pipeline
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.length_sort_window_batches = length_sort_window_batches
        self.worker_count = worker_count
        # A command slot per worker prevents unbounded shard prefetch.  The
        # completion queue has the same bound because each active shard emits
        # exactly one terminal message.
        self.multiprocessing_context = _WORKER_CONTEXT
        self.input_queues = tuple(
            cast(
                MultiprocessingQueue,
                self.multiprocessing_context.Queue(maxsize=1),
            )
            for _ in range(worker_count)
        )
        self.completion_queue = cast(
            MultiprocessingQueue,
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
        cross_shard_batcher = _LengthAwareBatcher(
            batch_size=self.batch_size,
            window_batches=self.length_sort_window_batches,
        )
        while True:
            command = self.input_queues[worker_id].get()
            if not isinstance(command, _ShardWork):
                raise CollateError("persistent worker received an invalid command")
            if command.stop:
                return
            shard_pipeline: WebDatasetPipeline | None = None
            try:
                shard_pipeline = self.pipeline._with_local_shards(  # pyright: ignore[reportPrivateUsage]
                    (command.local_path,),
                    (command.record,),
                    cycle_index=command.cycle_index,
                )
                samples = shard_pipeline._iter_paths(  # pyright: ignore[reportPrivateUsage]
                    (command.local_path,), (command.record,)
                )
                if self.drop_last:
                    # Keep decoded tails in the persistent worker so image/length
                    # grouping spans shard boundaries without retaining shard files.
                    for sample in samples:
                        for batch in cross_shard_batcher.add(sample):
                            yield _WorkerBatch(
                                worker_id, worker_pid, command.shard_path, batch
                            )
                else:
                    # Variable tail batches retain their historical per-shard
                    # semantics. Production training uses drop_last=true.
                    for batch in bucketed_batches(
                        samples,
                        batch_size=self.batch_size,
                        drop_last=False,
                        length_sort_window_batches=self.length_sort_window_batches,
                    ):
                        yield _WorkerBatch(
                            worker_id, worker_pid, command.shard_path, batch
                        )
            except BaseException as error:
                # Never let worker failures publish durable state.  The
                # completion message is bounded and only carries diagnostics.
                # Reject counters are best-effort here: a shard that crashed
                # mid-stream still reports the rejects seen so far, and an
                # uncreated shard pipeline reports the zero table.
                if shard_pipeline is not None:
                    try:
                        reject_counts = shard_pipeline.transparent_rejection_counts()
                    except BaseException:  # noqa: BLE001 - capture partial counters before re-raising
                        reject_counts = _zero_transparent_rejections()
                else:
                    reject_counts = _zero_transparent_rejections()
                self.completion_queue.put(
                    _WorkerCompletion(
                        worker_id=worker_id,
                        worker_pid=worker_pid,
                        shard_path=command.shard_path,
                        normal=False,
                        error=f"{type(error).__name__}: {error}",
                        transparent_rejections=reject_counts,
                    )
                )
                raise
            assert shard_pipeline is not None
            self.completion_queue.put(
                _WorkerCompletion(
                    worker_id=worker_id,
                    worker_pid=worker_pid,
                    shard_path=command.shard_path,
                    normal=True,
                    transparent_rejections=shard_pipeline.transparent_rejection_counts(),
                )
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
    completion_queue: MultiprocessingQueue,
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


def _active_condition_sample_indices(
    samples: tuple[PipelineSample, ...],
) -> torch.Tensor:
    active_samples: list[int] = []
    for row, sample in enumerate(samples):
        indices = sample.caption.condition_token_indices
        mask = sample.caption.condition_mask
        use_null = sample.caption.use_null_condition
        source = sample.caption.condition_source
        role = sample.caption.condition_role
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
                "condition token indices must be active positions in the serialized Qwen input"
            )
        if bool(indices) == use_null:
            raise CollateError(
                "condition token presence and null routing must be complementary"
            )
        if (source is None) != (role is None):
            raise CollateError("condition source and role must be present together")
        if (source is None) != use_null:
            raise CollateError("condition source and null routing must agree")
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
    key = (first.target_height, first.target_width)
    if any(
        (sample.target_height, sample.target_width) != key
        for sample in samples
    ):
        raise CollateError("batch samples must share one image bucket")
    if any(
        sample.image.shape != (3, first.target_height, first.target_width)
        or sample.image.dtype != torch.uint8
        for sample in samples
    ):
        raise CollateError("batch images must be RGB uint8 tensors at the target size")
    _validate_main_indices(samples)
    active_condition_sample_indices = _active_condition_sample_indices(samples)

    # Right-pad only to the longest serialized Qwen bucket in this image batch.
    # Valid token positions never move, and downstream attention masks remove the
    # extra positions before trainable DiT packing.
    dense_length = max(sample.caption.dense_length for sample in samples)
    input_ids = torch.full(
        (len(samples), dense_length), padding_token_id, dtype=torch.long
    )
    attention_mask = torch.zeros((len(samples), dense_length), dtype=torch.bool)
    for row, sample in enumerate(samples):
        length = len(sample.caption.input_ids)
        if length > sample.caption.dense_length:
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
    condition_indices, condition_mask = _index_tensor(
        tuple(sample.caption.condition_token_indices for sample in samples)
    )
    dropout_hits = dict.fromkeys(CAPTION_DROPOUT_KEYS, 0)
    for sample in samples:
        for key, hit in sample.caption.dropout_hits.as_mapping().items():
            dropout_hits[key] += int(hit)
    condition_sources: tuple[ConditionSource | None, ...] = tuple(
        sample.caption.condition_source for sample in samples
    )
    condition_roles: tuple[ConditionRole | None, ...] = tuple(
        sample.caption.condition_role for sample in samples
    )
    return TrainingBatch(
        images=torch.stack(tuple(sample.image for sample in samples)),
        input_ids=input_ids,
        attention_mask=attention_mask,
        main_token_indices=main_indices,
        main_mask=main_mask,
        main_token_lengths=main_token_lengths,
        condition_token_indices=condition_indices,
        condition_mask=condition_mask,
        active_condition_sample_indices=active_condition_sample_indices,
        condition_sources=condition_sources,
        condition_roles=condition_roles,
        condition_routes=count_condition_routes(condition_sources),
        sample_ids=torch.tensor(
            tuple(sample.sample_id for sample in samples), dtype=torch.long
        ),
        target_height=first.target_height,
        target_width=first.target_width,
        dense_length=dense_length,
        use_null_condition=torch.tensor(
            tuple(sample.caption.use_null_condition for sample in samples),
            dtype=torch.bool,
        ),
        all_condition_dropped=torch.tensor(
            tuple(sample.caption.all_condition_dropped for sample in samples),
            dtype=torch.bool,
        ),
        dropout_hits=CaptionDropoutCounts(**dropout_hits),
        source_shards=tuple(sample.source_shard for sample in samples),
        audits=tuple(sample.audit for sample in samples),
        rng_identities=tuple(sample.rng for sample in samples),
        spatial_crop=aggregate_spatial_crop(
            tuple(sample.audit for sample in samples)
        ),
        transparent=aggregate_transparent_white(samples),
        captions=tuple(sample.caption for sample in samples),
    )


class _LengthAwareBatcher:
    """Bounded sortish batching within each image shape.

    A window of N complete batches is sorted by declared Qwen dense length before
    collation. This limits padding while retaining a strict upper memory bound and
    avoiding the unbounded tails created by exact image-by-text bucket products.
    """

    def __init__(self, *, batch_size: int, window_batches: int) -> None:
        if (
            type(batch_size) is not int
            or batch_size <= 0
            or type(window_batches) is not int
            or not 1 <= window_batches <= 8
        ):
            raise CollateError("length-aware batcher fields are invalid")
        self.batch_size = batch_size
        self.window_batches = window_batches
        self._window_size = batch_size * window_batches
        self._pending: dict[tuple[int, int], list[PipelineSample]] = {}

    @staticmethod
    def _sort_key(sample: PipelineSample) -> tuple[int, int, int, int, str]:
        return (
            sample.caption.dense_length,
            sample.rng.cycle_index,
            sample.rng.caption_seed,
            sample.sample_id,
            sample.source_shard,
        )

    def _full_window(
        self, samples: list[PipelineSample]
    ) -> tuple[TrainingBatch, ...]:
        ordered = sorted(samples, key=self._sort_key)
        groups = [
            tuple(ordered[start : start + self.batch_size])
            for start in range(0, self._window_size, self.batch_size)
        ]
        # Do not emit monotonically short-to-long windows. The deterministic
        # rotation preserves replayability while distributing compute shapes.
        rotation = sum(sample.rng.caption_seed for sample in ordered) % len(groups)
        groups = groups[rotation:] + groups[:rotation]
        return tuple(collate_samples(group) for group in groups)

    def add(self, sample: PipelineSample) -> tuple[TrainingBatch, ...]:
        key = (sample.target_height, sample.target_width)
        bucket = self._pending.setdefault(key, [])
        bucket.append(sample)
        if len(bucket) < self._window_size:
            return ()
        if len(bucket) != self._window_size:
            raise CollateError("length-aware pending window exceeded its bound")
        del self._pending[key]
        return self._full_window(bucket)

    def finish(self, *, drop_last: bool) -> tuple[TrainingBatch, ...]:
        if type(drop_last) is not bool:
            raise CollateError("drop_last must be an exact boolean")
        if drop_last:
            self._pending.clear()
            return ()
        batches: list[TrainingBatch] = []
        for key in sorted(self._pending):
            ordered = sorted(self._pending[key], key=self._sort_key)
            for start in range(0, len(ordered), self.batch_size):
                batches.append(
                    collate_samples(tuple(ordered[start : start + self.batch_size]))
                )
        self._pending.clear()
        return tuple(batches)


def bucketed_batches(
    samples: Iterable[PipelineSample],
    *,
    batch_size: int,
    drop_last: bool,
    length_sort_window_batches: int = 1,
) -> Iterator[TrainingBatch]:
    if type(batch_size) is not int or batch_size <= 0 or type(drop_last) is not bool:
        raise CollateError("batch_size and drop_last are invalid")
    batcher = _LengthAwareBatcher(
        batch_size=batch_size,
        window_batches=length_sort_window_batches,
    )
    for sample in samples:
        yield from batcher.add(sample)
    yield from batcher.finish(drop_last=drop_last)


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


BatchItem = TypeVar("BatchItem")


def _build_batch_loader(
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
    length_sort_window_batches: int = 1,
    transparent_rejection_ledger: MutableMapping[str, int] | None = None,
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
    if transparent_rejection_ledger is not None and (
        set(transparent_rejection_ledger.keys()) != set(TRANSPARENT_REJECTION_KEYS)
    ):
        raise CollateError(
            "transparent_rejection_ledger must carry exactly "
            f"{TRANSPARENT_REJECTION_KEYS}"
        )

    dataset = _PersistentShardDataset(
        pipeline,
        batch_size=batch_size,
        drop_last=drop_last,
        length_sort_window_batches=length_sort_window_batches,
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
        if transparent_rejection_ledger is not None:
            # Normal completion finalizes the shard's reject counters; the
            # ledger is the durable parent-side aggregate (failed shards are
            # never retried in the same stream, so their partial counters
            # stay out of the running total).
            for key in TRANSPARENT_REJECTION_KEYS:
                transparent_rejection_ledger[key] += int(
                    completion.transparent_rejections[key]
                )
        client.acknowledge(descriptor)
        del queued[shard_path]
        available_workers.append(descriptor.worker_id)
        submit_available()

    def drain() -> Iterator[TrainingBatch]:
        try:
            submit_available()
            while True:
                if not queued:
                    if client.health():
                        return
                    time.sleep(_LEASE_RETRY_SECONDS)
                    submit_available()
                    continue
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
    "iter_service_batches",
]
