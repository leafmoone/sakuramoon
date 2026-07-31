"""Bucketed dense-Qwen batches with bounded DataLoader prefetch."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, replace

import torch
from torch.utils.data import DataLoader, IterableDataset

from sakuramoon.data.pipeline import (
    ImageAudit,
    PipelineSample,
    PipelineSampleError,
    RngIdentity,
    WebDatasetPipeline,
)
from sakuramoon.data.state import SingleProcessShardCoordinator


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
    releases: tuple[str, ...]
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
        input_ids[row, :length] = torch.tensor(sample.caption.input_ids, dtype=torch.long)
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
        releases=tuple(sample.release for sample in samples),
        audits=tuple(sample.audit for sample in samples),
        rng_identities=tuple(sample.rng for sample in samples),
    )


def bucketed_batches(
    samples: Iterable[PipelineSample],
    *,
    batch_size: int,
    drop_last: bool,
) -> Iterator[TrainingBatch]:
    if (
        type(batch_size) is not int
        or batch_size <= 0
        or type(drop_last) is not bool
    ):
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


def _build_batch_loader(
    dataset: BucketedBatchDataset,
    *,
    worker_count: int,
    ready_batches: int,
    pin_memory: bool,
) -> DataLoader[TrainingBatch]:
    """Build persistent workers with an exact divisible prefetch budget."""

    if (
        type(worker_count) is not int
        or worker_count <= 0
        or type(ready_batches) is not int
        or ready_batches < worker_count
        or ready_batches % worker_count
        or type(pin_memory) is not bool
    ):
        raise CollateError(
            "ready_batches must be a positive multiple of persistent worker_count"
        )
    return DataLoader(
        dataset,
        batch_size=None,
        num_workers=worker_count,
        persistent_workers=True,
        prefetch_factor=ready_batches // worker_count,
        pin_memory=pin_memory,
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
    """Drain each durable shard lease before publishing its completion."""

    if worker_count != 1:
        raise CollateError(
            "durable shard iteration currently requires exactly one worker"
        )
    if (
        type(shard_paths) is not tuple
        or not shard_paths
        or any(type(path) is not str or not path for path in shard_paths)
        or len(set(shard_paths)) != len(shard_paths)
    ):
        raise CollateError("durable shard iteration requires unique shard paths")
    for shard_path in shard_paths:
        with coordinator.lease(shard_path) as cached:
            if cached is None:
                continue
            if cached.fetched.relative_path != shard_path:
                raise PipelineSampleError("cache returned a different leased shard")
            shard_pipeline = pipeline._with_local_shards(  # pyright: ignore[reportPrivateUsage]
                (cached.fetched.path,)
            )
            dataset = BucketedBatchDataset(
                shard_pipeline,
                batch_size=batch_size,
                drop_last=drop_last,
            )
            loader = _build_batch_loader(
                dataset,
                worker_count=worker_count,
                ready_batches=ready_batches,
                pin_memory=pin_memory,
            )
            yield from loader


__all__ = [
    "BucketedBatchDataset",
    "CollateError",
    "TrainingBatch",
    "bucketed_batches",
    "collate_samples",
    "iter_leased_batches",
]
