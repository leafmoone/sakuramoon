"""Bucketed dense-Qwen batches with bounded DataLoader prefetch."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, replace

import torch
from torch.utils.data import DataLoader, IterableDataset

from sakuramoon.data.pipeline import ImageAudit, PipelineSample, RngIdentity


class CollateError(ValueError):
    """Samples cannot form one homogeneous training batch."""


@dataclass(frozen=True)
class TrainingBatch:
    images: torch.Tensor
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    main_token_indices: torch.Tensor
    main_mask: torch.Tensor
    artist_token_indices: torch.Tensor
    artist_mask: torch.Tensor
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


def collate_samples(
    samples: tuple[PipelineSample, ...], *, padding_token_id: int
) -> TrainingBatch:
    if not samples or type(padding_token_id) is not int or padding_token_id < 0:
        raise CollateError("collate requires samples and a valid padding token")
    first = samples[0]
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
    artist_indices, artist_mask = _index_tensor(
        tuple(sample.caption.artist_token_indices for sample in samples)
    )
    return TrainingBatch(
        images=torch.stack(tuple(sample.image for sample in samples)),
        input_ids=input_ids,
        attention_mask=attention_mask,
        main_token_indices=main_indices,
        main_mask=main_mask,
        artist_token_indices=artist_indices,
        artist_mask=artist_mask,
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
    padding_token_id: int,
    drop_last: bool,
) -> Iterator[TrainingBatch]:
    if type(batch_size) is not int or batch_size <= 0:
        raise CollateError("batch_size must be a positive integer")
    pending: dict[tuple[int, int, int], list[PipelineSample]] = {}
    for sample in samples:
        key = (sample.target_height, sample.target_width, sample.caption.dense_length)
        bucket = pending.setdefault(key, [])
        bucket.append(sample)
        if len(bucket) == batch_size:
            yield collate_samples(tuple(bucket), padding_token_id=padding_token_id)
            del pending[key]
    if not drop_last:
        for key in sorted(pending):
            yield collate_samples(tuple(pending[key]), padding_token_id=padding_token_id)


class BucketedBatchDataset(IterableDataset[TrainingBatch]):
    def __init__(
        self,
        samples: IterableDataset[PipelineSample],
        *,
        batch_size: int,
        padding_token_id: int,
        drop_last: bool,
    ) -> None:
        super().__init__()
        self.samples = samples
        self.batch_size = batch_size
        self.padding_token_id = padding_token_id
        self.drop_last = drop_last

    def __iter__(self) -> Iterator[TrainingBatch]:
        return bucketed_batches(
            self.samples,
            batch_size=self.batch_size,
            padding_token_id=self.padding_token_id,
            drop_last=self.drop_last,
        )


def build_batch_loader(
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


__all__ = [
    "BucketedBatchDataset",
    "CollateError",
    "TrainingBatch",
    "bucketed_batches",
    "build_batch_loader",
    "collate_samples",
]
