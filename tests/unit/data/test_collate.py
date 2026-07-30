from __future__ import annotations

import pytest
import torch

from sakuramoon.data.collate import (
    BucketedBatchDataset,
    CollateError,
    bucketed_batches,
    build_batch_loader,
    collate_samples,
)
from sakuramoon.data.pipeline import ImageAudit, PipelineSample, RngIdentity
from sakuramoon.data.serialize import SerializedCaption


def _sample(sample_id: int, *, width: int = 8, dense_length: int = 64) -> PipelineSample:
    input_ids = (10, 11, 12) if sample_id % 2 else (20, 21)
    caption = SerializedCaption(
        text="test",
        input_ids=input_ids,
        attention_mask=(True,) * len(input_ids),
        main_token_indices=tuple(range(len(input_ids))),
        main_mask=(True,) * len(input_ids),
        artist_token_indices=(),
        artist_mask=(),
        use_null_style=True,
        all_condition_dropped=False,
        selected_nl=None,
        body="",
        artist_text="",
        condition_tokens=5,
        condition_bucket=dense_length - 34,
        dense_length=dense_length,
        truncated=False,
    )
    return PipelineSample(
        sample_id=sample_id,
        release="r",
        image=torch.full((3, 8, width), sample_id, dtype=torch.uint8),
        target_height=8,
        target_width=width,
        caption=caption,
        audit=ImageAudit(8, 8, width, 8, (0, 0, width, 8), 1.0),
        rng=RngIdentity(1, "S0", 0, sample_id, sample_id + 1, sample_id + 2),
    )


def test_collate_pads_eot_and_preserves_structured_metadata() -> None:
    batch = collate_samples((_sample(1), _sample(2)), padding_token_id=248044)

    assert batch.images.shape == (2, 3, 8, 8)
    assert batch.input_ids.shape == (2, 64)
    assert batch.input_ids[0, 3:].eq(248044).all()
    assert batch.input_ids[1, 2:].eq(248044).all()
    assert torch.equal(batch.attention_mask.sum(dim=1), torch.tensor([3, 2]))
    assert batch.main_token_indices.shape == (2, 3)
    assert batch.main_token_indices[1, 2] == -1
    assert not batch.main_mask[1, 2]
    assert batch.artist_token_indices.shape == (2, 0)
    assert torch.equal(batch.sample_ids, torch.tensor([1, 2]))


def test_bucketed_batches_never_mix_image_or_text_buckets() -> None:
    samples = (_sample(1), _sample(2, width=16), _sample(3), _sample(4, width=16))

    batches = tuple(
        bucketed_batches(
            samples,
            batch_size=2,
            padding_token_id=248044,
            drop_last=True,
        )
    )

    assert len(batches) == 2
    assert {(batch.target_width, batch.dense_length) for batch in batches} == {
        (8, 64),
        (16, 64),
    }


def test_collate_rejects_mixed_bucket() -> None:
    with pytest.raises(CollateError, match="share"):
        collate_samples((_sample(1), _sample(2, width=16)), padding_token_id=248044)


class _EmptyDataset(torch.utils.data.IterableDataset[PipelineSample]):
    def __iter__(self):
        return iter(())


def test_loader_requires_exact_divisible_ready_batch_budget() -> None:
    dataset = BucketedBatchDataset(
        _EmptyDataset(), batch_size=2, padding_token_id=248044, drop_last=True
    )
    with pytest.raises(CollateError, match="multiple"):
        build_batch_loader(
            dataset,
            worker_count=2,
            ready_batches=3,
            pin_memory=True,
        )

    loader = build_batch_loader(
        dataset,
        worker_count=2,
        ready_batches=2,
        pin_memory=True,
    )
    assert loader.num_workers == 2
    assert loader.prefetch_factor == 1
    assert loader.persistent_workers is True
