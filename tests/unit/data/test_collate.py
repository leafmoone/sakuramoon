from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from sakuramoon.data.caption import CAPTION_DROPOUT_KEYS, CaptionDropoutHits
from sakuramoon.data.collate import (
    BucketedBatchDataset,
    CollateError,
    _build_batch_loader,  # pyright: ignore[reportPrivateUsage]
    bucketed_batches,
    collate_samples,
)
from sakuramoon.data.pipeline import ImageAudit, PipelineSample, RngIdentity
from sakuramoon.data.serialize import SerializedCaption
from sakuramoon.telemetry.metrics import DROPOUT_KEYS


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
        dropout_hits=CaptionDropoutHits(
            all_condition=False,
            nsfw=False,
            character=False,
            copyright=False,
            general=bool(sample_id % 2),
            artist=bool(sample_id % 2),
            candidate_source=False,
            long_names=False,
            long_no_names=False,
            short_vibes=False,
            nl2=False,
            nl3=False,
        ),
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
        padding_token_id=248044,
    )


def test_collate_pads_eot_and_preserves_structured_metadata() -> None:
    batch = collate_samples((_sample(1), _sample(2)))

    assert batch.images.shape == (2, 3, 8, 8)
    assert batch.input_ids.shape == (2, 64)
    assert batch.input_ids[0, 3:].eq(248044).all()
    assert batch.input_ids[1, 2:].eq(248044).all()
    assert torch.equal(batch.attention_mask.sum(dim=1), torch.tensor([3, 2]))
    assert batch.main_token_indices.shape == (2, 3)
    assert batch.main_token_indices[1, 2] == -1
    assert not batch.main_mask[1, 2]
    assert batch.main_token_lengths == (3, 2)
    assert batch.artist_token_indices.shape == (2, 0)
    assert batch.active_style_sample_indices.numel() == 0
    assert torch.equal(batch.sample_ids, torch.tensor([1, 2]))
    assert CAPTION_DROPOUT_KEYS == DROPOUT_KEYS
    dropout_hits = batch.dropout_hits.as_mapping()
    assert tuple(dropout_hits) == CAPTION_DROPOUT_KEYS
    assert dropout_hits["general"] == 1
    assert dropout_hits["artist"] == 1
    assert sum(dropout_hits.values()) == 2


def test_bucketed_batches_never_mix_image_or_text_buckets() -> None:
    samples = (_sample(1), _sample(2, width=16), _sample(3), _sample(4, width=16))

    batches = tuple(
        bucketed_batches(
            samples,
            batch_size=2,
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
        collate_samples((_sample(1), _sample(2, width=16)))


def test_collate_binds_padding_to_each_sample_framing() -> None:
    mismatched = replace(_sample(2), padding_token_id=0)
    with pytest.raises(CollateError, match="padding token"):
        collate_samples((_sample(1), mismatched))


@pytest.mark.parametrize(
    ("indices", "mask"),
    [
        ((0, 3), (True, True)),
        ((0, -1), (True, True)),
        ((0, 1), (True, False)),
        ((0, 1), (True,)),
    ],
)
def test_collate_rejects_invalid_main_index_metadata(
    indices: tuple[int, ...],
    mask: tuple[bool, ...],
) -> None:
    sample = _sample(1)
    invalid = replace(
        sample,
        caption=replace(
            sample.caption,
            main_token_indices=indices,
            main_mask=mask,
        ),
    )

    with pytest.raises(CollateError, match="main token indices"):
        collate_samples((invalid,))


def test_collate_builds_active_style_sample_plan_on_cpu() -> None:
    sample = _sample(1)
    active = replace(
        sample,
        caption=replace(
            sample.caption,
            main_token_indices=(0, 1),
            main_mask=(True, True),
            artist_token_indices=(2,),
            artist_mask=(True,),
            use_null_style=False,
        ),
    )

    batch = collate_samples((active, _sample(2)))

    assert batch.active_style_sample_indices.device.type == "cpu"
    assert torch.equal(batch.active_style_sample_indices, torch.tensor([0]))


@pytest.mark.parametrize(
    ("indices", "mask", "use_null"),
    [
        ((3,), (True,), False),
        ((2,), (False,), False),
        ((2,), (True,), True),
        ((), (), False),
    ],
)
def test_collate_rejects_invalid_artist_routing_metadata(
    indices: tuple[int, ...],
    mask: tuple[bool, ...],
    use_null: bool,
) -> None:
    sample = _sample(1)
    invalid = replace(
        sample,
        caption=replace(
            sample.caption,
            artist_token_indices=indices,
            artist_mask=mask,
            use_null_style=use_null,
        ),
    )

    with pytest.raises(CollateError, match="Artist token|Artist token presence"):
        collate_samples((invalid,))


class _EmptyDataset(torch.utils.data.IterableDataset[PipelineSample]):
    def __iter__(self):
        return iter(())


def test_loader_requires_exact_divisible_ready_batch_budget() -> None:
    dataset = BucketedBatchDataset(
        _EmptyDataset(), batch_size=2, drop_last=True
    )
    with pytest.raises(CollateError, match="multiple"):
        _build_batch_loader(
            dataset,
            worker_count=2,
            ready_batches=3,
            pin_memory=True,
        )

    loader = _build_batch_loader(
        dataset,
        worker_count=2,
        ready_batches=2,
        pin_memory=True,
    )
    assert loader.num_workers == 2
    assert loader.prefetch_factor == 1
    assert loader.persistent_workers is True


@pytest.mark.parametrize("drop_last", [0, "false"])
def test_bucketed_batches_requires_exact_boolean_drop_last(drop_last: object) -> None:
    with pytest.raises(CollateError, match="drop_last"):
        tuple(
            bucketed_batches(
                (_sample(1),),
                batch_size=1,
                drop_last=drop_last,  # pyright: ignore[reportArgumentType]
            )
        )
