from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from torch.utils.data import DataLoader, get_worker_info

from sakuramoon.data.caption import (
    CAPTION_DROPOUT_KEYS,
    CaptionPlan,
    ConditionRequest,
    Tag,
    empty_caption_dropout_hits,
)
from sakuramoon.data.collate import (
    BucketedBatchDataset,
    CollateError,
    _build_batch_loader,  # pyright: ignore[reportPrivateUsage]
    _shutdown_loader,  # pyright: ignore[reportPrivateUsage]
    bucketed_batches,
    collate_samples,
)
from sakuramoon.data.pipeline import ImageAudit, PipelineSample, RngIdentity
from sakuramoon.data.serialize import SerializedCaption
from sakuramoon.telemetry.metrics import DROPOUT_KEYS


def _sample(
    sample_id: int, *, width: int = 8, dense_length: int = 64
) -> PipelineSample:
    input_ids = (10, 11, 12) if sample_id % 2 else (20, 21)
    dropout_hits = replace(
        empty_caption_dropout_hits(),
        general=bool(sample_id % 2),
        artist=bool(sample_id % 2),
    )
    caption = SerializedCaption(
        plan=CaptionPlan(
            tags=(),
            condition=None,
            nl_text=None,
            selected_nl=None,
            all_condition_dropped=False,
            dropout_hits=dropout_hits,
        ),
        text="test",
        input_ids=input_ids,
        attention_mask=(True,) * len(input_ids),
        main_token_indices=tuple(range(len(input_ids))),
        main_mask=(True,) * len(input_ids),
        condition_token_indices=(),
        condition_mask=(),
        use_null_condition=True,
        condition_source=None,
        condition_role=None,
        all_condition_dropped=False,
        dropout_hits=dropout_hits,
        selected_nl=None,
        body="",
        condition_text="",
        condition_tokens=5,
        condition_bucket=dense_length - 34,
        dense_length=dense_length,
        truncated=False,
    )
    return PipelineSample(
        sample_id=sample_id,
        source_shard="data/test.tar",
        image=torch.full((3, 8, width), sample_id, dtype=torch.uint8),
        target_height=8,
        target_width=width,
        caption=caption,
        audit=ImageAudit(8, 8, width, 8, (0, 0, width, 8), 1.0),
        rng=RngIdentity(1, "S0", 0, sample_id, sample_id + 1, sample_id + 2),
        padding_token_id=248044,
    )


def test_service_batches_wait_for_a_transient_epoch_tail(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import sakuramoon.data.collate as collate_module

    record = collate_module.ShardRecord(path="data/remaining.tar", bytes=1)
    local_path = tmp_path / "remaining.tar"
    local_path.write_bytes(b"x")
    descriptor = collate_module.ShardLeaseDescriptor(
        lease_id="lease-1",
        worker_id=0,
        cycle_index=1,
        state_revision=0,
        record=record,
        local_path=local_path,
    )
    expected_batch = cast(Any, object())

    class FakeClient:
        identity = SimpleNamespace(worker_count=1)

        def __init__(self) -> None:
            self.lease_calls = 0

        def health(self) -> bool:
            return False

        def lease(self, worker_id: int):
            assert worker_id == 0
            self.lease_calls += 1
            return None if self.lease_calls == 1 else descriptor

        def acknowledge(self, _descriptor) -> None:
            raise AssertionError("batch must be observed before ACK")

    class FakeDataset:
        def __init__(self, *_args, **_kwargs) -> None:
            self.command = None

        def submit(self, worker_id: int, command) -> None:
            assert worker_id == 0
            self.command = command

        def stop(self, _worker_id: int, _record) -> None:
            pass

    class FakeLoader:
        def __init__(self, dataset: FakeDataset) -> None:
            self.dataset = dataset

        def __iter__(self):
            return self

        def __next__(self):
            command = self.dataset.command
            if command is None:
                raise AssertionError("loader advanced before a lease was available")
            self.dataset.command = None
            return collate_module._WorkerBatch(
                0,
                123,
                command.shard_path,
                expected_batch,
            )

    monkeypatch.setattr(collate_module, "_PersistentShardDataset", FakeDataset)
    monkeypatch.setattr(
        collate_module,
        "_build_batch_loader",
        lambda dataset, **_kwargs: FakeLoader(dataset),
    )
    monkeypatch.setattr(
        collate_module,
        "_shutdown_loader",
        lambda _loader, **_kwargs: None,
    )
    monkeypatch.setattr(collate_module.time, "sleep", lambda _seconds: None)

    client = FakeClient()
    stream = collate_module.iter_service_batches(
        cast(Any, SimpleNamespace(base_seed=1)),
        cast(Any, client),
        batch_size=1,
        worker_count=1,
        ready_batches=1,
        pin_memory=False,
        drop_last=True,
    )
    try:
        assert next(stream) is expected_batch
        assert client.lease_calls == 2
    finally:
        stream.close()


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
    assert batch.condition_token_indices.shape == (2, 0)
    assert batch.active_condition_sample_indices.numel() == 0
    assert batch.condition_sources == (None, None)
    assert batch.condition_roles == (None, None)
    assert batch.condition_routes.as_mapping() == {
        "artist_text": 0,
        "character_text": 0,
        "null": 2,
    }
    assert torch.equal(batch.sample_ids, torch.tensor([1, 2]))
    assert CAPTION_DROPOUT_KEYS == DROPOUT_KEYS
    dropout_hits = batch.dropout_hits.as_mapping()
    assert tuple(dropout_hits) == CAPTION_DROPOUT_KEYS
    assert dropout_hits["general"] == 1
    assert dropout_hits["artist"] == 1
    assert sum(dropout_hits.values()) == 2


def test_collate_dynamically_right_pads_mixed_text_buckets() -> None:
    short = _sample(1, dense_length=64)
    long = _sample(2, dense_length=128)

    batch = collate_samples((short, long))

    assert batch.dense_length == 128
    assert batch.input_ids.shape == (2, 128)
    assert batch.input_ids[0, len(short.caption.input_ids) :].eq(248044).all()
    assert batch.input_ids[1, len(long.caption.input_ids) :].eq(248044).all()
    assert torch.equal(
        batch.attention_mask.sum(dim=1),
        torch.tensor([len(short.caption.input_ids), len(long.caption.input_ids)]),
    )


def test_bucketed_batches_mix_text_lengths_but_never_image_buckets() -> None:
    samples = (
        _sample(1, dense_length=64),
        _sample(2, width=16, dense_length=128),
        _sample(3, dense_length=128),
        _sample(4, width=16, dense_length=64),
    )

    batches = tuple(
        bucketed_batches(
            samples,
            batch_size=2,
            drop_last=True,
        )
    )

    assert len(batches) == 2
    assert {(batch.target_width, batch.dense_length) for batch in batches} == {
        (8, 128),
        (16, 128),
    }


def test_length_sort_window_reduces_padding_without_mixing_image_buckets() -> None:
    samples = tuple(
        _sample(sample_id, dense_length=length)
        for sample_id, length in enumerate((64, 128, 64, 128), start=1)
    )

    batches = tuple(
        bucketed_batches(
            samples,
            batch_size=2,
            drop_last=True,
            length_sort_window_batches=2,
        )
    )

    assert len(batches) == 2
    assert sorted(batch.dense_length for batch in batches) == [64, 128]
    assert all(batch.images.shape[0] == 2 for batch in batches)


def test_length_sort_window_has_a_strict_per_image_bound() -> None:
    from sakuramoon.data.collate import (  # pyright: ignore[reportPrivateUsage]
        _LengthAwareBatcher,
    )

    batcher = _LengthAwareBatcher(batch_size=2, window_batches=2)
    assert not batcher.add(_sample(1, dense_length=128))
    assert not batcher.add(_sample(2, dense_length=64))
    assert not batcher.add(_sample(3, dense_length=128))
    batches = batcher.add(_sample(4, dense_length=64))

    assert len(batches) == 2
    assert sorted(batch.dense_length for batch in batches) == [64, 128]


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


def test_collate_builds_active_condition_sample_plan_on_cpu() -> None:
    sample = _sample(1)
    active = replace(
        sample,
        caption=replace(
            sample.caption,
            plan=replace(
                sample.caption.plan,
                condition=ConditionRequest(
                    source="artist_text",
                    role="style",
                    tags=(Tag("artist", "artist"),),
                ),
            ),
            main_token_indices=(0, 1),
            main_mask=(True, True),
            condition_token_indices=(2,),
            condition_mask=(True,),
            use_null_condition=False,
            condition_source="artist_text",
            condition_role="style",
        ),
    )

    batch = collate_samples((active, _sample(2)))

    assert batch.active_condition_sample_indices.device.type == "cpu"
    assert torch.equal(batch.active_condition_sample_indices, torch.tensor([0]))
    assert batch.condition_sources == ("artist_text", None)
    assert batch.condition_roles == ("style", None)
    assert batch.condition_routes.as_mapping() == {
        "artist_text": 1,
        "character_text": 0,
        "null": 1,
    }


@pytest.mark.parametrize(
    ("indices", "mask", "use_null"),
    [
        ((3,), (True,), False),
        ((2,), (False,), False),
        ((2,), (True,), True),
        ((), (), False),
    ],
)
def test_collate_rejects_invalid_condition_routing_metadata(
    indices: tuple[int, ...],
    mask: tuple[bool, ...],
    use_null: bool,
) -> None:
    sample = _sample(1)
    invalid = replace(
        sample,
        caption=replace(
            sample.caption,
            condition_token_indices=indices,
            condition_mask=mask,
            use_null_condition=use_null,
            condition_source="artist_text" if not use_null else None,
            condition_role="style" if not use_null else None,
        ),
    )

    with pytest.raises(CollateError, match="condition token|condition token presence"):
        collate_samples((invalid,))


def test_collate_rejects_missing_condition_source_or_role() -> None:
    sample = _sample(1)
    invalid = replace(
        sample,
        caption=replace(
            sample.caption,
            condition_token_indices=(2,),
            condition_mask=(True,),
            use_null_condition=False,
            condition_source=None,
            condition_role="style",
        ),
    )

    with pytest.raises(CollateError, match="condition source and role"):
        collate_samples((invalid,))


class _EmptyDataset(torch.utils.data.IterableDataset[PipelineSample]):
    def __iter__(self) -> Iterator[PipelineSample]:
        return iter(())


class _WorkerSeedDataset(torch.utils.data.IterableDataset[str]):
    def __iter__(self) -> Iterator[str]:
        info = get_worker_info()
        if info is None:
            raise RuntimeError("worker seed probe requires a DataLoader worker")
        yield f"{info.id}:{torch.initial_seed()}"


def _bootstrap_worker_seeds(worker_seed: int) -> tuple[str, ...]:
    parent_state = torch.get_rng_state()
    loader = _build_batch_loader(
        _WorkerSeedDataset(),
        worker_count=2,
        ready_batches=2,
        pin_memory=False,
        worker_seed=worker_seed,
    )
    assert torch.equal(torch.get_rng_state(), parent_state)
    try:
        worker_seeds = tuple(sorted(iter(loader)))
    finally:
        _shutdown_loader(loader)
    assert torch.equal(torch.get_rng_state(), parent_state)
    return worker_seeds


def test_loader_shutdown_only_suppresses_failure_while_preserving_an_exception() -> None:
    class FailingIterator:
        def _shutdown_workers(self) -> None:
            raise RuntimeError("worker terminated during shutdown")

    class FailingLoader:
        _iterator = FailingIterator()

    loader = cast(DataLoader[Any], FailingLoader())
    with pytest.raises(RuntimeError, match="terminated during shutdown"):
        _shutdown_loader(loader)

    _shutdown_loader(loader, suppress_worker_failure=True)


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
            worker_seed=44,
        )

    loader = _build_batch_loader(
        dataset,
        worker_count=2,
        ready_batches=2,
        pin_memory=True,
        worker_seed=44,
    )
    assert loader.num_workers == 2
    assert loader.prefetch_factor == 1
    assert loader.persistent_workers is True


def test_loader_worker_bootstrap_is_reproducible_without_global_rng_drift() -> None:
    original_state = torch.get_rng_state()
    try:
        torch.manual_seed(90210)  # pyright: ignore[reportUnknownMemberType]
        expected_parent_state = torch.get_rng_state()

        first = _bootstrap_worker_seeds(44)
        second = _bootstrap_worker_seeds(44)

        assert torch.equal(torch.get_rng_state(), expected_parent_state)
        assert first == second
        parsed = tuple(
            (int(worker_id), int(seed))
            for worker_id, seed in (value.split(":", maxsplit=1) for value in first)
        )
        assert tuple(worker_id for worker_id, _seed in parsed) == (0, 1)
        assert parsed[1][1] == parsed[0][1] + 1
    finally:
        torch.set_rng_state(original_state)


@pytest.mark.parametrize("worker_seed", [True, -1, 1.5, "44", 2**64, None])
def test_loader_rejects_invalid_worker_seed(worker_seed: object) -> None:
    with pytest.raises(CollateError, match="worker_seed"):
        _build_batch_loader(
            _WorkerSeedDataset(),
            worker_count=2,
            ready_batches=2,
            pin_memory=False,
            worker_seed=worker_seed,  # pyright: ignore[reportArgumentType]
        )


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
