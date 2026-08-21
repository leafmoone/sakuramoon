# pyright: reportPrivateUsage=false

from __future__ import annotations

import io
import tarfile
from pathlib import Path

import pytest
import torch
from PIL import Image

from sakuramoon.eval.features import (
    EvaluationFeatureError,
    iter_validation_image_batches,
)


def _solid_png(color: tuple[int, int, int]) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), color).save(buffer, format="PNG")
    return buffer.getvalue()


def _write_tar(path: Path, members: dict[str, bytes]) -> None:
    with tarfile.open(path, "w") as handle:
        for name, payload in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            handle.addfile(info, io.BytesIO(payload))


@pytest.fixture()
def shard_root(tmp_path: Path) -> Path:
    _write_tar(
        tmp_path / "a.tar",
        {
            "x1.png": _solid_png((255, 0, 0)),
            "x2.png": _solid_png((0, 255, 0)),
            "x3.png": _solid_png((0, 0, 255)),
        },
    )
    _write_tar(
        tmp_path / "b.tar",
        {
            "y1.png": _solid_png((10, 20, 30)),
            "y2.png": _solid_png((40, 50, 60)),
            "notes.txt": b"not-an-image\n",
        },
    )
    return tmp_path


def test_batches_follow_global_tar_order(
    shard_root: Path,
) -> None:
    batches = tuple(
        iter_validation_image_batches(
            shard_root,
            5,
            2,
            output_size=8,
        )
    )

    assert [len(batch.sample_ids) for batch in batches] == [2, 2, 1]
    assert [batch.sample_ids for batch in batches] == [
        ("a.tar::x1.png", "a.tar::x2.png"),
        ("a.tar::x3.png", "b.tar::y1.png"),
        ("b.tar::y2.png",),
    ]
    for batch in batches:
        assert batch.images.dtype is torch.uint8
        assert batch.images.shape[1:] == (3, 8, 8)
    red = batches[0].images[0]
    assert red.shape == (3, 8, 8)
    assert torch.all(red[0] == 255).item()
    assert torch.all(red[1] == 0).item()
    assert torch.all(red[2] == 0).item()


def test_count_truncates_before_later_archives(
    shard_root: Path,
) -> None:
    batches = tuple(iter_validation_image_batches(shard_root, 2, 2, output_size=8))

    assert [batch.sample_ids for batch in batches] == [
        ("a.tar::x1.png", "a.tar::x2.png"),
    ]
    assert batches[0].images.shape[0] == 2


def test_partial_final_batch(
    shard_root: Path,
) -> None:
    batches = tuple(iter_validation_image_batches(shard_root, 5, 4, output_size=8))

    assert [len(batch.sample_ids) for batch in batches] == [4, 1]


def test_incomplete_count_raises(
    shard_root: Path,
) -> None:
    with pytest.raises(EvaluationFeatureError, match="incomplete: 5/6"):
        tuple(iter_validation_image_batches(shard_root, 6, 2, output_size=8))


def test_duplicate_member_raises(tmp_path: Path) -> None:
    archive = tmp_path / "dup.tar"
    with tarfile.open(archive, "w") as handle:
        for name in ("img.png", "img.png"):
            info = tarfile.TarInfo(name)
            payload = _solid_png((1, 2, 3))
            info.size = len(payload)
            handle.addfile(info, io.BytesIO(payload))

    with pytest.raises(EvaluationFeatureError, match="duplicated"):
        tuple(iter_validation_image_batches(tmp_path, 1, 1, output_size=8))


def test_undecodable_image_raises(tmp_path: Path) -> None:
    _write_tar(tmp_path / "bad.tar", {"broken.jpg": b"definitely not a jpeg"})

    with pytest.raises(EvaluationFeatureError, match="cannot be decoded"):
        tuple(iter_validation_image_batches(tmp_path, 1, 1, output_size=8))


def test_max_workers_validation(
    shard_root: Path,
) -> None:
    with pytest.raises(ValueError, match="worker count"):
        tuple(
            iter_validation_image_batches(
                shard_root, 1, 1, output_size=8, max_workers=0
            )
        )
    with pytest.raises(ValueError, match="worker count"):
        tuple(
            iter_validation_image_batches(
                shard_root, 1, 1, output_size=8, max_workers=2.0
            )
        )


def test_single_worker_smoke(
    shard_root: Path,
) -> None:
    batches = tuple(
        iter_validation_image_batches(shard_root, 5, 3, output_size=8, max_workers=1)
    )

    assert [len(batch.sample_ids) for batch in batches] == [3, 2]
