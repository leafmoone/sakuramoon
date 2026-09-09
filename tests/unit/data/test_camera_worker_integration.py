"""Real-worker integration for the shifted-square camera viewport.

Exercises the production data path without config-root assets: a real
``WebDatasetPipeline`` (real ``__init__``, no ``object.__new__``), a real
shard tar on disk, the real ``_with_local_shards`` lease clone, and a
spawn-context DataLoader with the real ``collate_samples``. Proves the
camera policy survives process pickling, emits exactly one R x R view per
source sample, and aggregates into the batch-level camera counts.
"""

from __future__ import annotations

import io
import tarfile
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from sakuramoon.config.schema import DataBucketsConfig
from sakuramoon.data.buckets import generate_base_buckets, scale_buckets
from sakuramoon.data.camera_viewport import CameraViewportPolicy
from sakuramoon.data.caption import (
    CaptionDropoutProbabilities,
    CaptionFields,
    NlCandidates,
    NlDropoutProbabilities,
)
from sakuramoon.data.collate import TrainingBatch, collate_samples
from sakuramoon.data.manifest import ShardRecord
from sakuramoon.data.metadata import MetadataFieldMapping
from sakuramoon.data.pipeline import WebDatasetPipeline
from sakuramoon.data.serialize import MAIN_SUFFIX, SYSTEM_PREFIX, FramingContract

R = 256  # stage edge of the test bucket vocabulary (train.resolution = 256)

# Module-level definitions only: the spawn context pickles the dataset (and
# everything it references) by reference.


class _Tokenizer:
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        del add_special_tokens
        if not text:
            return []
        if text == SYSTEM_PREFIX:
            return list(range(34))
        if text == MAIN_SUFFIX:
            return list(range(100, 105))
        return [123]


def _identity_adapter(metadata: Mapping[str, object]) -> Mapping[str, object]:
    return metadata


def _fields(_raw: Mapping[str, object]) -> CaptionFields:
    return CaptionFields(
        nsfw=(),
        character=(),
        copyright=(),
        general=(),
        artists=(),
        candidate_tags=frozenset(),
        nl=NlCandidates(None, None, None, None, None),
    )


def _probabilities() -> CaptionDropoutProbabilities:
    nl = NlDropoutProbabilities(0.0, 0.0, 0.0, 0.0, 0.0)
    return CaptionDropoutProbabilities(
        condition_route=0.0,
        condition_only=0.0,
        tag=0.0,
        candidate_source=0.0,
        nl=nl,
    )


def _noop_observer(reason: str) -> None:
    del reason


def _collate(samples: list[object]) -> TrainingBatch:
    return collate_samples(tuple(samples))  # type: ignore[arg-type]


def _gradient_bytes(width: int, height: int) -> bytes:
    xs = (
        np.tile((np.arange(width, dtype=np.uint16) * 7 + 3), height)
        .reshape(height, width)
        .astype(np.uint8)
    )
    ys = (
        np.repeat((np.arange(height, dtype=np.uint16) * 13 + 5), width)
        .reshape(height, width)
        .astype(np.uint8)
    )
    zs = (
        (np.tile(np.arange(width, dtype=np.uint16), height)
        + np.repeat(np.arange(height, dtype=np.uint16) * 3, width)
        + 11)
        .reshape(height, width)
        .astype(np.uint8)
    )
    image = Image.fromarray(np.stack([xs, ys, zs], axis=-1))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _write_shard(directory: Path) -> int:
    shard = directory / "shard-000000.tar"
    entries = (
        (0, (512, 512)),  # square -> identity view
        (1, (1024, 512)),  # 2:1 horizontal
        (2, (512, 1024)),  # 2:1 vertical
    )
    with tarfile.open(shard, "w") as tar:
        for sample_id, (width, height) in entries:
            payload = _gradient_bytes(width, height)
            info = tarfile.TarInfo(name=f"{sample_id:06d}.png")
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
            meta = b'{"id": ' + str(sample_id).encode("ascii") + b"}"
            info = tarfile.TarInfo(name=f"{sample_id:06d}.json")
            info.size = len(meta)
            tar.addfile(info, io.BytesIO(meta))
    return shard.stat().st_size


def _buckets():
    return scale_buckets(
        generate_base_buckets(
            DataBucketsConfig(
                base_area_px=262144,
                quantum_px=32,
                min_short_edge_px=256,
                max_aspect_ratio=4.0,
                transpose_closed=True,
            )
        ),
        R,
    )


def test_real_worker_spawn_round_trip(tmp_path: Path) -> None:
    workdir = tmp_path / "workdir"
    workdir.mkdir()
    size = _write_shard(workdir)
    # Local shard paths are absolute (lease-local staging files); the
    # manifest record keeps the relative shard path and the local path must
    # end with it (validated by the pipeline).
    relative_path = "shard-000000.tar"
    local_path = workdir / relative_path

    pipeline = WebDatasetPipeline(
        shard_paths=(local_path,),
        shard_records=(ShardRecord(path=relative_path, bytes=size),),
        metadata_adapter=_identity_adapter,
        metadata_fields=MetadataFieldMapping(id_field="id"),
        buckets=_buckets(),
        min_crop_retention=0.8,
        probabilities=_probabilities(),
        condition_mode="artist_or_character",
        tokenizer=_Tokenizer(),
        framing=FramingContract(34, 5, 248044),
        caption_fields_parser=_fields,
        rejection_observer=_noop_observer,
        base_seed=7,
        stage="S0",
        cycle_index=0,
        spatial_policy=None,
        camera_policy=CameraViewportPolicy(enabled=True, probability=1.0),
        transparent_policy=None,
    )
    # The real lease clone is what a DataLoader worker would iterate.
    clone = pipeline._with_local_shards(  # type: ignore[reportPrivateUsage]
        (local_path,),
        (ShardRecord(path=relative_path, bytes=size),),
        cycle_index=0,
    )
    loader = DataLoader(
        clone,
        batch_size=3,
        num_workers=1,
        multiprocessing_context="spawn",
        collate_fn=_collate,
    )
    (batch,) = (batch for batch in loader)

    assert isinstance(batch, TrainingBatch)
    # One source sample -> one training view: 3 sources, 3 views.
    assert batch.images.shape == (3, 3, R, R)
    assert (batch.target_height, batch.target_width) == (R, R)
    assert len(batch.audits) == 3
    assert [audit.camera_orientation for audit in batch.audits] == [
        "square",
        "horizontal",
        "vertical",
    ]
    for audit in batch.audits:
        assert audit.crop_policy == "camera_viewport"
        assert audit.camera_selected is True
        assert audit.camera_applied is True
        assert audit.camera_fallback_reason == "none"
    # Batch-level camera counts: all selected, all applied, fixed keys.
    assert batch.camera_viewport.selected == 3
    assert batch.camera_viewport.applied == 3
    assert batch.camera_viewport.orientation_counts == {
        "horizontal": 1,
        "vertical": 1,
        "square": 1,
    }
    # The camera policy survived spawn pickling (proven by the applied
    # audits) and each view carries the real per-sample RNG identity.
    assert len(batch.rng_identities) == 3
    assert len({identity.sample_id for identity in batch.rng_identities}) == 3
    assert torch.isfinite(batch.images.float()).all()
