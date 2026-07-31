"""Bounded local-shard WebDataset processing for one training rank."""

from __future__ import annotations

import hashlib
import io
import json
import math
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
import webdataset as wds
from PIL import Image
from torch.utils.data import IterableDataset, get_worker_info

from sakuramoon.data.buckets import BucketShape, RejectionReason
from sakuramoon.data.caption import (
    CaptionDropoutProbabilities,
    CaptionFields,
    build_caption_plan,
)
from sakuramoon.data.image_ops import ImageRejected, prepare_image
from sakuramoon.data.manifest import DatasetManifest
from sakuramoon.data.metadata import MetadataRecord, parse_metadata
from sakuramoon.data.serialize import (
    FramingContract,
    SerializedCaption,
    TokenEncoder,
    serialize_caption,
)
from sakuramoon.data.state import ShardRunState, SingleProcessShardCoordinator

CaptionFieldsParser = Callable[[Mapping[str, object]], CaptionFields]
RejectionObserver = Callable[[RejectionReason], None]
_IMAGE_KEYS = ("jpg", "jpeg", "png", "webp")


class PipelineSampleError(ValueError):
    """A WebDataset sample cannot satisfy the training input contract."""


@dataclass(frozen=True)
class RngIdentity:
    base_seed: int
    stage: str
    pass_index: int
    sample_id: int
    caption_seed: int
    crop_seed: int


@dataclass(frozen=True)
class ImageAudit:
    source_width: int
    source_height: int
    resized_width: int
    resized_height: int
    crop_box: tuple[int, int, int, int]
    crop_retention: float


@dataclass(frozen=True)
class PipelineSample:
    sample_id: int
    release: str
    image: torch.Tensor
    target_height: int
    target_width: int
    caption: SerializedCaption
    audit: ImageAudit
    rng: RngIdentity
    padding_token_id: int


def _validate_local_shard_paths(shard_paths: tuple[Path, ...]) -> None:
    if type(shard_paths) is not tuple or not shard_paths:
        raise PipelineSampleError("pipeline requires local shard paths")
    if len(set(shard_paths)) != len(shard_paths):
        raise PipelineSampleError("pipeline shard paths must be unique")
    for path in shard_paths:
        if (
            not isinstance(path, Path)  # pyright: ignore[reportUnnecessaryIsInstance]
            or not path.is_absolute()
            or not path.is_file()
        ):
            raise PipelineSampleError(
                "pipeline shard paths must be absolute local regular files"
            )


def _domain_seed(
    base_seed: int,
    stage: str,
    pass_index: int,
    sample_id: int,
    domain: str,
) -> int:
    if (
        type(base_seed) is not int
        or base_seed < 0
        or not stage
        or type(pass_index) is not int
        or pass_index < 0
        or type(sample_id) is not int
        or sample_id <= 0
    ):
        raise PipelineSampleError("RNG identity fields are invalid")
    payload = f"{base_seed}\0{stage}\0{pass_index}\0{sample_id}\0{domain}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & (2**63 - 1)


def rng_identity(
    *, base_seed: int, stage: str, pass_index: int, sample_id: int
) -> RngIdentity:
    return RngIdentity(
        base_seed=base_seed,
        stage=stage,
        pass_index=pass_index,
        sample_id=sample_id,
        caption_seed=_domain_seed(
            base_seed, stage, pass_index, sample_id, "caption"
        ),
        crop_seed=_domain_seed(base_seed, stage, pass_index, sample_id, "crop"),
    )


def local_shard_order(
    cache_root: Path,
    manifest: DatasetManifest,
    state: ShardRunState,
) -> tuple[Path, ...]:
    """Put an interrupted active shard first and omit completed shards."""

    known = {shard.path for shard in manifest.shards}
    if not set(state.completed).issubset(known) or state.active not in known | {None}:
        raise PipelineSampleError("shard state does not match the manifest")
    remaining = [
        shard.path for shard in manifest.shards if shard.path not in state.completed
    ]
    if state.active is not None:
        remaining.remove(state.active)
        remaining.insert(0, state.active)
    return tuple(cache_root / path for path in remaining)


def _metadata(sample: Mapping[str, object]) -> Mapping[str, object]:
    payload = sample.get("json")
    if not isinstance(payload, bytes):
        raise PipelineSampleError("WebDataset sample must contain one JSON byte payload")
    try:
        document = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise PipelineSampleError("WebDataset metadata JSON is invalid") from None
    if not isinstance(document, dict):
        raise PipelineSampleError("WebDataset metadata must be a JSON object")
    mapping = cast(dict[object, object], document)
    if not all(isinstance(key, str) for key in mapping):
        raise PipelineSampleError("WebDataset metadata keys must be strings")
    return cast(dict[str, object], mapping)


def _image_bytes(sample: Mapping[str, object]) -> bytes:
    present = tuple(key for key in _IMAGE_KEYS if key in sample)
    if len(present) != 1 or not isinstance(sample[present[0]], bytes):
        raise PipelineSampleError("WebDataset sample must contain exactly one image")
    return cast(bytes, sample[present[0]])


def _uint8_chw(image: Image.Image) -> torch.Tensor:
    if image.mode != "RGB":
        raise PipelineSampleError("processed image must be RGB")
    storage = bytearray(image.tobytes())
    return (
        torch.frombuffer(storage, dtype=torch.uint8)
        .reshape(image.height, image.width, 3)
        .permute(2, 0, 1)
        .contiguous()
    )


class WebDatasetPipeline(IterableDataset[PipelineSample]):
    """Decode and serialize each non-validation sample exactly once."""

    def __init__(
        self,
        *,
        shard_paths: tuple[Path, ...],
        validation_ids: frozenset[int],
        buckets: tuple[BucketShape, ...],
        min_crop_retention: float,
        probabilities: CaptionDropoutProbabilities,
        tokenizer: TokenEncoder,
        framing: FramingContract,
        caption_fields_parser: CaptionFieldsParser,
        rejection_observer: RejectionObserver,
        base_seed: int,
        stage: str,
        pass_index: int,
    ) -> None:
        super().__init__()
        _validate_local_shard_paths(shard_paths)
        if (
            type(validation_ids) is not frozenset
            or any(type(item) is not int or item <= 0 for item in validation_ids)
            or type(buckets) is not tuple
            or not buckets
            or any(
                not isinstance(bucket, BucketShape)  # pyright: ignore[reportUnnecessaryIsInstance]
                for bucket in buckets
            )
            or len(set(buckets)) != len(buckets)
            or type(min_crop_retention) is not float
            or not math.isfinite(min_crop_retention)
            or not 0.0 <= min_crop_retention <= 1.0
            or not isinstance(  # pyright: ignore[reportUnnecessaryIsInstance]
                probabilities, CaptionDropoutProbabilities
            )
            or not isinstance(  # pyright: ignore[reportUnnecessaryIsInstance]
                framing, FramingContract
            )
            or not callable(caption_fields_parser)
            or not callable(rejection_observer)
            or type(base_seed) is not int
            or base_seed < 0
            or type(stage) is not str
            or not stage
            or stage != stage.strip()
            or type(pass_index) is not int
            or pass_index < 0
        ):
            raise PipelineSampleError("pipeline construction fields are invalid")
        self.shard_paths = shard_paths
        self.validation_ids = validation_ids
        self.buckets = buckets
        self.min_crop_retention = min_crop_retention
        self.probabilities = probabilities
        self.tokenizer = tokenizer
        self.framing = framing
        self.caption_fields_parser = caption_fields_parser
        self.rejection_observer = rejection_observer
        self.base_seed = base_seed
        self.stage = stage
        self.pass_index = pass_index
        self._lease_managed = False

    def _process(self, sample: Mapping[str, object]) -> PipelineSample | None:
        raw_metadata = _metadata(sample)
        metadata: MetadataRecord = parse_metadata(raw_metadata)
        if metadata.id in self.validation_ids:
            return None
        identity = rng_identity(
            base_seed=self.base_seed,
            stage=self.stage,
            pass_index=self.pass_index,
            sample_id=metadata.id,
        )
        fields = cast(object, self.caption_fields_parser(metadata.raw))
        if not isinstance(fields, CaptionFields):
            raise PipelineSampleError("caption field parser returned an invalid value")
        plan = build_caption_plan(
            fields, self.probabilities, seed=identity.caption_seed
        )
        caption = serialize_caption(plan, self.tokenizer, self.framing)
        try:
            with Image.open(io.BytesIO(_image_bytes(sample))) as decoded:
                decoded.load()
                processed = prepare_image(
                    decoded,
                    self.buckets,
                    min_crop_retention=self.min_crop_retention,
                    crop_seed=identity.crop_seed,
                )
        except ImageRejected as error:
            self.rejection_observer(cast(RejectionReason, error.reason))
            return None
        except (OSError, Image.DecompressionBombError):
            raise PipelineSampleError("WebDataset image decode failed") from None
        assignment = processed.assignment
        return PipelineSample(
            sample_id=metadata.id,
            release=metadata.release,
            image=_uint8_chw(processed.image),
            target_height=assignment.bucket.height,
            target_width=assignment.bucket.width,
            caption=caption,
            audit=ImageAudit(
                source_width=assignment.source_width,
                source_height=assignment.source_height,
                resized_width=assignment.resized_width,
                resized_height=assignment.resized_height,
                crop_box=processed.crop_box,
                crop_retention=assignment.crop_retention,
            ),
            rng=identity,
            padding_token_id=self.framing.padding_token_id,
        )

    def _iter_paths(self, shard_paths: tuple[Path, ...]) -> Iterator[PipelineSample]:
        _validate_local_shard_paths(shard_paths)
        urls = [str(path) for path in shard_paths]
        factory = cast(
            Callable[..., Iterable[dict[str, Any]]],
            wds.WebDataset,  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]
        )
        dataset = factory(
            urls,
            shardshuffle=False,
            empty_check=True,
        )
        for raw_sample in dataset:
            processed = self._process(cast(Mapping[str, object], raw_sample))
            if processed is not None:
                yield processed

    def _with_local_shards(
        self, shard_paths: tuple[Path, ...]
    ) -> WebDatasetPipeline:
        """Clone the validated processing contract onto prepared local shards."""

        pipeline = WebDatasetPipeline(
            shard_paths=shard_paths,
            validation_ids=self.validation_ids,
            buckets=self.buckets,
            min_crop_retention=self.min_crop_retention,
            probabilities=self.probabilities,
            tokenizer=self.tokenizer,
            framing=self.framing,
            caption_fields_parser=self.caption_fields_parser,
            rejection_observer=self.rejection_observer,
            base_seed=self.base_seed,
            stage=self.stage,
            pass_index=self.pass_index,
        )
        pipeline._lease_managed = True
        return pipeline

    def iter_leased_shards(
        self,
        coordinator: SingleProcessShardCoordinator,
        shard_paths: tuple[str, ...],
    ) -> Iterator[PipelineSample]:
        """Consume cached shards under the durable at-least-once lease boundary."""

        if (
            type(shard_paths) is not tuple
            or not shard_paths
            or any(
                type(path) is not str
                or not path
                or path != path.strip()
                or Path(path).is_absolute()
                for path in shard_paths
            )
            or len(set(shard_paths)) != len(shard_paths)
        ):
            raise PipelineSampleError("leased pipeline requires explicit shard paths")
        for shard_path in shard_paths:
            with coordinator.lease(shard_path) as cached:
                if cached is None:
                    continue
                if cached.fetched.relative_path != shard_path:
                    raise PipelineSampleError("cache returned a different leased shard")
                yield from self._iter_paths((cached.fetched.path,))

    def __iter__(self) -> Iterator[PipelineSample]:
        if get_worker_info() is not None and not self._lease_managed:
            raise PipelineSampleError(
                "worker iteration requires the durable shard lease entry point"
            )
        yield from self._iter_paths(self.shard_paths)


__all__ = [
    "CaptionFieldsParser",
    "ImageAudit",
    "PipelineSample",
    "PipelineSampleError",
    "RejectionObserver",
    "RngIdentity",
    "WebDatasetPipeline",
    "local_shard_order",
    "rng_identity",
]
