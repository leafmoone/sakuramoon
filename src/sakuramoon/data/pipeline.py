"""Bounded local-shard WebDataset processing for one training rank."""

from __future__ import annotations

import io
import json
import math
import random
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
from sakuramoon.data.manifest import ShardRecord
from sakuramoon.data.metadata import (
    MetadataFieldMapping,
    OperationalMetadataRecord,
    parse_shard_metadata,
)
from sakuramoon.data.serialize import (
    FramingContract,
    SerializedCaption,
    TokenEncoder,
    serialize_caption,
)

CaptionFieldsParser = Callable[[Mapping[str, object]], CaptionFields]
MetadataAdapter = Callable[[Mapping[str, object]], Mapping[str, object]]
RejectionObserver = Callable[[RejectionReason], None]
_IMAGE_KEYS = ("jpg", "jpeg", "png", "webp")


class PipelineSampleError(ValueError):
    """A WebDataset sample cannot satisfy the training input contract."""


@dataclass(frozen=True)
class RngIdentity:
    base_seed: int
    stage: str
    cycle_index: int
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
    source_shard: str
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


def _validate_shard_records(
    shard_paths: tuple[Path, ...], shard_records: tuple[ShardRecord, ...]
) -> None:
    if (
        type(shard_records) is not tuple
        or len(shard_records) != len(shard_paths)
        or any(
            not isinstance(record, ShardRecord)  # pyright: ignore[reportUnnecessaryIsInstance]
            for record in shard_records
        )
        or len({record.path for record in shard_records}) != len(shard_records)
    ):
        raise PipelineSampleError(
            "pipeline requires one unique manifest record per local shard"
        )
    for path, record in zip(shard_paths, shard_records, strict=True):
        record_parts = Path(record.path).parts
        if (
            len(path.parts) < len(record_parts)
            or path.parts[-len(record_parts) :] != record_parts
        ):
            raise PipelineSampleError(
                "local shard path must end with its manifest shard path"
            )


def _domain_seed(
    base_seed: int,
    stage: str,
    cycle_index: int,
    sample_id: int,
    domain: str,
) -> int:
    if (
        type(base_seed) is not int
        or base_seed < 0
        or not stage
        or type(cycle_index) is not int
        or cycle_index < 0
        or type(sample_id) is not int
        or sample_id <= 0
    ):
        raise PipelineSampleError("RNG identity fields are invalid")
    material = f"{base_seed}\0{stage}\0{cycle_index}\0{sample_id}\0{domain}"
    return random.Random(material).randrange(2**63)


def rng_identity(
    *, base_seed: int, stage: str, cycle_index: int, sample_id: int
) -> RngIdentity:
    return RngIdentity(
        base_seed=base_seed,
        stage=stage,
        cycle_index=cycle_index,
        sample_id=sample_id,
        caption_seed=_domain_seed(base_seed, stage, cycle_index, sample_id, "caption"),
        crop_seed=_domain_seed(base_seed, stage, cycle_index, sample_id, "crop"),
    )


def _metadata(sample: Mapping[str, object]) -> Mapping[str, object]:
    payload = sample.get("json")
    if not isinstance(payload, bytes):
        raise PipelineSampleError(
            "WebDataset sample must contain one JSON byte payload"
        )
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
    """Decode and serialize each sample from service-approved training shards."""

    def __init__(
        self,
        *,
        shard_paths: tuple[Path, ...],
        shard_records: tuple[ShardRecord, ...],
        metadata_adapter: MetadataAdapter,
        metadata_fields: MetadataFieldMapping,
        buckets: tuple[BucketShape, ...],
        min_crop_retention: float,
        probabilities: CaptionDropoutProbabilities,
        tokenizer: TokenEncoder,
        framing: FramingContract,
        caption_fields_parser: CaptionFieldsParser,
        rejection_observer: RejectionObserver,
        base_seed: int,
        stage: str,
        cycle_index: int,
    ) -> None:
        super().__init__()
        _validate_local_shard_paths(shard_paths)
        _validate_shard_records(shard_paths, shard_records)
        if (
            type(buckets) is not tuple
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
            or not isinstance(  # pyright: ignore[reportUnnecessaryIsInstance]
                metadata_fields, MetadataFieldMapping
            )
            or not callable(metadata_adapter)
            or not callable(caption_fields_parser)
            or not callable(rejection_observer)
            or type(base_seed) is not int
            or base_seed < 0
            or type(stage) is not str
            or not stage
            or stage != stage.strip()
            or type(cycle_index) is not int
            or cycle_index < 0
        ):
            raise PipelineSampleError("pipeline construction fields are invalid")
        self.shard_paths = shard_paths
        self.shard_records = shard_records
        self.metadata_adapter = metadata_adapter
        self.metadata_fields = metadata_fields
        self.buckets = buckets
        self.min_crop_retention = min_crop_retention
        self.probabilities = probabilities
        self.tokenizer = tokenizer
        self.framing = framing
        self.caption_fields_parser = caption_fields_parser
        self.rejection_observer = rejection_observer
        self.base_seed = base_seed
        self.stage = stage
        self.cycle_index = cycle_index
        self._lease_managed = False

    def _process(
        self,
        sample: Mapping[str, object],
        records_by_url: Mapping[str, ShardRecord],
    ) -> PipelineSample | None:
        raw_metadata = _metadata(sample)
        sample_url = sample.get("__url__")
        if not isinstance(sample_url, str) or sample_url not in records_by_url:
            raise PipelineSampleError(
                "WebDataset sample source is absent from trusted shard records"
            )
        adapted_metadata = cast(object, self.metadata_adapter(raw_metadata))
        if not isinstance(adapted_metadata, Mapping):
            raise PipelineSampleError("metadata adapter returned an invalid value")
        adapter_mapping = cast(Mapping[object, object], adapted_metadata)
        if not all(isinstance(key, str) for key in adapter_mapping):
            raise PipelineSampleError("metadata adapter keys must be strings")
        mapped_metadata = cast(Mapping[str, object], adapter_mapping)
        shard_record = records_by_url[sample_url]
        metadata: OperationalMetadataRecord = parse_shard_metadata(
            mapped_metadata,
            fields=self.metadata_fields,
        )
        identity = rng_identity(
            base_seed=self.base_seed,
            stage=self.stage,
            cycle_index=self.cycle_index,
            sample_id=metadata.id,
        )
        fields = cast(object, self.caption_fields_parser(raw_metadata))
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
            source_shard=shard_record.path,
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

    def _iter_paths(
        self,
        shard_paths: tuple[Path, ...],
        shard_records: tuple[ShardRecord, ...],
    ) -> Iterator[PipelineSample]:
        _validate_local_shard_paths(shard_paths)
        _validate_shard_records(shard_paths, shard_records)
        urls = [str(path) for path in shard_paths]
        records_by_url = dict(zip(urls, shard_records, strict=True))
        factory = cast(
            Callable[..., Iterable[dict[str, Any]]],
            wds.WebDataset,  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]
        )
        options: dict[str, object] = {
            "shardshuffle": False,
            "empty_check": True,
        }
        if self._lease_managed:
            options["workersplitter"] = None
        dataset = factory(urls, **options)
        for raw_sample in dataset:
            processed = self._process(
                cast(Mapping[str, object], raw_sample), records_by_url
            )
            if processed is not None:
                yield processed

    def _with_local_shards(
        self,
        shard_paths: tuple[Path, ...],
        shard_records: tuple[ShardRecord, ...],
        *,
        cycle_index: int,
    ) -> WebDatasetPipeline:
        """Clone the validated processing contract onto prepared local shards."""

        pipeline = WebDatasetPipeline(
            shard_paths=shard_paths,
            shard_records=shard_records,
            metadata_adapter=self.metadata_adapter,
            metadata_fields=self.metadata_fields,
            buckets=self.buckets,
            min_crop_retention=self.min_crop_retention,
            probabilities=self.probabilities,
            tokenizer=self.tokenizer,
            framing=self.framing,
            caption_fields_parser=self.caption_fields_parser,
            rejection_observer=self.rejection_observer,
            base_seed=self.base_seed,
            stage=self.stage,
            cycle_index=cycle_index,
        )
        pipeline._lease_managed = True
        return pipeline

    def __iter__(self) -> Iterator[PipelineSample]:
        if get_worker_info() is not None and not self._lease_managed:
            raise PipelineSampleError(
                "worker iteration requires the durable shard lease entry point"
            )
        yield from self._iter_paths(self.shard_paths, self.shard_records)


__all__ = [
    "CaptionFieldsParser",
    "ImageAudit",
    "MetadataAdapter",
    "PipelineSample",
    "PipelineSampleError",
    "RejectionObserver",
    "RngIdentity",
    "WebDatasetPipeline",
    "rng_identity",
]
