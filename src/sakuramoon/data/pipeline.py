"""Bounded local-shard WebDataset processing for one training rank."""

from __future__ import annotations

import io
import json
import math
import os
import random
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
import webdataset as wds
from PIL import Image
from torch.utils.data import IterableDataset, get_worker_info

from sakuramoon.config.schema import DataTransparentBackgroundConfig
from sakuramoon.data.buckets import BucketShape
from sakuramoon.data.caption import (
    CaptionDropoutProbabilities,
    CaptionFields,
    ConditionMode,
    build_caption_plan,
)
from sakuramoon.data.image_ops import (
    ImageRejected,
    normalize_image,
    prepare_image,
)
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
from sakuramoon.data.spatial_crop import (
    ShiftedBucketPlan,
    SpatialCropPolicy,
    plan_shifted_bucket,
)
from sakuramoon.data.transparent_white import (
    TRANSPARENT_REJECTION_KEYS,
    TransparentWhiteOutcome,
    TransparentWhiteTelemetry,
    apply_transparent_white,
)

CaptionFieldsParser = Callable[[Mapping[str, object]], CaptionFields]
MetadataAdapter = Callable[[Mapping[str, object]], Mapping[str, object]]
RejectionObserver = Callable[[str], None]
_IMAGE_KEYS = ("jpg", "jpeg", "png", "webp")
_MAX_DECODE_PIXELS = 100_000_000
_MAX_DECODE_DIMENSION = 32_768
_DRAFT_DECODE_MIN_PIXELS = int(
    os.environ.get("SAKURAMOON_DRAFT_DECODE_MIN_PIXELS", "16000000")
)
_SAMPLE_TRACE_PATH = "/root/sakuramoon-logs/sample-trace.log"

def _trace_sample(source_shard: str, sample_id: int, status: str) -> None:
    try:
        with open(_SAMPLE_TRACE_PATH, "a", encoding="utf-8") as fh:
            fh.write(f"{source_shard}\t{sample_id}\t{status}\n")
    except OSError:
        pass


def _probe_image_dimensions(image_bytes: bytes) -> tuple[int, int]:
    """Read header dimensions without decoding pixel data.

    Used only for the decode guard, whose pixel and dimension limits are
    invariant under EXIF rotation.
    """

    try:
        import imagesize
    except ImportError:
        pass
    else:
        try:
            width, height = imagesize.get(io.BytesIO(image_bytes))
        except (OSError, ValueError, SyntaxError, TypeError):
            width = height = 0
        if width > 0 and height > 0:
            return width, height
    with Image.open(io.BytesIO(image_bytes)) as probe:
        return probe.width, probe.height


def _source_dimensions(decoded: Image.Image) -> tuple[int, int]:
    """True post-EXIF dimensions matching ``ImageOps.exif_transpose``."""

    width, height = decoded.size
    if decoded.getexif().get(0x0112, 1) in (5, 6, 7, 8):
        width, height = height, width
    return width, height


def _draft_request_size(buckets: tuple[BucketShape, ...]) -> tuple[int, int]:
    """Request a reduced JPEG decode at twice the largest training bucket."""

    return (
        2 * max(shape.width for shape in buckets),
        2 * max(shape.height for shape in buckets),
    )


def _should_draft_decode(
    decoded: Image.Image, source_width: int, source_height: int
) -> bool:
    """Draft-scale only large JPEGs; other formats always decode in full."""

    if decoded.format != "JPEG":
        return False
    return source_width * source_height >= _DRAFT_DECODE_MIN_PIXELS


class PipelineSampleError(ValueError):
    """A WebDataset sample cannot satisfy the training input contract."""


class PipelineSampleRejected(Exception):
    """A valid metadata policy explicitly excludes one sample from training."""

    def __init__(self, reason: str) -> None:
        if (
            type(reason) is not str
            or not reason
            or reason != reason.strip()
            or any(character.isspace() for character in reason)
        ):
            raise PipelineSampleError("sample rejection reason is invalid")
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class RngIdentity:
    base_seed: int
    stage: str
    cycle_index: int
    sample_id: int
    caption_seed: int
    crop_seed: int
    spatial_policy_seed: int = 0
    spatial_zoom_seed: int = 0
    spatial_offset_x_seed: int = 0
    spatial_offset_y_seed: int = 0


@dataclass(frozen=True)
class ImageAudit:
    source_width: int
    source_height: int
    resized_width: int
    resized_height: int
    crop_box: tuple[int, int, int, int]
    crop_retention: float
    crop_policy: str = "aspect_bucket"
    spatial_selected: bool = False
    spatial_applied: bool = False
    spatial_fallback_reason: str = "none"
    base_crop_retention: float = 0.0
    final_crop_retention: float = 0.0
    requested_equivalent_zoom: float = 0.0
    actual_equivalent_zoom: float = 0.0
    normalized_offset_x: float = 0.0
    normalized_offset_y: float = 0.0


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
    # Per-sample transparent-white outcome.  Retained samples only carry
    # NOT_TAGGED or COMPOSITED (rejects never reach this object); the default
    # keeps the ordinary (untagged) path bit-identical.
    transparent_outcome: TransparentWhiteOutcome = TransparentWhiteOutcome.NOT_TAGGED


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
        spatial_policy_seed=_domain_seed(
            base_seed, stage, cycle_index, sample_id, "spatial-policy"
        ),
        spatial_zoom_seed=_domain_seed(
            base_seed, stage, cycle_index, sample_id, "spatial-zoom"
        ),
        spatial_offset_x_seed=_domain_seed(
            base_seed, stage, cycle_index, sample_id, "spatial-offset-x"
        ),
        spatial_offset_y_seed=_domain_seed(
            base_seed, stage, cycle_index, sample_id, "spatial-offset-y"
        ),
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

    condition_mode: ConditionMode

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
        condition_mode: ConditionMode,
        tokenizer: TokenEncoder,
        framing: FramingContract,
        caption_fields_parser: CaptionFieldsParser,
        rejection_observer: RejectionObserver,
        base_seed: int,
        stage: str,
        cycle_index: int,
        spatial_policy: SpatialCropPolicy | None = None,
        transparent_policy: DataTransparentBackgroundConfig | None = None,
        transparent_telemetry: TransparentWhiteTelemetry | None = None,
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
            or type(condition_mode) is not str
            or condition_mode not in {"artist", "artist_or_character"}
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
            or not (
                spatial_policy is None
                or isinstance(spatial_policy, SpatialCropPolicy)  # pyright: ignore[reportUnnecessaryIsInstance]
            )
            or not (
                transparent_policy is None
                or isinstance(transparent_policy, DataTransparentBackgroundConfig)  # pyright: ignore[reportUnnecessaryIsInstance]
            )
        ):
            raise PipelineSampleError("pipeline construction fields are invalid")
        self.shard_paths = shard_paths
        self.shard_records = shard_records
        self.metadata_adapter = metadata_adapter
        self.metadata_fields = metadata_fields
        self.buckets = buckets
        self.min_crop_retention = min_crop_retention
        self.probabilities = probabilities
        self.condition_mode = condition_mode
        self.tokenizer = tokenizer
        self.framing = framing
        self.caption_fields_parser = caption_fields_parser
        self.rejection_observer = rejection_observer
        self.base_seed = base_seed
        self.stage = stage
        self.cycle_index = cycle_index
        self.spatial_policy = spatial_policy
        self.transparent_policy = transparent_policy
        self.transparent_telemetry = (
            transparent_telemetry if transparent_telemetry is not None else TransparentWhiteTelemetry()
        )
        # Per-shard reject counters of the reliable worker->parent channel.
        # Each lease pipeline (and its local-shard clone) owns one fixed-key
        # dict; the counters ride the shard completion message, so a parent
        # can audit rejected samples that never produce a PipelineSample.
        self._shard_transparent_rejections: dict[str, int] = dict.fromkeys(
            TRANSPARENT_REJECTION_KEYS, 0
        )
        self._lease_managed = False

    def _process(
        self,
        sample: Mapping[str, object],
        records_by_url: Mapping[str, ShardRecord],
    ) -> PipelineSample | None:
        raw_metadata = _metadata(sample)
        sample_url = sample.get("__url__")
        sample_key = sample.get("__key__")
        if (
            not isinstance(sample_url, str)
            or sample_url not in records_by_url
            or not isinstance(sample_key, str)
            or not sample_key
        ):
            raise PipelineSampleError(
                "WebDataset sample source/key is absent from trusted shard records"
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
            sample_key=f"{sample_url}\0{sample_key}",
        )
        identity = rng_identity(
            base_seed=self.base_seed,
            stage=self.stage,
            cycle_index=self.cycle_index,
            sample_id=metadata.id,
        )
        try:
            fields = cast(object, self.caption_fields_parser(raw_metadata))
        except PipelineSampleRejected as rejected:
            # Per-sample skip logging is intentionally omitted (production
            # log-cadence reduction, 2026-09-04): rejections are still
            # accounted for via _trace_sample + rejection_observer.
            _trace_sample(
                shard_record.path,
                metadata.id,
                f"reject:{rejected.reason}",
            )
            self.rejection_observer(rejected.reason)
            return None
        if not isinstance(fields, CaptionFields):
            raise PipelineSampleError("caption field parser returned an invalid value")
        image_bytes = _image_bytes(sample)
        try:
            probe_width, probe_height = _probe_image_dimensions(image_bytes)
            if (
                probe_width * probe_height > _MAX_DECODE_PIXELS
                or max(probe_width, probe_height) > _MAX_DECODE_DIMENSION
            ):
                # Extreme decode-cost ceiling: beyond this, full decode spikes
                # worker CPU/RAM for a negligible number of samples. The earlier
                # training-stall hypothesis was disproven (NCCL comms), so these
                # images are otherwise trainable after resize like any other.
                print(
                    f"[data] skip pathological image "
                    f"{probe_width}x{probe_height} "
                    f"pixels={probe_width * probe_height}",
                    flush=True,
                )
                _trace_sample(shard_record.path, metadata.id, "skip_oversized")
                self.rejection_observer("decode_error")
                return None
            with Image.open(io.BytesIO(image_bytes)) as decoded:
                source_width, source_height = _source_dimensions(decoded)
                if _should_draft_decode(decoded, probe_width, probe_height):
                    decoded.draft("RGB", _draft_request_size(self.buckets))
                decoded.load()
                image_mode = decoded.mode
                # Transparent-background white-composite policy (sections 6-12):
                # alpha-first, applied BEFORE resize/crop. The ordinary (untagged)
                # path is bit-identical: NOT_TAGGED keeps the decoded image and the
                # original caption fields untouched.
                if self.transparent_policy is not None:
                    tw = apply_transparent_white(
                        decoded, fields, self.transparent_policy
                    )
                    self.transparent_telemetry.record(tw.outcome)
                    if tw.outcome.is_reject:
                        reason = tw.outcome.observer_reason
                        assert reason is not None  # reject outcomes always carry a reason
                        assert reason in self._shard_transparent_rejections
                        self._shard_transparent_rejections[reason] += 1
                        _trace_sample(
                            shard_record.path, metadata.id, f"reject:{reason}"
                        )
                        self.rejection_observer(reason)
                        return None
                    if tw.outcome is TransparentWhiteOutcome.COMPOSITED:
                        assert tw.image is not None
                        work_image = tw.image
                        work_fields = tw.fields
                        transparent_outcome = tw.outcome
                    else:
                        work_image = decoded
                        work_fields = fields
                        transparent_outcome = tw.outcome
                else:
                    work_image = decoded
                    work_fields = fields
                    transparent_outcome = TransparentWhiteOutcome.NOT_TAGGED
            plan = build_caption_plan(
                work_fields,
                self.probabilities,
                condition_mode=self.condition_mode,
                seed=identity.caption_seed,
            )
            caption = serialize_caption(plan, self.tokenizer, self.framing)
            processed = prepare_image(
                work_image,
                self.buckets,
                min_crop_retention=self.min_crop_retention,
                crop_seed=identity.crop_seed,
                source_size=(source_width, source_height),
            )
            spatial_plan: ShiftedBucketPlan | None = None
            spatial_image: Image.Image | None = None
            if self.spatial_policy is not None and self.spatial_policy.enabled:
                spatial_plan = plan_shifted_bucket(
                    processed.assignment,
                    self.spatial_policy,
                    source_size=(source_width, source_height),
                    policy_seed=identity.spatial_policy_seed,
                    zoom_seed=identity.spatial_zoom_seed,
                    offset_x_seed=identity.spatial_offset_x_seed,
                    offset_y_seed=identity.spatial_offset_y_seed,
                )
                if spatial_plan.applied:
                    normalized = normalize_image(work_image)
                    spatial_image = normalized.resize(
                        (spatial_plan.canvas_width, spatial_plan.canvas_height),
                        resample=Image.Resampling.LANCZOS,
                    ).crop(spatial_plan.crop_box)
        except ImageRejected as error:
            _trace_sample(shard_record.path, metadata.id, f"reject:{error.reason}")
            self.rejection_observer(error.reason)
            return None
        except (OSError, SyntaxError, ValueError, Image.DecompressionBombError):
            # A corrupt image is an individual bad sample, not a failed data
            # service. Drop it so one malformed archive member cannot abort
            # an otherwise healthy training run.
            _trace_sample(shard_record.path, metadata.id, "decode_error")
            self.rejection_observer("decode_error")
            return None
        assignment = processed.assignment
        if spatial_plan is not None and spatial_image is not None:
            sample_image: Image.Image = spatial_image
            audit = ImageAudit(
                source_width=assignment.source_width,
                source_height=assignment.source_height,
                resized_width=spatial_plan.canvas_width,
                resized_height=spatial_plan.canvas_height,
                crop_box=spatial_plan.crop_box,
                crop_retention=spatial_plan.final_crop_retention,
                crop_policy="shifted_bucket",
                spatial_selected=True,
                spatial_applied=True,
                spatial_fallback_reason="none",
                base_crop_retention=spatial_plan.base_crop_retention,
                final_crop_retention=spatial_plan.final_crop_retention,
                requested_equivalent_zoom=spatial_plan.requested_equivalent_zoom,
                actual_equivalent_zoom=spatial_plan.actual_equivalent_zoom,
                normalized_offset_x=spatial_plan.normalized_offset_x,
                normalized_offset_y=spatial_plan.normalized_offset_y,
            )
        else:
            # Ordinary aspect-bucket crop: bit-identical to the pre-spatial
            # behavior when no policy is enabled, and the literal fallback
            # when the policy is enabled but this sample was not selected or
            # the shifted-bucket geometry is infeasible.
            sample_image = processed.image
            normal_zoom = 1.0 / math.sqrt(assignment.crop_retention)
            if spatial_plan is None:
                selected = False
                fallback_reason = "none"
                requested_zoom = 0.0
            else:
                selected = spatial_plan.fallback_reason != "not_selected"
                fallback_reason = spatial_plan.fallback_reason
                requested_zoom = spatial_plan.requested_equivalent_zoom
            audit = ImageAudit(
                source_width=assignment.source_width,
                source_height=assignment.source_height,
                resized_width=assignment.resized_width,
                resized_height=assignment.resized_height,
                crop_box=processed.crop_box,
                crop_retention=assignment.crop_retention,
                crop_policy="aspect_bucket",
                spatial_selected=selected,
                spatial_applied=False,
                spatial_fallback_reason=fallback_reason,
                base_crop_retention=assignment.crop_retention,
                final_crop_retention=assignment.crop_retention,
                requested_equivalent_zoom=requested_zoom,
                actual_equivalent_zoom=normal_zoom,
                normalized_offset_x=0.0,
                normalized_offset_y=0.0,
            )
        _trace_sample(
            shard_record.path,
            metadata.id,
            "ok"
            + f" mode={image_mode}"
            + f" src={assignment.source_width}x{assignment.source_height}"
            + f" resized={assignment.resized_width}x{assignment.resized_height}"
            + f" bucket={assignment.bucket.width}x{assignment.bucket.height}"
            + f" cap={len(caption.input_ids)}"
            + (
                f" spatial={spatial_plan.fallback_reason}"
                if spatial_plan is not None
                else ""
            ),
        )
        return PipelineSample(
            sample_id=metadata.id,
            source_shard=shard_record.path,
            image=_uint8_chw(sample_image),
            target_height=assignment.bucket.height,
            target_width=assignment.bucket.width,
            caption=caption,
            audit=audit,
            rng=identity,
            padding_token_id=self.framing.padding_token_id,
            transparent_outcome=transparent_outcome,
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
            # The data service already assigns disjoint shards to every global
            # rank/worker pair. WebDataset must not split the one-shard lease
            # again based on RANK/WORLD_SIZE inherited by spawned workers.
            options["nodesplitter"] = None
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
            condition_mode=self.condition_mode,
            tokenizer=self.tokenizer,
            framing=self.framing,
            caption_fields_parser=self.caption_fields_parser,
            rejection_observer=self.rejection_observer,
            base_seed=self.base_seed,
            stage=self.stage,
            cycle_index=cycle_index,
            spatial_policy=self.spatial_policy,
            transparent_policy=self.transparent_policy,
            transparent_telemetry=self.transparent_telemetry,
        )
        pipeline._lease_managed = True
        return pipeline

    def transparent_rejection_counts(self) -> dict[str, int]:
        """Fixed-key reject counters of this pipeline's current shard.

        Call after the shard's sample stream ends: the returned snapshot
        carries the three reliable-channel reject keys (zero-filled) so a
        shard completion can audit rejected samples that never produced a
        :class:`PipelineSample`.  A conservation check against the shared
        telemetry counters runs first so a corrupted counter pair cannot
        be published.
        """
        self.transparent_telemetry.assert_conservation()
        return {
            key: self._shard_transparent_rejections[key]
            for key in TRANSPARENT_REJECTION_KEYS
        }

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
    "PipelineSampleRejected",
    "RejectionObserver",
    "RngIdentity",
    "WebDatasetPipeline",
    "rng_identity",
]
