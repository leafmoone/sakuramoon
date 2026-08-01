"""Governed ModelScope metadata and resolved-config data assembly."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from sakuramoon.config.schema import RuntimeConfig
from sakuramoon.data.buckets import generate_base_buckets, scale_buckets
from sakuramoon.data.caption import (
    CaptionDropoutProbabilities,
    CaptionFields,
    NlCandidates,
    NlDropoutProbabilities,
    Tag,
)
from sakuramoon.data.collate import (
    DataLeaseClient,
    TrainingBatch,
    iter_service_batches,
)
from sakuramoon.data.metadata import MetadataFieldMapping
from sakuramoon.data.pipeline import RejectionObserver, WebDatasetPipeline
from sakuramoon.data.serialize import FramingContract, TokenEncoder
from sakuramoon.data.service_protocol import ShardLeaseDescriptor
from sakuramoon.data.validation import load_validation_manifest_ids


class ProductionDataError(ValueError):
    """The governed production data boundary cannot be assembled exactly."""


PRODUCTION_METADATA_FIELDS = MetadataFieldMapping(
    id_field="id",
    width_field="width",
    height_field="height",
    caption_available_field="caption_available",
)


def _nested_mapping(raw: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = raw.get(key)
    if not isinstance(value, Mapping):
        raise ProductionDataError(f"ModelScope metadata {key} must be an object")
    mapping = cast(Mapping[object, object], value)
    if not all(type(item) is str for item in mapping):
        raise ProductionDataError(f"ModelScope metadata {key} keys must be strings")
    return cast(Mapping[str, object], mapping)


def adapt_modelscope_metadata(
    raw: Mapping[str, object],
) -> Mapping[str, object]:
    """Project the real nested dataset fields used by D021 trusted parsing."""

    image = _nested_mapping(raw, "image")
    captions = _nested_mapping(raw, "captions")
    multicaptions = _nested_mapping(raw, "multicaptions")
    caption_available = any(
        type(value) is str and bool(value.strip())
        for value in (*captions.values(), *multicaptions.values())
    )
    return {
        "id": raw.get("id"),
        "width": image.get("width"),
        "height": image.get("height"),
        "caption_available": caption_available,
    }


def _modelscope_tags(raw: Mapping[str, object], key: str) -> tuple[Tag, ...]:
    tags = _nested_mapping(raw, "tags").get(key)
    if type(tags) is not list:
        raise ProductionDataError(f"ModelScope metadata tags.{key} must be a list")
    items = cast(list[object], tags)
    if not all(type(item) is str for item in items):
        raise ProductionDataError(
            f"ModelScope metadata tags.{key} must contain only strings"
        )
    return tuple(Tag(text=item, canonical=item) for item in cast(list[str], items))


def _optional_text(mapping: Mapping[str, object], key: str) -> str | None:
    value = mapping.get(key)
    return value if type(value) is str and bool(value.strip()) else None


def parse_modelscope_caption_fields(
    raw: Mapping[str, object],
) -> CaptionFields:
    """Parse the governed real-dataset caption projection used by D014."""

    nsfw = raw.get("nsfw")
    if type(nsfw) is not str:
        raise ProductionDataError("ModelScope metadata nsfw must be a string")
    captions = _nested_mapping(raw, "captions")
    multicaptions = _nested_mapping(raw, "multicaptions")
    dropout = _nested_mapping(raw, "dropout")
    candidate_tags = dropout.get("candidate_tags")
    if type(candidate_tags) is not list:
        raise ProductionDataError(
            "ModelScope metadata dropout.candidate_tags must be a list"
        )
    candidate_items = cast(list[object], candidate_tags)
    if not all(type(item) is str for item in candidate_items):
        raise ProductionDataError(
            "ModelScope metadata dropout.candidate_tags must contain only strings"
        )
    return CaptionFields(
        nsfw=(Tag(nsfw, nsfw),),
        character=_modelscope_tags(raw, "character"),
        copyright=_modelscope_tags(raw, "copyright"),
        general=_modelscope_tags(raw, "general"),
        artists=_modelscope_tags(raw, "artist"),
        candidate_tags=frozenset(cast(list[str], candidate_items)),
        nl=NlCandidates(
            None,
            None,
            _optional_text(multicaptions, "vibes"),
            _optional_text(captions, "nl2"),
            _optional_text(captions, "nl3"),
        ),
    )


@dataclass(frozen=True, slots=True)
class ConfiguredDataLoader:
    batch_size: int
    worker_count: int
    ready_batches: int
    pin_memory: bool
    drop_last: bool

    @classmethod
    def from_config(cls, config: RuntimeConfig) -> ConfiguredDataLoader:
        if not isinstance(  # pyright: ignore[reportUnnecessaryIsInstance]
            config, RuntimeConfig
        ):
            raise ProductionDataError("resolved RuntimeConfig is required")
        return cls(
            batch_size=config.stage.local_batch,
            worker_count=config.data.cache.persistent_workers_per_rank,
            ready_batches=config.data.cache.ready_batches_per_rank,
            pin_memory=config.data.loader.pin_memory,
            drop_last=config.data.loader.drop_last,
        )

    def require_identity(self, client: DataLeaseClient) -> None:
        if client.identity.worker_count != self.worker_count:
            raise ProductionDataError(
                "resolved worker_count does not match the data service identity"
            )

    def batches(
        self, pipeline: WebDatasetPipeline, client: DataLeaseClient
    ) -> Iterator[TrainingBatch]:
        self.require_identity(client)
        return iter_service_batches(
            pipeline,
            client,
            batch_size=self.batch_size,
            worker_count=self.worker_count,
            ready_batches=self.ready_batches,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
        )


class _PreleasedClient:
    def __init__(
        self, delegate: DataLeaseClient, descriptor: ShardLeaseDescriptor
    ) -> None:
        self._delegate = delegate
        self._descriptor: ShardLeaseDescriptor | None = descriptor
        self.identity = delegate.identity

    def health(self) -> bool:
        return False

    def lease(self, worker_id: int) -> ShardLeaseDescriptor | None:
        if worker_id == 0 and self._descriptor is not None:
            descriptor, self._descriptor = self._descriptor, None
            return descriptor
        return self._delegate.lease(worker_id)

    def acknowledge(self, descriptor: ShardLeaseDescriptor) -> None:
        self._delegate.acknowledge(descriptor)


@dataclass(frozen=True, slots=True)
class ProductionPipelineFactory:
    config: RuntimeConfig
    validation_ids: frozenset[int]
    tokenizer: TokenEncoder
    framing: FramingContract
    rejection_observer: RejectionObserver
    pass_index: int

    def __post_init__(self) -> None:
        if (
            not isinstance(  # pyright: ignore[reportUnnecessaryIsInstance]
                self.config, RuntimeConfig
            )
            or type(self.validation_ids) is not frozenset
            or len(self.validation_ids) != self.config.data.validation.sample_count
            or any(
                type(value) is not int or value <= 0 for value in self.validation_ids
            )
            or not callable(self.rejection_observer)
            or type(self.pass_index) is not int
            or self.pass_index < 0
        ):
            raise ProductionDataError("production pipeline factory fields are invalid")

    @classmethod
    def from_config(
        cls,
        config: RuntimeConfig,
        *,
        repository_root: Path,
        tokenizer: TokenEncoder,
        framing: FramingContract,
        rejection_observer: Callable[[str], None],
        pass_index: int,
    ) -> ProductionPipelineFactory:
        if (
            not isinstance(  # pyright: ignore[reportUnnecessaryIsInstance]
                repository_root, Path
            )
            or not repository_root.is_absolute()
        ):
            raise ProductionDataError("repository_root must be an absolute path")
        configured_path = Path(config.data.validation.manifest_path)
        manifest_path = (
            configured_path
            if configured_path.is_absolute()
            else repository_root / configured_path
        )
        validation_ids = load_validation_manifest_ids(
            manifest_path,
            expected_sha256=config.data.validation.manifest_sha256,
            expected_count=config.data.validation.sample_count,
        )
        return cls(
            config=config,
            validation_ids=validation_ids,
            tokenizer=tokenizer,
            framing=framing,
            rejection_observer=cast(RejectionObserver, rejection_observer),
            pass_index=pass_index,
        )

    @property
    def loader(self) -> ConfiguredDataLoader:
        return ConfiguredDataLoader.from_config(self.config)

    def pipeline_for_lease(
        self, descriptor: ShardLeaseDescriptor
    ) -> WebDatasetPipeline:
        dropout = self.config.caption.dropout
        probabilities = CaptionDropoutProbabilities(
            nsfw=dropout.nsfw,
            character=dropout.character,
            copyright=dropout.copyright,
            general=dropout.general,
            artist=dropout.artist,
            candidate_source=dropout.candidate_source,
            nl=NlDropoutProbabilities(
                long_names=dropout.nl.long_names,
                long_no_names=dropout.nl.long_no_names,
                short_vibes=dropout.nl.short_vibes,
                nl2=dropout.nl.nl2,
                nl3=dropout.nl.nl3,
            ),
        )
        buckets = scale_buckets(
            generate_base_buckets(self.config.data.buckets),
            self.config.stage.resolution,
        )
        return WebDatasetPipeline(
            shard_paths=(descriptor.local_path,),
            shard_records=(descriptor.record,),
            metadata_adapter=adapt_modelscope_metadata,
            metadata_fields=PRODUCTION_METADATA_FIELDS,
            validation_ids=self.validation_ids,
            buckets=buckets,
            min_crop_retention=self.config.data.image.min_crop_retention,
            probabilities=probabilities,
            tokenizer=self.tokenizer,
            framing=self.framing,
            caption_fields_parser=parse_modelscope_caption_fields,
            rejection_observer=self.rejection_observer,
            base_seed=self.config.run.seed,
            stage=self.config.stage.name,
            pass_index=self.pass_index,
        )

    def batches(self, client: DataLeaseClient) -> Iterator[TrainingBatch]:
        """Consume service leases using only the five resolved loader controls."""

        loader = self.loader
        loader.require_identity(client)
        if client.health():
            return iter(())
        descriptor = client.lease(0)
        if descriptor is None:
            return iter(())
        return loader.batches(
            self.pipeline_for_lease(descriptor),
            _PreleasedClient(client, descriptor),
        )


__all__ = [
    "PRODUCTION_METADATA_FIELDS",
    "ConfiguredDataLoader",
    "ProductionDataError",
    "ProductionPipelineFactory",
    "adapt_modelscope_metadata",
    "parse_modelscope_caption_fields",
]
