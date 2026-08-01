"""Governed ModelScope metadata and resolved-config data assembly."""

from __future__ import annotations

import multiprocessing.reduction
import os
import secrets
import weakref
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Never, SupportsIndex, cast

from sakuramoon.config.resolve import resolved_config_sha256
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


def _require_spawn_serializable(value: object, name: str) -> None:
    """Exercise the exact pickler used to launch spawned DataLoader workers."""

    try:
        multiprocessing.reduction.ForkingPickler.dumps(value)
    except Exception as error:
        raise ProductionDataError(
            f"{name} must be serializable by the explicit spawn context"
        ) from error


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


@dataclass(frozen=True, slots=True)
class ProductionBatchStreamIdentity:
    """Immutable identities accepted at the production data-to-train boundary."""

    resolved_config_sha256: str
    loader: ConfiguredDataLoader
    manifest_sha256: str
    service_session_sha256: str
    factory_identity: str

    def __post_init__(self) -> None:
        digests = (
            self.resolved_config_sha256,
            self.manifest_sha256,
            self.service_session_sha256,
            self.factory_identity,
        )
        if (
            any(
                type(value) is not str
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
                for value in digests
            )
            or not isinstance(  # pyright: ignore[reportUnnecessaryIsInstance]
                self.loader, ConfiguredDataLoader
            )
        ):
            raise ProductionDataError("production batch stream identity is invalid")


class AcceptedProductionBatchStream(Iterator[TrainingBatch]):
    """Factory-issued process-local handle around the production batch iterator."""

    __slots__ = (
        "__weakref__",
        "_closed",
        "_identity",
        "_iterator",
        "_owner_pid",
        "_token",
    )

    def __init__(
        self,
        iterator: Iterator[TrainingBatch],
        identity: ProductionBatchStreamIdentity,
        *,
        token: str,
        authority: object,
    ) -> None:
        if authority is not _STREAM_AUTHORITY:
            raise ProductionDataError(
                "production batch streams must be issued by the production factory"
            )
        self._iterator = iterator
        self._identity = identity
        self._token = token
        self._owner_pid = os.getpid()
        self._closed = False

    def _require_live(self) -> None:
        if os.getpid() != self._owner_pid:
            raise ProductionDataError(
                "production batch stream cannot cross a process boundary"
            )
        if self._closed or _ACCEPTED_STREAMS.get(self._token) is not self:
            raise ProductionDataError(
                "production batch stream is closed or was not factory-issued"
            )

    @property
    def identity(self) -> ProductionBatchStreamIdentity:
        self._require_live()
        return self._identity

    def __iter__(self) -> AcceptedProductionBatchStream:
        self._require_live()
        return self

    def __next__(self) -> TrainingBatch:
        self._require_live()
        try:
            return next(self._iterator)
        except StopIteration:
            self._closed = True
            _ACCEPTED_STREAMS.pop(self._token, None)
            raise
        except BaseException:
            # Retire acceptance immediately, but leave close() responsible for
            # deterministic cleanup of the owned iterator in the caller's finally.
            _ACCEPTED_STREAMS.pop(self._token, None)
            raise

    def close(self) -> None:
        if os.getpid() != self._owner_pid:
            raise ProductionDataError(
                "production batch stream cannot cross a process boundary"
            )
        if self._closed:
            return
        try:
            close = getattr(self._iterator, "close", None)
            if callable(close):
                close()
        finally:
            self._closed = True
            _ACCEPTED_STREAMS.pop(self._token, None)

    def __reduce_ex__(self, protocol: SupportsIndex) -> Never:
        del protocol
        raise ProductionDataError(
            "production batch stream is process-local and cannot be serialized"
        )


_STREAM_AUTHORITY = object()
_ACCEPTED_STREAMS: weakref.WeakValueDictionary[
    str, AcceptedProductionBatchStream
] = weakref.WeakValueDictionary()


def _issue_batch_stream(
    iterator: Iterator[TrainingBatch],
    identity: ProductionBatchStreamIdentity,
) -> AcceptedProductionBatchStream:
    token = secrets.token_hex(32)
    stream = AcceptedProductionBatchStream(
        iterator,
        identity,
        token=token,
        authority=_STREAM_AUTHORITY,
    )
    _ACCEPTED_STREAMS[token] = stream
    return stream


def require_accepted_production_batch_stream(
    value: object,
) -> AcceptedProductionBatchStream:
    """Reject plain iterators and caller-built batches at the production boundary."""

    if not isinstance(value, AcceptedProductionBatchStream):
        raise ProductionDataError(
            "a factory-issued production batch stream is required"
        )
    value._require_live()  # pyright: ignore[reportPrivateUsage]
    return value


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


@dataclass(frozen=True, slots=True, init=False, weakref_slot=True)
class ProductionPipelineFactory:
    config: RuntimeConfig
    validation_ids: frozenset[int]
    tokenizer: TokenEncoder
    framing: FramingContract
    rejection_observer: RejectionObserver
    pass_index: int
    factory_identity: str
    _owner_pid: int

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise ProductionDataError(
            "production pipeline factories must be issued by from_config"
        )

    def _validate_fields(self) -> None:
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
            or type(self.factory_identity) is not str
            or len(self.factory_identity) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.factory_identity
            )
        ):
            raise ProductionDataError("production pipeline factory fields are invalid")
        _require_spawn_serializable(self, "production pipeline factory")

    def _require_governed_issuance(self) -> None:
        try:
            factory_identity = self.factory_identity
            owner_pid = self._owner_pid
        except AttributeError as error:
            raise ProductionDataError(
                "production pipeline factory was not issued by from_config"
            ) from error
        if (
            os.getpid() != owner_pid
            or _GOVERNED_FACTORIES.get(factory_identity) is not self
        ):
            raise ProductionDataError(
                "production pipeline factory was not issued by from_config in this process"
            )

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
        if cls is not ProductionPipelineFactory:
            raise ProductionDataError(
                "production pipeline factories must use the governed concrete class"
            )
        if not isinstance(  # pyright: ignore[reportUnnecessaryIsInstance]
            config, RuntimeConfig
        ):
            raise ProductionDataError("resolved RuntimeConfig is required")
        if not isinstance(  # pyright: ignore[reportUnnecessaryIsInstance]
            repository_root, Path
        ) or not repository_root.is_absolute():
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
        factory = object.__new__(cls)
        object.__setattr__(factory, "config", config)
        object.__setattr__(factory, "validation_ids", validation_ids)
        object.__setattr__(factory, "tokenizer", tokenizer)
        object.__setattr__(factory, "framing", framing)
        object.__setattr__(
            factory,
            "rejection_observer",
            cast(RejectionObserver, rejection_observer),
        )
        object.__setattr__(factory, "pass_index", pass_index)
        object.__setattr__(factory, "factory_identity", secrets.token_hex(32))
        object.__setattr__(factory, "_owner_pid", os.getpid())
        factory._validate_fields()
        _GOVERNED_FACTORIES[factory.factory_identity] = factory
        return factory

    @property
    def loader(self) -> ConfiguredDataLoader:
        self._require_governed_issuance()
        return ConfiguredDataLoader.from_config(self.config)

    def pipeline_for_lease(
        self, descriptor: ShardLeaseDescriptor
    ) -> WebDatasetPipeline:
        self._require_governed_issuance()
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
        pipeline = WebDatasetPipeline(
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
        _require_spawn_serializable(pipeline, "production pipeline")
        return pipeline

    def batches(self, client: DataLeaseClient) -> AcceptedProductionBatchStream:
        """Consume service leases using only the five resolved loader controls."""

        self._require_governed_issuance()
        loader = self.loader
        loader.require_identity(client)
        if client.identity.manifest_sha256 != self.config.data.manifest.sha256:
            raise ProductionDataError(
                "resolved manifest identity does not match the data service session"
            )
        identity = ProductionBatchStreamIdentity(
            resolved_config_sha256=resolved_config_sha256(self.config),
            loader=loader,
            manifest_sha256=self.config.data.manifest.sha256,
            service_session_sha256=client.identity.sha256,
            factory_identity=self.factory_identity,
        )
        if client.health():
            return _issue_batch_stream(iter(()), identity)
        descriptor = client.lease(0)
        if descriptor is None:
            return _issue_batch_stream(iter(()), identity)
        return _issue_batch_stream(
            loader.batches(
                self.pipeline_for_lease(descriptor),
                _PreleasedClient(client, descriptor),
            ),
            identity,
        )


_GOVERNED_FACTORIES: weakref.WeakValueDictionary[
    str, ProductionPipelineFactory
] = weakref.WeakValueDictionary()


__all__ = [
    "PRODUCTION_METADATA_FIELDS",
    "AcceptedProductionBatchStream",
    "ConfiguredDataLoader",
    "ProductionBatchStreamIdentity",
    "ProductionDataError",
    "ProductionPipelineFactory",
    "adapt_modelscope_metadata",
    "parse_modelscope_caption_fields",
    "require_accepted_production_batch_stream",
]
