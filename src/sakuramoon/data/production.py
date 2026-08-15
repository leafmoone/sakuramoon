"""Governed ModelScope metadata and resolved-config data assembly."""

from __future__ import annotations

import multiprocessing.reduction
import os
import re
import secrets
import weakref
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Never, SupportsIndex, cast

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
    CollateError,
    DataLeaseClient,
    TrainingBatch,
    iter_service_batches,
)
from sakuramoon.data.metadata import MetadataFieldMapping
from sakuramoon.data.pipeline import (
    PipelineSampleRejected,
    RejectionObserver,
    WebDatasetPipeline,
)
from sakuramoon.data.serialize import FramingContract, TokenEncoder
from sakuramoon.data.service_protocol import ShardLeaseDescriptor


class ProductionDataError(ValueError):
    """The governed production data boundary cannot be assembled exactly."""


PRODUCTION_METADATA_FIELDS = MetadataFieldMapping(
    id_field="id",
)

_V2_REQUIRED_TOP_LEVEL = frozenset(
    {
        "aesthetic",
        "anime_classification",
        "anime_completeness",
        "captions",
        "dropout",
        "id",
        "image",
        "join",
        "multicaptions",
        "nsfw",
        "quality",
        "rating",
        "schema_version",
        "source",
        "tags",
        "year",
    }
)
_V2_OPTIONAL_TOP_LEVEL = frozenset({"ai_image_corrupted"})
_RATINGS = frozenset({"safe", "general", "questionable", "explicit"})
_NSFW_VALUES = frozenset({"sfw", "questionable", "nsfw"})
_QUALITY_VALUES = frozenset(
    {"masterpiece", "best", "great", "good", "normal", "low", "worst"}
)
_COMPLETENESS_VALUES = frozenset({"polished", "rough", "monochrome"})
_CLASSIFICATION_VALUES = frozenset(
    {"3d", "bangumi", "comic", "illustration", "not_painting"}
)
_YEAR_PATTERN = re.compile(r"^year [0-9]{4}(?:, (?:newest|oldest))?$")
_MULTICAPTION_KEYS = frozenset({"long_names", "long_no_names", "short", "vibes"})


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


def _require_exact_keys(
    mapping: Mapping[str, object], expected: frozenset[str], *, group: str
) -> None:
    observed = frozenset(mapping)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ProductionDataError(
            f"ModelScope metadata {group} fields differ: missing={missing!r} extra={extra!r}"
        )


def _required_text(raw: Mapping[str, object], key: str) -> str:
    if key not in raw:
        raise ProductionDataError(f"ModelScope metadata {key} is required")
    value = raw[key]
    if (
        type(value) is not str
        or not value
        or value != value.strip()
        or "\n" in value
        or "<think>" in value
        or "</think>" in value
    ):
        raise ProductionDataError(
            f"ModelScope metadata {key} must be non-empty boundary text"
        )
    return value


def _tag_from_value(
    raw: Mapping[str, object], key: str, *, allowed: frozenset[str] | None = None
) -> tuple[Tag, ...]:
    value = _required_text(raw, key)
    if allowed is not None and value not in allowed:
        raise ProductionDataError(f"ModelScope metadata {key} has an invalid value")
    return (Tag(text=value, canonical=value),)


def _validate_v2_contract(raw: Mapping[str, object]) -> None:
    observed = frozenset(raw)
    if not _V2_REQUIRED_TOP_LEVEL.issubset(observed) or not observed.issubset(
        _V2_REQUIRED_TOP_LEVEL | _V2_OPTIONAL_TOP_LEVEL
    ):
        missing = sorted(_V2_REQUIRED_TOP_LEVEL - observed)
        extra = sorted(observed - _V2_REQUIRED_TOP_LEVEL - _V2_OPTIONAL_TOP_LEVEL)
        raise ProductionDataError(
            "ModelScope metadata v2 top-level fields differ: "
            f"missing={missing!r} extra={extra!r}"
        )
    if type(raw["schema_version"]) is not int or raw["schema_version"] != 1:
        raise ProductionDataError("ModelScope metadata schema_version must be 1")
    if type(raw["id"]) is not int or cast(int, raw["id"]) <= 0:
        raise ProductionDataError("ModelScope metadata id must be a positive integer")

    source = _nested_mapping(raw, "source")
    _require_exact_keys(
        source,
        frozenset({"dataset", "dataset_version", "release", "original_path"}),
        group="source",
    )
    if (
        source["dataset"] != "danbooru"
        or source["dataset_version"] != "5.9"
        or any(
            type(source[key]) is not str
            or not cast(str, source[key])
            or cast(str, source[key]) != cast(str, source[key]).strip()
            for key in ("release", "original_path")
        )
    ):
        raise ProductionDataError("ModelScope metadata source contract is invalid")

    image = _nested_mapping(raw, "image")
    _require_exact_keys(
        image, frozenset({"format", "width", "height"}), group="image"
    )
    if (
        image["format"] != "webp"
        or type(image["width"]) is not int
        or cast(int, image["width"]) <= 0
        or type(image["height"]) is not int
        or cast(int, image["height"]) <= 0
    ):
        raise ProductionDataError("ModelScope metadata image contract is invalid")

    tags = _nested_mapping(raw, "tags")
    _require_exact_keys(
        tags,
        frozenset({"general", "character", "artist", "copyright"}),
        group="tags",
    )
    captions = _nested_mapping(raw, "captions")
    _require_exact_keys(captions, frozenset({"nl2", "nl3"}), group="captions")
    multicaptions = _nested_mapping(raw, "multicaptions")
    unknown_multicaptions = frozenset(multicaptions) - _MULTICAPTION_KEYS
    if unknown_multicaptions:
        raise ProductionDataError(
            "ModelScope metadata multicaptions fields differ: "
            f"extra={sorted(unknown_multicaptions)!r}"
        )
    join = _nested_mapping(raw, "join")
    _require_exact_keys(
        join,
        frozenset({"multicaptions", "character_records"}),
        group="join",
    )
    if any(value not in {"matched", "missing"} for value in join.values()):
        raise ProductionDataError("ModelScope metadata join values are invalid")
    dropout = _nested_mapping(raw, "dropout")
    _require_exact_keys(
        dropout,
        frozenset({"candidate_tags", "candidate_source", "policy_version"}),
        group="dropout",
    )
    if (
        dropout["candidate_source"] != "popular_tags_intersection"
        or dropout["policy_version"] != "dropout_v1"
    ):
        raise ProductionDataError("ModelScope metadata dropout policy is invalid")

    if "ai_image_corrupted" in raw:
        if raw["ai_image_corrupted"] != "corrupted":
            raise ProductionDataError(
                "ModelScope metadata ai_image_corrupted must be absent or corrupted"
            )
        raise PipelineSampleRejected("ai_image_corrupted")


def adapt_modelscope_metadata(
    raw: Mapping[str, object],
) -> Mapping[str, object]:
    """Project only metadata consumed by training."""

    return {"id": raw.get("id")}


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


def _optional_text(
    mapping: Mapping[str, object], key: str, *, group: str
) -> str | None:
    value = mapping.get(key)
    if value is None:
        return None
    if type(value) is not str:
        raise ProductionDataError(
            f"ModelScope metadata {group}.{key} must be text or null"
        )
    stripped = value.strip()
    return stripped or None


def _short_vibes(multicaptions: Mapping[str, object]) -> str | None:
    short = _optional_text(multicaptions, "short", group="multicaptions")
    vibes = _optional_text(multicaptions, "vibes", group="multicaptions")
    if short is not None and vibes is not None:
        return f"{short}\n\n{vibes}"
    return short or vibes


def parse_modelscope_caption_fields(
    raw: Mapping[str, object],
) -> CaptionFields:
    """Parse the strict v2 metadata contract used for production conditioning."""

    _validate_v2_contract(raw)
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
    year = _required_text(raw, "year")
    if _YEAR_PATTERN.fullmatch(year) is None:
        raise ProductionDataError("ModelScope metadata year has an invalid value")
    year_tags = tuple(
        Tag(text=component, canonical=component) for component in year.split(", ")
    )
    return CaptionFields(
        nsfw=_tag_from_value(raw, "nsfw", allowed=_NSFW_VALUES),
        character=_modelscope_tags(raw, "character"),
        copyright=_modelscope_tags(raw, "copyright"),
        general=_modelscope_tags(raw, "general"),
        artists=_modelscope_tags(raw, "artist"),
        candidate_tags=frozenset(cast(list[str], candidate_items)),
        nl=NlCandidates(
            _optional_text(multicaptions, "long_names", group="multicaptions"),
            _optional_text(multicaptions, "long_no_names", group="multicaptions"),
            _short_vibes(multicaptions),
            _optional_text(captions, "nl2", group="captions"),
            _optional_text(captions, "nl3", group="captions"),
        ),
        rating=_tag_from_value(raw, "rating", allowed=_RATINGS),
        year=year_tags,
        aesthetic=(
            ()
            if raw["aesthetic"] == " "
            else _tag_from_value(raw, "aesthetic")
        ),
        quality=_tag_from_value(raw, "quality", allowed=_QUALITY_VALUES),
        anime_completeness=_tag_from_value(
            raw, "anime_completeness", allowed=_COMPLETENESS_VALUES
        ),
        anime_classification=_tag_from_value(
            raw, "anime_classification", allowed=_CLASSIFICATION_VALUES
        ),
    )


@dataclass(frozen=True, slots=True)
class ConfiguredDataLoader:
    batch_size: int
    worker_count: int
    ready_batches: int
    pin_memory: bool
    drop_last: bool
    length_sort_window_batches: int = 1

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
            length_sort_window_batches=config.data.loader.length_sort_window_batches,
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
            length_sort_window_batches=self.length_sort_window_batches,
        )


@dataclass(frozen=True, slots=True)
class ProductionBatchStreamIdentity:
    """Immutable identities accepted at the production data-to-train boundary."""

    loader: ConfiguredDataLoader
    dataset_id: str
    session_id: str

    def __post_init__(self) -> None:
        if (
            type(self.dataset_id) is not str
            or not self.dataset_id
            or type(self.session_id) is not str
            or not self.session_id
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
        "_ready_batch_depth",
        "_token",
    )

    def __init__(
        self,
        iterator: Iterator[TrainingBatch],
        identity: ProductionBatchStreamIdentity,
        *,
        token: str,
        authority: object,
        ready_batch_depth_snapshot: Callable[[], int] | None = None,
    ) -> None:
        if authority is not _STREAM_AUTHORITY:
            raise ProductionDataError(
                "production batch streams must be issued by the production factory"
            )
        self._iterator = iterator
        self._identity = identity
        inferred_snapshot = getattr(iterator, "ready_batch_depth_snapshot", None)
        snapshot = (
            ready_batch_depth_snapshot
            if ready_batch_depth_snapshot is not None
            else inferred_snapshot
        )
        self._ready_batch_depth = snapshot if callable(snapshot) else None
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

    def ready_batch_depth_snapshot(self) -> int:
        self._require_live()
        if self._ready_batch_depth is None:
            raise ProductionDataError(
                "live DataLoader ready-batch depth is unavailable"
            )
        try:
            depth = self._ready_batch_depth()
        except (CollateError, NotImplementedError, OSError):
            raise ProductionDataError(
                "live DataLoader ready-batch depth is unsupported"
            ) from None
        if type(depth) is not int or depth < 0:
            raise ProductionDataError("live DataLoader ready-batch depth is invalid")
        return depth

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
    *,
    ready_batch_depth_snapshot: Callable[[], int] | None = None,
) -> AcceptedProductionBatchStream:
    token = secrets.token_hex(32)
    stream = AcceptedProductionBatchStream(
        iterator,
        identity,
        token=token,
        authority=_STREAM_AUTHORITY,
        ready_batch_depth_snapshot=ready_batch_depth_snapshot,
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
    tokenizer: TokenEncoder
    framing: FramingContract
    rejection_observer: RejectionObserver
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
            or not callable(self.rejection_observer)
            or type(self.factory_identity) is not str
            or not self.factory_identity
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
        factory = object.__new__(cls)
        object.__setattr__(factory, "config", config)
        object.__setattr__(factory, "tokenizer", tokenizer)
        object.__setattr__(factory, "framing", framing)
        object.__setattr__(
            factory,
            "rejection_observer",
            cast(RejectionObserver, rejection_observer),
        )
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
            tag=dropout.tag,
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
            buckets=buckets,
            min_crop_retention=self.config.data.image.min_crop_retention,
            probabilities=probabilities,
            tokenizer=self.tokenizer,
            framing=self.framing,
            caption_fields_parser=parse_modelscope_caption_fields,
            rejection_observer=self.rejection_observer,
            base_seed=self.config.run.seed,
            stage=self.config.stage.name,
            cycle_index=descriptor.cycle_index,
        )
        _require_spawn_serializable(pipeline, "production pipeline")
        return pipeline

    def batches(self, client: DataLeaseClient) -> AcceptedProductionBatchStream:
        """Consume service leases using only the resolved loader controls."""

        self._require_governed_issuance()
        loader = self.loader
        loader.require_identity(client)
        identity = ProductionBatchStreamIdentity(
            loader=loader,
            dataset_id=client.identity.dataset_id,
            session_id=client.identity.session_id,
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
