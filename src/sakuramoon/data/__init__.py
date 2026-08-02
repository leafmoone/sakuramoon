"""Lazy public data API that does not import service-owned transport in trainers."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sakuramoon.data.manifest import (
        DatasetManifest,
        DatasetManifestError,
        DatasetSourceIdentity,
        ShardRecord,
        load_dataset_manifest,
    )
    from sakuramoon.data.metadata import (
        DuplicateIdReport,
        MetadataError,
        MetadataRecord,
        OperationalMetadataRecord,
        parse_metadata,
        scan_duplicate_ids,
    )
    from sakuramoon.data.modelscope import (
        DatasetAuthenticationError,
        DatasetTransportError,
        FetchedShard,
        ModelScopeDatasetTransport,
        ShardIntegrityError,
        ensure_dataset_manifest,
        fetch_dataset_shard,
        validate_remote_manifest,
    )
    from sakuramoon.data.production import (
        PRODUCTION_METADATA_FIELDS,
        AcceptedProductionBatchStream,
        ConfiguredDataLoader,
        ProductionBatchStreamIdentity,
        ProductionDataError,
        ProductionPipelineFactory,
        adapt_modelscope_metadata,
        parse_modelscope_caption_fields,
        require_accepted_production_batch_stream,
    )
    from sakuramoon.data.validation import (
        VALIDATION_SELECTION_SEED,
        VALIDATION_SHARD_COUNT,
        VALIDATION_SHARD_PATHS,
        PreparedValidationShards,
        ValidationPromptError,
        ValidationPromptSample,
        ValidationSelection,
        ValidationSelectionError,
        ValidationSelectionExistsError,
        canonical_validation_selection_bytes,
        ensure_validation_selection,
        load_validation_prompt_samples,
        load_validation_selection,
        parse_validation_selection,
        prepare_validation_shards,
        select_validation_shards,
        validate_selection_manifest,
        write_validation_selection,
    )

_EXPORTS = {
    "AcceptedProductionBatchStream": (
        "sakuramoon.data.production",
        "AcceptedProductionBatchStream",
    ),
    "VALIDATION_SELECTION_SEED": (
        "sakuramoon.data.validation",
        "VALIDATION_SELECTION_SEED",
    ),
    "VALIDATION_SHARD_COUNT": (
        "sakuramoon.data.validation",
        "VALIDATION_SHARD_COUNT",
    ),
    "VALIDATION_SHARD_PATHS": (
        "sakuramoon.data.validation",
        "VALIDATION_SHARD_PATHS",
    ),
    "DatasetAuthenticationError": (
        "sakuramoon.data.modelscope",
        "DatasetAuthenticationError",
    ),
    "DatasetManifest": ("sakuramoon.data.manifest", "DatasetManifest"),
    "DatasetManifestError": ("sakuramoon.data.manifest", "DatasetManifestError"),
    "DatasetSourceIdentity": ("sakuramoon.data.manifest", "DatasetSourceIdentity"),
    "DatasetTransportError": ("sakuramoon.data.modelscope", "DatasetTransportError"),
    "DuplicateIdReport": ("sakuramoon.data.metadata", "DuplicateIdReport"),
    "FetchedShard": ("sakuramoon.data.modelscope", "FetchedShard"),
    "MetadataError": ("sakuramoon.data.metadata", "MetadataError"),
    "MetadataRecord": ("sakuramoon.data.metadata", "MetadataRecord"),
    "OperationalMetadataRecord": (
        "sakuramoon.data.metadata",
        "OperationalMetadataRecord",
    ),
    "ConfiguredDataLoader": (
        "sakuramoon.data.production",
        "ConfiguredDataLoader",
    ),
    "PRODUCTION_METADATA_FIELDS": (
        "sakuramoon.data.production",
        "PRODUCTION_METADATA_FIELDS",
    ),
    "ModelScopeDatasetTransport": (
        "sakuramoon.data.modelscope",
        "ModelScopeDatasetTransport",
    ),
    "ProductionDataError": ("sakuramoon.data.production", "ProductionDataError"),
    "ProductionBatchStreamIdentity": (
        "sakuramoon.data.production",
        "ProductionBatchStreamIdentity",
    ),
    "ProductionPipelineFactory": (
        "sakuramoon.data.production",
        "ProductionPipelineFactory",
    ),
    "PreparedValidationShards": (
        "sakuramoon.data.validation",
        "PreparedValidationShards",
    ),
    "ShardIntegrityError": ("sakuramoon.data.modelscope", "ShardIntegrityError"),
    "ShardRecord": ("sakuramoon.data.manifest", "ShardRecord"),
    "ValidationPromptError": (
        "sakuramoon.data.validation",
        "ValidationPromptError",
    ),
    "ValidationPromptSample": (
        "sakuramoon.data.validation",
        "ValidationPromptSample",
    ),
    "ValidationSelection": ("sakuramoon.data.validation", "ValidationSelection"),
    "ValidationSelectionError": (
        "sakuramoon.data.validation",
        "ValidationSelectionError",
    ),
    "ValidationSelectionExistsError": (
        "sakuramoon.data.validation",
        "ValidationSelectionExistsError",
    ),
    "canonical_validation_selection_bytes": (
        "sakuramoon.data.validation",
        "canonical_validation_selection_bytes",
    ),
    "fetch_dataset_shard": ("sakuramoon.data.modelscope", "fetch_dataset_shard"),
    "ensure_dataset_manifest": (
        "sakuramoon.data.modelscope",
        "ensure_dataset_manifest",
    ),
    "adapt_modelscope_metadata": (
        "sakuramoon.data.production",
        "adapt_modelscope_metadata",
    ),
    "ensure_validation_selection": (
        "sakuramoon.data.validation",
        "ensure_validation_selection",
    ),
    "load_validation_prompt_samples": (
        "sakuramoon.data.validation",
        "load_validation_prompt_samples",
    ),
    "load_validation_selection": (
        "sakuramoon.data.validation",
        "load_validation_selection",
    ),
    "load_dataset_manifest": ("sakuramoon.data.manifest", "load_dataset_manifest"),
    "parse_metadata": ("sakuramoon.data.metadata", "parse_metadata"),
    "parse_modelscope_caption_fields": (
        "sakuramoon.data.production",
        "parse_modelscope_caption_fields",
    ),
    "require_accepted_production_batch_stream": (
        "sakuramoon.data.production",
        "require_accepted_production_batch_stream",
    ),
    "scan_duplicate_ids": ("sakuramoon.data.metadata", "scan_duplicate_ids"),
    "parse_validation_selection": (
        "sakuramoon.data.validation",
        "parse_validation_selection",
    ),
    "prepare_validation_shards": (
        "sakuramoon.data.validation",
        "prepare_validation_shards",
    ),
    "select_validation_shards": (
        "sakuramoon.data.validation",
        "select_validation_shards",
    ),
    "validate_remote_manifest": (
        "sakuramoon.data.modelscope",
        "validate_remote_manifest",
    ),
    "validate_selection_manifest": (
        "sakuramoon.data.validation",
        "validate_selection_manifest",
    ),
    "write_validation_selection": (
        "sakuramoon.data.validation",
        "write_validation_selection",
    ),
}

__all__ = [
    "PRODUCTION_METADATA_FIELDS",
    "VALIDATION_SELECTION_SEED",
    "VALIDATION_SHARD_COUNT",
    "VALIDATION_SHARD_PATHS",
    "AcceptedProductionBatchStream",
    "ConfiguredDataLoader",
    "DatasetAuthenticationError",
    "DatasetManifest",
    "DatasetManifestError",
    "DatasetSourceIdentity",
    "DatasetTransportError",
    "DuplicateIdReport",
    "FetchedShard",
    "MetadataError",
    "MetadataRecord",
    "ModelScopeDatasetTransport",
    "OperationalMetadataRecord",
    "PreparedValidationShards",
    "ProductionBatchStreamIdentity",
    "ProductionDataError",
    "ProductionPipelineFactory",
    "ShardIntegrityError",
    "ShardRecord",
    "ValidationPromptError",
    "ValidationPromptSample",
    "ValidationSelection",
    "ValidationSelectionError",
    "ValidationSelectionExistsError",
    "adapt_modelscope_metadata",
    "canonical_validation_selection_bytes",
    "ensure_dataset_manifest",
    "ensure_validation_selection",
    "fetch_dataset_shard",
    "load_dataset_manifest",
    "load_validation_prompt_samples",
    "load_validation_selection",
    "parse_metadata",
    "parse_modelscope_caption_fields",
    "parse_validation_selection",
    "prepare_validation_shards",
    "require_accepted_production_batch_stream",
    "scan_duplicate_ids",
    "select_validation_shards",
    "validate_remote_manifest",
    "validate_selection_manifest",
    "write_validation_selection",
]


def __getattr__(name: str) -> Any:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    module_name, attribute = target
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value
