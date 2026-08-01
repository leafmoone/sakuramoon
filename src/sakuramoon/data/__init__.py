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
        parse_metadata,
        scan_duplicate_ids,
    )
    from sakuramoon.data.modelscope import (
        DatasetAuthenticationError,
        DatasetTransportError,
        FetchedShard,
        ModelScopeDatasetTransport,
        ShardIntegrityError,
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
        VALIDATION_SAMPLE_COUNT,
        TrainingExclusionReport,
        ValidationEntry,
        ValidationManifestError,
        ValidationSelection,
        ValidationSelectionError,
        ValidationStratum,
        exclude_validation_records,
        load_validation_manifest_ids,
        select_validation_records,
        validation_manifest_bytes,
    )

_EXPORTS = {
    "AcceptedProductionBatchStream": (
        "sakuramoon.data.production",
        "AcceptedProductionBatchStream",
    ),
    "VALIDATION_SAMPLE_COUNT": (
        "sakuramoon.data.validation",
        "VALIDATION_SAMPLE_COUNT",
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
    "ShardIntegrityError": ("sakuramoon.data.modelscope", "ShardIntegrityError"),
    "ShardRecord": ("sakuramoon.data.manifest", "ShardRecord"),
    "TrainingExclusionReport": (
        "sakuramoon.data.validation",
        "TrainingExclusionReport",
    ),
    "ValidationEntry": ("sakuramoon.data.validation", "ValidationEntry"),
    "ValidationSelection": ("sakuramoon.data.validation", "ValidationSelection"),
    "ValidationSelectionError": (
        "sakuramoon.data.validation",
        "ValidationSelectionError",
    ),
    "ValidationManifestError": (
        "sakuramoon.data.validation",
        "ValidationManifestError",
    ),
    "ValidationStratum": ("sakuramoon.data.validation", "ValidationStratum"),
    "exclude_validation_records": (
        "sakuramoon.data.validation",
        "exclude_validation_records",
    ),
    "fetch_dataset_shard": ("sakuramoon.data.modelscope", "fetch_dataset_shard"),
    "adapt_modelscope_metadata": (
        "sakuramoon.data.production",
        "adapt_modelscope_metadata",
    ),
    "load_validation_manifest_ids": (
        "sakuramoon.data.validation",
        "load_validation_manifest_ids",
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
    "select_validation_records": (
        "sakuramoon.data.validation",
        "select_validation_records",
    ),
    "validate_remote_manifest": (
        "sakuramoon.data.modelscope",
        "validate_remote_manifest",
    ),
    "validation_manifest_bytes": (
        "sakuramoon.data.validation",
        "validation_manifest_bytes",
    ),
}

__all__ = [
    "PRODUCTION_METADATA_FIELDS",
    "VALIDATION_SAMPLE_COUNT",
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
    "ProductionBatchStreamIdentity",
    "ProductionDataError",
    "ProductionPipelineFactory",
    "ShardIntegrityError",
    "ShardRecord",
    "TrainingExclusionReport",
    "ValidationEntry",
    "ValidationManifestError",
    "ValidationSelection",
    "ValidationSelectionError",
    "ValidationStratum",
    "adapt_modelscope_metadata",
    "exclude_validation_records",
    "fetch_dataset_shard",
    "load_dataset_manifest",
    "load_validation_manifest_ids",
    "parse_metadata",
    "parse_modelscope_caption_fields",
    "require_accepted_production_batch_stream",
    "scan_duplicate_ids",
    "select_validation_records",
    "validate_remote_manifest",
    "validation_manifest_bytes",
]


def __getattr__(name: str) -> Any:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    module_name, attribute = target
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value
