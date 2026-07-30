"""WebDataset manifest and ModelScope streaming helpers."""

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
from sakuramoon.data.validation import (
    VALIDATION_SAMPLE_COUNT,
    TrainingExclusionReport,
    ValidationEntry,
    ValidationSelection,
    ValidationSelectionError,
    ValidationStratum,
    exclude_validation_records,
    select_validation_records,
    validation_manifest_bytes,
)

__all__ = [
    "VALIDATION_SAMPLE_COUNT",
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
    "ShardIntegrityError",
    "ShardRecord",
    "TrainingExclusionReport",
    "ValidationEntry",
    "ValidationSelection",
    "ValidationSelectionError",
    "ValidationStratum",
    "exclude_validation_records",
    "fetch_dataset_shard",
    "load_dataset_manifest",
    "parse_metadata",
    "scan_duplicate_ids",
    "select_validation_records",
    "validate_remote_manifest",
    "validation_manifest_bytes",
]
