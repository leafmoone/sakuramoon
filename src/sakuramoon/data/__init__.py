"""WebDataset manifest and ModelScope streaming helpers."""

from sakuramoon.data.manifest import (
    DatasetManifest,
    DatasetManifestError,
    DatasetSourceIdentity,
    ShardRecord,
    load_dataset_manifest,
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

__all__ = [
    "DatasetAuthenticationError",
    "DatasetManifest",
    "DatasetManifestError",
    "DatasetSourceIdentity",
    "DatasetTransportError",
    "FetchedShard",
    "ModelScopeDatasetTransport",
    "ShardIntegrityError",
    "ShardRecord",
    "fetch_dataset_shard",
    "load_dataset_manifest",
    "validate_remote_manifest",
]
