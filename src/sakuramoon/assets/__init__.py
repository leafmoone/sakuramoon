"""Immutable asset manifest and preflight interfaces."""

from sakuramoon.assets.bindings import AssetBindingError
from sakuramoon.assets.inspect import (
    AssetPreflightError,
    InspectionIssue,
    InspectionReport,
    VerifiedAssetFile,
    VerifiedAssetSelection,
    inspect_databases,
    inspect_reference_repositories,
    inspect_runtime_models,
    require_databases_ready,
    require_runtime_assets_ready,
    require_verified_selection,
)
from sakuramoon.assets.manifest import AssetManifest, ManifestError, load_manifest

__all__ = [
    "AssetBindingError",
    "AssetManifest",
    "AssetPreflightError",
    "InspectionIssue",
    "InspectionReport",
    "ManifestError",
    "VerifiedAssetFile",
    "VerifiedAssetSelection",
    "inspect_databases",
    "inspect_reference_repositories",
    "inspect_runtime_models",
    "load_manifest",
    "require_databases_ready",
    "require_runtime_assets_ready",
    "require_verified_selection",
]
