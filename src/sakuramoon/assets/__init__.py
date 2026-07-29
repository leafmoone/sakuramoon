"""Immutable asset manifest and preflight interfaces."""

from sakuramoon.assets.bindings import AssetBindingError, require_runtime_assets_match
from sakuramoon.assets.inspect import (
    AssetPreflightError,
    InspectionIssue,
    InspectionReport,
    inspect_assets,
    require_assets_ready,
)
from sakuramoon.assets.manifest import AssetManifest, ManifestError, load_manifest

__all__ = [
    "AssetBindingError",
    "AssetManifest",
    "AssetPreflightError",
    "InspectionIssue",
    "InspectionReport",
    "ManifestError",
    "inspect_assets",
    "load_manifest",
    "require_assets_ready",
    "require_runtime_assets_match",
]
