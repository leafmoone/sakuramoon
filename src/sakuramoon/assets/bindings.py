"""Exact binding between runtime asset configuration and the locked manifest."""

from __future__ import annotations

import hashlib
from pathlib import Path

from sakuramoon.assets.manifest import QwenAsset, VaeAsset, load_manifest
from sakuramoon.config.schema import AssetsConfig


class AssetBindingError(ValueError):
    """Raised when runtime configuration does not select the locked assets."""


def _manifest_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_runtime_assets_match(config: AssetsConfig, manifest_path: Path) -> None:
    """Reject runtime asset fields that differ from the immutable manifest."""

    manifest = load_manifest(manifest_path)
    models = {asset.kind: asset for asset in manifest.models}
    qwen = models["qwen"]
    vae = models["vae"]
    if not isinstance(qwen, QwenAsset) or not isinstance(vae, VaeAsset):
        raise AssetBindingError("asset manifest model kinds are invalid")
    if qwen.lock_state != "ready" or vae.lock_state != "ready":
        raise AssetBindingError("runtime models are not fully locked")

    manifest_sha256 = _manifest_sha256(manifest_path)
    expected_qwen = {
        "repo_id": qwen.source.repo_id,
        "revision": qwen.source.revision,
        "local_path": qwen.local_path,
        "manifest_sha256": manifest_sha256,
        "tokenizer_sha256": qwen.summary.tokenizer_sha256,
        "dtype": qwen.summary.dtype,
        "frozen": qwen.summary.frozen,
        "layers": qwen.summary.layers,
        "hidden_size": qwen.summary.hidden_size,
        "use_cache": qwen.summary.use_cache,
        "visual_path_enabled": qwen.summary.visual_path_enabled,
    }
    expected_vae = {
        "repo_id": vae.source.repo_id,
        "revision": vae.source.revision,
        "local_path": vae.local_path,
        "manifest_sha256": manifest_sha256,
        "dtype": vae.summary.dtype,
        "frozen": vae.summary.frozen,
        "latent_channels": vae.summary.latent_channels,
        "downsample_factor": vae.summary.downsample_factor,
        "sample_posterior": vae.summary.sample_posterior,
    }
    mismatches = [
        *(f"qwen.{name}" for name, value in expected_qwen.items() if getattr(config.qwen, name) != value),
        *(f"vae.{name}" for name, value in expected_vae.items() if getattr(config.vae, name) != value),
    ]
    if mismatches:
        raise AssetBindingError("runtime asset mismatch: " + ",".join(sorted(mismatches)))
