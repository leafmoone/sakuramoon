from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from sakuramoon.assets import (
    AssetBindingError,
    load_manifest,
    require_runtime_assets_match,
)
from sakuramoon.assets.manifest import QwenAsset, VaeAsset
from sakuramoon.config.schema import AssetsConfig


def _runtime_assets(manifest_path: Path) -> AssetsConfig:
    manifest = load_manifest(manifest_path)
    qwen, vae = manifest.models
    assert isinstance(qwen, QwenAsset)
    assert isinstance(vae, VaeAsset)
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    return AssetsConfig.model_validate(
        {
            "qwen": {
                "repo_id": qwen.source.repo_id,
                "revision": qwen.source.revision,
                "local_path": qwen.local_path,
                "manifest_sha256": digest,
                "tokenizer_sha256": qwen.summary.tokenizer_sha256,
                "dtype": qwen.summary.dtype,
                "frozen": qwen.summary.frozen,
                "layers": qwen.summary.layers,
                "hidden_size": qwen.summary.hidden_size,
                "use_cache": qwen.summary.use_cache,
                "visual_path_enabled": qwen.summary.visual_path_enabled,
            },
            "vae": {
                "repo_id": vae.source.repo_id,
                "revision": vae.source.revision,
                "local_path": vae.local_path,
                "manifest_sha256": digest,
                "dtype": vae.summary.dtype,
                "frozen": vae.summary.frozen,
                "latent_channels": vae.summary.latent_channels,
                "downsample_factor": vae.summary.downsample_factor,
                "sample_posterior": vae.summary.sample_posterior,
            },
        },
        strict=True,
    )


def test_runtime_assets_match_locked_production_models() -> None:
    manifest_path = Path("assets/manifest.toml")

    require_runtime_assets_match(_runtime_assets(manifest_path), manifest_path)


@pytest.mark.parametrize(
    ("table", "field", "value"),
    [
        ("qwen", "revision", "c" * 40),
        ("qwen", "manifest_sha256", "0" * 64),
        ("qwen", "tokenizer_sha256", "1" * 64),
        ("vae", "revision", "d" * 40),
        ("vae", "manifest_sha256", "2" * 64),
    ],
)
def test_runtime_asset_mismatch_is_rejected(table: str, field: str, value: str) -> None:
    manifest_path = Path("assets/manifest.toml")
    config = _runtime_assets(manifest_path)
    payload = config.model_dump(mode="python")
    payload[table][field] = value
    mismatched = AssetsConfig.model_validate(payload, strict=True)

    with pytest.raises(AssetBindingError, match=f"{table}.{field}"):
        require_runtime_assets_match(mismatched, manifest_path)
