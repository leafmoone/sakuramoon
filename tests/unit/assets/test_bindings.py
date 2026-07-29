from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Protocol

import pytest

from sakuramoon.assets import (
    AssetBindingError,
    AssetPreflightError,
    load_manifest,
    require_runtime_assets_ready,
)
from sakuramoon.assets import inspect as inspect_module
from sakuramoon.assets.manifest import QwenAsset, VaeAsset
from sakuramoon.config.schema import AssetsConfig


class SyntheticAssetTree(Protocol):
    root: Path
    manifest_path: Path
    payload: dict[str, Any]

    def write_manifest(self) -> None: ...


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


def test_runtime_readiness_binds_and_verifies_one_manifest_snapshot(
    synthetic_assets: SyntheticAssetTree,
) -> None:
    readiness = require_runtime_assets_ready(
        _runtime_assets(synthetic_assets.manifest_path),
        synthetic_assets.manifest_path,
        root=synthetic_assets.root,
    )

    assert readiness.manifest_sha256 == hashlib.sha256(
        synthetic_assets.manifest_path.read_bytes()
    ).hexdigest()
    assert readiness.verified_path("qwen_text_encoder", "model.safetensors").is_file()


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
def test_runtime_asset_mismatch_is_rejected(
    synthetic_assets: SyntheticAssetTree,
    table: str,
    field: str,
    value: str,
) -> None:
    config = _runtime_assets(synthetic_assets.manifest_path)
    payload = config.model_dump(mode="python")
    payload[table][field] = value
    mismatched = AssetsConfig.model_validate(payload, strict=True)

    with pytest.raises(AssetBindingError, match=f"{table}.{field}"):
        require_runtime_assets_ready(
            mismatched,
            synthetic_assets.manifest_path,
            root=synthetic_assets.root,
        )


def test_runtime_manifest_must_be_root_confined(
    synthetic_assets: SyntheticAssetTree,
) -> None:
    outside = (
        synthetic_assets.root.parent / f"{synthetic_assets.root.name}-external-manifest.toml"
    )
    outside.write_bytes(synthetic_assets.manifest_path.read_bytes())

    with pytest.raises(ValueError, match="inside the repository root"):
        require_runtime_assets_ready(
            _runtime_assets(synthetic_assets.manifest_path),
            outside,
            root=synthetic_assets.root,
        )


def test_runtime_manifest_final_symlink_is_rejected(
    synthetic_assets: SyntheticAssetTree,
) -> None:
    link = synthetic_assets.root / "manifest-link.toml"
    link.symlink_to(synthetic_assets.manifest_path)

    with pytest.raises(ValueError, match="symlink path components"):
        require_runtime_assets_ready(
            _runtime_assets(synthetic_assets.manifest_path),
            link,
            root=synthetic_assets.root,
        )


def test_atomic_manifest_replacement_during_readiness_fails_closed(
    synthetic_assets: SyntheticAssetTree,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _runtime_assets(synthetic_assets.manifest_path)
    real_sha256 = inspect_module._sha256  # pyright: ignore[reportPrivateUsage]
    replaced = False

    def replace_manifest_once(path: Path) -> str:
        nonlocal replaced
        if not replaced:
            replaced = True
            replacement = synthetic_assets.manifest_path.with_suffix(".replacement")
            replacement.write_bytes(synthetic_assets.manifest_path.read_bytes())
            replacement.replace(synthetic_assets.manifest_path)
        return real_sha256(path)

    monkeypatch.setattr(inspect_module, "_sha256", replace_manifest_once)

    with pytest.raises(AssetPreflightError, match="manifest identity drifted"):
        require_runtime_assets_ready(
            config,
            synthetic_assets.manifest_path,
            root=synthetic_assets.root,
        )


def test_same_inode_same_size_manifest_content_drift_fails_with_stale_stat(
    synthetic_assets: SyntheticAssetTree,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    readiness = require_runtime_assets_ready(
        _runtime_assets(synthetic_assets.manifest_path),
        synthetic_assets.manifest_path,
        root=synthetic_assets.root,
    )
    real_stat_identity = inspect_module._stat_identity  # pyright: ignore[reportPrivateUsage]
    locked_identity = real_stat_identity(synthetic_assets.manifest_path)
    original = synthetic_assets.manifest_path.read_bytes()
    changed = original.replace(b"manifest_revision = 1", b"manifest_revision = 2", 1)
    assert changed != original
    assert len(changed) == len(original)
    inode_before = synthetic_assets.manifest_path.stat().st_ino
    with synthetic_assets.manifest_path.open("r+b") as handle:
        handle.write(changed)
        handle.truncate()
    assert synthetic_assets.manifest_path.stat().st_ino == inode_before

    def stale_manifest_identity(path: Path) -> object:
        if path == synthetic_assets.manifest_path:
            return locked_identity
        return real_stat_identity(path)

    monkeypatch.setattr(inspect_module, "_stat_identity", stale_manifest_identity)

    with pytest.raises(AssetPreflightError, match="manifest identity drifted"):
        readiness.require_unchanged()


def test_selection_rehashes_only_the_small_manifest_before_model_use(
    synthetic_assets: SyntheticAssetTree,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    readiness = require_runtime_assets_ready(
        _runtime_assets(synthetic_assets.manifest_path),
        synthetic_assets.manifest_path,
        root=synthetic_assets.root,
    )
    real_sha256 = inspect_module._sha256  # pyright: ignore[reportPrivateUsage]
    hashed_paths: list[Path] = []

    def record_sha256(path: Path) -> str:
        hashed_paths.append(path)
        return real_sha256(path)

    monkeypatch.setattr(inspect_module, "_sha256", record_sha256)

    readiness.verified_path("qwen_text_encoder", "model.safetensors")

    assert hashed_paths == [synthetic_assets.manifest_path]


def test_manifest_revalidation_io_error_is_redacted(
    synthetic_assets: SyntheticAssetTree,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    readiness = require_runtime_assets_ready(
        _runtime_assets(synthetic_assets.manifest_path),
        synthetic_assets.manifest_path,
        root=synthetic_assets.root,
    )
    secret = "SENSITIVE-MANIFEST-PATH"

    def deny_manifest_read(path: Path) -> str:
        if path == synthetic_assets.manifest_path:
            raise PermissionError(secret)
        raise AssertionError(f"model payload was unexpectedly rehashed: {path.name}")

    monkeypatch.setattr(inspect_module, "_sha256", deny_manifest_read)

    with pytest.raises(
        AssetPreflightError,
        match="manifest could not be revalidated",
    ) as raised:
        readiness.require_unchanged()
    assert secret not in str(raised.value)


def test_verified_file_identity_must_be_rechecked_before_use(
    synthetic_assets: SyntheticAssetTree,
) -> None:
    readiness = require_runtime_assets_ready(
        _runtime_assets(synthetic_assets.manifest_path),
        synthetic_assets.manifest_path,
        root=synthetic_assets.root,
    )
    weights = synthetic_assets.root / "model/qwen/model.safetensors"
    weights.write_bytes(b"x" * weights.stat().st_size)

    with pytest.raises(AssetPreflightError, match="identity drifted"):
        readiness.verified_path("qwen_text_encoder", "model.safetensors")
