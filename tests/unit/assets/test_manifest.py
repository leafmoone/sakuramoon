from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import pytest
from pydantic import ValidationError

from sakuramoon.assets.manifest import (
    AssetManifest,
    ManifestError,
    QwenAsset,
    VaeAsset,
    load_manifest,
)


class SyntheticAssetTree(Protocol):
    root: Path
    manifest_path: Path
    payload: dict[str, Any]

    def write_manifest(self) -> None: ...


def test_production_manifest_records_confirmed_models_and_optional_databases() -> None:
    manifest = load_manifest(Path("assets/manifest.toml"))

    assert [asset.kind for asset in manifest.models] == ["qwen", "vae"]
    qwen, vae = manifest.models
    assert isinstance(qwen, QwenAsset)
    assert isinstance(vae, VaeAsset)
    assert qwen.lock_state == "ready"
    assert qwen.blockers == ()
    assert qwen.source.revision == "7a387e3c9c23e913bbbe0b20fcbd1c5b38c0b96f"
    assert qwen.summary.layers == 24
    assert qwen.summary.hidden_size == 2048
    assert vae.lock_state == "ready"
    assert vae.blockers == ()
    assert vae.source.repo_id == "microsoft/Mage-Flow"
    assert vae.summary.latent_channels == 128
    assert vae.summary.downsample_factor == 16
    assert vae.summary.sample_posterior is False
    assert vae.summary.posterior_mean_required is True
    assert len(manifest.databases) == 3
    assert all(asset.lock_state == "ready" for asset in manifest.databases)
    assert all(not asset.required_for_runtime for asset in manifest.databases)
    assert manifest.databases[0].source.origin_kind == "user_derived"


def test_unknown_key_is_rejected(synthetic_assets: SyntheticAssetTree) -> None:
    synthetic_assets.payload["unexpected"] = True
    synthetic_assets.write_manifest()

    with pytest.raises(ManifestError, match="unexpected"):
        load_manifest(synthetic_assets.manifest_path)


@pytest.mark.parametrize("invalid", (True, 0, 1, "false"))
def test_database_runtime_flag_rejects_true_and_wrong_toml_types(
    synthetic_assets: SyntheticAssetTree,
    invalid: object,
) -> None:
    synthetic_assets.payload["databases"][0]["required_for_runtime"] = invalid
    synthetic_assets.write_manifest()

    with pytest.raises(ManifestError, match="databases.0.required_for_runtime"):
        load_manifest(synthetic_assets.manifest_path)


def test_database_runtime_flag_rejects_null_during_strict_model_validation(
    synthetic_assets: SyntheticAssetTree,
) -> None:
    synthetic_assets.payload["databases"][0]["required_for_runtime"] = None

    with pytest.raises(ValidationError, match="databases.0.required_for_runtime"):
        AssetManifest.model_validate(synthetic_assets.payload, strict=True)


def test_path_traversal_is_rejected(synthetic_assets: SyntheticAssetTree) -> None:
    synthetic_assets.payload["models"][0]["local_path"] = "../model/qwen"
    synthetic_assets.write_manifest()

    with pytest.raises(ManifestError, match="models.0"):
        load_manifest(synthetic_assets.manifest_path)


def test_ready_third_party_vae_is_rejected(synthetic_assets: SyntheticAssetTree) -> None:
    synthetic_assets.payload["models"][1]["source"]["repo_id"] = "third-party/converted-vae"
    synthetic_assets.write_manifest()

    with pytest.raises(ManifestError, match="models.1.vae"):
        load_manifest(synthetic_assets.manifest_path)


def test_ready_asset_requires_every_sha(synthetic_assets: SyntheticAssetTree) -> None:
    synthetic_assets.payload["models"][0]["files"][2].pop("sha256")
    synthetic_assets.write_manifest()

    with pytest.raises(ManifestError, match="models.0.qwen"):
        load_manifest(synthetic_assets.manifest_path)


def test_ready_asset_cannot_omit_required_weight_kind(
    synthetic_assets: SyntheticAssetTree,
) -> None:
    synthetic_assets.payload["models"][0]["files"] = synthetic_assets.payload["models"][0][
        "files"
    ][:2]
    synthetic_assets.write_manifest()

    with pytest.raises(ManifestError, match="models.0.qwen"):
        load_manifest(synthetic_assets.manifest_path)


def test_summary_hash_must_match_file_lock(synthetic_assets: SyntheticAssetTree) -> None:
    synthetic_assets.payload["models"][1]["summary"]["config_sha256"] = "f" * 64
    synthetic_assets.write_manifest()

    with pytest.raises(ManifestError, match="models.1.vae"):
        load_manifest(synthetic_assets.manifest_path)
