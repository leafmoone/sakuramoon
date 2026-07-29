from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import pytest

from sakuramoon.assets import AssetPreflightError, inspect_assets, require_assets_ready
from sakuramoon.assets import inspect as inspect_module
from sakuramoon.cli.inspect import main


class SyntheticAssetTree(Protocol):
    root: Path
    manifest_path: Path
    payload: dict[str, Any]

    def write_manifest(self) -> None: ...


def test_ready_synthetic_tree_passes(synthetic_assets: SyntheticAssetTree) -> None:
    report = require_assets_ready(synthetic_assets.manifest_path, root=synthetic_assets.root)

    assert report.ok
    assert report.issues == ()


def test_hash_drift_fails_before_use(synthetic_assets: SyntheticAssetTree) -> None:
    (synthetic_assets.root / "model/qwen/model.safetensors").write_bytes(b"drift")

    report = inspect_assets(synthetic_assets.manifest_path, root=synthetic_assets.root)

    assert not report.ok
    assert {issue.code for issue in report.issues} == {"byte_size_mismatch", "sha256_mismatch"}
    with pytest.raises(AssetPreflightError, match="asset preflight failed"):
        require_assets_ready(synthetic_assets.manifest_path, root=synthetic_assets.root)


def test_missing_file_fails(synthetic_assets: SyntheticAssetTree) -> None:
    (synthetic_assets.root / "db/metadata.db").unlink()

    report = inspect_assets(synthetic_assets.manifest_path, root=synthetic_assets.root)

    assert any(issue.code == "missing_file" for issue in report.issues)


def test_symlink_is_rejected(synthetic_assets: SyntheticAssetTree) -> None:
    weights = synthetic_assets.root / "model/qwen/model.safetensors"
    weights.unlink()
    weights.symlink_to(synthetic_assets.root / "model/qwen/tokenizer.json")

    report = inspect_assets(synthetic_assets.manifest_path, root=synthetic_assets.root)

    assert any(issue.code == "unsafe_path" for issue in report.issues)


def test_manifest_parent_symlink_is_rejected(synthetic_assets: SyntheticAssetTree) -> None:
    link = synthetic_assets.root / "linked-assets"
    link.symlink_to(synthetic_assets.manifest_path.parent, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink path components"):
        inspect_assets(Path("linked-assets/manifest.toml"), root=synthetic_assets.root)


def test_blocked_asset_does_not_hash_undeclared_weight(
    synthetic_assets: SyntheticAssetTree,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    qwen = synthetic_assets.payload["models"][0]
    qwen["lock_state"] = "blocked"
    qwen["blockers"] = ["missing_weight_sha256"]
    qwen["files"][2].pop("sha256")
    synthetic_assets.write_manifest()
    real_sha256 = inspect_module._sha256  # pyright: ignore[reportPrivateUsage]

    def guarded_sha256(path: Path) -> str:
        if path.name == "model.safetensors":
            raise AssertionError("blocked weight must not be read")
        return real_sha256(path)

    monkeypatch.setattr(inspect_module, "_sha256", guarded_sha256)

    report = inspect_assets(synthetic_assets.manifest_path, root=synthetic_assets.root)

    assert any(issue.code == "asset_blocked" for issue in report.issues)


def test_qwen_and_vae_config_mismatch_are_reported(synthetic_assets: SyntheticAssetTree) -> None:
    qwen = synthetic_assets.root / "model/qwen/config.json"
    qwen.write_text('{"text_config":{"num_hidden_layers":23,"hidden_size":2048}}', encoding="utf-8")
    vae = synthetic_assets.root / "model/vae/config.json"
    vae.write_text('{"latent_channels":64,"downsample_factor":16,"sample_posterior":true}', encoding="utf-8")

    report = inspect_assets(synthetic_assets.manifest_path, root=synthetic_assets.root)

    codes = {issue.code for issue in report.issues}
    assert "qwen_architecture_mismatch" in codes
    assert "vae_interface_mismatch" in codes


def test_cli_exit_codes_and_json(synthetic_assets: SyntheticAssetTree, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(("--manifest", str(synthetic_assets.manifest_path), "--root", str(synthetic_assets.root))) == 0
    assert '"ok":true' in capsys.readouterr().out

    synthetic_assets.payload["models"][0]["lock_state"] = "blocked"
    synthetic_assets.payload["models"][0]["blockers"] = ["manual_blocker"]
    synthetic_assets.write_manifest()
    assert main(("--manifest", str(synthetic_assets.manifest_path), "--root", str(synthetic_assets.root))) == 1
    assert '"ok":false' in capsys.readouterr().out
