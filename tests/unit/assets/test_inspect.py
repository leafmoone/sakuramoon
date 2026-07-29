from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Any, Protocol

import pytest

from sakuramoon.assets import (
    AssetPreflightError,
    inspect_databases,
    inspect_reference_repositories,
    inspect_runtime_models,
    load_manifest,
    require_databases_ready,
    require_runtime_assets_ready,
)
from sakuramoon.assets import inspect as inspect_module
from sakuramoon.assets.manifest import QwenAsset, VaeAsset
from sakuramoon.cli.inspect import main
from sakuramoon.config.schema import AssetsConfig


class SyntheticAssetTree(Protocol):
    root: Path
    manifest_path: Path
    payload: dict[str, Any]

    def write_manifest(self) -> None: ...


def _runtime_assets(tree: SyntheticAssetTree) -> AssetsConfig:
    manifest = load_manifest(tree.manifest_path)
    qwen, vae = manifest.models
    assert isinstance(qwen, QwenAsset)
    assert isinstance(vae, VaeAsset)
    digest = hashlib.sha256(tree.manifest_path.read_bytes()).hexdigest()
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


def test_ready_synthetic_runtime_models_pass(synthetic_assets: SyntheticAssetTree) -> None:
    report = inspect_runtime_models(
        synthetic_assets.manifest_path,
        root=synthetic_assets.root,
    )

    assert report.ok
    assert report.issues == ()


def test_runtime_does_not_depend_on_optional_databases_or_references(
    synthetic_assets: SyntheticAssetTree,
) -> None:
    synthetic_assets.payload["databases"][0]["required_for_runtime"] = False
    synthetic_assets.write_manifest()
    (synthetic_assets.root / "db/metadata.db").rename(
        synthetic_assets.root / "db/metadata.db.absent"
    )
    for reference in synthetic_assets.payload["references"]:
        path = synthetic_assets.root / str(reference["local_path"])
        path.rename(path.with_name(f"{path.name}.absent"))

    readiness = require_runtime_assets_ready(
        _runtime_assets(synthetic_assets),
        synthetic_assets.manifest_path,
        root=synthetic_assets.root,
    )
    reference_report = inspect_reference_repositories(
        synthetic_assets.manifest_path,
        root=synthetic_assets.root,
    )

    assert readiness.files
    assert not reference_report.ok
    assert {issue.code for issue in reference_report.issues} == {"missing_reference"}


def test_size_drift_skips_payload_hash(
    synthetic_assets: SyntheticAssetTree,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    weights = synthetic_assets.root / "model/qwen/model.safetensors"
    weights.write_bytes(b"drift")
    real_sha256 = inspect_module._sha256  # pyright: ignore[reportPrivateUsage]

    def guarded_sha256(path: Path) -> str:
        if path == weights:
            raise AssertionError("size mismatch must skip the payload hash")
        return real_sha256(path)

    monkeypatch.setattr(inspect_module, "_sha256", guarded_sha256)

    report = inspect_runtime_models(
        synthetic_assets.manifest_path,
        root=synthetic_assets.root,
    )

    assert {issue.code for issue in report.issues} == {"byte_size_mismatch"}
    with pytest.raises(AssetPreflightError, match="runtime model preflight failed"):
        require_runtime_assets_ready(
            _runtime_assets(synthetic_assets),
            synthetic_assets.manifest_path,
            root=synthetic_assets.root,
        )


def test_required_model_missing_hard_fails(synthetic_assets: SyntheticAssetTree) -> None:
    weights = synthetic_assets.root / "model/qwen/model.safetensors"
    weights.unlink()

    report = inspect_runtime_models(
        synthetic_assets.manifest_path,
        root=synthetic_assets.root,
    )

    assert "missing_file" in {issue.code for issue in report.issues}
    with pytest.raises(AssetPreflightError, match="runtime model preflight failed"):
        require_runtime_assets_ready(
            _runtime_assets(synthetic_assets),
            synthetic_assets.manifest_path,
            root=synthetic_assets.root,
        )


def test_required_model_same_size_hash_drift_hard_fails(
    synthetic_assets: SyntheticAssetTree,
) -> None:
    weights = synthetic_assets.root / "model/qwen/model.safetensors"
    weights.write_bytes(b"x" * weights.stat().st_size)

    report = inspect_runtime_models(
        synthetic_assets.manifest_path,
        root=synthetic_assets.root,
    )

    assert "sha256_mismatch" in {issue.code for issue in report.issues}
    with pytest.raises(AssetPreflightError, match="runtime model preflight failed"):
        require_runtime_assets_ready(
            _runtime_assets(synthetic_assets),
            synthetic_assets.manifest_path,
            root=synthetic_assets.root,
        )


def test_selected_database_missing_fails_without_hash_or_open(
    synthetic_assets: SyntheticAssetTree,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = synthetic_assets.root / "db/metadata.db"
    database.unlink()
    real_sha256 = inspect_module._sha256  # pyright: ignore[reportPrivateUsage]

    def guarded_sha256(path: Path) -> str:
        if path.name == "metadata.db":
            raise AssertionError("missing database must fail before payload access")
        return real_sha256(path)

    monkeypatch.setattr(inspect_module, "_sha256", guarded_sha256)

    report = inspect_databases(
        synthetic_assets.manifest_path,
        root=synthetic_assets.root,
        asset_ids=("metadata_db",),
    )

    assert {issue.code for issue in report.issues} == {"missing_file"}
    with pytest.raises(AssetPreflightError, match="database asset audit failed"):
        require_databases_ready(
            synthetic_assets.manifest_path,
            root=synthetic_assets.root,
            asset_ids=("metadata_db",),
        )


def test_selected_database_size_mismatch_skips_hash(
    synthetic_assets: SyntheticAssetTree,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = synthetic_assets.root / "db/metadata.db"
    database.write_bytes(b"short")
    real_sha256 = inspect_module._sha256  # pyright: ignore[reportPrivateUsage]

    def guarded_sha256(path: Path) -> str:
        if path == database:
            raise AssertionError("database size mismatch must fail before hashing")
        return real_sha256(path)

    monkeypatch.setattr(inspect_module, "_sha256", guarded_sha256)

    report = inspect_databases(
        synthetic_assets.manifest_path,
        root=synthetic_assets.root,
        asset_ids=("metadata_db",),
    )

    assert {issue.code for issue in report.issues} == {"byte_size_mismatch"}


def test_selected_database_hash_drift_fails_before_database_use(
    synthetic_assets: SyntheticAssetTree,
) -> None:
    database = synthetic_assets.root / "db/metadata.db"
    database.write_bytes(b"x" * database.stat().st_size)

    report = inspect_databases(
        synthetic_assets.manifest_path,
        root=synthetic_assets.root,
        asset_ids=("metadata_db",),
    )

    assert {issue.code for issue in report.issues} == {"sha256_mismatch"}
    with pytest.raises(AssetPreflightError, match="database asset audit failed"):
        require_databases_ready(
            synthetic_assets.manifest_path,
            root=synthetic_assets.root,
            asset_ids=("metadata_db",),
        )


def test_selected_database_returns_revalidated_identity(
    synthetic_assets: SyntheticAssetTree,
) -> None:
    readiness = require_databases_ready(
        synthetic_assets.manifest_path,
        root=synthetic_assets.root,
        asset_ids=("metadata_db",),
    )
    database = synthetic_assets.root / "db/metadata.db"
    database.write_bytes(b"x" * database.stat().st_size)

    with pytest.raises(AssetPreflightError, match="identity drifted"):
        readiness.verified_path("metadata_db", "metadata.db")


def test_verified_model_root_is_bound_to_the_complete_selection(
    synthetic_assets: SyntheticAssetTree,
) -> None:
    selection = require_runtime_assets_ready(
        _runtime_assets(synthetic_assets),
        synthetic_assets.manifest_path,
        root=synthetic_assets.root,
    )

    assert selection.verified_root("qwen_text_encoder") == (
        synthetic_assets.root / "model/qwen"
    )
    assert selection.verified_root("mage_vae") == synthetic_assets.root / "model/vae"
    with pytest.raises(AssetPreflightError, match="asset root was not verified"):
        selection.verified_root("unknown_model")

    weights = synthetic_assets.root / "model/qwen/model.safetensors"
    weights.write_bytes(b"x" * weights.stat().st_size)
    with pytest.raises(AssetPreflightError, match="identity drifted"):
        selection.verified_root("qwen_text_encoder")


def test_database_audit_requires_explicit_known_unique_selection(
    synthetic_assets: SyntheticAssetTree,
) -> None:
    with pytest.raises(ValueError, match="at least one explicit"):
        inspect_databases(
            synthetic_assets.manifest_path,
            root=synthetic_assets.root,
            asset_ids=(),
        )
    with pytest.raises(ValueError, match="unknown database"):
        inspect_databases(
            synthetic_assets.manifest_path,
            root=synthetic_assets.root,
            asset_ids=("unknown_db",),
        )
    with pytest.raises(ValueError, match="must be unique"):
        inspect_databases(
            synthetic_assets.manifest_path,
            root=synthetic_assets.root,
            asset_ids=("metadata_db", "metadata_db"),
        )


def test_symlinked_model_file_is_rejected(synthetic_assets: SyntheticAssetTree) -> None:
    weights = synthetic_assets.root / "model/qwen/model.safetensors"
    weights.unlink()
    weights.symlink_to(synthetic_assets.root / "model/qwen/tokenizer.json")

    report = inspect_runtime_models(
        synthetic_assets.manifest_path,
        root=synthetic_assets.root,
    )

    assert any(issue.code == "unsafe_path" for issue in report.issues)


def test_manifest_parent_symlink_is_rejected(synthetic_assets: SyntheticAssetTree) -> None:
    link = synthetic_assets.root / "linked-assets"
    link.symlink_to(synthetic_assets.manifest_path.parent, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink path components"):
        inspect_runtime_models(Path("linked-assets/manifest.toml"), root=synthetic_assets.root)


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

    report = inspect_runtime_models(
        synthetic_assets.manifest_path,
        root=synthetic_assets.root,
    )

    assert any(issue.code == "asset_blocked" for issue in report.issues)


def test_qwen_and_vae_config_mismatch_are_reported(
    synthetic_assets: SyntheticAssetTree,
) -> None:
    qwen = synthetic_assets.root / "model/qwen/config.json"
    qwen.write_text(
        '{"text_config":{"num_hidden_layers":23,"hidden_size":2048}}',
        encoding="utf-8",
    )
    vae = synthetic_assets.root / "model/vae/config.json"
    vae.write_text(
        '{"latent_channels":64,"downsample_factor":16,"sample_posterior":true}',
        encoding="utf-8",
    )

    report = inspect_runtime_models(
        synthetic_assets.manifest_path,
        root=synthetic_assets.root,
    )

    codes = {issue.code for issue in report.issues}
    assert "qwen_architecture_mismatch" in codes
    assert "vae_interface_mismatch" in codes


def test_reference_origin_diagnostic_redacts_credentials(
    synthetic_assets: SyntheticAssetTree,
) -> None:
    secret = "TOP-SECRET-TOKEN"
    reference = synthetic_assets.root / "reference/HDM"
    subprocess.run(
        (
            "git",
            "-C",
            str(reference),
            "remote",
            "set-url",
            "origin",
            f"https://{secret}@example.invalid/HDM.git",
        ),
        check=True,
    )

    report = inspect_reference_repositories(
        synthetic_assets.manifest_path,
        root=synthetic_assets.root,
    )
    serialized = report.to_json()

    assert not report.ok
    assert "origin_mismatch" in serialized
    assert secret not in serialized
    assert secret not in str(report.issues)


def test_reference_git_audit_disables_hostile_local_configuration(
    synthetic_assets: SyntheticAssetTree,
) -> None:
    reference = synthetic_assets.root / "reference/HDM"
    marker = synthetic_assets.root / "hostile-git-config-executed"
    hostile = synthetic_assets.root / "hostile-git-command.sh"
    hostile.write_text(
        f"#!/bin/sh\nprintf executed >> {marker}\nexit 0\n",
        encoding="utf-8",
    )
    hostile.chmod(0o700)
    hooks = synthetic_assets.root / "hostile-hooks"
    hooks.mkdir()
    hook = hooks / "post-index-change"
    hook.write_text(
        f"#!/bin/sh\nprintf hook >> {marker}\nexit 0\n",
        encoding="utf-8",
    )
    hook.chmod(0o700)
    for key, value in (
        ("core.fsmonitor", str(hostile)),
        ("core.hooksPath", str(hooks)),
        ("core.pager", str(hostile)),
        ("pager.status", str(hostile)),
        ("diff.external", str(hostile)),
        ("interactive.diffFilter", str(hostile)),
    ):
        subprocess.run(
            ("git", "-C", str(reference), "config", key, value),
            check=True,
        )

    report = inspect_reference_repositories(
        synthetic_assets.manifest_path,
        root=synthetic_assets.root,
    )

    assert report.ok
    assert not marker.exists()


def test_reference_git_helper_rejects_unlisted_commands_without_execution(
    synthetic_assets: SyntheticAssetTree,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference = load_manifest(synthetic_assets.manifest_path).references[0]

    def forbidden_run(*args: object, **kwargs: object) -> None:
        raise AssertionError(f"unexpected subprocess: {args!r} {kwargs!r}")

    monkeypatch.setattr(inspect_module.subprocess, "run", forbidden_run)

    assert (
        inspect_module._git(  # pyright: ignore[reportPrivateUsage]
            reference,
            synthetic_assets.root,
            "show",
            "HEAD:train.py",
        )
        is None
    )


def test_cli_scopes_and_exit_codes(
    synthetic_assets: SyntheticAssetTree,
    capsys: pytest.CaptureFixture[str],
) -> None:
    common = ("--manifest", str(synthetic_assets.manifest_path), "--root", str(synthetic_assets.root))
    assert main(common) == 0
    assert '"ok":true' in capsys.readouterr().out

    assert main((*common, "--scope", "databases", "--asset-id", "metadata_db")) == 0
    assert '"ok":true' in capsys.readouterr().out

    synthetic_assets.payload["models"][0]["lock_state"] = "blocked"
    synthetic_assets.payload["models"][0]["blockers"] = ["manual_blocker"]
    synthetic_assets.write_manifest()
    assert main(common) == 1
    assert '"ok":false' in capsys.readouterr().out
