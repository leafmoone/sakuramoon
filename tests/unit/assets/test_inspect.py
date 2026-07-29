from __future__ import annotations

import gc
import hashlib
import json
import subprocess
import weakref
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any, Protocol

import pytest

from sakuramoon.assets import (
    AssetPreflightError,
    VerifiedAssetFile,
    VerifiedAssetSelection,
    inspect_databases,
    inspect_reference_repositories,
    inspect_runtime_models,
    load_manifest,
    require_databases_ready,
    require_runtime_assets_ready,
    require_verified_selection,
)
from sakuramoon.assets import inspect as inspect_module
from sakuramoon.assets.manifest import QwenAsset, VaeAsset
from sakuramoon.cli import inspect as cli_inspect_module
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


def test_hash_io_error_is_a_redacted_relative_issue(
    synthetic_assets: SyntheticAssetTree,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    weights = synthetic_assets.root / "model/qwen/model.safetensors"
    secret = "SENSITIVE-ABSOLUTE-PATH"
    real_sha256 = inspect_module._sha256  # pyright: ignore[reportPrivateUsage]

    def deny_weight_read(path: Path) -> str:
        if path == weights:
            raise PermissionError(secret)
        return real_sha256(path)

    monkeypatch.setattr(inspect_module, "_sha256", deny_weight_read)

    report = inspect_runtime_models(
        synthetic_assets.manifest_path,
        root=synthetic_assets.root,
    )
    matching = [issue for issue in report.issues if issue.code == "file_read_error"]
    assert [(issue.asset_id, issue.detail) for issue in matching] == [
        ("qwen_text_encoder", "model.safetensors")
    ]
    assert secret not in report.to_json()

    common = (
        "--manifest",
        str(synthetic_assets.manifest_path),
        "--root",
        str(synthetic_assets.root),
    )
    assert main(common) == 1
    captured = capsys.readouterr()
    assert json.loads(captured.out)["ok"] is False
    assert secret not in captured.out
    assert captured.err == ""


def test_config_read_io_is_not_misreported_as_invalid_json(
    synthetic_assets: SyntheticAssetTree,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = synthetic_assets.root / "model/qwen/config.json"
    real_read_text = Path.read_text

    def deny_config_read(path: Path, *args: object, **kwargs: object) -> str:
        if path == config_path:
            raise PermissionError("SENSITIVE-CONFIG-PATH")
        return real_read_text(path, *args, **kwargs)  # pyright: ignore[reportArgumentType]

    monkeypatch.setattr(Path, "read_text", deny_config_read)

    report = inspect_runtime_models(
        synthetic_assets.manifest_path,
        root=synthetic_assets.root,
    )
    qwen_codes = {
        issue.code for issue in report.issues if issue.asset_id == "qwen_text_encoder"
    }
    assert qwen_codes == {"file_read_error"}
    assert "SENSITIVE-CONFIG-PATH" not in report.to_json()


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


def test_untrusted_wrapper_argument_requires_a_live_selection_capability(
    synthetic_assets: SyntheticAssetTree,
) -> None:
    selection = require_runtime_assets_ready(
        _runtime_assets(synthetic_assets),
        synthetic_assets.manifest_path,
        root=synthetic_assets.root,
    )

    assert require_verified_selection(selection) is selection
    with pytest.raises(AssetPreflightError, match="capability is not verified"):
        require_verified_selection(object())

    weights = synthetic_assets.root / "model/qwen/model.safetensors"
    weights.write_bytes(b"x" * weights.stat().st_size)
    with pytest.raises(AssetPreflightError, match="identity drifted"):
        require_verified_selection(selection)


def test_selection_capability_rejects_subclasses_directly_and_through_a_wrapper() -> None:
    def forged_require_unchanged(self: object) -> None:
        del self
        raise AssertionError("a subclass override must never be invoked")

    def forged_verified_root(self: object, asset_id: str) -> Path:
        del self, asset_id
        return Path("/tmp/unverified-model-cache")

    forged_type = type(
        "ForgedSelection",
        (VerifiedAssetSelection,),
        {
            "require_unchanged": forged_require_unchanged,
            "verified_root": forged_verified_root,
        },
    )
    forged = object.__new__(forged_type)

    with pytest.raises(AssetPreflightError, match="capability is not verified"):
        require_verified_selection(forged)

    def narrow(value: object) -> VerifiedAssetSelection:
        return require_verified_selection(value)

    with pytest.raises(AssetPreflightError, match="capability is not verified"):
        narrow(forged)


def test_selection_capability_rejects_equal_direct_construction_and_object_new(
    synthetic_assets: SyntheticAssetTree,
) -> None:
    issued = require_runtime_assets_ready(
        _runtime_assets(synthetic_assets),
        synthetic_assets.manifest_path,
        root=synthetic_assets.root,
    )
    equal_direct_construction = replace(issued)
    uninitialized = object.__new__(VerifiedAssetSelection)

    assert type(equal_direct_construction) is VerifiedAssetSelection
    assert equal_direct_construction == issued
    assert equal_direct_construction is not issued
    with pytest.raises(AssetPreflightError, match="capability is not verified"):
        require_verified_selection(equal_direct_construction)
    with pytest.raises(AssetPreflightError, match="capability is not verified"):
        require_verified_selection(uninitialized)


def test_issued_selection_rejects_nested_file_subclass_without_dispatch(
    synthetic_assets: SyntheticAssetTree,
) -> None:
    selection = require_runtime_assets_ready(
        _runtime_assets(synthetic_assets),
        synthetic_assets.manifest_path,
        root=synthetic_assets.root,
    )

    def forged_require_unchanged(self: object) -> Path:
        del self
        raise AssertionError("a nested subclass override must never be invoked")

    forged_type = type(
        "ForgedAssetFile",
        (VerifiedAssetFile,),
        {"require_unchanged": forged_require_unchanged},
    )
    forged = object.__new__(forged_type)
    object.__setattr__(selection, "files", (forged, *selection.files[1:]))

    with pytest.raises(AssetPreflightError, match="file capability is not verified"):
        selection.require_unchanged()
    with pytest.raises(AssetPreflightError, match="file capability is not verified"):
        require_verified_selection(selection)


def test_issued_selection_rejects_equal_but_unissued_file(
    synthetic_assets: SyntheticAssetTree,
) -> None:
    selection = require_runtime_assets_ready(
        _runtime_assets(synthetic_assets),
        synthetic_assets.manifest_path,
        root=synthetic_assets.root,
    )
    issued_file = selection.files[0]
    equal_direct_construction = replace(issued_file)

    assert type(equal_direct_construction) is VerifiedAssetFile
    assert equal_direct_construction == issued_file
    assert equal_direct_construction is not issued_file
    object.__setattr__(
        selection,
        "files",
        (equal_direct_construction, *selection.files[1:]),
    )

    with pytest.raises(AssetPreflightError, match="file capability is not verified"):
        require_verified_selection(selection)
    with pytest.raises(AssetPreflightError, match="file capability is not verified"):
        equal_direct_construction.require_unchanged()


def test_issued_capability_registries_are_thread_safe_and_release_on_gc(
    synthetic_assets: SyntheticAssetTree,
) -> None:
    gc.collect()
    selection_count_before = len(
        inspect_module._ISSUED_SELECTIONS  # pyright: ignore[reportPrivateUsage]
    )
    file_count_before = len(
        inspect_module._ISSUED_FILES  # pyright: ignore[reportPrivateUsage]
    )
    selection = require_runtime_assets_ready(
        _runtime_assets(synthetic_assets),
        synthetic_assets.manifest_path,
        root=synthetic_assets.root,
    )
    selection_reference = weakref.ref(selection)
    file_references = tuple(weakref.ref(item) for item in selection.files)
    issued_file_count = len(selection.files)
    selections = (selection,) * 32

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = tuple(executor.map(require_verified_selection, selections))
    assert all(item is selection for item in results)
    assert len(inspect_module._ISSUED_SELECTIONS) == selection_count_before + 1  # pyright: ignore[reportPrivateUsage]
    assert len(inspect_module._ISSUED_FILES) == file_count_before + issued_file_count  # pyright: ignore[reportPrivateUsage]

    del results
    del selections
    del selection
    gc.collect()

    assert selection_reference() is None
    assert all(reference() is None for reference in file_references)
    assert len(inspect_module._ISSUED_SELECTIONS) == selection_count_before  # pyright: ignore[reportPrivateUsage]
    assert len(inspect_module._ISSUED_FILES) == file_count_before  # pyright: ignore[reportPrivateUsage]


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
    subprocess.run(
        ("git", "-C", str(reference), "config", "core.fsmonitor", str(hostile)),
        check=True,
    )
    subprocess.run(
        ("git", "-C", str(reference), "config", "core.hooksPath", str(hooks)),
        check=True,
    )
    subprocess.run(
        ("git", "-C", str(reference), "config", "core.pager", str(hostile)),
        check=True,
    )
    subprocess.run(
        ("git", "-C", str(reference), "config", "pager.status", str(hostile)),
        check=True,
    )
    subprocess.run(
        ("git", "-C", str(reference), "config", "diff.external", str(hostile)),
        check=True,
    )
    subprocess.run(
        (
            "git",
            "-C",
            str(reference),
            "config",
            "interactive.diffFilter",
            str(hostile),
        ),
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


@pytest.mark.parametrize("missing", ("manifest", "root"))
def test_cli_preflight_errors_are_json_without_traceback(
    synthetic_assets: SyntheticAssetTree,
    capsys: pytest.CaptureFixture[str],
    missing: str,
) -> None:
    manifest = synthetic_assets.manifest_path
    root = synthetic_assets.root
    if missing == "manifest":
        manifest = synthetic_assets.root / "assets/missing.toml"
    else:
        root = synthetic_assets.root / "missing-root"

    assert main(("--manifest", str(manifest), "--root", str(root))) == 2
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["ok"] is False
    assert "Traceback" not in captured.out
    assert captured.err == ""


def test_cli_last_resort_io_error_is_redacted(
    synthetic_assets: SyntheticAssetTree,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret = "SENSITIVE-RAW-IO-PATH"

    def raise_io(*args: object, **kwargs: object) -> None:
        raise PermissionError(secret)

    monkeypatch.setattr(cli_inspect_module, "inspect_runtime_models", raise_io)

    assert cli_inspect_module.main(("--root", str(synthetic_assets.root))) == 2
    captured = capsys.readouterr()
    assert json.loads(captured.out) == {
        "error": "asset inspection I/O failed",
        "ok": False,
    }
    assert secret not in captured.out
    assert captured.err == ""
