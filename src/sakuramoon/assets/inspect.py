"""Root-confined runtime readiness and explicit asset audit boundaries."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

from sakuramoon.assets.bindings import require_runtime_assets_match_snapshot
from sakuramoon.assets.manifest import (
    AssetManifest,
    DatabaseAsset,
    FileLock,
    ModelAsset,
    QwenAsset,
    ReferenceAsset,
    VaeAsset,
    parse_manifest_bytes,
)
from sakuramoon.config.schema import AssetsConfig


class AssetPreflightError(RuntimeError):
    """Raised before asset use when a lock is incomplete or has drifted."""


@dataclass(frozen=True)
class InspectionIssue:
    asset_id: str
    code: str
    detail: str


@dataclass(frozen=True)
class InspectionReport:
    manifest_revision: int
    ok: bool
    issues: tuple[InspectionIssue, ...]

    def to_json(self) -> str:
        return json.dumps(
            {
                "manifest_revision": self.manifest_revision,
                "ok": self.ok,
                "issues": [asdict(issue) for issue in self.issues],
            },
            sort_keys=True,
            separators=(",", ":"),
        )


@dataclass(frozen=True)
class _StatIdentity:
    device: int
    inode: int
    bytes: int
    modified_ns: int
    changed_ns: int


@dataclass(frozen=True)
class VerifiedAssetFile:
    """A manifest-locked file identity that must be rechecked before opening."""

    asset_id: str
    relative_path: str
    kind: str
    bytes: int
    sha256: str
    _base: Path
    _path: Path
    _identity: _StatIdentity

    def require_unchanged(self) -> Path:
        """Return the verified path only while its filesystem identity is unchanged."""

        if (
            _safe_path(self._base, self.relative_path) != self._path
            or self._path.is_symlink()
            or _stat_identity(self._path) != self._identity
        ):
            raise AssetPreflightError(
                f"verified asset identity drifted: {self.asset_id}:{self.relative_path}"
            )
        return self._path

    def verified_root(self) -> Path:
        """Return this file's asset root only while its identity is unchanged."""

        self.require_unchanged()
        return self._base


@dataclass(frozen=True)
class VerifiedAssetSelection:
    """Consumable identities from one manifest and filesystem verification pass."""

    manifest_revision: int
    manifest_sha256: str
    files: tuple[VerifiedAssetFile, ...]
    _root: Path
    _manifest_relative_path: str
    _manifest_path: Path
    _manifest_identity: _StatIdentity

    def require_unchanged(self) -> None:
        """Hard-fail if the manifest or any verified file changed after inspection."""

        if (
            _safe_path(self._root, self._manifest_relative_path) != self._manifest_path
            or self._manifest_path.is_symlink()
            or _stat_identity(self._manifest_path) != self._manifest_identity
        ):
            raise AssetPreflightError("verified asset manifest identity drifted")
        for verified_file in self.files:
            verified_file.require_unchanged()

    def verified_path(self, asset_id: str, relative_path: str) -> Path:
        """Revalidate the complete selection and return one declared file path."""

        self.require_unchanged()
        matches = [
            item
            for item in self.files
            if item.asset_id == asset_id and item.relative_path == relative_path
        ]
        if len(matches) != 1:
            raise AssetPreflightError(f"file was not verified: {asset_id}:{relative_path}")
        return matches[0].require_unchanged()

    def verified_root(self, asset_id: str) -> Path:
        """Return one manifest-bound asset root after revalidating every locked file."""

        self.require_unchanged()
        matches = [item for item in self.files if item.asset_id == asset_id]
        roots = {item.verified_root() for item in matches}
        if not matches or len(roots) != 1:
            raise AssetPreflightError(f"asset root was not verified: {asset_id}")
        root = roots.pop()
        if _safe_path(self._root, root.relative_to(self._root).as_posix()) != root:
            raise AssetPreflightError(f"verified asset root drifted: {asset_id}")
        return root


def require_verified_selection(value: object) -> VerifiedAssetSelection:
    """Narrow an untrusted wrapper argument to a live verified asset capability."""

    if not isinstance(value, VerifiedAssetSelection):
        raise AssetPreflightError("asset selection capability is not verified")
    value.require_unchanged()
    return value


@dataclass(frozen=True)
class _ManifestSnapshot:
    manifest: AssetManifest
    sha256: str
    root: Path
    relative_path: str
    path: Path
    identity: _StatIdentity


def _stat_identity(path: Path) -> _StatIdentity | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    return _StatIdentity(
        device=stat.st_dev,
        inode=stat.st_ino,
        bytes=stat.st_size,
        modified_ns=stat.st_mtime_ns,
        changed_ns=stat.st_ctime_ns,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_path(root: Path, relative: str) -> Path | None:
    root = root.resolve()
    candidate = root / relative
    current = root
    for part in Path(relative).parts:
        current = current / part
        if current.is_symlink():
            return None
    try:
        candidate.resolve(strict=False).relative_to(root)
    except ValueError:
        return None
    return candidate


def _read_manifest_snapshot(manifest_path: Path, root: Path) -> _ManifestSnapshot:
    resolved_root = root.resolve(strict=True)
    candidate = manifest_path if manifest_path.is_absolute() else resolved_root / manifest_path
    try:
        lexical_relative = candidate.relative_to(resolved_root)
        candidate.resolve(strict=False).relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError("asset manifest must remain inside the repository root") from exc
    if ".." in lexical_relative.parts:
        raise ValueError("asset manifest must use a normalized repository-relative path")
    safe_manifest = _safe_path(resolved_root, lexical_relative.as_posix())
    if safe_manifest is None:
        raise ValueError("asset manifest must not contain symlink path components")
    before = _stat_identity(safe_manifest)
    if before is None or not safe_manifest.is_file():
        raise AssetPreflightError("asset manifest is missing or is not a regular file")
    try:
        payload = safe_manifest.read_bytes()
    except OSError as exc:
        raise AssetPreflightError("asset manifest could not be read") from exc
    after = _stat_identity(safe_manifest)
    if after is None or before != after:
        raise AssetPreflightError("asset manifest changed while it was being read")
    return _ManifestSnapshot(
        manifest=parse_manifest_bytes(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        root=resolved_root,
        relative_path=lexical_relative.as_posix(),
        path=safe_manifest,
        identity=after,
    )


def _inspect_file(
    asset_id: str,
    base: Path,
    lock: FileLock,
    *,
    verify_hash: bool,
) -> tuple[list[InspectionIssue], VerifiedAssetFile | None]:
    path = _safe_path(base, lock.path)
    if path is None:
        return [InspectionIssue(asset_id, "unsafe_path", lock.path)], None
    before = _stat_identity(path)
    if before is None:
        return [InspectionIssue(asset_id, "missing_file", lock.path)], None
    if not path.is_file():
        return [InspectionIssue(asset_id, "not_regular_file", lock.path)], None
    if before.bytes != lock.bytes:
        return [
            InspectionIssue(
                asset_id,
                "byte_size_mismatch",
                f"{lock.path}: expected {lock.bytes}, observed {before.bytes}; sha256 skipped",
            )
        ], None
    if not verify_hash:
        return [], None
    if lock.sha256 is None:
        return [InspectionIssue(asset_id, "missing_sha256", lock.path)], None
    observed_sha256 = _sha256(path)
    after = _stat_identity(path)
    if after is None or before != after or path.is_symlink():
        return [InspectionIssue(asset_id, "file_changed_during_check", lock.path)], None
    if observed_sha256 != lock.sha256:
        return [InspectionIssue(asset_id, "sha256_mismatch", lock.path)], None
    return [], VerifiedAssetFile(
        asset_id=asset_id,
        relative_path=lock.path,
        kind=lock.kind,
        bytes=lock.bytes,
        sha256=lock.sha256,
        _base=base,
        _path=path,
        _identity=after,
    )


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = cast(object, json.loads(path.read_text(encoding="utf-8")))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return cast(dict[str, Any], payload) if isinstance(payload, dict) else None


def _inspect_model_summary(root: Path, asset: ModelAsset) -> list[InspectionIssue]:
    base = _safe_path(root, asset.local_path)
    if base is None:
        return [InspectionIssue(asset.asset_id, "unsafe_path", asset.local_path)]
    config_lock = next((item for item in asset.files if item.kind == "config"), None)
    if config_lock is None:
        return [InspectionIssue(asset.asset_id, "missing_config_lock", "config")]
    config_path = _safe_path(base, config_lock.path)
    payload = _read_json(config_path) if config_path is not None else None
    if payload is None:
        return [InspectionIssue(asset.asset_id, "invalid_config_json", config_lock.path)]
    issues: list[InspectionIssue] = []
    if isinstance(asset, QwenAsset):
        text_config = payload.get("text_config")
        if not isinstance(text_config, dict):
            return [InspectionIssue(asset.asset_id, "invalid_qwen_config", "text_config")]
        typed_text_config = cast(dict[str, Any], text_config)
        observed = {
            "layers": typed_text_config.get("num_hidden_layers"),
            "hidden_size": typed_text_config.get("hidden_size"),
        }
        expected = {"layers": asset.summary.layers, "hidden_size": asset.summary.hidden_size}
        if observed != expected:
            issues.append(
                InspectionIssue(
                    asset.asset_id,
                    "qwen_architecture_mismatch",
                    json.dumps(observed, sort_keys=True),
                )
            )
    else:
        observed = {
            "latent_channels": payload.get("latent_channels"),
            "downsample_factor": payload.get("downsample_factor"),
            "sample_posterior": payload.get("sample_posterior"),
        }
        expected = {
            "latent_channels": asset.summary.latent_channels,
            "downsample_factor": asset.summary.downsample_factor,
            "sample_posterior": asset.summary.sample_posterior,
        }
        if observed != expected:
            issues.append(
                InspectionIssue(
                    asset.asset_id,
                    "vae_interface_mismatch",
                    json.dumps(observed, sort_keys=True),
                )
            )
    return issues


_REFERENCE_GIT_COMMANDS = frozenset(
    {
        ("rev-parse", "--verify", "HEAD^{commit}"),
        ("remote", "get-url", "origin"),
        ("status", "--porcelain=v1", "--untracked-files=no"),
    }
)


def _git(reference: ReferenceAsset, root: Path, *args: str) -> str | None:
    if args not in _REFERENCE_GIT_COMMANDS:
        return None
    path = _safe_path(root, reference.local_path)
    if path is None:
        return None
    try:
        result = subprocess.run(
            (
                "git",
                "--no-pager",
                "--no-optional-locks",
                "-c",
                "core.fsmonitor=false",
                "-c",
                "core.hooksPath=/dev/null",
                "-c",
                "core.pager=cat",
                "-c",
                "pager.status=false",
                "-c",
                "diff.external=",
                "-c",
                "interactive.diffFilter=",
                "-C",
                str(path),
                *args,
            ),
            check=True,
            capture_output=True,
            env={
                "GIT_CONFIG_GLOBAL": "/dev/null",
                "GIT_CONFIG_NOSYSTEM": "1",
                "GIT_CONFIG_SYSTEM": "/dev/null",
                "GIT_EXTERNAL_DIFF": "",
                "GIT_OPTIONAL_LOCKS": "0",
                "GIT_PAGER": "cat",
                "GIT_TERMINAL_PROMPT": "0",
                "PAGER": "cat",
            },
            stdin=subprocess.DEVNULL,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def _inspect_reference(root: Path, reference: ReferenceAsset) -> list[InspectionIssue]:
    issues: list[InspectionIssue] = []
    base = _safe_path(root, reference.local_path)
    if base is None or not base.is_dir():
        return [InspectionIssue(reference.asset_id, "missing_reference", reference.local_path)]
    head = _git(reference, root, "rev-parse", "--verify", "HEAD^{commit}")
    if head != reference.commit:
        issues.append(InspectionIssue(reference.asset_id, "commit_mismatch", head or "unavailable"))
    origin = _git(reference, root, "remote", "get-url", "origin")
    if origin != reference.origin_url:
        issues.append(InspectionIssue(reference.asset_id, "origin_mismatch", "redacted"))
    tracked_status = _git(reference, root, "status", "--porcelain=v1", "--untracked-files=no")
    if tracked_status is None or tracked_status:
        issues.append(
            InspectionIssue(
                reference.asset_id,
                "tracked_worktree_not_clean",
                "dirty" if tracked_status else "unavailable",
            )
        )
    for license_lock in reference.licenses:
        file_lock = FileLock(
            path=license_lock.path,
            kind="license",
            bytes=license_lock.bytes,
            sha256=license_lock.sha256,
        )
        file_issues, _ = _inspect_file(reference.asset_id, base, file_lock, verify_hash=True)
        issues.extend(file_issues)
    return issues


def _inspect_locked_asset(
    root: Path,
    asset: ModelAsset | DatabaseAsset,
    *,
    verify_hashes: bool,
) -> tuple[list[InspectionIssue], list[VerifiedAssetFile]]:
    issues = [InspectionIssue(asset.asset_id, "asset_blocked", blocker) for blocker in asset.blockers]
    verified: list[VerifiedAssetFile] = []
    base = _safe_path(root, asset.local_path)
    if base is None or not base.is_dir():
        issues.append(InspectionIssue(asset.asset_id, "missing_asset_root", asset.local_path))
        return issues, verified
    for lock in asset.files:
        file_issues, verified_file = _inspect_file(
            asset.asset_id,
            base,
            lock,
            verify_hash=verify_hashes,
        )
        issues.extend(file_issues)
        if verified_file is not None:
            verified.append(verified_file)
    if isinstance(asset, (QwenAsset, VaeAsset)):
        issues.extend(_inspect_model_summary(root, asset))
    return issues, verified


def _report(manifest: AssetManifest, issues: list[InspectionIssue]) -> InspectionReport:
    ordered = tuple(sorted(issues, key=lambda item: (item.asset_id, item.code, item.detail)))
    return InspectionReport(manifest.manifest_revision, not ordered, ordered)


def _inspect_runtime_snapshot(
    snapshot: _ManifestSnapshot,
    root: Path,
) -> tuple[InspectionReport, tuple[VerifiedAssetFile, ...]]:
    issues: list[InspectionIssue] = []
    verified: list[VerifiedAssetFile] = []
    verify_hashes = not any(asset.lock_state == "blocked" for asset in snapshot.manifest.models)
    for asset in snapshot.manifest.models:
        asset_issues, asset_files = _inspect_locked_asset(
            root,
            asset,
            verify_hashes=verify_hashes,
        )
        issues.extend(asset_issues)
        verified.extend(asset_files)
    return _report(snapshot.manifest, issues), tuple(verified)


def _selected_databases(
    manifest: AssetManifest,
    asset_ids: Sequence[str],
) -> tuple[DatabaseAsset, ...]:
    selected_ids = tuple(asset_ids)
    if not selected_ids:
        raise ValueError("database audit requires at least one explicit asset ID")
    if len(set(selected_ids)) != len(selected_ids):
        raise ValueError("database audit asset IDs must be unique")
    inventory = {asset.asset_id: asset for asset in manifest.databases}
    unknown = sorted(set(selected_ids) - inventory.keys())
    if unknown:
        raise ValueError("unknown database asset IDs: " + ",".join(unknown))
    return tuple(inventory[asset_id] for asset_id in selected_ids)


def _inspect_database_snapshot(
    snapshot: _ManifestSnapshot,
    root: Path,
    asset_ids: Sequence[str],
) -> tuple[InspectionReport, tuple[VerifiedAssetFile, ...]]:
    selected = _selected_databases(snapshot.manifest, asset_ids)
    issues: list[InspectionIssue] = []
    verified: list[VerifiedAssetFile] = []
    verify_hashes = not any(asset.lock_state == "blocked" for asset in selected)
    for asset in selected:
        asset_issues, asset_files = _inspect_locked_asset(
            root,
            asset,
            verify_hashes=verify_hashes,
        )
        issues.extend(asset_issues)
        verified.extend(asset_files)
    return _report(snapshot.manifest, issues), tuple(verified)


def _selection(
    snapshot: _ManifestSnapshot,
    files: tuple[VerifiedAssetFile, ...],
) -> VerifiedAssetSelection:
    selection = VerifiedAssetSelection(
        manifest_revision=snapshot.manifest.manifest_revision,
        manifest_sha256=snapshot.sha256,
        files=files,
        _root=snapshot.root,
        _manifest_relative_path=snapshot.relative_path,
        _manifest_path=snapshot.path,
        _manifest_identity=snapshot.identity,
    )
    selection.require_unchanged()
    return selection


def inspect_runtime_models(manifest_path: Path, *, root: Path) -> InspectionReport:
    """Diagnostic-only model lock inspection; it does not bind runtime config."""

    snapshot = _read_manifest_snapshot(manifest_path, root)
    report, _ = _inspect_runtime_snapshot(snapshot, root.resolve())
    return report


def require_runtime_assets_ready(
    config: AssetsConfig,
    manifest_path: Path,
    *,
    root: Path,
) -> VerifiedAssetSelection:
    """Bind config and verify runtime models from one root-confined manifest snapshot."""

    snapshot = _read_manifest_snapshot(manifest_path, root)
    require_runtime_assets_match_snapshot(config, snapshot.manifest, snapshot.sha256)
    report, files = _inspect_runtime_snapshot(snapshot, root.resolve())
    if not report.ok:
        codes = ",".join(sorted({issue.code for issue in report.issues}))
        raise AssetPreflightError(f"runtime model preflight failed: {codes}")
    return _selection(snapshot, files)


def inspect_databases(
    manifest_path: Path,
    *,
    root: Path,
    asset_ids: Sequence[str],
) -> InspectionReport:
    """Hash selected DB payloads without opening a database or reading rows."""

    snapshot = _read_manifest_snapshot(manifest_path, root)
    report, _ = _inspect_database_snapshot(snapshot, root.resolve(), asset_ids)
    return report


def require_databases_ready(
    manifest_path: Path,
    *,
    root: Path,
    asset_ids: Sequence[str],
) -> VerifiedAssetSelection:
    """Verify selected DB identities before any database library opens them."""

    snapshot = _read_manifest_snapshot(manifest_path, root)
    report, files = _inspect_database_snapshot(snapshot, root.resolve(), asset_ids)
    if not report.ok:
        codes = ",".join(sorted({issue.code for issue in report.issues}))
        raise AssetPreflightError(f"database asset audit failed: {codes}")
    return _selection(snapshot, files)


def inspect_reference_repositories(manifest_path: Path, *, root: Path) -> InspectionReport:
    """Audit ignored development references without affecting runtime readiness."""

    snapshot = _read_manifest_snapshot(manifest_path, root)
    issues: list[InspectionIssue] = []
    for reference in snapshot.manifest.references:
        issues.extend(_inspect_reference(root.resolve(), reference))
    return _report(snapshot.manifest, issues)


def iter_declared_paths(manifest: AssetManifest) -> Iterable[str]:
    """Yield logical asset roots for repository-boundary audits."""

    for asset in (*manifest.models, *manifest.databases, *manifest.references):
        yield asset.local_path
