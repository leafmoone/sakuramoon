"""Root-confined runtime readiness and explicit asset audit boundaries."""

from __future__ import annotations

import hashlib
import json
import subprocess
import threading
import weakref
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


class _IdentityWeakRegistry:
    """Track immutable issuance snapshots without extending object lifetime."""

    def __init__(self) -> None:
        self._records: dict[
            int,
            tuple[weakref.ReferenceType[object], tuple[object, ...]],
        ] = {}
        self._lock = threading.Lock()

    def issue(self, value: object, fingerprint: tuple[object, ...]) -> None:
        identity = id(value)

        def discard(reference: weakref.ReferenceType[object]) -> None:
            with self._lock:
                record = self._records.get(identity)
                if record is not None and record[0] is reference:
                    del self._records[identity]

        reference = weakref.ref(value, discard)
        with self._lock:
            self._records[identity] = (reference, fingerprint)

    def contains(self, value: object, fingerprint: tuple[object, ...]) -> bool:
        with self._lock:
            identity = id(value)
            record = self._records.get(identity)
            if record is None:
                return False
            reference, issued_fingerprint = record
            if reference() is not value:
                return False
            if issued_fingerprint != fingerprint:
                del self._records[identity]
                return False
            return True

    def __len__(self) -> int:
        with self._lock:
            return len(self._records)


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


@dataclass(frozen=True, slots=True, weakref_slot=True)
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

        fingerprint = _verified_file_fingerprint(self)
        if fingerprint is None or not _ISSUED_FILES.contains(self, fingerprint):
            raise AssetPreflightError("asset file capability is not verified")
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

        VerifiedAssetFile.require_unchanged(self)
        return self._base


@dataclass(frozen=True, slots=True, weakref_slot=True)
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

        fingerprint = _verified_selection_fingerprint(self)
        if fingerprint is None or not _ISSUED_SELECTIONS.contains(self, fingerprint):
            raise AssetPreflightError("asset selection capability is not verified")
        safe_manifest = _safe_path(self._root, self._manifest_relative_path)
        if (
            safe_manifest != self._manifest_path
            or self._manifest_path.is_symlink()
            or _stat_identity(self._manifest_path) != self._manifest_identity
        ):
            raise AssetPreflightError("verified asset manifest identity drifted")
        try:
            observed_sha256 = _sha256(self._manifest_path)
        except OSError as exc:
            raise AssetPreflightError(
                "verified asset manifest could not be revalidated"
            ) from exc
        if (
            observed_sha256 != self.manifest_sha256
            or self._manifest_path.is_symlink()
            or _stat_identity(self._manifest_path) != self._manifest_identity
        ):
            raise AssetPreflightError("verified asset manifest identity drifted")
        for verified_file in self.files:
            file_fingerprint = _verified_file_fingerprint(verified_file)
            if (
                file_fingerprint is None
                or not _ISSUED_FILES.contains(verified_file, file_fingerprint)
            ):
                raise AssetPreflightError("asset file capability is not verified")
            VerifiedAssetFile.require_unchanged(verified_file)

    def verified_path(self, asset_id: str, relative_path: str) -> Path:
        """Revalidate the complete selection and return one declared file path."""

        VerifiedAssetSelection.require_unchanged(self)
        matches = [
            item
            for item in self.files
            if item.asset_id == asset_id and item.relative_path == relative_path
        ]
        if len(matches) != 1:
            raise AssetPreflightError(f"file was not verified: {asset_id}:{relative_path}")
        return VerifiedAssetFile.require_unchanged(matches[0])

    def verified_root(self, asset_id: str) -> Path:
        """Return one manifest-bound asset root after revalidating every locked file."""

        VerifiedAssetSelection.require_unchanged(self)
        matches = [item for item in self.files if item.asset_id == asset_id]
        roots = {VerifiedAssetFile.verified_root(item) for item in matches}
        if not matches or len(roots) != 1:
            raise AssetPreflightError(f"asset root was not verified: {asset_id}")
        root = roots.pop()
        if _safe_path(self._root, root.relative_to(self._root).as_posix()) != root:
            raise AssetPreflightError(f"verified asset root drifted: {asset_id}")
        return root


def require_verified_selection(value: object) -> VerifiedAssetSelection:
    """Narrow an untrusted wrapper argument to a live verified asset capability."""

    if type(value) is not VerifiedAssetSelection:
        raise AssetPreflightError("asset selection capability is not verified")
    selection = value
    fingerprint = _verified_selection_fingerprint(selection)
    if fingerprint is None or not _ISSUED_SELECTIONS.contains(selection, fingerprint):
        raise AssetPreflightError("asset selection capability is not verified")
    VerifiedAssetSelection.require_unchanged(selection)
    return selection


_ISSUED_FILES = _IdentityWeakRegistry()
_ISSUED_SELECTIONS = _IdentityWeakRegistry()
_PATH_TYPE = type(Path())
def _path_fingerprint(value: object) -> tuple[str, ...] | None:
    if type(value) is not _PATH_TYPE:
        return None
    return value.parts


def _stat_fingerprint(value: object) -> tuple[int, int, int, int, int] | None:
    if type(value) is not _StatIdentity:
        return None
    state = vars(value)
    names = ("device", "inode", "bytes", "modified_ns", "changed_ns")
    if set(state) != set(names) or any(type(state[name]) is not int for name in names):
        return None
    return cast(tuple[int, int, int, int, int], tuple(state[name] for name in names))


def _verified_file_fingerprint(value: object) -> tuple[object, ...] | None:
    if type(value) is not VerifiedAssetFile:
        return None
    try:
        asset_id = object.__getattribute__(value, "asset_id")
        relative_path = object.__getattribute__(value, "relative_path")
        kind = object.__getattribute__(value, "kind")
        byte_count = object.__getattribute__(value, "bytes")
        sha256 = object.__getattribute__(value, "sha256")
        raw_base = object.__getattribute__(value, "_base")
        raw_path = object.__getattribute__(value, "_path")
        raw_identity = object.__getattribute__(value, "_identity")
    except AttributeError:
        return None
    if any(type(item) is not str for item in (asset_id, relative_path, kind, sha256)):
        return None
    if type(byte_count) is not int:
        return None
    base = _path_fingerprint(raw_base)
    path = _path_fingerprint(raw_path)
    identity = _stat_fingerprint(raw_identity)
    if base is None or path is None or identity is None:
        return None
    return (
        "file",
        asset_id,
        relative_path,
        kind,
        byte_count,
        sha256,
        base,
        path,
        identity,
    )


def _verified_selection_fingerprint(value: object) -> tuple[object, ...] | None:
    if type(value) is not VerifiedAssetSelection:
        return None
    try:
        manifest_revision = object.__getattribute__(value, "manifest_revision")
        manifest_sha256 = object.__getattribute__(value, "manifest_sha256")
        raw_files = object.__getattribute__(value, "files")
        raw_root = object.__getattribute__(value, "_root")
        manifest_relative_path = object.__getattribute__(
            value,
            "_manifest_relative_path",
        )
        raw_manifest_path = object.__getattribute__(value, "_manifest_path")
        raw_identity = object.__getattribute__(value, "_manifest_identity")
    except AttributeError:
        return None
    if type(manifest_revision) is not int:
        return None
    if type(manifest_sha256) is not str:
        return None
    if type(manifest_relative_path) is not str:
        return None
    if type(raw_files) is not tuple:
        return None
    files = cast(tuple[object, ...], raw_files)
    root = _path_fingerprint(raw_root)
    manifest_path = _path_fingerprint(raw_manifest_path)
    identity = _stat_fingerprint(raw_identity)
    if root is None or manifest_path is None or identity is None:
        return None
    return (
        "selection",
        manifest_revision,
        manifest_sha256,
        tuple(id(item) for item in files),
        root,
        manifest_relative_path,
        manifest_path,
        identity,
    )


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
    try:
        root = root.resolve()
    except (OSError, RuntimeError):
        return None
    candidate = root / relative
    current = root
    for part in Path(relative).parts:
        current = current / part
        if current.is_symlink():
            return None
    try:
        candidate.resolve(strict=False).relative_to(root)
    except (OSError, RuntimeError, ValueError):
        return None
    return candidate


def _read_manifest_snapshot(manifest_path: Path, root: Path) -> _ManifestSnapshot:
    try:
        resolved_root = root.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise AssetPreflightError("repository root is missing or inaccessible") from exc
    if not resolved_root.is_dir():
        raise AssetPreflightError("repository root is not a directory")
    candidate = manifest_path if manifest_path.is_absolute() else resolved_root / manifest_path
    try:
        lexical_relative = candidate.relative_to(resolved_root)
        candidate.resolve(strict=False).relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError("asset manifest must remain inside the repository root") from exc
    except (OSError, RuntimeError) as exc:
        raise AssetPreflightError("asset manifest path could not be resolved") from exc
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
    try:
        observed_sha256 = _sha256(path)
    except OSError:
        return [InspectionIssue(asset_id, "file_read_error", lock.path)], None
    after = _stat_identity(path)
    if after is None or before != after or path.is_symlink():
        return [InspectionIssue(asset_id, "file_changed_during_check", lock.path)], None
    if observed_sha256 != lock.sha256:
        return [InspectionIssue(asset_id, "sha256_mismatch", lock.path)], None
    verified_file = VerifiedAssetFile(
        asset_id=asset_id,
        relative_path=lock.path,
        kind=lock.kind,
        bytes=lock.bytes,
        sha256=lock.sha256,
        _base=base,
        _path=path,
        _identity=after,
    )
    fingerprint = _verified_file_fingerprint(verified_file)
    if fingerprint is None:
        raise AssetPreflightError("asset file capability could not be issued")
    _ISSUED_FILES.issue(verified_file, fingerprint)
    return [], verified_file


def _read_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        payload = cast(object, json.loads(path.read_text(encoding="utf-8")))
    except OSError:
        return None, "file_read_error"
    except (UnicodeError, json.JSONDecodeError):
        return None, "invalid_config_json"
    if not isinstance(payload, dict):
        return None, "invalid_config_json"
    return cast(dict[str, Any], payload), None


def _inspect_model_summary(root: Path, asset: ModelAsset) -> list[InspectionIssue]:
    base = _safe_path(root, asset.local_path)
    if base is None:
        return [InspectionIssue(asset.asset_id, "unsafe_path", asset.local_path)]
    config_lock = next((item for item in asset.files if item.kind == "config"), None)
    if config_lock is None:
        return [InspectionIssue(asset.asset_id, "missing_config_lock", "config")]
    config_path = _safe_path(base, config_lock.path)
    if config_path is None:
        return [InspectionIssue(asset.asset_id, "unsafe_path", config_lock.path)]
    payload, read_error = _read_json(config_path)
    if payload is None:
        return [
            InspectionIssue(
                asset.asset_id,
                read_error or "invalid_config_json",
                config_lock.path,
            )
        ]
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
    fingerprint = _verified_selection_fingerprint(selection)
    if fingerprint is None:
        raise AssetPreflightError("asset selection capability could not be issued")
    _ISSUED_SELECTIONS.issue(selection, fingerprint)
    VerifiedAssetSelection.require_unchanged(selection)
    return selection


def inspect_runtime_models(manifest_path: Path, *, root: Path) -> InspectionReport:
    """Diagnostic-only model lock inspection; it does not bind runtime config."""

    snapshot = _read_manifest_snapshot(manifest_path, root)
    report, _ = _inspect_runtime_snapshot(snapshot, snapshot.root)
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
    report, files = _inspect_runtime_snapshot(snapshot, snapshot.root)
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
    report, _ = _inspect_database_snapshot(snapshot, snapshot.root, asset_ids)
    return report


def require_databases_ready(
    manifest_path: Path,
    *,
    root: Path,
    asset_ids: Sequence[str],
) -> VerifiedAssetSelection:
    """Verify selected DB identities before any database library opens them."""

    snapshot = _read_manifest_snapshot(manifest_path, root)
    report, files = _inspect_database_snapshot(snapshot, snapshot.root, asset_ids)
    if not report.ok:
        codes = ",".join(sorted({issue.code for issue in report.issues}))
        raise AssetPreflightError(f"database asset audit failed: {codes}")
    return _selection(snapshot, files)


def inspect_reference_repositories(manifest_path: Path, *, root: Path) -> InspectionReport:
    """Audit ignored development references without affecting runtime readiness."""

    snapshot = _read_manifest_snapshot(manifest_path, root)
    issues: list[InspectionIssue] = []
    for reference in snapshot.manifest.references:
        issues.extend(_inspect_reference(snapshot.root, reference))
    return _report(snapshot.manifest, issues)


def iter_declared_paths(manifest: AssetManifest) -> Iterable[str]:
    """Yield logical asset roots for repository-boundary audits."""

    for asset in (*manifest.models, *manifest.databases, *manifest.references):
        yield asset.local_path
