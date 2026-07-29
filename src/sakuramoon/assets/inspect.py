"""Fail-closed, read-only asset inspection before any model or DB load."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

from sakuramoon.assets.manifest import (
    AssetManifest,
    DatabaseAsset,
    FileLock,
    ModelAsset,
    QwenAsset,
    ReferenceAsset,
    VaeAsset,
    load_manifest,
)


class AssetPreflightError(RuntimeError):
    """Raised when an asset is not fully locked or has drifted."""


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


def _inspect_file(
    asset_id: str,
    base: Path,
    lock: FileLock,
    *,
    verify_hash: bool,
) -> list[InspectionIssue]:
    path = _safe_path(base, lock.path)
    if path is None:
        return [InspectionIssue(asset_id, "unsafe_path", lock.path)]
    try:
        stat = path.stat()
    except OSError:
        return [InspectionIssue(asset_id, "missing_file", lock.path)]
    issues: list[InspectionIssue] = []
    if not path.is_file():
        issues.append(InspectionIssue(asset_id, "not_regular_file", lock.path))
        return issues
    if stat.st_size != lock.bytes:
        issues.append(
            InspectionIssue(
                asset_id,
                "byte_size_mismatch",
                f"{lock.path}: expected {lock.bytes}, observed {stat.st_size}",
            )
        )
    if verify_hash:
        if lock.sha256 is None:
            issues.append(InspectionIssue(asset_id, "missing_sha256", lock.path))
        elif _sha256(path) != lock.sha256:
            issues.append(InspectionIssue(asset_id, "sha256_mismatch", lock.path))
    return issues


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
                InspectionIssue(asset.asset_id, "qwen_architecture_mismatch", json.dumps(observed, sort_keys=True))
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
                InspectionIssue(asset.asset_id, "vae_interface_mismatch", json.dumps(observed, sort_keys=True))
            )
    return issues


def _git(reference: ReferenceAsset, root: Path, *args: str) -> str | None:
    path = _safe_path(root, reference.local_path)
    if path is None:
        return None
    try:
        result = subprocess.run(
            ("git", "-C", str(path), *args),
            check=True,
            capture_output=True,
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
    head = _git(reference, root, "rev-parse", "HEAD")
    if head != reference.commit:
        issues.append(InspectionIssue(reference.asset_id, "commit_mismatch", head or "unavailable"))
    origin = _git(reference, root, "remote", "get-url", "origin")
    if origin != reference.origin_url:
        issues.append(InspectionIssue(reference.asset_id, "origin_mismatch", origin or "unavailable"))
    tracked_status = _git(reference, root, "status", "--porcelain", "--untracked-files=no")
    if tracked_status is None or tracked_status:
        issues.append(
            InspectionIssue(reference.asset_id, "tracked_worktree_not_clean", tracked_status or "unavailable")
        )
    for license_lock in reference.licenses:
        file_lock = FileLock(
            path=license_lock.path,
            kind="license",
            bytes=license_lock.bytes,
            sha256=license_lock.sha256,
        )
        issues.extend(_inspect_file(reference.asset_id, base, file_lock, verify_hash=True))
    return issues


def _inspect_locked_asset(
    root: Path,
    asset: ModelAsset | DatabaseAsset,
    *,
    verify_hashes: bool,
) -> list[InspectionIssue]:
    issues = [InspectionIssue(asset.asset_id, "asset_blocked", blocker) for blocker in asset.blockers]
    base = _safe_path(root, asset.local_path)
    if base is None or not base.is_dir():
        issues.append(InspectionIssue(asset.asset_id, "missing_asset_root", asset.local_path))
        return issues
    for lock in asset.files:
        issues.extend(_inspect_file(asset.asset_id, base, lock, verify_hash=verify_hashes))
    if isinstance(asset, (QwenAsset, VaeAsset)):
        issues.extend(_inspect_model_summary(root, asset))
    return issues


def inspect_assets(manifest_path: Path, *, root: Path) -> InspectionReport:
    """Inspect every declared asset without loading model weights or DB rows."""

    resolved_root = root.resolve()
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
    manifest: AssetManifest = load_manifest(safe_manifest)
    issues: list[InspectionIssue] = []
    verify_hashes = not any(
        asset.lock_state == "blocked"
        for asset in (
            *manifest.models,
            *(database for database in manifest.databases if database.required_for_runtime),
        )
    )
    for asset in manifest.models:
        issues.extend(_inspect_locked_asset(resolved_root, asset, verify_hashes=verify_hashes))
    for asset in manifest.databases:
        if asset.required_for_runtime:
            issues.extend(_inspect_locked_asset(resolved_root, asset, verify_hashes=verify_hashes))
    for reference in manifest.references:
        issues.extend(_inspect_reference(resolved_root, reference))
    ordered = tuple(sorted(issues, key=lambda item: (item.asset_id, item.code, item.detail)))
    return InspectionReport(manifest.manifest_revision, not ordered, ordered)


def require_assets_ready(manifest_path: Path, *, root: Path) -> InspectionReport:
    """Hard-fail before model or database loading when any lock is incomplete or drifted."""

    report = inspect_assets(manifest_path, root=root)
    if not report.ok:
        codes = ",".join(sorted({issue.code for issue in report.issues}))
        raise AssetPreflightError(f"asset preflight failed: {codes}")
    return report


def iter_declared_paths(manifest: AssetManifest) -> Iterable[str]:
    """Yield logical asset roots for repository-boundary audits."""

    for asset in (*manifest.models, *manifest.databases, *manifest.references):
        yield asset.local_path
