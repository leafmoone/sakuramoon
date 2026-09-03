"""Asset contract for the frozen PE-Spatial-B16-512 iREPA teacher.

The teacher weights are a local, fingerprint-bound asset
(``model/pe_spatial_b16_512/`` by default): the SHA256 of the weight file
and the ``asset.json`` metadata are verified fail-closed at load time.
Absent or disabled iREPA never touches this directory: the checks below
are only invoked when ``config.irepa.enabled is True``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from sakuramoon.config.schema import IRepaConfig

PE_SPATIAL_TEACHER_FILENAME = "PE-Spatial-B16-512.pt"
PE_SPATIAL_ASSET_METADATA_FILENAME = "asset.json"

# Approved upstream asset: Hugging Face facebook/PE-Spatial-B16-512, file
# PE-Spatial-B16-512.pt, fetched 2025-09-02.  The SHA256 below is the
# verification constant; a mismatched local file is a hard failure, never
# an auto-accepted re-download.
APPROVED_PE_SPATIAL_B16_512_SHA256 = (
    "86217607f0bb28c0adb5ac3f9b0608ae22f6fb634bf1c16b2316847e8148a2a5"
)
APPROVED_PE_SPATIAL_B16_512_SIZE_BYTES = 345_783_707

_SHA256_CHUNK_BYTES = 1 << 20


@dataclass(frozen=True)
class PeSpatialTeacherFingerprint:
    """Binding between the teacher asset and the approved model identity."""

    teacher_id: str
    filename: str
    sha256: str
    file_size_bytes: int
    patch_size: int
    width: int
    depth: int
    input_normalization: str
    source_model: str


APPROVED_PE_SPATIAL_B16_512 = PeSpatialTeacherFingerprint(
    teacher_id="facebook/PE-Spatial-B16-512",
    filename=PE_SPATIAL_TEACHER_FILENAME,
    sha256=APPROVED_PE_SPATIAL_B16_512_SHA256,
    file_size_bytes=APPROVED_PE_SPATIAL_B16_512_SIZE_BYTES,
    patch_size=16,
    width=768,
    depth=12,
    input_normalization="minus_one_to_one",
    source_model="facebook/PE-Spatial-B16-512",
)


def sha256_file(path: Path) -> str:
    """Stream a file's SHA256 in fixed-size chunks without full residency."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_SHA256_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _require_inside_root(resolved: Path, root: Path, label: str) -> None:
    if not resolved.is_relative_to(root):
        raise ValueError(f"{label} escapes the repository root")


def _require_regular_file(path: Path, label: str) -> None:
    if path.is_symlink():
        raise ValueError(f"{label} must not be a symlink: {path.as_posix()}")
    if not path.is_file():
        raise FileNotFoundError(f"required {label} file is missing: {path.as_posix()}")


def verify_pe_spatial_teacher_asset(
    asset_dir: Path,
    expected: PeSpatialTeacherFingerprint,
    *,
    repository_root: Path | None = None,
) -> Path:
    """Fail-closed verification of a local PE-Spatial teacher asset.

    Returns the verified weight file path.  Missing files raise
    ``FileNotFoundError``; any fingerprint, metadata, size, hash, symlink,
    or path-escape mismatch raises ``ValueError``.
    """

    if repository_root is None:
        resolved_root: Path | None = None
        resolved_dir = asset_dir
    else:
        # Fail closed on any ``..`` component before resolution: a relative
        # or absolute asset path that contains a parent traversal is
        # rejected even when it would resolve back inside the root.
        if ".." in Path(asset_dir).parts:
            raise ValueError("teacher asset directory escapes the repository root")
        resolved_root = repository_root.resolve()
        resolved_dir = (repository_root / asset_dir).resolve()
        _require_inside_root(resolved_dir, resolved_root, "teacher asset directory")

    if not resolved_dir.is_dir():
        raise FileNotFoundError(
            f"required PE-Spatial teacher asset directory is missing: "
            f"{resolved_dir.as_posix()}"
        )

    metadata_path = resolved_dir / PE_SPATIAL_ASSET_METADATA_FILENAME
    _require_regular_file(metadata_path, "teacher asset metadata")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"teacher asset metadata is not readable JSON: {exc}") from exc
    if not isinstance(metadata, dict):
        raise TypeError("teacher asset metadata must be a JSON object")
    metadata = cast("dict[str, object]", metadata)

    for field in (
        "teacher_id",
        "filename",
        "sha256",
        "file_size_bytes",
        "patch_size",
        "width",
        "depth",
        "input_normalization",
        "source_model",
    ):
        actual = metadata.get(field)
        want = getattr(expected, field)
        if type(actual) is not type(want) or actual != want:
            raise ValueError(
                f"teacher asset metadata field {field} does not match the "
                f"approved fingerprint (expected {want!r}, got {actual!r})"
            )

    weight_path = resolved_dir / expected.filename
    _require_regular_file(weight_path, "teacher weight file")
    if resolved_root is not None:
        _require_inside_root(weight_path, resolved_root, "teacher weight file")

    size = weight_path.stat().st_size
    if size != expected.file_size_bytes:
        raise ValueError(
            "teacher weight file size does not match the approved "
            f"fingerprint (expected {expected.file_size_bytes}, got {size})"
        )
    actual_sha256 = sha256_file(weight_path)
    if actual_sha256 != expected.sha256:
        raise ValueError(
            "teacher weight file SHA256 does not match the approved "
            f"fingerprint (expected {expected.sha256}, got {actual_sha256})"
        )
    return weight_path


def write_pe_spatial_asset_metadata(
    asset_dir: Path,
    fingerprint: PeSpatialTeacherFingerprint,
) -> Path:
    """Write the canonical ``asset.json`` for an approved teacher asset."""

    asset_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "teacher_id": fingerprint.teacher_id,
        "filename": fingerprint.filename,
        "sha256": fingerprint.sha256,
        "file_size_bytes": fingerprint.file_size_bytes,
        "patch_size": fingerprint.patch_size,
        "width": fingerprint.width,
        "depth": fingerprint.depth,
        "input_normalization": fingerprint.input_normalization,
        "source_model": fingerprint.source_model,
    }
    metadata_path = asset_dir / PE_SPATIAL_ASSET_METADATA_FILENAME
    metadata_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata_path


def require_local_pe_spatial_teacher(
    repository_root: Path,
    teacher_local_path: str,
    *,
    expected: PeSpatialTeacherFingerprint = APPROVED_PE_SPATIAL_B16_512,
) -> Path:
    """Verify the teacher asset at the configured repository-relative path.

    Only call this when ``config.irepa.enabled is True``; absent or
    disabled iREPA must perform no asset I/O at all.
    """

    return verify_pe_spatial_teacher_asset(
        Path(teacher_local_path),
        expected,
        repository_root=repository_root,
    )


def pe_spatial_teacher_required(irepa: IRepaConfig | None) -> bool:
    """Whether the frozen teacher asset is required for this config."""

    return irepa is not None and irepa.enabled is True


__all__ = [
    "APPROVED_PE_SPATIAL_B16_512",
    "APPROVED_PE_SPATIAL_B16_512_SHA256",
    "APPROVED_PE_SPATIAL_B16_512_SIZE_BYTES",
    "PE_SPATIAL_ASSET_METADATA_FILENAME",
    "PE_SPATIAL_TEACHER_FILENAME",
    "PeSpatialTeacherFingerprint",
    "pe_spatial_teacher_required",
    "require_local_pe_spatial_teacher",
    "sha256_file",
    "verify_pe_spatial_teacher_asset",
    "write_pe_spatial_asset_metadata",
]
