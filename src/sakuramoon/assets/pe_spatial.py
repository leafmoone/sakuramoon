"""Local PE-Spatial teacher asset checks for iREPA.

The teacher weights are a local asset (``model/pe_spatial_b16_512/`` by
default): the weight file's presence and size and the ``asset.json`` model
metadata are checked at load time, without reading the weight contents.
Absent or disabled iREPA never touches this directory: the checks below
are only invoked when ``config.irepa.enabled is True``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from sakuramoon.config.schema import IRepaConfig

PE_SPATIAL_TEACHER_FILENAME = "PE-Spatial-B16-512.pt"
PE_SPATIAL_ASSET_METADATA_FILENAME = "asset.json"

APPROVED_PE_SPATIAL_B16_512_SIZE_BYTES = 345_783_707


@dataclass(frozen=True)
class PeSpatialTeacherFingerprint:
    """Expected local asset metadata for the approved teacher model."""

    teacher_id: str
    filename: str
    file_size_bytes: int
    patch_size: int
    width: int
    depth: int
    input_normalization: str
    source_model: str


APPROVED_PE_SPATIAL_B16_512 = PeSpatialTeacherFingerprint(
    teacher_id="facebook/PE-Spatial-B16-512",
    filename=PE_SPATIAL_TEACHER_FILENAME,
    file_size_bytes=APPROVED_PE_SPATIAL_B16_512_SIZE_BYTES,
    patch_size=16,
    width=768,
    depth=12,
    input_normalization="minus_one_to_one",
    source_model="facebook/PE-Spatial-B16-512",
)


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
    """Check the local PE-Spatial teacher files and model metadata.

    Returns the weight file path without reading its contents. Missing
    files raise ``FileNotFoundError``; metadata, size, symlink, or
    path-escape mismatches raise ``ValueError``.
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
                f"approved model specification (expected {want!r}, got {actual!r})"
            )

    weight_path = resolved_dir / expected.filename
    _require_regular_file(weight_path, "teacher weight file")
    if resolved_root is not None:
        _require_inside_root(weight_path, resolved_root, "teacher weight file")

    size = weight_path.stat().st_size
    if size != expected.file_size_bytes:
        raise ValueError(
            "teacher weight file size does not match the approved "
            f"model specification (expected {expected.file_size_bytes}, got {size})"
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
    "APPROVED_PE_SPATIAL_B16_512_SIZE_BYTES",
    "PE_SPATIAL_ASSET_METADATA_FILENAME",
    "PE_SPATIAL_TEACHER_FILENAME",
    "PeSpatialTeacherFingerprint",
    "pe_spatial_teacher_required",
    "require_local_pe_spatial_teacher",
    "verify_pe_spatial_teacher_asset",
    "write_pe_spatial_asset_metadata",
]
