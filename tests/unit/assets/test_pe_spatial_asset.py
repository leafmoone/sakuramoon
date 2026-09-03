"""Asset-contract tests for the frozen PE-Spatial-B16-512 iREPA teacher.

Covers: correct fingerprint accepted, wrong hash / size / metadata
rejected, symlink and path-escape rejected, and absent/disabled iREPA
performing no asset requirement.  Uses a small deterministic synthetic
weight file (never the real 346MB asset).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch

from sakuramoon.assets.pe_spatial import (
    PE_SPATIAL_ASSET_METADATA_FILENAME,
    PE_SPATIAL_TEACHER_FILENAME,
    PeSpatialTeacherFingerprint,
    pe_spatial_teacher_required,
    require_local_pe_spatial_teacher,
    sha256_file,
    verify_pe_spatial_teacher_asset,
    write_pe_spatial_asset_metadata,
)
from sakuramoon.config.schema import IRepaConfig


def _irepa(**overrides: Any) -> IRepaConfig:
    fields: dict[str, Any] = {
        "enabled": False,
        "teacher_id": "facebook/PE-Spatial-B16-512",
        "teacher_local_path": "model/pe_spatial_b16_512",
        "tap_slot": 8,
        "projector_kernel_size": 3,
        "spatial_norm": "zscore",
        "spatial_norm_gamma": 0.6,
        "spatial_norm_eps": 0.000001,
        "loss": "cosine",
        "weight": 0.5,
        "ramp_in_updates": 1000,
        "ramp_out_after_updates": None,
        "ramp_out_updates": 1000,
    }
    fields.update(overrides)
    return IRepaConfig(**fields)


def _fingerprint(
    weight_path: Path,
    **overrides: Any,
) -> PeSpatialTeacherFingerprint:
    values: dict[str, Any] = {
        "teacher_id": "facebook/PE-Spatial-B16-512",
        "filename": PE_SPATIAL_TEACHER_FILENAME,
        "sha256": sha256_file(weight_path),
        "file_size_bytes": weight_path.stat().st_size,
        "patch_size": 16,
        "width": 768,
        "depth": 12,
        "input_normalization": "minus_one_to_one",
        "source_model": "facebook/PE-Spatial-B16-512",
    }
    values.update(overrides)
    return PeSpatialTeacherFingerprint(**values)


def _make_asset(
    tmp_path: Path,
    *,
    mutate_meta: Any = None,
) -> tuple[Path, Path, PeSpatialTeacherFingerprint]:
    """Build a valid synthetic asset directory; return (root, dir, fp)."""

    root = tmp_path / "repo"
    asset_dir = root / "model" / "pe_spatial_b16_512"
    asset_dir.mkdir(parents=True)
    weight_path = asset_dir / PE_SPATIAL_TEACHER_FILENAME
    torch.save(
        {"state_dict": {"conv1.weight": torch.arange(64.0).reshape(8, 8)}},
        weight_path,
    )
    fingerprint = _fingerprint(weight_path)
    if mutate_meta is not None:
        fingerprint = mutate_meta(fingerprint)
    write_pe_spatial_asset_metadata(asset_dir, fingerprint)
    return root, asset_dir, _fingerprint(weight_path)


def _verify(root: Path, fingerprint: PeSpatialTeacherFingerprint) -> Path:
    return verify_pe_spatial_teacher_asset(
        Path("model/pe_spatial_b16_512"),
        fingerprint,
        repository_root=root,
    )


def test_correct_fingerprint_accepted(tmp_path: Path) -> None:
    root, asset_dir, fingerprint = _make_asset(tmp_path)

    weight_path = _verify(root, fingerprint)
    assert weight_path == asset_dir / PE_SPATIAL_TEACHER_FILENAME

    loaded = require_local_pe_spatial_teacher(
        root, "model/pe_spatial_b16_512", expected=fingerprint
    )
    assert loaded == weight_path


def test_wrong_hash_rejected(tmp_path: Path) -> None:
    def flip_hash(fp: PeSpatialTeacherFingerprint) -> PeSpatialTeacherFingerprint:
        return _fingerprint(
            _weights(tmp_path), sha256="0" * 64
        )

    root, _, _ = _make_asset(tmp_path, mutate_meta=flip_hash)
    actual = _fingerprint(_weights(tmp_path))
    with pytest.raises(ValueError, match="field sha256"):
        _verify(root, actual)


def _weights(tmp_path: Path) -> Path:
    return (
        tmp_path
        / "repo"
        / "model"
        / "pe_spatial_b16_512"
        / PE_SPATIAL_TEACHER_FILENAME
    )


def test_wrong_size_rejected(tmp_path: Path) -> None:
    root, _, _ = _make_asset(
        tmp_path,
        mutate_meta=lambda fp: _fingerprint(
            _weights(tmp_path), file_size_bytes=fp.file_size_bytes + 1
        ),
    )
    with pytest.raises(ValueError, match="field file_size_bytes"):
        _verify(root, _fingerprint(_weights(tmp_path)))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("width", 512),
        ("depth", 8),
        ("patch_size", 32),
        ("input_normalization", "zero_to_one"),
        ("teacher_id", "facebook/PE-Spatial-S16-512"),
        ("source_model", "other/model"),
    ],
)
def test_wrong_metadata_rejected(
    tmp_path: Path, field: str, value: Any
) -> None:
    def corrupt(fp: PeSpatialTeacherFingerprint) -> PeSpatialTeacherFingerprint:
        return _fingerprint(_weights(tmp_path), **{field: value})

    root, _, _ = _make_asset(tmp_path, mutate_meta=corrupt)
    with pytest.raises(ValueError, match=f"field {field}"):
        _verify(root, _fingerprint(_weights(tmp_path)))


def test_hash_rejects_file_modified_after_metadata(tmp_path: Path) -> None:
    root, asset_dir, fingerprint = _make_asset(tmp_path)
    # Corrupt the weight file after the metadata was written: the SHA256
    # verification must reject the drift even though the metadata is
    # internally consistent.
    weight_path = asset_dir / PE_SPATIAL_TEACHER_FILENAME
    weight_path.write_bytes(b"\x00" * weight_path.stat().st_size)
    with pytest.raises(ValueError, match="SHA256"):
        _verify(root, fingerprint)


def test_symlink_weight_rejected(tmp_path: Path) -> None:
    root, asset_dir, _ = _make_asset(tmp_path)
    real = asset_dir / "real.pt"
    real.write_bytes(b"weights")
    weight_path = asset_dir / PE_SPATIAL_TEACHER_FILENAME
    weight_path.unlink()
    weight_path.symlink_to(real)

    fingerprint = _fingerprint(real, file_size_bytes=7)
    write_pe_spatial_asset_metadata(asset_dir, fingerprint)
    with pytest.raises(ValueError, match="symlink"):
        _verify(root, fingerprint)


def test_path_escape_rejected(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    (root / "a" / "b").mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    weight_path = outside / PE_SPATIAL_TEACHER_FILENAME
    torch.save({"x": torch.zeros(1)}, weight_path)
    fingerprint = _fingerprint(weight_path)
    write_pe_spatial_asset_metadata(outside, fingerprint)

    with pytest.raises(ValueError, match="escapes the repository root"):
        verify_pe_spatial_teacher_asset(
            Path("a/b/../../outside"),
            fingerprint,
            repository_root=root,
        )


def test_missing_directory_raises_file_not_found(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    fingerprint = PeSpatialTeacherFingerprint(
        teacher_id="facebook/PE-Spatial-B16-512",
        filename=PE_SPATIAL_TEACHER_FILENAME,
        sha256="0" * 64,
        file_size_bytes=1,
        patch_size=16,
        width=768,
        depth=12,
        input_normalization="minus_one_to_one",
        source_model="facebook/PE-Spatial-B16-512",
    )
    with pytest.raises(FileNotFoundError):
        _verify(root, fingerprint)


def test_absent_or_disabled_requires_no_teacher() -> None:
    assert pe_spatial_teacher_required(None) is False
    assert pe_spatial_teacher_required(_irepa()) is False
    assert pe_spatial_teacher_required(_irepa(enabled=True)) is True


def test_asset_json_roundtrip_matches_spec_fields(tmp_path: Path) -> None:
    _, asset_dir, _ = _make_asset(tmp_path)
    payload = json.loads(
        (asset_dir / PE_SPATIAL_ASSET_METADATA_FILENAME).read_text(encoding="utf-8")
    )
    assert set(payload) == {
        "teacher_id",
        "filename",
        "sha256",
        "file_size_bytes",
        "patch_size",
        "width",
        "depth",
        "input_normalization",
        "source_model",
    }
    assert payload["input_normalization"] == "minus_one_to_one"
    assert payload["patch_size"] == 16
    assert payload["width"] == 768
    assert payload["depth"] == 12
