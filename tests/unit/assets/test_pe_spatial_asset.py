"""Asset-contract tests for the frozen PE-Spatial-B16-512 iREPA teacher.

Covers: local files, size and model metadata checks without weight-content
reads, symlink and path-escape rejection, and absent/disabled iREPA requiring
no asset. Uses small synthetic files, never the real teacher weights.
"""

from __future__ import annotations

import io
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pytest

from sakuramoon.assets.pe_spatial import (
    PE_SPATIAL_ASSET_METADATA_FILENAME,
    PE_SPATIAL_TEACHER_FILENAME,
    PeSpatialTeacherFingerprint,
    pe_spatial_teacher_required,
    require_local_pe_spatial_teacher,
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
    mutate_meta: Callable[[PeSpatialTeacherFingerprint], PeSpatialTeacherFingerprint] | None = None,
) -> tuple[Path, Path, PeSpatialTeacherFingerprint]:
    """Build a valid synthetic asset directory; return (root, dir, fp)."""

    root = tmp_path / "repo"
    asset_dir = root / "model" / "pe_spatial_b16_512"
    asset_dir.mkdir(parents=True)
    weight_path = asset_dir / PE_SPATIAL_TEACHER_FILENAME
    weight_path.write_bytes(b"synthetic teacher weights")
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


def test_metadata_without_sha256_accepted(tmp_path: Path) -> None:
    root, asset_dir, fingerprint = _make_asset(tmp_path)

    weight_path = _verify(root, fingerprint)
    assert weight_path == asset_dir / PE_SPATIAL_TEACHER_FILENAME

    loaded = require_local_pe_spatial_teacher(
        root, "model/pe_spatial_b16_512", expected=fingerprint
    )
    assert loaded == weight_path


@pytest.mark.parametrize("extra_value", [None, "not a digest", {"uninspected": [1, False]}])
def test_extra_metadata_sha256_is_ignored(tmp_path: Path, extra_value: Any) -> None:
    root, asset_dir, fingerprint = _make_asset(tmp_path)
    metadata_path = asset_dir / PE_SPATIAL_ASSET_METADATA_FILENAME
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    payload["sha256"] = extra_value
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")

    assert _verify(root, fingerprint) == asset_dir / PE_SPATIAL_TEACHER_FILENAME
    assert require_local_pe_spatial_teacher(
        root, "model/pe_spatial_b16_512", expected=fingerprint
    ) == asset_dir / PE_SPATIAL_TEACHER_FILENAME


def _weights(tmp_path: Path) -> Path:
    return (
        tmp_path
        / "repo"
        / "model"
        / "pe_spatial_b16_512"
        / PE_SPATIAL_TEACHER_FILENAME
    )


def test_wrong_size_metadata_rejected(tmp_path: Path) -> None:
    root, _, _ = _make_asset(
        tmp_path,
        mutate_meta=lambda fp: _fingerprint(
            _weights(tmp_path), file_size_bytes=fp.file_size_bytes + 1
        ),
    )
    with pytest.raises(ValueError, match="field file_size_bytes"):
        _verify(root, _fingerprint(_weights(tmp_path)))


@pytest.mark.parametrize("size_delta", [-1, 1])
def test_wrong_weight_file_size_rejected(tmp_path: Path, size_delta: int) -> None:
    root, asset_dir, fingerprint = _make_asset(tmp_path)
    weight_path = asset_dir / PE_SPATIAL_TEACHER_FILENAME
    weight_path.write_bytes(b"x" * (fingerprint.file_size_bytes + size_delta))

    with pytest.raises(ValueError, match="teacher weight file size"):
        _verify(root, fingerprint)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("filename", "other.pt"),
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


def test_asset_checks_do_not_read_weight_contents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, asset_dir, fingerprint = _make_asset(tmp_path)
    weight_path = asset_dir / PE_SPATIAL_TEACHER_FILENAME
    original_open = io.open

    def guarded_open(file: Any, *args: Any, **kwargs: Any) -> Any:
        if not isinstance(file, int) and Path(file) == weight_path:
            pytest.fail("asset checks must not open the weight file")
        return cast("io.IOBase", original_open(file, *args, **kwargs))

    monkeypatch.setattr(io, "open", guarded_open)
    monkeypatch.setattr("builtins.open", guarded_open)

    assert _verify(root, fingerprint) == weight_path
    assert require_local_pe_spatial_teacher(
        root, "model/pe_spatial_b16_512", expected=fingerprint
    ) == weight_path


@pytest.mark.parametrize(
    "filename", [PE_SPATIAL_TEACHER_FILENAME, PE_SPATIAL_ASSET_METADATA_FILENAME]
)
def test_symlink_file_rejected(tmp_path: Path, filename: str) -> None:
    root, asset_dir, fingerprint = _make_asset(tmp_path)
    path = asset_dir / filename
    real = asset_dir / f"real-{filename}"
    path.rename(real)
    path.symlink_to(real)

    with pytest.raises(ValueError, match="symlink"):
        _verify(root, fingerprint)


def test_path_escape_rejected(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    (root / "a" / "b").mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    weight_path = outside / PE_SPATIAL_TEACHER_FILENAME
    weight_path.write_bytes(b"synthetic teacher weights")
    fingerprint = _fingerprint(weight_path)
    write_pe_spatial_asset_metadata(outside, fingerprint)

    for asset_path in (Path("a/b/../../outside"), outside):
        with pytest.raises(ValueError, match="escapes the repository root"):
            verify_pe_spatial_teacher_asset(
                asset_path,
                fingerprint,
                repository_root=root,
            )


def test_missing_directory_raises_file_not_found(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    fingerprint = PeSpatialTeacherFingerprint(
        teacher_id="facebook/PE-Spatial-B16-512",
        filename=PE_SPATIAL_TEACHER_FILENAME,
        file_size_bytes=1,
        patch_size=16,
        width=768,
        depth=12,
        input_normalization="minus_one_to_one",
        source_model="facebook/PE-Spatial-B16-512",
    )
    with pytest.raises(FileNotFoundError):
        _verify(root, fingerprint)


@pytest.mark.parametrize(
    "filename", [PE_SPATIAL_TEACHER_FILENAME, PE_SPATIAL_ASSET_METADATA_FILENAME]
)
def test_missing_file_raises_file_not_found(tmp_path: Path, filename: str) -> None:
    root, asset_dir, fingerprint = _make_asset(tmp_path)
    (asset_dir / filename).unlink()

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
