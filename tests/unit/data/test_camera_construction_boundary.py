"""Construction-boundary contract for the shifted-square camera viewport.

The camera square-bucket requirement (``camera_stage_edge``) applies only
while the camera is EFFECTIVELY ACTIVE: a policy is present AND enabled
AND probability > 0.0. On an ordinary-legal stage vocabulary that contains
NO square bucket at all:

* every effectively-off shape (absent policy, enabled=false with p=0,
  enabled=false with p=1, enabled=true with p=0) constructs through the
  real ``WebDatasetPipeline.__init__`` (no ``object.__new__``), runs the
  ordinary path end to end, and leases a clone without any camera-specific
  requirement;
* an effectively-active camera policy must fail construction explicitly
  (``CameraViewportError``), never degrade into per-sample handling;
* the production factory applies the identical gate: the off shapes issue
  a lease pipeline from a real resolved config, and the active shape is
  still accepted on the standard square vocabulary.

Direct construction, factory issuance, and the lease clone (which re-runs
the real ``__init__``) therefore agree on activation.
"""

from __future__ import annotations

import io
import tarfile
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from sakuramoon.config import load_config
from sakuramoon.config.schema import DataBucketsConfig, RuntimeConfig
from sakuramoon.data.buckets import BucketShape, generate_base_buckets, scale_buckets
from sakuramoon.data.camera_viewport import (
    CameraViewportError,
    CameraViewportPolicy,
)
from sakuramoon.data.caption import (
    CaptionDropoutProbabilities,
    CaptionFields,
    NlCandidates,
    NlDropoutProbabilities,
)
from sakuramoon.data.manifest import ShardRecord
from sakuramoon.data.metadata import MetadataFieldMapping
from sakuramoon.data.pipeline import WebDatasetPipeline
from sakuramoon.data.production import ProductionPipelineFactory
from sakuramoon.data.serialize import (
    MAIN_SUFFIX,
    SYSTEM_PREFIX,
    FramingContract,
)
from sakuramoon.data.service_protocol import ShardLeaseDescriptor

R = 256  # stage edge of the no-square vocabulary
_SHARD_NAME = "shard-000000.tar"

# Module-level definitions only: the production factory exercises the spawn
# pickler on every field it is handed.


class _Tokenizer:
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        del add_special_tokens
        if not text:
            return []
        if text == SYSTEM_PREFIX:
            return list(range(34))
        if text == MAIN_SUFFIX:
            return list(range(100, 105))
        return [123]


def _identity_adapter(metadata: Mapping[str, object]) -> Mapping[str, object]:
    return metadata


def _fields(_raw: Mapping[str, object]) -> CaptionFields:
    return CaptionFields(
        nsfw=(),
        character=(),
        copyright=(),
        general=(),
        artists=(),
        candidate_tags=frozenset(),
        nl=NlCandidates(None, None, None, None, None),
    )


def _probabilities() -> CaptionDropoutProbabilities:
    nl = NlDropoutProbabilities(0.0, 0.0, 0.0, 0.0, 0.0)
    return CaptionDropoutProbabilities(
        condition_route=0.0,
        condition_only=0.0,
        tag=0.0,
        candidate_source=0.0,
        nl=nl,
    )


def _noop_observer(reason: str) -> None:
    del reason


def _gradient_png(width: int, height: int) -> bytes:
    xs = (
        np.tile((np.arange(width, dtype=np.uint16) * 7 + 3), height)
        .reshape(height, width)
        .astype(np.uint8)
    )
    ys = (
        np.repeat((np.arange(height, dtype=np.uint16) * 13 + 5), width)
        .reshape(height, width)
        .astype(np.uint8)
    )
    zs = (
        (np.tile(np.arange(width, dtype=np.uint16), height)
        + np.repeat(np.arange(height, dtype=np.uint16) * 3, width)
        + 11)
        .reshape(height, width)
        .astype(np.uint8)
    )
    buffer = io.BytesIO()
    Image.fromarray(np.stack([xs, ys, zs], axis=-1)).save(buffer, format="PNG")
    return buffer.getvalue()


def _no_square_buckets() -> tuple[BucketShape, ...]:
    """The real stage-256 vocabulary minus its (only) square bucket."""

    config = DataBucketsConfig(
        base_area_px=262144,
        quantum_px=32,
        min_short_edge_px=256,
        max_aspect_ratio=4.0,
        transpose_closed=True,
    )
    buckets = tuple(
        bucket
        for bucket in scale_buckets(generate_base_buckets(config), R)
        if bucket.width != bucket.height
    )
    assert buckets, "the stage vocabulary must not reduce to nothing"
    assert all(bucket.width != bucket.height for bucket in buckets)
    return buckets


def _write_shard(directory: Path) -> int:
    """Two ordinary-legal samples (2:1 and 1:2) for the no-square vocabulary."""

    shard = directory / _SHARD_NAME
    entries = (
        (1, (512, 256)),  # 2:1 horizontal
        (2, (256, 512)),  # 1:2 vertical
    )
    with tarfile.open(shard, "w") as tar:
        for sample_id, (width, height) in entries:
            payload = _gradient_png(width, height)
            info = tarfile.TarInfo(name=f"{sample_id:06d}.png")
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
            meta = b'{"id": ' + str(sample_id).encode("ascii") + b"}"
            info = tarfile.TarInfo(name=f"{sample_id:06d}.json")
            info.size = len(meta)
            tar.addfile(info, io.BytesIO(meta))
    return shard.stat().st_size


def _pipeline(
    tmp_path: Path,
    size: int,
    *,
    camera_policy: CameraViewportPolicy | None,
) -> WebDatasetPipeline:
    local_path = tmp_path / _SHARD_NAME
    return WebDatasetPipeline(
        shard_paths=(local_path,),
        shard_records=(ShardRecord(path=_SHARD_NAME, bytes=size),),
        metadata_adapter=_identity_adapter,
        metadata_fields=MetadataFieldMapping(id_field="id"),
        buckets=_no_square_buckets(),
        min_crop_retention=0.8,
        probabilities=_probabilities(),
        condition_mode="artist_or_character",
        tokenizer=_Tokenizer(),
        framing=FramingContract(34, 5, 248044),
        caption_fields_parser=_fields,
        rejection_observer=_noop_observer,
        base_seed=7,
        stage="S0",
        cycle_index=0,
        spatial_policy=None,
        camera_policy=camera_policy,
        transparent_policy=None,
    )


_OFF_SHAPES: tuple[tuple[str, CameraViewportPolicy | None], ...] = (
    ("absent", None),
    ("disabled_p0", CameraViewportPolicy(False, 0.0)),
    ("disabled_p1", CameraViewportPolicy(False, 1.0)),
    ("enabled_p0", CameraViewportPolicy(True, 0.0)),
)


class TestDirectConstructionBoundary:
    @pytest.mark.parametrize(
        ("label", "camera_policy"),
        _OFF_SHAPES,
        ids=[label for label, _ in _OFF_SHAPES],
    )
    def test_off_shapes_construct_and_run_ordinary_path(
        self,
        tmp_path: Path,
        label: str,
        camera_policy: CameraViewportPolicy | None,
    ) -> None:
        del label
        size = _write_shard(tmp_path)
        # Real __init__: an effectively-off camera must not trigger the
        # square-bucket requirement on this no-square vocabulary.
        pipeline = _pipeline(tmp_path, size, camera_policy=camera_policy)
        sample = {
            "__url__": _SHARD_NAME,
            "__key__": "000001.png",
            "json": b'{"id": 1}',
            "png": _gradient_png(512, 256),
        }
        result = pipeline._process(  # type: ignore[reportPrivateUsage]
            sample,
            {_SHARD_NAME: ShardRecord(path=_SHARD_NAME, bytes=size)},
        )
        assert result is not None
        assert result.audit.crop_policy == "aspect_bucket"
        assert result.audit.camera_policy == "none"
        assert result.audit.camera_selected is False
        assert result.audit.camera_applied is False

    def test_off_shapes_lease_clone_uses_ordinary_path(self, tmp_path: Path) -> None:
        # The clone re-runs the real __init__ with the same policy, so it
        # must apply the same (absent) camera requirement.
        size = _write_shard(tmp_path)
        for _label, camera_policy in _OFF_SHAPES:
            pipeline = _pipeline(tmp_path, size, camera_policy=camera_policy)
            clone = pipeline._with_local_shards(  # type: ignore[reportPrivateUsage]
                (tmp_path / _SHARD_NAME,),
                (ShardRecord(path=_SHARD_NAME, bytes=size),),
                cycle_index=1,
            )
            sample = {
                "__url__": _SHARD_NAME,
                "__key__": "000002.png",
                "json": b'{"id": 2}',
                "png": _gradient_png(256, 512),
            }
            result = clone._process(  # type: ignore[reportPrivateUsage]
                sample,
                {_SHARD_NAME: ShardRecord(path=_SHARD_NAME, bytes=size)},
            )
            assert result is not None
            assert result.audit.crop_policy == "aspect_bucket"

    @pytest.mark.parametrize(
        "probability",
        [0.01, 0.25, 1.0],
        ids=["p0.01", "p0.25", "p1.0"],
    )
    def test_active_camera_fails_fast_on_no_square_vocabulary(
        self, tmp_path: Path, probability: float
    ) -> None:
        size = _write_shard(tmp_path)
        with pytest.raises(CameraViewportError, match="square bucket"):
            _pipeline(
                tmp_path, size, camera_policy=CameraViewportPolicy(True, probability)
            )


class _TmpConfigRoot:
    """A config root mirroring the real one, for synthetic test configs."""

    def __init__(self, tmp_path: Path) -> None:
        self.root = tmp_path / "config"
        self.root.mkdir(exist_ok=True)
        for item in Path("config").glob("*.toml"):
            target = self.root / item.name
            if not target.exists():
                target.symlink_to(item.resolve())

    def write(self, name: str, text: str) -> str:
        (self.root / name).write_text(text, encoding="utf-8")
        return name


def _camera_toml(*, enabled: bool, probability: str) -> str:
    return (
        'extends = ["train_g1.toml"]\n'
        "\n"
        "[data.camera_viewport]\n"
        f"enabled = {str(enabled).lower()}\n"
        'mode = "hdm_shifted_square_v2"\n'
        f"probability = {probability}\n"
        'viewport = "stage_square"\n'
        'offset_distribution = "uniform_long_axis_inclusive"\n'
        'zoom_source = "natural_source_aspect"\n'
    )


def _is_active_camera(camera: CameraViewportPolicy | None) -> bool:
    return camera is not None and camera.enabled and camera.probability > 0.0


class TestFactoryConstructionBoundary:
    def _lease_descriptor(self, tmp_path: Path) -> ShardLeaseDescriptor:
        _write_shard(tmp_path)
        return ShardLeaseDescriptor(
            lease_id="lease-0",
            worker_id=0,
            cycle_index=0,
            state_revision=1,
            record=ShardRecord(path=_SHARD_NAME, bytes=1),
            local_path=tmp_path / _SHARD_NAME,
        )

    def _factory(
        self, config: RuntimeConfig, tmp_path: Path
    ) -> ProductionPipelineFactory:
        return ProductionPipelineFactory.from_config(
            config,
            repository_root=tmp_path,
            tokenizer=_Tokenizer(),
            framing=FramingContract(34, 5, 248044),
            rejection_observer=_noop_observer,
        )

    @pytest.mark.parametrize(
        ("name", "toml_text", "expect_policy"),
        [
            ("train_g1.toml", None, False),
            (
                "factory_off_disabled_p0.toml",
                _camera_toml(enabled=False, probability="0.0"),
                True,
            ),
            (
                "factory_off_disabled_p1.toml",
                _camera_toml(enabled=False, probability="1.0"),
                True,
            ),
            (
                "factory_off_enabled_p0.toml",
                _camera_toml(enabled=True, probability="0.0"),
                True,
            ),
        ],
        ids=["absent", "disabled_p0", "disabled_p1", "enabled_p0"],
    )
    def test_off_shapes_issue_lease_pipeline(
        self,
        tmp_path: Path,
        name: str,
        toml_text: str | None,
        expect_policy: bool,
    ) -> None:
        root = _TmpConfigRoot(tmp_path)
        if toml_text is not None:
            name = root.write(name, toml_text)
        config = load_config(
            Path(name), config_root=root.root, validate_secrets=False
        ).config
        factory = self._factory(config, tmp_path)
        # Real factory issuance on an effectively-off camera must not add
        # any camera-specific config requirement.
        pipeline = factory.pipeline_for_lease(self._lease_descriptor(tmp_path))
        assert isinstance(pipeline, WebDatasetPipeline)
        camera = pipeline.camera_policy
        assert (camera is not None) is expect_policy
        assert not _is_active_camera(camera)

    def test_active_shape_still_accepted_on_square_vocabulary(
        self, tmp_path: Path
    ) -> None:
        root = _TmpConfigRoot(tmp_path)
        config = load_config(
            Path("train_g1_camera_v2_p100.toml"),
            config_root=root.root,
            validate_secrets=False,
        ).config
        factory = self._factory(config, tmp_path)
        pipeline = factory.pipeline_for_lease(self._lease_descriptor(tmp_path))
        assert isinstance(pipeline, WebDatasetPipeline)
        assert _is_active_camera(pipeline.camera_policy)
