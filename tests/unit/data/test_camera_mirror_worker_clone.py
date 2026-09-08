"""Worker-clone mirror-policy propagation contracts (100U canary erratum).

The 100U canary's recorded exposure root cause was wrong: its own rank0
severity histogram (``severity_lt2 + severity_2to4 + severity_ge4 ==
vertical_applied`` by the fixed conservation invariant) proves 2221
vertical-applied rows with latent shift >= 2.0 existed, yet
``mirror_eligible`` was 0 for all 40000 logical samples.  The actual defect:
``ProductionPipelineFactory.pipeline_for_lease()`` passes the enabled mirror
policy to the parent pipeline, but the persistent data workers cross
``WebDatasetPipeline._with_local_shards()`` -- which omitted
``mirror_policy`` -- so every shard-local clone fell back to the ``__init__``
default (``None``) and never entered the mirror branch.

These tests pin:

* the clone contract (identity preservation of every policy + telemetry);
* the v1 eligibility boundary (1.99 / 2.0 / > 4 / horizontal / odd viewport);
* the canary-specific ``eligible == severe_vertical`` invariant, exercised
  through a CLONED pipeline -- never the parent;
* the full persistent-worker boundary
  (``_PersistentShardDataset -> _with_local_shards -> _iter_paths ->
  collate_samples``) through a real spawned DataLoader worker, which fails
  on the pre-fix code (``applied == 0``, ``physical == logical``) and passes
  after the one-line propagation fix.
"""

from __future__ import annotations

import io
import sys
import tarfile
import tempfile
import tomllib
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
from PIL import Image

from sakuramoon.config import load_config
from sakuramoon.config.schema import (
    DataBucketsConfig,
    DataTransparentBackgroundConfig,
)
from sakuramoon.data.buckets import (
    generate_base_buckets,
    scale_buckets,
)
from sakuramoon.data.camera_viewport import (
    MIRROR_MODE_V1,
    CameraMirrorCounts,
    CameraMirrorPolicy,
    CameraViewportPolicy,
    aggregate_camera_mirror,
    camera_mirror_eligible,
    camera_stage_edge,
)
from sakuramoon.data.caption import (
    CaptionDropoutProbabilities,
    CaptionFields,
    NlCandidates,
    NlDropoutProbabilities,
)
from sakuramoon.data.collate import (
    _build_batch_loader,  # pyright: ignore[reportPrivateUsage]
    _PersistentShardDataset,  # pyright: ignore[reportPrivateUsage]
    _ShardWork,  # pyright: ignore[reportPrivateUsage]
    _shutdown_loader,  # pyright: ignore[reportPrivateUsage]
    _WorkerBatch,
    _WorkerDone,
)
from sakuramoon.data.manifest import ShardRecord
from sakuramoon.data.metadata import MetadataFieldMapping
from sakuramoon.data.pipeline import (
    PipelineSample,
    WebDatasetPipeline,
)
from sakuramoon.data.production import (
    _require_spawn_serializable,  # pyright: ignore[reportPrivateUsage]
)
from sakuramoon.data.serialize import (
    MAIN_SUFFIX,
    SYSTEM_PREFIX,
    FramingContract,
)
from sakuramoon.data.spatial_crop import SpatialCropPolicy
from sakuramoon.data.transparent_white import TransparentWhiteTelemetry

_STAGING_EDGE = 512
_SHARD = "data/synthetic/mirror-worker-clone-000000.tar"


# ---------------------------------------------------------------------------
# Fixtures (module-level so the spawned DataLoader worker can re-import them)
# ---------------------------------------------------------------------------


def _buckets():
    config = DataBucketsConfig(
        base_area_px=262144,
        quantum_px=32,
        min_short_edge_px=256,
        max_aspect_ratio=4.0,
        shape_count=17,
        transpose_closed=True,
    )
    return scale_buckets(generate_base_buckets(config), _STAGING_EDGE)


BUCKETS = _buckets()


def _gradient_png(width: int, height: int) -> bytes:
    """A lossless RGB image with a vertical gradient (top != bottom)."""

    x = np.arange(width, dtype=np.uint8)
    y = np.arange(height, dtype=np.uint8).reshape(-1, 1)
    red = np.broadcast_to(x, (height, width))
    green = np.broadcast_to(y, (height, width))
    blue = np.broadcast_to((x + y) // 2, (height, width))
    array = np.stack((red, green, blue), axis=-1).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(array, mode="RGB").save(buffer, format="PNG")
    return buffer.getvalue()


class _Tokenizer:
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        del add_special_tokens
        if not text:
            # The real tokenizer encodes the empty string to no tokens; the
            # collate condition-source/null-routing invariant relies on that
            # (no condition -> no condition tokens -> use_null_condition).
            return []
        if text == SYSTEM_PREFIX:
            return list(range(34))
        if text == MAIN_SUFFIX:
            return list(range(100, 105))
        return [123]


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


def _identity_adapter(metadata: object) -> object:
    return metadata


def _rejection_observer(_reason: str) -> None:
    return None


def _camera_policy(probability: float = 1.0) -> CameraViewportPolicy:
    return CameraViewportPolicy(
        enabled=True,
        probability=probability,
        min_equivalent_zoom=1.10,
        max_equivalent_zoom=1.50,
    )


def _mirror_policy(
    *,
    enabled: bool = True,
    pair_probability: float = 1.0,
) -> CameraMirrorPolicy:
    return CameraMirrorPolicy(
        enabled=enabled,
        mode=MIRROR_MODE_V1,
        min_latent_shift=2.0,
        pair_probability=pair_probability,
        pair_weight=1.0,
    )


def _spatial_policy() -> SpatialCropPolicy:
    return SpatialCropPolicy(
        enabled=True,
        probability=0.0,
        min_equivalent_zoom=1.10,
        max_equivalent_zoom=1.50,
        min_crop_retention=0.8,
    )


def _transparent_policy() -> DataTransparentBackgroundConfig:
    return DataTransparentBackgroundConfig(enabled=False)


def _pipeline(
    *,
    camera_policy: CameraViewportPolicy | None,
    mirror_policy: CameraMirrorPolicy | None = None,
    spatial_policy: SpatialCropPolicy | None = None,
    transparent_policy: DataTransparentBackgroundConfig | None = None,
    transparent_telemetry: TransparentWhiteTelemetry | None = None,
) -> WebDatasetPipeline:
    pipeline = object.__new__(WebDatasetPipeline)
    pipeline.metadata_adapter = _identity_adapter
    pipeline.metadata_fields = MetadataFieldMapping(id_field="id")
    pipeline.base_seed = 7
    pipeline.stage = "S0"
    pipeline.cycle_index = 0
    pipeline.caption_fields_parser = _fields
    pipeline.probabilities = _probabilities()
    pipeline.condition_mode = "artist_or_character"
    pipeline.tokenizer = _Tokenizer()
    pipeline.framing = FramingContract(34, 5, 248044)
    pipeline.buckets = BUCKETS
    pipeline.min_crop_retention = 0.8
    pipeline.rejection_observer = _rejection_observer
    pipeline.spatial_policy = spatial_policy
    pipeline.transparent_policy = transparent_policy
    pipeline.transparent_telemetry = (
        transparent_telemetry
        if transparent_telemetry is not None
        else TransparentWhiteTelemetry()  # pyright: ignore[reportAttributeAccessIssue]
    )
    pipeline.camera_policy = camera_policy
    pipeline.mirror_policy = mirror_policy
    pipeline._camera_stage_edge = camera_stage_edge(BUCKETS)  # pyright: ignore[reportAttributeAccessIssue, reportPrivateUsage]
    return cast(Any, pipeline)


def _process_sample(
    pipeline: WebDatasetPipeline,
    image_bytes: bytes,
    *,
    sample_id: int,
):
    sample = {
        "__url__": _SHARD,
        "__key__": f"synthetic/{sample_id:06d}",
        "json": b'{"id": ' + str(sample_id).encode("ascii") + b"}",
        "png": image_bytes,
    }
    result = pipeline._process(  # pyright: ignore[reportPrivateUsage]
        sample,
        {_SHARD: ShardRecord(path=_SHARD, bytes=1)},
    )
    assert result is not None, "decodable sample must not be rejected"
    return result


def _clone(
    pipeline: WebDatasetPipeline,
    *,
    cycle_index: int = 0,
    shard_path: Path | None = None,
):
    """Drive the exact production worker boundary (_with_local_shards).

    The clone's ``__init__`` validates that the shard path is an absolute
    local regular file, so pass a real temporary file (the clone never
    opens it during ``_process``; only ``_iter_paths`` reads the tar).
    """

    shard = shard_path if shard_path is not None else Path(_SHARD)
    # The record carries the relative manifest path (ShardRecord contract);
    # the pipeline validates the local file path separately.
    return pipeline._with_local_shards(  # pyright: ignore[reportPrivateUsage]
        (shard,),
        (ShardRecord(path=_SHARD, bytes=1),),
        cycle_index=cycle_index,
    )


# ---------------------------------------------------------------------------
# Clone contract
# ---------------------------------------------------------------------------


def _dummy_shard(tmp_path: Path) -> Path:
    # The local file path must end with the manifest shard path
    # (_validate_shard_records), so reproduce the manifest layout.
    shard = tmp_path / _SHARD
    shard.parent.mkdir(parents=True, exist_ok=True)
    shard.write_bytes(b"")
    return shard


def test_clone_preserves_policy_identity(tmp_path: Path):
    spatial = _spatial_policy()
    camera = _camera_policy()
    mirror = _mirror_policy()
    transparent = _transparent_policy()
    telemetry = TransparentWhiteTelemetry()
    parent = _pipeline(
        camera_policy=camera,
        mirror_policy=mirror,
        spatial_policy=spatial,
        transparent_policy=transparent,
        transparent_telemetry=telemetry,
    )
    clone = _clone(parent, cycle_index=3, shard_path=_dummy_shard(tmp_path))
    # Identity sharing is the existing convention for these immutable
    # (or deliberately shared) policy objects.
    assert clone.spatial_policy is spatial
    assert clone.camera_policy is camera
    assert clone.mirror_policy is mirror
    assert clone.transparent_policy is transparent
    # Shared telemetry semantics are preserved (same counter object).
    assert clone.transparent_telemetry is telemetry
    assert clone.cycle_index == 3
    assert clone._lease_managed is True  # pyright: ignore[reportPrivateUsage]


def test_clone_preserves_none_policies(tmp_path: Path):
    parent = _pipeline(camera_policy=_camera_policy())
    assert parent.mirror_policy is None
    assert parent.spatial_policy is None
    assert parent.transparent_policy is None
    clone = _clone(parent, shard_path=_dummy_shard(tmp_path))
    assert clone.mirror_policy is None
    assert clone.camera_policy is parent.camera_policy
    assert clone.spatial_policy is None
    assert clone.transparent_policy is None


# ---------------------------------------------------------------------------
# v1 eligibility boundary
# ---------------------------------------------------------------------------


def _plan(
    *,
    applied: bool = True,
    orientation: str = "vertical",
    viewport: int = 256,
    shift: float = 2.0,
):
    return SimpleNamespace(
        applied=applied,
        fallback_reason="none",
        orientation=orientation,
        viewport=viewport,
        full_width=viewport,
        full_height=viewport,
        crop_box=(0, 0, viewport, viewport),
        left=0,
        top=0,
        equivalent_zoom=1.25,
        retention=0.9,
        normalized_offset=0.2,
        signed_pixel_center_shift=shift * 16.0,
        absolute_pixel_center_shift=shift * 16.0,
        latent_center_shift=shift,
        camera_shift_x=0.0,
        camera_shift_y=shift * 16.0,
    )


def test_v1_eligibility_boundary_values() -> None:
    # The canary contract: viewport 256 (even), min_latent_shift exactly 2.0.
    assert camera_mirror_eligible(_plan(shift=1.99), min_latent_shift=2.0) is False
    assert camera_mirror_eligible(_plan(shift=2.0), min_latent_shift=2.0) is True
    assert camera_mirror_eligible(_plan(shift=4.5), min_latent_shift=2.0) is True
    # Horizontal views are excluded regardless of severity.
    assert (
        camera_mirror_eligible(
            _plan(orientation="horizontal", shift=3.0), min_latent_shift=2.0
        )
        is False
    )
    # Odd viewports cannot give the exact signed-shift antisymmetry.
    assert (
        camera_mirror_eligible(_plan(viewport=255, shift=3.0), min_latent_shift=2.0)
        is False
    )
    # Fallback / not-selected plans are ineligible by construction.
    assert (
        camera_mirror_eligible(_plan(applied=False, shift=3.0), min_latent_shift=2.0)
        is False
    )


# ---------------------------------------------------------------------------
# Eligibility through the CLONED pipeline (never the parent)
# ---------------------------------------------------------------------------


def _scan_cloned(
    image_bytes: bytes,
    *,
    mirror_policy: CameraMirrorPolicy,
    count: int,
    shard_path: Path,
) -> list[PipelineSample]:
    parent = _pipeline(camera_policy=_camera_policy(), mirror_policy=mirror_policy)
    clone = _clone(parent, shard_path=shard_path)
    rows: list[PipelineSample] = []
    for sample_id in range(1, count + 1):
        rows.append(_process_sample(clone, image_bytes, sample_id=sample_id))
    return rows


def _severe_vertical(rows: list[PipelineSample]) -> list[PipelineSample]:
    return [
        row
        for row in rows
        if row.audit.camera_applied
        and row.audit.camera_orientation == "vertical"
        and abs(row.audit.camera_latent_center_shift) >= 2.0
    ]


def test_cloned_pipeline_applies_severe_vertical_rows(tmp_path: Path) -> None:
    image_bytes = _gradient_png(512, 1024)
    rows = _scan_cloned(
        image_bytes,
        mirror_policy=_mirror_policy(),
        count=200,
        shard_path=_dummy_shard(tmp_path),
    )
    severe = _severe_vertical(rows)
    assert len(severe) >= 5, f"expected >=5 severe vertical rows, got {len(severe)}"
    for row in rows:
        audit = row.audit
        if audit.camera_applied and audit.camera_orientation == "vertical":
            # Canary invariant at the row level (pair_probability = 1.0,
            # even viewport): eligible is EXACTLY the severe-vertical set.
            expect = abs(audit.camera_latent_center_shift) >= 2.0
            assert row.mirror_eligible is expect, (
                f"row {row.sample_id}: shift="
                f"{audit.camera_latent_center_shift:.4f} eligible="
                f"{row.mirror_eligible}"
            )
            assert row.mirror_selected is expect
            if expect:
                assert row.mirror is not None, (
                    f"row {row.sample_id}: severe vertical row must carry a "
                    "mirror payload through the cloned pipeline"
                )
            else:
                assert row.mirror is None
        else:
            # Horizontal / non-applied rows never enter the mirror path.
            assert row.mirror_eligible is False
            assert row.mirror_selected is False
            assert row.mirror is None
            if audit.camera_applied:
                assert audit.camera_orientation == "horizontal"


# ---------------------------------------------------------------------------
# Canary-specific consistency (pinned to the v2 canary policy, §7)
# ---------------------------------------------------------------------------


def test_canary_policy_invariant_on_cloned_pipeline(tmp_path: Path) -> None:
    config = load_config(
        Path("train_g1_camera_v2_p25_mirror_v2_canary.toml"),
        config_root=Path("config"),
        validate_secrets=False,
    )
    mirror_cfg = config.config.data.camera_mirror_balance
    assert mirror_cfg is not None
    assert mirror_cfg.mode == MIRROR_MODE_V1
    assert mirror_cfg.min_latent_shift == 2.0
    assert mirror_cfg.pair_probability == 1.0
    assert mirror_cfg.pair_weight == 1.0
    policy = CameraMirrorPolicy.from_config(mirror_cfg)

    image_bytes = _gradient_png(512, 1024)
    rows = _scan_cloned(
        image_bytes,
        mirror_policy=policy,
        count=160,
        shard_path=_dummy_shard(tmp_path),
    )
    counts = aggregate_camera_mirror(rows)
    severe = _severe_vertical(rows)
    # The v1 canary contract: eligible == severe-vertical, selected ==
    # eligible, applied == selected, no degradation, exact view accounting.
    assert counts.mirror_eligible == len(severe)
    assert counts.mirror_selected == counts.mirror_eligible
    assert counts.mirror_applied == counts.mirror_selected
    assert counts.mirror_eligible >= 3, f"expected >=3 severe rows, got {len(severe)}"
    assert counts.severity_lt2 + counts.severity_2to4 + counts.severity_ge4 == (
        counts.vertical_applied
    )
    assert counts.mirror_extra_views == counts.mirror_applied
    assert counts.physical_views == counts.logical_samples + counts.mirror_extra_views
    assert counts.physical_views > counts.logical_samples


# ---------------------------------------------------------------------------
# Persistent-worker regression (the exact production structural boundary)
# ---------------------------------------------------------------------------


def _make_shard(path: Path, layout: tuple[tuple[int, int], ...]) -> ShardRecord:
    with tarfile.open(path, "w") as tar:
        for index, (width, height) in enumerate(layout):
            image_bytes = _gradient_png(width, height)
            for name, payload in (
                (f"{index:06d}.png", image_bytes),
                (f"{index:06d}.json", b'{"id": ' + str(index).encode() + b"}"),
            ):
                info = tarfile.TarInfo(name=name)
                info.size = len(payload)
                tar.addfile(info, io.BytesIO(payload))
    # The record carries the relative manifest path (ShardRecord contract);
    # the absolute local file path is tracked separately by _ShardWork.
    return ShardRecord(path=_SHARD, bytes=path.stat().st_size)


def test_persistent_worker_exposes_mirror_through_clone() -> None:
    # 48 tall + 16 wide synthetic sources: the tall population drives the
    # vertical-applied severe rows, the wide population proves horizontal
    # rows never enter the mirror path inside the worker.
    layout = ((512, 1024),) * 48 + ((1024, 512),) * 16
    with tempfile.TemporaryDirectory(prefix="mbs-worker-clone-") as tmp:
        local_path = Path(tmp) / _SHARD
        local_path.parent.mkdir(parents=True, exist_ok=True)
        record = _make_shard(local_path, layout)

        parent = _pipeline(
            camera_policy=_camera_policy(),
            mirror_policy=_mirror_policy(),
        )
        # The same explicit spawn-serializability gate the production
        # factory applies before handing the pipeline to workers.
        _require_spawn_serializable(parent, "worker-clone test pipeline")

        dataset = _PersistentShardDataset(
            parent,
            batch_size=8,
            drop_last=True,
            length_sort_window_batches=1,
            worker_count=1,
        )
        loader = _build_batch_loader(
            dataset,
            worker_count=1,
            ready_batches=1,
            pin_memory=False,
            worker_seed=parent.base_seed,
            in_order=False,
        )
        work = _ShardWork(
            shard_path=str(record.path),
            local_path=local_path,
            record=record,
            cycle_index=0,
        )
        iterator = iter(loader)
        dataset.submit(0, work)
        batches = []
        try:
            while True:
                item = next(iterator)
                if isinstance(item, _WorkerBatch):
                    batches.append(item.batch)
                elif isinstance(item, _WorkerDone):
                    break
                else:
                    raise TypeError(
                        "persistent worker output channel yielded an "
                        "unsupported item type"
                    )
        finally:
            dataset.stop(0, record)
            _shutdown_loader(
                loader, suppress_worker_failure=sys.exc_info()[0] is not None
            )

    assert batches, "persistent worker produced no batches"
    logical = 0
    agg: CameraMirrorCounts | None = None
    for batch in batches:
        # The core regression assertion: the worker-produced batch carries
        # the camera-mirror counters through the cloned pipeline.
        assert batch.camera_mirror is not None, (
            "worker-produced TrainingBatch must carry camera_mirror counts "
            "(pre-fix code produces the strict-zero table because the "
            "clone lost the mirror policy)"
        )
        assert len(batch.mirror) == len(batch.audits)
        logical += len(batch.audits)
        if agg is None:
            agg = batch.camera_mirror
        else:
            agg = _add_mirror_counts(agg, batch.camera_mirror)
        for row_index, audit in enumerate(batch.audits):
            payload = batch.mirror[row_index]
            if (
                audit.camera_applied
                and audit.camera_orientation == "vertical"
                and abs(audit.camera_latent_center_shift) >= 2.0
            ):
                assert payload is not None, (
                    f"row {audit} severe vertical row lost its mirror "
                    "payload in the persistent worker"
                )
            else:
                assert payload is None

    assert agg is not None
    assert agg.logical_samples == logical == 64
    # The production symptom this fix removes: pre-fix code reports
    # applied == 0 and physical == logical here.
    assert agg.mirror_applied >= 1
    assert agg.vertical_applied >= 1
    severe_vertical = agg.severity_2to4 + agg.severity_ge4
    assert severe_vertical >= 1
    assert agg.mirror_eligible == severe_vertical
    assert agg.mirror_selected == agg.mirror_eligible
    assert agg.mirror_applied == agg.mirror_selected
    assert agg.mirror_extra_views == agg.mirror_applied
    assert agg.physical_views == agg.logical_samples + agg.mirror_extra_views
    assert agg.physical_views > agg.logical_samples
    # Horizontal rows never entered the mirror path.
    assert agg.original_start + agg.original_center + agg.original_end == (
        agg.mirror_applied
    )


def _add_mirror_counts(
    a: CameraMirrorCounts, b: CameraMirrorCounts
) -> CameraMirrorCounts:
    fields = (
        "logical_samples",
        "vertical_applied",
        "mirror_eligible",
        "mirror_selected",
        "mirror_applied",
        "severity_lt2",
        "severity_2to4",
        "severity_ge4",
        "original_start",
        "original_center",
        "original_end",
        "mirror_start",
        "mirror_center",
        "mirror_end",
        "mirror_extra_views",
        "physical_views",
    )
    return CameraMirrorCounts(
        **{name: getattr(a, name) + getattr(b, name) for name in fields}
    )  # pyright: ignore[reportCallIssue]


# ---------------------------------------------------------------------------
# v2 canary config identity
# ---------------------------------------------------------------------------


def _config_diff_keys(a: Mapping, b: Mapping, prefix: str = "") -> set[str]:
    keys: set[str] = set()
    for name in set(a) | set(b):
        path = f"{prefix}.{name}" if prefix else str(name)
        if name not in a or name not in b:
            keys.add(path)
            continue
        value_a, value_b = a[name], b[name]
        if isinstance(value_a, Mapping) and isinstance(value_b, Mapping):
            keys |= _config_diff_keys(value_a, value_b, path)
        elif value_a != value_b:
            keys.add(path)
    return keys


def test_v2_canary_config_differs_only_in_identity() -> None:
    v1 = load_config(
        Path("train_g1_camera_v2_p25_mirror_canary.toml"),
        config_root=Path("config"),
        validate_secrets=False,
    )
    v2 = load_config(
        Path("train_g1_camera_v2_p25_mirror_v2_canary.toml"),
        config_root=Path("config"),
        validate_secrets=False,
    )
    before = tomllib.loads(v1.resolved_toml)
    after = tomllib.loads(v2.resolved_toml)
    diff = _config_diff_keys(before, after)
    identity_keys = {
        "run.run_id",
        "paths.run_dir",
        "paths.checkpoint_dir",
        "paths.artifact_dir",
        "logging.local_jsonl_path",
        "wandb.retry_jsonl_path",
        "evaluation.output_dir",
    }
    assert diff == identity_keys, diff
    # Everything the treatment inherits stays byte-identical.
    assert after["data"]["camera_mirror_balance"] == before["data"][
        "camera_mirror_balance"
    ]
    assert after["stage"]["canary_stop_successful_update"] == 118200
    assert after["stage"]["planned_updates"] == before["stage"]["planned_updates"]
    assert after["data"]["camera_viewport"] == before["data"]["camera_viewport"]
    assert v2.config.run.run_id == "g1_camera_v2_p25_mirror_v2"
