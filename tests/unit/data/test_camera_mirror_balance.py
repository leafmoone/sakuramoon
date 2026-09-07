"""Guarantees for the v1 vertical mirror-balanced supervision policy.

Covers the mirror-balance design contracts end to end at the data layer:

* strict config surface (v1 mode literal, exact 2.0 threshold, pair weight
  exactly 1.0, enabled implies camera_viewport enabled);
* pair geometry (k_mirror = available - k, exact signed-shift antisymmetry,
  same zoom / retention / full canvas, mirror-of-mirror round trip,
  start/end/center side exchange, odd viewport excluded);
* fixed-key mirror counters and their conservation equations;
* worker-level pipeline behavior (disabled bit-identity, eligibility and
  selection, payload coherence against an independent canvas re-derivation,
  degradation never rejects);
* collate-level row alignment and payload validation.
"""

from __future__ import annotations

import io
import shutil
import tomllib
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch
from PIL import Image

from sakuramoon.config import ConfigurationError, load_config
from sakuramoon.config.schema import (
    DataBucketsConfig,
    DataCameraMirrorBalanceConfig,
)
from sakuramoon.data.buckets import (
    BucketRejection,
    assign_bucket,
    generate_base_buckets,
    scale_buckets,
)
from sakuramoon.data.camera_viewport import (
    MIRROR_MODE_V1,
    CameraMirrorCounts,
    CameraMirrorPayload,
    CameraMirrorPolicy,
    CameraViewportError,
    CameraViewportPlan,
    CameraViewportPolicy,
    aggregate_camera_mirror,
    camera_mirror_eligible,
    camera_stage_edge,
    mirror_severity_band,
    mirror_side,
    plan_camera_viewport,
    plan_mirror_geometry,
    zero_camera_mirror_counts,
)
from sakuramoon.data.caption import (
    CaptionDropoutProbabilities,
    CaptionFields,
    CaptionPlan,
    NlCandidates,
    NlDropoutProbabilities,
    empty_caption_dropout_hits,
)
from sakuramoon.data.collate import CollateError, collate_samples
from sakuramoon.data.image_ops import normalize_image
from sakuramoon.data.manifest import ShardRecord
from sakuramoon.data.metadata import MetadataFieldMapping
from sakuramoon.data.pipeline import (
    ImageAudit,
    PipelineSample,
    RngIdentity,
    WebDatasetPipeline,
)
from sakuramoon.data.serialize import (
    MAIN_SUFFIX,
    SYSTEM_PREFIX,
    FramingContract,
    SerializedCaption,
)
from sakuramoon.data.transparent_white import TransparentWhiteTelemetry

_STAGING_EDGE = 512
_SHARD = "data/synthetic/mirror-000000.tar"


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


def _assignment(width: int, height: int):
    result = assign_bucket(width, height, BUCKETS, min_crop_retention=0.8)
    assert not isinstance(result, BucketRejection)
    return result


def _plan(
    width: int,
    height: int,
    *,
    probability: float = 1.0,
    policy_seed: int = 1,
    offset_seed: int = 1,
):
    policy = CameraViewportPolicy(
        enabled=True,
        probability=probability,
        min_equivalent_zoom=1.10,
        max_equivalent_zoom=1.50,
    )
    return plan_camera_viewport(
        _assignment(width, height),
        policy,
        buckets=BUCKETS,
        stage_edge=_STAGING_EDGE,
        source_size=(width, height),
        policy_seed=policy_seed,
        offset_seed=offset_seed,
    )


def _vertical_plans(seeds: int = 200):
    """Deterministic vertical camera plans (512x1024 portrait source)."""

    return [_plan(512, 1024, offset_seed=seed) for seed in range(seeds)]


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


# ---------------------------------------------------------------------------
# Config schema
# ---------------------------------------------------------------------------


class _TmpConfigRoot:
    """A config root mirroring the real one, for synthetic test configs."""

    def __init__(self, tmp_path: Path) -> None:
        self.root = tmp_path / "config"
        self.root.mkdir(exist_ok=True)
        for item in Path("config").glob("*.toml"):
            target = self.root / item.name
            if not target.exists():
                # The config loader forbids symlinks, so the config root is
                # built from plain copies (read-only for the test's lifetime).
                shutil.copy2(item.resolve(), target)

    def write(self, name: str, text: str) -> str:
        (self.root / name).write_text(text, encoding="utf-8")
        return name


CAMERA_TABLE = """
[data.camera_viewport]
enabled = true
mode = "hdm_shifted_square_v2"
probability = 0.25
viewport = "stage_square"
min_equivalent_zoom = 1.10
max_equivalent_zoom = 1.50
offset_distribution = "uniform_long_axis_inclusive"
zoom_source = "natural_source_aspect"
fallback_to_aspect_bucket = true
"""

DISABLED_CAMERA_TABLE = """
[data.camera_viewport]
enabled = false
mode = "hdm_shifted_square_v2"
probability = 0.0
viewport = "stage_square"
min_equivalent_zoom = 1.10
max_equivalent_zoom = 1.50
offset_distribution = "uniform_long_axis_inclusive"
zoom_source = "natural_source_aspect"
fallback_to_aspect_bucket = true
"""

MIRROR_TABLE = """
[data.camera_mirror_balance]
enabled = {enabled}
mode = "vertical_mirror_pair_v1"
min_latent_shift = {min_latent_shift}
pair_probability = {pair_probability}
pair_weight = {pair_weight}
"""

# First-version mutual exclusion: an enabled camera_viewport requires the
# production spatial p50 policy to be disabled (exactly as the p25 canary
# lineage does).
SPATIAL_OFF_TABLE = """
[data.spatial_crop]
enabled = false
mode = "shifted_bucket"
probability = 0.0
min_equivalent_zoom = 1.02
max_equivalent_zoom = 1.10
zoom_distribution = "sqrt_uniform_high"
offset_distribution = "uniform_independent"
fallback_to_aspect_bucket = true
"""


def _load_config(name: str):
    return load_config(
        Path(name), config_root=Path("config"), validate_secrets=False
    )


def _load_canary_table(
    tmp_path: Path, *, camera: str, mirror: str
) -> None:
    root = _TmpConfigRoot(tmp_path)
    name = root.write(
        "mirror_cfg_test.toml",
        'extends = ["train_g1_cmuon_production.toml"]\n'
        f"{camera}\n{SPATIAL_OFF_TABLE}\n{mirror}",
    )
    load_config(
        Path(name), config_root=root.root, validate_secrets=False
    )


def _reject_canary_table(tmp_path: Path, *, camera: str, mirror: str) -> None:
    with pytest.raises(ConfigurationError):
        _load_canary_table(tmp_path, camera=camera, mirror=mirror)


def test_schema_accepts_canary_values(tmp_path: Path) -> None:
    _load_canary_table(
        tmp_path,
        camera=CAMERA_TABLE,
        mirror=MIRROR_TABLE.format(
            enabled="true",
            min_latent_shift="2.0",
            pair_probability="1.0",
            pair_weight="1.0",
        ),
    )


def test_schema_disabled_default_is_absent_and_inert(tmp_path: Path) -> None:
    # Absent table: the production resolved TOML keeps zero mirror bytes
    # (byte-identity is pinned separately by the camera config suite).
    loaded = _load_config("train_g1_cmuon_production.toml")
    assert loaded.config.data.camera_mirror_balance is None
    assert "camera_mirror" not in loaded.resolved_toml

    # Disabled table: valid, and the policy object stays off.
    root = _TmpConfigRoot(tmp_path)
    name = root.write(
        "mirror_off.toml",
        'extends = ["train_g1_cmuon_production.toml"]\n'
        f"{CAMERA_TABLE}\n"
        + SPATIAL_OFF_TABLE
        + MIRROR_TABLE.format(
            enabled="false",
            min_latent_shift="2.0",
            pair_probability="0.0",
            pair_weight="1.0",
        ),
    )
    loaded = load_config(
        Path(name), config_root=root.root, validate_secrets=False
    )
    assert loaded.config.data.camera_mirror_balance.enabled is False
    policy = CameraMirrorPolicy.from_config(
        loaded.config.data.camera_mirror_balance
    )
    assert policy is not None and policy.enabled is False


def test_schema_enabled_requires_camera_viewport_enabled(
    tmp_path: Path,
) -> None:
    _reject_canary_table(
        tmp_path,
        camera=DISABLED_CAMERA_TABLE,
        mirror=MIRROR_TABLE.format(
            enabled="true",
            min_latent_shift="2.0",
            pair_probability="1.0",
            pair_weight="1.0",
        ),
    )


def test_schema_min_latent_shift_must_be_exact_2(tmp_path: Path) -> None:
    for value in ("1.9", "2.01", "3.0"):
        _reject_canary_table(
            tmp_path,
            camera=CAMERA_TABLE,
            mirror=MIRROR_TABLE.format(
                enabled="true",
                min_latent_shift=value,
                pair_probability="1.0",
                pair_weight="1.0",
            ),
        )


def test_schema_pair_weight_must_be_exact_1(tmp_path: Path) -> None:
    for value in ("0.5", "2.0"):
        _reject_canary_table(
            tmp_path,
            camera=CAMERA_TABLE,
            mirror=MIRROR_TABLE.format(
                enabled="true",
                min_latent_shift="2.0",
                pair_probability="1.0",
                pair_weight=value,
            ),
        )


def test_schema_pair_probability_bounds(tmp_path: Path) -> None:
    for value in ("-0.1", "1.5"):
        _reject_canary_table(
            tmp_path,
            camera=CAMERA_TABLE,
            mirror=MIRROR_TABLE.format(
                enabled="true",
                min_latent_shift="2.0",
                pair_probability=value,
                pair_weight="1.0",
            ),
        )
    # 0.0 is a valid v1 edge (eligible but never selected).
    _load_canary_table(
        tmp_path,
        camera=CAMERA_TABLE,
        mirror=MIRROR_TABLE.format(
            enabled="true",
            min_latent_shift="2.0",
            pair_probability="0.0",
            pair_weight="1.0",
        ),
    )


def test_schema_mode_literal_is_v1_only(tmp_path: Path) -> None:
    root = _TmpConfigRoot(tmp_path)
    name = root.write(
        "mirror_bad_mode.toml",
        'extends = ["train_g1_cmuon_production.toml"]\n'
        f"{CAMERA_TABLE}\n"
        + SPATIAL_OFF_TABLE
        + "[data.camera_mirror_balance]\n"
        "enabled = true\n"
        'mode = "vertical_mirror_pair_v2"\n'
        "min_latent_shift = 2.0\n"
        "pair_probability = 1.0\n"
        "pair_weight = 1.0\n",
    )
    with pytest.raises(ConfigurationError):
        load_config(Path(name), config_root=root.root, validate_secrets=False)


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


def test_canary_config_changes_only_mirror_and_identity() -> None:
    p25 = _load_config("train_g1_camera_v2_p25.toml")
    canary = _load_config("train_g1_camera_v2_p25_mirror_canary.toml")
    before = tomllib.loads(p25.resolved_toml)
    after = tomllib.loads(canary.resolved_toml)
    diff = _config_diff_keys(before, after)

    # The canary may ONLY touch run/artifact identity, the mirror table, and
    # the governed runtime stop cap (launch-readiness: an invocation stop,
    # never a stage-budget shrink): no LR, batch, optimizer, scheduler,
    # checkpoint cadence, architecture or camera-policy leaf may differ.
    identity_keys = {
        "run.run_id",
        "paths.run_dir",
        "paths.checkpoint_dir",
        "paths.artifact_dir",
        "logging.local_jsonl_path",
        "wandb.retry_jsonl_path",
        "evaluation.output_dir",
    }
    # The mirror section is absent from the baseline resolved TOML
    # (exclude_none), so its whole section path shows up as the single
    # mirror-related diff key; its leaf values are pinned below.
    mirror_section = "data.camera_mirror_balance"
    # The stop cap is a new stage leaf absent from the baseline resolved
    # TOML (exclude_none), so it surfaces as one leaf diff key; every other
    # stage leaf (planned_updates et al.) must remain inherited.
    stop_cap_key = "stage.canary_stop_successful_update"
    assert diff <= identity_keys | {mirror_section, stop_cap_key}, diff
    assert mirror_section in diff, diff
    assert stop_cap_key in diff, diff
    assert after["stage"]["canary_stop_successful_update"] == 118200
    assert after["stage"]["planned_updates"] == before["stage"]["planned_updates"]
    assert "camera_mirror_balance" not in before["data"]

    mirror = after["data"]["camera_mirror_balance"]
    assert set(mirror) == {
        "enabled",
        "mode",
        "min_latent_shift",
        "pair_probability",
        "pair_weight",
    }
    assert mirror == {
        "enabled": True,
        "mode": "vertical_mirror_pair_v1",
        "min_latent_shift": 2.0,
        "pair_probability": 1.0,
        "pair_weight": 1.0,
    }
    # The camera viewport policy itself is untouched.
    assert after["data"]["camera_viewport"] == before["data"]["camera_viewport"]
    assert canary.config.run.run_id == "g1_camera_v2_p25_mirror"


# ---------------------------------------------------------------------------
# Policy object
# ---------------------------------------------------------------------------


def test_policy_from_config_none() -> None:
    assert CameraMirrorPolicy.from_config(None) is None


def test_policy_from_config_canary() -> None:
    config = DataCameraMirrorBalanceConfig(
        enabled=True,
        mode=MIRROR_MODE_V1,
        min_latent_shift=2.0,
        pair_probability=1.0,
        pair_weight=1.0,
    )
    policy = CameraMirrorPolicy.from_config(config)
    assert policy is not None
    assert policy.enabled is True
    assert policy.mode == MIRROR_MODE_V1
    assert policy.min_latent_shift == 2.0
    assert policy.pair_probability == 1.0
    assert policy.pair_weight == 1.0


def test_policy_rejects_invalid_fields() -> None:
    with pytest.raises(CameraViewportError):
        CameraMirrorPolicy(
            enabled=True,
            mode="other_mode",
            min_latent_shift=2.0,
            pair_probability=1.0,
            pair_weight=1.0,
        )
    with pytest.raises(CameraViewportError):
        CameraMirrorPolicy(
            enabled=True,
            mode=MIRROR_MODE_V1,
            min_latent_shift=2,  # int is not the exact float contract
            pair_probability=1.0,
            pair_weight=1.0,
        )
    with pytest.raises(CameraViewportError):
        CameraMirrorPolicy(
            enabled=True,
            mode=MIRROR_MODE_V1,
            min_latent_shift=2.0,
            pair_probability=1.0,
            pair_weight=0.999,
        )
    with pytest.raises(CameraViewportError):
        CameraMirrorPolicy(
            enabled=True,
            mode=MIRROR_MODE_V1,
            min_latent_shift=2.0,
            pair_probability=1.5,
            pair_weight=1.0,
        )


# ---------------------------------------------------------------------------
# Pair geometry
# ---------------------------------------------------------------------------


def test_fixture_yields_applied_vertical_plans() -> None:
    plans = _vertical_plans(24)
    assert len(plans) == 24
    for plan in plans:
        assert plan.applied is True
        assert plan.orientation == "vertical"
        assert plan.fallback_reason == "none"


def test_even_viewport_shifts_are_whole_pixel() -> None:
    # Even viewport + even full canvas: the center shift is an exact integer
    # number of pixels, so the signed shift is integer-valued (no half-pixel
    # center case exists in the v1 production vocabulary).
    for plan in _vertical_plans(64):
        assert plan.viewport % 2 == 0
        assert plan.signed_pixel_center_shift.is_integer()


def test_mirror_geometry_antisymmetry_and_invariants() -> None:
    for plan in _vertical_plans(200):
        assert plan.viewport % 2 == 0
        geometry = plan_mirror_geometry(plan)
        assert geometry is not None, f"even vertical plan {plan} has no mirror"
        mirror_top, mirror_box, mirror_signed, mirror_shift_y, mirror_offset = (
            geometry
        )
        available = plan.full_height - plan.viewport
        assert mirror_top == available - plan.top
        assert mirror_box == (
            0,
            mirror_top,
            plan.viewport,
            mirror_top + plan.viewport,
        )
        # Exact float antisymmetry (the v1 hard gate; integer-valued in the
        # even-viewport vocabulary, verified exact for every canvas).
        assert mirror_signed == -plan.signed_pixel_center_shift
        assert abs(mirror_signed) == abs(plan.signed_pixel_center_shift)
        assert mirror_shift_y == -plan.camera_shift_y
        # The mirror offset is its own top/available (geometry-derived); it
        # equals 1 - offset up to the division rounding, and the side band
        # swaps exactly.
        assert 0.0 <= mirror_offset <= 1.0
        assert mirror_offset == pytest.approx(
            1.0 - plan.normalized_offset, abs=1e-12
        )
        # Side exchange: start <-> end, center -> center.
        assert mirror_side(mirror_offset) == {
            "start": "end",
            "end": "start",
            "center": "center",
        }[mirror_side(plan.normalized_offset)]


def test_mirror_of_mirror_returns_original() -> None:
    for seed in (0, 1, 2, 17, 63):
        plan = _plan(512, 1024, offset_seed=seed)
        assert plan.applied and plan.orientation == "vertical"
        (
            mirror_top,
            mirror_box,
            mirror_signed,
            mirror_shift_y,
            mirror_offset,
        ) = plan_mirror_geometry(plan)
        mirrored = CameraViewportPlan(
            applied=True,
            fallback_reason="none",
            orientation="vertical",
            viewport=plan.viewport,
            full_width=plan.full_width,
            full_height=plan.full_height,
            crop_box=mirror_box,
            left=0,
            top=mirror_top,
            equivalent_zoom=plan.equivalent_zoom,
            retention=plan.retention,
            normalized_offset=mirror_offset,
            signed_pixel_center_shift=mirror_signed,
            absolute_pixel_center_shift=abs(mirror_signed),
            latent_center_shift=abs(mirror_signed) / 16.0,
            camera_shift_x=plan.camera_shift_x,
            camera_shift_y=mirror_shift_y,
        )
        round_trip = plan_mirror_geometry(mirrored)
        assert round_trip is not None
        assert round_trip[0] == plan.top
        assert round_trip[1] == plan.crop_box
        assert round_trip[2] == plan.signed_pixel_center_shift
        assert round_trip[3] == plan.camera_shift_y
        assert round_trip[4] == plan.normalized_offset


def test_eligibility_threshold_and_populations() -> None:
    plans = _vertical_plans(200)
    below = [p for p in plans if p.latent_center_shift < 2.0]
    at_or_above = [p for p in plans if p.latent_center_shift >= 2.0]
    assert below and at_or_above  # both populations are reachable
    for plan in below:
        assert camera_mirror_eligible(plan, min_latent_shift=2.0) is False
    for plan in at_or_above:
        assert camera_mirror_eligible(plan, min_latent_shift=2.0) is True


def test_horizontal_and_fallback_are_ineligible() -> None:
    horizontal = _plan(1024, 512, offset_seed=3)
    assert horizontal.applied is True
    assert horizontal.orientation == "horizontal"
    assert (
        camera_mirror_eligible(horizontal, min_latent_shift=2.0) is False
    )

    # policy_seed 0 draws 0.844..., which is >= probability 0.5, so the
    # camera plan is not selected (applied=False). An enabled policy requires
    # probability > 0, so 0.0 is not a valid "off" knob here.
    not_selected = _plan(
        512, 1024, probability=0.5, policy_seed=0, offset_seed=3
    )
    assert not_selected.applied is False
    assert not_selected.fallback_reason == "not_selected"
    assert (
        camera_mirror_eligible(not_selected, min_latent_shift=2.0) is False
    )

    near_square = _plan(600, 600, offset_seed=3)
    assert near_square.applied is False
    assert (
        camera_mirror_eligible(near_square, min_latent_shift=2.0) is False
    )


def test_odd_viewport_is_ineligible_and_has_no_geometry() -> None:
    # Synthetic odd-viewport vertical plan (not producible by the stage-512
    # bucket vocabulary, exercised directly at the guard level).
    viewport = 511
    full_height = 765
    plan = CameraViewportPlan(
        applied=True,
        fallback_reason="none",
        orientation="vertical",
        viewport=viewport,
        full_width=viewport,
        full_height=full_height,
        crop_box=(0, 0, viewport, viewport),
        left=0,
        top=0,
        equivalent_zoom=full_height / viewport,
        retention=0.9,
        normalized_offset=0.0,
        signed_pixel_center_shift=-127.0,
        absolute_pixel_center_shift=127.0,
        latent_center_shift=127.0 / 16.0,
        camera_shift_x=0.0,
        camera_shift_y=1.0 - full_height / viewport,
    )
    # latent 7.94 >= 2.0: only the odd viewport keeps it ineligible.
    assert camera_mirror_eligible(plan, min_latent_shift=2.0) is False
    assert plan_mirror_geometry(plan) is None


# ---------------------------------------------------------------------------
# Severity and side labels
# ---------------------------------------------------------------------------


def test_severity_bands() -> None:
    assert mirror_severity_band(0.0) == "lt2"
    assert mirror_severity_band(1.999) == "lt2"
    assert mirror_severity_band(2.0) == "2to4"
    assert mirror_severity_band(3.999) == "2to4"
    assert mirror_severity_band(4.0) == "ge4"
    assert mirror_severity_band(16.0) == "ge4"
    for bad in (-0.1, float("nan"), float("inf")):
        with pytest.raises(CameraViewportError):
            mirror_severity_band(bad)
    with pytest.raises(CameraViewportError):
        mirror_severity_band(2)  # int is not the float contract


def test_side_bands() -> None:
    assert mirror_side(0.0) == "start"
    assert mirror_side(1.0 / 3.0 - 1e-9) == "start"
    assert mirror_side(1.0 / 3.0) == "center"
    assert mirror_side(0.5) == "center"
    assert mirror_side(2.0 / 3.0) == "end"
    assert mirror_side(1.0) == "end"
    for bad in (-1e-9, 1.0 + 1e-9):
        with pytest.raises(CameraViewportError):
            mirror_side(bad)


# ---------------------------------------------------------------------------
# Fixed-key counters
# ---------------------------------------------------------------------------


def test_zero_counts_table() -> None:
    counts = zero_camera_mirror_counts(4)
    assert counts.logical_samples == 4
    assert counts.physical_views == 4
    assert counts.mirror_extra_views == 0
    assert counts.mirror_applied == 0
    assert counts.vertical_applied == 0
    assert counts.severity_lt2 == 0
    assert counts.severity_2to4 == 0
    assert counts.severity_ge4 == 0


def _valid_counts(**overrides: int) -> CameraMirrorCounts:
    base: dict[str, int] = {
        "logical_samples": 10,
        "vertical_applied": 4,
        "mirror_eligible": 3,
        "mirror_selected": 2,
        "mirror_applied": 2,
        "severity_lt2": 1,
        "severity_2to4": 2,
        "severity_ge4": 1,
        "original_start": 1,
        "original_center": 1,
        "original_end": 0,
        "mirror_start": 1,
        "mirror_center": 1,
        "mirror_end": 0,
        "mirror_extra_views": 2,
        "physical_views": 12,
    }
    base.update(overrides)
    return CameraMirrorCounts(**base)  # type: ignore[arg-type]


def test_counts_accept_valid_table() -> None:
    _valid_counts()


def test_counts_reject_violations() -> None:
    for overrides in (
        {"mirror_applied": 3},  # applied > selected
        {"mirror_selected": 4},  # selected > eligible
        {"mirror_eligible": 5},  # eligible > vertical
        {"vertical_applied": 5},  # vertical > logical
        {"severity_lt2": 2},  # severity sum != vertical
        {"original_start": 2},  # side sum != applied
        {"mirror_end": 1},  # mirror side sum != applied
        {"mirror_extra_views": 1},  # extra != applied
        {"physical_views": 13},  # physical != logical + extra
        {"mirror_applied": -1},
    ):
        with pytest.raises(CameraViewportError):
            _valid_counts(**overrides)
    with pytest.raises(CameraViewportError):
        _valid_counts(logical_samples=True)  # bool is not the int contract


def _fake_sample(
    *,
    camera_applied: bool = False,
    orientation: str = "none",
    latent: float = 0.0,
    offset: float = 0.0,
    eligible: bool = False,
    selected: bool = False,
    mirror: Any = None,
) -> SimpleNamespace:
    audit = SimpleNamespace(
        camera_applied=camera_applied,
        camera_orientation=orientation,
        camera_latent_center_shift=latent,
        camera_normalized_offset=offset,
    )
    return SimpleNamespace(
        audit=audit,
        mirror=mirror,
        mirror_eligible=eligible,
        mirror_selected=selected,
    )


def test_aggregate_mixed_batch_conservation() -> None:
    samples = (
        _fake_sample(),  # ordinary, no camera
        _fake_sample(
            camera_applied=True,
            orientation="vertical",
            latent=3.0,
            offset=0.1,
            eligible=True,
            selected=True,
            mirror=SimpleNamespace(normalized_offset=0.9),
        ),
        _fake_sample(
            camera_applied=True, orientation="vertical", latent=1.0
        ),  # vertical but lt2: not eligible
        _fake_sample(
            camera_applied=True,
            orientation="horizontal",
            latent=5.0,
        ),  # horizontal: never eligible
        _fake_sample(
            camera_applied=True,
            orientation="vertical",
            latent=8.0,
            offset=0.8,
            eligible=True,  # eligible but not selected (pair_probability 0)
        ),
    )
    counts = aggregate_camera_mirror(samples)
    assert counts.logical_samples == 5
    assert counts.vertical_applied == 3
    assert counts.mirror_eligible == 2
    assert counts.mirror_selected == 1
    assert counts.mirror_applied == 1
    assert counts.severity_lt2 == 1
    assert counts.severity_2to4 == 1
    assert counts.severity_ge4 == 1
    assert counts.original_start == 1
    assert counts.mirror_end == 1
    assert counts.mirror_extra_views == 1
    assert counts.physical_views == 6


def test_aggregate_all_zero_returns_zero_table() -> None:
    counts = aggregate_camera_mirror(
        (_fake_sample(), _fake_sample())
    )
    assert counts == zero_camera_mirror_counts(2)


# ---------------------------------------------------------------------------
# Pipeline worker-level behavior
# ---------------------------------------------------------------------------


class _Tokenizer:
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        del add_special_tokens
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


def _pipeline(
    *,
    camera_policy: CameraViewportPolicy | None,
    mirror_policy: CameraMirrorPolicy | None = None,
) -> WebDatasetPipeline:
    pipeline = object.__new__(WebDatasetPipeline)
    pipeline.metadata_adapter = lambda metadata: metadata
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
    pipeline.rejection_observer = lambda reason: None
    pipeline.spatial_policy = None
    pipeline.transparent_policy = None
    pipeline.transparent_telemetry = TransparentWhiteTelemetry()  # pyright: ignore[reportAttributeAccessIssue]
    pipeline.camera_policy = camera_policy
    pipeline.mirror_policy = mirror_policy
    pipeline._camera_stage_edge = camera_stage_edge(BUCKETS)  # pyright: ignore[reportAttributeAccessIssue, reportPrivateUsage]
    return cast(Any, pipeline)


def _process_sample(
    pipeline: WebDatasetPipeline,
    image_bytes: bytes,
    *,
    sample_id: int = 1,
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


def test_disabled_paths_are_bit_identical() -> None:
    image_bytes = _gradient_png(512, 1024)
    reference = _pipeline(camera_policy=_camera_policy())
    disabled = _pipeline(
        camera_policy=_camera_policy(),
        mirror_policy=_mirror_policy(enabled=False, pair_probability=0.0),
    )
    for sample_id in range(1, 9):
        base_result = _process_sample(
            reference, image_bytes, sample_id=sample_id
        )
        off_result = _process_sample(
            disabled, image_bytes, sample_id=sample_id
        )
        assert torch.equal(base_result.image, off_result.image)
        assert base_result.audit == off_result.audit
        assert base_result.caption == off_result.caption
        for result in (base_result, off_result):
            assert result.mirror is None
            assert result.mirror_eligible is False
            assert result.mirror_selected is False


def test_horizontal_camera_never_mirrors() -> None:
    image_bytes = _gradient_png(1024, 512)
    pipeline = _pipeline(
        camera_policy=_camera_policy(),
        mirror_policy=_mirror_policy(pair_probability=1.0),
    )
    saw_horizontal = False
    for sample_id in range(1, 33):
        result = _process_sample(pipeline, image_bytes, sample_id=sample_id)
        assert result.mirror is None
        assert result.mirror_eligible is False
        assert result.mirror_selected is False
        if result.audit.camera_applied:
            assert result.audit.camera_orientation == "horizontal"
            saw_horizontal = True
    assert saw_horizontal


def test_pair_selection_payload_and_antisymmetry() -> None:
    image_bytes = _gradient_png(512, 1024)
    with_mirror = _pipeline(
        camera_policy=_camera_policy(),
        mirror_policy=_mirror_policy(pair_probability=1.0),
    )
    without = _pipeline(camera_policy=_camera_policy())

    checked_pairs = 0
    for sample_id in range(1, 65):
        result = _process_sample(
            with_mirror, image_bytes, sample_id=sample_id
        )
        audit = result.audit
        if not audit.camera_applied or audit.camera_orientation != "vertical":
            continue
        expected_eligible = audit.camera_latent_center_shift >= 2.0
        assert result.mirror_eligible is expected_eligible
        if not expected_eligible:
            assert result.mirror_selected is False
            assert result.mirror is None
            continue
        # pair_probability = 1.0 selects every eligible sample.
        assert result.mirror_selected is True
        payload = result.mirror
        assert payload is not None
        assert result.mirror is payload

        # The audit and the original image are untouched by the mirror.
        baseline = _process_sample(
            without, image_bytes, sample_id=sample_id
        )
        assert baseline.audit == audit
        assert torch.equal(baseline.image, result.image)
        assert baseline.caption == result.caption

        # Payload geometry matches the independent plan re-derivation.
        left, top, right, bottom = audit.crop_box
        assert left == 0  # vertical crops anchor at the canvas left edge
        viewport = right - left
        assert viewport == bottom - top
        assert viewport % 2 == 0
        plan = CameraViewportPlan(
            applied=True,
            fallback_reason="none",
            orientation="vertical",
            viewport=viewport,
            full_width=audit.camera_full_width,
            full_height=audit.camera_full_height,
            crop_box=audit.crop_box,
            left=left,
            top=top,
            equivalent_zoom=audit.camera_equivalent_zoom,
            retention=audit.camera_final_retention,
            normalized_offset=audit.camera_normalized_offset,
            signed_pixel_center_shift=audit.camera_pixel_center_shift,
            absolute_pixel_center_shift=abs(audit.camera_pixel_center_shift),
            latent_center_shift=audit.camera_latent_center_shift,
            camera_shift_x=audit.camera_shift_x,
            camera_shift_y=audit.camera_shift_y,
        )
        (
            _mirror_top,
            mirror_box,
            mirror_signed,
            mirror_shift_y,
            mirror_offset,
        ) = plan_mirror_geometry(plan)
        assert payload.crop_box == mirror_box
        assert mirror_box[1] == audit.camera_full_height - viewport - top
        # Exact signed-shift antisymmetry against the audit record.
        assert payload.signed_pixel_center_shift == -audit.camera_pixel_center_shift
        assert payload.signed_pixel_center_shift == mirror_signed
        assert payload.camera_shift_y == -audit.camera_shift_y
        assert payload.camera_shift_y == mirror_shift_y
        assert payload.normalized_offset == mirror_offset
        assert 0.0 <= payload.normalized_offset <= 1.0

        # The mirror image is the exact crop of the SAME normalized canvas.
        work = Image.open(io.BytesIO(image_bytes))
        work.load()
        canvas = normalize_image(work).resize(
            (audit.camera_full_width, audit.camera_full_height),
            resample=Image.Resampling.LANCZOS,
        )
        expected = canvas.crop(payload.crop_box)
        expected_tensor = torch.from_numpy(
            np.asarray(expected).copy()
        ).permute(2, 0, 1).contiguous()
        assert torch.equal(payload.mirror_image, expected_tensor)
        assert payload.mirror_image.shape == (3, viewport, viewport)
        assert payload.mirror_image.dtype == torch.uint8

        # A vertical gradient makes the two crops distinguishable unless the
        # original is exactly centered (top == available - top).
        if top != payload.crop_box[1]:
            assert not torch.equal(payload.mirror_image, result.image)
        checked_pairs += 1
    assert checked_pairs >= 1, "no mirror pair was exercised"


def test_pair_probability_zero_selects_nothing() -> None:
    image_bytes = _gradient_png(512, 1024)
    pipeline = _pipeline(
        camera_policy=_camera_policy(),
        mirror_policy=_mirror_policy(pair_probability=0.0),
    )
    saw_eligible = False
    for sample_id in range(1, 65):
        result = _process_sample(pipeline, image_bytes, sample_id=sample_id)
        audit = result.audit
        if not audit.camera_applied:
            continue
        if audit.camera_orientation == "vertical" and (
            audit.camera_latent_center_shift >= 2.0
        ):
            saw_eligible = True
            assert result.mirror_eligible is True
            assert result.mirror_selected is False
            assert result.mirror is None
    assert saw_eligible


def test_mirror_rng_streams_are_isolated() -> None:
    # The mirror selection draw lives in its own domain: two pipelines that
    # differ only in the mirror policy produce identical camera plans and
    # identical camera-seed domains.
    image_bytes = _gradient_png(512, 1024)
    with_mirror = _pipeline(
        camera_policy=_camera_policy(),
        mirror_policy=_mirror_policy(pair_probability=0.5),
    )
    without = _pipeline(camera_policy=_camera_policy())
    for sample_id in range(1, 17):
        a = _process_sample(with_mirror, image_bytes, sample_id=sample_id)
        b = _process_sample(without, image_bytes, sample_id=sample_id)
        assert a.audit == b.audit
        assert a.rng.camera_policy_seed == b.rng.camera_policy_seed
        assert a.rng.camera_offset_seed == b.rng.camera_offset_seed
        assert a.rng.mirror_policy_seed > 0


# ---------------------------------------------------------------------------
# Collate level
# ---------------------------------------------------------------------------


def _collate_sample(
    sample_id: int,
    *,
    width: int = 8,
    mirror: Any = None,
    eligible: bool = False,
    selected: bool = False,
    camera_vertical: bool = False,
) -> PipelineSample:
    input_ids = (10, 11, 12) if sample_id % 2 else (20, 21)
    dropout_hits = empty_caption_dropout_hits()
    caption = SerializedCaption(
        plan=CaptionPlan(
            tags=(),
            condition=None,
            nl_text=None,
            selected_nl=None,
            all_condition_dropped=False,
            dropout_hits=dropout_hits,
        ),
        text="test",
        input_ids=input_ids,
        attention_mask=(True,) * len(input_ids),
        main_token_indices=tuple(range(len(input_ids))),
        main_mask=(True,) * len(input_ids),
        condition_token_indices=(),
        condition_mask=(),
        use_null_condition=True,
        condition_source=None,
        condition_role=None,
        all_condition_dropped=False,
        dropout_hits=dropout_hits,
        selected_nl=None,
        body="",
        condition_text="",
        condition_tokens=5,
        condition_bucket=64 - 34,
        dense_length=64,
        truncated=False,
    )
    audit = (
        ImageAudit(
            8,
            8,
            width,
            8,
            (0, 0, width, 8),
            1.0,
            crop_policy="camera_viewport",
            camera_policy="hdm_shifted_square_v2",
            camera_selected=True,
            camera_applied=True,
            camera_orientation="vertical",
            camera_equivalent_zoom=1.25,
            camera_final_retention=1.0,
            camera_normalized_offset=0.1,
            camera_pixel_center_shift=-32.0,
            camera_latent_center_shift=2.0,
            camera_shift_y=-0.25,
            camera_full_width=8,
            camera_full_height=16,
        )
        if camera_vertical
        else ImageAudit(8, 8, width, 8, (0, 0, width, 8), 1.0)
    )
    return PipelineSample(
        sample_id=sample_id,
        source_shard="data/test.tar",
        image=torch.full((3, 8, width), sample_id, dtype=torch.uint8),
        target_height=8,
        target_width=width,
        caption=caption,
        audit=audit,
        rng=RngIdentity(1, "S0", 0, sample_id, sample_id + 1, sample_id + 2),
        padding_token_id=248044,
        mirror=mirror,
        mirror_eligible=eligible,
        mirror_selected=selected,
    )


def _payload(height: int, width: int, *, box: tuple[int, int, int, int]):
    return CameraMirrorPayload(
        mirror_image=torch.full((3, height, width), 3, dtype=torch.uint8),
        crop_box=box,
        signed_pixel_center_shift=-32.0,
        camera_shift_y=-0.25,
        normalized_offset=0.9,
    )


def test_collate_row_alignment_and_counts() -> None:
    payload = _payload(8, 8, box=(0, 0, 8, 8))
    samples = (
        _collate_sample(1),
        _collate_sample(
            2,
            mirror=payload,
            eligible=True,
            selected=True,
            camera_vertical=True,
        ),
    )
    batch = collate_samples(samples)
    assert batch.mirror == (None, payload)
    assert batch.camera_mirror is not None
    assert batch.camera_mirror.logical_samples == 2
    assert batch.camera_mirror.vertical_applied == 1
    assert batch.camera_mirror.mirror_eligible == 1
    assert batch.camera_mirror.mirror_selected == 1
    assert batch.camera_mirror.mirror_applied == 1
    assert batch.camera_mirror.severity_2to4 == 1
    assert batch.camera_mirror.original_start == 1
    assert batch.camera_mirror.mirror_end == 1
    assert batch.camera_mirror.mirror_extra_views == 1
    assert batch.camera_mirror.physical_views == 3


def test_collate_rejects_invalid_mirror_payloads() -> None:
    good = _payload(8, 8, box=(0, 0, 8, 8))
    base = (
        _collate_sample(1),
        _collate_sample(
            2,
            mirror=good,
            eligible=True,
            selected=True,
            camera_vertical=True,
        ),
    )
    collate_samples(base)  # sanity: the good batch collates

    wrong_shape = _collate_sample(
        2,
        mirror=_payload(4, 4, box=(0, 0, 4, 4)),
        eligible=True,
        selected=True,
        camera_vertical=True,
    )
    with pytest.raises(CollateError):
        collate_samples((_collate_sample(1), wrong_shape))

    bad_box = _collate_sample(
        2,
        mirror=_payload(8, 8, box=(1, 0, 9, 8)),
        eligible=True,
        selected=True,
        camera_vertical=True,
    )
    with pytest.raises(CollateError):
        collate_samples((_collate_sample(1), bad_box))

    not_eligible = _collate_sample(
        2, mirror=good, eligible=False, selected=True, camera_vertical=True
    )
    with pytest.raises(CollateError):
        collate_samples((_collate_sample(1), not_eligible))

    not_selected = _collate_sample(
        2, mirror=good, eligible=True, selected=False, camera_vertical=True
    )
    with pytest.raises(CollateError):
        collate_samples((_collate_sample(1), not_selected))


def test_legacy_batches_keep_pre_mirror_contract() -> None:
    batch = collate_samples((_collate_sample(1), _collate_sample(2)))
    # Collate always fills the row-aligned mirror tuple; a batch without any
    # mirror sample is all-None (the runtime's no-pair path), with the zero
    # counts table and no extra physical views.
    assert batch.mirror == (None, None)
    assert all(payload is None for payload in batch.mirror)
    assert batch.camera_mirror == zero_camera_mirror_counts(2)


# ---------------------------------------------------------------------------
# Degrade semantics (canary readiness fix C): a mirror-branch failure must
# preserve the eligibility/selection decision history so that
# (selected - applied) is the deterministic degraded count, and the
# original sample always survives.
# ---------------------------------------------------------------------------


def test_pipeline_degrade_preserves_decision_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sakuramoon.data.pipeline as pipeline_module

    def _boom(_plan):  # type: ignore[no-redef]
        raise ValueError("synthetic mirror construction failure")

    monkeypatch.setattr(pipeline_module, "plan_mirror_geometry", _boom)
    image_bytes = _gradient_png(512, 1024)
    pipeline = _pipeline(
        camera_policy=_camera_policy(),
        mirror_policy=_mirror_policy(pair_probability=1.0),
    )
    saw_degrade = False
    for sample_id in range(1, 129):
        result = _process_sample(pipeline, image_bytes, sample_id=sample_id)
        if not (result.mirror_eligible and result.mirror_selected):
            continue
        saw_degrade = True
        assert result.mirror is None
        # The decision history is preserved, not reset.
        assert result.mirror_eligible is True
        assert result.mirror_selected is True
        # The original sample continues: camera still applied, original
        # crop emitted.
        assert result.audit.camera_applied
        assert result.image is not None
    assert saw_degrade, "no eligible+selected sample was exercised"


def test_collate_selected_degraded_keeps_decision_history() -> None:
    # eligible=1, selected=1, payload=None must produce eligible 1,
    # selected 1, applied 0 -- NOT 0/0/0.
    samples = (
        _collate_sample(1),
        _collate_sample(
            2,
            mirror=None,
            eligible=True,
            selected=True,
            camera_vertical=True,
        ),
    )
    batch = collate_samples(samples)
    assert batch.mirror == (None, None)
    counts = batch.camera_mirror
    assert counts is not None
    assert counts.mirror_eligible == 1
    assert counts.mirror_selected == 1
    assert counts.mirror_applied == 0
    assert counts.mirror_extra_views == 0
    assert counts.physical_views == 2
