# pyright: reportPrivateUsage=false

from __future__ import annotations

import math
from pathlib import Path

import torch

from sakuramoon.conditioning.rope import full_canvas_crop_coordinates
from sakuramoon.data.caption import (
    CaptionPlan,
    CaptionTag,
    ConditionRequest,
    Tag,
    empty_caption_dropout_hits,
)
from sakuramoon.data.serialize import (
    MAIN_SUFFIX,
    SYSTEM_PREFIX,
    FramingContract,
    serialize_caption,
)
from sakuramoon.train import sampling

_CANONICAL_VARIANTS = (
    "A-base",
    "B-base",
    "A-with-B",
    "B-with-A",
    "A-null",
    "B-null",
    "A-with-BA",
    "B-with-BA",
)
_CAMERA_VARIANTS = (
    "A-camera-h-center",
    "A-camera-h-end",
    "A-camera-v-center",
    "A-camera-v-end",
)


class _Tokenizer:
    pad_token_id = 248044

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        if text == SYSTEM_PREFIX:
            return list(range(100, 134))
        if text == MAIN_SUFFIX:
            return list(range(200, 205))
        return [1000 + index for index, _character in enumerate(text)]


def _prompt(
    label: str,
    condition_tag: str,
) -> sampling._PostDropoutPrompt:
    plan = CaptionPlan(
        tags=(
            CaptionTag("general", Tag(f"subject_{label}", f"subject_{label}")),
            CaptionTag("year", Tag("year 2026", "year 2026")),
        ),
        condition=ConditionRequest(
            source="artist_text",
            role="style",
            tags=(Tag(condition_tag, condition_tag),),
        ),
        nl_text=f"lighting {label}",
        selected_nl="short_vibes",
        all_condition_dropped=False,
        dropout_hits=empty_caption_dropout_hits(),
    )
    framing = FramingContract(34, 5, _Tokenizer.pad_token_id)
    return sampling._PostDropoutPrompt(
        sample_id=f"sample-{label}",
        caption=serialize_caption(plan, _Tokenizer(), framing),
        plan=plan,
        observed_height=256,
        observed_width=256,
    )


def _items(resolution: int = 256) -> tuple[sampling.TrainingSampleItem, ...]:
    framing = FramingContract(34, 5, _Tokenizer.pad_token_id)
    return sampling._build_variant_items(
        sampling._PromptPair(
            _prompt("A", "artist_a"),
            _prompt("B", "artist_b"),
        ),
        tokenizer=_Tokenizer(),
        framing=framing,
        resolution=resolution,
    )


def test_camera_crop_protocol_preserves_condition_layout_and_cfg_shape() -> None:
    assert sampling._VARIANT_COUNT == 12
    assert sampling._CFG_BRANCH_COUNT == 24
    assert sampling._DIAGNOSTIC_ITEM_INDICES == (0, 2, 4)
    assert sampling._GEOMETRY_PROTOCOL == "camera-crop-v1"
    assert sampling._VARIANT_NAMES == _CANONICAL_VARIANTS + _CAMERA_VARIANTS
    assert sampling._VARIANT_COUNT == len(sampling._VARIANT_NAMES)


def test_first_eight_variants_keep_identity_geometry() -> None:
    items = {item.variant: item for item in _items(256)}
    for variant in _CANONICAL_VARIANTS:
        item = items[variant]
        assert item.zoom == 1.0
        assert item.virtual_canvas_size == (256, 256)
        assert item.crop_box == (0, 0, 256, 256)
        assert item.coordinate_type == "canonical_full_canvas"
        assert (item.height, item.width) == (256, 256)


def _expected_camera_geometry(
    resolution: int,
) -> dict[str, tuple[tuple[int, int], tuple[int, int, int, int]]]:
    half = resolution // 2
    return {
        "A-camera-h-center": (
            (resolution, resolution * 2),
            (half, 0, half + resolution, resolution),
        ),
        "A-camera-h-end": (
            (resolution, resolution * 2),
            (resolution, 0, resolution * 2, resolution),
        ),
        "A-camera-v-center": (
            (resolution * 2, resolution),
            (0, half, resolution, half + resolution),
        ),
        "A-camera-v-end": (
            (resolution * 2, resolution),
            (0, resolution, resolution, resolution * 2),
        ),
    }


def _assert_camera_items(resolution: int) -> None:
    items = {item.variant: item for item in _items(resolution)}
    tokens = resolution // 16
    for variant, (canvas, crop_box) in _expected_camera_geometry(resolution).items():
        item = items[variant]
        assert item.virtual_canvas_size == canvas
        assert item.crop_box == crop_box
        assert item.coordinate_type == "camera_full_canvas_crop"
        assert (item.height, item.width) == (resolution, resolution)
        canvas_area = float(canvas[0] * canvas[1])
        left, top, right, bottom = crop_box
        crop_area = float((right - left) * (bottom - top))
        # equivalent_zoom = sqrt(canvas/crop) and area retention = 0.5,
        # derived from the same canvas/crop pair the coordinate map uses.
        assert item.zoom == math.sqrt(canvas_area / crop_area)
        assert item.zoom == math.sqrt(2.0)
        assert crop_area / canvas_area == 0.5
        left, top, right, bottom = crop_box
        assert 0 <= left < right <= canvas[1]
        assert 0 <= top < bottom <= canvas[0]
        coordinates = full_canvas_crop_coordinates(
            tokens,
            tokens,
            full_height=canvas[0],
            full_width=canvas[1],
            crop_box=crop_box,
            device=torch.device("cpu"),
        )
        assert coordinates.shape == (tokens * tokens, 2)
        # The item's stored geometry must reproduce its exact coordinate map.
        maps = sampling._coordinate_maps(
            (item,), device=torch.device("cpu")
        )
        torch.testing.assert_close(maps[0], coordinates)


def test_camera_variants_match_geometric_definition_at_256() -> None:
    _assert_camera_items(256)


def test_camera_variants_match_geometric_definition_at_512() -> None:
    _assert_camera_items(512)


def test_camera_variants_change_only_geometry_from_a_base() -> None:
    items = {item.variant: item for item in _items(256)}
    base = items["A-base"]
    for variant in _CAMERA_VARIANTS:
        item = items[variant]
        assert item.main_source == base.main_source
        assert item.condition_sources == base.condition_sources
        assert item.sample_id == base.sample_id
        assert item.caption == base.caption
        assert item.plan == base.plan
        assert (item.height, item.width) == (base.height, base.width)
        assert item.crop_box != base.crop_box


def test_cfg_branches_share_the_conditional_coordinate_maps() -> None:
    items = _items(256)
    maps = sampling._coordinate_maps(items, device=torch.device("cpu"))
    assert len(maps) == sampling._VARIANT_COUNT
    # The sampler's CFG branch layout duplicates the identical conditional
    # map tuple, so the conditional and unconditional halves always sit in
    # the same image coordinate frame.
    branch_coordinates = maps + maps
    for index in range(sampling._VARIANT_COUNT):
        torch.testing.assert_close(
            branch_coordinates[index + sampling._VARIANT_COUNT],
            branch_coordinates[index],
        )
    source = Path(sampling.__file__).read_text(encoding="utf-8")
    assert "conditional_coordinates + conditional_coordinates" in source


def test_all_variants_share_the_same_initial_noise() -> None:
    noise = sampling._shared_initial_noise(
        height=256,
        width=256,
        shared_seed=12345,
        device=torch.device("cpu"),
    )

    assert noise.shape == (12, 128, 16, 16)
    torch.testing.assert_close(noise, noise[0:1].expand_as(noise))
