# pyright: reportPrivateUsage=false

from __future__ import annotations

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


def _items() -> tuple[sampling.TrainingSampleItem, ...]:
    framing = FramingContract(34, 5, _Tokenizer.pad_token_id)
    return sampling._build_variant_items(
        sampling._PromptPair(
            _prompt("A", "artist_a"),
            _prompt("B", "artist_b"),
        ),
        tokenizer=_Tokenizer(),
        framing=framing,
        resolution=256,
    )


def test_tiered_zoom_protocol_preserves_condition_layout_and_cfg_shape() -> None:
    assert sampling._VARIANT_COUNT == 12
    assert sampling._CFG_BRANCH_COUNT == 24
    assert sampling._DIAGNOSTIC_ITEM_INDICES == (0, 2, 4)
    assert sampling._GEOMETRY_PROTOCOL == "tiered-zoom-v1"
    assert sampling._VARIANT_NAMES == (
        "A-base",
        "B-base",
        "A-with-B",
        "B-with-A",
        "A-null",
        "B-null",
        "A-with-BA",
        "B-with-BA",
        "A-zoom-mild",
        "A-zoom-strong",
        "A-shift-zoom-mild",
        "A-shift-zoom-strong",
    )


def test_tiered_zoom_geometry_matches_mild_and_legacy_strong_coordinates() -> None:
    items = {item.variant: item for item in _items()}
    expected = {
        "A-zoom-mild": (282, (13, 13, 269, 269)),
        "A-zoom-strong": (384, (64, 64, 320, 320)),
        "A-shift-zoom-mild": (282, (19, 19, 275, 275)),
        "A-shift-zoom-strong": (384, (96, 96, 352, 352)),
    }
    for variant, (virtual, crop_box) in expected.items():
        item = items[variant]
        assert item.virtual_canvas_size == (virtual, virtual)
        assert item.crop_box == crop_box
        assert item.zoom == virtual / 256
        left, top, right, bottom = crop_box
        assert 0 <= left < right <= virtual
        assert 0 <= top < bottom <= virtual
        coordinates = full_canvas_crop_coordinates(
            16,
            16,
            full_height=virtual,
            full_width=virtual,
            crop_box=crop_box,
            device=torch.device("cpu"),
        )
        assert coordinates.shape == (256, 2)

    assert (
        items["A-zoom-mild"].virtual_canvas_size
        == items["A-shift-zoom-mild"].virtual_canvas_size
    )
    assert (
        items["A-zoom-strong"].virtual_canvas_size
        == items["A-shift-zoom-strong"].virtual_canvas_size
    )


def test_spatial_variants_change_only_geometry_from_a_base() -> None:
    items = {item.variant: item for item in _items()}
    base = items["A-base"]
    for variant in (
        "A-zoom-mild",
        "A-zoom-strong",
        "A-shift-zoom-mild",
        "A-shift-zoom-strong",
    ):
        item = items[variant]
        assert item.main_source == base.main_source
        assert item.condition_sources == base.condition_sources
        assert item.sample_id == base.sample_id
        assert item.caption == base.caption
        assert item.plan == base.plan
        assert (item.height, item.width) == (base.height, base.width)
        assert item.crop_box != base.crop_box


def test_all_variants_share_the_same_initial_noise() -> None:
    noise = sampling._shared_initial_noise(
        height=256,
        width=256,
        shared_seed=12345,
        device=torch.device("cpu"),
    )

    assert noise.shape == (12, 128, 16, 16)
    torch.testing.assert_close(noise, noise[0:1].expand_as(noise))
