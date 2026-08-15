from __future__ import annotations

import pytest

from sakuramoon.data.caption import (
    CaptionPlan,
    CaptionTag,
    StyleCondition,
    Tag,
    empty_caption_dropout_hits,
)
from sakuramoon.data.serialize import (
    MAIN_SUFFIX,
    SYSTEM_PREFIX,
    FramingContract,
    serialize_caption,
)


class _Tokenizer:
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        if text == SYSTEM_PREFIX:
            return list(range(100, 134))
        if text == MAIN_SUFFIX:
            return list(range(200, 205))
        return [1000 + index for index, _character in enumerate(text)]


def _plan(kind: str = "artist") -> CaptionPlan:
    return CaptionPlan(
        tags=(
            CaptionTag("general", Tag("long_hair", "long_hair")),
            CaptionTag("year", Tag("year 2026", "year 2026")),
        ),
        style_condition=StyleCondition(
            kind=kind,  # pyright: ignore[reportArgumentType]
            tags=(Tag(f"{kind}_name", f"{kind}_name"),),
        ),
        nl_text="soft lighting",
        selected_nl="short_vibes",
        all_condition_dropped=False,
        dropout_hits=empty_caption_dropout_hits(),
    )


@pytest.mark.parametrize("kind", ["artist", "character"])
def test_serialization_normalizes_only_tokenizer_facing_underscores(kind: str) -> None:
    plan = _plan(kind)

    caption = serialize_caption(plan, _Tokenizer(), FramingContract(34, 5, 248044))

    assert caption.body == "long hair, year 2026\n\nsoft lighting"
    assert caption.condition_text == f"{kind} name"
    assert caption.text == (
        SYSTEM_PREFIX
        + "long hair, year 2026\n\nsoft lighting"
        + MAIN_SUFFIX
        + f"{kind} name"
    )
    assert caption.plan.tags[0].tag.text == "long_hair"
    assert caption.plan.tags[0].tag.canonical == "long_hair"
    assert caption.plan.style_condition is not None
    assert caption.plan.style_condition.tags[0].text == f"{kind}_name"
    assert caption.condition_token_indices
    assert caption.use_null_condition is False
    assert caption.condition_kind == kind


def test_serialization_truncates_only_complete_globally_ordered_tags() -> None:
    tags = tuple(
        CaptionTag("general", Tag(f"tag_{index:03d}", f"tag_{index:03d}"))
        for index in range(80)
    )
    requested = CaptionPlan(
        tags=tags,
        style_condition=None,
        nl_text=None,
        selected_nl=None,
        all_condition_dropped=False,
        dropout_hits=empty_caption_dropout_hits(),
    )

    caption = serialize_caption(requested, _Tokenizer(), FramingContract(34, 5, 248044))

    assert caption.truncated is True
    assert caption.plan.tags
    assert caption.plan.tags == tags[: len(caption.plan.tags)]
    assert caption.body == ", ".join(
        item.tag.text.replace("_", " ") for item in caption.plan.tags
    )
    assert caption.condition_token_indices == ()
    assert caption.use_null_condition is True
    assert caption.condition_kind is None
