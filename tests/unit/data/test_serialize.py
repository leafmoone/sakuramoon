from __future__ import annotations

import pytest

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


class _Tokenizer:
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        if text == SYSTEM_PREFIX:
            return list(range(100, 134))
        if text == MAIN_SUFFIX:
            return list(range(200, 205))
        return [1000 + index for index, _character in enumerate(text)]


def _plan(source: str = "artist_text") -> CaptionPlan:
    is_artist = source == "artist_text"
    role = "style" if is_artist else "identity"
    tag_name = "artist_name" if is_artist else "character_name"
    return CaptionPlan(
        tags=(
            CaptionTag("general", Tag("long_hair", "long_hair")),
            CaptionTag("year", Tag("year 2026", "year 2026")),
        ),
        condition=ConditionRequest(
            source=source,  # pyright: ignore[reportArgumentType]
            role=role,  # pyright: ignore[reportArgumentType]
            tags=(Tag(tag_name, tag_name),),
        ),
        nl_text="soft lighting",
        selected_nl="short_vibes",
        all_condition_dropped=False,
        dropout_hits=empty_caption_dropout_hits(),
    )


@pytest.mark.parametrize(
    ("source", "role", "tag_name", "expected_condition"),
    [
        ("artist_text", "style", "artist_name", "style reference: artist name"),
        (
            "character_text",
            "identity",
            "character_name",
            "character identity: character name",
        ),
    ],
)
def test_serialization_normalizes_tags_and_encodes_explicit_role(
    source: str,
    role: str,
    tag_name: str,
    expected_condition: str,
) -> None:
    plan = _plan(source)

    caption = serialize_caption(plan, _Tokenizer(), FramingContract(34, 5, 248044))

    assert caption.body == "long hair, year 2026\n\nsoft lighting"
    assert caption.condition_text == expected_condition
    assert caption.text == (
        SYSTEM_PREFIX
        + "long hair, year 2026\n\nsoft lighting"
        + MAIN_SUFFIX
        + expected_condition
    )
    assert caption.plan.tags[0].tag.text == "long_hair"
    assert caption.plan.tags[0].tag.canonical == "long_hair"
    assert caption.plan.condition is not None
    assert caption.plan.condition.tags[0].text == tag_name
    assert caption.condition_token_indices
    assert caption.use_null_condition is False
    assert caption.condition_source == source
    assert caption.condition_role == role


def test_serialization_truncates_only_complete_globally_ordered_tags() -> None:
    tags = tuple(
        CaptionTag("general", Tag(f"tag_{index:03d}", f"tag_{index:03d}"))
        for index in range(80)
    )
    requested = CaptionPlan(
        tags=tags,
        condition=None,
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
    assert caption.condition_source is None
    assert caption.condition_role is None
