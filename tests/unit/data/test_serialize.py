from __future__ import annotations

from sakuramoon.data.caption import (
    CaptionPlan,
    CaptionTag,
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


def _plan() -> CaptionPlan:
    return CaptionPlan(
        tags=(
            CaptionTag("general", Tag("long_hair", "long_hair")),
            CaptionTag("year", Tag("year 2026", "year 2026")),
        ),
        artists=(Tag("artist_name", "artist_name"),),
        nl_text="soft lighting",
        selected_nl="short_vibes",
        all_condition_dropped=False,
        dropout_hits=empty_caption_dropout_hits(),
    )


def test_serialization_normalizes_only_tokenizer_facing_underscores() -> None:
    plan = _plan()

    caption = serialize_caption(plan, _Tokenizer(), FramingContract(34, 5, 248044))

    assert caption.body == "long hair, year 2026\n\nsoft lighting"
    assert caption.artist_text == "artist name"
    assert caption.text == (
        SYSTEM_PREFIX
        + "long hair, year 2026\n\nsoft lighting"
        + MAIN_SUFFIX
        + "artist name"
    )
    assert caption.plan.tags[0].tag.text == "long_hair"
    assert caption.plan.tags[0].tag.canonical == "long_hair"
    assert caption.plan.artists[0].text == "artist_name"
    assert caption.artist_token_indices
    assert caption.use_null_style is False


def test_serialization_truncates_only_complete_globally_ordered_tags() -> None:
    tags = tuple(
        CaptionTag("general", Tag(f"tag_{index:03d}", f"tag_{index:03d}"))
        for index in range(80)
    )
    requested = CaptionPlan(
        tags=tags,
        artists=(),
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
    assert caption.artist_token_indices == ()
    assert caption.use_null_style is True
