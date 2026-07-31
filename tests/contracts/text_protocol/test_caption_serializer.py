from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from sakuramoon.data.caption import CaptionPlan, Tag
from sakuramoon.data.serialize import (
    CONDITION_BUCKETS,
    MAIN_SUFFIX,
    SYSTEM_PREFIX,
    CaptionSerializationError,
    FramingContract,
    TokenEncoder,
    serialize_caption,
)


class _CharacterTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        if text == SYSTEM_PREFIX:
            return list(range(1000, 1034))
        if text == MAIN_SUFFIX:
            return list(range(2000, 2005))
        return [ord(character) + 1 for character in text]


def _framing(tokenizer: _CharacterTokenizer) -> FramingContract:
    return FramingContract(
        prefix_tokens=len(tokenizer.encode(SYSTEM_PREFIX, add_special_tokens=False)),
        suffix_tokens=len(tokenizer.encode(MAIN_SUFFIX, add_special_tokens=False)),
        padding_token_id=0,
    )


def _plan(
    *,
    nsfw: tuple[Tag, ...] = (),
    character: tuple[Tag, ...] = (),
    copyright: tuple[Tag, ...] = (),
    general: tuple[Tag, ...] = (),
    artists: tuple[Tag, ...] = (),
    nl_text: str | None = None,
    selected_nl: str | None = None,
    all_condition_dropped: bool = False,
) -> CaptionPlan:
    return CaptionPlan(
        nsfw=nsfw,
        character=character,
        copyright=copyright,
        general=general,
        artists=artists,
        nl_text=nl_text,
        selected_nl=selected_nl,  # type: ignore[arg-type]
        all_condition_dropped=all_condition_dropped,
    )


def _tag(text: str) -> Tag:
    return Tag(text, f"canonical:{text}")


def test_fixed_framing_category_order_and_join_rules() -> None:
    tokenizer = _CharacterTokenizer()
    plan = _plan(
        nsfw=(_tag("safe"),),
        character=(_tag("alice"),),
        copyright=(_tag("wonderland"),),
        general=(_tag("blue dress"),),
        nl_text="soft light",
        selected_nl="short_vibes",
    )

    result = serialize_caption(plan, tokenizer, _framing(tokenizer))

    assert result.body == "safe, alice, wonderland, blue dress\n\nsoft light"
    assert result.text == SYSTEM_PREFIX + result.body + MAIN_SUFFIX
    assert "Tags:" not in result.text
    assert "Description:" not in result.text
    assert result.selected_nl == "short_vibes"


@pytest.mark.parametrize(
    ("plan", "body"),
    [
        (_plan(general=(_tag("tag only"),)), "tag only"),
        (_plan(nl_text="NL only", selected_nl="nl2"), "NL only"),
        (_plan(), ""),
    ],
)
def test_tags_only_nl_only_and_empty_body(plan: CaptionPlan, body: str) -> None:
    tokenizer = _CharacterTokenizer()
    result = serialize_caption(plan, tokenizer, _framing(tokenizer))
    assert result.body == body
    assert result.text == SYSTEM_PREFIX + body + MAIN_SUFFIX


def test_artist_is_structurally_after_every_main_token() -> None:
    tokenizer = _CharacterTokenizer()
    result = serialize_caption(
        _plan(general=(_tag("main tag"),), artists=(_tag("artist one"),)),
        tokenizer,
        _framing(tokenizer),
    )

    assert result.artist_text == "artist one"
    assert result.text.endswith(MAIN_SUFFIX + "artist one")
    assert result.main_token_indices == tuple(range(len(result.main_token_indices)))
    assert result.artist_token_indices == tuple(
        range(len(result.main_token_indices), len(result.input_ids))
    )
    assert not set(result.main_token_indices) & set(result.artist_token_indices)
    assert result.artist_mask == (True,) * len(result.artist_token_indices)
    assert result.use_null_style is False


def test_all_condition_empty_template_uses_null_style() -> None:
    tokenizer = _CharacterTokenizer()
    result = serialize_caption(
        _plan(all_condition_dropped=True), tokenizer, _framing(tokenizer)
    )
    assert result.text == SYSTEM_PREFIX + MAIN_SUFFIX
    assert result.body == result.artist_text == ""
    assert result.artist_token_indices == ()
    assert result.use_null_style is True
    assert result.all_condition_dropped is True


def test_truncation_removes_nl_before_complete_low_priority_tag() -> None:
    tokenizer = _CharacterTokenizer()
    result = serialize_caption(
        _plan(
            nsfw=(_tag("n" * 100),),
            character=(_tag("c" * 100),),
            copyright=(_tag("p" * 100),),
            general=(_tag("g" * 250),),
            nl_text="description " * 100,
            selected_nl="long_names",
        ),
        tokenizer,
        _framing(tokenizer),
    )

    assert result.truncated is True
    assert result.selected_nl is None
    assert "description" not in result.body
    assert "g" * 250 not in result.body
    assert "n" * 100 in result.body
    assert "c" * 100 in result.body
    assert "p" * 100 in result.body


def test_oversized_tag_is_removed_as_one_boundary() -> None:
    tokenizer = _CharacterTokenizer()
    result = serialize_caption(
        _plan(general=(_tag("x" * 600),)), tokenizer, _framing(tokenizer)
    )
    assert result.body == ""
    assert result.truncated is True


def test_artist_reservation_keeps_at_least_one_complete_source() -> None:
    tokenizer = _CharacterTokenizer()
    result = serialize_caption(
        _plan(artists=(_tag("a" * 400), _tag("b" * 200))),
        tokenizer,
        _framing(tokenizer),
    )
    assert result.artist_text == "a" * 400
    assert result.truncated is True
    assert result.use_null_style is False


def test_artist_reservation_skips_oversized_first_source_for_later_valid_source() -> None:
    tokenizer = _CharacterTokenizer()
    result = serialize_caption(
        _plan(artists=(_tag("a" * 600), _tag("valid artist"))),
        tokenizer,
        _framing(tokenizer),
    )

    assert result.artist_text == "valid artist"
    assert result.truncated is True
    assert result.use_null_style is False


def test_unique_oversized_artist_is_a_hard_failure() -> None:
    tokenizer = _CharacterTokenizer()
    with pytest.raises(CaptionSerializationError, match="Artist"):
        serialize_caption(
            _plan(artists=(_tag("a" * 600),)), tokenizer, _framing(tokenizer)
        )


def test_all_oversized_artists_are_a_hard_failure() -> None:
    tokenizer = _CharacterTokenizer()
    with pytest.raises(CaptionSerializationError, match="Artist"):
        serialize_caption(
            _plan(artists=(_tag("a" * 600), _tag("b" * 700))),
            tokenizer,
            _framing(tokenizer),
        )


def test_condition_bucket_and_dense_length_use_measured_prefix() -> None:
    tokenizer = _CharacterTokenizer()
    framing = _framing(tokenizer)
    result = serialize_caption(
        _plan(general=(_tag("short"),)), tokenizer, framing
    )
    assert result.condition_bucket in CONDITION_BUCKETS
    assert result.condition_bucket >= result.condition_tokens
    assert result.dense_length == framing.prefix_tokens + result.condition_bucket
    assert result.attention_mask == (True,) * len(result.input_ids)
    assert framing.padding_token_id not in result.input_ids
    assert tuple(framing.prefix_tokens + bucket for bucket in CONDITION_BUCKETS) == (
        98,
        162,
        226,
        290,
        354,
        418,
        482,
        546,
    )


def test_changed_framing_token_counts_fail() -> None:
    tokenizer = _CharacterTokenizer()
    framing = _framing(tokenizer)
    with pytest.raises(CaptionSerializationError, match="counts changed"):
        class _ChangedTokenizer(_CharacterTokenizer):
            def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
                encoded = super().encode(text, add_special_tokens=add_special_tokens)
                return encoded + [9999] if text == SYSTEM_PREFIX else encoded

        serialize_caption(
            _plan(),
            _ChangedTokenizer(),
            framing,
        )


@pytest.mark.parametrize("padding_token_id", [-1, True])
def test_framing_rejects_invalid_padding_token_id(padding_token_id: object) -> None:
    with pytest.raises(CaptionSerializationError, match="padding token"):
        FramingContract(
            prefix_tokens=34,
            suffix_tokens=5,
            padding_token_id=padding_token_id,  # pyright: ignore[reportArgumentType]
        )


def test_tokenizer_negative_token_id_is_rejected() -> None:
    class _NegativeTokenTokenizer(_CharacterTokenizer):
        def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
            if text == "invalid":
                return [-1]
            return super().encode(text, add_special_tokens=add_special_tokens)

    tokenizer = _NegativeTokenTokenizer()
    with pytest.raises(CaptionSerializationError, match="invalid token IDs"):
        serialize_caption(
            _plan(general=(_tag("invalid"),)), tokenizer, _framing(tokenizer)
        )


def test_real_local_tokenizer_framing_and_segment_equivalence() -> None:
    repository_root = Path(__file__).parents[3]
    tokenizer = cast(
        PreTrainedTokenizerBase,
        AutoTokenizer.from_pretrained(  # pyright: ignore[reportUnknownMemberType]
            repository_root / "model/qwen_3.5_2B",
            local_files_only=True,
            trust_remote_code=False,
            use_fast=True,
            padding_side="right",
        ),
    )
    encoder = cast(TokenEncoder, tokenizer)
    prefix_ids = encoder.encode(SYSTEM_PREFIX, add_special_tokens=False)
    suffix_ids = encoder.encode(MAIN_SUFFIX, add_special_tokens=False)

    assert len(prefix_ids) == 34
    assert len(suffix_ids) == 5
    assert tokenizer.pad_token_id == 248044
    for body, artist in (
        ("", ""),
        ("blue sky", ""),
        ("blue sky, 1girl\n\nsoft light", "sample artist"),
        ("Japanese text", "artist:name"),
    ):
        segmented = (
            prefix_ids
            + encoder.encode(body, add_special_tokens=False)
            + suffix_ids
            + encoder.encode(artist, add_special_tokens=False)
        )
        whole = encoder.encode(
            SYSTEM_PREFIX + body + MAIN_SUFFIX + artist,
            add_special_tokens=False,
        )
        assert segmented == whole
