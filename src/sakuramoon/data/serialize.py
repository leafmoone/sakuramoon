"""Fixed Qwen framing with structured main and Artist token indices."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from sakuramoon.data.caption import CATEGORY_ORDER, CaptionPlan, Tag

SYSTEM_PREFIX = (
    "<|im_start|>system\n"
    "Describe the image by detailing the color, shape, size, texture, quantity, "
    "text, spatial relationships of the objects and background:<|im_end|>\n"
    "<|im_start|>user\n"
)
MAIN_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"
CONDITION_BUCKETS = (64, 128, 192, 256, 320, 384, 448, 512)
TEXT_CONDITION_MAX = 512
EXPECTED_PREFIX_TOKENS = 34
EXPECTED_SUFFIX_TOKENS = 5


class CaptionSerializationError(ValueError):
    """Caption cannot be serialized without violating a locked boundary."""


class TokenEncoder(Protocol):
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]: ...


@dataclass(frozen=True)
class FramingContract:
    prefix_tokens: int
    suffix_tokens: int
    padding_token_id: int

    def __post_init__(self) -> None:
        if (
            self.prefix_tokens != EXPECTED_PREFIX_TOKENS
            or self.suffix_tokens != EXPECTED_SUFFIX_TOKENS
        ):
            raise CaptionSerializationError(
                "Qwen framing must measure as 34 prefix and 5 suffix tokens"
            )


@dataclass(frozen=True)
class SerializedCaption:
    text: str
    input_ids: tuple[int, ...]
    attention_mask: tuple[bool, ...]
    main_token_indices: tuple[int, ...]
    main_mask: tuple[bool, ...]
    artist_token_indices: tuple[int, ...]
    artist_mask: tuple[bool, ...]
    use_null_style: bool
    all_condition_dropped: bool
    selected_nl: str | None
    body: str
    artist_text: str
    condition_tokens: int
    condition_bucket: int
    dense_length: int
    truncated: bool


def _encode(tokenizer: TokenEncoder, text: str) -> tuple[int, ...]:
    encoded = tokenizer.encode(text, add_special_tokens=False)
    if any(type(token) is not int for token in encoded):
        raise CaptionSerializationError("tokenizer returned invalid token IDs")
    return tuple(encoded)


def _body(categories: dict[str, list[Tag]], nl_text: str | None) -> str:
    tags = [tag.text for category in CATEGORY_ORDER for tag in categories[category]]
    tag_text = ", ".join(tags)
    if tag_text and nl_text:
        return f"{tag_text}\n\n{nl_text}"
    return tag_text or (nl_text or "")


def _artist_text(artists: list[Tag]) -> str:
    return ", ".join(tag.text for tag in artists)


def _bucket_for(tokens: int) -> int:
    for bucket in CONDITION_BUCKETS:
        if tokens <= bucket:
            return bucket
    raise CaptionSerializationError("caption exceeds the 512 condition-token limit")


def serialize_caption(
    plan: CaptionPlan,
    tokenizer: TokenEncoder,
    framing: FramingContract,
) -> SerializedCaption:
    """Encode structural segments and truncate only complete NL/tag boundaries."""

    prefix_ids = _encode(tokenizer, SYSTEM_PREFIX)
    suffix_ids = _encode(tokenizer, MAIN_SUFFIX)
    if len(prefix_ids) != framing.prefix_tokens or len(suffix_ids) != framing.suffix_tokens:
        raise CaptionSerializationError("tokenizer framing token counts changed")

    categories = {
        category: list(getattr(plan, category)) for category in CATEGORY_ORDER
    }
    artists = list(plan.artists)
    nl_text = plan.nl_text
    truncated = False

    if artists:
        fitting_artists: list[Tag] = []
        for artist in artists:
            individual_ids = _encode(tokenizer, artist.text)
            if not individual_ids:
                raise CaptionSerializationError("tokenizer produced no Artist tokens")
            if len(suffix_ids) + len(individual_ids) <= TEXT_CONDITION_MAX:
                fitting_artists.append(artist)
            else:
                truncated = True
        if not fitting_artists:
            raise CaptionSerializationError(
                "Artist segment cannot fit without splitting a tag"
            )
        artists = fitting_artists

    artist_text = _artist_text(artists)
    artist_ids = _encode(tokenizer, artist_text)
    while len(suffix_ids) + len(artist_ids) > TEXT_CONDITION_MAX and len(artists) > 1:
        artists.pop()
        truncated = True
        artist_text = _artist_text(artists)
        artist_ids = _encode(tokenizer, artist_text)
    if len(suffix_ids) + len(artist_ids) > TEXT_CONDITION_MAX:
        raise CaptionSerializationError("Artist segment cannot fit without splitting a tag")

    while True:
        body = _body(categories, nl_text)
        body_ids = _encode(tokenizer, body)
        condition_tokens = len(body_ids) + len(suffix_ids) + len(artist_ids)
        if condition_tokens <= TEXT_CONDITION_MAX:
            break
        if nl_text is not None:
            nl_text = None
            truncated = True
            continue
        removed = False
        for category in reversed(CATEGORY_ORDER):
            if categories[category]:
                categories[category].pop()
                truncated = True
                removed = True
                break
        if not removed:
            raise CaptionSerializationError("fixed framing exceeds the condition limit")

    input_ids = prefix_ids + body_ids + suffix_ids + artist_ids
    if framing.padding_token_id in input_ids:
        raise CaptionSerializationError("padding token appears in the valid Qwen sequence")
    main_length = len(prefix_ids) + len(body_ids) + len(suffix_ids)
    main_indices = tuple(range(main_length))
    artist_indices = tuple(range(main_length, len(input_ids)))
    condition_bucket = _bucket_for(condition_tokens)
    return SerializedCaption(
        text=SYSTEM_PREFIX + body + MAIN_SUFFIX + artist_text,
        input_ids=input_ids,
        attention_mask=(True,) * len(input_ids),
        main_token_indices=main_indices,
        main_mask=(True,) * len(main_indices),
        artist_token_indices=artist_indices,
        artist_mask=(True,) * len(artist_indices),
        use_null_style=not bool(artist_indices),
        all_condition_dropped=plan.all_condition_dropped,
        selected_nl=plan.selected_nl if nl_text is not None else None,
        body=body,
        artist_text=artist_text,
        condition_tokens=condition_tokens,
        condition_bucket=condition_bucket,
        dense_length=len(prefix_ids) + condition_bucket,
        truncated=truncated,
    )
