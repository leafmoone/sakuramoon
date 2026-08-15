"""Fixed Qwen framing with structured main and condition-token indices."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Protocol

from sakuramoon.data.caption import (
    CaptionDropoutHits,
    CaptionPlan,
    CaptionTag,
    ConditionRequest,
    ConditionRole,
    ConditionSource,
    Tag,
)

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
CONDITION_ROLE_PREFIXES: dict[ConditionRole, str] = {
    "style": "style reference: ",
    "identity": "character identity: ",
}


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
        if type(self.padding_token_id) is not int or self.padding_token_id < 0:
            raise CaptionSerializationError(
                "Qwen padding token ID must be a non-negative integer"
            )
        if (
            self.prefix_tokens != EXPECTED_PREFIX_TOKENS
            or self.suffix_tokens != EXPECTED_SUFFIX_TOKENS
        ):
            raise CaptionSerializationError(
                "Qwen framing must measure as 34 prefix and 5 suffix tokens"
            )


@dataclass(frozen=True)
class SerializedCaption:
    plan: CaptionPlan
    text: str
    input_ids: tuple[int, ...]
    attention_mask: tuple[bool, ...]
    main_token_indices: tuple[int, ...]
    main_mask: tuple[bool, ...]
    condition_token_indices: tuple[int, ...]
    condition_mask: tuple[bool, ...]
    use_null_condition: bool
    condition_source: ConditionSource | None
    condition_role: ConditionRole | None
    all_condition_dropped: bool
    dropout_hits: CaptionDropoutHits
    selected_nl: str | None
    body: str
    condition_text: str
    condition_tokens: int
    condition_bucket: int
    dense_length: int
    truncated: bool


def _encode(tokenizer: TokenEncoder, text: str) -> tuple[int, ...]:
    encoded = tokenizer.encode(text, add_special_tokens=False)
    if any(type(token) is not int or token < 0 for token in encoded):
        raise CaptionSerializationError("tokenizer returned invalid token IDs")
    return tuple(encoded)


def _display_text(tag: Tag) -> str:
    """Normalize only the tokenizer-facing surface; canonical IDs stay untouched."""

    return tag.text.replace("_", " ")


def _body(tags: list[CaptionTag], nl_text: str | None) -> str:
    tag_text = ", ".join(_display_text(item.tag) for item in tags)
    if tag_text and nl_text:
        return f"{tag_text}\n\n{nl_text}"
    return tag_text or (nl_text or "")


def _condition_text(condition: ConditionRequest | None) -> str:
    if condition is None:
        return ""
    prefix = CONDITION_ROLE_PREFIXES[condition.role]
    return prefix + ", ".join(_display_text(tag) for tag in condition.tags)


def render_caption_segments(plan: CaptionPlan) -> tuple[str, str]:
    """Render the governed main and role-explicit condition segments."""

    return _body(list(plan.tags), plan.nl_text), _condition_text(plan.condition)


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
    if (
        len(prefix_ids) != framing.prefix_tokens
        or len(suffix_ids) != framing.suffix_tokens
    ):
        raise CaptionSerializationError("tokenizer framing token counts changed")

    tags = list(plan.tags)
    condition_source = None if plan.condition is None else plan.condition.source
    condition_role = None if plan.condition is None else plan.condition.role
    condition_tags = [] if plan.condition is None else list(plan.condition.tags)
    nl_text = plan.nl_text
    truncated = False

    def current_plan() -> CaptionPlan:
        return dataclasses.replace(
            plan,
            tags=tuple(tags),
            condition=(
                ConditionRequest(
                    source=condition_source,
                    role=condition_role,
                    tags=tuple(condition_tags),
                )
                if condition_source is not None
                and condition_role is not None
                and condition_tags
                else None
            ),
            nl_text=nl_text,
            selected_nl=plan.selected_nl if nl_text is not None else None,
        )

    if condition_tags:
        if condition_source is None or condition_role is None:
            raise CaptionSerializationError("condition source or role is missing")
        fitting_tags: list[Tag] = []
        for tag in condition_tags:
            individual = ConditionRequest(
                source=condition_source,
                role=condition_role,
                tags=(tag,),
            )
            individual_ids = _encode(tokenizer, _condition_text(individual))
            if not individual_ids:
                raise CaptionSerializationError(
                    "tokenizer produced no condition tokens"
                )
            if len(suffix_ids) + len(individual_ids) <= TEXT_CONDITION_MAX:
                fitting_tags.append(tag)
            else:
                truncated = True
        if not fitting_tags:
            raise CaptionSerializationError(
                "condition segment cannot fit without splitting a tag"
            )
        condition_tags = fitting_tags

    _body_text, condition_text = render_caption_segments(current_plan())
    condition_ids = _encode(tokenizer, condition_text)
    while (
        len(suffix_ids) + len(condition_ids) > TEXT_CONDITION_MAX
        and len(condition_tags) > 1
    ):
        condition_tags.pop()
        truncated = True
        _body_text, condition_text = render_caption_segments(current_plan())
        condition_ids = _encode(tokenizer, condition_text)
    if len(suffix_ids) + len(condition_ids) > TEXT_CONDITION_MAX:
        raise CaptionSerializationError(
            "condition segment cannot fit without splitting a tag"
        )

    while True:
        body, condition_text = render_caption_segments(current_plan())
        body_ids = _encode(tokenizer, body)
        condition_tokens = len(body_ids) + len(suffix_ids) + len(condition_ids)
        if condition_tokens <= TEXT_CONDITION_MAX:
            break
        if nl_text is not None:
            nl_text = None
            truncated = True
            continue
        if not tags:
            raise CaptionSerializationError("fixed framing exceeds the condition limit")
        tags.pop()
        truncated = True

    resolved_plan = current_plan()
    if render_caption_segments(resolved_plan) != (body, condition_text):
        raise RuntimeError("resolved caption plan differs from serialized segments")
    input_ids = prefix_ids + body_ids + suffix_ids + condition_ids
    if framing.padding_token_id in input_ids:
        raise CaptionSerializationError(
            "padding token appears in the valid Qwen sequence"
    )
    main_length = len(prefix_ids) + len(body_ids) + len(suffix_ids)
    main_indices = tuple(range(main_length))
    condition_indices = tuple(range(main_length, len(input_ids)))
    condition_bucket = _bucket_for(condition_tokens)
    return SerializedCaption(
        plan=resolved_plan,
        text=SYSTEM_PREFIX + body + MAIN_SUFFIX + condition_text,
        input_ids=input_ids,
        attention_mask=(True,) * len(input_ids),
        main_token_indices=main_indices,
        main_mask=(True,) * len(main_indices),
        condition_token_indices=condition_indices,
        condition_mask=(True,) * len(condition_indices),
        use_null_condition=not bool(condition_indices),
        condition_source=(
            None if resolved_plan.condition is None else resolved_plan.condition.source
        ),
        condition_role=(
            None if resolved_plan.condition is None else resolved_plan.condition.role
        ),
        all_condition_dropped=resolved_plan.all_condition_dropped,
        dropout_hits=resolved_plan.dropout_hits,
        selected_nl=resolved_plan.selected_nl,
        body=body,
        condition_text=condition_text,
        condition_tokens=condition_tokens,
        condition_bucket=condition_bucket,
        dense_length=len(prefix_ids) + condition_bucket,
        truncated=truncated,
    )
