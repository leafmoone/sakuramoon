"""Deterministic validation prompt identities."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import cast

from sakuramoon.data.caption import CaptionDropoutHits, CaptionPlan, Tag
from sakuramoon.data.serialize import (
    MAIN_SUFFIX,
    SYSTEM_PREFIX,
    render_caption_segments,
)

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_NO_DROPOUT = CaptionDropoutHits(
    all_condition=False,
    nsfw=False,
    character=False,
    copyright=False,
    general=False,
    artist=False,
    candidate_source=False,
    long_names=False,
    long_no_names=False,
    short_vibes=False,
    nl2=False,
    nl3=False,
)
_CAPTION_TAG_FIELDS = ("nsfw", "character", "copyright", "general", "artists")


def _safe_id(name: str, value: str) -> None:
    if type(value) is not str or _SAFE_ID.fullmatch(value) is None:
        raise ValueError(f"{name} is invalid")


def caption_plan_prompt_text(plan: CaptionPlan) -> str:
    """Return the exact untruncated Qwen text surface for a caption plan."""

    body, artist_text = render_caption_segments(plan)
    return f"{SYSTEM_PREFIX}{body}{MAIN_SUFFIX}{artist_text}"


def _caption_plan_mapping(plan: CaptionPlan) -> dict[str, object]:
    return {
        **{
            field: [
                {"canonical": tag.canonical, "text": tag.text}
                for tag in getattr(plan, field)
            ]
            for field in _CAPTION_TAG_FIELDS
        },
        "nl_text": plan.nl_text,
        "selected_nl": plan.selected_nl,
    }


def _parse_caption_tags(value: object, field: str) -> tuple[Tag, ...]:
    if type(value) is not list:
        raise ValueError(f"prompt caption {field} must be an array")
    result: list[Tag] = []
    for raw_tag in cast(list[object], value):
        if type(raw_tag) is not dict:
            raise ValueError(f"prompt caption {field} tag must be an object")
        tag = cast(dict[str, object], raw_tag)
        if set(tag) != {"canonical", "text"}:
            raise ValueError(f"prompt caption {field} tag fields are invalid")
        result.append(Tag(cast(str, tag["text"]), cast(str, tag["canonical"])))
    return tuple(result)


def _parse_caption_plan(value: object) -> CaptionPlan | None:
    if value is None:
        return None
    if type(value) is not dict:
        raise ValueError("prompt caption plan must be an object or null")
    document = cast(dict[str, object], value)
    if set(document) != {*_CAPTION_TAG_FIELDS, "nl_text", "selected_nl"}:
        raise ValueError("prompt caption plan fields are invalid")
    nl_text = document["nl_text"]
    selected_nl = document["selected_nl"]
    if nl_text is not None and type(nl_text) is not str:
        raise ValueError("prompt caption NL text is invalid")
    if selected_nl is not None and selected_nl not in (
        "long_names",
        "long_no_names",
        "short_vibes",
        "nl2",
        "nl3",
    ):
        raise ValueError("prompt caption NL branch is invalid")
    return CaptionPlan(
        nsfw=_parse_caption_tags(document["nsfw"], "nsfw"),
        character=_parse_caption_tags(document["character"], "character"),
        copyright=_parse_caption_tags(document["copyright"], "copyright"),
        general=_parse_caption_tags(document["general"], "general"),
        artists=_parse_caption_tags(document["artists"], "artists"),
        nl_text=nl_text,
        selected_nl=selected_nl,
        all_condition_dropped=False,
        dropout_hits=_NO_DROPOUT,
    )


@dataclass(frozen=True, slots=True)
class PromptCase:
    prompt_id: str
    prompt: str
    conditions: tuple[str, ...]
    seed: int
    height: int
    width: int
    caption_plan: CaptionPlan | None = None

    def __post_init__(self) -> None:
        _safe_id("prompt_id", self.prompt_id)
        if (
            type(self.prompt) is not str
            or not self.prompt.strip()
            or (self.caption_plan is None and self.prompt != self.prompt.strip())
            or "<think>" in self.prompt
            or "</think>" in self.prompt
        ):
            raise ValueError("prompt text is invalid")
        if (
            type(self.conditions) is not tuple
            or any(
                type(value) is not str
                or not value
                or value != value.strip()
                or ", " in value
                or "\n" in value
                or "<think>" in value
                or "</think>" in value
                for value in self.conditions
            )
            or len(set(self.conditions)) != len(self.conditions)
        ):
            raise ValueError("prompt conditions must be complete tag boundaries")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("prompt seed must be a nonnegative integer")
        if any(
            type(value) is not int or value <= 0 or value % 16 != 0
            for value in (self.height, self.width)
        ):
            raise ValueError("prompt dimensions must be positive multiples of 16")
        if self.caption_plan is not None:
            plan = self.caption_plan
            typed_tags = all(
                type(getattr(plan, field)) is tuple
                and all(type(tag) is Tag for tag in getattr(plan, field))
                for field in _CAPTION_TAG_FIELDS
            )
            has_content = (
                any(getattr(plan, field) for field in _CAPTION_TAG_FIELDS)
                or plan.nl_text is not None
            )
            if (
                type(plan) is not CaptionPlan
                or not typed_tags
                or not has_content
                or plan.all_condition_dropped
                or any(plan.dropout_hits.as_mapping().values())
                or self.prompt != caption_plan_prompt_text(plan)
            ):
                raise ValueError("structured evaluator caption plan is invalid")

    def as_mapping(self) -> dict[str, object]:
        return {
            "caption_plan": (
                _caption_plan_mapping(self.caption_plan)
                if self.caption_plan is not None
                else None
            ),
            "conditions": list(self.conditions),
            "height": self.height,
            "prompt": self.prompt,
            "prompt_id": self.prompt_id,
            "seed": self.seed,
            "width": self.width,
        }


@dataclass(frozen=True, slots=True)
class PromptManifest:
    cases: tuple[PromptCase, ...]

    def __post_init__(self) -> None:
        if type(self.cases) is not tuple or any(
            type(case) is not PromptCase for case in self.cases
        ):
            raise TypeError(
                "prompt manifest cases must be an immutable PromptCase tuple"
            )
        if not self.cases:
            raise ValueError("prompt manifest must not be empty")
        identifiers = tuple(case.prompt_id for case in self.cases)
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("prompt IDs must be unique")

    def canonical_bytes(self) -> bytes:
        return (
            json.dumps(
                {
                    "cases": [case.as_mapping() for case in self.cases],
                    "schema_version": 1,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode()

    @classmethod
    def from_canonical_bytes(cls, payload: bytes) -> PromptManifest:
        if type(payload) is not bytes:
            raise TypeError("prompt manifest payload must be bytes")
        try:
            parsed: object = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise ValueError("prompt manifest must be valid canonical JSON") from None
        if type(parsed) is not dict:
            raise ValueError("prompt manifest root must be an object")
        document = cast(dict[str, object], parsed)
        if set(document) != {"cases", "schema_version"}:
            raise ValueError("prompt manifest root fields are invalid")
        if document["schema_version"] != 1:
            raise ValueError("prompt manifest schema version is invalid")
        raw_cases = document["cases"]
        if type(raw_cases) is not list:
            raise ValueError("prompt manifest cases must be an array")
        cases: list[PromptCase] = []
        expected_fields = {
            "caption_plan",
            "conditions",
            "height",
            "prompt",
            "prompt_id",
            "seed",
            "width",
        }
        for raw_case in cast(list[object], raw_cases):
            if type(raw_case) is not dict:
                raise ValueError("prompt manifest case must be an object")
            case = cast(dict[str, object], raw_case)
            if set(case) != expected_fields:
                raise ValueError("prompt manifest case fields are invalid")
            raw_conditions = case["conditions"]
            if type(raw_conditions) is not list:
                raise ValueError("prompt manifest conditions must be a string array")
            condition_values = cast(list[object], raw_conditions)
            if any(type(value) is not str for value in condition_values):
                raise ValueError("prompt manifest conditions must be a string array")
            cases.append(
                PromptCase(
                    prompt_id=cast(str, case["prompt_id"]),
                    prompt=cast(str, case["prompt"]),
                    conditions=tuple(cast(str, value) for value in condition_values),
                    seed=cast(int, case["seed"]),
                    height=cast(int, case["height"]),
                    width=cast(int, case["width"]),
                    caption_plan=_parse_caption_plan(case["caption_plan"]),
                )
            )
        manifest = cls(tuple(cases))
        if manifest.canonical_bytes() != payload:
            raise ValueError("prompt manifest must use canonical serialization")
        return manifest
