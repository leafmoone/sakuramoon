"""Deterministic caption categories and explicit dropout decisions."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Literal

CATEGORY_ORDER = ("nsfw", "character", "copyright", "general")
ALL_CONDITION_DROPOUT = 0.10
NlBranch = Literal["long_names", "long_no_names", "short_vibes", "nl2", "nl3"]
NL_BRANCHES: tuple[NlBranch, ...] = (
    "long_names",
    "long_no_names",
    "short_vibes",
    "nl2",
    "nl3",
)


class CaptionError(ValueError):
    """Structured caption input or explicit probabilities are invalid."""


@dataclass(frozen=True)
class Tag:
    text: str
    canonical: str

    def __post_init__(self) -> None:
        if (
            type(self.text) is not str
            or type(self.canonical) is not str
            or not self.text
            or not self.canonical
            or self.text != self.text.strip()
            or self.canonical != self.canonical.strip()
            or ", " in self.text
            or "\n" in self.text
            or "<think>" in self.text
            or "</think>" in self.text
        ):
            raise CaptionError("tag text and canonical ID must be non-empty boundaries")


@dataclass(frozen=True)
class NlCandidates:
    long_names: str | None
    long_no_names: str | None
    short_vibes: str | None
    nl2: str | None
    nl3: str | None

    def available(self) -> tuple[tuple[NlBranch, str], ...]:
        result: list[tuple[NlBranch, str]] = []
        for branch in NL_BRANCHES:
            value = getattr(self, branch)
            if value is not None and value.strip():
                if "<think>" in value or "</think>" in value:
                    raise CaptionError("NL text must not contain thinking markers")
                result.append((branch, value.strip()))
        return tuple(result)


@dataclass(frozen=True)
class CaptionFields:
    nsfw: tuple[Tag, ...]
    character: tuple[Tag, ...]
    copyright: tuple[Tag, ...]
    general: tuple[Tag, ...]
    artists: tuple[Tag, ...]
    candidate_tags: frozenset[str]
    nl: NlCandidates

    def __post_init__(self) -> None:
        if type(self.candidate_tags) is not frozenset or any(
            type(candidate) is not str
            or not candidate
            or candidate != candidate.strip()
            for candidate in self.candidate_tags
        ):
            raise CaptionError(
                "candidate deletion IDs must be a frozenset of trim-stable strings"
            )


@dataclass(frozen=True)
class NlDropoutProbabilities:
    long_names: float
    long_no_names: float
    short_vibes: float
    nl2: float
    nl3: float


@dataclass(frozen=True)
class CaptionDropoutProbabilities:
    nsfw: float
    character: float
    copyright: float
    general: float
    artist: float
    candidate_source: float
    nl: NlDropoutProbabilities

    def __post_init__(self) -> None:
        values = (
            self.nsfw,
            self.character,
            self.copyright,
            self.general,
            self.artist,
            self.candidate_source,
            self.nl.long_names,
            self.nl.long_no_names,
            self.nl.short_vibes,
            self.nl.nl2,
            self.nl.nl3,
        )
        if any(type(value) is not float or not 0.0 <= value <= 1.0 for value in values):
            raise CaptionError("all non-global dropout probabilities must be explicit floats")
        nl_values = (
            self.nl.long_names,
            self.nl.long_no_names,
            self.nl.short_vibes,
            self.nl.nl2,
            self.nl.nl3,
        )
        if len(set(nl_values)) != 1:
            raise CaptionError("all five NL dropout probabilities must be equal")


@dataclass(frozen=True)
class CaptionPlan:
    nsfw: tuple[Tag, ...]
    character: tuple[Tag, ...]
    copyright: tuple[Tag, ...]
    general: tuple[Tag, ...]
    artists: tuple[Tag, ...]
    nl_text: str | None
    selected_nl: NlBranch | None
    all_condition_dropped: bool

    def __post_init__(self) -> None:
        has_content = any(
            (self.nsfw, self.character, self.copyright, self.general, self.artists)
        ) or self.nl_text is not None
        if self.all_condition_dropped and has_content:
            raise CaptionError("all-condition dropout plan must be empty")
        if (self.nl_text is None) != (self.selected_nl is None):
            raise CaptionError("NL text and selected branch must be present together")


def _seed(seed: int, domain: str) -> int:
    payload = f"{seed}\0{domain}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _drop(seed: int, domain: str, probability: float) -> bool:
    value = _seed(seed, domain) / 2**64
    return value < probability


def _shuffle(tags: tuple[Tag, ...], seed: int, domain: str) -> tuple[Tag, ...]:
    shuffled = list(tags)
    random.Random(_seed(seed, domain)).shuffle(shuffled)
    return tuple(shuffled)


def build_caption_plan(
    fields: CaptionFields,
    probabilities: CaptionDropoutProbabilities,
    *,
    seed: int,
) -> CaptionPlan:
    """Apply the fixed global dropout and all explicit per-field probabilities."""

    if type(seed) is not int or seed < 0:
        raise CaptionError("caption seed must be a non-negative integer")
    if _drop(seed, "all_condition", ALL_CONDITION_DROPOUT):
        return CaptionPlan((), (), (), (), (), None, None, True)

    categories: dict[str, tuple[Tag, ...]] = {}
    for category in CATEGORY_ORDER:
        probability = getattr(probabilities, category)
        source = getattr(fields, category)
        categories[category] = (
            ()
            if _drop(seed, f"dropout:{category}", probability)
            else _shuffle(source, seed, f"shuffle:{category}")
        )

    if _drop(seed, "dropout:candidate_source", probabilities.candidate_source):
        for category in CATEGORY_ORDER:
            categories[category] = tuple(
                tag
                for tag in categories[category]
                if tag.canonical not in fields.candidate_tags
            )

    artists = (
        ()
        if _drop(seed, "dropout:artist", probabilities.artist)
        else _shuffle(fields.artists, seed, "shuffle:artist")
    )

    available_nl: tuple[tuple[NlBranch, str], ...] = tuple(
        (branch, text)
        for branch, text in fields.nl.available()
        if not _drop(seed, f"dropout:nl:{branch}", getattr(probabilities.nl, branch))
    )
    if available_nl:
        selected_nl, nl_text = available_nl[
            random.Random(_seed(seed, "select:nl")).randrange(len(available_nl))
        ]
    else:
        selected_nl, nl_text = None, None
    return CaptionPlan(
        nsfw=categories["nsfw"],
        character=categories["character"],
        copyright=categories["copyright"],
        general=categories["general"],
        artists=artists,
        nl_text=nl_text,
        selected_nl=selected_nl,
        all_condition_dropped=False,
    )
