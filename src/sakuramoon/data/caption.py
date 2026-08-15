"""Deterministic per-tag caption dropout and global tag ordering."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Literal, cast

ALL_CONDITION_DROPOUT = 0.10
TAG_SOURCE_ORDER = (
    "rating",
    "year",
    "aesthetic",
    "quality",
    "anime_completeness",
    "anime_classification",
    "nsfw",
    "character",
    "copyright",
    "general",
    "artist",
)
BODY_TAG_SOURCE_ORDER = TAG_SOURCE_ORDER[:-1]
CANDIDATE_SOURCE_FIELDS = frozenset({"nsfw", "character", "copyright", "general"})
CAPTION_DROPOUT_KEYS: tuple[str, ...] = (
    "all_condition",
    *TAG_SOURCE_ORDER,
    "candidate_source",
    "long_names",
    "long_no_names",
    "short_vibes",
    "nl2",
    "nl3",
)
STYLE_CONDITION_ROUTE_KEYS: tuple[str, ...] = ("artist", "character", "null")
TagSource = Literal[
    "rating",
    "year",
    "aesthetic",
    "quality",
    "anime_completeness",
    "anime_classification",
    "nsfw",
    "character",
    "copyright",
    "general",
    "artist",
]
StyleConditionKind = Literal["artist", "character"]
StyleConditionMode = Literal["artist", "artist_or_character"]
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
            or "\r" in self.text
            or "\0" in self.text
            or "\n" in self.canonical
            or "\r" in self.canonical
            or "\0" in self.canonical
            or "<think>" in self.text
            or "</think>" in self.text
        ):
            raise CaptionError("tag text and canonical ID must be non-empty boundaries")


@dataclass(frozen=True)
class CaptionTag:
    source: TagSource
    tag: Tag

    def __post_init__(self) -> None:
        if (
            type(self.source) is not str
            or self.source not in TAG_SOURCE_ORDER
            or type(self.tag) is not Tag
        ):
            raise CaptionError("caption tag source or value is invalid")


@dataclass(frozen=True)
class StyleCondition:
    kind: StyleConditionKind
    tags: tuple[Tag, ...]

    def __post_init__(self) -> None:
        if (
            type(self.kind) is not str
            or self.kind not in {"artist", "character"}
            or type(self.tags) is not tuple
            or not self.tags
            or any(type(tag) is not Tag for tag in self.tags)
        ):
            raise CaptionError("style condition kind or tags are invalid")


@dataclass(frozen=True)
class StyleConditionRouteCounts:
    artist: int
    character: int
    null: int

    def __post_init__(self) -> None:
        if any(
            type(value) is not int or value < 0 for value in self.as_mapping().values()
        ):
            raise CaptionError(
                "style condition route counts must be non-negative integers"
            )

    def as_mapping(self) -> dict[str, int]:
        return {
            "artist": self.artist,
            "character": self.character,
            "null": self.null,
        }


def count_style_condition_routes(
    kinds: tuple[StyleConditionKind | None, ...],
) -> StyleConditionRouteCounts:
    if type(kinds) is not tuple or any(
        kind is not None
        and (type(kind) is not str or kind not in {"artist", "character"})
        for kind in kinds
    ):
        raise CaptionError("style condition route kinds are invalid")
    counts = dict.fromkeys(STYLE_CONDITION_ROUTE_KEYS, 0)
    for kind in kinds:
        counts["null" if kind is None else kind] += 1
    return StyleConditionRouteCounts(
        artist=counts["artist"],
        character=counts["character"],
        null=counts["null"],
    )


@dataclass(frozen=True)
class NlCandidates:
    long_names: str | None
    long_no_names: str | None
    short_vibes: str | None
    nl2: str | None
    nl3: str | None

    def __post_init__(self) -> None:
        if any(
            value is not None and type(value) is not str
            for value in (
                self.long_names,
                self.long_no_names,
                self.short_vibes,
                self.nl2,
                self.nl3,
            )
        ):
            raise CaptionError("NL candidates must be text or null")

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
    rating: tuple[Tag, ...] = ()
    year: tuple[Tag, ...] = ()
    aesthetic: tuple[Tag, ...] = ()
    quality: tuple[Tag, ...] = ()
    anime_completeness: tuple[Tag, ...] = ()
    anime_classification: tuple[Tag, ...] = ()

    def __post_init__(self) -> None:
        for source in TAG_SOURCE_ORDER:
            values = self.artists if source == "artist" else getattr(self, source)
            if type(values) is not tuple or any(type(tag) is not Tag for tag in values):
                raise CaptionError(f"caption field {source} must be an exact Tag tuple")
        if type(self.nl) is not NlCandidates:
            raise CaptionError("caption NL candidates are invalid")
        if type(self.candidate_tags) is not frozenset or any(
            type(candidate) is not str
            or not candidate
            or candidate != candidate.strip()
            or "\n" in candidate
            or "\r" in candidate
            or "\0" in candidate
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
    tag: float
    candidate_source: float
    nl: NlDropoutProbabilities

    def __post_init__(self) -> None:
        if type(self.nl) is not NlDropoutProbabilities:
            raise CaptionError("NL dropout probabilities use an invalid type")
        values = (
            self.tag,
            self.candidate_source,
            self.nl.long_names,
            self.nl.long_no_names,
            self.nl.short_vibes,
            self.nl.nl2,
            self.nl.nl3,
        )
        if any(type(value) is not float or not 0.0 <= value <= 1.0 for value in values):
            raise CaptionError("all dropout probabilities must be explicit floats")
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
class CaptionDropoutHits:
    all_condition: bool
    rating: bool
    year: bool
    aesthetic: bool
    quality: bool
    anime_completeness: bool
    anime_classification: bool
    nsfw: bool
    character: bool
    copyright: bool
    general: bool
    artist: bool
    candidate_source: bool
    long_names: bool
    long_no_names: bool
    short_vibes: bool
    nl2: bool
    nl3: bool

    def __post_init__(self) -> None:
        if any(type(value) is not bool for value in self.as_mapping().values()):
            raise CaptionError("caption dropout hits must be exact booleans")

    def as_mapping(self) -> dict[str, bool]:
        return {
            "all_condition": self.all_condition,
            "rating": self.rating,
            "year": self.year,
            "aesthetic": self.aesthetic,
            "quality": self.quality,
            "anime_completeness": self.anime_completeness,
            "anime_classification": self.anime_classification,
            "nsfw": self.nsfw,
            "character": self.character,
            "copyright": self.copyright,
            "general": self.general,
            "artist": self.artist,
            "candidate_source": self.candidate_source,
            "long_names": self.long_names,
            "long_no_names": self.long_no_names,
            "short_vibes": self.short_vibes,
            "nl2": self.nl2,
            "nl3": self.nl3,
        }


@dataclass(frozen=True)
class CaptionDropoutCounts:
    all_condition: int
    rating: int
    year: int
    aesthetic: int
    quality: int
    anime_completeness: int
    anime_classification: int
    nsfw: int
    character: int
    copyright: int
    general: int
    artist: int
    candidate_source: int
    long_names: int
    long_no_names: int
    short_vibes: int
    nl2: int
    nl3: int

    def __post_init__(self) -> None:
        if any(
            type(value) is not int or value < 0 for value in self.as_mapping().values()
        ):
            raise CaptionError("caption dropout counts must be non-negative integers")

    def as_mapping(self) -> dict[str, int]:
        return {
            "all_condition": self.all_condition,
            "rating": self.rating,
            "year": self.year,
            "aesthetic": self.aesthetic,
            "quality": self.quality,
            "anime_completeness": self.anime_completeness,
            "anime_classification": self.anime_classification,
            "nsfw": self.nsfw,
            "character": self.character,
            "copyright": self.copyright,
            "general": self.general,
            "artist": self.artist,
            "candidate_source": self.candidate_source,
            "long_names": self.long_names,
            "long_no_names": self.long_no_names,
            "short_vibes": self.short_vibes,
            "nl2": self.nl2,
            "nl3": self.nl3,
        }


def empty_caption_dropout_hits(*, all_condition: bool = False) -> CaptionDropoutHits:
    if type(all_condition) is not bool:
        raise CaptionError("all-condition dropout state must be an exact boolean")
    return CaptionDropoutHits(
        all_condition=all_condition,
        rating=False,
        year=False,
        aesthetic=False,
        quality=False,
        anime_completeness=False,
        anime_classification=False,
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


@dataclass(frozen=True)
class CaptionPlan:
    tags: tuple[CaptionTag, ...]
    style_condition: StyleCondition | None
    nl_text: str | None
    selected_nl: NlBranch | None
    all_condition_dropped: bool
    dropout_hits: CaptionDropoutHits

    def __post_init__(self) -> None:
        has_content = bool(
            self.tags or self.style_condition is not None or self.nl_text is not None
        )
        if type(self.tags) is not tuple or any(
            type(tag) is not CaptionTag for tag in self.tags
        ):
            raise CaptionError("caption plan tags must be an exact CaptionTag tuple")
        if self.style_condition is not None and type(self.style_condition) is not StyleCondition:
            raise CaptionError("caption plan style condition is invalid")
        if (
            type(self.all_condition_dropped) is not bool
            or type(self.dropout_hits) is not CaptionDropoutHits
            or self.all_condition_dropped != self.dropout_hits.all_condition
        ):
            raise CaptionError("all-condition result and dropout hit must agree")
        if self.all_condition_dropped and has_content:
            raise CaptionError("all-condition dropout plan must be empty")
        if self.nl_text is not None and (
            type(self.nl_text) is not str
            or not self.nl_text
            or self.nl_text != self.nl_text.strip()
            or "<think>" in self.nl_text
            or "</think>" in self.nl_text
        ):
            raise CaptionError("caption plan NL text is invalid")
        if (self.nl_text is None) != (self.selected_nl is None):
            raise CaptionError("NL text and selected branch must be present together")
        if self.selected_nl is not None and self.selected_nl not in NL_BRANCHES:
            raise CaptionError("selected NL branch is invalid")


def _seed(seed: int, domain: str) -> int:
    return random.Random(f"{seed}\0{domain}").getrandbits(64)


def _drop(seed: int, domain: str, probability: float) -> bool:
    value = _seed(seed, domain) / 2**64
    return value < probability


def _shuffle_tags(tags: tuple[CaptionTag, ...], seed: int) -> tuple[CaptionTag, ...]:
    shuffled = list(tags)
    random.Random(_seed(seed, "shuffle:all_tags")).shuffle(shuffled)
    return tuple(shuffled)


def _field_tags(fields: CaptionFields, source: TagSource) -> tuple[Tag, ...]:
    if source == "artist":
        return fields.artists
    value = getattr(fields, source)
    return cast(tuple[Tag, ...], value)


def _drop_source_tags(
    fields: CaptionFields,
    source: TagSource,
    *,
    candidate_source_hit: bool,
    probability: float,
    seed: int,
) -> tuple[tuple[Tag, ...], bool]:
    indexed = tuple(enumerate(_field_tags(fields, source)))
    if candidate_source_hit and source in CANDIDATE_SOURCE_FIELDS:
        indexed = tuple(
            (index, tag)
            for index, tag in indexed
            if tag.canonical not in fields.candidate_tags
        )
    retained: list[Tag] = []
    hit = False
    for index, tag in indexed:
        domain = f"dropout:tag:{source}:{index}:{tag.canonical}"
        if _drop(seed, domain, probability):
            hit = True
        else:
            retained.append(tag)
    return tuple(retained), hit


def _caption_tags(source: TagSource, tags: tuple[Tag, ...]) -> tuple[CaptionTag, ...]:
    return tuple(CaptionTag(source=source, tag=tag) for tag in tags)


def _select_style_condition(
    mode: StyleConditionMode,
    *,
    artists: tuple[Tag, ...],
    characters: tuple[Tag, ...],
    seed: int,
) -> StyleCondition | None:
    if mode == "artist":
        return StyleCondition(kind="artist", tags=artists) if artists else None
    if artists and characters:
        kind: StyleConditionKind = (
            "artist"
            if (_seed(seed, "select:style_condition:artist_or_character") & 1) == 0
            else "character"
        )
        selected = artists if kind == "artist" else characters
        return StyleCondition(kind=kind, tags=selected)
    if artists:
        return StyleCondition(kind="artist", tags=artists)
    if characters:
        return StyleCondition(kind="character", tags=characters)
    return None


def build_caption_plan(
    fields: CaptionFields,
    probabilities: CaptionDropoutProbabilities,
    *,
    style_condition_mode: StyleConditionMode,
    seed: int,
) -> CaptionPlan:
    """Apply all-condition, candidate, per-tag, global-shuffle, and NL rules."""

    if type(fields) is not CaptionFields:
        raise CaptionError("caption fields must use the strict CaptionFields type")
    if type(probabilities) is not CaptionDropoutProbabilities:
        raise CaptionError("caption probabilities use an invalid type")
    if (
        type(style_condition_mode) is not str
        or style_condition_mode not in {"artist", "artist_or_character"}
    ):
        raise CaptionError("style condition mode is invalid")
    if type(seed) is not int or seed < 0:
        raise CaptionError("caption seed must be a non-negative integer")
    if _drop(seed, "all_condition", ALL_CONDITION_DROPOUT):
        return CaptionPlan(
            tags=(),
            style_condition=None,
            nl_text=None,
            selected_nl=None,
            all_condition_dropped=True,
            dropout_hits=empty_caption_dropout_hits(all_condition=True),
        )

    candidate_source_hit = _drop(
        seed, "dropout:candidate_source", probabilities.candidate_source
    )
    rating, rating_hit = _drop_source_tags(
        fields,
        "rating",
        candidate_source_hit=candidate_source_hit,
        probability=probabilities.tag,
        seed=seed,
    )
    year, year_hit = _drop_source_tags(
        fields,
        "year",
        candidate_source_hit=candidate_source_hit,
        probability=probabilities.tag,
        seed=seed,
    )
    aesthetic, aesthetic_hit = _drop_source_tags(
        fields,
        "aesthetic",
        candidate_source_hit=candidate_source_hit,
        probability=probabilities.tag,
        seed=seed,
    )
    quality, quality_hit = _drop_source_tags(
        fields,
        "quality",
        candidate_source_hit=candidate_source_hit,
        probability=probabilities.tag,
        seed=seed,
    )
    anime_completeness, anime_completeness_hit = _drop_source_tags(
        fields,
        "anime_completeness",
        candidate_source_hit=candidate_source_hit,
        probability=probabilities.tag,
        seed=seed,
    )
    anime_classification, anime_classification_hit = _drop_source_tags(
        fields,
        "anime_classification",
        candidate_source_hit=candidate_source_hit,
        probability=probabilities.tag,
        seed=seed,
    )
    nsfw, nsfw_hit = _drop_source_tags(
        fields,
        "nsfw",
        candidate_source_hit=candidate_source_hit,
        probability=probabilities.tag,
        seed=seed,
    )
    character, character_hit = _drop_source_tags(
        fields,
        "character",
        candidate_source_hit=candidate_source_hit,
        probability=probabilities.tag,
        seed=seed,
    )
    copyright_tags, copyright_hit = _drop_source_tags(
        fields,
        "copyright",
        candidate_source_hit=candidate_source_hit,
        probability=probabilities.tag,
        seed=seed,
    )
    general, general_hit = _drop_source_tags(
        fields,
        "general",
        candidate_source_hit=candidate_source_hit,
        probability=probabilities.tag,
        seed=seed,
    )
    artists, artist_hit = _drop_source_tags(
        fields,
        "artist",
        candidate_source_hit=candidate_source_hit,
        probability=probabilities.tag,
        seed=seed,
    )

    style_condition = _select_style_condition(
        style_condition_mode,
        artists=artists,
        characters=character,
        seed=seed,
    )
    body_characters = (
        ()
        if style_condition is not None and style_condition.kind == "character"
        else character
    )
    body_artists = (
        artists
        if style_condition is not None and style_condition.kind == "character"
        else ()
    )
    tags = _shuffle_tags(
        (
            *_caption_tags("rating", rating),
            *_caption_tags("year", year),
            *_caption_tags("aesthetic", aesthetic),
            *_caption_tags("quality", quality),
            *_caption_tags("anime_completeness", anime_completeness),
            *_caption_tags("anime_classification", anime_classification),
            *_caption_tags("nsfw", nsfw),
            *_caption_tags("character", body_characters),
            *_caption_tags("copyright", copyright_tags),
            *_caption_tags("general", general),
            *_caption_tags("artist", body_artists),
        ),
        seed,
    )

    nl_hits = {
        branch: _drop(
            seed,
            f"dropout:nl:{branch}",
            getattr(probabilities.nl, branch),
        )
        for branch in NL_BRANCHES
    }
    available_nl: tuple[tuple[NlBranch, str], ...] = tuple(
        (branch, text) for branch, text in fields.nl.available() if not nl_hits[branch]
    )
    if available_nl:
        selected_nl, nl_text = available_nl[
            random.Random(_seed(seed, "select:nl")).randrange(len(available_nl))
        ]
    else:
        selected_nl, nl_text = None, None

    return CaptionPlan(
        tags=tags,
        style_condition=style_condition,
        nl_text=nl_text,
        selected_nl=selected_nl,
        all_condition_dropped=False,
        dropout_hits=CaptionDropoutHits(
            all_condition=False,
            rating=rating_hit,
            year=year_hit,
            aesthetic=aesthetic_hit,
            quality=quality_hit,
            anime_completeness=anime_completeness_hit,
            anime_classification=anime_classification_hit,
            nsfw=nsfw_hit,
            character=character_hit,
            copyright=copyright_hit,
            general=general_hit,
            artist=artist_hit,
            candidate_source=candidate_source_hit,
            long_names=nl_hits["long_names"],
            long_no_names=nl_hits["long_no_names"],
            short_vibes=nl_hits["short_vibes"],
            nl2=nl_hits["nl2"],
            nl3=nl_hits["nl3"],
        ),
    )
