"""Database-derived fixed prompt cohort for periodic training samples."""

from __future__ import annotations

from dataclasses import dataclass

from sakuramoon.data.caption import (
    CaptionPlan,
    CaptionTag,
    ConditionRequest,
    Tag,
    empty_caption_dropout_hits,
)


@dataclass(frozen=True, slots=True)
class FixedPromptRecord:
    """A frozen prompt projection of one high-favorite database record."""

    sample_id: str
    post_id: int
    artist: str
    fav_count: int
    score: int
    rating: str
    year: str
    aesthetic: str
    quality: str
    anime_completeness: str
    anime_classification: str
    nsfw: str
    character: tuple[str, ...]
    copyright: tuple[str, ...]
    general: tuple[str, ...]

    def caption_plan(self) -> CaptionPlan:
        category_values = (
            ("rating", (self.rating,)),
            ("year", (self.year,)),
            ("aesthetic", (self.aesthetic,)),
            ("quality", (self.quality,)),
            ("anime_completeness", (self.anime_completeness,)),
            ("anime_classification", (self.anime_classification,)),
            ("nsfw", (self.nsfw,)),
            ("character", self.character),
            ("copyright", self.copyright),
            ("general", self.general),
        )
        tags = tuple(
            CaptionTag(source, Tag(text, text))
            for source, values in category_values
            for text in values
        )
        return CaptionPlan(
            tags=tags,
            condition=ConditionRequest(
                source="artist_text",
                role="style",
                tags=(Tag(self.artist, self.artist),),
            ),
            nl_text=None,
            selected_nl=None,
            all_condition_dropped=False,
            dropout_hits=empty_caption_dropout_hits(),
        )


# Highest-favorite general-rated records returned by the Danbooru database at
# selection time. The normalized quality labels are retained in the prompts.
FIXED_NEUTRAL_PROMPTS: tuple[FixedPromptRecord, FixedPromptRecord] = (
    FixedPromptRecord(
        sample_id="danbooru:3013073",
        post_id=3013073,
        artist="hiten_(hitenkei)",
        fav_count=191,
        score=142,
        rating="general",
        year="year 2018",
        aesthetic="excellent",
        quality="best",
        anime_completeness="polished",
        anime_classification="illustration",
        nsfw="sfw",
        character=("kagurazaka_reina",),
        copyright=("original",),
        general=(
            "1girl",
            "backlighting",
            "blonde_hair",
            "blue_eyes",
            "blush",
            "cafe",
            "coffee",
            "cup",
            "depth_of_field",
            "indoors",
            "long_hair",
            "looking_outside",
            "looking_to_the_side",
            "off-shoulder_sweater",
            "sitting",
            "snow",
            "solo",
            "sweater",
            "teacup",
            "window",
            "wistful",
        ),
    ),
    FixedPromptRecord(
        sample_id="danbooru:3444635",
        post_id=3444635,
        artist="wlop",
        fav_count=172,
        score=107,
        rating="general",
        year="year 2019",
        aesthetic="excellent",
        quality="best",
        anime_completeness="polished",
        anime_classification="illustration",
        nsfw="sfw",
        character=(
            "2b_(nier:automata)",
            "lady_maria_of_the_astral_clocktower",
            "lunafreya_nox_fleuret",
        ),
        copyright=(
            "bloodborne",
            "final_fantasy",
            "final_fantasy_xv",
            "nier:automata",
            "nier_(series)",
        ),
        general=(
            "3girls",
            "armchair",
            "black_dress",
            "blonde_hair",
            "boots",
            "closed_eyes",
            "crown_braid",
            "dress",
            "earrings",
            "high_heels",
            "jewelry",
            "multiple_girls",
            "short_hair",
            "sitting",
            "sword",
            "white_dress",
            "white_hair",
            "wine_glass",
        ),
    ),
)

FIXED_NEUTRAL_SHARED_SEED = 6632148857128391041


def fixed_neutral_provenance() -> dict[str, object]:
    return {
        "selection": "highest-fav-count general-rated record per artist",
        "source": "danbooru",
        "records": [
            {
                "sample_id": record.sample_id,
                "post_id": record.post_id,
                "artist": record.artist,
                "fav_count": record.fav_count,
                "score": record.score,
                "rating": record.rating,
            }
            for record in FIXED_NEUTRAL_PROMPTS
        ],
    }


__all__ = [
    "FIXED_NEUTRAL_PROMPTS",
    "FIXED_NEUTRAL_SHARED_SEED",
    "FixedPromptRecord",
    "fixed_neutral_provenance",
]
