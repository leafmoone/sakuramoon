from __future__ import annotations

from dataclasses import replace

import pytest

from sakuramoon.data.caption import (
    ALL_CONDITION_DROPOUT,
    BODY_TAG_SOURCE_ORDER,
    CAPTION_DROPOUT_KEYS,
    CaptionDropoutProbabilities,
    CaptionError,
    CaptionFields,
    CaptionPlan,
    CaptionTag,
    ConditionRequest,
    NlCandidates,
    NlDropoutProbabilities,
    Tag,
    empty_caption_dropout_hits,
)
from sakuramoon.data.caption import build_caption_plan as _build_caption_plan


def _nl_probabilities(value: float = 0.0) -> NlDropoutProbabilities:
    return NlDropoutProbabilities(value, value, value, value, value)


def _probabilities(
    *, tag: float = 0.0, candidate_source: float = 0.0, nl: float = 0.0
) -> CaptionDropoutProbabilities:
    return CaptionDropoutProbabilities(
        tag=tag,
        candidate_source=candidate_source,
        nl=_nl_probabilities(nl),
    )


def build_caption_plan(
    fields: CaptionFields,
    probabilities: CaptionDropoutProbabilities,
    *,
    seed: int,
    condition_mode: str = "artist",
) -> CaptionPlan:
    return _build_caption_plan(
        fields,
        probabilities,
        condition_mode=condition_mode,  # pyright: ignore[reportArgumentType]
        seed=seed,
    )


def _fields() -> CaptionFields:
    return CaptionFields(
        nsfw=(Tag("nsfw", "nsfw"),),
        character=(
            Tag("alice", "alice"),
            Tag("candidate_character", "candidate_character"),
        ),
        copyright=(Tag("wonderland", "wonderland"),),
        general=(
            Tag("blue_dress", "blue_dress"),
            Tag("candidate_general", "candidate_general"),
            Tag("soft_light", "soft_light"),
        ),
        artists=(
            Tag("sample_artist", "sample_artist"),
            Tag("candidate_general", "candidate_general"),
        ),
        candidate_tags=frozenset({"candidate_character", "candidate_general"}),
        nl=NlCandidates(
            long_names="A detailed scene.",
            long_no_names=None,
            short_vibes="soft light",
            nl2=None,
            nl3=None,
        ),
        rating=(Tag("safe", "safe"),),
        year=(Tag("year 2026", "year 2026"), Tag("newest", "newest")),
        aesthetic=(Tag("masterpiece", "masterpiece"),),
        quality=(Tag("best", "best"),),
        anime_completeness=(Tag("polished", "polished"),),
        anime_classification=(Tag("illustration", "illustration"),),
    )


def _seed_for_global_dropout(expected: bool) -> int:
    for seed in range(10_000):
        plan = build_caption_plan(_fields(), _probabilities(), seed=seed)
        if plan.all_condition_dropped is expected:
            return seed
    raise AssertionError("test could not find deterministic global dropout seed")


def _seed_for_partial_general_dropout() -> int:
    source_count = len(_fields().general)
    for seed in range(10_000):
        plan = build_caption_plan(_fields(), _probabilities(tag=0.5), seed=seed)
        retained = tuple(item for item in plan.tags if item.source == "general")
        if not plan.all_condition_dropped and 0 < len(retained) < source_count:
            return seed
    raise AssertionError("test could not find a partial per-tag dropout seed")


def test_all_condition_probability_is_fixed_and_produces_only_global_hit() -> None:
    assert ALL_CONDITION_DROPOUT == 0.10
    plan = build_caption_plan(
        _fields(), _probabilities(), seed=_seed_for_global_dropout(True)
    )

    assert plan.all_condition_dropped is True
    assert plan.tags == ()
    assert plan.condition is None
    assert plan.nl_text is None
    assert tuple(plan.dropout_hits.as_mapping()) == CAPTION_DROPOUT_KEYS
    assert plan.dropout_hits.as_mapping() == {
        key: key == "all_condition" for key in CAPTION_DROPOUT_KEYS
    }


def test_all_non_artist_tags_are_globally_shuffled_deterministically() -> None:
    fields = _fields()
    expected = {
        (source, tag.canonical)
        for source in BODY_TAG_SOURCE_ORDER
        for tag in getattr(fields, source)
    }
    selected: tuple[CaptionTag, ...] | None = None
    grouped_sources = tuple(
        source for source in BODY_TAG_SOURCE_ORDER for _tag in getattr(fields, source)
    )
    for seed in range(10_000):
        plan = build_caption_plan(fields, _probabilities(), seed=seed)
        if (
            not plan.all_condition_dropped
            and tuple(item.source for item in plan.tags) != grouped_sources
        ):
            selected = plan.tags
            repeated = build_caption_plan(fields, _probabilities(), seed=seed)
            assert repeated.tags == selected
            break

    assert selected is not None
    assert {(item.source, item.tag.canonical) for item in selected} == expected


def test_each_tag_drops_independently_within_one_source() -> None:
    fields = _fields()
    seed = _seed_for_partial_general_dropout()
    plan = build_caption_plan(fields, _probabilities(tag=0.5), seed=seed)
    retained = tuple(item.tag for item in plan.tags if item.source == "general")

    assert 0 < len(retained) < len(fields.general)
    assert plan.dropout_hits.general is True


def test_unified_tag_probability_applies_to_every_non_nl_field() -> None:
    plan = build_caption_plan(
        _fields(),
        _probabilities(tag=1.0),
        seed=_seed_for_global_dropout(False),
    )

    assert plan.tags == ()
    assert plan.condition is None
    assert plan.nl_text in {"A detailed scene.", "soft light"}
    hits = plan.dropout_hits.as_mapping()
    assert all(hits[source] for source in (*BODY_TAG_SOURCE_ORDER, "artist"))
    assert hits["candidate_source"] is False


def test_candidate_source_uses_canonical_ids_and_never_deletes_artist() -> None:
    plan = build_caption_plan(
        _fields(),
        _probabilities(candidate_source=1.0),
        seed=_seed_for_global_dropout(False),
    )
    retained = {(item.source, item.tag.canonical) for item in plan.tags}

    assert ("character", "candidate_character") not in retained
    assert ("general", "candidate_general") not in retained
    assert ("general", "blue_dress") in retained
    assert plan.condition is not None
    assert plan.condition.source == "artist_text"
    assert plan.condition.role == "style"
    assert tuple(tag.canonical for tag in plan.condition.tags) == (
        "sample_artist",
        "candidate_general",
    )
    assert plan.dropout_hits.candidate_source is True


def test_artist_order_is_fixed_after_independent_dropout() -> None:
    fields = _fields()
    plan = build_caption_plan(
        fields, _probabilities(), seed=_seed_for_global_dropout(False)
    )

    assert plan.condition == ConditionRequest(
        source="artist_text",
        role="style",
        tags=fields.artists,
    )


def test_artist_or_character_routing_is_deterministic_and_complementary() -> None:
    fields = _fields()
    expected = {
        (source, tag.canonical)
        for source in BODY_TAG_SOURCE_ORDER
        for tag in getattr(fields, source)
    } | {("artist", tag.canonical) for tag in fields.artists}
    selected_sources: set[str] = set()

    for seed in range(10_000):
        plan = build_caption_plan(
            fields,
            _probabilities(),
            condition_mode="artist_or_character",
            seed=seed,
        )
        if plan.all_condition_dropped:
            continue
        assert plan.condition is not None
        assert plan == build_caption_plan(
            fields,
            _probabilities(),
            condition_mode="artist_or_character",
            seed=seed,
        )
        condition = plan.condition
        selected_sources.add(condition.source)
        routed = {(item.source, item.tag.canonical) for item in plan.tags}
        condition_tag_source = (
            "artist" if condition.source == "artist_text" else "character"
        )
        routed.update((condition_tag_source, tag.canonical) for tag in condition.tags)
        assert routed == expected
        assert len(plan.tags) + len(condition.tags) == len(expected)
        if condition.source == "artist_text":
            assert condition.role == "style"
            assert not any(item.source == "artist" for item in plan.tags)
            assert {
                item.tag.canonical for item in plan.tags if item.source == "character"
            } == {tag.canonical for tag in fields.character}
        else:
            assert condition.role == "identity"
            assert not any(item.source == "character" for item in plan.tags)
            assert {
                item.tag.canonical for item in plan.tags if item.source == "artist"
            } == {tag.canonical for tag in fields.artists}
        if selected_sources == {"artist_text", "character_text"}:
            break

    assert selected_sources == {"artist_text", "character_text"}


@pytest.mark.parametrize(
    ("fields", "expected_source"),
    [
        (replace(_fields(), character=()), "artist_text"),
        (replace(_fields(), artists=()), "character_text"),
        (replace(_fields(), character=(), artists=()), None),
    ],
)
def test_artist_or_character_routes_only_available_source(
    fields: CaptionFields, expected_source: str | None
) -> None:
    plan = build_caption_plan(
        fields,
        _probabilities(),
        condition_mode="artist_or_character",
        seed=_seed_for_global_dropout(False),
    )

    assert (
        None if plan.condition is None else plan.condition.source
    ) == expected_source
    if expected_source is not None:
        selected_tag_source = (
            "artist" if expected_source == "artist_text" else "character"
        )
        assert not any(item.source == selected_tag_source for item in plan.tags)


def test_nl_selects_at_most_one_available_complete_branch() -> None:
    plan = build_caption_plan(
        _fields(), _probabilities(), seed=_seed_for_global_dropout(False)
    )

    assert plan.selected_nl in {"long_names", "short_vibes"}
    assert plan.nl_text in {"A detailed scene.", "soft light"}


def test_explicit_nl_dropout_removes_all_available_branches() -> None:
    plan = build_caption_plan(
        _fields(),
        _probabilities(nl=1.0),
        seed=_seed_for_global_dropout(False),
    )

    assert plan.selected_nl is None
    assert plan.nl_text is None
    assert all(
        getattr(plan.dropout_hits, branch)
        for branch in ("long_names", "long_no_names", "short_vibes", "nl2", "nl3")
    )


@pytest.mark.parametrize(
    "candidate_tags",
    [
        frozenset({""}),
        frozenset({" candidate"}),
        frozenset({"candidate "}),
        frozenset({1}),
        {"candidate"},
    ],
)
def test_candidate_deletion_ids_require_exact_canonical_boundaries(
    candidate_tags: object,
) -> None:
    fields = _fields()
    with pytest.raises(CaptionError, match="candidate deletion IDs"):
        CaptionFields(
            nsfw=fields.nsfw,
            character=fields.character,
            copyright=fields.copyright,
            general=fields.general,
            artists=fields.artists,
            candidate_tags=candidate_tags,  # pyright: ignore[reportArgumentType]
            nl=fields.nl,
        )


def test_five_nl_probabilities_must_remain_equal() -> None:
    with pytest.raises(CaptionError, match="must be equal"):
        CaptionDropoutProbabilities(
            tag=0.1,
            candidate_source=0.3,
            nl=NlDropoutProbabilities(0.1, 0.2, 0.1, 0.1, 0.1),
        )


@pytest.mark.parametrize("value", [-0.1, 1.1, 0, True])
def test_probabilities_require_explicit_valid_floats(value: object) -> None:
    with pytest.raises(CaptionError, match="explicit floats"):
        CaptionDropoutProbabilities(
            tag=value,  # pyright: ignore[reportArgumentType]
            candidate_source=0.3,
            nl=_nl_probabilities(),
        )


@pytest.mark.parametrize(
    "text",
    [
        "",
        " bad",
        "bad, boundary",
        "line\nbreak",
        "line\rbreak",
        "nul\0tag",
        "<think>bad</think>",
    ],
)
def test_tag_boundaries_and_thinking_markers_are_rejected(text: str) -> None:
    with pytest.raises(CaptionError):
        Tag(text, "canonical")


@pytest.mark.parametrize(
    "canonical", ["", " bad", "bad ", "line\nbreak", "line\rbreak", "nul\0id"]
)
def test_tag_canonical_ids_must_be_trim_stable(canonical: str) -> None:
    with pytest.raises(CaptionError, match="boundaries"):
        Tag("valid", canonical)


def test_nl_candidates_reject_non_text_values_immediately() -> None:
    with pytest.raises(CaptionError, match="text or null"):
        NlCandidates(1, None, None, None, None)  # pyright: ignore[reportArgumentType]


def test_caption_probabilities_reject_invalid_nl_container_immediately() -> None:
    with pytest.raises(CaptionError, match="invalid type"):
        CaptionDropoutProbabilities(
            tag=0.1,
            candidate_source=0.3,
            nl=object(),  # pyright: ignore[reportArgumentType]
        )


@pytest.mark.parametrize("seed", [-1, True, "1"])
def test_caption_seed_requires_a_non_negative_integer(seed: object) -> None:
    with pytest.raises(CaptionError, match="caption seed"):
        build_caption_plan(
            _fields(),
            _probabilities(),
            seed=seed,  # pyright: ignore[reportArgumentType]
        )


@pytest.mark.parametrize("mode", ["", "artist_or_character ", "character", True, 1])
def test_condition_mode_is_strict(mode: object) -> None:
    with pytest.raises(CaptionError, match="condition mode"):
        _build_caption_plan(
            _fields(),
            _probabilities(),
            condition_mode=mode,  # pyright: ignore[reportArgumentType]
            seed=_seed_for_global_dropout(False),
        )


def test_nl_thinking_markers_are_rejected_when_consumed() -> None:
    fields = _fields()
    invalid = CaptionFields(
        nsfw=fields.nsfw,
        character=fields.character,
        copyright=fields.copyright,
        general=fields.general,
        artists=fields.artists,
        candidate_tags=fields.candidate_tags,
        nl=NlCandidates("<think>hidden</think>", None, None, None, None),
        rating=fields.rating,
        year=fields.year,
        aesthetic=fields.aesthetic,
        quality=fields.quality,
        anime_completeness=fields.anime_completeness,
        anime_classification=fields.anime_classification,
    )

    with pytest.raises(CaptionError, match="thinking"):
        build_caption_plan(
            invalid, _probabilities(), seed=_seed_for_global_dropout(False)
        )


def test_all_condition_plan_cannot_carry_content() -> None:
    with pytest.raises(CaptionError, match="must be empty"):
        CaptionPlan(
            tags=(CaptionTag("rating", Tag("safe", "safe")),),
            condition=None,
            nl_text=None,
            selected_nl=None,
            all_condition_dropped=True,
            dropout_hits=empty_caption_dropout_hits(all_condition=True),
        )
