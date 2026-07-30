from __future__ import annotations

import pytest

from sakuramoon.data.caption import (
    ALL_CONDITION_DROPOUT,
    CaptionDropoutProbabilities,
    CaptionError,
    CaptionFields,
    CaptionPlan,
    NlCandidates,
    NlDropoutProbabilities,
    Tag,
    build_caption_plan,
)


def _nl_probabilities(value: float = 0.0) -> NlDropoutProbabilities:
    return NlDropoutProbabilities(value, value, value, value, value)


def _probabilities(**changes: float) -> CaptionDropoutProbabilities:
    values = {
        "nsfw": 0.0,
        "character": 0.0,
        "copyright": 0.0,
        "general": 0.0,
        "artist": 0.0,
        "candidate_source": 0.0,
    }
    values.update(changes)
    return CaptionDropoutProbabilities(
        nsfw=values["nsfw"],
        character=values["character"],
        copyright=values["copyright"],
        general=values["general"],
        artist=values["artist"],
        candidate_source=values["candidate_source"],
        nl=_nl_probabilities(),
    )


def _fields() -> CaptionFields:
    return CaptionFields(
        nsfw=(Tag("safe", "rating:safe"),),
        character=(Tag("alice", "character:alice"),),
        copyright=(Tag("wonderland", "copyright:wonderland"),),
        general=(Tag("blue dress", "general:blue_dress"),),
        artists=(Tag("sample artist", "artist:sample"),),
        candidate_tags=frozenset({"character:alice", "general:blue_dress"}),
        nl=NlCandidates(
            long_names="A detailed scene.",
            long_no_names=None,
            short_vibes="soft light",
            nl2=None,
            nl3=None,
        ),
    )


def _seed_for_global_dropout(expected: bool) -> int:
    for seed in range(1000):
        plan = build_caption_plan(_fields(), _probabilities(), seed=seed)
        if plan.all_condition_dropped is expected:
            return seed
    raise AssertionError("test could not find deterministic global dropout seed")


def test_all_condition_probability_is_fixed_and_produces_empty_plan() -> None:
    assert ALL_CONDITION_DROPOUT == 0.10
    plan = build_caption_plan(
        _fields(), _probabilities(), seed=_seed_for_global_dropout(True)
    )
    assert plan.all_condition_dropped is True
    assert plan.nsfw == plan.character == plan.copyright == plan.general == ()
    assert plan.artists == ()
    assert plan.nl_text is None


def test_category_order_data_and_internal_shuffle_are_deterministic() -> None:
    fields = _fields()
    fields = CaptionFields(
        nsfw=fields.nsfw,
        character=fields.character,
        copyright=fields.copyright,
        general=(
            Tag("first", "general:first"),
            Tag("second", "general:second"),
            Tag("third", "general:third"),
        ),
        artists=fields.artists,
        candidate_tags=fields.candidate_tags,
        nl=fields.nl,
    )
    seed = _seed_for_global_dropout(False)
    first = build_caption_plan(fields, _probabilities(), seed=seed)
    repeated = build_caption_plan(fields, _probabilities(), seed=seed)
    assert first == repeated
    assert set(first.general) == set(fields.general)


def test_candidate_dropout_deletes_canonical_matches_only_from_tag_categories() -> None:
    plan = build_caption_plan(
        _fields(),
        _probabilities(candidate_source=1.0),
        seed=_seed_for_global_dropout(False),
    )
    assert plan.character == ()
    assert plan.general == ()
    assert plan.nsfw and plan.copyright
    assert plan.artists
    assert plan.nl_text is not None


def test_explicit_category_and_artist_dropout() -> None:
    plan = build_caption_plan(
        _fields(),
        _probabilities(nsfw=1.0, general=1.0, artist=1.0),
        seed=_seed_for_global_dropout(False),
    )
    assert plan.nsfw == ()
    assert plan.general == ()
    assert plan.character and plan.copyright
    assert plan.artists == ()


def test_nl_selects_at_most_one_currently_available_branch() -> None:
    plan = build_caption_plan(
        _fields(), _probabilities(), seed=_seed_for_global_dropout(False)
    )
    assert plan.selected_nl in {"long_names", "short_vibes"}
    assert plan.nl_text in {"A detailed scene.", "soft light"}


def test_explicit_nl_dropout_removes_all_available_branches() -> None:
    probabilities = CaptionDropoutProbabilities(
        nsfw=0.0,
        character=0.0,
        copyright=0.0,
        general=0.0,
        artist=0.0,
        candidate_source=0.0,
        nl=NlDropoutProbabilities(1.0, 1.0, 1.0, 1.0, 1.0),
    )
    plan = build_caption_plan(
        _fields(), probabilities, seed=_seed_for_global_dropout(False)
    )
    assert plan.selected_nl is None
    assert plan.nl_text is None


def test_five_nl_probabilities_must_remain_equal() -> None:
    with pytest.raises(CaptionError, match="must be equal"):
        CaptionDropoutProbabilities(
            nsfw=0.0,
            character=0.0,
            copyright=0.0,
            general=0.0,
            artist=0.0,
            candidate_source=0.0,
            nl=NlDropoutProbabilities(0.1, 0.2, 0.1, 0.1, 0.1),
        )


@pytest.mark.parametrize("value", [-0.1, 1.1, 0, True])
def test_unresolved_probabilities_require_explicit_valid_floats(value: object) -> None:
    with pytest.raises(CaptionError, match="explicit floats"):
        CaptionDropoutProbabilities(
            nsfw=value,  # type: ignore[arg-type]
            character=0.0,
            copyright=0.0,
            general=0.0,
            artist=0.0,
            candidate_source=0.0,
            nl=_nl_probabilities(),
        )


@pytest.mark.parametrize(
    "text", ["", " bad", "bad, boundary", "line\nbreak", "<think>bad</think>"]
)
def test_tag_boundaries_and_thinking_markers_are_rejected(text: str) -> None:
    with pytest.raises(CaptionError):
        Tag(text, "canonical")


def test_nl_thinking_markers_are_rejected_when_consumed() -> None:
    fields = _fields()
    fields = CaptionFields(
        nsfw=fields.nsfw,
        character=fields.character,
        copyright=fields.copyright,
        general=fields.general,
        artists=fields.artists,
        candidate_tags=fields.candidate_tags,
        nl=NlCandidates("<think>hidden</think>", None, None, None, None),
    )
    with pytest.raises(CaptionError, match="thinking"):
        build_caption_plan(fields, _probabilities(), seed=_seed_for_global_dropout(False))


def test_all_condition_plan_cannot_carry_content() -> None:
    with pytest.raises(CaptionError, match="must be empty"):
        CaptionPlan(
            nsfw=(Tag("safe", "rating:safe"),),
            character=(),
            copyright=(),
            general=(),
            artists=(),
            nl_text=None,
            selected_nl=None,
            all_condition_dropped=True,
        )
