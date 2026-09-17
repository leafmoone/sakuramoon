# pyright: reportPrivateUsage=false

from __future__ import annotations

import random

import pytest
from pydantic import ValidationError

from sakuramoon.config.schema import TrainingSamplingConfig
from sakuramoon.data.caption import (
    CaptionPlan,
    CaptionTag,
    ConditionRequest,
    Tag,
    empty_caption_dropout_hits,
)
from sakuramoon.data.serialize import (
    MAIN_SUFFIX,
    SYSTEM_PREFIX,
    FramingContract,
    serialize_caption,
)
from sakuramoon.train import sampling


class _Tokenizer:
    pad_token_id = 248044

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        if text == SYSTEM_PREFIX:
            return list(range(100, 134))
        if text == MAIN_SUFFIX:
            return list(range(200, 205))
        return [1000 + index for index, _character in enumerate(text)]


def _prompt(
    label: str,
    *,
    condition: str | None = "cond",
    height: int = 512,
    width: int = 512,
) -> sampling._PostDropoutPrompt:
    if condition is None:
        # A fully dropped sample: the plan is empty and flagged as such.
        plan = CaptionPlan(
            tags=(),
            condition=None,
            nl_text=None,
            selected_nl=None,
            all_condition_dropped=True,
            dropout_hits=empty_caption_dropout_hits(all_condition=True),
        )
    else:
        plan = CaptionPlan(
            tags=(
                CaptionTag("general", Tag(f"subject_{label}", f"subject_{label}")),
                CaptionTag("year", Tag("year 2026", "year 2026")),
            ),
            condition=ConditionRequest(
                source="artist_text",
                role="style",
                tags=(Tag(condition, condition),),
            ),
            nl_text=f"lighting {label}",
            selected_nl="short_vibes",
            all_condition_dropped=False,
            dropout_hits=empty_caption_dropout_hits(),
        )
    framing = FramingContract(34, 5, _Tokenizer.pad_token_id)
    return sampling._PostDropoutPrompt(
        sample_id=f"sample-{label}",
        caption=serialize_caption(plan, _Tokenizer(), framing),
        plan=plan,
        observed_height=height,
        observed_width=width,
    )


class TestRawSchema:
    """Config-level invariants for the raw cohort count."""

    def test_default_is_disabled(self) -> None:
        config = TrainingSamplingConfig()
        assert config.raw_image_count == 0

    def test_disabled_does_not_disturb_cohort_consistency(self) -> None:
        config = TrainingSamplingConfig(
            image_count=60,
            fixed_cohort="locked",
            raw_image_count=0,
        )
        assert config.raw_image_count == 0

    def test_max_count_is_accepted(self) -> None:
        config = TrainingSamplingConfig(raw_image_count=12)
        assert config.raw_image_count == 12

    def test_count_above_variant_count_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            TrainingSamplingConfig(raw_image_count=13)

    def test_negative_count_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            TrainingSamplingConfig(raw_image_count=-1)


class TestSelectRawCandidates:
    """Pure-function coverage of the deterministic raw selection."""

    def _candidates(self) -> tuple[sampling._PostDropoutPrompt, ...]:
        return (
            _prompt("a", condition="cond_a"),
            _prompt("b", condition="cond_b", height=384, width=640),
            _prompt("c", condition=None),  # fully dropped: ineligible
            _prompt("d", condition="cond_d"),
            _prompt("e", condition="cond_e"),
            _prompt("f", condition="cond_f"),
            _prompt("g", condition="cond_g"),
            _prompt("h", condition="cond_h"),
            _prompt("i", condition="cond_i"),
            _prompt("j", condition="cond_j"),
            _prompt("k", condition="cond_k"),
            _prompt("l", condition="cond_l"),
            _prompt("m", condition="cond_m"),
        )

    def test_selects_exactly_count_distinct_conditioned(self) -> None:
        candidates = self._candidates()
        selector = random.Random("sakuramoon\0training-sample-raw\062000")
        selected = sampling._select_raw_candidates(candidates, selector, 12)
        assert len(selected) == 12
        assert len({item.sample_id for item in selected}) == 12
        assert all(item.plan.condition is not None for item in selected)
        # The fully-dropped sample is never eligible.
        assert all(item.sample_id != "sample-c" for item in selected)

    def test_selection_is_deterministic_for_seed_and_pool(self) -> None:
        candidates = self._candidates()
        first = sampling._select_raw_candidates(
            candidates, random.Random("seed-a"), 5
        )
        second = sampling._select_raw_candidates(
            candidates, random.Random("seed-a"), 5
        )
        assert first == second
        other = sampling._select_raw_candidates(
            candidates, random.Random("seed-b"), 5
        )
        assert first != other

    def test_insufficient_conditioned_candidates_raise(self) -> None:
        candidates = (
            _prompt("a", condition="cond_a"),
            _prompt("b", condition=None),
        )
        with pytest.raises(sampling.TrainingSamplingError):
            sampling._select_raw_candidates(
                candidates, random.Random("x"), 3
            )

    def test_invalid_count_raises(self) -> None:
        candidates = self._candidates()
        with pytest.raises(sampling.TrainingSamplingError):
            sampling._select_raw_candidates(
                candidates, random.Random("x"), 0
            )
        with pytest.raises(sampling.TrainingSamplingError):
            sampling._select_raw_candidates(
                candidates, random.Random("x"), 13
            )


class TestBuildRawItems:
    """Raw items carry the real prompt at the observed training shape."""

    def test_preserves_observed_shape_and_canonical_geometry(self) -> None:
        selected = (
            _prompt("a", condition="cond_a", height=384, width=640),
            _prompt("b", condition="cond_b", height=512, width=512),
        )
        items = sampling._build_raw_items(selected, start_ordinal=60)
        assert len(items) == 2
        assert items[0].height == 384
        assert items[0].width == 640
        assert items[1].height == 512
        assert items[1].width == 512
        assert (items[0].ordinal, items[1].ordinal) == (60, 61)
        for item in items:
            assert item.variant == "raw"
            assert item.zoom == 1.0
            assert item.coordinate_type == "canonical_full_canvas"
            assert item.virtual_canvas_size == (item.height, item.width)
            assert item.crop_box == (0, 0, item.width, item.height)
            assert item.main_source == "A"
            assert item.condition_sources == ("A",)

    def test_keeps_the_post_dropout_caption_untouched(self) -> None:
        selected = (_prompt("a", condition="cond_a"),)
        (item,) = sampling._build_raw_items(selected, start_ordinal=0)
        assert item.caption is selected[0].caption
        assert item.plan is selected[0].plan
        assert item.sample_id == "sample-a"


class TestRawCohortConstants:
    """The raw variant name is part of the item type but not of the
    12-variant pair layout."""

    def test_raw_is_not_in_the_pair_variant_layout(self) -> None:
        assert "raw" not in sampling._VARIANT_NAMES
        assert len(sampling._VARIANT_NAMES) == sampling._VARIANT_COUNT

    def test_raw_is_accepted_as_variant_name(self) -> None:
        item_variant: sampling.VariantName = "raw"
        assert item_variant == "raw"
