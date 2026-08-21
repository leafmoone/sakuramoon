# pyright: reportPrivateUsage=false

from __future__ import annotations

import pytest
from pydantic import ValidationError

from sakuramoon.config.schema import TrainingSamplingConfig
from sakuramoon.train.sampling import (
    _LOCKED_FIXED_PAIR_COUNT,
    _LOCKED_TOTAL_VARIANT_COUNT,
    _TOTAL_VARIANT_COUNT,
    _VARIANT_COUNT,
    _pinned_selector_update,
)


class TestPinnedSelectorUpdate:
    """Pure-function coverage for the longitudinal dynamic selector pin."""

    def test_update_below_pin_is_not_pinned(self) -> None:
        assert _pinned_selector_update(61_000, 62_000) is None

    def test_update_at_pin_is_pinned(self) -> None:
        assert _pinned_selector_update(62_000, 62_000) == 62_000

    def test_update_above_pin_is_pinned(self) -> None:
        assert _pinned_selector_update(62_800, 62_000) == 62_000

    def test_no_pin_never_pins(self) -> None:
        assert _pinned_selector_update(62_800, None) is None


class TestLockedCohortConstants:
    """The locked fixed cohort expands to 4 pairs of 12 variants each."""

    def test_locked_total_state_count_is_60(self) -> None:
        assert _LOCKED_FIXED_PAIR_COUNT == 4
        assert _LOCKED_TOTAL_VARIANT_COUNT == 60
        # 12 dynamic + 4 * 12 locked fixed.
        assert _LOCKED_TOTAL_VARIANT_COUNT == (
            _VARIANT_COUNT + _LOCKED_FIXED_PAIR_COUNT * _VARIANT_COUNT
        )

    def test_locked_total_is_larger_than_neutral_total(self) -> None:
        assert _LOCKED_TOTAL_VARIANT_COUNT > _TOTAL_VARIANT_COUNT


class TestSchemaCohortConsistency:
    """Config-level invariants for the new fixed_cohort / pin knobs."""

    def test_defaults_are_neutral_and_unpinned(self) -> None:
        config = TrainingSamplingConfig()
        assert config.fixed_cohort == "neutral"
        assert config.longitudinal_pin_update is None
        assert config.image_count == 12

    def test_locked_cohort_requires_image_count_60(self) -> None:
        config = TrainingSamplingConfig(
            image_count=60,
            fixed_cohort="locked",
            longitudinal_pin_update=62_000,
        )
        assert config.fixed_cohort == "locked"
        assert config.image_count == 60
        assert config.longitudinal_pin_update == 62_000

    def test_locked_cohort_rejects_mismatched_image_count(self) -> None:
        with pytest.raises(ValidationError):
            TrainingSamplingConfig(image_count=24, fixed_cohort="locked")

    def test_neutral_cohort_allows_single_cohort_count(self) -> None:
        config = TrainingSamplingConfig(image_count=12, fixed_cohort="neutral")
        assert config.fixed_cohort == "neutral"

    def test_pin_must_be_positive_or_none(self) -> None:
        with pytest.raises(ValidationError):
            TrainingSamplingConfig(longitudinal_pin_update=0)
