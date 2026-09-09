# pyright: reportPrivateUsage=false

from __future__ import annotations

import pytest
from pydantic import ValidationError

from sakuramoon.config.schema import TrainingSamplingConfig
from sakuramoon.train.sampling import (
    _LOCKED_FIXED_PAIR_COUNT,
    _LOCKED_TOTAL_VARIANT_COUNT,
    _VARIANT_COUNT,
    _locked_cohort_for,
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

    def test_locked_total_is_larger_than_dynamic_cohort(self) -> None:
        assert _LOCKED_TOTAL_VARIANT_COUNT > _VARIANT_COUNT


class TestSchemaCohortConsistency:
    """Config-level invariants for the new fixed_cohort / pin knobs."""

    def test_defaults_are_none_cohort_and_unpinned(self) -> None:
        config = TrainingSamplingConfig()
        assert config.fixed_cohort == "none"
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

    def test_none_cohort_allows_single_cohort_count(self) -> None:
        config = TrainingSamplingConfig(image_count=12, fixed_cohort="none")
        assert config.fixed_cohort == "none"

    @pytest.mark.parametrize(
        ("fixed_cohort", "image_count"),
        [("none", 60), ("none", 24), ("locked", 12), ("locked", 24), ("none", 1)],
    )
    def test_cohort_count_mapping_rejects_every_mismatch(
        self, fixed_cohort: str, image_count: int
    ) -> None:
        # The cohort mode and the count map 1:1; every other combination
        # fails fast instead of being guessed at runtime.
        with pytest.raises(ValidationError):
            TrainingSamplingConfig.model_validate(
                {"fixed_cohort": fixed_cohort, "image_count": image_count}
            )

    def test_runtime_selector_uses_fixed_cohort_not_the_count(self) -> None:
        # The runtime picks the locked mode from the explicit cohort mode;
        # image_count is only a consistency assertion on the 1:1 mapping.
        assert _locked_cohort_for("none", 12) is False
        assert _locked_cohort_for("locked", 60) is True
        with pytest.raises(ValueError, match="image_count"):
            _locked_cohort_for("none", 60)
        with pytest.raises(ValueError, match="image_count"):
            _locked_cohort_for("locked", 12)
        with pytest.raises(ValueError):
            _locked_cohort_for("neutral", 12)

    def test_live_train_g1_config_still_resolves(self) -> None:
        # The production G1 ramp (locked cohort, 60 images) keeps resolving
        # under the strict 1:1 mapping.
        from pathlib import Path

        from sakuramoon.config import load_config

        loaded = load_config(
            Path("train_g1.toml"),
            config_root=Path("config"),
            validate_secrets=False,
        )
        sampling = loaded.config.sampling.training
        assert sampling.fixed_cohort == "locked"
        assert sampling.image_count == 60

    def test_fixed_cohort_rejects_removed_neutral_value(self) -> None:
        with pytest.raises(ValidationError):
            TrainingSamplingConfig.model_validate(
                {"fixed_cohort": "neutral", "image_count": 12}
            )

    def test_pin_must_be_positive_or_none(self) -> None:
        with pytest.raises(ValidationError):
            TrainingSamplingConfig(longitudinal_pin_update=0)
