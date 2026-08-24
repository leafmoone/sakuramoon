"""Unit tests for the transparent-background white-composite policy.

Covers the comparison-only tag identity, the alpha-first composite, the strict
rejections (special alpha / conflicting background / missing alpha), the
trigger-tag rewrite + NL clear, the ``fake_transparency``-does-not-trigger rule,
the section-13 telemetry conservation, and the candidate-source normalization
fix in :mod:`sakuramoon.data.caption`.
"""

from __future__ import annotations

from typing import Any

import pytest
from PIL import Image

from sakuramoon.config.schema import DataTransparentBackgroundConfig
from sakuramoon.data.caption import (
    CaptionFields,
    NlCandidates,
    Tag,
    _drop_source_tags,
)
from sakuramoon.data.tag_identity import tag_match_key
from sakuramoon.data.transparent_white import (
    TransparentWhiteOutcome,
    TransparentWhiteTelemetry,
    apply_transparent_white,
    composite_to_white,
    has_valid_alpha,
    is_transparent_tagged,
)


def _fields(
    *general: Tag,
    nl: NlCandidates | None = None,
    candidate: frozenset[str] = frozenset(),
) -> CaptionFields:
    return CaptionFields(
        nsfw=(),
        character=(),
        copyright=(),
        general=tuple(general),
        artists=(),
        candidate_tags=candidate,
        nl=nl if nl is not None else NlCandidates(None, None, None, None, None),
    )


def _config(
    enabled: bool = True,
    special: tuple[str, ...] = (),
    conflict: tuple[str, ...] = (),
) -> DataTransparentBackgroundConfig:
    return DataTransparentBackgroundConfig(
        enabled=enabled,
        special_alpha_tags=special,
        conflict_background_tags=conflict,
    )


def _rgba(color: tuple[int, int, int, int]) -> Image.Image:
    return Image.new("RGBA", (4, 4), color)


def _rgb(color: tuple[int, int, int]) -> Image.Image:
    return Image.new("RGB", (4, 4), color)


class TestTagMatchKey:
    def test_normalizes_case_and_separators(self) -> None:
        assert tag_match_key("transparent_background") == "transparent_background"
        assert tag_match_key("Transparent Background") == "transparent_background"
        assert tag_match_key("WHITE BACKGROUND") == "white_background"
        assert tag_match_key("White_Background") == "white_background"

    def test_rejects_invalid_values(self) -> None:
        with pytest.raises(ValueError):
            tag_match_key(123)  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            tag_match_key("")
        with pytest.raises(ValueError):
            tag_match_key(" padded ")
        with pytest.raises(ValueError):
            tag_match_key("bad\ncontrol")


class TestHasValidAlpha:
    def test_rgba_and_la(self) -> None:
        assert has_valid_alpha(_rgba((0, 0, 0, 255)))
        assert has_valid_alpha(Image.new("LA", (4, 4), (128, 128)))

    def test_palette_transparency(self) -> None:
        palette = Image.new("P", (4, 4))
        palette.info["transparency"] = 0
        assert has_valid_alpha(palette)
        assert not has_valid_alpha(Image.new("P", (4, 4)))

    def test_rgb_has_no_alpha(self) -> None:
        assert not has_valid_alpha(_rgb((255, 255, 255)))


class TestCompositeToWhite:
    def test_transparent_pixel_becomes_white(self) -> None:
        result = composite_to_white(_rgba((0, 0, 0, 0)))
        assert result.mode == "RGB"
        assert result.getpixel((0, 0)) == (255, 255, 255)

    def test_opaque_is_a_noop(self) -> None:
        source = _rgba((10, 20, 30, 255))
        result = composite_to_white(source)
        assert result.mode == "RGB"
        assert result.getpixel((0, 0)) == (10, 20, 30)

    def test_rejects_out_of_range_color(self) -> None:
        with pytest.raises(ValueError):
            composite_to_white(_rgba((0, 0, 0, 255)), color=300)


class TestApplyPolicy:
    def test_not_tagged_leaves_sample_unchanged(self) -> None:
        image = _rgba((0, 0, 0, 0))
        fields = _fields(Tag("cat", "cat"))
        result = apply_transparent_white(image, fields, _config())
        assert result.outcome is TransparentWhiteOutcome.NOT_TAGGED
        assert result.image is image
        assert result.fields is fields

    def test_composited_rewrites_tag_and_clears_nl(self) -> None:
        fields = _fields(
            Tag("transparent_background", "transparent_background"),
            Tag("cat", "cat"),
            nl=NlCandidates("nl", "nl", "nl", "nl", "nl"),
        )
        result = apply_transparent_white(_rgba((0, 0, 0, 0)), fields, _config())
        assert result.outcome is TransparentWhiteOutcome.COMPOSITED
        assert result.image is not None and result.image.mode == "RGB"
        assert result.image.getpixel((0, 0)) == (255, 255, 255)
        rewritten = result.fields.general
        # trigger rewritten: display text changes, canonical (seed domain) kept
        assert rewritten[0].text == "white_background"
        assert rewritten[0].canonical == "transparent_background"
        # sibling tag is untouched
        assert rewritten[1] == Tag("cat", "cat")
        # every NL candidate cleared (sample-level NL off)
        assert result.fields.nl == NlCandidates(None, None, None, None, None)

    def test_missing_alpha_rejects(self) -> None:
        fields = _fields(Tag("transparent_background", "transparent_background"))
        result = apply_transparent_white(_rgb((255, 255, 255)), fields, _config())
        assert result.outcome is TransparentWhiteOutcome.REJECT_MISSING_ALPHA
        assert result.image is None
        assert result.outcome.observer_reason == "reject_missing_alpha"

    def test_special_alpha_rejects(self) -> None:
        fields = _fields(
            Tag("transparent_background", "transparent_background"),
            Tag("alpha_transparency", "alpha_transparency"),
        )
        config = _config(special=("alpha_transparency", "thumbnail_surprise"))
        result = apply_transparent_white(_rgba((0, 0, 0, 0)), fields, config)
        assert result.outcome is TransparentWhiteOutcome.REJECT_SPECIAL_ALPHA

    def test_conflicting_background_rejects(self) -> None:
        fields = _fields(
            Tag("transparent_background", "transparent_background"),
            Tag("black_background", "black_background"),
        )
        config = _config(conflict=("black_background",))
        result = apply_transparent_white(_rgba((0, 0, 0, 0)), fields, config)
        assert result.outcome is TransparentWhiteOutcome.REJECT_CONFLICT_BG

    def test_fake_transparency_alone_does_not_trigger(self) -> None:
        # "fake_transparency" is not the trigger tag; without the real
        # "transparent_background" tag the policy is a no-op.
        fields = _fields(Tag("fake_transparency", "fake_transparency"))
        result = apply_transparent_white(_rgba((0, 0, 0, 0)), fields, _config())
        assert result.outcome is TransparentWhiteOutcome.NOT_TAGGED

    def test_disabled_config_is_a_noop(self) -> None:
        fields = _fields(Tag("transparent_background", "transparent_background"))
        result = apply_transparent_white(
            _rgba((0, 0, 0, 0)), fields, _config(enabled=False)
        )
        assert result.outcome is TransparentWhiteOutcome.NOT_TAGGED

    def test_disabled_config_reports_untagged(self) -> None:
        fields = _fields(Tag("transparent_background", "transparent_background"))
        assert is_transparent_tagged(fields, _config(enabled=False)) is False
        assert is_transparent_tagged(fields, _config()) is True


class TestTelemetry:
    def test_conservation_holds(self) -> None:
        telemetry = TransparentWhiteTelemetry()
        for outcome in (
            TransparentWhiteOutcome.COMPOSITED,
            TransparentWhiteOutcome.COMPOSITED,
            TransparentWhiteOutcome.REJECT_MISSING_ALPHA,
            TransparentWhiteOutcome.REJECT_SPECIAL_ALPHA,
            TransparentWhiteOutcome.REJECT_CONFLICT_BG,
            TransparentWhiteOutcome.NOT_TAGGED,
        ):
            telemetry.record(outcome)
        telemetry.assert_conservation()
        assert telemetry.tagged == 5  # NOT_TAGGED is not a tagged sample
        assert telemetry.composited == 2
        assert telemetry.nl_suppressed == 2

    def test_snapshot_keys_and_conservation_guard(self) -> None:
        telemetry = TransparentWhiteTelemetry()
        telemetry.record(TransparentWhiteOutcome.COMPOSITED)
        snapshot: dict[str, Any] = telemetry.snapshot()
        assert set(snapshot) == {
            "transparent_tagged",
            "transparent_composited",
            "transparent_missing_alpha",
            "transparent_special_alpha",
            "transparent_conflict_bg",
            "transparent_nl_suppressed",
        }
        assert snapshot["transparent_tagged"] == 1

    def test_conservation_violation_raises(self) -> None:
        telemetry = TransparentWhiteTelemetry()
        # Manually break the invariant (simulating a bug).
        telemetry.tagged = 1
        telemetry.composited = 0
        with pytest.raises(AssertionError):
            telemetry.assert_conservation()


class TestCandidateSourceNormalization:
    def test_space_form_candidate_matches_underscore_tag(self) -> None:
        fields = _fields(
            Tag("transparent_background", "transparent_background"),
            Tag("other", "other"),
            candidate=frozenset({"Transparent Background"}),
        )
        retained, _hit = _drop_source_tags(
            fields,
            "general",
            candidate_source_hit=True,
            probability=0.0,
            seed=12345,
        )
        # The normalized candidate matches the underscore tag, which is dropped;
        # the unrelated tag survives.
        assert retained == (Tag("other", "other"),)

    def test_no_candidate_hit_keeps_all(self) -> None:
        fields = _fields(
            Tag("transparent_background", "transparent_background"),
            candidate=frozenset({"unrelated"}),
        )
        retained, _hit = _drop_source_tags(
            fields,
            "general",
            candidate_source_hit=True,
            probability=0.0,
            seed=12345,
        )
        assert retained == (Tag("transparent_background", "transparent_background"),)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
