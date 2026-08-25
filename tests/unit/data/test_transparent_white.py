"""Unit tests for the transparent-background white-composite policy.

Covers the comparison-only tag identity, the effective-alpha-first composite
(a fully-opaque RGBA/LA/P/WebP alpha band is a missing-alpha reject, not a
silent no-op composite), the strict rejections (special alpha / conflicting
background / missing alpha), the trigger-tag rewrite + NL clear, the
``fake_transparency``-does-not-trigger rule, the section-13 telemetry
conservation, and the candidate-source normalization fix in
:mod:`sakuramoon.data.caption`.
"""

from __future__ import annotations

import io
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
    TRANSPARENT_REJECTION_KEYS,
    TransparentWhiteCounts,
    TransparentWhiteOutcome,
    TransparentWhiteTelemetry,
    aggregate_transparent_white,
    apply_transparent_white,
    composite_to_white,
    has_effective_alpha,
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


class TestHasEffectiveAlpha:
    def test_semi_transparent_rgba_and_la(self) -> None:
        assert has_effective_alpha(_rgba((0, 0, 0, 128)))
        assert has_effective_alpha(Image.new("LA", (4, 4), (128, 128)))

    def test_fully_opaque_rgba_is_not_effective(self) -> None:
        assert not has_effective_alpha(_rgba((0, 0, 0, 255)))

    def test_fully_opaque_la_is_not_effective(self) -> None:
        assert not has_effective_alpha(Image.new("LA", (4, 4), (128, 255)))

    def test_palette_transparency(self) -> None:
        palette = Image.new("P", (4, 4))
        palette.info["transparency"] = 0
        assert has_effective_alpha(palette)
        assert not has_effective_alpha(Image.new("P", (4, 4)))

    def test_palette_transparency_unused_index_is_not_effective(self) -> None:
        # Every pixel sits on palette index 1; the transparent index 0 is
        # never used, so the alpha band is fully opaque.
        palette = Image.new("P", (4, 4), 1)
        palette.info["transparency"] = 0
        assert not has_effective_alpha(palette)

    def test_webp_alpha_survives_decode(self) -> None:
        buffered = io.BytesIO()
        _rgba((0, 0, 0, 128)).save(buffered, format="WEBP")
        buffered.seek(0)
        assert has_effective_alpha(Image.open(buffered))

    def test_webp_without_alpha_is_not_effective(self) -> None:
        buffered = io.BytesIO()
        _rgb((255, 255, 255)).save(buffered, format="WEBP")
        buffered.seek(0)
        assert not has_effective_alpha(Image.open(buffered))

    def test_rgb_has_no_alpha(self) -> None:
        assert not has_effective_alpha(_rgb((255, 255, 255)))


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

    def test_fully_opaque_rgba_rejects_missing_alpha(self) -> None:
        # A fully-opaque alpha band is a no-op composite -> explicit reject,
        # never a silent identity transform.
        fields = _fields(Tag("transparent_background", "transparent_background"))
        result = apply_transparent_white(_rgba((10, 20, 30, 255)), fields, _config())
        assert result.outcome is TransparentWhiteOutcome.REJECT_MISSING_ALPHA
        assert result.image is None

    def test_fully_opaque_webp_rejects_missing_alpha(self) -> None:
        fields = _fields(Tag("transparent_background", "transparent_background"))
        buffered = io.BytesIO()
        _rgba((10, 20, 30, 255)).save(buffered, format="WEBP")
        buffered.seek(0)
        result = apply_transparent_white(Image.open(buffered), fields, _config())
        assert result.outcome is TransparentWhiteOutcome.REJECT_MISSING_ALPHA

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


class TestTransparentWhiteCounts:
    def test_zero_counts_construct(self) -> None:
        counts = TransparentWhiteCounts(0, 0, 0, 0, 0, 0)
        assert counts.tagged == 0

    def test_valid_nonzero_counts(self) -> None:
        counts = TransparentWhiteCounts(
            tagged=3,
            composited=2,
            missing_alpha=1,
            special_alpha=0,
            conflict_bg=0,
            nl_suppressed=2,
        )
        assert counts.tagged == counts.composited + counts.missing_alpha

    def test_conservation_violation_raises(self) -> None:
        with pytest.raises(ValueError, match="conservation"):
            TransparentWhiteCounts(
                tagged=4,
                composited=2,
                missing_alpha=1,
                special_alpha=0,
                conflict_bg=0,
                nl_suppressed=2,
            )

    def test_negative_counter_raises(self) -> None:
        with pytest.raises(ValueError, match="nonnegative"):
            TransparentWhiteCounts(
                tagged=1,
                composited=0,
                missing_alpha=-1,
                special_alpha=0,
                conflict_bg=0,
                nl_suppressed=0,
            )

    def test_non_int_counter_raises(self) -> None:
        for value in (1.0, True, "1"):
            with pytest.raises(ValueError, match="nonnegative"):
                TransparentWhiteCounts(
                    tagged=value,  # type: ignore[arg-type]
                    composited=1,
                    missing_alpha=0,
                    special_alpha=0,
                    conflict_bg=0,
                    nl_suppressed=1,
                )

    def test_nl_suppressed_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="conservation"):
            TransparentWhiteCounts(
                tagged=1,
                composited=1,
                missing_alpha=0,
                special_alpha=0,
                conflict_bg=0,
                nl_suppressed=0,
            )


class _FakeSample:
    def __init__(self, outcome: TransparentWhiteOutcome) -> None:
        self.transparent_outcome = outcome


class TestAggregateTransparentWhite:
    def test_empty_stream_is_zero_counts(self) -> None:
        counts = aggregate_transparent_white([])
        assert counts == TransparentWhiteCounts(0, 0, 0, 0, 0, 0)

    def test_mixed_outcomes_count_only_retained(self) -> None:
        samples = [
            _FakeSample(TransparentWhiteOutcome.COMPOSITED),
            _FakeSample(TransparentWhiteOutcome.NOT_TAGGED),
            _FakeSample(TransparentWhiteOutcome.COMPOSITED),
        ]
        counts = aggregate_transparent_white(samples)
        assert counts.tagged == 2
        assert counts.composited == 2
        assert counts.nl_suppressed == 2
        assert counts.missing_alpha == 0

    def test_reject_outcome_on_retained_sample_raises(self) -> None:
        for reject in (
            TransparentWhiteOutcome.REJECT_MISSING_ALPHA,
            TransparentWhiteOutcome.REJECT_SPECIAL_ALPHA,
            TransparentWhiteOutcome.REJECT_CONFLICT_BG,
        ):
            with pytest.raises(ValueError, match="rejected"):
                aggregate_transparent_white([_FakeSample(reject)])

    def test_invalid_outcome_type_raises(self) -> None:
        sample = _FakeSample("not_tagged")  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="missing or invalid"):
            aggregate_transparent_white([sample])

    def test_sample_without_outcome_raises(self) -> None:
        with pytest.raises(AttributeError):
            aggregate_transparent_white([object()])


class TestAggregateConservation:
    """Section-13 conservation across the two channels.

    The batch stream only sees retained samples (composited + untagged),
    while the per-shard rejection ledger sees the discarded ones.  At the
    cumulative level the two channels must add up exactly:

    ``tagged_total == composited_total + missing + special + conflict``
    and ``nl_suppressed_total == composited_total``.
    """

    def test_batch_stream_plus_rejection_ledger_conserve(self) -> None:
        # One training stream: per-batch retained-sample aggregates.
        batches: list[TransparentWhiteCounts] = [
            # (composited per batch, nl_suppressed mirrors composited)
            aggregate_transparent_white(
                [
                    _FakeSample(TransparentWhiteOutcome.COMPOSITED),
                    _FakeSample(TransparentWhiteOutcome.NOT_TAGGED),
                    _FakeSample(TransparentWhiteOutcome.COMPOSITED),
                    _FakeSample(TransparentWhiteOutcome.NOT_TAGGED),
                ]
            ),
            aggregate_transparent_white(
                [
                    _FakeSample(TransparentWhiteOutcome.COMPOSITED),
                    _FakeSample(TransparentWhiteOutcome.NOT_TAGGED),
                ]
            ),
            aggregate_transparent_white([_FakeSample(TransparentWhiteOutcome.NOT_TAGGED)]),
        ]
        composited_total = sum(batch.composited for batch in batches)
        nl_total = sum(batch.nl_suppressed for batch in batches)
        assert composited_total == 3
        assert nl_total == composited_total

        # The reliable per-shard rejection channel (fixed keys, cumulative).
        ledger: dict[str, int] = dict.fromkeys(TRANSPARENT_REJECTION_KEYS, 0)
        for key, total in (
            ("reject_missing_alpha", 2),
            ("reject_special_alpha", 1),
            ("reject_conflict_bg", 1),
        ):
            ledger[key] = total

        # A tagged sample either survives into the batch stream as
        # composited or is counted exactly once in the ledger: the tagged
        # total observed at the source is the sum of both channels.
        tagged_total = composited_total + sum(ledger.values())
        assert tagged_total == 7  # 3 composited + 2 missing + 1 special + 1 conflict
        assert nl_total == composited_total
        for batch in batches:
            # Per-batch invariant: in the retained-sample view the reject
            # counters are structurally zero and tagged == composited.
            assert (
                batch.tagged
                == batch.composited
                + batch.missing_alpha
                + batch.special_alpha
                + batch.conflict_bg
            )
            assert batch.missing_alpha == 0
            assert batch.nl_suppressed == batch.composited

    def test_rejection_keys_are_fixed_and_disjoint(self) -> None:
        assert len(TRANSPARENT_REJECTION_KEYS) == 3
        assert set(TRANSPARENT_REJECTION_KEYS) == {
            "reject_missing_alpha",
            "reject_special_alpha",
            "reject_conflict_bg",
        }
        # The fixed keys must not overlap the per-batch count field names.
        count_fields = {
            "tagged",
            "composited",
            "missing_alpha",
            "special_alpha",
            "conflict_bg",
            "nl_suppressed",
        }
        assert set(TRANSPARENT_REJECTION_KEYS).isdisjoint(count_fields)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
