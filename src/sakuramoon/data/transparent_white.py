"""Transparent-background white-composite policy (data strategy, sections 6-13).

This is a *data-strategy only* policy. It does not touch the model, loss,
optimizer, LR, batch, or the shifted-bucket spatial crop. For a sample whose
general tag semantically matches the trigger tag (``transparent_background``,
matched via the comparison-only :func:`tag_match_key` so the space form also
matches) the policy:

1. requires an effective per-pixel alpha channel (at least one pixel below
   full opacity);
2. if present, composites the alpha onto a solid white canvas BEFORE
   resize/crop, rewrites the trigger tag to ``white_background`` (which the
   tokenizer surface serializes as "white background"), clears every
   training NL candidate (sample-level NL off), aligns the candidate
   deletion set with the rewrite, and collapses post-rewrite white-family
   duplicates (see :func:`apply_transparent_white`); other tags are kept;
3. if no effective alpha (no alpha channel, or a fully-opaque RGBA/LA/P/WebP
   alpha band), skips the sample with an explicit ``missing_alpha`` reason
   (v1) - a fully-opaque alpha band would make the composite a silent identity
   transform, so it is rejected like a missing channel;
4. if the sample carries a special alpha effect or a conflicting explicit
   background, rejects it strictly and counts it (``fake_transparency`` alone
   does NOT trigger).

Rejected or missing-alpha samples never reach Qwen/VAE/DiT. The ordinary
(untagged) path is untouched and bit-identical: :func:`apply_transparent_white`
returns ``NOT_TAGGED`` for it, leaving the image and fields unchanged.

The rewritten trigger tag keeps its *original* ``canonical`` (the raw
``transparent_background``) so the dropout seed domain stays on the original
value, while ``text`` becomes ``white_background`` for display/serialization.

Candidate alignment and co-occurrence: on a composited sample the trigger
drop IDs in ``candidate_tags`` would dangle (the trigger tag no longer
exists) while a pre-existing ``white_background`` entry would delete the
very background tag the composite now carries, so the deletion set is
rewritten trigger -> replacement and collapsed to one spelling per
comparison-only match key (the lexicographically minimal string wins; the
normalized membership test downstream is a functional no-op and the
dropout seed domain stays on the raw ``Tag.canonical``).  A
self-duplicated trigger entry - or an explicit white tag co-occurring when
the conflict set does not cover it - leaves at most one white-family entry
in the rewritten general list: the first occurrence per match key wins, so
the documented seed-domain intent is preserved for the trigger.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
from enum import Enum

from PIL import Image, ImageOps

from sakuramoon.config.schema import DataTransparentBackgroundConfig
from sakuramoon.data.caption import CaptionFields, NlCandidates, Tag
from sakuramoon.data.tag_identity import tag_match_key

__all__ = [
    "TRANSPARENT_REJECTION_KEYS",
    "TransparentWhiteCounts",
    "TransparentWhiteOutcome",
    "TransparentWhiteResult",
    "TransparentWhiteTelemetry",
    "aggregate_transparent_white",
    "apply_transparent_white",
    "composite_to_white",
    "has_effective_alpha",
    "is_transparent_tagged",
]


class TransparentWhiteOutcome(str, Enum):
    """Terminal outcome of the transparent-background policy for one sample."""

    NOT_TAGGED = "not_tagged"
    COMPOSITED = "composited"
    REJECT_MISSING_ALPHA = "reject_missing_alpha"
    REJECT_SPECIAL_ALPHA = "reject_special_alpha"
    REJECT_CONFLICT_BG = "reject_conflict_bg"

    @property
    def is_reject(self) -> bool:
        return self in _REJECT_OUTCOMES

    @property
    def observer_reason(self) -> str | None:
        """Reason string for the generic rejection observer (None if kept)."""
        if self is TransparentWhiteOutcome.NOT_TAGGED:
            return None
        return self.value


_REJECT_OUTCOMES = frozenset(
    {
        TransparentWhiteOutcome.REJECT_MISSING_ALPHA,
        TransparentWhiteOutcome.REJECT_SPECIAL_ALPHA,
        TransparentWhiteOutcome.REJECT_CONFLICT_BG,
    }
)

#: Fixed key set of the reliable worker->parent rejection channel.  Rejected
#: samples never produce a :class:`PipelineSample`, so their counters cannot
#: ride the batch/telemetry stream; each shard completion carries exactly
#: these keys (zero-filled when the shard rejected nothing).
TRANSPARENT_REJECTION_KEYS: tuple[str, ...] = (
    TransparentWhiteOutcome.REJECT_MISSING_ALPHA.value,
    TransparentWhiteOutcome.REJECT_SPECIAL_ALPHA.value,
    TransparentWhiteOutcome.REJECT_CONFLICT_BG.value,
)


def has_effective_alpha(image: Image.Image) -> bool:
    """True when the decoded image carries an *effective* alpha channel.

    ``RGBA``/``LA`` images, palette (``P``) images advertising transparency,
    and alpha-channel WebP (which decodes to ``RGBA``) qualify only when the
    alpha band is actually below full opacity somewhere (``min < 255``).
    A fully-opaque alpha band makes the white composite a no-op, so it is
    routed to the explicit ``missing_alpha`` reject instead of a silent
    identity transform.
    """

    if image.size[0] == 0 or image.size[1] == 0:
        return False
    if image.mode in ("RGBA", "LA"):
        alpha = image.getchannel("A")
    elif image.mode == "P" and "transparency" in image.info:
        alpha = image.convert("RGBA").getchannel("A")
    else:
        return False
    low, _high = alpha.getextrema()
    return low < 255


def composite_to_white(image: Image.Image, *, color: int = 255) -> Image.Image:
    """Composite an oriented alpha image onto a solid white canvas (RGB).

    Runs BEFORE resize/crop and before the RGB normalize. It is deterministic
    (no RNG) and bit-stable for a given image; fully-opaque alphas composite to
    a no-op. Returns a fresh ``RGB`` image (EXIF orientation already applied).
    """

    if not 0 <= color <= 255:
        raise ValueError("transparent-white composite color must be within [0, 255]")
    oriented = ImageOps.exif_transpose(image)
    rgba = oriented.convert("RGBA")
    background = Image.new("RGBA", rgba.size, (color, color, color, 255))
    return Image.alpha_composite(background, rgba).convert("RGB")


def is_transparent_tagged(fields: CaptionFields, config: DataTransparentBackgroundConfig) -> bool:
    """Whether the sample's general tags match the trigger (comparison-only)."""

    if not config.enabled:
        return False
    trigger_key = tag_match_key(config.trigger_tag)
    return any(tag_match_key(tag.canonical) == trigger_key for tag in fields.general)


@dataclass(frozen=True)
class TransparentWhiteResult:
    """Outcome of the policy for one sample.

    ``COMPOSITED``: ``image`` is the fresh white-composited RGB image and
    ``fields`` carries the rewritten general tag, the aligned candidate
    deletion set, and cleared NL.
    ``NOT_TAGGED``: ``image``/``fields`` are the originals (ordinary path).
    Rejections: ``image`` is ``None`` and the sample must be dropped.
    """

    outcome: TransparentWhiteOutcome
    image: Image.Image | None
    fields: CaptionFields


def apply_transparent_white(
    image: Image.Image,
    fields: CaptionFields,
    config: DataTransparentBackgroundConfig,
) -> TransparentWhiteResult:
    """Apply the transparent-background white-composite policy to one sample."""

    if not is_transparent_tagged(fields, config):
        return TransparentWhiteResult(
            outcome=TransparentWhiteOutcome.NOT_TAGGED, image=image, fields=fields
        )

    trigger_key = tag_match_key(config.trigger_tag)
    general_keys = [tag_match_key(tag.canonical) for tag in fields.general]

    # (4a) special alpha effect -> strict reject + count.
    special_keys = frozenset(tag_match_key(tag) for tag in config.special_alpha_tags)
    if special_keys and special_keys.intersection(general_keys):
        return TransparentWhiteResult(
            outcome=TransparentWhiteOutcome.REJECT_SPECIAL_ALPHA,
            image=None,
            fields=fields,
        )

    # (4b) conflicting explicit background -> strict reject + count.
    conflict_keys = frozenset(
        tag_match_key(tag) for tag in config.conflict_background_tags
    )
    if conflict_keys and conflict_keys.intersection(general_keys):
        return TransparentWhiteResult(
            outcome=TransparentWhiteOutcome.REJECT_CONFLICT_BG,
            image=None,
            fields=fields,
        )

    # (3) no effective alpha (fully-opaque included) -> explicit skip (v1).
    if not has_effective_alpha(image):
        return TransparentWhiteResult(
            outcome=TransparentWhiteOutcome.REJECT_MISSING_ALPHA,
            image=None,
            fields=fields,
        )

    # (2) usable alpha -> composite white + rewrite tag + clear training NL.
    composited = composite_to_white(image, color=_composite_color(config))
    new_general = tuple(
        Tag(text=config.replacement_tag, canonical=tag.canonical)
        if key == trigger_key
        else tag
        for tag, key in zip(fields.general, general_keys, strict=True)
    )
    # (2b) white-dedup: a self-duplicated trigger entry rewrites twice, and
    # an explicit white-family tag can co-occur when the conflict set does
    # not cover it; keep the first occurrence per match key.
    replacement_key = tag_match_key(config.replacement_tag)
    seen_replacement = False
    deduped_general: list[Tag] = []
    for tag in new_general:
        if tag_match_key(tag.text) == replacement_key:
            if seen_replacement:
                continue
            seen_replacement = True
        deduped_general.append(tag)
    cleared_nl = NlCandidates(None, None, None, None, None)
    rewritten_fields = replace(
        fields,
        general=tuple(deduped_general),
        candidate_tags=_rewrite_candidate_tags(fields.candidate_tags, config),
        nl=cleared_nl,
    )
    return TransparentWhiteResult(
        outcome=TransparentWhiteOutcome.COMPOSITED,
        image=composited,
        fields=rewritten_fields,
    )


def _rewrite_candidate_tags(
    candidates: frozenset[str],
    config: DataTransparentBackgroundConfig,
) -> frozenset[str]:
    """Align the candidate deletion set with a composited sample.

    Trigger-matching drop IDs become the replacement tag, and the result is
    collapsed to one spelling per comparison-only match key (the
    lexicographically minimal string wins) so the normalized membership test
    downstream sees exactly one drop ID per candidate.  The collapse is a
    functional no-op (both spellings match the same tag) and never touches
    the dropout seed domain, which stays on the raw ``Tag.canonical``.
    """

    trigger_key = tag_match_key(config.trigger_tag)
    replacement_key = tag_match_key(config.replacement_tag)
    groups: dict[str, list[str]] = {}
    for candidate in candidates:
        key = tag_match_key(candidate)
        if key == trigger_key:
            key = replacement_key
            value = config.replacement_tag
        else:
            value = candidate
        groups.setdefault(key, []).append(value)
    return frozenset(min(values) for values in groups.values())


_COMPOSITE_COLORS = {"white": 255}


def _composite_color(config: DataTransparentBackgroundConfig) -> int:
    try:
        return _COMPOSITE_COLORS[config.composite_color]
    except KeyError as error:  # pragma: no cover - guarded by the Literal type
        raise ValueError(
            f"unknown transparent-white composite color: {config.composite_color}"
        ) from error


class TransparentWhiteTelemetry:
    """Fixed counters with the section-13 conservation invariants.

    ``tagged = composited + missing_alpha + special_alpha + conflict_bg`` and
    ``nl_suppressed == composited``. Counters start at 0 (never NaN) and only
    increment, so they are safe to aggregate across DataLoader workers.
    """

    __slots__ = (
        "composited",
        "conflict_bg",
        "missing_alpha",
        "nl_suppressed",
        "special_alpha",
        "tagged",
    )

    def __init__(self) -> None:
        self.tagged = 0
        self.composited = 0
        self.missing_alpha = 0
        self.special_alpha = 0
        self.conflict_bg = 0
        self.nl_suppressed = 0

    def record(self, outcome: TransparentWhiteOutcome) -> None:
        if outcome is TransparentWhiteOutcome.NOT_TAGGED:
            return
        self.tagged += 1
        if outcome is TransparentWhiteOutcome.COMPOSITED:
            self.composited += 1
            self.nl_suppressed += 1
        elif outcome is TransparentWhiteOutcome.REJECT_MISSING_ALPHA:
            self.missing_alpha += 1
        elif outcome is TransparentWhiteOutcome.REJECT_SPECIAL_ALPHA:
            self.special_alpha += 1
        elif outcome is TransparentWhiteOutcome.REJECT_CONFLICT_BG:
            self.conflict_bg += 1

    def assert_conservation(self) -> None:
        if self.tagged != (
            self.composited
            + self.missing_alpha
            + self.special_alpha
            + self.conflict_bg
        ):
            raise AssertionError(
                "transparent-white conservation violated: "
                "tagged != composited + missing + special + conflict"
            )
        if self.nl_suppressed != self.composited:
            raise AssertionError(
                "transparent-white conservation violated: "
                "nl_suppressed != composited"
            )

    def snapshot(self) -> dict[str, int]:
        self.assert_conservation()
        return {
            "transparent_tagged": self.tagged,
            "transparent_composited": self.composited,
            "transparent_missing_alpha": self.missing_alpha,
            "transparent_special_alpha": self.special_alpha,
            "transparent_conflict_bg": self.conflict_bg,
            "transparent_nl_suppressed": self.nl_suppressed,
        }


@dataclass(frozen=True)
class TransparentWhiteCounts:
    """Fixed-key per-batch counters for the transparent-white policy.

    Aggregated from the per-sample outcomes of the *retained* samples of one
    batch, mirroring :class:`~sakuramoon.data.spatial_crop.SpatialCropCounts`.
    Rejected samples produce no :class:`~sakuramoon.data.pipeline.PipelineSample`
    and therefore no per-batch count: their counters are structurally zero in
    the batch view and flow to the parent through the reliable per-shard
    completion channel (``TRANSPARENT_REJECTION_KEYS``) instead.  The section-13
    conservation invariants are enforced at construction time so a corrupted
    aggregate can never reach the training metric.
    """

    tagged: int
    composited: int
    missing_alpha: int
    special_alpha: int
    conflict_bg: int
    nl_suppressed: int

    def __post_init__(self) -> None:
        for name in (
            "tagged",
            "composited",
            "missing_alpha",
            "special_alpha",
            "conflict_bg",
            "nl_suppressed",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"transparent-white counter {name} must be a nonnegative integer")
        if self.tagged != (
            self.composited + self.missing_alpha + self.special_alpha + self.conflict_bg
        ):
            raise ValueError(
                "transparent-white conservation violated: "
                "tagged != composited + missing + special + conflict"
            )
        if self.nl_suppressed != self.composited:
            raise ValueError(
                "transparent-white conservation violated: "
                "nl_suppressed != composited"
            )


def aggregate_transparent_white(samples: Iterable[object]) -> TransparentWhiteCounts:
    """Aggregate the fixed transparent-white counters from per-sample outcomes.

    ``samples`` are the retained samples of one batch; each carries a
    ``transparent_outcome`` of type :class:`TransparentWhiteOutcome`.  The
    aggregate is spawn-picklable (frozen plain ints) and safe to cross the
    DataLoader worker boundary.
    """

    tagged = 0
    composited = 0
    for sample in samples:
        outcome = sample.transparent_outcome  # type: ignore[attr-defined]
        if not isinstance(outcome, TransparentWhiteOutcome):
            raise TypeError("transparent outcome is missing or invalid on a sample")
        if outcome is TransparentWhiteOutcome.NOT_TAGGED:
            continue
        tagged += 1
        if outcome is TransparentWhiteOutcome.COMPOSITED:
            composited += 1
        else:
            # Retained samples can only be NOT_TAGGED or COMPOSITED; a reject
            # outcome here means the sample channel was fed a discarded sample.
            raise ValueError(
                f"retained sample carries a rejected transparent outcome: {outcome}"
            )
    return TransparentWhiteCounts(
        tagged=composited,
        composited=composited,
        missing_alpha=0,
        special_alpha=0,
        conflict_bg=0,
        nl_suppressed=composited,
    )
