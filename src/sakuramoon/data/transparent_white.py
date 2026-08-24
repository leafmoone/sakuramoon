"""Transparent-background white-composite policy (data strategy, sections 6-13).

This is a *data-strategy only* policy. It does not touch the model, loss,
optimizer, LR, batch, or the shifted-bucket spatial crop. For a sample whose
general tag semantically matches the trigger tag (``transparent_background``,
matched via the comparison-only :func:`tag_match_key` so the space form also
matches) the policy:

1. requires a usable per-pixel alpha channel;
2. if present, composites the alpha onto a solid white canvas BEFORE
   resize/crop, rewrites the trigger tag to ``white_background`` (which the
   tokenizer surface serializes as "white background"), and clears every
   training NL candidate (sample-level NL off); other tags are kept;
3. if no usable alpha, skips the sample with an explicit ``missing_alpha``
   reason (v1);
4. if the sample carries a special alpha effect or a conflicting explicit
   background, rejects it strictly and counts it (``fake_transparency`` alone
   does NOT trigger).

Rejected or missing-alpha samples never reach Qwen/VAE/DiT. The ordinary
(untagged) path is untouched and bit-identical: :func:`apply_transparent_white`
returns ``NOT_TAGGED`` for it, leaving the image and fields unchanged.

The rewritten trigger tag keeps its *original* ``canonical`` (the raw
``transparent_background``) so the dropout seed domain stays on the original
value, while ``text`` becomes ``white_background`` for display/serialization.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum

from PIL import Image, ImageOps

from sakuramoon.config.schema import DataTransparentBackgroundConfig
from sakuramoon.data.caption import CaptionFields, NlCandidates, Tag
from sakuramoon.data.tag_identity import tag_match_key

__all__ = [
    "TransparentWhiteOutcome",
    "TransparentWhiteResult",
    "TransparentWhiteTelemetry",
    "apply_transparent_white",
    "composite_to_white",
    "has_valid_alpha",
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


def has_valid_alpha(image: Image.Image) -> bool:
    """True when the decoded image carries a usable per-pixel alpha channel.

    ``RGBA`` and ``LA`` always qualify. A palette (``P``) image qualifies only
    when it advertises transparency. Fully-opaque alphas still qualify (the
    composite is then a no-op and stays well defined); they are not special.
    """

    if image.mode in ("RGBA", "LA"):
        return True
    if image.mode == "P" and "transparency" in image.info:
        return True
    return False


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
    ``fields`` carries the rewritten general tag plus cleared NL.
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

    # (3) no usable alpha -> explicit skip (v1).
    if not has_valid_alpha(image):
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
    cleared_nl = NlCandidates(None, None, None, None, None)
    rewritten_fields = replace(fields, general=new_general, nl=cleared_nl)
    return TransparentWhiteResult(
        outcome=TransparentWhiteOutcome.COMPOSITED,
        image=composited,
        fields=rewritten_fields,
    )


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
        "tagged",
        "composited",
        "missing_alpha",
        "special_alpha",
        "conflict_bg",
        "nl_suppressed",
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
