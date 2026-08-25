"""Comparison-only tag identity: separator/case-insensitive matching keys.

This module provides a *comparison-only* normalizer for tag strings. It is used
to make equality / membership tests robust to the ASCII-space vs underscore
separator (and case) drift between per-image tag lists and reference tables.

It is strictly NOT a rewrite:
  * it never mutates ``Tag.text`` / ``Tag.canonical`` / any display surface, and
  * it is never used for the deterministic dropout RNG seed, which keeps using
    the raw ``Tag.canonical`` (see ``caption._drop_source_tags``).

The normalizer is total and idempotent: applying it twice gives the same key,
and two keys that are equal represent the same tag under the "only the
separator and case differ" equivalence.
"""

from __future__ import annotations

__all__ = ["tag_match_key"]

# Characters that must never appear inside a tag identity. A well-formed tag
# string is a single tokenizer-facing token: no newlines / carriage returns /
# NUL / ASCII control characters. This mirrors the strict boundary checks in
# ``caption.Tag`` and the candidate-ID validation in ``CaptionFields``.
_FORBIDDEN = ("\n", "\r", "\0")


def tag_match_key(value: str) -> str:
    """Return the comparison-only identity key for a tag string.

    The key is ``value.casefold()`` with ASCII spaces collapsed to underscores,
    so that ``"transparent background"`` and ``"transparent_background"`` (and
    their case variants) map to the same key ``"transparent_background"``.

    Arguments:
        value: a non-empty, control-free tag string (exact ``str`` type).

    Returns:
        The normalized comparison key (``str``).

    Raises:
        ValueError: if ``value`` is not an exact ``str``, is empty, is not
            trim-stable, or contains a forbidden control character.
    """
    if type(value) is not str:
        raise ValueError("tag_match_key requires an exact str value")
    if not value:
        raise ValueError("tag_match_key requires a non-empty tag string")
    if value != value.strip():
        raise ValueError("tag_match_key input must be trim-stable")
    for forbidden in _FORBIDDEN:
        if forbidden in value:
            raise ValueError("tag_match_key input must not contain control characters")
    return value.casefold().replace(" ", "_")
