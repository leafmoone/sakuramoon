"""SR-clean-v1 label-pool logic and deterministic stats helpers.

Single source of truth shared by the label-analysis tools
(``tools/analyze_sr_hr_quality_labels.py`` for local shard mirrors and
``tools/stream_scan_v2_repo.py`` for HTTP-range streaming scans of the raw
modelscope repo). Everything here is pure (no I/O, no torch) and depends only
on the standard library, so it is trivially unit-testable.

The pool predicates operate on :class:`RawMeta` — the label fields exactly as
read from a danbooru-v2 shard ``.json``. They are label-only: nothing here
reads pixels or applies a clean-score gate.
"""

from __future__ import annotations

import hashlib
import math
import random
from collections.abc import Sequence
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# constants / rules
# ---------------------------------------------------------------------------
QUALITY_ORDINAL: dict[str, int] = {
    "masterpiece": 6,
    "best": 5,
    "great": 4,
    "good": 3,
    "normal": 2,
    "low": 1,
    "worst": 0,
}
PRIORITY_QUALITY = frozenset({"masterpiece", "best", "great"})
P1_QUALITY = frozenset({"masterpiece", "best", "great", "good", "normal"})
CORE_CLASSIFICATION = frozenset({"illustration", "bangumi", "comic"})
DEFAULT_SEED = 20260831
DEFAULT_TOTAL_EXPOSURES = 6_000_000


def quality_ordinal(tier: str) -> int:
    """Ordinal rank of a quality tier; unknown/absent -> -1."""
    return QUALITY_ORDINAL.get(tier, -1)


# ---------------------------------------------------------------------------
# raw sample record
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RawMeta:
    """One danbooru-v2 image, fields exactly as read from the tar ``.json``."""

    sample_id: str
    shard: str
    width_meta: int  # original-post dims from the meta (may differ from webp)
    height_meta: int
    nsfw: str
    year: int
    quality_tier: str  # "" when absent
    completeness: str  # anime_completeness, "" when absent
    classification: str  # anime_classification, "" when absent
    ai_corrupted: bool
    tags_general: tuple[str, ...]

    @property
    def sample_hash10k(self) -> int:
        d = hashlib.blake2b(self.sample_id.encode(), digest_size=4).digest()
        return int.from_bytes(d, "little") % 10_000


# ---------------------------------------------------------------------------
# pool predicates (label-only)
# ---------------------------------------------------------------------------
def in_p0(m: RawMeta) -> bool:
    """P0 = existing strict priority (project definition, label-only)."""
    return (
        m.quality_tier in PRIORITY_QUALITY
        and m.completeness == "polished"
        and m.classification in CORE_CLASSIFICATION
    )


def in_p1(m: RawMeta) -> bool:
    """P1 = SR-clean-v1: polished + core classification + tier >= normal."""
    return (
        m.completeness == "polished"
        and m.classification in CORE_CLASSIFICATION
        and m.quality_tier in P1_QUALITY
    )


def in_p2(m: RawMeta) -> bool:
    """P2 = SR-clean-wide: polished + core classification, any tier."""
    return m.completeness == "polished" and m.classification in CORE_CLASSIFICATION


def in_p3(m: RawMeta) -> bool:
    """P3 = line-art-extra: monochrome OR comic, not corrupted, not
    not_painting (population is already hard-eligibility-filtered)."""
    return (
        (m.completeness == "monochrome" or m.classification == "comic")
        and not m.ai_corrupted
        and m.classification != "not_painting"
    )


def in_p4(m: RawMeta) -> bool:
    """P4 = rough-extra (statistics only)."""
    return m.completeness == "rough"


POOLS = (
    ("P0_priority", in_p0),
    ("P1_sr_clean_v1", in_p1),
    ("P2_sr_clean_wide", in_p2),
    ("P3_lineart_extra", in_p3),
    ("P4_rough_extra", in_p4),
)


# ---------------------------------------------------------------------------
# small stats helpers (stdlib only, deterministic)
# ---------------------------------------------------------------------------
def _pctl(sorted_vals: Sequence[float], p: float) -> float | None:
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * (p / 100.0)
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return sorted_vals[int(k)]
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


def _pctl_block(vals: Sequence[float]) -> dict[str, float | None]:
    s = sorted(vals)
    out = {f"p{q}": _pctl(s, q) for q in (10, 25, 50, 75, 90)}
    out["min"] = s[0] if s else None
    out["max"] = s[-1] if s else None
    out["mean"] = (sum(s) / len(s)) if s else None
    return out


def _ranks(xs: Sequence[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _pearson(a: Sequence[float], b: Sequence[float]) -> float | None:
    n = len(a)
    if n < 2:
        return None
    ma = sum(a) / n
    mb = sum(b) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da = math.sqrt(sum((x - ma) ** 2 for x in a))
    db = math.sqrt(sum((y - mb) ** 2 for y in b))
    if da == 0.0 or db == 0.0:
        return None
    return num / (da * db)


def _spearman(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    return _pearson(_ranks(xs), _ranks(ys))


def _det_sample(ids: Sequence[str], k: int, seed: int) -> list[str]:
    """Deterministic sample: sorted input, fixed-seed MT19937."""
    if len(ids) <= k:
        return sorted(ids)
    return random.Random(seed).sample(sorted(ids), k)


# ---------------------------------------------------------------------------
# crop-position proxy for a square-bucket HR crop
# ---------------------------------------------------------------------------
def _crop_positions(width: int, height: int, bucket: int) -> tuple[int, int]:
    """(positions_exact, positions_stride64) for a square bucket crop."""
    nx = max(1, width - bucket + 1)
    ny = max(1, height - bucket + 1)
    exact = nx * ny
    nx64 = (width - bucket) // 64 + 1 if width >= bucket else 0
    ny64 = (height - bucket) // 64 + 1 if height >= bucket else 0
    return exact, (nx64 * ny64)


_CROP_THRESHOLDS = (1, 4, 16, 64, 256)


def _crop_bucket_hist(positions: list[int]) -> dict[str, int]:
    hist: dict[str, int] = {"==1": 0, **{f">{t}": 0 for t in _CROP_THRESHOLDS}}
    for p in positions:
        if p == 1:
            hist["==1"] += 1
        for t in _CROP_THRESHOLDS:
            if p > t:
                hist[f">{t}"] += 1
    return hist
