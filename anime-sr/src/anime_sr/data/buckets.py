"""Resolution buckets + deterministic HR crops (plan §10.1, data-contract §1).

Fixed 4×: HR sizes are multiples of 64 (latent divisible by 4), LQ = HR/4 is
a multiple of 16. Crops are full square (100% retention — the §10.4
80%-retention floor is satisfied trivially) at a deterministic offset so a
given (sample, seed) always yields the same crop (resume-safe).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from anime_sr.config.schema import Config
from anime_sr.data.index import MetaRecord
from anime_sr.data.index import eligible_buckets as _index_eligible_buckets


@dataclass(frozen=True)
class Bucket:
    """One training bucket: LQ side + HR side (fixed 4×)."""

    lq: int  # LQ edge (multiple of 16)
    hr: int  # HR edge (multiple of 64)

    @property
    def latent(self) -> int:
        """HR latent edge (HR / 16)."""
        return self.hr // 16


def check_buckets(cfg: Config) -> list[Bucket]:
    """Derive + validate the frozen 4× bucket table (lq_sizes × hr_multiplier)."""
    out: list[Bucket] = []
    for lq in cfg.buckets.lq_sizes:
        hr = lq * cfg.buckets.hr_multiplier
        if lq % cfg.buckets.lq_multiple:
            raise ValueError(f"bucket lq={lq} violates the {cfg.buckets.lq_multiple}-multiple contract")
        if hr % cfg.buckets.hr_multiple:
            raise ValueError(f"bucket hr={hr} violates the {cfg.buckets.hr_multiple}-multiple contract")
        if (hr // 16) % cfg.buckets.latent_divisible_by:
            raise ValueError(f"bucket {lq}->{hr} latent {hr // 16} not divisible by {cfg.buckets.latent_divisible_by}")
        out.append(Bucket(lq, hr))
    return out


def eligible_hr_buckets(rec: MetaRecord, cfg: Config) -> list[Bucket]:
    """Buckets whose full square HR crop fits in the image (data/index owns
    the size check; this returns the aligned Bucket objects)."""
    sizes = _index_eligible_buckets(rec, cfg)
    return [b for b in check_buckets(cfg) if b.hr in sizes]


def crop_box(width: int, height: int, hr: int, seed: int) -> tuple[int, int]:
    """Deterministic crop top-left for an HR square of edge ``hr``.

    Center-anchored with a seeded jitter clipped to the valid range, so
    (width, height, hr, seed) reproduces the exact box across resumes.
    x and y use independent hash streams (a single 64-bit value cannot
    supply two unbiased coordinates after one modulo).
    """
    if width < hr or height < hr:
        raise ValueError(f"image {width}x{height} cannot crop HR {hr}")
    if width == hr and height == hr:
        return 0, 0
    hx = hashlib.blake2b(f"crop-x|{seed}|{width}|{height}|{hr}".encode(), digest_size=8).digest()
    hy = hashlib.blake2b(f"crop-y|{seed}|{width}|{height}|{hr}".encode(), digest_size=8).digest()
    x_max, y_max = width - hr, height - hr
    x = int.from_bytes(hx, "little") % (x_max + 1)
    y = int.from_bytes(hy, "little") % (y_max + 1)
    return x, y


def assert_aligned(h: int, w: int, lq_h: int, lq_w: int) -> None:
    """Data-contract §1 tensor alignment (HR 64-multiple, LQ 16-multiple)."""
    if h % 64 or w % 64:
        raise ValueError(f"HR {w}x{h} edges must be multiples of 64")
    if lq_h % 16 or lq_w % 16:
        raise ValueError(f"LQ {lq_w}x{lq_h} edges must be multiples of 16")
