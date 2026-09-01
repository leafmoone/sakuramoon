"""Resolution buckets + deterministic HR crops (plan §10.1, data-contract §1).

Fixed 4×: HR sizes are multiples of 64 (latent divisible by 4), LQ = HR/4 is
a multiple of 16. Crops are full square (100% retention — the §10.4
80%-retention floor is satisfied trivially) at a deterministic offset so a
given (sample, seed) always yields the same crop (resume-safe).
"""

from __future__ import annotations

from dataclasses import dataclass

from anime_sr.config.schema import Config
from anime_sr.data.index import MetaRecord
from anime_sr.data.index import eligible_buckets as _index_eligible_buckets

# re-export: the canonical crop_box lives in the torch-free top-level module
# (spawn workers import it without triggering the torch-heavy data package)


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


def assert_aligned(h: int, w: int, lq_h: int, lq_w: int) -> None:
    """Data-contract §1 tensor alignment (HR 64-multiple, LQ 16-multiple)."""
    if h % 64 or w % 64:
        raise ValueError(f"HR {w}x{h} edges must be multiples of 64")
    if lq_h % 16 or lq_w % 16:
        raise ValueError(f"LQ {lq_w}x{lq_h} edges must be multiples of 16")
