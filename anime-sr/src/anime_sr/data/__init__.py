"""Data services (milestone M1, plan §10-§11).

- ``index``: shard scan → SR eligibility + validation split (§10)
- ``buckets``: frozen 4× bucket table + deterministic crops (§10.1)
- ``degradation``: deterministic P0-P4 anime degradation chain (§11)
- ``pipeline``: SRDataset (webp → HR crop → LQ) for training (§10-§11)

Contract: ``anime-sr/docs/data-contract.md`` + ``anime-sr/docs/degradation-v1.md``.
"""

from anime_sr.data.buckets import Bucket, check_buckets, crop_box
from anime_sr.data.degradation import (
    DegradationParams,
    apply_degradation,
    degrade_hr,
    exposure_seed,
    sample_params,
)
from anime_sr.data.index import (
    Eligibility,
    MetaRecord,
    build_index,
    evaluate_eligibility,
    iter_index,
)

__all__ = [
    "Bucket",
    "DegradationParams",
    "Eligibility",
    "MetaRecord",
    "apply_degradation",
    "build_index",
    "check_buckets",
    "crop_box",
    "degrade_hr",
    "evaluate_eligibility",
    "exposure_seed",
    "iter_index",
    "sample_params",
]
