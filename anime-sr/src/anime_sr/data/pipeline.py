"""M1 training dataset: webp → HR crop → deterministic LQ (plan §10-§11).

The dataset reads the M1 index (``sr-eligibility-v1``), decodes webp from
the extracted shard directory (``webp/<shard>/<id>.webp``), crops the HR
square (deterministic offset, ``buckets.crop_box``) and applies the
exposure-deterministic degradation chain (``degradation.degrade_hr``).

``__getitem__`` returns (hr_crop, lq, meta) for the default exposure;
training with a multi-exposure schedule calls :meth:`fetch` with explicit
``exposure_index`` / ``data_cycle`` (resume contract, plan §11.5).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from anime_sr.config.schema import Config
from anime_sr.data import index as index_mod
from anime_sr.data.buckets import check_buckets, crop_box
from anime_sr.data.degradation import degrade_hr

_EXPOSURE_PER_CYCLE = 25  # exposures per image before the data cycle advances (§11.5)


def find_eligibility(index_dir: str | Path) -> str:
    """Locate the eligibility table (parquet preferred, JSONL fallback)."""
    d = Path(index_dir)
    for name in ("sr-eligibility-v1.parquet", "sr-eligibility-v1.jsonl"):
        if (d / name).is_file():
            return str(d / name)
    raise FileNotFoundError(f"no eligibility table in {d} (run cli/index_dataset.py first)")


def _blake2b_u64(s: str) -> int:
    """Stable 64-bit hash (platform-independent, stdlib only)."""
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "little")


def box_seed(sample_id: str, data_cycle: int, exposure_index: int) -> int:
    """Crop-box stream: independent from the degradation exposure seed."""
    return _blake2b_u64(f"box|{sample_id}|{data_cycle}|{exposure_index}")


@dataclass(frozen=True)
class SampleMeta:
    sample_id: str
    shard: str
    rel_path: str
    width: int
    height: int
    is_validation: bool


class SRDataset(Dataset):
    """train / validation split of the M1 index for one HR bucket."""

    def __init__(
        self,
        index_dir: str | Path,
        webp_dir: str | Path,
        cfg: Config,
        bucket_hr: int = 1024,
        split: str = "train",
        global_seed: int = 42,
    ) -> None:
        if split not in ("train", "validation"):
            raise ValueError(f"split must be train|validation, got {split}")
        if bucket_hr not in {b.hr for b in check_buckets(cfg)}:
            raise ValueError(f"bucket_hr {bucket_hr} not in the frozen bucket table")
        self.webp_dir = Path(webp_dir)
        self.cfg = cfg
        self.bucket = next(b for b in check_buckets(cfg) if b.hr == bucket_hr)
        self.split = split
        self.global_seed = global_seed
        self.samples: list[SampleMeta] = []
        for row in index_mod.iter_index(find_eligibility(index_dir)):
            meta = SampleMeta(
                sample_id=str(row["sample_id"]),
                shard=str(row["shard"]),
                rel_path=str(row["rel_path"]),
                width=int(row["width"]),
                height=int(row["height"]),
                is_validation=bool(row["is_validation"]),
            )
            if meta.is_validation != (split == "validation"):
                continue
            if not row["eligible_train"]:
                continue
            if bucket_hr not in (row["eligible_buckets"] or []):
                continue
            self.samples.append(meta)
        if not self.samples:
            raise RuntimeError(f"no {split} samples eligible for HR bucket {bucket_hr} in {index_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def _webp_path(self, meta: SampleMeta) -> Path:
        # danbooru-v2 member: danbooru/5.9/1_2024/<id>.webp → <shard>/<id>.webp
        img_id = meta.rel_path.rsplit("/", 1)[-1]
        return self.webp_dir / meta.shard / img_id

    def decode_hr(self, meta: SampleMeta) -> torch.Tensor:
        """webp → [3, H, W] fp32 in [-1, 1] (full image, no crop)."""
        p = self._webp_path(meta)
        if not p.is_file():
            raise FileNotFoundError(f"webp missing (extract shards first): {p}")
        img = Image.open(p).convert("RGB")
        if (img.width, img.height) != (meta.width, meta.height):
            raise RuntimeError(f"size mismatch for {p}: {img.size} vs ({meta.width}, {meta.height})")
        # .copy(): PIL's __array__ is a read-only view of its internal buffer;
        # torch.from_numpy on a non-writable array is undefined behavior.
        arr = np.asarray(img, dtype=np.uint8).copy()
        t = torch.from_numpy(arr).permute(2, 0, 1).float()
        return t * (2.0 / 255.0) - 1.0

    def crop(self, meta: SampleMeta, data_cycle: int, exposure_index: int) -> tuple[int, int]:
        seed_box = box_seed(meta.sample_id, data_cycle, exposure_index)
        return crop_box(meta.width, meta.height, self.bucket.hr, seed_box)

    def fetch(
        self,
        i: int,
        exposure_index: int = 0,
        data_cycle: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, SampleMeta]:
        """(hr_crop [3,B,B], lq [3,B/4,B/4], meta) — deterministic exposure."""
        meta = self.samples[i % len(self.samples)]
        hr = self.decode_hr(meta)
        x, y = self.crop(meta, data_cycle, exposure_index)
        hr_crop = hr[..., y : y + self.bucket.hr, x : x + self.bucket.hr].contiguous()
        lq, _ = degrade_hr(
            hr_crop,
            self.cfg,
            global_seed=self.global_seed,
            sample_id=meta.sample_id,
            data_cycle=data_cycle,
            exposure_index=exposure_index,
        )
        return hr_crop, lq, meta

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, SampleMeta]:
        """Default exposure (0, 0) for sample ``i``; use ``fetch`` for the
        multi-exposure training schedule."""
        return self.fetch(i)


def make_loader(
    dataset: SRDataset,
    *,
    batch_size: int = 8,
    num_workers: int = 0,
    shuffle: bool = True,
    seed: int = 42,
) -> torch.utils.data.DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        pin_memory=False,
    )


__all__ = [
    "_EXPOSURE_PER_CYCLE",
    "SRDataset",
    "SampleMeta",
    "box_seed",
    "find_eligibility",
    "make_loader",
]
