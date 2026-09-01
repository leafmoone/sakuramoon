"""M1 training dataset: webp → HR crop → deterministic LQ (plan §10-§11).

The dataset reads the M1 index (``sr-eligibility-v1``) and decodes webp
either from the extracted shard directory (``webp/<shard>/<id>.webp``,
legacy / salt5 path) or IN PLACE from the pinned shard tar member at
(``webp_offset``, ``webp_size``) (``tar_dir`` set — the streaming
tar-direct path, 2026-09-01: the corpus is never materialized as a webp
tree; the 512 GiB service cache is the only resident store). Both paths
hand back identical bytes, so the decode is bit-exact across modes. The
dataset then crops the HR square (deterministic offset,
``buckets.crop_box``) and applies the exposure-deterministic degradation
chain (``degradation.degrade_hr``).

``__getitem__`` returns (hr_crop, lq, meta) for the default exposure;
training with a multi-exposure schedule calls :meth:`fetch` with explicit
``exposure_index`` / ``data_cycle`` (resume contract, plan §11.5).
"""

from __future__ import annotations

import io
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from anime_sr.config.schema import Config
from anime_sr.data import index as index_mod
from anime_sr.data.buckets import check_buckets, crop_box
from anime_sr.data.codec_bank import CodecBank
from anime_sr.data.degradation import degrade_hr, exposure_seed
from anime_sr.torchfree_fetch import box_seed  # re-export (see __all__)

_EXPOSURE_PER_CYCLE = 25  # exposures per image before the data cycle advances (§11.5)


def find_eligibility(index_dir: str | Path) -> str:
    """Locate the eligibility table (parquet preferred, JSONL fallback)."""
    d = Path(index_dir)
    for name in ("sr-eligibility-v1.parquet", "sr-eligibility-v1.jsonl"):
        if (d / name).is_file():
            return str(d / name)
    raise FileNotFoundError(f"no eligibility table in {d} (run cli/index_dataset.py first)")


@dataclass(frozen=True)
class SampleMeta:
    sample_id: str
    shard: str
    rel_path: str
    width: int
    height: int
    is_validation: bool
    # §10.4 pool from the index row ("priority" | "regular" | "aux"); the
    # P1 pool sampler (data/pool_sampler.py) consumes this. Default
    # "regular" only protects hand-built test metas.
    sampling_pool: str = "regular"
    # tar-direct read coordinates (streaming path): byte offset / size of
    # the webp member inside the shard tar (0 = extracted-webp path).
    offset: int = 0
    size: int = 0


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
        bank: CodecBank | None = None,
        tar_dir: str | Path | None = None,
    ) -> None:
        if split not in ("train", "validation"):
            raise ValueError(f"split must be train|validation, got {split}")
        if bucket_hr not in {b.hr for b in check_buckets(cfg)}:
            raise ValueError(f"bucket_hr {bucket_hr} not in the frozen bucket table")
        if bank is not None:
            lo, hi = cfg.degradation.codec_bank_batch_fraction
            if not (lo <= cfg.degradation.codec_bank_fraction <= hi):
                raise ValueError(
                    f"codec_bank_fraction {cfg.degradation.codec_bank_fraction} "
                    f"outside the frozen range {lo}-{hi} (plan §11.4)"
                )
        self.webp_dir = Path(webp_dir)
        # Streaming (tar-direct) path: decode webp members in place from the
        # pinned shard tars (no extracted webp tree; the corpus is consumed
        # as a 512 GiB-bounded stream, 2026-09-01). ``webp_dir`` is then a
        # placeholder and must be a valid path argument only.
        self.tar_dir = Path(tar_dir) if tar_dir is not None else None
        self.cfg = cfg
        self.bucket = next(b for b in check_buckets(cfg) if b.hr == bucket_hr)
        self.split = split
        self.global_seed = global_seed
        self.bank = bank
        # §10.5 clean score is a FROZEN offline sidecar (P1-4, 2026-08-29):
        # training never computes/appends it at decode time; the trainer
        # loads the sidecar once at start-up (gate + report).
        self.samples: list[SampleMeta] = []
        n_no_coords = 0
        for row in index_mod.iter_index(find_eligibility(index_dir)):
            if row.get("is_validation") != (split == "validation"):
                continue
            if not row["eligible_train"]:
                continue
            if bucket_hr not in (row["eligible_buckets"] or []):
                continue
            offset = int(row.get("webp_offset") or 0)
            size = int(row.get("webp_size") or 0)
            if self.tar_dir is not None and (offset <= 0 or size <= 0):
                n_no_coords += 1
                continue
            meta = SampleMeta(
                sample_id=str(row["sample_id"]),
                shard=str(row["shard"]),
                rel_path=str(row["rel_path"]),
                width=int(row["width"]),
                height=int(row["height"]),
                is_validation=bool(row["is_validation"]),
                sampling_pool=str(row.get("sampling_pool") or "regular"),
                offset=offset,
                size=size,
            )
            self.samples.append(meta)
        if self.tar_dir is not None and n_no_coords:
            raise RuntimeError(
                f"{n_no_coords} eligible rows lack tar-direct coordinates "
                f"(webp_offset/webp_size) in {index_dir} — rebuild the index "
                f"with the streaming scan (scan_shard_full)"
            )
        if not self.samples:
            raise RuntimeError(f"no {split} samples eligible for HR bucket {bucket_hr} in {index_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def _webp_path(self, meta: SampleMeta) -> Path:
        # danbooru-v2 member: danbooru/5.9/1_2024/<id>.webp → <shard>/<id>.webp
        img_id = meta.rel_path.rsplit("/", 1)[-1]
        return self.webp_dir / meta.shard / img_id

    def _tar_path(self, meta: SampleMeta) -> Path:
        # pin-dir convention: <tar_dir>/<shard> where <shard> is the frozen
        # release-prefixed flat name (the index ``shard`` column)
        return self.tar_dir / meta.shard

    def _open_webp(self, meta: SampleMeta) -> Image.Image:
        """Open the sample's webp: extracted file (webp_dir) or the tar
        member at (offset, size) (tar_dir). Both paths hand back IDENTICAL
        bytes, so the PIL decode downstream is bit-exact across modes."""
        if self.tar_dir is None:
            p = self._webp_path(meta)
            if not p.is_file():
                raise FileNotFoundError(f"webp missing (extract shards first): {p}")
            return Image.open(p)
        p = self._tar_path(meta)
        if not p.is_file():
            raise FileNotFoundError(
                f"tar shard missing (the window driver did not pin it): {p}"
            )
        with p.open("rb") as fh:
            fh.seek(meta.offset)
            data = fh.read(meta.size)
        if len(data) != meta.size:
            raise RuntimeError(
                f"truncated webp member for {meta.sample_id} in {p}: "
                f"got {len(data)} bytes, want {meta.size}"
            )
        return Image.open(io.BytesIO(data))

    def decode_hr(self, meta: SampleMeta) -> torch.Tensor:
        """webp → [3, H, W] fp32 in [-1, 1] (full image, no crop)."""
        img = self._open_webp(meta).convert("RGB")
        if (img.width, img.height) != (meta.width, meta.height):
            raise RuntimeError(
                f"size mismatch for {meta.sample_id} ({self._tar_path(meta) if self.tar_dir else self._webp_path(meta)}): "
                f"{img.size} vs ({meta.width}, {meta.height})"
            )
        # .copy(): PIL's __array__ is a read-only view of its internal buffer;
        # torch.from_numpy on a non-writable array is undefined behavior.
        arr = np.asarray(img, dtype=np.uint8).copy()
        t = torch.from_numpy(arr).permute(2, 0, 1).float()
        out = t * (2.0 / 255.0) - 1.0
        return out

    def decode_hr_timed(self, meta: SampleMeta) -> tuple[torch.Tensor, dict[str, float]]:
        """:meth:`decode_hr` with shard/decode sub-stage wall times (M1 #8
        producer profiling, §13 M3 data-throughput gate).

        Returns ``(hr_full, {"shard": t, "decode": t})`` — bit-exact with
        ``decode_hr`` (same operations, same order), split into the webp
        open/read (``shard``: file open on the extracted-webp path, or
        open+seek+member-read on the tar-direct path) and the PIL decode
        -> tensor (``decode``; the lazy file read lands in this window).
        """
        t_open = time.perf_counter()
        img = self._open_webp(meta)
        t_shard = time.perf_counter() - t_open
        t_dec0 = time.perf_counter()
        img = img.convert("RGB")
        if (img.width, img.height) != (meta.width, meta.height):
            raise RuntimeError(
                f"size mismatch for {meta.sample_id}: {img.size} vs ({meta.width}, {meta.height})"
            )
        arr = np.asarray(img, dtype=np.uint8).copy()
        t = torch.from_numpy(arr).permute(2, 0, 1).float()
        out = t * (2.0 / 255.0) - 1.0
        t_decode = time.perf_counter() - t_dec0
        return out, {"shard": t_shard, "decode": t_decode}

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
        if self.bank is not None:
            seed = exposure_seed(self.global_seed, meta.sample_id, data_cycle, exposure_index)
            frac = self.cfg.degradation.codec_bank_fraction
            if self.bank.bank_fraction_hit(meta.sample_id, seed, frac):
                # §11.4: 10-20% of the batch takes its LQ from the offline
                # real-codec bank instead of the synthetic chain (deterministic
                # per sample + exposure seed; the synthetic lq is kept as the
                # fallback for samples missing from the bank).
                try:
                    arr = self.bank.variants_for(meta.sample_id, seed, k=1)[0]
                except KeyError:
                    arr = None
                if arr is not None:
                    want = (self.bucket.hr // 4, self.bucket.hr // 4)
                    if (arr.shape[0], arr.shape[1]) != want:
                        raise RuntimeError(
                            f"codec bank LQ {arr.shape[1]}x{arr.shape[0]} for {meta.sample_id} "
                            f"does not match bucket {self.bucket.hr} ({want}); rebuild the bank "
                            f"for this bucket"
                        )
                    lq = (
                        torch.from_numpy(arr.copy())
                        .permute(2, 0, 1)
                        .float()
                        .mul(2.0 / 255.0)
                        .sub(1.0)
                        .to(hr_crop.device)
                    )
        return hr_crop, lq, meta

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, SampleMeta]:
        """Default exposure (0, 0) for sample ``i``; use ``fetch`` for the
        multi-exposure training schedule."""
        return self.fetch(i)


def make_loader(
    dataset: torch.utils.data.Dataset,
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
