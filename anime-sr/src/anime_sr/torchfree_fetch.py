"""Torch-free sample fetch for the SPAWN producer pool (2026-09-01).

latent_flow ``producer="spawn"`` runs its worker pool with the ``spawn``
start method: each worker is a FRESH interpreter that must never import
torch. A torch import commits ~5 GiB of virtual address space per
process; the 09-01 salt9 death loop (host kernel
``vm.overcommit_memory=2``, CommitLimit ~453 GiB shared across tenants)
showed that 32-64 torch-holding workers per rank exhaust the host budget
— forked workers counted the parent's torch commit through the fork, and
spawned workers importing torch would commit the same ~5 GiB themselves.

Everything here is numpy/PIL/stdlib. ``load_hr_numpy`` is the torch-free
half of ``SRDataset._open_webp`` + ``decode_hr`` (pipeline.py): same
open/seek/read, same ``Image.open(io.BytesIO(...))``, same
``convert("RGB")``, same ``np.asarray(dtype=np.uint8).copy()``. The
unit test pins bit-exactness against the torch path, and the crop stage
reuses the exact ``crop_box``/``box_seed`` the trainer uses, so a
``producer="spawn"`` batch is bit-identical to the thread/process
producer for the same (sample, step) — only the process boundary moved.

MODULE LOCATION IS LOAD-BEARING (2026-09-01, salt9 probe test): this
module lives at the ``anime_sr`` TOP LEVEL, not in ``anime_sr.data``,
because ``anime_sr.data.__init__`` eagerly imports its submodules
(degradation, pipeline — torch). A spawn worker importing
``anime_sr.data.torchfree_fetch`` would run that package ``__init__``
and load torch in every worker (~5 GiB committed each — the exact VA
death this producer exists to avoid). Only ``anime_sr/__init__.py``
(clean: docstring + ``__version__``) may sit above this module. This
module therefore imports NOTHING from ``anime_sr`` at runtime.
"""

from __future__ import annotations

import hashlib
import io
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image

if TYPE_CHECKING:
    # typing only (never executed): SampleMeta for the annotations
    from anime_sr.data.pipeline import SampleMeta

# ---------------------------------------------------------------------------
# crop-box stream (moved here from pipeline.py, then from data/buckets.py:
# spawn workers must build the deterministic box without importing the
# torch-heavy ``anime_sr.data`` package; pipeline/buckets re-export — the
# hash formats are FROZEN, resume-safe)
# ---------------------------------------------------------------------------


def _blake2b_u64(s: str) -> int:
    """Stable 64-bit hash (platform-independent, stdlib only)."""
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "little")


def box_seed(sample_id: str, data_cycle: int, exposure_index: int) -> int:
    """Crop-box stream: independent from the degradation exposure seed."""
    return _blake2b_u64(f"box|{sample_id}|{data_cycle}|{exposure_index}")


def crop_box(width: int, height: int, hr: int, seed: int) -> tuple[int, int]:
    """Deterministic crop top-left for an HR square of edge ``hr``.

    Center-anchored with a seeded jitter clipped to the valid range, so
    (width, height, hr, seed) reproduces the exact box across resumes.
    x and y use independent hash streams (a single 64-bit value cannot
    supply two unbiased coordinates after one modulo).

    (Canonical home: this torch-free module; ``anime_sr.data.buckets``
    re-exports it — the hash format is FROZEN, resume-safe.)"""
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


# ---------------------------------------------------------------------------
# raw webp -> numpy (bit-exact source of SRDataset.decode_hr)
# ---------------------------------------------------------------------------


def load_hr_numpy(
    meta: SampleMeta,
    tar_dir: str | Path | None,
    webp_dir: str | Path | None,
) -> tuple[np.ndarray, dict[str, float]]:
    """meta -> (H, W, 3) uint8 RGB array + stage times {"shard", "decode"}.

    Stage split mirrors ``SRDataset.decode_hr_timed``: ``shard`` = the webp
    open/read (file open on the extracted path, or open+seek+member-read on
    the tar-direct path — the lazy PIL read is NOT in it), ``decode`` =
    convert + size check + asarray (the lazy file read lands in it)."""
    if tar_dir is None:
        img_id = meta.rel_path.rsplit("/", 1)[-1]
        p = Path(webp_dir) / meta.shard / img_id
        if not p.is_file():
            raise FileNotFoundError(f"webp missing (extract shards first): {p}")
        t_open = time.perf_counter()
        img = Image.open(p)
        t_shard = time.perf_counter() - t_open
    else:
        p = Path(tar_dir) / meta.shard
        if not p.is_file():
            raise FileNotFoundError(
                f"tar shard missing (the window driver did not pin it): {p}"
            )
        t_open = time.perf_counter()
        with p.open("rb") as fh:
            fh.seek(meta.offset)
            data = fh.read(meta.size)
        t_shard = time.perf_counter() - t_open
        if len(data) != meta.size:
            raise RuntimeError(
                f"truncated webp member for {meta.sample_id} in {p}: "
                f"got {len(data)} bytes, want {meta.size}"
            )
        img = Image.open(io.BytesIO(data))
    t_dec0 = time.perf_counter()
    img = img.convert("RGB")
    if (img.width, img.height) != (meta.width, meta.height):
        raise RuntimeError(
            f"size mismatch for {meta.sample_id}: {img.size} "
            f"vs ({meta.width}, {meta.height})"
        )
    arr = np.asarray(img, dtype=np.uint8).copy()
    t_decode = time.perf_counter() - t_dec0
    return arr, {"shard": t_shard, "decode": t_decode}


# ---------------------------------------------------------------------------
# spawn-pool worker body (module-level: picklable by reference for spawn)
# ---------------------------------------------------------------------------

_W: dict[str, Any] = {"tar_dir": None, "webp_dir": None, "bucket_hr": 1024}


def init_worker(
    tar_dir: str | None, webp_dir: str | None, bucket_hr: int
) -> None:
    """Pool initializer: this module (and its imports) must stay
    torch-free — the VA budget of the spawn workers depends on it."""
    _W["tar_dir"] = tar_dir
    _W["webp_dir"] = webp_dir
    _W["bucket_hr"] = int(bucket_hr)


def fetch_crop_numpy_worker(
    meta: SampleMeta,
    step: int,
    exposure_per_cycle: int,
    *,
    tar_dir: str | Path | None,
    webp_dir: str | Path | None,
    bucket_hr: int,
) -> tuple[np.ndarray, dict[str, float]]:
    """Pure core: (meta, step, epc) -> (hr_crop uint8 HWC [B, B, 3],
    stage times).

    The crop box is the trainer's exact deterministic rule
    (``crop_box`` + ``box_seed`` over the dynamic exposure identity);
    the numpy slice is the bit-exact stand-in for the torch slice
    (``hr_full[..., y:y+B, x:x+B]`` after the per-element uint8->fp32
    linear map commutes with the slice). Degradation (seeded torch ops)
    stays on the consumer so the lq is bit-identical to the other
    producers."""
    st: dict[str, float] = {}
    arr, dec = load_hr_numpy(meta, tar_dir, webp_dir)
    st["shard"] = dec["shard"]
    st["decode"] = dec["decode"]
    t_c0 = time.perf_counter()
    b = int(bucket_hr)
    x, y = crop_box(
        meta.width,
        meta.height,
        b,
        box_seed(meta.sample_id, step // exposure_per_cycle, step % exposure_per_cycle),
    )
    crop = np.ascontiguousarray(arr[y : y + b, x : x + b])
    st["crop"] = time.perf_counter() - t_c0
    return crop, st


def fetch_crop_numpy(
    args: tuple[SampleMeta, int, int],
) -> tuple[np.ndarray, dict[str, float]]:
    """Spawn-pool worker body (module-level: picklable by reference for
    spawn); delegates to :func:`fetch_crop_numpy_worker` with the
    pool-initializer state."""
    meta, step, epc = args
    return fetch_crop_numpy_worker(
        meta,
        step,
        epc,
        tar_dir=_W["tar_dir"],
        webp_dir=_W["webp_dir"],
        bucket_hr=_W["bucket_hr"],
    )


def _probe_modules() -> list[str]:
    """Test/ops probe: names of loaded top-level modules (VA-budget
    canary: must never contain 'torch')."""
    import sys

    return sorted({m.split(".")[0] for m in sys.modules})


__all__ = [
    "box_seed",
    "crop_box",
    "fetch_crop_numpy",
    "fetch_crop_numpy_worker",
    "init_worker",
    "load_hr_numpy",
]
