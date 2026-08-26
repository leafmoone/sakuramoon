"""Offline Mage-VAE z_hr pre-encoder (plan §4.3, §20 step 2).

Encodes the frozen Mage-VAE posterior-mean latent of every selected HR crop
(z_hr = E(hr_crop), 128ch at bucket/16) into a resume-safe fp16 store that
the latent flow trainer (M3/M4) reads at data time. The LQ anchor z_lr is
NOT stored: it follows the per-exposure degradation draw (plan §4.3) and is
computed at data time (§M4 async prefetch).

Crop selection mirrors the codec bank builder (blake2b rank of sample ids),
default salt ``cbank`` so the latent set aligns with the codec bank's crop
set. Crops are decoded exactly like ``SRDataset`` (PIL RGB, [-1,1] fp32,
exposure (0,0) crop box) so pre-encoded z_hr matches the training target.

Usage (server):

    python -m anime_sr.cli.encode_latents \
        --config config/base.toml config/data.toml \
        --index-dir <index> --webp-dir <webp> \
        --out-dir <latents> --bucket-hr 1024 \
        --n-crops 10000 --vae <mage-vae.safetensors> --batch 8

Resume-safe: existing latents with the expected byte size are skipped; the
index is written atomically at the end.
"""

from __future__ import annotations

import argparse
import hashlib
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from anime_sr.config.loader import load_config
from anime_sr.config.schema import Config
from anime_sr.data.buckets import crop_box
from anime_sr.data.latent_store import LatentStore
from anime_sr.data.pipeline import SampleMeta, SRDataset, box_seed
from anime_sr.vae.mage import load_frozen_vae


def _blake2b_u64(s: str) -> int:
    return int.from_bytes(
        hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "little"
    )


def select_crops(
    samples: Sequence[SampleMeta], n_crops: int, salt: str = "cbank"
) -> list[int]:
    """Deterministic crop subset: rank sample ids by blake2b(salt|sid), take n.

    Returns indices into ``samples`` (stable: the dataset order is kept, so
    the caller can index its dataset/latent set directly).
    """
    if n_crops > len(samples):
        raise SystemExit(
            f"error: --n-crops {n_crops} > eligible samples {len(samples)}"
        )
    order = sorted(
        range(len(samples)),
        key=lambda i: _blake2b_u64(f"{salt}|{samples[i].sample_id}"),
    )
    return order[:n_crops]


def _hr_crop_tensor(
    meta: SampleMeta,
    webp_dir: str,
    bucket_hr: int,
) -> torch.Tensor:
    """Decode + crop one HR crop, byte-identical to SRDataset.fetch (0,0)."""
    img_id = meta.rel_path.rsplit("/", 1)[-1]
    p = Path(webp_dir) / meta.shard / img_id
    if not p.is_file():
        raise FileNotFoundError(f"webp missing: {p}")
    img = Image.open(p).convert("RGB")
    if (img.width, img.height) != (meta.width, meta.height):
        raise RuntimeError(
            f"size mismatch for {p}: {img.size} vs ({meta.width}, {meta.height})"
        )
    arr = np.asarray(img, dtype=np.uint8).copy()
    t = torch.from_numpy(arr).permute(2, 0, 1).float()
    t = t * (2.0 / 255.0) - 1.0
    x, y = crop_box(meta.width, meta.height, bucket_hr, box_seed(meta.sample_id, 0, 0))
    return t[..., y : y + bucket_hr, x : x + bucket_hr].contiguous()


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Pre-encode z_hr with the frozen Mage-VAE (plan §4.3)."
    )
    ap.add_argument("--config", nargs="+", required=True, help="TOML config files")
    ap.add_argument("--index-dir", required=True)
    ap.add_argument("--webp-dir", required=True)
    ap.add_argument("--out-dir", required=True, help="latent store output dir")
    ap.add_argument("--bucket-hr", type=int, default=1024)
    ap.add_argument("--n-crops", type=int, default=10_000)
    ap.add_argument(
        "--vae", default="", help="Mage-VAE weights (default: cfg.vae.path)"
    )
    ap.add_argument(
        "--select-salt",
        default="cbank",
        help="crop-selection salt (cbank = codec-bank set)",
    )
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args(argv)

    cfg: Config = load_config(*args.config)
    vae_path = args.vae or cfg.vae.path
    if not vae_path:
        raise SystemExit(
            "error: VAE weights missing: pass --vae or set [vae].path in a config overlay"
        )

    ds = SRDataset(
        args.index_dir, args.webp_dir, cfg, bucket_hr=args.bucket_hr, split="train"
    )
    picks = select_crops(ds.samples, args.n_crops, args.select_salt)
    metas = [ds.samples[i] for i in picks]
    print(
        f"[latents] {len(ds.samples)} eligible samples for bucket {args.bucket_hr}; "
        f"encoding {len(metas)} crops (salt={args.select_salt!r}, batch={args.batch}, device={args.device})"
    )

    vae = load_frozen_vae(vae_path, device=args.device, dtype=torch.bfloat16)
    store = LatentStore(args.out_dir, args.bucket_hr)

    n_skip = n_enc = 0
    z_sum = 0.0
    z_cnt = 0
    t0 = time.time()
    for start in range(0, len(metas), args.batch):
        chunk = metas[start : start + args.batch]
        crops = torch.stack(
            [_hr_crop_tensor(m, args.webp_dir, args.bucket_hr) for m in chunk]
        ).to(device=args.device, dtype=torch.bfloat16)
        z = vae.encode(crops)  # [B, 128, g, g] bf16, no_grad, frozen
        for m, z_i in zip(chunk, z):
            if store.write(m.sample_id, z_i):
                n_enc += 1
                z_sum += float(z_i.abs().mean().item())
                z_cnt += 1
            else:
                n_skip += 1
        done = min(start + args.batch, len(metas))
        if done % 200 < args.batch or done == len(metas):
            print(
                f"[latents] {done}/{len(metas)} (enc {n_enc} skip {n_skip} "
                f"{time.time() - t0:.0f}s)"
            )

    store.finalize_index([m.sample_id for m in metas])
    mean_abs = z_sum / z_cnt if z_cnt else float("nan")
    if n_enc and not (0.01 < mean_abs < 10.0):
        raise SystemExit(
            f"error: z_hr sanity check failed (mean |z| = {mean_abs:.4f}); "
            "check the VAE weights / bucket / dtype"
        )
    print(
        f"[latents] done: {n_enc} encoded, {n_skip} resumed, mean|z|={mean_abs:.4f} in {time.time() - t0:.0f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
