"""Offline codec bank builder (plan §11.4, step 6).

Builds a deterministic bank of real-codec LQ variants (webp/avif/h264/h265/
av1/mpeg4, 4:2:0/4:2:2, range mismatch, double transcode) for a seeded
subset of train-eligible crops. Training workers never touch ffmpeg: at
data time ``SRDataset.fetch`` substitutes the synthetic LQ with a bank
variant for the deterministic 10-20% of the batch.

Usage (server):

    python -m anime_sr.cli.build_codec_bank \
        --config config/base.toml config/data.toml \
        --index-dir <index> --webp-dir <webp> \
        --out-dir <bank> --bucket-hr 1024 \
        --n-crops 10000 --variants 2 --workers 32

Resume-safe: existing variant files with the expected byte size are
skipped; the index is rewritten atomically at the end.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from anime_sr.config.loader import load_config
from anime_sr.config.schema import Config
from anime_sr.data.buckets import crop_box
from anime_sr.data.codec_bank import (
    ENCODE_PROFILES,
    INDEX_NAME,
    VARIANT_DIR,
    CodecVariant,
    encode_variant,
    sample_variants,
)
from anime_sr.data.pipeline import SRDataset, box_seed

BUILD_SEED_DEFAULT = 1_000_003  # recipe seed: bank content is build-seed-stable


def _blake2b_u64(s: str) -> int:
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "little")


def _variant_at(seed: int, j: int) -> CodecVariant:
    """The j-th deterministic draw of the recipe stream (salted by j)."""
    return sample_variants(seed, j + 1)[j]


def select_crops(ds: SRDataset, n_crops: int) -> list[int]:
    """Deterministic crop subset: rank sample ids by blake2b, take n."""
    if n_crops > len(ds.samples):
        raise SystemExit(f"error: --n-crops {n_crops} > eligible samples {len(ds.samples)}")
    order = sorted(range(len(ds.samples)), key=lambda i: _blake2b_u64(f"cbank|{ds.samples[i].sample_id}"))
    return order[:n_crops]


def _crop_rgb(
    sample_id: str,
    shard: str,
    rel_path: str,
    width: int,
    height: int,
    webp_dir: str,
    bucket_hr: int,
) -> np.ndarray:
    """Decode + crop one HR crop (same crop_box derivation as SRDataset)."""
    img_id = rel_path.rsplit("/", 1)[-1]
    p = Path(webp_dir) / shard / img_id
    if not p.is_file():
        raise FileNotFoundError(f"webp missing: {p}")
    img = Image.open(p).convert("RGB")
    if (img.width, img.height) != (width, height):
        raise RuntimeError(f"size mismatch for {p}: {img.size} vs ({width}, {height})")
    x, y = crop_box(width, height, bucket_hr, box_seed(sample_id, 0, 0))
    arr = np.asarray(img, dtype=np.uint8)[y : y + bucket_hr, x : x + bucket_hr]
    if arr.shape[0] != bucket_hr or arr.shape[1] != bucket_hr:
        raise RuntimeError(f"crop overflow for {p}: box ({x},{y}) {bucket_hr}x{bucket_hr} in {width}x{height}")
    return arr.copy()


def build_crop_job(job: dict[str, Any]) -> dict[str, Any]:
    """One pool task: decode a crop, encode its k variants, write the bins.

    ``job`` keys: sample_id, shard, rel_path, width, height, webp_dir,
    out_dir, bucket_hr, lq, seed, k, codecs.
    Returns {"sample_id", "entries", "skipped", "encoded"}.
    """
    sid = job["sample_id"]
    seed = _blake2b_u64(f"cbank|{sid}|{job['seed']}")
    allowed = set(job["codecs"])
    # walk the draw stream, keeping only variants whose codec is in the subset
    variants: list[CodecVariant] = []
    j = 0
    while len(variants) < job["k"] and j < 64:
        v = _variant_at(seed, j)
        if v.codec in allowed:
            variants.append(v)
        j += 1
    if not variants:
        raise RuntimeError(f"no variants for {sid} within codec subset {sorted(allowed)}")

    crop = _crop_rgb(
        sid,
        job["shard"],
        job["rel_path"],
        job["width"],
        job["height"],
        job["webp_dir"],
        job["bucket_hr"],
    )
    out = Path(job["out_dir"]) / VARIANT_DIR
    out.mkdir(parents=True, exist_ok=True)
    work = Path(job["out_dir"]) / "_work" / sid
    lq_w = lq_h = job["lq"]
    entries: list[dict] = []
    n_skip = n_enc = 0
    for v in variants:
        p = out / f"{v.variant_id}.bin"
        want = lq_w * lq_h * 3
        if p.is_file() and p.stat().st_size == want:
            n_skip += 1
        else:
            raw, n_bytes = encode_variant(crop, v, lq_w, lq_h, work)
            assert n_bytes == want
            tmp = p.with_suffix(".part")
            tmp.write_bytes(raw.read_bytes())
            tmp.rename(p)
            n_enc += 1
        entries.append(
            {
                "variant_id": v.variant_id,
                "codec": v.codec,
                "pix_fmt": v.pix_fmt,
                "range_mismatch": v.range_mismatch,
                "passes": v.passes,
                "quality": v.quality,
                "lq_w": lq_w,
                "lq_h": lq_h,
                "bytes": want,
            }
        )
    # drop the per-crop scratch (keep only the .bin variants + the index)
    shutil.rmtree(work, ignore_errors=True)
    return {"sample_id": sid, "entries": entries, "skipped": n_skip, "encoded": n_enc}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build the offline real-codec LQ bank (§11.4).")
    ap.add_argument("--config", nargs="+", required=True, help="TOML config files")
    ap.add_argument("--index-dir", required=True)
    ap.add_argument("--webp-dir", required=True)
    ap.add_argument("--out-dir", required=True, help="bank output dir")
    ap.add_argument("--bucket-hr", type=int, default=1024)
    ap.add_argument("--n-crops", type=int, default=10_000)
    ap.add_argument("--variants", type=int, default=2, help="versions per crop (1-2, §11.4)")
    ap.add_argument("--codecs", default="webp,avif,h264,h265,av1,mpeg4")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=BUILD_SEED_DEFAULT, help="bank recipe seed")
    args = ap.parse_args(argv)

    codecs = [c.strip() for c in args.codecs.split(",") if c.strip()]
    bad = [c for c in codecs if c not in ENCODE_PROFILES]
    if bad:
        raise SystemExit(f"error: unknown codec(s) {bad}; supported: {sorted(ENCODE_PROFILES)}")
    if not (1 <= args.variants <= 2):
        raise SystemExit(f"error: --variants must be 1-2 (§11.4), got {args.variants}")

    cfg: Config = load_config(*args.config)
    ds = SRDataset(args.index_dir, args.webp_dir, cfg, bucket_hr=args.bucket_hr, split="train")
    plan_min = cfg.degradation.codec_bank_hr_crops_min
    if args.n_crops < plan_min:
        print(f"[warn] n-crops {args.n_crops} < plan min {plan_min} (reduced bank for the small Phase I)")
    picks = select_crops(ds, args.n_crops)
    print(
        f"[bank] {len(ds.samples)} eligible samples for bucket {args.bucket_hr}; "
        f"building {args.n_crops} crops x {args.variants} variants"
    )

    out_dir = Path(args.out_dir)
    jobs = []
    for i in picks:
        m = ds.samples[i]
        jobs.append(
            {
                "sample_id": m.sample_id,
                "shard": m.shard,
                "rel_path": m.rel_path,
                "width": m.width,
                "height": m.height,
                "webp_dir": str(args.webp_dir),
                "out_dir": str(out_dir),
                "bucket_hr": args.bucket_hr,
                "lq": args.bucket_hr // 4,
                "seed": args.seed,
                "k": args.variants,
                "codecs": codecs,
            }
        )

    t0 = time.time()
    results: list[dict[str, Any]] = []
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(build_crop_job, j): j for j in jobs}
        for fut in as_completed(futs):
            r = fut.result()  # propagates worker tracebacks (no squashed errors)
            results.append(r)
            done += 1
            if done % 100 == 0 or done == len(jobs):
                print(f"[bank] {done}/{len(jobs)} crops ({time.time() - t0:.0f}s)")

    # atomic index write
    samples_idx = {r["sample_id"]: r["entries"] for r in sorted(results, key=lambda r: r["sample_id"])}
    doc = {"version": 1, "bucket_hr": args.bucket_hr, "seed": args.seed, "codecs": codecs, "samples": samples_idx}
    p = out_dir / INDEX_NAME
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(doc), encoding="utf-8")
    tmp.rename(p)
    n_variants = sum(len(e) for e in samples_idx.values())
    print(f"[bank] wrote {p} ({len(samples_idx)} samples, {n_variants} variants)")
    cc = Counter(e["codec"] for entries in samples_idx.values() for e in entries)
    print("[bank] codec mix: " + ", ".join(f"{k}:{v}" for k, v in sorted(cc.items())))
    print(f"[bank] done in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
