"""Tar-direct canary: bit-exact decode comparison, file mode vs tar mode.

For a sample set drawn from the REAL corpus (default: the first N eligible
train samples of the given bucket), decode the same image twice —

  * file mode:  SRDataset(tar_dir=None)   -> Image.open(webp_dir/<shard>/<id>.webp)
  * tar mode:   SRDataset(tar_dir=shards) -> seek+read of the webp MEMBER
    inside the pinned shard tar

and assert torch.equal on the decode_hr outputs. PASS = every tensor
identical; the tar-direct path is then drop-in safe for training (the
slot stream itself is untouched by this refactor).

Usage (salt8, DTK python + SR repo layout):
  python3 tar_mode_canary.py \
      --index-dir /root/private_data/anime-sr/srv2-run/data/index \
      --webp-dir  /root/private_data/anime-sr/srv2-run/data/webp \
      --shard-dir /root/private_data/anime-sr/srv2-run/tars \
      --code-src  /root/private_data/anime-sr/sakuramoon-dev/anime-sr/src \
      --bucket-hr 1024 --num 48
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-dir", required=True)
    ap.add_argument("--webp-dir", required=True, help="extracted webp tree (file mode)")
    ap.add_argument("--shard-dir", required=True, help="pinned shard tar dir (tar mode)")
    ap.add_argument("--code-src", required=True)
    ap.add_argument("--bucket-hr", type=int, default=1024)
    ap.add_argument("--num", type=int, default=48, help="samples to compare per mode")
    args = ap.parse_args(argv)

    import torch

    sys.path.insert(0, args.code_src)
    from anime_sr.config.loader import load_config
    from anime_sr.data.pipeline import SRDataset

    cfg = load_config(
        str(Path(__file__).resolve().parents[3] / "anime-sr/config/base.toml"),
        str(Path(__file__).resolve().parents[3] / "anime-sr/config/data.toml"),
    )

    # one index, two data planes over the same rows
    ds_file = SRDataset(
        args.index_dir, args.webp_dir, cfg, bucket_hr=args.bucket_hr, split="train"
    )
    ds_tar = SRDataset(
        args.index_dir, args.shard_dir, cfg,
        bucket_hr=args.bucket_hr, split="train", tar_dir=args.shard_dir,
    )
    if len(ds_file.samples) != len(ds_tar.samples):
        print(
            f"[canary] FAIL: sample counts differ "
            f"(file={len(ds_file.samples)} tar={len(ds_tar.samples)})",
            flush=True,
        )
        return 1
    if any(
        a.sample_id != b.sample_id for a, b in zip(ds_file.samples, ds_tar.samples)
    ):
        print("[canary] FAIL: sample order differs between modes", flush=True)
        return 1

    n = min(args.num, len(ds_file.samples))
    print(f"[canary] comparing {n} samples (bucket {args.bucket_hr}) ...", flush=True)
    t0 = time.time()
    for k in range(n):
        meta_f = ds_file.samples[k]
        meta_t = ds_tar.samples[k]
        a, _t_file = ds_file.decode_hr_timed(meta_f)
        b, _t_tar = ds_tar.decode_hr_timed(meta_t)
        if a.shape != b.shape:
            print(
                f"[canary] FAIL sample {k} ({meta_f.sample_id}): "
                f"shape {tuple(a.shape)} != {tuple(b.shape)}",
                flush=True,
            )
            return 1
        if not torch.equal(a, b):
            diff = (a - b).abs()
            print(
                f"[canary] FAIL sample {k} ({meta_f.sample_id}): tensors differ "
                f"(max abs diff {diff.max().item():.3e})",
                flush=True,
            )
            return 1
    print(
        f"[canary] PASS: {n}/{n} samples bit-identical (file vs tar) "
        f"in {time.time() - t0:.1f}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
