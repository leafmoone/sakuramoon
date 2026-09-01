"""Extract the file-mode webp comparison tree for the bit-exact canary.

``tar_mode_canary.py`` compares file-mode vs tar-mode decoding of the first
N eligible train samples. File mode needs the extracted layout
``<webp-dir>/<shard>/<img-id>.webp`` (pipeline ``_webp_path``; the shard
column carries the ``.tar`` suffix). This helper builds the SAME file-mode
SRDataset the canary builds (same index rows, same bucket/split filters,
same order), then extracts just the webp members those N samples need
from the pinned shard tars — the corpus is never fully materialized; the
tree stays a small canary artifact.

Run (after the index pass and before tar_mode_canary.py):
  python3 extract_canary_webp.py \
    --index-dir /root/private_data/anime-sr/data/index \
    --shard-dir /root/srv2-run/pins \
    --webp-dir  /root/srv2-run/webp-canary \
    --code-src  /root/private_data/anime-sr/sakuramoon-dev/anime-sr/src \
    --bucket-hr 1024 --num 48
"""

from __future__ import annotations

import argparse
import sys
import tarfile
from collections import OrderedDict
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--index-dir", required=True)
    ap.add_argument("--shard-dir", required=True, help="pinned flat shard tar dir")
    ap.add_argument("--webp-dir", required=True, help="output file-mode webp tree root")
    ap.add_argument("--code-src", required=True, help="anime-sr/src")
    ap.add_argument("--bucket-hr", type=int, default=1024)
    ap.add_argument("--num", type=int, default=48)
    args = ap.parse_args()

    sys.path.insert(0, args.code_src)
    from anime_sr.config.loader import load_config
    from anime_sr.data.pipeline import SRDataset

    # Identical construction to tar_mode_canary's ds_file (file mode):
    # same rows, same filters, same order -> same first-N selection.
    cfg = load_config(
        str(Path(__file__).resolve().parents[3] / "anime-sr/config/base.toml"),
        str(Path(__file__).resolve().parents[3] / "anime-sr/config/data.toml"),
    )
    ds = SRDataset(args.index_dir, args.webp_dir, cfg, bucket_hr=args.bucket_hr, split="train")
    metas = ds.samples[: args.num]
    if not metas:
        print("no eligible train samples; check the index and bucket", flush=True)
        return 1

    # Shards in first-appearance order; per-shard the member names needed.
    needed: OrderedDict[str, set[str]] = OrderedDict()
    for meta in metas:
        needed.setdefault(meta.shard, set()).add(meta.rel_path)

    webp_root = Path(args.webp_dir)
    shard_dir = Path(args.shard_dir)
    total_members = 0
    for shard, members in needed.items():
        tar_path = shard_dir / shard
        if not tar_path.is_file():
            print(
                f"[extract] FAIL: pinned tar missing (window driver not pinning yet?): {tar_path}",
                flush=True,
            )
            return 2
        out_dir = webp_root / shard
        out_dir.mkdir(parents=True, exist_ok=True)
        have = {p.name for p in out_dir.glob("*.webp")}
        missing = {m.rsplit("/", 1)[-1] for m in members} - have
        if not missing:
            print(f"[extract] {shard}: {len(members)} members already present", flush=True)
            continue
        with tarfile.open(tar_path) as tf:
            for member_name in sorted(missing):
                m = tf.getmember(member_name)
                fh = tf.extractfile(m)
                if fh is None:
                    print(
                        f"[extract] FAIL {shard}: member {member_name} is not a file",
                        flush=True,
                    )
                    return 2
                with open(out_dir / member_name, "wb") as out:
                    out.write(fh.read())
        total_members += len(missing)
        print(
            f"[extract] {shard}: wrote {len(missing)} members ({len(members)} needed)",
            flush=True,
        )

    print(
        f"[extract] done: {len(metas)} samples across {len(needed)} shards, "
        f"{total_members} members (fresh) into {webp_root}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
