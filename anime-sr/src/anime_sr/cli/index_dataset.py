"""CLI: build the M1 index from danbooru-v2 shards (plan §10, step 4).

Usage (on the training host, after the shards are downloaded):

    python -m anime_sr.cli.index_dataset \
        --shard-dir /root/private_data/anime-sr/danbooru-v2/data/1_2024 \
        --out-dir /root/private_data/anime-sr/data/index \
        --config anime-sr/config/base.toml anime-sr/config/data.toml \
        [--validation-permille 300]

Progress + traceback on error (repo rule: no JSON-squashed errors).
"""

from __future__ import annotations

import argparse
import glob
import sys
import tarfile
from pathlib import Path

from anime_sr.config.loader import load_config
from anime_sr.data import index


def extract_shards(shard_dir: str, webp_dir: str, progress_every: int = 1000) -> int:
    """Extract webp members to ``webp_dir/<shard>/<id>.webp`` (resume-safe).

    Needed by the pipeline (fast random access vs tar streaming). Returns
    the number of files extracted.
    """
    out_root = Path(webp_dir)
    n_out = 0
    for shard in sorted(glob.glob(str(Path(shard_dir) / "shard-*.tar"))):
        dest = out_root / Path(shard).name
        dest.mkdir(parents=True, exist_ok=True)
        n_in_shard = 0
        with tarfile.open(shard, "r") as tf:
            for member in tf.getmembers():
                if not (member.isfile() and member.name.endswith(".webp")):
                    continue
                img_id = member.name.rsplit("/", 1)[-1]
                target = dest / img_id
                if target.is_file() and target.stat().st_size == member.size:
                    continue  # resume: skip already-extracted (byte check)
                f = tf.extractfile(member)
                if f is None:
                    raise RuntimeError(f"cannot read {member.name} from {shard}")
                tmp = target.with_suffix(".part")
                tmp.write_bytes(f.read())
                tmp.rename(target)
                n_out += 1
                n_in_shard += 1
                if n_in_shard % progress_every == 0:
                    print(f"[extract] {Path(shard).name}: {n_in_shard} webp extracted", flush=True)
        print(f"[extract] {Path(shard).name}: done ({n_in_shard} in shard)")
    print(f"[extract] total {n_out} webp → {out_root}")
    return n_out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build the M1 data index (plan §10).")
    ap.add_argument("--shard-dir", required=True, help="directory with shard-*.tar")
    ap.add_argument("--out-dir", required=True, help="index output directory")
    ap.add_argument("--config", nargs="+", default=["anime-sr/config/base.toml", "anime-sr/config/data.toml"])
    ap.add_argument("--validation-permille", type=int, default=None, help="override split fraction (‰)")
    ap.add_argument("--extract", action="store_true", help="extract webp for the pipeline")
    ap.add_argument("--webp-dir", default=None, help="extraction target (default <out-dir>/../webp)")
    args = ap.parse_args(argv)

    cfg = load_config(*args.config)
    if args.validation_permille is not None:
        cfg.validation.validation_permille = args.validation_permille

    shards = sorted(glob.glob(str(Path(args.shard_dir) / "shard-*.tar")))
    if not shards:
        print(f"error: no shard-*.tar under {args.shard_dir}", file=sys.stderr)
        return 2

    # Uncaught exceptions propagate (repo rule: CLI prints the traceback,
    # no JSON-squashed errors) → interpreter exit code 1.
    report = index.build_index(shards, cfg, args.out_dir)
    print(f"[index] wrote {args.out_dir} (eligibility writer: {report['writer']['sr-eligibility-v1']})")
    if args.extract:
        webp_dir = args.webp_dir or str(Path(args.out_dir).parent / "webp")
        extract_shards(args.shard_dir, webp_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
