#!/usr/bin/env python3
"""Set C selection builder (deterministic, seed 42) — run on sakrua10.

Regenerated 2026-08-30: the original rgb-eval-c-selection.json was lost with
the old container (OOM re-provisioning).  Same intent as the original plan
("60 real danbooru original small-web images -> LQ 256, human-eval only"):

  - pool: the 3 extracted webp shards under /root/private_data/anime-sr/data/webp
    (29,629 danbooru-5.9 1_2024 originals, persistent volume)
  - size + safety filter from the danbooru-v2 sidecars (same 29,629 ids):
      nsfw == "sfw"  AND  512 <= min(w,h) <= 1024  AND  max(w,h) <= 2048
  - 20 images per pool shard (60 total); per-shard candidates sorted by id;
    a single random.Random(42) drives all samples, shards in ascending order
  - output: /root/private_data/anime-sr/data/c-selection.json
      [{"sid","src","w","h"}, ...]  (src = pool webp path on the persistent
      volume; rgb_eval load_set_c contract)

Usage:
    /usr/local/bin/python3.11 c_selection_builder.py
"""
import json
import random
import tarfile
from pathlib import Path

POOL_SHARDS = [
    "/root/private_data/anime-sr/data/webp/shard-000000.tar",
    "/root/private_data/anime-sr/data/webp/shard-000001.tar",
    "/root/private_data/anime-sr/data/webp/shard-000002.tar",
]
DB_V2 = "/root/private_data/anime-sr/danbooru-v2/data/1_2024"
OUT = "/root/private_data/anime-sr/data/c-selection.json"
PER_SHARD = 20
SEED = 42


def pool_stems() -> dict[str, set[str]]:
    stems: dict[str, set[str]] = {}
    for p in POOL_SHARDS:
        d = Path(p)
        stems[p] = {f.stem for f in d.iterdir() if f.suffix == ".webp"}
    return stems


def sidecar_meta() -> dict[str, tuple[int, int, str]]:
    meta: dict[str, tuple[int, int, str]] = {}
    for i in range(3):
        t = tarfile.open(f"{DB_V2}/shard-{i:06d}.tar")
        for m in t.getmembers():
            if not m.name.endswith(".json"):
                continue
            d = json.load(t.extractfile(m))
            w = d.get("image", {}).get("width")
            h = d.get("image", {}).get("height")
            if w is None or h is None:
                continue
            meta[str(d["id"])] = (w, h, d.get("nsfw", ""))
    return meta


def main() -> None:
    stems = pool_stems()
    meta = sidecar_meta()
    in_pool = set().union(*stems.values())
    rng = random.Random(SEED)
    sel: list[dict] = []
    for p in POOL_SHARDS:  # ascending shard order
        cands = sorted(
            (
                sid
                for sid in stems[p]
                if sid in meta
                and meta[sid][2] == "sfw"
                and 512 <= min(meta[sid][0], meta[sid][1]) <= 1024
                and max(meta[sid][0], meta[sid][1]) <= 2048
            ),
            key=int,
        )
        picks = rng.sample(cands, PER_SHARD) if len(cands) >= PER_SHARD else cands
        for sid in sorted(picks):
            w, h, _ = meta[sid]
            sel.append({"sid": sid, "src": f"{p}/{sid}.webp", "w": w, "h": h})
    Path(OUT).write_text(json.dumps(sel, indent=1))
    print(f"[c-selection] {len(sel)} entries -> {OUT}")
    assert len(sel) == 3 * PER_SHARD, f"expected {3 * PER_SHARD}, got {len(sel)}"
    for e in sel:
        assert Path(e["src"]).is_file(), f"missing {e['src']}"
    print("[c-selection] all src paths verified on volume")


if __name__ == "__main__":
    main()
