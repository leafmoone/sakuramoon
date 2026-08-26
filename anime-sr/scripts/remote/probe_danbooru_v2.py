"""Danbooru-v2 shard probe: validate ModelScope auth + ranged download, size the corpus.

Stdlib-only (urllib). Mirrors the G1 downloader's API contract:
  listing: GET /api/v1/datasets/{repo}/repo/tree?Revision=...&Recursive=True
  blob:    GET /api/v1/datasets/{repo}/repo?Revision=...&FilePath=...
  auth:    Authorization: Bearer <token> + Cookie: m_session_id=<token>

Run:  env2sh-generated env sourced first (MODELSCOPE_API_TOKEN + https_proxy set).
"""

from __future__ import annotations

import json
import os
import tarfile
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

HOST = "modelscope.cn"
REPO = "leafmoone/webdataset_danbooru_v2"
REV = "master"
SHARD = "data/1_2024/shard-000000.tar"
SHARD_BYTES = 2_148_085_760
OUT = "/root/group_data/anime-sr/danbooru-v2/data/1_2024/shard-000000.tar"
CHUNKS = 4
IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp")
META_EXTS = (".json", ".txt")


def get(url: str, rng: tuple[int, int] | None = None) -> urllib.request.addinfourl:
    req = urllib.request.Request(url, headers={
        "Authorization": f"Bearer {os.environ['MODELSCOPE_API_TOKEN']}",
        "Cookie": f"m_session_id={os.environ['MODELSCOPE_API_TOKEN']}",
    })
    if rng is not None:
        req.add_header("Range", f"bytes={rng[0]}-{rng[1]}")
    return urllib.request.urlopen(req, timeout=600)


def main() -> None:
    print("step 1: listing probe", flush=True)
    tree = (
        f"https://{HOST}/api/v1/datasets/{REPO}/repo/tree"
        f"?Revision={REV}&Recursive=True&PageNumber=1&PageSize=5"
    )
    doc = json.load(get(tree))
    entries = doc["Data"]["Files"]
    tars = [e for e in entries if str(e.get("Path", "")).endswith(".tar")]
    print(f"listing ok: {len(entries)} entries on page 1, tars={len(tars)}", flush=True)
    for e in entries[:3]:
        print("  sample:", e.get("Path"), e.get("Size"), flush=True)

    print("step 2: ranged download of", SHARD, flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    part = OUT + ".part"
    blob = (
        f"https://{HOST}/api/v1/datasets/{REPO}/repo"
        f"?Revision={REV}&FilePath={urllib.parse.quote(SHARD)}"
    )

    def fetch(i: int) -> None:
        s, e = i * SHARD_BYTES // CHUNKS, (i + 1) * SHARD_BYTES // CHUNKS - 1
        r = get(blob, (s, e))
        data = r.read()
        want = e - s + 1
        if len(data) != want:
            raise SystemExit(f"range chunk {i}: got {len(data)} want {want}")
        with open(part, "r+b") as f:
            f.seek(s)
            f.write(data)
        print(f"  chunk {i + 1}/{CHUNKS} ok ({s}-{e})", flush=True)

    with open(part, "wb") as f:
        f.truncate(SHARD_BYTES)
    with ThreadPoolExecutor(CHUNKS) as pool:
        list(pool.map(fetch, range(CHUNKS)))
    os.replace(part, OUT)

    print("step 3: counting entries", flush=True)
    n_img = n_meta = 0
    names: list[str] = []
    with tarfile.open(OUT) as t:
        for m in t.getmembers():
            low = m.name.lower()
            if low.endswith(IMG_EXTS):
                n_img += 1
                if len(names) < 8:
                    names.append(m.name)
            elif low.endswith(META_EXTS):
                n_meta += 1
    print(
        f"PROBE RESULT images={n_img} meta={n_meta} per shard (~{SHARD_BYTES // n_img // 1024} KiB/img)\n"
        f"sample: {names}",
        flush=True,
    )
    print(
        f"SHARDS_FOR_200K={-(-200_000 // n_img)} SHARDS_FOR_500K={-(-500_000 // n_img)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
