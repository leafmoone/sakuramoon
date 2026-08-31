"""Regenerate the SR data-service manifest from the live ModelScope listing.

The upload stage commits shards to leafmoone/SR_v2 incrementally, so the
data-service manifest must mirror the repo's ACTUAL file set: a superset
manifest (full corpus vs partially uploaded repo) makes the prefetch loop
spin on permanent 404s for not-yet-uploaded shards and deadlocks the
startup barrier. Re-run after the upload completes to restore the full
corpus (the previous manifest is kept as a *.full-<n>-<date>.json backup).

Output format: sakuramoon DatasetManifest schema_version 3:
  {aggregates: {bytes, shards}, dataset_id: "<repo>@<revision>",
   schema_version: 3, shards: [{bytes, path}... sorted by path],
   source: {repo_id, revision}}

Run (needs the egress proxy env sourced; token file holds MODELSCOPE_API_TOKEN=):
  python3 sr_manifest_from_repo.py \
    --repo leafmoone/SR_v2 --revision master \
    --token-file /root/private_data/anime-sr/sr-ds/.env.service \
    --out /root/private_data/anime-sr/sr-ds/data/dataset-manifest-sr-v2.json \
    --backup
"""

from __future__ import annotations

import argparse
import json
import os
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

HOST = "modelscope.cn"
PAGE_SIZE = 100
MAX_PAGES = 1000  # 100k files; the SR corpus is ~2k


def _token(token_file: Path) -> str:
    for line in token_file.read_text(encoding="utf-8").splitlines():
        if line.startswith("MODELSCOPE_API_TOKEN="):
            return line.split("=", 1)[1].strip()
    raise SystemExit(f"no MODELSCOPE_API_TOKEN in {token_file}")


def _get_json(url: str, token: str) -> dict:
    request = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Cookie": f"m_session_id={token}",
        },
    )
    last: Exception | None = None
    for attempt in range(5):
        try:
            with urllib.request.urlopen(
                request, timeout=60, context=ssl.create_default_context()
            ) as response:
                if response.status == 200:
                    document = json.loads(response.read())
                    if document.get("Code") not in (None, 200):
                        raise RuntimeError(f"api error: {document.get('Code')} {document.get('Message')}")
                    return document
                last = RuntimeError(f"http {response.status}")
        except Exception as exc:  # noqa: BLE001 - transient network/auth retry
            last = exc
        time.sleep(min(2 ** (attempt + 1), 30))
    raise SystemExit(f"listing failed after retries: {last}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo", default="leafmoone/SR_v2")
    ap.add_argument("--revision", default="master")
    ap.add_argument("--token-file", default="/root/private_data/anime-sr/sr-ds/.env.service")
    ap.add_argument("--out", required=True)
    ap.add_argument("--backup", action="store_true", help="keep the existing manifest as a timestamped backup")
    args = ap.parse_args()

    namespace, _, name = args.repo.partition("/")
    if not namespace or not name:
        raise SystemExit("--repo must be owner/name")
    token = _token(Path(args.token_file))

    shards: dict[str, int] = {}
    page = 1
    while page <= MAX_PAGES:
        query = urllib.parse.urlencode(
            {
                "Revision": args.revision,
                "Recursive": "True",
                "PageNumber": page,
                "PageSize": PAGE_SIZE,
            }
        )
        url = f"https://{HOST}/api/v1/datasets/{namespace}/{name}/repo/tree?{query}"
        document = _get_json(url, token)
        data = document["Data"]
        entries = data["Files"]
        if not isinstance(entries, list) or not entries:
            break
        for entry in entries:
            if not isinstance(entry, dict) or entry.get("Type") in {"tree", "directory"}:
                continue
            path = entry.get("Path")
            size = entry.get("Size")
            if isinstance(path, str) and isinstance(size, int) and path.endswith(".tar"):
                shards[path] = size
        if len(entries) < PAGE_SIZE:
            break
        page += 1
    else:
        raise SystemExit("listing exceeded the page limit; repo may be inconsistent")

    if not shards:
        raise SystemExit("no .tar shards listed; refusing to write an empty manifest")

    ordered = sorted(shards)
    manifest = {
        "aggregates": {
            "bytes": sum(shards.values()),
            "shards": len(ordered),
        },
        "dataset_id": f"{args.repo}@{args.revision}",
        "schema_version": 3,
        "shards": [{"bytes": shards[path], "path": path} for path in ordered],
        "source": {"repo_id": args.repo, "revision": args.revision},
    }

    out = Path(args.out)
    if args.backup and out.exists():
        old = json.loads(out.read_text(encoding="utf-8"))
        n = old.get("aggregates", {}).get("shards", -1)
        backup = out.with_name(
            f"{out.stem}.full-{n}-{time.strftime('%Y%m%d-%H%M%S')}{out.suffix}"
        )
        os.replace(out, backup)
        print(f"[manifest] backed up previous manifest ({n} shards) -> {backup}")

    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_name(f".{out.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(manifest), encoding="utf-8")
    os.replace(temporary, out)

    by_release = Counter(path.split("/")[1] for path in ordered)
    print(
        f"[manifest] wrote {out}: {len(ordered)} shards "
        f"({manifest['aggregates']['bytes'] / 1e9:.1f} GB) "
        f"by release: {dict(by_release)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
