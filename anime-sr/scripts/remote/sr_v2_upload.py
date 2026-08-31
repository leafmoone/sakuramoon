"""SR_v2 build stage 3: upload finished shards to leafmoone/SR_v2.

Mechanics (modelscope_hub internals, no git):
  * LFS blobs: validate_blobs (dedupe by sha256) -> PUT to the presigned URL.
    Blob PUTs run in a worker pool (parallel-safe, server-side dedupe).
  * Commits: create_commit (LFS pointer) runs SERIALLY on the main loop so
    concurrent commits can never race on master.

Sidecar protocol (written by the extract stage / this stage):
  <tar>.sha256  sha256 of the tar (from extract; recomputed here if missing)
  <tar>.done    tar is complete
  <tar>.blob    blob is uploaded / already on the server
  <tar>.commit  pointer is committed to master

Idempotent: every stage is skipped when its marker exists; safe to restart.

Run (needs LD_LIBRARY_PATH for the torch import chain, token file):
  python3.11 srv2_upload.py \
    --repo leafmoone/SR_v2 \
    --out-dir /root/private_data/anime-sr/sr-v2-build/out \
    --blob-workers 12
"""

from __future__ import annotations

import argparse
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(32 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo", default="leafmoone/SR_v2")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--blob-workers", type=int, default=12)
    ap.add_argument("--token-file", default="/root/private_data/anime-sr/.ms-token")
    ap.add_argument("--raw-dir", default=None, help="if set: corrupted outputs are cleared for re-extraction")
    ap.add_argument("--idle-limit", type=int, default=30 * 60)
    args = ap.parse_args()

    import os

    tok = None
    for line in Path(args.token_file).read_text(encoding="utf-8").splitlines():
        if line.startswith("MODELSCOPE_API_TOKEN="):
            tok = line.split("=", 1)[1].strip()
    if not tok:
        raise SystemExit(f"no token in {args.token_file}")
    os.environ["MODELSCOPE_API_TOKEN"] = tok

    from modelscope_hub.api import HubApi

    h = HubApi(token=tok)
    client = h.uploader._client  # LegacyClient: validate_blobs / upload_blob / create_commit
    repo_id = args.repo
    repo_type = "dataset"
    out_dir = Path(args.out_dir)

    def find_ready(marker: str, exclude: str | None = None) -> list[Path]:
        res = []
        for p in out_dir.rglob(f"*.{marker}"):
            tar = p.with_name(p.name[: -len(f".{marker}")])
            if not tar.exists() or tar.suffix != ".tar":
                continue
            if exclude is not None and (tar.parent / (tar.name + f".{exclude}")).exists():
                continue
            res.append(tar)
        return res

    raw_dir = Path(args.raw_dir) if args.raw_dir else None

    def blob_worker(tar: Path) -> tuple[str, str]:
        rel = tar.relative_to(out_dir).as_posix()
        sha_file = tar.parent / (tar.name + ".sha256")
        sha = sha_file.read_text(encoding="utf-8").strip() if sha_file.exists() else None
        real = _sha256_file(tar)
        if sha is None:
            sha_file.write_text(real + "\n", encoding="utf-8")
        elif sha != real:
            # output corrupted after extraction (concurrent re-extraction race):
            # clear it and the raw extract marker so the extract stage redoes it
            print(f"[up] CORRUPT {rel}: sidecar != file sha; clearing for re-extraction", flush=True)
            for mark in ("", ".done", ".blob", ".commit", ".sha256"):
                (tar.parent / (tar.name + mark)).unlink(missing_ok=True)
            if raw_dir is not None:
                srcbase = tar.name[: tar.name.index("-p2-")]
                d = tar.relative_to(out_dir).parent
                raw_shard = raw_dir / d / (srcbase + ".tar")
                raw_shard.with_name(raw_shard.name + ".extracted").unlink(missing_ok=True)
            return ("requeue", rel)
        size = tar.stat().st_size
        for attempt in range(5):
            try:
                need = client.validate_blobs(repo_id, repo_type, [{"oid": sha, "size": size}])
                href = need.get(sha)
                if href:
                    with open(tar, "rb") as fh:
                        client.upload_blob(href, data=fh, size=size, timeout=(60, 600))
                (tar.parent / (tar.name + ".blob")).touch()
                return ("blob-ok", rel)
            except Exception as exc:  # noqa: BLE001 - retried per-file
                print(f"[up] blob retry {attempt + 1}/5 {rel}: {type(exc).__name__} {str(exc)[:160]}", flush=True)
                time.sleep(min(2 ** (attempt + 1), 60))
        return ("blob-fail", rel)

    def commit_one(tar: Path) -> tuple[str, str]:
        rel = tar.relative_to(out_dir).as_posix()
        sha = (tar.parent / (tar.name + ".sha256")).read_text(encoding="utf-8").strip()
        size = tar.stat().st_size
        for attempt in range(5):
            try:
                client.create_commit(
                    repo_id=repo_id,
                    repo_type=repo_type,
                    operations=[
                        {
                            "action": "create",
                            "path": rel,
                            "type": "lfs",
                            "size": size,
                            "sha256": sha,
                            "content": "",
                            "encoding": "",
                        }
                    ],
                    commit_message=f"SR_v2: {rel}",
                    revision="master",
                )
                (tar.parent / (tar.name + ".commit")).touch()
                return ("commit-ok", rel)
            except Exception as exc:  # noqa: BLE001 - serial committer retries
                print(f"[up] commit retry {attempt + 1}/5 {rel}: {type(exc).__name__} {str(exc)[:160]}", flush=True)
                time.sleep(min(2 ** (attempt + 1), 60))
        return ("commit-fail", rel)

    # restart-safe: count markers from previous runs
    committed = 0
    committed_bytes = 0
    for cm in out_dir.rglob("*.commit"):
        tar = cm.with_name(cm.name[: -len(".commit")])
        if tar.exists():
            committed += 1
            committed_bytes += tar.stat().st_size
    blob_fails: list[str] = []
    commit_fails: list[str] = []
    t0 = time.monotonic()
    last_activity = t0
    print(f"[up] start: repo={repo_id} already-committed={committed} ({committed_bytes / 1e9:.1f} GB)", flush=True)

    with ThreadPoolExecutor(max_workers=args.blob_workers) as pool:
        pending_blob: dict = {}
        while True:
            # keep the blob pool fed
            if len(pending_blob) < args.blob_workers:
                for tar in find_ready("done", exclude="blob"):
                    if tar not in [t for t in pending_blob]:
                        pending_blob[pool.submit(blob_worker, tar)] = tar
            # collect blob results
            for fut in list(pending_blob):
                if not fut.done():
                    continue
                tar = pending_blob.pop(fut)
                status, rel = fut.result()
                last_activity = time.monotonic()
                if status == "blob-ok":
                    print(f"[up] blob {rel}", flush=True)
                elif status == "requeue":
                    pass  # cleared for re-extraction; will reappear as a new .done
                else:
                    blob_fails.append(rel)
            # serial committer
            for tar in find_ready("blob", exclude="commit"):
                status, rel = commit_one(tar)
                if status == "commit-ok":
                    committed += 1
                    committed_bytes += tar.stat().st_size
                    last_activity = time.monotonic()
                    print(f"[up] committed {committed} {rel} ({committed_bytes / 1e9:.1f} GB total)", flush=True)
                else:
                    commit_fails.append(rel)
            # monitor daemon: exit only after the queue has been empty AND no
            # activity for idle_limit (i.e. the extract stage is finished)
            if time.monotonic() - last_activity > args.idle_limit:
                print(f"[up] DONE (idle) committed={committed} bytes={committed_bytes / 1e9:.1f} GB", flush=True)
                break
            time.sleep(5)
    if blob_fails or commit_fails:
        raise SystemExit(f"failures: blob={blob_fails[:10]} commit={commit_fails[:10]}")


if __name__ == "__main__":
    main()
