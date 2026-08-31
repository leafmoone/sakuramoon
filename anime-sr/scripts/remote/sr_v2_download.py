"""SR_v2 build stage 1: parallel resumable downloader for P2 shards (salt5).

Downloads only the shards listed in the P2 manifest (full-file, per-file size
verification, .part resume). Nothing else is touched.

Run:
  python3.11 srv2_download.py \
    --manifest /root/private_data/anime-sr/sr-v2-build/p2-manifest.json \
    --shard-list /root/private_data/anime-sr/reports-m4-sr-clean-v1-fullrepo/shard-list.tsv \
    --raw-dir /root/private_data/anime-sr/sr-v2-build/raw \
    --workers 16
"""

from __future__ import annotations

import argparse
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote

import requests

_tls = threading.local()
_stats_lock = threading.Lock()
_bytes_run = 0.0
_files_done = 0
_files_total = 0
_bytes_total = 0.0
_failures: list[str] = []
_t0 = 0.0
_last_tick = 0.0
_last_bytes = 0.0


def _set_globals(files_total: int, bytes_total: float) -> None:
    global _files_total, _bytes_total, _t0, _last_tick, _last_bytes
    _files_total = files_total
    _bytes_total = bytes_total
    _t0 = time.monotonic()
    _last_tick = _t0
    _last_bytes = 0.0


def _sess(tok: str) -> requests.Session:
    s = getattr(_tls, "s", None)
    if s is None:
        s = requests.Session()
        s.headers.update({"Authorization": f"Bearer {tok}", "Cookie": f"m_session_id={tok}"})
        _tls.s = s
    return s


def _resolve(sess: requests.Session, repo: str, rev: str, path: str) -> str:
    r = sess.get(
        f"https://modelscope.cn/api/v1/datasets/{repo}/repo?Revision={rev}&FilePath={quote(path)}",
        headers={"Range": "bytes=0-0"},
        timeout=60,
        allow_redirects=True,
    )
    r.raise_for_status()
    return r.url


def _download_one(item: tuple[str, int], tok: str, repo: str, rev: str, raw_dir: Path) -> tuple[str, str, int]:
    path, size = item
    global _bytes_run, _files_done
    final = raw_dir / path
    part = final.with_name(final.name + ".part")
    final.parent.mkdir(parents=True, exist_ok=True)
    if final.exists() and final.stat().st_size == size:
        return ("skip", path, size)
    for attempt in range(8):
        try:
            sess = _sess(tok)
            cdn = _resolve(sess, repo, rev, path)
            start = part.stat().st_size if part.exists() else 0
            if start >= size:
                start = 0
                part.unlink(missing_ok=True)
            headers = {"Range": f"bytes={start}-{size - 1}"} if start else {}
            r = sess.get(cdn, headers=headers, timeout=180, stream=True)
            if r.status_code == 416:
                r.close()
                start = 0
                part.unlink(missing_ok=True)
                continue
            r.raise_for_status()
            got = start
            mode = "ab" if start else "wb"
            with open(part, mode) as fh:
                for chunk in r.iter_content(16 * 1024 * 1024):
                    if chunk:
                        fh.write(chunk)
                        got += len(chunk)
            if got != size:
                raise RuntimeError(f"size mismatch {got} != {size}")
            final.parent.mkdir(parents=True, exist_ok=True)
            os.replace(part, final)
            with _stats_lock:
                _bytes_run += (got - start)
                _files_done += 1
            print(f"[dl] {path} {size / 1e9:.2f}GB ({'resume' if start else 'fresh'})", flush=True)
            return ("ok", path, size)
        except (requests.RequestException, OSError, RuntimeError) as exc:
            print(f"[dl] retry {attempt + 1}/8 {path}: {type(exc).__name__} {str(exc)[:120]}", flush=True)
            time.sleep(min(2 ** attempt, 30))
    with _stats_lock:
        _failures.append(path)
    return ("fail", path, 0)


def _ticker() -> None:
    global _last_tick, _last_bytes
    while True:
        time.sleep(20)
        with _stats_lock:
            done, br, bt, ft = _files_done, _bytes_run, _bytes_total, _files_total
            now = time.monotonic()
            rate = (br - _last_bytes) / (now - _last_tick) if now > _last_tick else 0.0
            _last_tick, _last_bytes = now, br
        remaining = bt - br if br < bt else 0
        eta = remaining / rate if rate > 1 else 0
        print(
            f"[dl] {done}/{ft} shards, {br / 1e9:.1f}/{bt / 1e9:.0f} GB, "
            f"{rate / 1e6:.0f} MB/s, eta {eta / 3600:.1f}h",
            flush=True,
        )


def main() -> None:
    global _bytes_total
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--shard-list", required=True, help="path<TAB>size per shard")
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--repo", default="leafmoone/webdataset_danbooru_v2")
    ap.add_argument("--rev", default="master")
    ap.add_argument("--token-file", default="/root/private_data/anime-sr/.ms-token")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    import json

    tok = None
    for line in Path(args.token_file).read_text(encoding="utf-8").splitlines():
        if line.startswith("MODELSCOPE_API_TOKEN="):
            tok = line.split("=", 1)[1].strip()
    if not tok:
        raise SystemExit(f"no token in {args.token_file}")

    with open(args.manifest, encoding="utf-8") as fh:
        manifest = json.load(fh)
    shards: dict[str, int] = {}
    with open(args.shard_list, encoding="utf-8") as fh:
        for ln in fh:
            p, s = ln.rstrip("\n").split("\t")
            shards[p] = int(s)
    items = sorted((p, shards[p]) for p in manifest if p in shards)
    _bytes_total = float(sum(s for _, s in items))
    _set_globals(len(items), _bytes_total)
    print(
        f"[dl] start: {len(items)} shards, {_bytes_total / 1e12:.2f} TB, workers={args.workers}",
        flush=True,
    )

    threading.Thread(target=_ticker, daemon=True).start()
    t_start = time.monotonic()
    n_ok = n_skip = n_fail = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for status, path, size in ex.map(lambda it: _download_one(it, tok, args.repo, args.rev, Path(args.raw_dir)), items):
            if status == "ok":
                n_ok += 1
            elif status == "skip":
                n_skip += 1
            else:
                n_fail += 1
    print(
        f"[dl] DONE ok={n_ok} skip={n_skip} fail={n_fail} elapsed={time.monotonic() - t_start:.0f}s failures={_failures}",
        flush=True,
    )
    if n_fail:
        raise SystemExit(f"{n_fail} shards failed: {_failures}")


if __name__ == "__main__":
    main()
