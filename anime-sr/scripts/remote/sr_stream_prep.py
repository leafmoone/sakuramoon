"""SR streaming prep (Stage A consumer) for the SR data service.

Pulls SR_v2 shards from the running SR data service (config/sr_data_service.toml,
512 GiB streaming cache window), extracts them into the persistent webp tree,
and ACKs so the service can LRU-evict processed tars. The corpus is consumed
as a stream — it is never held whole in the service cache.

Pipeline per shard (lease-driven, deterministic service order):
  lease(worker) -> hard-link the cached tar into the flat shard dir under the
  frozen release-prefixed name (shard-<release>-<NNNNNN>.tar — cross-release
  basenames collide, so the salt5 387-tar snapshot naming is the convention)
  -> extract webp (byte-size resume-safe) -> record state -> ack.

Index rebuilds run every --rebuild-every newly-extracted shards over ALL
extracted shards on disk (build_index is full-overwrite), with a persistent
per-shard webp size cache so each rebuild only header-scans the NEW shard
dirs. Cross-shard duplicate ids resolve in global sorted shard order
(later shard wins), identical to the serial collect_webp_sizes semantics.

Idempotent and restart-safe: extracted webp are skipped by byte check, the
service queue state re-leases an in-flight shard to the same worker id, and
linked tars survive service eviction (hard link).

Usage (salt8):
  python3 sr_stream_prep.py \
      --socket /run/sakuramoon/sr-data-service.sock \
      --service-workers 16 \
      --workers 8 \
      --shard-dir /root/private_data/anime-sr/srv2-run/shards \
      --webp-dir /root/private_data/anime-sr/srv2-run/data/webp \
      --index-dir /root/private_data/anime-sr/srv2-run/data/index \
      --code-src /root/private_data/anime-sr/sakuramoon-dev/anime-sr/src \
      --rebuild-every 250 [--rebuild-interval 1800] [--once]
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import signal
import sys
import tarfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))  # sakuramoon (service client)

from sakuramoon.data.client import (
    DataServiceClient,
    DataServiceUnavailable,
)

STATE_NAME = "sr-stream-state.json"
SIZES_NAME = "sr-stream-sizes.pkl"
_SHARD_BASE_RE = re.compile(r"^shard-(\d+)-p2-\d+\.tar$")

_stop = threading.Event()


def _log(worker: str, msg: str) -> None:
    print(f"[prep:{worker}] {msg}", flush=True)


def target_name(local_path: Path) -> str:
    """Map a cached shard to its frozen flat name.

    local_path = <cache root>/data/<release>/shard-NNNNNN-p2-00.tar
    -> shard-<release>-NNNNNN.tar (salt5 snapshot convention; cross-release
    shard-000000 exists in every release, so the basename alone is unsafe).
    """
    base = local_path.name
    release = local_path.parent.name
    m = _SHARD_BASE_RE.match(base)
    num = m.group(1) if m else base[: -len(".tar")]
    return f"shard-{release}-{num}.tar"


def extract_one(tar_path: Path, webp_dir: Path) -> int:
    """Extract webp members of one shard (byte-size resume-safe)."""
    dest = webp_dir / tar_path.name
    dest.mkdir(parents=True, exist_ok=True)
    n_out = 0
    with tarfile.open(tar_path, "r") as tf:
        for member in tf.getmembers():
            if not (member.isfile() and member.name.endswith(".webp")):
                continue
            img_id = member.name.rsplit("/", 1)[-1]
            target = dest / img_id
            if target.is_file() and target.stat().st_size == member.size:
                continue
            f = tf.extractfile(member)
            if f is None:
                raise RuntimeError(f"cannot read {member.name} from {tar_path}")
            tmp = target.with_suffix(".part")
            tmp.write_bytes(f.read())
            tmp.rename(target)
            n_out += 1
    return n_out


class State:
    """Extracted/acked shard bookkeeping (restart-safe JSON)."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.extracted: dict[str, dict] = {}
        self._lock = threading.Lock()
        if path.is_file():
            doc = json.loads(path.read_text(encoding="utf-8"))
            self.extracted = doc.get("extracted", {})

    def mark_extracted(self, name: str, n_webp: int, total_bytes: int) -> None:
        with self._lock:
            self.extracted[name] = {"n_webp": n_webp, "bytes": total_bytes}
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(
                json.dumps(
                    {"extracted": self.extracted},
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            tmp.replace(self.path)

    def names(self) -> list[str]:
        with self._lock:
            return sorted(self.extracted)


def scan_shard_sizes(
    webp_dir: Path, shard_name: str
) -> dict[str, tuple[int, int]]:
    from anime_sr.data.index import webp_header_size

    dest = webp_dir / shard_name
    out: dict[str, tuple[int, int]] = {}
    if not dest.is_dir():
        return out
    for f in sorted(os.listdir(dest)):
        if not f.endswith(".webp") or f.endswith(".part"):
            continue
        p = dest / f
        with open(p, "rb") as fh:
            s = webp_header_size(fh.read(64))
        if s is not None:
            out[f[: -len(".webp")]] = s
    return out


def rebuild_index(
    args: argparse.Namespace,
    state: State,
    sizes_cache: dict[str, dict[str, tuple[int, int]]],
    sizes_path: Path,
) -> dict:
    """Full-overwrite rebuild over all extracted shards; incremental size scan."""
    sys.path.insert(0, args.code_src)
    from anime_sr.config.loader import load_config
    from anime_sr.data import index

    cfg = load_config(
        str(REPO_ROOT / "anime-sr/config/base.toml"),
        str(REPO_ROOT / "anime-sr/config/data.toml"),
    )
    names = state.names()
    shards = [args.shard_dir / n for n in names if (args.shard_dir / n).is_file()]
    if not shards:
        _log("index", "no extracted shards to index")
        return {}

    # Header-scan only the shard dirs missing from the persistent cache.
    to_scan = [n for n in names if n not in sizes_cache]
    if to_scan:
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=args.size_scan_workers) as pool:
            results = dict(zip(to_scan, pool.map(lambda n: scan_shard_sizes(args.webp_dir, n), to_scan)))
        sizes_cache.update(results)
        sizes_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = sizes_path.with_suffix(".tmp")
        with open(tmp, "wb") as fh:
            pickle.dump(sizes_cache, fh)
        tmp.replace(sizes_path)
        _log(
            "index",
            f"scanned {len(to_scan)} new shard dirs in {time.time() - t0:.0f}s "
            f"({sum(len(v) for v in results.values())} sizes)",
        )

    # Merge in GLOBAL sorted shard order so cross-shard duplicate ids resolve
    # identically to the serial collect_webp_sizes (later shard wins).
    merged: dict[str, tuple[int, int]] = {}
    for n in names:
        merged.update(sizes_cache.get(n, {}))

    t0 = time.time()
    report = index.build_index(shards, cfg, args.index_dir, size_overrides=merged)
    _log(
        "index",
        f"rebuilt over {len(shards)} shards in {time.time() - t0:.0f}s: "
        f"{report.get('n_images')} images, eligible={report.get('n_eligible_train')}, "
        f"size_missing={report.get('n_size_missing')}",
    )
    return report


def run_worker(
    client: DataServiceClient,
    worker_id: int,
    args: argparse.Namespace,
    state: State,
    stats: dict,
    stats_lock: threading.Lock,
) -> None:
    name = f"w{worker_id:02d}"
    failures = 0
    while not _stop.is_set():
        try:
            desc = client.lease(worker_id)
        except DataServiceUnavailable as error:
            failures += 1
            if failures % 6 == 1:
                _log(name, f"service unavailable ({error}); retrying")
            time.sleep(10.0)
            continue
        if desc is None:
            _log(name, "service reports corpus done")
            return
        failures = 0
        target = args.shard_dir / target_name(desc.local_path)
        if not target.exists():
            args.shard_dir.mkdir(parents=True, exist_ok=True)
            os.link(desc.local_path, target)
        t0 = time.time()
        n_webp = extract_one(target, args.webp_dir)
        total_bytes = target.stat().st_size
        state.mark_extracted(target.name, n_webp, total_bytes)
        client.acknowledge(desc)
        with stats_lock:
            stats["shards"] = stats.get("shards", 0) + 1
            stats["webp"] = stats.get("webp", 0) + n_webp
        _log(
            name,
            f"{target.name}: {n_webp} webp in {time.time() - t0:.0f}s "
            f"(total {stats['shards']} shards, {stats['webp']} webp)",
        )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--socket", required=True, help="SR data-service unix socket")
    ap.add_argument(
        "--service-workers",
        type=int,
        default=16,
        help="session worker_count the service was started with",
    )
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--shard-dir", required=True)
    ap.add_argument("--webp-dir", required=True)
    ap.add_argument("--index-dir", required=True)
    ap.add_argument("--code-src", required=True, help="anime-sr src/ dir")
    ap.add_argument("--rebuild-every", type=int, default=250)
    ap.add_argument("--rebuild-interval", type=float, default=1800.0)
    ap.add_argument("--size-scan-workers", type=int, default=16)
    ap.add_argument(
        "--once",
        action="store_true",
        help="single index rebuild over current disk state, no service session",
    )
    args = ap.parse_args()
    args.shard_dir = Path(args.shard_dir)
    args.webp_dir = Path(args.webp_dir)
    args.index_dir = Path(args.index_dir)

    shard_dir = Path(args.shard_dir)
    state = State(shard_dir / STATE_NAME)
    sizes_path = shard_dir / SIZES_NAME
    sizes_cache: dict[str, dict[str, tuple[int, int]]] = {}
    if sizes_path.is_file():
        with open(sizes_path, "rb") as fh:
            sizes_cache = pickle.load(fh)

    if args.once:
        rebuild_index(args, state, sizes_cache, sizes_path)
        return 0

    signal.signal(signal.SIGTERM, lambda *_a: _stop.set())

    # The service blocks client connections until its startup barrier (the
    # first worker_count shards ready), which during a degraded download
    # window (proxy flap) can take a long time: wait it out, not crash.
    client: DataServiceClient | None = None
    while client is None:
        try:
            client = DataServiceClient(
                Path(args.socket),
                worker_count=args.service_workers,
                request_timeout_seconds=60.0,
            )
            break
        except DataServiceUnavailable as error:
            if _stop.is_set():
                return 1
            _log("main", f"service not ready yet ({error}); waiting")
            time.sleep(15.0)
    _log("main", f"connected to {args.socket}; corpus session ready")

    stats: dict[str, int] = {}
    stats_lock = threading.Lock()
    indexed_count = len(state.names())
    threads = [
        threading.Thread(
            target=run_worker,
            args=(client, i, args, state, stats, stats_lock),
            name=f"prep-w{i:02d}",
            daemon=True,
        )
        for i in range(args.workers)
    ]
    for t in threads:
        t.start()

    last_rebuild = time.time()
    exit_code = 0
    while True:
        time.sleep(5.0)
        if _stop.is_set() and all(not t.is_alive() for t in threads):
            break
        names = state.names()
        newly = len(names) - indexed_count
        if (
            not _stop.is_set()
            and names
            and (newly >= args.rebuild_every or (time.time() - last_rebuild) >= args.rebuild_interval)
        ):
            try:
                rebuild_index(args, state, sizes_cache, sizes_path)
                indexed_count = len(state.names())
                last_rebuild = time.time()
            except Exception:  # noqa: BLE001 - progress must survive index hiccups
                _log("index", f"rebuild failed:\n{traceback_text()}")

    if all(not t.is_alive() for t in threads) and state.names():
        # Corpus stream finished (or we stopped): final full rebuild.
        try:
            rebuild_index(args, state, sizes_cache, sizes_path)
        except Exception:  # noqa: BLE001
            _log("index", f"final rebuild failed:\n{traceback_text()}")
            exit_code = 1

    with stats_lock:
        _log("main", f"stopped: {stats.get('shards', 0)} shards streamed, "
             f"{stats.get('webp', 0)} webp extracted this run")
    return exit_code


def traceback_text() -> str:
    import traceback

    return traceback.format_exc()


if __name__ == "__main__":
    raise SystemExit(main())
