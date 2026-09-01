"""SR streaming prep (Stage A consumer) for the SR data service.

Three modes (--mode), all lease-driven against the 512 GiB-bounded
streaming cache:

  extract  (legacy / fallback, default)
      Pulls shards, extracts them into the persistent webp tree, ACKs so
      the service can LRU-evict processed tars. Index rebuilds every
      --rebuild-every newly-extracted shards over ALL extracted shards.

  index  (streaming tar-direct path, 2026-09-01)
      ONE PASS over the whole corpus: lease each shard (queue order),
      single-pass scan with member coordinates (scan_shard_full: meta +
      webp offset/size + REAL header dims), write a per-shard index
      partition, ACK. When the queue drains, the CPU-only merge
      (build_index_from_records) assembles the global eligibility table —
      the frozen sample set (n) the trainer's §11.5 stream is built over.
      No webp is ever extracted; the corpus is never resident.

  window (streaming tar-direct training-time driver, 2026-09-01)
      Tracks the trainer's deterministic slot stream (the step heartbeat
      the trainer writes every 50 steps) and keeps the shards the NEXT
      --window-steps of the stream touch pinned: request(worker, path)
      downloads + pins on demand, hard-link into the pin dir (the trainer
      decodes webp members in place from there), ACK + unlink when a shard
      leaves the window (the 512 GiB LRU may then evict it). Validation
      shards stay pinned for the whole run (a small fixed set).

Usage (salt8):
  python3 sr_stream_prep.py --socket /run/sakuramoon/sr-data-service.sock \
      --mode index --workers 8 \
      --index-dir /root/private_data/anime-sr/srv2-run/data/index \
      --code-src /root/private_data/anime-sr/sakuramoon-dev/anime-sr/src

  python3 sr_stream_prep.py --socket /run/sakuramoon/sr-data-service.sock \
      --mode window --index-dir .../data/index \
      --shard-dir /root/private_data/anime-sr/srv2-run/tars \
      --state-file /root/private_data/anime-sr/srv2-run/output/.../step-heartbeat.json \
      --bucket-hr 1024 --bs 8 --world 2 --window-steps 300 --lease-slots 4
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import pickle
import queue
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
_FLAT_NAME_RE = re.compile(r"^shard-(.+)-(\d+)\.tar$")

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


def flat_to_relpath(flat: str) -> str:
    """Inverse of target_name: flat pin name -> repo-relative tar path."""
    m = _FLAT_NAME_RE.match(flat)
    if m is None:
        raise ValueError(f"flat shard name is not in the frozen form: {flat}")
    release, num = m.group(1), m.group(2)
    return f"data/{release}/shard-{num}-p2-00.tar"


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
                continue
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
        p = webp_dir / shard_name / f
        with open(p, "rb") as fh:
            s = webp_header_size(fh.read(64))
        if s is not None:
            out[f[: -len(".webp")]] = s
    return out


def _load_anime_sr_cfg(args: argparse.Namespace):
    sys.path.insert(0, args.code_src)
    from anime_sr.config.loader import load_config

    # The window driver must see EXACTLY the trainer's config stack (the
    # slot map, bucket table, gates and val split are pure functions of it);
    # venue overlays (e.g. sampling.enabled for streaming) only work when
    # both processes pass the same --config files.
    if args.config:
        return load_config(*[str(Path(c)) for c in args.config])
    return load_config(
        str(REPO_ROOT / "anime-sr/config/base.toml"),
        str(REPO_ROOT / "anime-sr/config/data.toml"),
    )


def rebuild_index(
    args: argparse.Namespace,
    state: State,
    sizes_cache: dict[str, dict[str, tuple[int, int]]],
    sizes_path: Path,
) -> dict:
    """Full-overwrite rebuild over all extracted shards; incremental size scan."""
    cfg = _load_anime_sr_cfg(args)
    from anime_sr.data import index

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


# ---------------------------------------------------------------------------
# mode: index — the one-shot streaming index pass (no extraction)
# ---------------------------------------------------------------------------
def run_index_mode(args: argparse.Namespace, client: DataServiceClient) -> int:
    """Lease every shard (queue order), scan_shard_full it into a per-shard
    partition, ACK; merge all partitions into the global index at the end."""
    cfg = _load_anime_sr_cfg(args)
    from anime_sr.data.index import (
        MetaRecord,
        ShardSummary,
        build_index_from_records,
        scan_shard_full,
    )

    parts_dir = Path(args.index_dir) / "partitions"
    parts_dir.mkdir(parents=True, exist_ok=True)

    def ack_with_retry(name: str, desc: object) -> None:
        """ACK must land before the worker leases the next shard.

        The service commits state (NFS write) under its lock, so the ack
        round-trip can exceed the client socket timeout; the client's
        transport retry then delivers a duplicate, which the service
        (protocol v5) accepts as a no-op. Retry until it lands; the
        partition is already durable, so this loop can never lose work."""
        attempt = 0
        while not _stop.is_set():
            try:
                client.acknowledge(desc)
                return
            except DataServiceUnavailable as error:
                attempt += 1
                _log(
                    name,
                    f"ack not confirmed for {desc.local_path.name} "
                    f"(attempt {attempt}: {error}); retrying",
                )
                time.sleep(5.0)

    def worker(worker_id: int) -> None:
        name = f"idx{worker_id:02d}"
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
            flat = target_name(desc.local_path)
            part = parts_dir / f"{flat}.jsonl"
            done = parts_dir / f"{flat}.jsonl.done"
            if done.is_file():
                ack_with_retry(name, desc)
                continue
            t0 = time.time()
            records, summary = scan_shard_full(desc.local_path, flat)
            tmp = part.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as fh:
                fh.writelines(
                    json.dumps(dataclasses.asdict(rec)) + "\n" for rec in records
                )
            tmp.replace(part)
            (parts_dir / f"{flat}.summary.json").write_text(
                json.dumps(dataclasses.asdict(summary)), encoding="utf-8"
            )
            done.touch()
            ack_with_retry(name, desc)
            _log(
                name,
                f"{flat}: {summary.n_images} images in {time.time() - t0:.0f}s "
                f"(webp_missing={summary.n_webp_missing} "
                f"bad_header={summary.n_webp_bad_header})",
            )

    threads = [
        threading.Thread(target=worker, args=(i,), name=f"prep-idx-{i}", daemon=True)
        for i in range(args.workers)
    ]
    for t in threads:
        t.start()
    while any(t.is_alive() for t in threads) and not _stop.is_set():
        time.sleep(5)
    for t in threads:
        t.join(timeout=1.0)

    # merge: all partitions -> records + summaries -> global artifacts
    all_records: list[MetaRecord] = []
    summaries: list[ShardSummary] = []
    for part in sorted(parts_dir.glob("*.jsonl")):
        with part.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    all_records.append(MetaRecord(**json.loads(line)))
        summ = parts_dir / f"{part.stem}.summary.json"
        if summ.is_file():
            summaries.append(ShardSummary(**json.loads(summ.read_text(encoding="utf-8"))))
    n_parts = len(list(parts_dir.glob("*.jsonl.done")))
    _log("index", f"merging {n_parts} partitions ({len(all_records)} records) into the global index")
    t0 = time.time()
    report = build_index_from_records(all_records, summaries, cfg, args.index_dir)
    n_missing = int(report.get("n_webp_missing") or 0)
    n_bad = int(report.get("n_webp_bad_header") or 0)
    _log(
        "index",
        f"merged in {time.time() - t0:.0f}s: {report.get('n_images')} images, "
        f"eligible={report.get('n_eligible_train')}, webp_missing={n_missing}, "
        f"bad_header={n_bad}",
    )
    if n_missing or n_bad:
        _log("index", "COVERAGE FAILURE: a meta has no webp member or its header is unreadable — aborting (no silent sample skipping)")
        return 1
    (Path(args.index_dir) / "COMPLETE.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _log("index", "streaming index COMPLETE")
    return 0


# ---------------------------------------------------------------------------
# mode: window — the training-time demand driver (pin/lease/ack)
# ---------------------------------------------------------------------------
_CKPT_STEP_CACHE: dict[str, tuple[float, int]] = {}


def _ckpt_step(path: str | Path) -> int:
    """The trainer step stored in a checkpoint (v2 meta["step"] or v1
    "step"), cached by (path, mtime). Returns 0 when the file is missing or
    unreadable — the conservative release gate (nothing may be released)
    on any read failure."""
    key = str(path)
    try:
        mtime = Path(path).stat().st_mtime
    except OSError:
        return 0
    hit = _CKPT_STEP_CACHE.get(key)
    if hit is not None and hit[0] == mtime:
        return hit[1]
    step = 0
    try:
        import torch  # lazy: keep this script importable without DTK

        doc = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(doc, dict):
            meta = doc.get("meta")
            if isinstance(meta, dict) and "step" in meta:
                step = int(meta["step"])
            elif "step" in doc:
                step = int(doc["step"])
    except (OSError, RuntimeError, ValueError, TypeError, KeyError):
        step = 0
    _CKPT_STEP_CACHE[key] = (mtime, step)
    return step


def run_window_mode(args: argparse.Namespace, client: DataServiceClient) -> int:
    """Keep the shards the trainer's next --window-steps touch pinned.

    The demand set is the window oracle (anime_sr.data.stream) over the
    trainer's exact §11.5 slot stream (same formula, same SlotMap salt,
    same clean-score gate as the trainer). Steps come from the trainer's
    step heartbeat (rank 0, every 50 steps); before the first heartbeat the
    driver pins [0, window) (warm start before the trainer begins)."""
    cfg = _load_anime_sr_cfg(args)
    from anime_sr.data.pipeline import SRDataset
    from anime_sr.data.stream import window_shards
    from anime_sr.train.latent_flow import (
        _build_slot_map,
        clean_score_gate_retained,
    )

    shard_dir = Path(args.shard_dir)
    shard_dir.mkdir(parents=True, exist_ok=True)

    ds = SRDataset(
        args.index_dir, shard_dir, cfg,
        bucket_hr=args.bucket_hr, split="train", tar_dir=shard_dir,
    )
    # Mirror the trainer exactly (latent_flow.run_latent_flow): the
    # clean-score gate filters ds.samples BEFORE the stream is built, and
    # on-fly mode's legacy order is the identity — the window driver must
    # reproduce the trainer's §11.5 slot stream sample-for-sample.
    retained = clean_score_gate_retained(
        [m.sample_id for m in ds.samples], args.index_dir, cfg.filter.clean_score_min
    )
    if retained is not None:
        ds.samples = [m for m in ds.samples if m.sample_id in retained]
        if not ds.samples:
            _log("window", "clean-score gate removed all train samples")
            return 1
    n = len(ds.samples)
    slot_map = _build_slot_map(ds, cfg, list(range(n)))
    shards = [m.shard for m in ds.samples]

    val_shards: set[str] = set()
    try:
        val_ds = SRDataset(
            args.index_dir, shard_dir, cfg,
            bucket_hr=args.bucket_hr, split="validation", tar_dir=shard_dir,
        )
        val_shards = {m.shard for m in val_ds.samples}
    except (RuntimeError, ValueError):
        pass

    _log(
        "window",
        f"stream built: n={n}, bs={args.bs}, world={args.world}, "
        f"window={args.window_steps} steps, val_shards={len(val_shards)}, "
        f"sampling={'on' if slot_map.enabled else 'off (legacy)'}",
    )

    q: queue.Queue[tuple[str, int]] = queue.Queue()
    wlock = threading.Lock()
    pinned: dict[str, tuple[object, int]] = {}
    pending: dict[str, int] = {}
    # slots currently in flight on a worker (from queue pickup until the pin
    # succeeds); a slot must not be re-assigned while its job is sleeping or
    # re-queued, or the service would see two concurrent leases for one
    # worker id ("already holds a lease" retry storm)
    inflight: set[int] = set()
    # epoch-boundary retry counts per slot (stall diagnostics)
    stall_notes: dict[int, int] = {}

    def _recover_stale_lease(slot: int, rel: str, flat: str) -> bool:
        """request() failed because worker `slot` already holds a lease —
        typically a STALE lease left by a previous driver run (a crash or
        restart never ACKs it; the new driver has no memory of it).
        Recover through the protocol: lease(slot) idempotently returns the
        held descriptor. Same path -> adopt it (no state change); different
        path -> ACK it to unblock the worker (its row becomes completed in
        the current cycle; if this run's stream still needs that shard
        later in the cycle, the epoch-boundary branch below will wait — a
        wait that can only end with a service restart, which the stall
        log calls out explicitly). Returns True when the worker was
        unblocked (or the job was satisfied)."""
        try:
            held = client.lease(slot)
        except DataServiceUnavailable as error:
            _log("pin", f"stale-lease recovery for slot {slot} failed: {error}")
            return False
        if held is None:
            return False  # a race cleared it (or the queue drained); retry
        if held.record.path == rel:
            link = shard_dir / flat
            if not link.exists():
                try:
                    os.link(held.local_path, link)
                except OSError:
                    pass
            with wlock:
                pending.pop(flat, None)
                inflight.discard(slot)
                pinned[flat] = (held, slot)
            _log("pin", f"adopted held lease for {flat} (stale-lease recovery)")
            return True
        try:
            client.acknowledge(held)
        except DataServiceUnavailable as error:
            _log("pin", f"stale-lease release failed for {held.record.path}: {error}")
            return False
        _log(
            "pin",
            f"released stale lease {held.record.path} from slot {slot}; "
            f"retrying {flat}",
        )
        return True

    def lease_worker() -> None:
        name = "pin"
        while not _stop.is_set():
            try:
                flat, slot = q.get(timeout=1.0)
            except queue.Empty:
                continue
            with wlock:
                inflight.add(slot)
            rel = flat_to_relpath(flat)
            try:
                desc = client.request(slot, rel, timeout_seconds=args.request_timeout)
                link = shard_dir / flat
                if not link.exists():
                    os.link(desc.local_path, link)
                with wlock:
                    pending.pop(flat, None)
                    inflight.discard(slot)
                    pinned[flat] = (desc, slot)
                _log(name, f"pinned {flat} (slot {slot})")
            except DataServiceUnavailable as error:
                msg = str(error)
                if "already holds a lease" in msg:
                    if not _recover_stale_lease(slot, rel, flat):
                        with wlock:
                            pending.pop(flat, None)
                        time.sleep(10.0)
                        q.put((flat, slot))
                    continue
                with wlock:
                    pending.pop(flat, None)
                _log(name, f"service unavailable while pinning {flat}: {msg}")
                time.sleep(10.0)
                q.put((flat, slot))
            except Exception as error:  # noqa: BLE001 - surface, retry, never die
                with wlock:
                    pending.pop(flat, None)
                msg = str(error)
                if "completed in this cycle" in msg:
                    # epoch boundary: the queue rolled over; retry after the
                    # service has rebuilt the next cycle. A DEMAND-DRIVEN
                    # cycle only rolls when EVERY row is completed, so a
                    # stalled boundary means rows are stuck (e.g. a stale
                    # lease just got ACKed for a shard this cycle still
                    # needs) — the only recovery is a service restart,
                    # which resets the queue to a fresh cycle.
                    stall_notes[slot] = stall_notes.get(slot, 0) + 1
                    note = ""
                    if stall_notes[slot] >= 20:
                        note = (
                            " — STALL: the cycle cannot roll without a "
                            "queue-order consumer; restart the data service "
                            "to reset the queue"
                        )
                    if stall_notes[slot] in (1, 20) or stall_notes[slot] % 40 == 0:
                        _log(
                            name,
                            f"epoch boundary at {flat} (x{stall_notes[slot]}); "
                            f"retrying in 15s{note}",
                        )
                    time.sleep(15.0)
                    q.put((flat, slot))
                else:
                    _log(name, f"pin failed for {flat}: {type(error).__name__} {msg[:200]}")
                    time.sleep(15.0)
                    q.put((flat, slot))

    threads = [
        threading.Thread(target=lease_worker, name=f"prep-pin-{i}", daemon=True)
        for i in range(args.lease_slots)
    ]
    for t in threads:
        t.start()

    state_file = Path(args.state_file)
    last_step: int | None = None
    while not _stop.is_set():
        step = 0
        if state_file.is_file():
            try:
                doc = json.loads(state_file.read_text(encoding="utf-8"))
                step = max(0, int(doc["step"]))
            except (OSError, ValueError, KeyError):
                step = 0
        if step != last_step:
            window = set(
                window_shards(
                    step,
                    step + args.window_steps,
                    bs=args.bs,
                    world=args.world,
                    n=n,
                    slot_map=slot_map,
                    shards=shards,
                )
            )
            last_step = step
            # Release gate (2026-09-01, salt9 death-loop fix): a pinned shard
            # is the ONLY in-cycle copy — the data service refuses same-cycle
            # re-serves ("completed in this cycle"). Before the first
            # checkpoint a death restarts the trainer at step 0 (no ckpt to
            # seed the heartbeat), so the window rolls back and re-requests
            # exactly the shards the dead run already released; with the pins
            # unlinked the service hard-refuses -> missing pin -> trainer
            # death -> restart -> the same refusal, forever. A shard may
            # therefore be released only when the trainer can never re-read
            # it: the resume origin O (latest ckpt step; 0 = no ckpt yet)
            # has passed it. The re-read region is [O, current sample-cycle
            # boundary) in step units: O sits at most one checkpoint grid
            # (15625 steps) + one window behind `step`, so the region — and
            # the retained pin set — stays bounded (~one grid + window,
            # ~260 GiB on the venue's local disk) and drains as O advances.
            # Pre-checkpoint O=0 makes the region the whole current cycle:
            # nothing is released until the first ckpt exists (bounded by
            # the first-grid step count).
            resume_origin = (
                _ckpt_step(args.resume_ckpt) if args.resume_ckpt else 0
            )
            cycle_pos = step * args.bs * args.world
            cycle_end_pos = (cycle_pos // n + 1) * n
            re_read_end = (cycle_end_pos + args.bs * args.world - 1) // (
                args.bs * args.world
            )  # ceil: keep the boundary-straddling step conservative
            re_read = (
                set(
                    window_shards(
                        resume_origin,
                        re_read_end,
                        bs=args.bs,
                        world=args.world,
                        n=n,
                        slot_map=slot_map,
                        shards=shards,
                    )
                )
                if 0 < resume_origin < re_read_end
                else (set(shards) if 0 < n else set())
            )
            # pin: everything the window (or val) needs
            for flat in sorted(window | val_shards):
                with wlock:
                    if flat in pinned or flat in pending:
                        continue
                    # restart recovery: a pin file left on disk by a
                    # previous driver run IS the local in-cycle copy —
                    # adopt it without a service round-trip. This also
                    # covers shards the service's cycle ledger already
                    # marks completed (self-ACKed via the stale-lease
                    # recovery path): a fresh request for them would be
                    # refused ("completed in this cycle"), but the file
                    # on disk is exactly what the trainer needs
                    # (2026-09-01 salt9 pre-ckpt restart deadlock, 2nd
                    # form).
                    if (shard_dir / flat).exists():
                        pinned[flat] = (None, None)
                        _log(
                            "window",
                            f"adopted on-disk pin for {flat} (restart recovery)",
                        )
                        continue
                    # a free slot: not carrying a pending pin AND not in
                    # flight on a worker (sleeping / re-queued job)
                    busy = set(pending.values()) | inflight
                    free = [s for s in range(args.lease_slots) if s not in busy]
                    if not free:
                        continue
                    slot = free[0]
                    pending[flat] = slot
                    q.put((flat, slot))
            # release: shards that left the window (val stays pinned);
            # ACK outside the lock (network), the LRU may then evict the tar
            released: list[tuple[str, object, int]] = []
            with wlock:
                for flat in list(pinned):
                    # val stays pinned; the live window stays pinned; the
                    # restart re-read region (release gate above) stays
                    # pinned — releasing it is the salt9 death loop.
                    if flat in val_shards or flat in window or flat in re_read:
                        continue
                    desc, slot = pinned.pop(flat)
                    released.append((flat, desc, slot))
            for flat, desc, slot in released:
                try:
                    os.unlink(shard_dir / flat)
                except OSError as error:
                    _log("window", f"unlink {flat} failed: {error}")
                if desc is None:
                    # adopted from disk (no service lease was ever held
                    # by this driver run): unlink alone releases it
                    _log("window", f"released {flat} (adopted pin; no ack)")
                    continue
                try:
                    client.acknowledge(desc)
                    _log("window", f"released {flat}")
                except DataServiceUnavailable as error:
                    _log("window", f"ack {flat} failed ({error}); keeping pinned")
                    with wlock:
                        pinned[flat] = (desc, slot)
            _log(
                "window",
                f"step={step}: window touches {len(window)} shards, "
                f"pinned={len(pinned)}",
            )
        time.sleep(5)
    for t in threads:
        t.join(timeout=1.0)
    return 0


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
        "--mode",
        choices=("extract", "index", "window"),
        default="extract",
        help="extract=legacy webp tree (default); index=one-shot streaming "
        "index pass (no extraction); window=training-time pin driver",
    )
    ap.add_argument("--service-workers", type=int, default=16)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--shard-dir", required=True)
    ap.add_argument("--webp-dir", default=None, help="extract mode only")
    ap.add_argument("--index-dir", required=True)
    ap.add_argument("--code-src", required=True)
    ap.add_argument(
        "--config",
        nargs="+",
        default=None,
        help="anime-sr config stack (the SAME files the trainer passes via "
        "--config); window mode must see the trainer's exact overlays. "
        "Default: base.toml + data.toml",
    )
    ap.add_argument("--rebuild-every", type=int, default=250)
    ap.add_argument("--rebuild-interval", type=int, default=1800)
    ap.add_argument("--size-scan-workers", type=int, default=8)
    ap.add_argument("--once", action="store_true")
    # window mode
    ap.add_argument("--state-file", default=None, help="window mode: trainer step heartbeat")
    ap.add_argument("--bucket-hr", type=int, default=1024)
    ap.add_argument("--bs", type=int, default=8, help="window mode: train batch size per rank")
    ap.add_argument("--world", type=int, default=2, help="window mode: DDP world size")
    ap.add_argument("--window-steps", type=int, default=300)
    ap.add_argument("--request-timeout", type=float, default=1800.0)
    ap.add_argument("--lease-slots", type=int, default=4)
    ap.add_argument(
        "--resume-ckpt",
        default=None,
        help=(
            "window mode: the latest.pt the trainer resumes from. Release "
            "gate: a pinned shard is the ONLY in-cycle copy (the service "
            "refuses same-cycle re-serves), so a shard is released only "
            "when the trainer can never re-read it — i.e. the resume "
            "origin (ckpt step; 0 = no checkpoint yet) has passed it. "
            "Pre-checkpoint: nothing is ever released."
        ),
    )
    args = ap.parse_args()

    if args.mode == "extract" and not args.webp_dir:
        raise SystemExit("--webp-dir is required for extract mode")
    if args.mode == "window" and not args.state_file:
        raise SystemExit("--state-file is required for window mode")

    signal.signal(signal.SIGTERM, lambda *_: _stop.set())
    signal.signal(signal.SIGINT, lambda *_: _stop.set())

    client: DataServiceClient | None = None
    while client is None:
        try:
            client = DataServiceClient(
                Path(args.socket),
                worker_count=args.service_workers,
                request_timeout_seconds=30.0,
            )
        except DataServiceUnavailable:
            if _stop.is_set():
                raise SystemExit(1)
            _log("main", "service not ready yet (data service is unavailable); waiting")
            time.sleep(15.0)

    exit_code = 0
    if args.mode == "index":
        exit_code = run_index_mode(args, client)
        return exit_code
    if args.mode == "window":
        exit_code = run_window_mode(args, client)
        return exit_code

    # extract mode (legacy)
    state = State(Path(args.shard_dir) / STATE_NAME)
    sizes_path = Path(args.shard_dir) / SIZES_NAME
    sizes_cache: dict[str, dict[str, tuple[int, int]]] = {}
    if sizes_path.is_file():
        with open(sizes_path, "rb") as fh:
            sizes_cache = pickle.load(fh)
    stats: dict = {}
    stats_lock = threading.Lock()
    last_rebuild = time.time()
    n_at_rebuild = len(state.names())

    threads = [
        threading.Thread(
            target=run_worker,
            args=(client, i, args, state, stats, stats_lock),
            name=f"prep-{i}",
            daemon=True,
        )
        for i in range(args.workers)
    ]
    for t in threads:
        t.start()
    while any(t.is_alive() for t in threads) and not _stop.is_set():
        time.sleep(5)
        if (
            not args.once
            and time.time() - last_rebuild >= args.rebuild_interval
            and len(state.names()) - n_at_rebuild >= args.rebuild_every
        ):
            last_rebuild = time.time()
            n_at_rebuild = len(state.names())
            try:
                rebuild_index(args, state, sizes_cache, sizes_path)
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
