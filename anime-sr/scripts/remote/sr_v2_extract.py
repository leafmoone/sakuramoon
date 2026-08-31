"""SR_v2 build stage 2: extract P2 pairs from downloaded shards, repack <=2GiB tars.

Output layout mirrors the source repo:
  <out-dir>/<dir>/<srcbase>-p2-NN.tar   (+ .sha256 sidecar, + .done marker)
Each output tar contains the original member names (<...>/<id>.webp|.png|.jpg +
<...>/<id>.json), img-then-json order, exactly the source format.

Watches --raw-dir for fully-downloaded shards (size-verified against
--shard-list) that are listed in the manifest and not yet extracted; exits
when every manifest shard is extracted or has failed twice.

Run:
  python3.11 srv2_extract.py \
    --manifest /root/private_data/anime-sr/sr-v2-build/p2-manifest.json \
    --shard-list /root/private_data/anime-sr/reports-m4-sr-clean-v1-fullrepo/shard-list.tsv \
    --raw-dir /root/private_data/anime-sr/sr-v2-build/raw \
    --out-dir /root/private_data/anime-sr/sr-v2-build/out \
    --workers 8
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import posixpath
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

IMG_EXTS = (".webp", ".png", ".jpg")


class _HashedFile:
    """write-through wrapper so the tar bytes are hashed while being written."""

    def __init__(self, fh, hasher: hashlib._Hash) -> None:
        self.fh = fh
        self.h = hasher

    def write(self, b: bytes) -> int:
        n = self.fh.write(b)
        self.h.update(b)
        return n

    def flush(self) -> None:
        self.fh.flush()

    def tell(self) -> int:
        return self.fh.tell()

    def seek(self, pos: int, whence: int = 0) -> int:
        return self.fh.seek(pos, whence)


def _pad512(n: int) -> int:
    return (n + 511) // 512 * 512


def _extract_one(
    shard_rel: str,
    ids: list[str],
    size_expected: int,
    raw_dir: Path,
    out_dir: Path,
    max_bytes: int,
) -> tuple[str, str, int, int]:
    """Return (status, shard_rel, emitted, missing). status: ok|error."""
    p2set = set(ids)
    raw = raw_dir / shard_rel
    base = shard_rel.rsplit("/", 1)[1][:-len(".tar")]
    # flat layout identical to the source repo: <out>/<dir>/<base>-p2-NN.tar
    # (in-progress chunks use a hidden .tmp-NN name + .done marker, so the
    # uploader never sees a partial tar)
    out_sub = out_dir / shard_rel.rsplit("/", 1)[0]
    out_sub.mkdir(parents=True, exist_ok=True)

    pending: dict[str, dict] = {}
    emitted: set[str] = set()
    cur_fh = None
    cur_tar = None
    cur_h = None
    cur_bytes = 0
    chunk_no = 0
    n_chunks = 0

    def _tmp_name() -> Path:
        # UNIQUE per (shard, chunk): concurrent shards in the same output
        # directory must never share a temp path (two workers writing one
        # inode = interleaved corruption)
        return out_sub / f".tmp-{base}-p2-{chunk_no:02d}.tar"

    def close_chunk() -> None:
        nonlocal cur_fh, cur_tar, cur_h, cur_bytes, chunk_no, n_chunks
        if cur_tar is None:
            return
        if cur_bytes > 0:
            cur_tar.close()
            cur_fh.close()
            name = f"{base}-p2-{chunk_no:02d}.tar"
            final = out_sub / name
            _tmp_name().rename(final)
            # read-back verification: the file on disk must equal the byte
            # stream that was hashed (guards against any concurrent-write or
            # filesystem anomaly before the sidecar is ever trusted)
            h2 = hashlib.sha256()
            with open(final, "rb") as fh:
                for blk in iter(lambda: fh.read(32 * 1024 * 1024), b""):
                    h2.update(blk)
            if h2.hexdigest() != cur_h.hexdigest():
                final.unlink(missing_ok=True)
                raise RuntimeError(f"post-write hash mismatch for {final.name}")
            (out_sub / (name + ".sha256")).write_text(cur_h.hexdigest() + "\n", encoding="utf-8")
            (out_sub / (name + ".done")).touch()
            n_chunks += 1
        else:
            cur_tar.close()
            cur_fh.close()
            _tmp_name().unlink(missing_ok=True)
        chunk_no += 1
        cur_tar = cur_fh = cur_h = None
        cur_bytes = 0

    def open_chunk() -> None:
        nonlocal cur_fh, cur_tar, cur_h, cur_bytes
        cur_h = hashlib.sha256()
        # two-phase lifecycle: close_chunk() closes on rollover/finish
        cur_fh = open(_tmp_name(), "wb")  # noqa: SIM115
        cur_tar = tarfile.open(  # noqa: SIM115
            fileobj=_HashedFile(cur_fh, cur_h), mode="w", format=tarfile.USTAR_FORMAT
        )
        cur_bytes = 0

    def emit(stem: str) -> None:
        nonlocal cur_bytes
        p = pending.pop(stem)
        img_name, img_data = p["img"]
        js_name, js_data = p["json"]
        need = _pad512(len(img_data)) + _pad512(len(js_data)) + 1024
        if cur_bytes > 0 and cur_bytes + need > max_bytes:
            close_chunk()
            open_chunk()
        im = tarfile.TarInfo(img_name)
        im.size = len(img_data)
        cur_tar.addfile(im, io.BytesIO(img_data))
        jm = tarfile.TarInfo(js_name)
        jm.size = len(js_data)
        cur_tar.addfile(jm, io.BytesIO(js_data))
        cur_bytes += need
        emitted.add(stem)

    try:
        open_chunk()
        with tarfile.open(raw, "r") as tf:
            for m in tf:
                if not m.isfile():
                    continue
                b = posixpath.basename(m.name)
                stem = None
                kind = None
                for ext in IMG_EXTS:
                    if b.endswith(ext):
                        stem = b[: -len(ext)]
                        kind = "img"
                        break
                if stem is None and b.endswith(".json"):
                    stem = b[: -len(".json")]
                    kind = "json"
                if stem is None or stem not in p2set:
                    continue
                fobj = tf.extractfile(m)
                data = fobj.read() if fobj else b""
                p = pending.setdefault(stem, {})
                p[kind] = (m.name, data)
                if "img" in p and "json" in p:
                    emit(stem)
        close_chunk()
        missing = len(p2set - emitted)
        raw.with_name(raw.name + ".extracted").touch()
        print(
            f"[ex] {shard_rel}: emitted={len(emitted)} missing={missing} chunks={n_chunks}",
            flush=True,
        )
        return ("ok", shard_rel, len(emitted), missing)
    except Exception as exc:  # noqa: BLE001 - per-item isolation, reported via status
        print(f"[ex] ERROR {shard_rel}: {type(exc).__name__} {str(exc)[:200]}", flush=True)
        return ("error", shard_rel, len(emitted), -1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--shard-list", required=True)
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-bytes", type=int, default=2147483648)
    args = ap.parse_args()

    with open(args.manifest, encoding="utf-8") as fh:
        manifest = json.load(fh)
    sizes: dict[str, int] = {}
    with open(args.shard_list, encoding="utf-8") as fh:
        for ln in fh:
            p, s = ln.rstrip("\n").split("\t")
            sizes[p] = int(s)
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)

    failed: dict[str, int] = {}
    emitted_total = 0
    missing_total = 0
    submitted: set[str] = set()
    t0 = time.monotonic()
    last_progress = t0
    IDLE_LIMIT = 30 * 60  # no new shard ready for 30 min -> the download stage died
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures: dict = {}
        while True:
            # idempotent progress: count .extracted markers on disk (self-heals if
            # markers are removed for re-extraction)
            n_marker = 0
            if time.monotonic() - last_progress > IDLE_LIMIT:
                raise SystemExit(
                    f"stalled: no progress for {IDLE_LIMIT // 60} min "
                    f"(markers={n_marker}/{len(manifest)}); check the download stage"
                )
            for shard_rel in manifest:
                raw = raw_dir / shard_rel
                if raw.with_name(raw.name + ".extracted").exists():
                    n_marker += 1
                    continue
                if shard_rel in submitted or failed.get(shard_rel, 0) >= 2:
                    continue
                if raw.exists() and raw.stat().st_size == sizes[shard_rel]:
                    futures[
                        ex.submit(
                            _extract_one,
                            shard_rel,
                            manifest[shard_rel],
                            sizes[shard_rel],
                            raw_dir,
                            out_dir,
                            args.max_bytes,
                        )
                    ] = shard_rel
                    submitted.add(shard_rel)
                    last_progress = time.monotonic()
            for fut in list(futures):
                if not fut.done():
                    continue
                shard_rel = futures.pop(fut)
                submitted.discard(shard_rel)
                status, _, emitted, missing = fut.result()
                if status == "ok":
                    emitted_total += emitted
                    missing_total += missing
                    last_progress = time.monotonic()
                else:
                    failed[shard_rel] = failed.get(shard_rel, 0) + 1
            n_hard_fail = sum(1 for c in failed.values() if c >= 2)
            if n_marker + n_hard_fail >= len(manifest) and not futures:
                break
            time.sleep(10)
    print(
        f"[ex] DONE extracted_markers={n_marker} hard_failed={n_hard_fail} "
        f"emitted={emitted_total} missing_pairs={missing_total} elapsed={time.monotonic() - t0:.0f}s",
        flush=True,
    )
    if n_hard_fail:
        raise SystemExit(f"{n_hard_fail} shards failed twice: {[s for s, c in failed.items() if c >= 2][:20]}")


if __name__ == "__main__":
    main()
