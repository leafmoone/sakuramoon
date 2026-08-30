"""Full-repo streaming label scan of the raw modelscope v2 repo.

READ-ONLY. Streams EVERY shard of ``leafmoone/webdataset_danbooru_v2`` over
HTTP Range and reads ONLY the label metadata:

  * every ``*.json`` member: tar header + payload fetched and parsed
    (danbooru meta fields, the authoritative label source);
  * every ``*.webp`` member: 576 bytes fetched — the 512B tar header plus the
    64B RIFF/VP8X header (true shipped pixel size). Image pixel data is NEVER
    fetched and NOTHING is written to disk beyond per-shard checkpoint
    aggregates under ``--out-dir/checkpoints/``.

No local shard mirror, no downloader. The tar has no member index, so each
member is located by walking 512B headers with Range requests (the only way
to read the JSON without fetching the images). Throughput is request-bound
(~13ms/req keep-alive to the CDN): the default 64 parallel shards finish the
~3000-shard repo in a few hours.

webp<->json pairing is by member basename (``<id>.webp`` / ``<id>.json``),
order-independent.

Phases:
  scan   stream the (missing) shards in parallel, one checkpoint pickle each
  merge  read all checkpoints -> repo-wide SR-clean statistics + outputs
  all    scan then merge (default)

Eligibility/pools: the pipeline's own rules (classify_quality /
evaluate_eligibility / is_validation) with webp-corrected dims, identical to
``tools/analyze_sr_hr_quality_labels.py``; pools P0..P4 come from
``anime_sr.data.sr_clean_pools``. Nothing gates on clean_score.

Run (from the execution tree's anime-sr/ dir, on salt5):
  LD_LIBRARY_PATH=/opt/dtk-26.04/lib:/opt/dtk-26.04/hip/lib:/opt/hyhal/lib \
  PYTHONPATH=src python3.11 tools/stream_scan_v2_repo.py \
    --out-dir /root/private_data/anime-sr/reports-m4-sr-clean-v1-fullrepo
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import posixpath
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from urllib.parse import quote

import requests
from anime_sr.config.loader import load_config
from anime_sr.config.schema import Config
from anime_sr.data.index import (
    MetaRecord,
    _parse_meta,
    classify_quality,
    evaluate_eligibility,
    is_validation,
    webp_header_size,
)
from anime_sr.data.sr_clean_pools import (
    DEFAULT_SEED,
    DEFAULT_TOTAL_EXPOSURES,
    POOLS,
    RawMeta,
    _crop_bucket_hist,
    _crop_positions,
    _det_sample,
    _pctl_block,
)

DEFAULT_REPO = "leafmoone/webdataset_danbooru_v2"
_ZERO512 = b"\x00" * 512


# ---------------------------------------------------------------------------
# tar header walk (pure, unit-testable)
# ---------------------------------------------------------------------------
def pad512(n: int) -> int:
    return (n + 511) // 512 * 512


def parse_tar_header(hdr: bytes) -> tuple[str, int] | None:
    """(member_name, size) from a 512B ustar header; None at end-of-archive."""
    if len(hdr) < 512 or hdr == _ZERO512:
        return None
    name = hdr[0:100].rstrip(b"\x00").decode("utf-8", "replace")
    size_field = hdr[124:136].rstrip(b"\x00").strip()
    size = int(size_field, 8) if size_field else 0
    return name, size


def walk_shard(
    fetch,
    total: int,
    window: int,
    handle_webp,
    handle_json,
) -> tuple[int, int]:
    """Walk one tar stream via a ``fetch(start, end) -> bytes`` range callable.

    Handlers are called with the member basename stem (name without its final
    ``.webp``/``.json`` suffix):

      handle_webp(stem, data64) : first min(64, size) bytes of the webp payload
      handle_json(stem, payload): the full json payload

    Returns (n_webp, n_json). At most one window of bytes is buffered at a
    time; ``fetch`` must clamp end to total-1 (HttpFeed does).
    """
    off = 0
    buf = b""
    buf_start = 0
    n_webp = 0
    n_json = 0
    while off < total:
        if off < buf_start or off + 512 > buf_start + len(buf):
            buf = fetch(off, min(total - 1, off + window - 1))
            buf_start = off
        parsed = parse_tar_header(buf[off - buf_start : off - buf_start + 512])
        if parsed is None:
            break
        name, size = parsed
        stem = posixpath.basename(name)
        if stem.endswith(".webp"):
            stem = stem[: -len(".webp")]
        elif stem.endswith(".json"):
            stem = stem[: -len(".json")]
        start = off + 512
        if name.endswith(".webp") and size > 0:
            n_webp += 1
            take = min(64, size)
            end = start + take - 1
            if start >= buf_start and end < buf_start + len(buf):
                data = buf[start - buf_start : start - buf_start + take]
            else:
                data = fetch(start, end)
            handle_webp(stem, data)
        elif name.endswith(".json") and size > 0:
            n_json += 1
            end = start + size - 1
            if start >= buf_start and end < buf_start + len(buf):
                payload = buf[start - buf_start : start - buf_start + size]
            else:
                payload = fetch(start, end)
            handle_json(stem, payload)
        off += 512 + pad512(size)
    return n_webp, n_json


# ---------------------------------------------------------------------------
# HTTP range feed (keep-alive, retry, CDN re-resolve)
# ---------------------------------------------------------------------------
class HttpFeed:
    def __init__(self, session: requests.Session, blob_url: str, total: int, timeout: int = 30) -> None:
        self.session = session
        self.blob_url = blob_url
        self.total = total
        self.timeout = timeout
        self.cdn: str | None = None
        self.n_req = 0
        self.bytes_fetched = 0

    def _resolve(self) -> None:
        r = self.session.get(self.blob_url, headers={"Range": "bytes=0-0"}, timeout=self.timeout, allow_redirects=True)
        self.n_req += 1
        self.bytes_fetched += len(r.content)
        r.raise_for_status()
        self.cdn = r.url

    def get(self, s: int, e: int) -> bytes:
        e = min(e, self.total - 1)
        last_exc: Exception | None = None
        for attempt in range(6):
            if self.cdn is None:
                try:
                    self._resolve()
                except requests.RequestException as exc:
                    last_exc = exc
                    time.sleep(min(2**attempt, 15))
                    continue
            try:
                assert self.cdn is not None
                r = self.session.get(self.cdn, headers={"Range": f"bytes={s}-{e}"}, timeout=self.timeout)
                self.n_req += 1
                self.bytes_fetched += len(r.content)
                if r.status_code in (400, 401, 403, 404, 410):
                    self.cdn = None  # signature likely expired -> re-resolve
                    time.sleep(1 + attempt)
                    continue
                r.raise_for_status()
                return r.content
            except requests.RequestException as exc:
                last_exc = exc
                time.sleep(min(2**attempt, 15))
        raise RuntimeError(f"range fetch {s}-{e} failed after retries: {last_exc}")


# ---------------------------------------------------------------------------
# shard listing (modelscope tree API)
# ---------------------------------------------------------------------------
def list_repo_shards(
    session: requests.Session,
    tok: str,
    repo: str,
    rev: str,
    prefixes: tuple[str, ...] | None = None,
) -> list[tuple[str, int]]:
    session.headers.update({"Authorization": f"Bearer {tok}", "Cookie": f"m_session_id={tok}"})
    tars: list[tuple[str, int]] = []
    page = 1
    while True:
        url = (
            f"https://modelscope.cn/api/v1/datasets/{repo}/repo/tree"
            f"?Revision={rev}&Recursive=True&PageNumber={page}&PageSize=200"
        )
        r = session.get(url, timeout=60)
        r.raise_for_status()
        files = r.json().get("Data", {}).get("Files", []) or []
        for f in files:
            if f["Type"] == "blob" and f["Path"].endswith(".tar"):
                p = f["Path"]
                if prefixes is None or any(p.startswith(x) for x in prefixes):
                    tars.append((p, int(f.get("Size", 0))))
        if len(files) < 200 or page > 500:
            break
        page += 1
    return sorted(tars)


def load_token(token_file: str | None) -> str:
    env = os.environ.get("MODELSCOPE_API_TOKEN")
    if env:
        return env
    if token_file:
        for line in Path(token_file).read_text(encoding="utf-8").splitlines():
            if line.startswith("MODELSCOPE_API_TOKEN="):
                return line.split("=", 1)[1].strip()
    raise SystemExit("no token: set MODELSCOPE_API_TOKEN or pass --token-file")


# ---------------------------------------------------------------------------
# worker: stream one shard (module globals set in main before fork)
# ---------------------------------------------------------------------------
_CFG: Config
_BUCKET_HR: int
_WINDOW: int
_TOK: str
_REPO: str
_REV: str


def _stream_shard(task: tuple[str, int]) -> dict:
    path, total = task
    cfg = _CFG
    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {_TOK}", "Cookie": f"m_session_id={_TOK}"})
    blob_url = (
        f"https://modelscope.cn/api/v1/datasets/{_REPO}/repo?Revision={_REV}&FilePath={quote(path)}"
    )
    feed = HttpFeed(session, blob_url, total)
    t0 = time.monotonic()

    nsfw_c: Counter[str] = Counter()
    n_unresolved = 0
    n_fallback = 0
    n_val = 0
    webp_sizes: dict[str, tuple[int, int]] = {}  # basename stem -> shipped dims
    pop: list[tuple] = []
    cq: Counter = Counter()
    clsq: Counter = Counter()
    ccls: Counter = Counter()
    c3: Counter = Counter()

    def h_webp(stem: str, data: bytes) -> None:
        sz = webp_header_size(data)
        if sz is not None:
            webp_sizes[stem] = sz

    def h_json(stem: str, payload: bytes) -> None:
        nonlocal n_unresolved, n_fallback, n_val
        rec = _parse_meta(payload)
        if rec is None:
            n_unresolved += 1
            return
        nsfw_c[rec.nsfw or "unknown"] += 1
        # shipped webp dims when the paired webp member gave them, else the
        # original-post dims from the meta (counted as a fallback)
        w, h = webp_sizes.get(stem, (rec.width, rec.height))
        if stem not in webp_sizes:
            n_fallback += 1
        mrec = MetaRecord(
            sample_id=rec.sample_id,
            shard=path,
            rel_path="",
            width=w,
            height=h,
            nsfw=rec.nsfw,
            year=rec.year,
            quality="",
            tags_general=rec.tags_general,
            quality_tier=rec.quality_tier,
            anime_completeness=rec.anime_completeness,
            anime_classification=rec.anime_classification,
            ai_corrupted=rec.ai_corrupted,
        )
        mrec = classify_quality(mrec, cfg)
        el = evaluate_eligibility(mrec, cfg)
        if el.eligible_train and _BUCKET_HR in el.eligible_buckets:
            if is_validation(mrec, cfg):
                n_val += 1
            else:
                comp = rec.anime_completeness or "unknown"
                cls = rec.anime_classification or "unknown"
                tier = rec.quality_tier or "unknown"
                pop.append((rec.sample_id, w, h, tier, comp, cls, rec.ai_corrupted))
                cq[(comp, tier)] += 1
                clsq[(cls, tier)] += 1
                ccls[(comp, cls)] += 1
                c3[(comp, cls, tier)] += 1

    n_webp, n_json = walk_shard(feed.get, total, _WINDOW, h_webp, h_json)
    return {
        "path": path,
        "size": total,
        "n_json": n_json,
        "n_webp": n_webp,
        "n_unresolved": n_unresolved,
        "nsfw": dict(nsfw_c),
        "n_size_fallback": n_fallback,
        "n_val_eligible": n_val,
        "cross_cq": dict(cq),
        "cross_cls_q": dict(clsq),
        "cross_c_cls": dict(ccls),
        "c3": dict(c3),
        "pop": pop,
        "requests": feed.n_req,
        "bytes_fetched": feed.bytes_fetched,
        "elapsed_s": round(time.monotonic() - t0, 2),
    }


# ---------------------------------------------------------------------------
# merge: checkpoints -> repo-wide outputs
# ---------------------------------------------------------------------------
def _ckpt_name(path: str) -> str:
    return path.replace("/", "__") + ".pkl"


def merge(out_dir: Path, cfg: Config, bucket_hr: int, seed: int, total_exposures: int) -> dict:
    t0 = time.monotonic()
    ckdir = out_dir / "checkpoints"
    ckpts = sorted(ckdir.glob("*.pkl"))
    if not ckpts:
        raise SystemExit(f"no checkpoints under {ckdir}")

    pop: list[tuple] = []
    nsfw_dist: Counter[str] = Counter()
    n_json = 0
    n_webp = 0
    n_unresolved = 0
    n_fallback = 0
    n_val = 0
    cq: Counter = Counter()
    clsq: Counter = Counter()
    ccls: Counter = Counter()
    c3: Counter = Counter()
    shard_info: dict[str, dict] = {}
    total_requests = 0
    total_bytes = 0
    total_elapsed = 0.0
    for cp in ckpts:
        with cp.open("rb") as fh:
            r = pickle.load(fh)
        pop.extend(r["pop"])
        nsfw_dist.update(r["nsfw"])
        n_json += r["n_json"]
        n_webp += r["n_webp"]
        n_unresolved += r["n_unresolved"]
        n_fallback += r["n_size_fallback"]
        n_val += r["n_val_eligible"]
        cq.update(r["cross_cq"])
        clsq.update(r["cross_cls_q"])
        ccls.update(r["cross_c_cls"])
        c3.update(r["c3"])
        total_requests += r["requests"]
        total_bytes += r["bytes_fetched"]
        total_elapsed += r["elapsed_s"]
        shard_info[r["path"]] = {
            "n_images": r["n_json"],
            "n_unresolved_json": r["n_unresolved"],
            "n_webp": r["n_webp"],
            "n_1024_eligible_train": len(r["pop"]),
            "n_1024_eligible_validation": r["n_val_eligible"],
            "requests": r["requests"],
            "elapsed_s": r["elapsed_s"],
        }

    # rebuild RawMeta for the pool predicates
    metas: dict[str, RawMeta] = {}
    sizes: dict[str, tuple[int, int]] = {}
    for sid, w, h, tier, comp, cls, ai in pop:
        metas[sid] = RawMeta(
            sample_id=sid,
            shard=sid,
            width_meta=w,
            height_meta=h,
            nsfw="",
            year=0,
            quality_tier=tier,
            completeness=comp,
            classification=cls,
            ai_corrupted=ai,
            tags_general=(),
        )
        sizes[sid] = (w, h)

    # per-directory breakdown (shard path prefix)
    per_dir: dict[str, dict] = {}
    for p in shard_info:
        d = p.rsplit("/", 1)[0]
        per_dir.setdefault(
            d,
            {"n_shards": 0, "n_images": 0, "n_unresolved": 0, "n_webp": 0, "n_1024_eligible_train": 0},
        )
        per_dir[d]["n_shards"] += 1
        per_dir[d]["n_images"] += shard_info[p]["n_images"]
        per_dir[d]["n_unresolved"] += shard_info[p]["n_unresolved_json"]
        per_dir[d]["n_webp"] += shard_info[p]["n_webp"]
        per_dir[d]["n_1024_eligible_train"] += shard_info[p]["n_1024_eligible_train"]

    # schema notes for unresolved json (probed on salt5, 08-31):
    # artstation-2D json carry id:"" (no danbooru post id), no
    # quality/anime_completeness/anime_classification fields, and their image
    # members are .jpg (no webp) -> not indexable by the pipeline (sample_id
    # is the join key); background-2D is one shard / 98 images (negligible);
    # the danbooru dirs show the normal ~0.2% malformed/null-dim rate.
    non_danbooru = ("data/artstation-2D", "data/background-2D")
    unresolved_notes = {
        "data/artstation-2D": (
            "artstation 架构：id 为空（无 danbooru post id）+ 无 quality/anime_completeness/"
            "anime_classification 字段 + 图像成员为 .jpg（无 webp）→ 无法建索引（0 eligible 是"
            "「不可索引」而非「无好图」；标签/尺寸本身可读，若需利用要单独打标）"
        ),
        "data/background-2D": "1 片 98 图，量级可忽略，未单独探测 schema",
    }
    danbooru_json = sum(v["n_images"] for k, v in per_dir.items() if k not in non_danbooru)
    danbooru_unres = sum(v["n_unresolved"] for k, v in per_dir.items() if k not in non_danbooru)
    for k in per_dir:
        if k not in unresolved_notes and k not in non_danbooru:
            unresolved_notes[k] = (
                f"malformed / 维度为 null（{danbooru_unres}/{danbooru_json} = "
                f"{(danbooru_unres / danbooru_json) if danbooru_json else 0:.2%}，与生产语料同档）"
            )

    def _table2(counter: Counter) -> dict[str, dict[str, int]]:
        out: dict[str, dict[str, int]] = {}
        for (a, b), n in counter.items():
            out.setdefault(a, {})[b] = n
        return out

    cross = {
        "completeness_x_quality": _table2(cq),
        "classification_x_quality": _table2(clsq),
        "completeness_x_classification": _table2(ccls),
    }
    top30_3d = [
        {"completeness": a, "classification": b, "quality": q, "n": n} for (a, b, q), n in c3.most_common(30)
    ]

    pool_ids: dict[str, list[str]] = {}
    for name, pred in POOLS:
        pool_ids[name] = sorted(sid for sid in metas if pred(metas[sid]))
    p0 = set(pool_ids["P0_priority"])
    p1 = set(pool_ids["P1_sr_clean_v1"])
    p2 = set(pool_ids["P2_sr_clean_wide"])
    n_full = len(metas)

    pool_sizes = [
        {"pool": name, "n": len(pool_ids[name]), "share_of_full": (len(pool_ids[name]) / n_full) if n_full else 0.0}
        for name, _ in POOLS
    ]

    def _dim_list(ids: list[str]) -> tuple[list[int], list[int]]:
        ws: list[int] = []
        hs: list[int] = []
        for sid in ids:
            w, h = sizes[sid]
            ws.append(w)
            hs.append(h)
        return ws, hs

    p1_ids = pool_ids["P1_sr_clean_v1"]
    p1_ws, p1_hs = _dim_list(p1_ids)
    p1_detail = {
        "n_unique_images": len(p1_ids),
        "quality": dict(Counter(metas[s].quality_tier or "unknown" for s in p1_ids)),
        "classification": dict(Counter(metas[s].classification or "unknown" for s in p1_ids)),
        "completeness": dict(Counter(metas[s].completeness or "unknown" for s in p1_ids)),
        "width": _pctl_block(p1_ws),
        "height": _pctl_block(p1_hs),
        "min_dim": _pctl_block([min(w, h) for w, h in zip(p1_ws, p1_hs)]),
        "max_dim": _pctl_block([max(w, h) for w, h in zip(p1_ws, p1_hs)]),
    }
    p1_exact: list[int] = []
    p1_64: list[int] = []
    for sid in p1_ids:
        e, s64 = _crop_positions(sizes[sid][0], sizes[sid][1], bucket_hr)
        p1_exact.append(e)
        p1_64.append(s64)
    p1_detail["crop_flexibility_exact"] = _crop_bucket_hist(p1_exact)
    p1_detail["crop_flexibility_stride64"] = _crop_bucket_hist(p1_64)

    exposures = []
    for name, _ in POOLS:
        if name in ("P3_lineart_extra", "P4_rough_extra"):
            continue
        ids = pool_ids[name]
        n = len(ids)
        pos64 = [_crop_positions(sizes[s][0], sizes[s][1], bucket_hr)[1] for s in ids]
        exposures.append(
            {
                "pool": name,
                "n": n,
                "mean_exposures_per_source": (total_exposures / n) if n else None,
                "sum_crop_positions_stride64": sum(pos64),
                "mean_exposures_per_possible_crop_proxy": (total_exposures / sum(pos64)) if pos64 else None,
            }
        )
    exposures.append(
        {
            "pool": "full_eligible",
            "n": n_full,
            "mean_exposures_per_source": (total_exposures / n_full) if n_full else None,
            "sum_crop_positions_stride64": None,
            "mean_exposures_per_possible_crop_proxy": None,
        }
    )

    def _diff_stats(added: set[str]) -> dict:
        tiers = Counter(metas[s].quality_tier or "unknown" for s in added)
        cls_c = Counter(metas[s].classification or "unknown" for s in added)
        comp = Counter(metas[s].completeness or "unknown" for s in added)
        return {
            "n_added": len(added),
            "quality_tier": dict(tiers),
            "classification": dict(cls_c),
            "completeness": dict(comp),
        }

    diffs = {
        "P1_minus_P0": _diff_stats(p1 - p0),
        "P2_minus_P1": _diff_stats(p2 - p1),
    }

    sample_lists: dict[str, list[str]] = {
        "p0_priority_samples": _det_sample(pool_ids["P0_priority"], 100, seed),
        "p1_added_good_normal_samples": _det_sample(sorted(p1 - p0), 200, seed),
        "p2_low_worst_samples": _det_sample(sorted(p2 - p1), 200, seed),
        "p3_lineart_samples": _det_sample(pool_ids["P3_lineart_extra"], 200, seed),
        "p4_rough_samples": _det_sample(pool_ids["P4_rough_extra"], 100, seed),
    }

    answers = {
        "mode": "full-repo streaming scan (raw labels only, no frozen table)",
        "q1_full_1024_eligible_train": n_full,
        "q2_p0_strict_priority": len(p0),
        "q3_p1_sr_clean_v1": len(p1),
        "q4_p1_minus_p0_is_good_normal_polished": diffs["P1_minus_P0"],
        "q5_p2_minus_p1_low_worst": diffs["P2_minus_P1"],
        "q6_recommendation": "NOT SET (full-repo scan is sizing data for future corpus expansion)",
    }
    full = {
        "params": {
            "mode": "full-repo-streaming",
            "bucket_hr": bucket_hr,
            "seed": seed,
            "total_exposures": total_exposures,
            "n_shards": len(shard_info),
        },
        "shard_scan": shard_info,
        "coverage": {
            "n_scanned_json": n_json,
            "n_scanned_webp": n_webp,
            "n_unresolved_json": n_unresolved,
            "unresolved_by_dir": {k: v["n_unresolved"] for k, v in sorted(per_dir.items())},
            "unresolved_notes": unresolved_notes,
            "n_full_eligible_train": n_full,
            "n_full_eligible_validation": n_val,
            "nsfw_distribution": dict(nsfw_dist),
            "n_webp_size_fallback_to_meta": n_fallback,
        },
        "stream": {
            "requests": total_requests,
            "bytes_fetched": total_bytes,
            "wall_s_sum": round(total_elapsed, 1),
        },
        "per_dir": per_dir,
        "cross_tables": cross,
        "top30_3d": top30_3d,
        "pool_sizes": pool_sizes,
        "p1_detail": p1_detail,
        "exposures_6m": exposures,
        "diffs": diffs,
        "answers": answers,
        "sample_counts": {k: len(v) for k, v in sample_lists.items()},
        "elapsed_s": round(time.monotonic() - t0, 2),
    }
    (out_dir / "full.json").write_text(json.dumps(full, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out_dir / "sr-clean-v1-fullrepo-summary.json").write_text(
        json.dumps(
            {"answers": answers, "pool_sizes": pool_sizes, "per_dir": per_dir, "coverage": full["coverage"]},
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    with (out_dir / "sr-clean-v1-fullrepo-sample-ids.txt").open("w", encoding="utf-8") as fh:
        fh.write("\n".join(sorted(p1)) + "\n")
    for name, ids in sample_lists.items():
        with (out_dir / f"{name}.txt").open("w", encoding="utf-8") as fh:
            for sid in ids:
                fh.write(f"{sid}\n")

    def _write_csv(fname: str, header: list[str], rows: list[list]) -> None:
        with (out_dir / fname).open("w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(header)
            w.writerows(rows)

    for tname, table in cross.items():
        keys_b = sorted({b for sub in table.values() for b in sub})
        _write_csv(
            f"{tname}.csv",
            [tname.split("_x_")[0], *keys_b],
            [[a, *[sub.get(b, 0) for b in keys_b]] for a, sub in sorted(table.items())],
        )
    _write_csv(
        "top30_3d.csv",
        ["completeness", "classification", "quality", "n"],
        [[d["completeness"], d["classification"], d["quality"], d["n"]] for d in top30_3d],
    )
    _write_csv(
        "pool_sizes.csv",
        ["pool", "n", "share_of_full"],
        [[d["pool"], d["n"], d["share_of_full"]] for d in pool_sizes],
    )
    _write_csv(
        "exposures_6m.csv",
        ["pool", "n", "mean_exposures_per_source", "sum_crop_positions_stride64", "mean_exposures_per_possible_crop_proxy"],
        [
            [d["pool"], d["n"], d["mean_exposures_per_source"], d["sum_crop_positions_stride64"], d["mean_exposures_per_possible_crop_proxy"]]
            for d in exposures
        ],
    )
    _write_csv(
        "per_dir.csv",
        ["dir", "n_shards", "n_images", "n_unresolved", "n_webp", "n_1024_eligible_train"],
        [
            [d, v["n_shards"], v["n_images"], v["n_unresolved"], v["n_webp"], v["n_1024_eligible_train"]]
            for d, v in sorted(per_dir.items())
        ],
    )

    md: list[str] = []
    md.append("# Full-repo streaming scan — SR-clean label statistics (v2 repo)")
    md.append("")
    md.append(f"- shards: **{len(shard_info)}**; json members: {n_json}; webp headers: {n_webp}; unresolved json: {n_unresolved}")
    md.append(f"- nsfw distribution (scanned): {dict(nsfw_dist)}")
    md.append(f"- population: **{n_full}** 1024-eligible train images (+{n_val} validation)")
    md.append(f"- webp-size fallbacks to meta dims: {n_fallback}")
    md.append(f"- stream cost: {total_requests} range requests, {total_bytes / 1e9:.1f} GB transferred (transient 8KB windows; only json+headers useful; zero disk writes)")
    md.append("")
    md.append("## Per directory")
    md.append("")
    md.append("| dir | shards | images | unresolved | webp members | 1024-eligible train |")
    md.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for d, v in sorted(per_dir.items()):
        md.append(
            f"| {d} | {v['n_shards']} | {v['n_images']} | {v['n_unresolved']} | {v['n_webp']} | {v['n_1024_eligible_train']} |"
        )
    md.append("")
    md.append("## 未解析 json 分解（schema 说明）")
    md.append("")
    md.append("| dir | unresolved | 说明 |")
    md.append("| --- | ---: | --- |")
    for d, v in sorted(per_dir.items(), key=lambda kv: -kv[1]["n_unresolved"]):
        if v["n_unresolved"]:
            md.append(f"| {d} | {v['n_unresolved']} | {unresolved_notes.get(d, '')} |")
    md.append("")
    md.append(
        f"- webp 尺寸回退：{n_fallback} 张（tar 内无配对 .webp 成员或头不可读 → 用 meta 尺寸；"
        "几乎全部位于 2_2026.1，该目录图像为原始 png/jpg 无缩放，meta 尺寸=实际尺寸（抽验 3/3）→ 无高估）"
    )
    for d, v in sorted(per_dir.items()):
        if v["n_images"] and v["n_1024_eligible_train"] and v["n_webp"] < 0.5 * v["n_images"]:
            md.append(
                f"- {d}：仅 {v['n_webp']}/{v['n_images']} 图成员为 webp，其余为原始 png/jpg（无缩放步骤）；"
                "抽验 meta 尺寸=实际文件尺寸（2_2026.1 3/3）→ eligible 判定可靠，"
                "但该目录与 danbooru webp 语料不是同一图像管线"
            )
    art_n = per_dir.get("data/artstation-2D", {}).get("n_images", 0)
    if art_n:
        md.append(
            f"- artstation-2D 的 {art_n} 张图 json 可读（tags+尺寸齐全）但不可索引；若未来要利用需单独打 danbooru 风格标签并赋 id"
        )
    md.append("")
    md.append("## Pool sizes")
    md.append("")
    md.append("| pool | n | share of full |")
    md.append("| --- | ---: | ---: |")
    for d in pool_sizes:
        md.append(f"| {d['pool']} | {d['n']} | {d['share_of_full']:.1%} |")
    md.append("")
    md.append("## 6M repetition intensity (reference; 6M was the M4 run size)")
    md.append("")
    md.append("| pool | n | mean exposures/source | sum crop-positions (stride64) | mean exposures/possible-crop proxy |")
    md.append("| --- | ---: | ---: | ---: | ---: |")
    for d in exposures:
        sum64 = d["sum_crop_positions_stride64"]
        proxy = d["mean_exposures_per_possible_crop_proxy"]
        md.append(
            f"| {d['pool']} | {d['n']} | {d['mean_exposures_per_source'] or 0:.0f} | "
            f"{sum64 if sum64 is not None else '-'} | "
            f"{(format(proxy, '.2f') if proxy else '-')} |"
        )
    md.append("")
    md.append("## Diffs")
    md.append("")
    md.append(f"- P1 - P0: +{diffs['P1_minus_P0']['n_added']} (quality: {diffs['P1_minus_P0']['quality_tier']})")
    md.append(f"- P2 - P1: +{diffs['P2_minus_P1']['n_added']} (quality: {diffs['P2_minus_P1']['quality_tier']})")
    (out_dir / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "out_dir": str(out_dir),
                "n_shards": len(shard_info),
                "n_full": n_full,
                "pools": {d["pool"]: d["n"] for d in pool_sizes},
                "requests": total_requests,
            }
        )
    )
    return full


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    global _CFG, _BUCKET_HR, _WINDOW, _TOK, _REPO, _REV
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--rev", default="master")
    ap.add_argument("--token-file", default="/root/private_data/anime-sr/.ms-token")
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--shards", default=None, help="comma-separated shard-path prefixes to restrict (default: all)")
    ap.add_argument("--limit", type=int, default=0, help="scan at most N shards (0 = all)")
    ap.add_argument("--window", type=int, default=8192, help="byte window per fresh Range fetch")
    ap.add_argument("--bucket-hr", type=int, default=1024)
    ap.add_argument("--config", nargs="+", default=["config/base.toml", "config/data.toml", "config/m4_1024.toml"])
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--total-exposures", type=int, default=DEFAULT_TOTAL_EXPOSURES)
    ap.add_argument("--phase", choices=["scan", "merge", "all"], default="all")
    ap.add_argument("--no-resume", action="store_true", help="re-scan shards even if a checkpoint exists")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    ckdir = out_dir / "checkpoints"
    ckdir.mkdir(parents=True, exist_ok=True)

    cfg: Config = load_config(*args.config)

    if args.phase in ("scan", "all"):
        _CFG = cfg
        _BUCKET_HR = args.bucket_hr
        _WINDOW = args.window
        _TOK = load_token(args.token_file)
        _REPO = args.repo
        _REV = args.rev

        session = requests.Session()
        prefixes = tuple(p for p in args.shards.split(",") if p) if args.shards else None
        shards = list_repo_shards(session, _TOK, args.repo, args.rev, prefixes)
        if args.limit:
            shards = shards[: args.limit]
        (out_dir / "shard-list.tsv").write_text("".join(f"{p}\t{s}\n" for p, s in shards), encoding="utf-8")
        total = len(shards)
        pending = [t for t in shards if args.no_resume or not (ckdir / _ckpt_name(t[0])).exists()]
        print(f"[stream] repo={args.repo} shards={total} pending={len(pending)} workers={args.workers}", flush=True)
        if pending:
            t_start = time.monotonic()
            done = 0
            with ProcessPoolExecutor(max_workers=max(1, args.workers)) as ex:
                for res in ex.map(_stream_shard, pending):
                    done += 1
                    with (ckdir / _ckpt_name(res["path"])).open("wb") as fh:
                        pickle.dump(res, fh)
                    elapsed = time.monotonic() - t_start
                    eta = (elapsed / done) * (len(pending) - done)
                    print(
                        f"[{done}/{len(pending)}] {res['path']} json={res['n_json']} "
                        f"pop+{len(res['pop'])} req={res['requests']} "
                        f"req/s={res['requests'] / res['elapsed_s']:.0f} "
                        f"elapsed={elapsed / 60:.1f}m eta={eta / 60:.1f}m",
                        flush=True,
                    )
        else:
            print("[stream] all shards already checkpointed; skipping scan", flush=True)

    if args.phase in ("merge", "all"):
        merge(out_dir, cfg, args.bucket_hr, args.seed, args.total_exposures)


if __name__ == "__main__":
    # uncaught exceptions print the full traceback and exit(1) (project CLI rule)
    main()
