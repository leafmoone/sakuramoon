"""M4-1024 HR-GT quality pool statistics — SR-clean-v1 candidate definition.

READ-ONLY label analysis over the production M4 corpus.

Label source (authoritative, per the 08-31 work order):
  the raw danbooru-v2 webdataset shards of the modelscope repo
  ``leafmoone/webdataset_danbooru_v2`` (master, ``data/1_2024/shard-NNNNNN.tar``),
  streamed tar-by-tar IN PARALLEL: every ``*.json`` member is parsed for the
  danbooru meta fields, every ``*.webp`` member is read for its 64-byte header
  (true shipped pixel size) but NEVER decoded. The shards must be a local
  mirror of the repo (``--shard-dir``), produced by
  ``scripts/remote/dl_danbooru_v2.sh`` (resumable ranged downloader).

Eligibility source (derived, filter only):
  the production ``sr-eligibility-v1`` table (``--index-dir``). It defines the
  1024 train-eligible population (eligible_train AND bucket in
  eligible_buckets AND NOT is_validation). The tool RE-DERIVES eligibility
  from the raw labels with the pipeline's own rules (classify_quality /
  evaluate_eligibility / is_validation) and reports any mismatch with the
  frozen table.

Hard constraints honoured:
  * no training logic / index / model / flow / loss / sampler changes;
  * clean_score stays a correlation analysis only (no hard gate,
    clean_score_min stays -1.0);
  * the production index is never written; the only outputs go to --out-dir;
  * deterministic: sorted iteration + fixed sampling seed.

Outputs (all under --out-dir, e.g. reports/m4-sr-clean-v1/):
  full.json                       complete machine-readable result
  summary.md                      human-readable report
  *.csv                           cross tables / pool sizes / percentiles
  p0_priority_samples.txt         100 deterministic sample ids
  p1_added_good_normal_samples.txt  200 (P1 - P0) ids
  p2_low_worst_samples.txt        200 (P2 - P1) ids
  p3_lineart_samples.txt          200 P3 ids
  p4_rough_samples.txt            100 P4 ids
  sr-clean-v1-sample-ids.txt      FULL P1 candidate id list (sorted)
  sr-clean-v1-summary.json        compact summary + the 6 answers

Run (from the execution tree's anime-sr/ dir):
  python tools/analyze_sr_hr_quality_labels.py \
    --shard-dir /root/private_data/anime-sr/danbooru-v2/data/1_2024 \
    --index-dir /root/private_data/anime-sr/data/index-p1formal \
    --bucket-hr 1024 \
    --clean-score /root/private_data/anime-sr/data/index-p1formal/clean-score-v1.jsonl \
    --out-dir /root/private_data/anime-sr/reports-m4-sr-clean-v1
"""

from __future__ import annotations

import argparse
import csv
import json
import posixpath
import tarfile
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

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
    _spearman,
    quality_ordinal,
)


# ---------------------------------------------------------------------------
# raw shard scanning (parallel workers)
# ---------------------------------------------------------------------------
def _scan_shard(path: str) -> tuple[str, dict[str, RawMeta], dict[str, tuple[int, int]], int]:
    """Stream one shard tar; parse every .json, header-read every .webp.

    Returns (shard_name, metas by id, webp pixel sizes by id, n_unresolved).
    """
    metas: dict[str, RawMeta] = {}
    sizes: dict[str, tuple[int, int]] = {}
    n_unresolved = 0
    with tarfile.open(path, "r") as tf:
        for member in tf.getmembers():
            if not member.isfile():
                continue
            name = member.name
            f = tf.extractfile(member)
            if f is None:
                continue
            if name.endswith(".json"):
                base = name.rsplit(".", 1)[0]
                if base.endswith(".json"):
                    base = base.rsplit(".", 1)[0]
                rec = _parse_meta(f.read())
                if rec is None:
                    n_unresolved += 1
                    continue
                metas[rec.sample_id] = RawMeta(
                    sample_id=rec.sample_id,
                    shard=Path(path).name,
                    width_meta=rec.width,
                    height_meta=rec.height,
                    nsfw=rec.nsfw,
                    year=rec.year,
                    quality_tier=rec.quality_tier,
                    completeness=rec.anime_completeness,
                    classification=rec.anime_classification,
                    ai_corrupted=rec.ai_corrupted,
                    tags_general=rec.tags_general,
                )
            elif name.endswith(".webp"):
                # member names carry a repo path prefix (danbooru/5.9/1_2024/<id>.webp);
                # the id is the basename, matching the json's content ``id``.
                base = posixpath.basename(name)
                base = base.rsplit(".", 1)[0]
                sz = webp_header_size(f.read(64))
                if sz is not None:
                    sizes[base] = sz
    return Path(path).name, metas, sizes, n_unresolved


# ---------------------------------------------------------------------------
# full-repo population (no frozen table: rules applied to raw labels only)
# ---------------------------------------------------------------------------
def select_full_repo_population(
    all_metas: dict[str, RawMeta],
    sizes: dict[str, tuple[int, int]],
    cfg: Config,
    bucket_hr: int,
) -> tuple[set[str], Counter[str], int]:
    """Re-derive the train population from raw labels alone (full-repo mode).

    Applies the pipeline's own rules (classify_quality / evaluate_eligibility /
    is_validation) with webp-corrected dims (meta fallback) to every scanned
    image. Returns (train_ids, nsfw distribution of ALL scanned,
    n_1024_eligible_validation).
    """
    train: set[str] = set()
    nsfw_dist: Counter[str] = Counter()
    n_val = 0
    for sid, m in all_metas.items():
        nsfw_dist[m.nsfw or "unknown"] += 1
        w, h = sizes.get(sid, (m.width_meta, m.height_meta))
        rec = MetaRecord(
            sample_id=m.sample_id,
            shard=m.shard,
            rel_path="",
            width=w,
            height=h,
            nsfw=m.nsfw,
            year=m.year,
            quality="",
            tags_general=m.tags_general,
            quality_tier=m.quality_tier,
            anime_completeness=m.completeness,
            anime_classification=m.classification,
            ai_corrupted=m.ai_corrupted,
        )
        rec = classify_quality(rec, cfg)
        el = evaluate_eligibility(rec, cfg)
        if el.eligible_train and bucket_hr in el.eligible_buckets:
            if is_validation(rec, cfg):
                n_val += 1
            else:
                train.add(sid)
    return train, nsfw_dist, n_val


# ---------------------------------------------------------------------------
# eligibility table
# ---------------------------------------------------------------------------
def _load_eligibility_rows(index_dir: Path) -> list[dict]:
    for cand in (index_dir / "sr-eligibility-v1.parquet", index_dir / "sr-eligibility-v1.jsonl"):
        if cand.exists():
            if cand.suffix == ".parquet":
                import pyarrow.parquet as pq

                return pq.read_table(cand).to_pylist()
            with cand.open("r", encoding="utf-8") as fh:
                return [json.loads(line) for line in fh if line.strip()]
    raise FileNotFoundError(f"no sr-eligibility-v1 (parquet/jsonl) under {index_dir}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--shard-dir", required=True, help="dir with the danbooru-v2 shard tars (repo mirror)")
    ap.add_argument(
        "--index-dir",
        default=None,
        help="dir with sr-eligibility-v1 (production eligibility table); OMIT for full-repo mode (population re-derived from raw labels only)",
    )
    ap.add_argument("--bucket-hr", type=int, default=1024)
    ap.add_argument("--config", nargs="+", default=["config/base.toml", "config/data.toml", "config/m4_1024.toml"])
    ap.add_argument("--clean-score", default=None, help="clean-score sidecar jsonl (correlation only; never gates)")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--total-exposures", type=int, default=DEFAULT_TOTAL_EXPOSURES)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--verdict-file", default=None, help="optional text file with the analyst verdict (embedded verbatim)")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    t0 = time.monotonic()
    cfg: Config = load_config(*args.config)
    shard_dir = Path(args.shard_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    full_repo = args.index_dir is None

    # ---- 1. population: frozen eligibility table OR re-derived (full-repo)
    rows: dict[str, dict] = {}
    if not full_repo:
        for r in _load_eligibility_rows(Path(args.index_dir)):
            if r.get("eligible_train") and args.bucket_hr in (r.get("eligible_buckets") or []) and not r.get("is_validation"):
                rows[str(r["sample_id"])] = r

    # ---- 2. stream the raw v2 shards in parallel -------------------------
    shard_paths = sorted(str(p) for p in shard_dir.iterdir() if p.name.endswith(".tar"))
    if not shard_paths:
        raise SystemExit(f"no .tar shards under {shard_dir}")
    all_metas: dict[str, RawMeta] = {}
    all_sizes: dict[str, tuple[int, int]] = {}
    scan_stats: dict[str, dict] = {}
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as ex:
        for shard_name, metas, sizes, n_unres in ex.map(_scan_shard, shard_paths):
            all_metas.update(metas)
            all_sizes.update(sizes)
            scan_stats[shard_name] = {"n_images": len(metas), "n_unresolved_json": n_unres}

    if full_repo:
        pop_ids, nsfw_dist, n_val_eligible = select_full_repo_population(all_metas, all_sizes, cfg, args.bucket_hr)
        missing_in_raw: list[str] = []
    else:
        pop_ids = set(rows)
        nsfw_dist: Counter[str] = Counter()
        n_val_eligible = 0
        # coverage: every population id must be present in the raw shards
        missing_in_raw = sorted(sid for sid in pop_ids if sid not in all_metas)
    n_full = len(pop_ids)

    # ---- 3. raw sizes: shipped webp header (fall back to meta dims) ------
    sizes: dict[str, tuple[int, int]] = {}
    n_size_fallback = 0
    for sid in pop_ids:
        if sid in all_sizes:
            sizes[sid] = all_sizes[sid]
        else:
            m = all_metas[sid]
            sizes[sid] = (m.width_meta, m.height_meta)
            n_size_fallback += 1

    # ---- 4. re-derive eligibility from raw labels; cross-check (frozen only)
    field_mismatch: Counter[str] = Counter()
    elig_mismatch = 0
    webp_dim_checked = 0
    webp_dim_mismatch = 0
    for sid, row in rows.items():
        m = all_metas[sid]
        if sid in all_sizes:
            webp_dim_checked += 1
            if all_sizes[sid] != (int(row["width"]), int(row["height"])):
                webp_dim_mismatch += 1
        for f_raw, f_par in (
            ("quality_tier", "quality"),
            ("completeness", "anime_completeness"),
            ("classification", "anime_classification"),
            ("nsfw", "nsfw"),
        ):
            if getattr(m, f_raw) != (row.get(f_par) or ""):
                field_mismatch[f_raw] += 1
        if m.ai_corrupted != bool(row.get("ai_corrupted")):
            field_mismatch["ai_corrupted"] += 1
        # re-derive eligibility (webp-corrected dims, as the production pipeline did)
        rec = MetaRecord(
            sample_id=m.sample_id,
            shard=m.shard,
            rel_path="",
            width=sizes[sid][0],
            height=sizes[sid][1],
            nsfw=m.nsfw,
            year=m.year,
            quality="",
            tags_general=m.tags_general,
            quality_tier=m.quality_tier,
            anime_completeness=m.completeness,
            anime_classification=m.classification,
            ai_corrupted=m.ai_corrupted,
        )
        rec = classify_quality(rec, cfg)
        el = evaluate_eligibility(rec, cfg)
        if el.eligible_train != bool(row.get("eligible_train")) or set(el.eligible_buckets) != set(
            row.get("eligible_buckets") or []
        ):
            elig_mismatch += 1

    # ---- 5. cross tables (full population, raw labels) -------------------
    def _table2(field_a: str, field_b: str) -> dict[str, dict[str, int]]:
        c: Counter[tuple[str, str]] = Counter()
        for sid in pop_ids:
            m = all_metas[sid]
            c[(getattr(m, field_a), getattr(m, field_b))] += 1
        out: dict[str, dict[str, int]] = {}
        for (a, b), n in c.items():
            out.setdefault(a or "unknown", {})[b or "unknown"] = n
        return out

    cross = {
        "completeness_x_quality": _table2("completeness", "quality_tier"),
        "classification_x_quality": _table2("classification", "quality_tier"),
        "completeness_x_classification": _table2("completeness", "classification"),
    }
    c3: Counter[tuple[str, str, str]] = Counter()
    for sid in pop_ids:
        m = all_metas[sid]
        c3[(m.completeness or "unknown", m.classification or "unknown", m.quality_tier or "unknown")] += 1
    top30_3d = [
        {"completeness": a, "classification": b, "quality": q, "n": n}
        for (a, b, q), n in c3.most_common(30)
    ]

    # ---- 6. pools ---------------------------------------------------------
    pool_ids: dict[str, list[str]] = {}
    for name, pred in POOLS:
        pool_ids[name] = sorted(sid for sid in pop_ids if pred(all_metas[sid]))
    p0 = set(pool_ids["P0_priority"])
    p1 = set(pool_ids["P1_sr_clean_v1"])
    p2 = set(pool_ids["P2_sr_clean_wide"])

    pool_sizes = []
    for name, _ in POOLS:
        n = len(pool_ids[name])
        pool_sizes.append(
            {
                "pool": name,
                "n": n,
                "share_of_full": (n / n_full) if n_full else 0.0,
                "in_p1_share": (n / len(p1)) if (n and p1) else None,
            }
        )

    # ---- 7. P1 detail ------------------------------------------------------
    def _dim_list(ids: list[str]) -> tuple[list[int], list[int]]:
        ws, hs = [], []
        for sid in ids:
            w, h = sizes[sid]
            ws.append(w)
            hs.append(h)
        return ws, hs

    p1_ids = pool_ids["P1_sr_clean_v1"]
    p1_ws, p1_hs = _dim_list(p1_ids)
    p1_detail = {
        "n_unique_images": len(p1_ids),
        "quality": dict(Counter(all_metas[s].quality_tier or "unknown" for s in p1_ids)),
        "classification": dict(Counter(all_metas[s].classification or "unknown" for s in p1_ids)),
        "completeness": dict(Counter(all_metas[s].completeness or "unknown" for s in p1_ids)),
        "width": _pctl_block(p1_ws),
        "height": _pctl_block(p1_hs),
        "min_dim": _pctl_block([min(w, h) for w, h in zip(p1_ws, p1_hs)]),
        "max_dim": _pctl_block([max(w, h) for w, h in zip(p1_ws, p1_hs)]),
    }
    p1_exact: list[int] = []
    p1_64: list[int] = []
    for sid in p1_ids:
        e, s64 = _crop_positions(sizes[sid][0], sizes[sid][1], args.bucket_hr)
        p1_exact.append(e)
        p1_64.append(s64)
    p1_detail["crop_flexibility_exact"] = _crop_bucket_hist(p1_exact)
    p1_detail["crop_flexibility_stride64"] = _crop_bucket_hist(p1_64)

    # ---- 8. 6M repetition intensity ---------------------------------------
    exposures = []
    crop_sums: dict[str, int] = {}
    for name, _ in POOLS:
        if name == "P3_lineart_extra" or name == "P4_rough_extra":
            continue
        ids = pool_ids[name]
        n = len(ids)
        pos64 = [
            _crop_positions(sizes[s][0], sizes[s][1], args.bucket_hr)[1] for s in ids
        ]
        crop_sums[name] = sum(pos64)
        exposures.append(
            {
                "pool": name,
                "n": n,
                "mean_exposures_per_source": (args.total_exposures / n) if n else None,
                "sum_crop_positions_stride64": sum(pos64),
                "mean_exposures_per_possible_crop_proxy": (args.total_exposures / sum(pos64)) if pos64 else None,
            }
        )
    exposures.append(
        {
            "pool": "full_eligible",
            "n": n_full,
            "mean_exposures_per_source": (args.total_exposures / n_full) if n_full else None,
            "sum_crop_positions_stride64": None,
            "mean_exposures_per_possible_crop_proxy": None,
        }
    )

    # ---- 9. clean-score correlation (NO gating) ----------------------------
    clean: dict[str, float] = {}
    clean_section: dict = {"status": "unavailable"}
    if args.clean_score and Path(args.clean_score).exists():
        with Path(args.clean_score).open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    doc = json.loads(line)
                    clean[str(doc["sample_id"])] = float(doc["score"])
        groups = {}
        for name in ("P0_priority", "P1_sr_clean_v1", "P2_sr_clean_wide", "P3_lineart_extra", "P4_rough_extra", "full_eligible"):
            ids = pool_ids.get(name, sorted(pop_ids))
            covered = [clean[s] for s in ids if s in clean]
            groups[name] = {
                "n": len(ids),
                "n_covered": len(covered),
                "coverage": (len(covered) / len(ids)) if ids else 0.0,
                **_pctl_block(covered),
            }
        pop_sorted = sorted(pop_ids)
        ordv = [float(quality_ordinal(all_metas[s].quality_tier)) for s in pop_sorted if s in clean]
        scv = [clean[s] for s in pop_sorted if s in clean]
        spearman = _spearman(ordv, scv)
        polished = [clean[s] for s in pop_sorted if all_metas[s].completeness == "polished" and s in clean]
        non_polished = [clean[s] for s in pop_sorted if all_metas[s].completeness != "polished" and s in clean]
        by_class: dict[str, dict] = {}
        for cl in sorted({all_metas[s].classification or "unknown" for s in pop_sorted}):
            vals = [clean[s] for s in pop_sorted if (all_metas[s].classification or "unknown") == cl and s in clean]
            by_class[cl] = {"n_covered": len(vals), **_pctl_block(vals)}
        clean_section = {
            "status": "ok",
            "note": "correlation analysis only; clean_score_min stays -1.0; no data removed",
            "groups": groups,
            "spearman_quality_ordinal_vs_score": spearman,
            "polished_vs_non_polished": {
                "polished": {"n": len(polished), **_pctl_block(polished)},
                "non_polished": {"n": len(non_polished), **_pctl_block(non_polished)},
            },
            "by_classification": by_class,
        }

    # ---- 10. nested-rule diffs ---------------------------------------------
    def _diff_stats(added: set[str]) -> dict:
        tiers = Counter(all_metas[s].quality_tier or "unknown" for s in added)
        cls = Counter(all_metas[s].classification or "unknown" for s in added)
        comp = Counter(all_metas[s].completeness or "unknown" for s in added)
        covered = [clean[s] for s in added if s in clean] if clean else []
        return {
            "n_added": len(added),
            "quality_tier": dict(tiers),
            "classification": dict(cls),
            "completeness": dict(comp),
            "clean_score": ({"n_covered": len(covered), **_pctl_block(covered)} if covered else None),
        }

    diffs = {
        "P1_minus_P0": _diff_stats(p1 - p0),
        "P2_minus_P1": _diff_stats(p2 - p1),
    }

    # ---- 11. deterministic samples ------------------------------------------
    sample_lists: dict[str, list[str]] = {
        "p0_priority_samples": _det_sample(pool_ids["P0_priority"], 100, args.seed),
        "p1_added_good_normal_samples": _det_sample(sorted(p1 - p0), 200, args.seed),
        "p2_low_worst_samples": _det_sample(sorted(p2 - p1), 200, args.seed),
        "p3_lineart_samples": _det_sample(pool_ids["P3_lineart_extra"], 200, args.seed),
        "p4_rough_samples": _det_sample(pool_ids["P4_rough_extra"], 100, args.seed),
    }

    # ---- 12. write outputs -----------------------------------------------------
    verdict = (
        Path(args.verdict_file).read_text(encoding="utf-8").strip()
        if args.verdict_file
        else "NOT SET (analyst verdict pending; data above)"
    )
    if full_repo:
        answers = {
            "mode": "full-repo (population re-derived from raw labels only; no frozen table)",
            "q1_full_1024_eligible_train": n_full,
            "q2_p0_strict_priority": len(p0),
            "q3_p1_sr_clean_v1": len(p1),
            "q4_p1_minus_p0_is_good_normal_polished": diffs["P1_minus_P0"],
            "q5_p2_minus_p1_low_worst": diffs["P2_minus_P1"],
            "q6_recommendation": verdict,
        }
    else:
        answers = {
            "q1_full_1024_eligible_train": n_full,
            "q2_p0_strict_priority": len(p0),
            "q3_p1_sr_clean_v1": len(p1),
            "q4_p1_minus_p0_is_good_normal_polished": diffs["P1_minus_P0"],
            "q5_p2_minus_p1_low_worst": diffs["P2_minus_P1"],
            "q6_recommendation": verdict,
        }
    full = {
        "params": {
            "mode": "full-repo" if full_repo else "frozen-population",
            "shard_dir": str(shard_dir),
            "index_dir": str(args.index_dir),
            "bucket_hr": args.bucket_hr,
            "config": args.config,
            "clean_score": args.clean_score,
            "seed": args.seed,
            "total_exposures": args.total_exposures,
            "workers": args.workers,
        },
        "shard_scan": scan_stats,
        "coverage": {
            "n_scanned_images": len(all_metas),
            "n_full_eligible_train": n_full,
            "n_full_eligible_validation": n_val_eligible,
            "nsfw_distribution": dict(nsfw_dist) if full_repo else None,
            "population_missing_in_raw_shards": len(missing_in_raw),
            "population_missing_ids_head": missing_in_raw[:50],
            "n_webp_size_fallback_to_meta": n_size_fallback,
        },
        "cross_check": (
            {
                "mode": "full-repo",
                "note": "no frozen table; eligibility re-derived from raw labels only",
            }
            if full_repo
            else {
                "label_field_mismatches_raw_vs_frozen": dict(field_mismatch),
                "eligibility_rederive_mismatches": elig_mismatch,
                "webp_dim_checked": webp_dim_checked,
                "webp_dim_mismatches_raw_tar_vs_frozen": webp_dim_mismatch,
            }
        ),
        "cross_tables": cross,
        "top30_3d": top30_3d,
        "pool_sizes": pool_sizes,
        "p1_detail": p1_detail,
        "exposures_6m": exposures,
        "clean_score": clean_section,
        "diffs": diffs,
        "answers": answers,
        "sample_counts": {k: len(v) for k, v in sample_lists.items()},
        "elapsed_s": round(time.monotonic() - t0, 2),
    }
    (out_dir / "full.json").write_text(json.dumps(full, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out_dir / "sr-clean-v1-summary.json").write_text(
        json.dumps({"answers": answers, "pool_sizes": pool_sizes, "params": full["params"]}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with (out_dir / "sr-clean-v1-sample-ids.txt").open("w", encoding="utf-8") as fh:
        fh.write("\n".join(sorted(p1)) + "\n")
    for name, ids in sample_lists.items():
        with (out_dir / f"{name}.txt").open("w", encoding="utf-8") as fh:
            for sid in ids:
                m = all_metas[sid]
                fh.write(f"{sid}\t{m.shard}\t{sid}.webp\n")
    # CSVs
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

    # Markdown summary
    md: list[str] = []
    md.append("# M4-1024 HR-GT quality pool statistics (SR-clean-v1)")
    md.append("")
    if full_repo:
        md.append("- mode: **full-repo** — population re-derived from raw labels only (no frozen table)")
        md.append(f"- scanned: **{len(all_metas)}** images across {len(scan_stats)} shards")
        md.append(f"- nsfw distribution (scanned): {dict(nsfw_dist)}")
        md.append(f"- population: **{n_full}** 1024-eligible train images (+{n_val_eligible} validation)")
        md.append(f"- webp-size fallbacks to meta dims: {n_size_fallback}")
    else:
        md.append(f"- population: **{n_full}** 1024-eligible train images (frozen eligibility table, raw v2 labels)")
        md.append(
            "- shards scanned: "
            + ", ".join(str(k) + "=" + str(v["n_images"]) for k, v in sorted(scan_stats.items()))
        )
        md.append(f"- coverage gaps (population ids missing in raw): {len(missing_in_raw)}; webp-size fallbacks: {n_size_fallback}")
        md.append(
            f"- cross-check vs frozen table: label mismatches {dict(field_mismatch) or '{}'}, eligibility re-derive mismatches {elig_mismatch}, "
            f"tar-webp dim mismatches {webp_dim_mismatch}/{webp_dim_checked}"
        )
    md.append("")
    md.append("## Pool sizes")
    md.append("")
    md.append("| pool | n | share of full |")
    md.append("| --- | ---: | ---: |")
    for d in pool_sizes:
        md.append(f"| {d['pool']} | {d['n']} | {d['share_of_full']:.1%} |")
    md.append("")
    md.append("## 6M repetition intensity")
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
    md.append("")
    if clean_section["status"] == "ok":
        md.append("## Clean-score (correlation only, no gating)")
        md.append("")
        md.append(f"- Spearman(quality ordinal, score) = {clean_section['spearman_quality_ordinal_vs_score']}")
        pp = clean_section["polished_vs_non_polished"]
        md.append(f"- polished mean {pp['polished'].get('mean')} (n={pp['polished']['n']}) vs non-polished {pp['non_polished'].get('mean')} (n={pp['non_polished']['n']})")
        md.append("")
    md.append("## Verdict")
    md.append("")
    md.append(answers["q6_recommendation"])
    (out_dir / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print(json.dumps({"out_dir": str(out_dir), "n_full": n_full, "pools": {d["pool"]: d["n"] for d in pool_sizes}, "elapsed_s": full["elapsed_s"]}))


if __name__ == "__main__":
    main()
