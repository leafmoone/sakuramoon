"""Offline / incremental clean-score precompute (P1-4, 2026-08-29).

The clean score is a FROZEN sidecar that training only READS (the legacy
lazy path raced DDP ranks / producer workers on O_APPEND to the same JSONL
behind a process-local lock — not cross-process synchronization). This CLI
is the SINGLE WRITER: it walks the eligible TRAIN index once, decodes each
webp (full image, no crop — the score is a property of the HR image, not a
bucket), computes the clean score, and appends one JSONL line per sample.

Incremental + resumable: sample_ids already present in the sidecar are
skipped, so a killed/interrupted run continues where it stopped. The
sidecar is re-readable at any time (last line per id wins on re-runs).

On completion it prints the distribution report (percentiles, per-pool /
quality / completeness / classification stats, candidate-threshold
keep/exclude counts) and writes it to ``<index-dir>/clean-score-report.json``
— the numbers the user reads before deciding a ``clean_score_min`` gate.

Usage (server):

    python -m anime_sr.cli.clean_score_precompute \
        --config config/base.toml config/data.toml \
        --index-dir <index> --webp-dir <webp> \
        [--limit 0] [--every 500]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from anime_sr.config.loader import load_config
from anime_sr.data import index as index_mod
from anime_sr.data.clean_score import (
    CLEAN_SCORE_CACHE_NAME,
    build_clean_score_report,
    compute_clean_score,
)
from anime_sr.data.pipeline import find_eligibility


def _decode_full_hr(index_dir_path, meta_row: dict, webp_dir) -> torch.Tensor:
    """webp -> [3, H, W] fp32 in [-1, 1] (full image; same decode contract
    as SRDataset.decode_hr)."""
    img_id = str(meta_row["rel_path"]).rsplit("/", 1)[-1]
    p = webp_dir / str(meta_row["shard"]) / img_id
    if not p.is_file():
        raise FileNotFoundError(f"webp missing (extract shards first): {p}")
    img = Image.open(p).convert("RGB")
    if (img.width, img.height) != (int(meta_row["width"]), int(meta_row["height"])):
        raise RuntimeError(f"size mismatch for {p}: {img.size} vs row dims")
    arr = np.asarray(img, dtype=np.uint8).copy()
    t = torch.from_numpy(arr).permute(2, 0, 1).float()
    return t * (2.0 / 255.0) - 1.0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Offline/incremental clean-score precompute (single writer)")
    ap.add_argument("--config", nargs="+", required=True, help="TOML config files")
    ap.add_argument("--index-dir", required=True)
    ap.add_argument("--webp-dir", required=True)
    ap.add_argument("--limit", type=int, default=0, help="score at most N samples (0 = all)")
    ap.add_argument("--every", type=int, default=500, help="progress log interval")
    args = ap.parse_args(argv)

    cfg = load_config(*args.config)
    index_dir = args.index_dir
    webp_dir = args.webp_dir
    sidecar_path = _sidecar_path(index_dir)  # frozen sidecar under the index dir

    rows = []
    for row in index_mod.iter_index(find_eligibility(index_dir)):
        if row.get("is_validation") or not row.get("eligible_train"):
            continue
        rows.append(row)
    total = len(rows)
    if args.limit > 0:
        rows = rows[: args.limit]

    # incremental: load existing scores, skip them (last line already wins
    # inside the loader; we only append NEW sample_ids here)
    existing: dict[str, float] = {}
    if sidecar_path.is_file():
        with open(sidecar_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    existing[str(rec["sample_id"])] = float(rec["score"])
    todo = [r for r in rows if str(r["sample_id"]) not in existing]
    print(
        f"[clean-score] {total} eligible train samples, {len(existing)} already "
        f"scored, {len(todo)} to do",
        flush=True,
    )

    done = 0
    t0 = time.perf_counter()
    with open(sidecar_path, "a", encoding="utf-8") as f:
        for row in todo:
            hr = _decode_full_hr(index_dir, row, webp_dir)
            score = round(compute_clean_score(hr), 6)
            f.write(json.dumps({"sample_id": str(row["sample_id"]), "score": score}) + "\n")
            done += 1
            if args.every > 0 and done % args.every == 0:
                f.flush()
                rate = done / max(time.perf_counter() - t0, 1e-9)
                eta = (len(todo) - done) / max(rate, 1e-9)
                print(
                    f"[clean-score] {done}/{len(todo)} "
                    f"({rate:.1f}/s, eta {eta / 60:.1f} min)",
                    flush=True,
                )
        f.flush()

    print(f"[clean-score] done: {done} new scores @ {sidecar_path}", flush=True)

    # distribution report (rank-free, offline): the numbers the user reads
    # before picking a clean_score_min threshold
    all_ids = [str(r["sample_id"]) for r in rows]
    report = build_clean_score_report(
        index_dir, all_ids, cfg.filter.clean_score_candidates
    )
    p = report["percentiles"]
    print(
        f"[clean-score] coverage {report['n_covered']}/{report['n_requested']} "
        f"({report['coverage']:.1%}); p10={p['p10']:.3f} p25={p['p25']:.3f} "
        f"p50={p['p50']:.3f} p75={p['p75']:.3f} p90={p['p90']:.3f} "
        f"mean={p['mean']:.3f}",
        flush=True,
    )
    for t, v in report["candidate_thresholds"].items():
        print(
            f"[clean-score] candidate min={t}: keep {v['kept']} / exclude {v['excluded']}",
            flush=True,
        )
    report_path = _report_path(index_dir)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[clean-score] report -> {report_path}", flush=True)
    return 0


def _sidecar_path(index_dir: str) -> Path:
    return Path(index_dir) / CLEAN_SCORE_CACHE_NAME


def _report_path(index_dir: str) -> Path:
    return Path(index_dir) / "clean-score-report.json"


if __name__ == "__main__":
    raise SystemExit(main())
