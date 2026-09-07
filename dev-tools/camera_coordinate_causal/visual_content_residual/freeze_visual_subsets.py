"""PHASE A: freeze the visual-content subsets for the 512 mirror pairs.

This script reads ONLY:
  * the frozen pair manifest (pair-manifest.json, sha-pinned),
  * the frozen CLIP feature audit (features/clip-audit.json),
  * the frozen PE-Spatial feature audit (features/pe-audit.json).

It computes the rank-normalized composite visual asymmetry

    A_visual = 0.5 * ( R_img + R_pe )
      R_img  = percentile rank (average ties) of |CLIP image-full delta|
      R_pe   = percentile rank (average ties) of |PE retained delta|

and freezes the geometry-stratified plus sensitivity subsets. The frozen
manifest + its SHA are the only bridge to PHASE B (analyze.py). No per-pair
mirror statistic, no checkpoint loss, no report of the prior generation is
read here; subset selection is therefore pre-registered with respect to the
downstream statistics.

Outputs (under --out):
  visual-subsets.json          canonical JSON (sorted keys, compact)
  visual-subsets-frozen.json   sha256 of the above + subset sizes
  logs/freeze.log
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import contracts as C


def log(out_dir: Path, msg: str) -> None:
    line = f"[visual-freeze {time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    with (out_dir / "logs" / "freeze.log").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_pair_indices(doc: dict, field: str, expected: int, label: str) -> dict[int, dict]:
    """Check a feature audit row list covers exactly 0..expected-1, once each."""
    out: dict[int, dict] = {}
    rows = doc.get(field)
    if not isinstance(rows, list):
        raise TypeError(f"{label}: missing row list {field!r}")
    for row in rows:
        i = int(row["pair_index"])
        if i in out:
            raise ValueError(f"{label}: duplicate pair_index {i}")
        out[i] = row
    if sorted(out.keys()) != list(range(expected)):
        missing = set(range(expected)) - set(out.keys())
        raise ValueError(f"{label}: pair_index set != 0..{expected - 1}; missing {len(missing)}")
    return out


def finite(rows: dict[int, dict], keys: dict[str, tuple[str, ...]], label: str) -> None:
    for i, row in rows.items():
        for name, path in keys.items():
            v = row
            for p in path:
                v = v[p]
            fv = float(v)
            if not np.isfinite(fv):
                raise ValueError(f"{label}: nonfinite {name} at pair_index {i}")


def build_per_pair(manifest: dict, clip: dict, pe: dict, n: int) -> dict:
    img_rows = load_pair_indices(clip, "image_rows", n, "clip image")
    text_status = str(clip.get("text_status"))
    if text_status not in ("AVAILABLE", "NOT_AVAILABLE"):
        raise ValueError(f"clip text_status {text_status!r} not recognized")
    text_rows = load_pair_indices(clip, "text_rows", n, "clip text") if text_status == "AVAILABLE" else {}
    pe_rows = load_pair_indices(pe, "rows", n, "pe")

    finite(img_rows, {"sim_full_top": ("sim_full_top",), "sim_full_bot": ("sim_full_bot",)}, "clip image")
    if text_rows:
        finite(text_rows, {"sim_text_top": ("sim_text_top",), "sim_text_bot": ("sim_text_bot",)}, "clip text")
    finite(pe_rows, {"ret_top": ("top", "retained"), "ret_bot": ("bottom", "retained")}, "pe")

    c_img = np.empty(n, dtype=np.float64)
    c_pe = np.empty(n, dtype=np.float64)
    c_text = np.empty(n, dtype=np.float64)
    for i in range(n):
        c_img[i] = float(img_rows[i]["sim_full_top"]) - float(img_rows[i]["sim_full_bot"])
        c_pe[i] = float(pe_rows[i]["top"]["retained"]) - float(pe_rows[i]["bottom"]["retained"])
        if text_rows:
            c_text[i] = float(text_rows[i]["sim_text_top"]) - float(text_rows[i]["sim_text_bot"])

    a_img = np.abs(c_img)
    a_pe = np.abs(c_pe)
    r_img = C.percentile_rank_average_ties(a_img)
    r_pe = C.percentile_rank_average_ties(a_pe)
    a_visual = C.composite_visual_score(r_img, r_pe)

    pairs = manifest["pairs"]
    if len(pairs) != n or [int(p["pair_index"]) for p in pairs] != list(range(n)):
        raise ValueError("manifest pairs must be exactly 0..n-1 in order")
    strata = [C.geometry_stratum(p) for p in pairs]

    per_pair: dict[int, dict] = {}
    for i in range(n):
        rec = {
            "C_img_signed": float(c_img[i]),
            "C_pe_signed": float(c_pe[i]),
            "A_img": float(a_img[i]),
            "A_pe": float(a_pe[i]),
            "R_img": float(r_img[i]),
            "R_pe": float(r_pe[i]),
            "A_visual": float(a_visual[i]),
            "geometry_stratum": strata[i],
        }
        if text_rows:
            rec["C_text_signed"] = float(c_text[i])
        per_pair[str(i)] = rec
    return {
        "per_pair": per_pair,
        "c_img": c_img,
        "c_pe": c_pe,
        "a_img": a_img,
        "a_pe": a_pe,
        "r_img": r_img,
        "r_pe": r_pe,
        "a_visual": a_visual,
        "strata": strata,
        "text_status": text_status,
    }


def build_subsets(pp: dict, n: int) -> dict[str, list[int]]:
    a_visual = pp["a_visual"]
    strata = pp["strata"]
    subsets: dict[str, list[int]] = {
        "ALL": list(range(n)),
        "VISUAL_GEO_BALANCED_50": C.geometry_stratified_select(a_visual, strata, C.V50_TARGET),
        "VISUAL_GEO_BALANCED_25": C.geometry_stratified_select(a_visual, strata, C.V25_TARGET),
        "GLOBAL_VISUAL_50": C.global_lowest_select(a_visual, C.V50_TARGET),
        "GLOBAL_VISUAL_25": C.global_lowest_select(a_visual, C.V25_TARGET),
        "CLIP_IMAGE_ONLY_GEO_50": C.geometry_stratified_select(pp["r_img"], strata, C.V50_TARGET),
        "CLIP_IMAGE_ONLY_GEO_25": C.geometry_stratified_select(pp["r_img"], strata, C.V25_TARGET),
        "PE_ONLY_GEO_50": C.geometry_stratified_select(pp["r_pe"], strata, C.V50_TARGET),
        "PE_ONLY_GEO_25": C.geometry_stratified_select(pp["r_pe"], strata, C.V25_TARGET),
    }
    subsets["JOINT_LOW_VISUAL_50_INTERSECTION"] = sorted(
        C.joint_lowest_intersection(pp["a_img"], pp["a_pe"], C.V50_TARGET)
    )
    return subsets


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--clip", type=Path, required=True)
    ap.add_argument("--pe", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    log(args.out, f"inputs manifest={args.manifest} clip={args.clip} pe={args.pe}")

    # -- manifest sha gate (frozen identity of the pair set) --
    man_bytes = args.manifest.read_bytes()
    man_sha = C.sha256_of_bytes(man_bytes)
    if man_sha != C.PAIR_MANIFEST_SHA:
        log(args.out, f"STOP: manifest sha mismatch ({man_sha[:12]}... != pinned)")
        return 2
    manifest = json.loads(man_bytes)
    n = len(manifest["pairs"])
    if n != C.N_PAIRS:
        log(args.out, f"STOP: n_pairs {n} != {C.N_PAIRS}")
        return 2
    log(args.out, f"manifest verified (sha match, n={n})")

    clip_bytes = args.clip.read_bytes()
    pe_bytes = args.pe.read_bytes()
    clip = json.loads(clip_bytes)
    pe = json.loads(pe_bytes)

    pp = build_per_pair(manifest, clip, pe, n)
    subsets = build_subsets(pp, n)

    cells: dict[str, int] = {}
    for s in pp["strata"]:
        cells[s] = cells.get(s, 0) + 1

    doc = {
        "schema_version": C.SCHEMA_VERSION,
        "base_sha": C.BASE_SHA,
        "pair_manifest_sha": man_sha,
        "clip_feature_sha": C.sha256_of_bytes(clip_bytes),
        "pe_feature_sha": C.sha256_of_bytes(pe_bytes),
        "n_pairs": n,
        "score_definition": "A_visual = 0.5*(R_img + R_pe); R_* = percentile rank (average ties) of the absolute single-metric delta; CLIP text excluded from the primary score",
        "rank_definition": "percentile rank in [0,1], lowest asymmetry -> 0, highest -> 1, average ranks for ties, deterministic",
        "geometry_strata_definition": "cell = original_side x zoom_band(equivalent_zoom) x latent_shift_band(latent_shift); zoom mild<1.20<=medium<1.35<=strong; latent <2, [2,4), >=4 (frozen VBS definitions)",
        "selection_rule": "geometry-stratified proportional allocation + largest remainder (ties by cell name); within cell ascending (score, pair_index); global/joint subsets use the same tie break",
        "clip_text_status": pp["text_status"],
        "geometry_cells": cells,
        "subsets": subsets,
        "subset_sizes": {k: len(v) for k, v in subsets.items()},
        "per_pair": pp["per_pair"],
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    problems = C.verify_subset_doc(doc)
    if problems:
        log(args.out, f"STOP: subset doc structural problems: {problems}")
        return 2

    body = C.canon_json(doc)
    out_file = args.out / "visual-subsets.json"
    out_file.write_text(body, encoding="utf-8")
    sha = C.sha256_of_bytes(body.encode("utf-8"))
    frozen = {
        "schema_version": C.SCHEMA_VERSION,
        "frozen_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sha256_visual_subsets": sha,
        "subset_sizes": doc["subset_sizes"],
        "n_pairs": n,
    }
    (args.out / "visual-subsets-frozen.json").write_text(C.canon_json(frozen), encoding="utf-8")
    log(args.out, f"frozen sha={sha[:12]}... sizes={doc['subset_sizes']}")
    log(args.out, "FREEZE DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
