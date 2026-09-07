"""Vertical Bottom Supervision Audit - stage 1: build + freeze the pair manifest.

Reads ONLY:
  * /tmp/camera-coordinate-causal/stage1-manifest.json (historical, immutable)
  * the 7 local s0-validation-50k-v1 shards (source images + metadata json)
and writes ONLY:
  * /tmp/camera-vertical-bottom/pair-manifest.json           (frozen)
  * /tmp/camera-vertical-bottom/pair-manifest-frozen.json    (freeze receipt)
  * /tmp/camera-vertical-bottom/sources/<shard-rel>/<sample_id>.{bin,json}
  * /tmp/camera-vertical-bottom/build-pairs.log

Selection contract (spec s10-12): same-source mirrored vertical pairs from
stage1 units; dedup by (source_shard, sample_id) keeping the smallest unit id;
strata = anchor-side (START|END) x zoom band (mild|medium|strong); 512 target,
256 minimum else STOP; deterministic sample-key ordering (seed recorded for
auditability; ordering is seed-independent by construction).

Exit codes: 0 = frozen manifest written; 1 = STOP_BELOW_MINIMUM; 2 = fatal.
"""
from __future__ import annotations

import argparse
import json
import sys
import tarfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import contracts as C


def log(msg: str, log_path: Path | None = None) -> None:
    line = f"[build-pairs {time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if log_path is not None:
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")


def unit_bundle_path(unit_id: int) -> Path:
    return C.EVIDENCE_ROOT / "units" / f"unit-{int(unit_id):05d}.pt"


def _index_shard(tar_path: Path) -> dict[int, tuple[str, str]]:
    """Map sample_id -> (image member, json member) for one shard tar."""
    out: dict[int, tuple[str, str]] = {}
    with tarfile.open(tar_path, "r") as tf:
        members = tf.getmembers()
    img_by_id: dict[int, str] = {}
    json_by_id: dict[int, str] = {}
    for m in members:
        base = m.name.rsplit("/", 1)[-1]
        stem, dot, ext = base.rpartition(".")
        if not dot or not stem.isdigit():
            continue
        sid = int(stem)
        if ext in ("jpg", "jpeg", "png", "webp"):
            img_by_id[sid] = m.name
        elif ext == "json":
            json_by_id[sid] = m.name
    for sid in set(img_by_id) & set(json_by_id):
        out[sid] = (img_by_id[sid], json_by_id[sid])
    return out


def _extract_sample(
    tar_path: Path,
    index: dict[int, tuple[str, str]],
    sample_id: int,
    out_dir: Path,
    img_out: Path,
    json_out: Path,
) -> tuple[str, bool]:
    if img_out.is_file() and json_out.is_file():
        return C.sha256_file(img_out), True
    if sample_id not in index:
        raise RuntimeError(f"sample {sample_id} missing from {tar_path.name}")
    img_member, json_member = index[sample_id]
    with tarfile.open(tar_path, "r") as tf:
        img_bytes = tf.extractfile(img_member).read()
        json_bytes = tf.extractfile(json_member).read()
    out_dir.mkdir(parents=True, exist_ok=True)
    img_out.write_bytes(img_bytes)
    json_out.write_bytes(json_bytes)
    return C.sha256_file(img_out), False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True, help="worktree (provenance only)")
    ap.add_argument("--out-root", type=Path, default=C.AUDIT_ROOT)
    ap.add_argument("--target", type=int, default=C.N_PAIRS_TARGET)
    ap.add_argument("--minimum", type=int, default=C.N_PAIRS_MIN)
    args = ap.parse_args()

    log_path = args.out_root / "build-pairs.log"
    args.out_root.mkdir(parents=True, exist_ok=True)

    if not C.STAGE1_MANIFEST.is_file():
        log(f"FATAL: missing {C.STAGE1_MANIFEST}", log_path)
        return 2
    with open(C.STAGE1_MANIFEST, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest.get("status") != "OK":
        log("FATAL: stage1 manifest status != OK", log_path)
        return 2
    rows = manifest["units"]
    log(f"stage1 units: {len(rows)}", log_path)

    # ---- eligibility sweep (spec s10) ----
    eligible: list[dict] = []
    reasons: dict[str, int] = {}
    for u in rows:
        ok, reason = C.pair_eligible(u)
        if ok:
            eligible.append(u)
        else:
            key = reason.split(":", 1)[0]
            reasons[key] = reasons.get(key, 0) + 1
    log(f"eligible: {len(eligible)}  rejections: {reasons}", log_path)

    unique, dedup_removed = C.select_unique_pairs(eligible)
    log(f"unique sources: {len(unique)} (dedup removed {dedup_removed})", log_path)

    sel = C.stratified_selection(unique, target=args.target, minimum=args.minimum, seed=C.MASTER_SEED_VBS)
    if sel["status"] == "STOP_BELOW_MINIMUM":
        doc = {
            "label": "VERTICAL BOTTOM SUPERVISION PAIR MANIFEST",
            "status": "STOP_BELOW_MINIMUM",
            "n_eligible": len(eligible),
            "n_unique": len(unique),
            "minimum": args.minimum,
            "rejection_reasons": reasons,
            "seed": C.MASTER_SEED_VBS,
        }
        C.write_frozen(args.out_root / "pair-manifest.json", doc)
        log(f"STOP: pool {len(unique)} < minimum {args.minimum}", log_path)
        return 1
    log(f"selected: {sel['n_selected']}  allocation: {sel['allocation']}", log_path)

    # ---- extract sources + build pair rows (spec s13) ----
    src_dir = args.out_root / "sources"
    shard_index: dict[str, dict[int, tuple[str, str]]] = {}
    pairs_out: list[dict] = []
    t_vals = C.t_values()
    n_extracted = 0
    n_cached = 0
    for i, u in enumerate(sel["selected"]):
        g = C.pair_geometry(u)
        shard_rel = str(u["source_shard"])
        shard_path = C.SHARD_ROOT / shard_rel
        if not shard_path.is_file():
            log(f"FATAL: shard missing {shard_path}", log_path)
            return 2
        if shard_rel not in shard_index:
            shard_index[shard_rel] = _index_shard(shard_path)
            log(f"indexed {shard_rel}: {len(shard_index[shard_rel])} samples", log_path)
        rel_dir = Path(shard_rel).parent
        sdir = src_dir / rel_dir
        img_out = sdir / f"{int(u['sample_id'])}.bin"
        json_out = sdir / f"{int(u['sample_id'])}.json"
        sha, cached = _extract_sample(
            shard_path, shard_index[shard_rel], int(u["sample_id"]), sdir, img_out, json_out
        )
        if cached:
            n_cached += 1
        else:
            n_extracted += 1
        anchor_side = g["anchor_tertile"]
        mirror_side = "END" if anchor_side == "START" else "START"
        anchor_is_top = anchor_side == "START"  # TOP crop = k_start side
        zoom_b = C.zoom_band(g["zoom"])
        row = {
            # spec s13 fields
            "pair_id": i,
            "source_shard": shard_rel,
            "sample_id": int(u["sample_id"]),
            "original_unit": int(u["unit"]),
            "original_side": anchor_side,
            "target": [C.VIEWPORT, C.VIEWPORT],
            "full_canvas": {"width": C.VIEWPORT, "height": g["full_height"]},
            "equivalent_zoom": g["zoom"],
            "available": g["available"],
            "k_start": g["k_start"],
            "k_end": g["k_end"],
            "norm_start": g["norm_start"],
            "norm_end": g["norm_end"],
            "signed_shift_start": g["latent_shift_top"],
            "signed_shift_end": g["latent_shift_bottom"],
            "absolute_shift": abs(g["latent_shift_top"]),
            "latent_shift": abs(g["latent_shift_top"]) / C.VAE_SCALE,
            "crop_box_start": list(g["crop_box_top"]),
            "crop_box_end": list(g["crop_box_bottom"]),
            "source_image_sha256": sha,
            "qwen_bundle_sha256": str(u["file_sha256"]),
            "selection_stratum": f"{anchor_side}|{zoom_b}",
            # machine-use auxiliaries (not part of the s13 minimal set)
            "pair_index": i,
            "source_image_rel": str(img_out.relative_to(args.out_root)),
            "anchor_unit": {
                "unit": int(u["unit"]),
                "unit_file": str(unit_bundle_path(int(u["unit"]))),
                "unit_file_sha256": str(u["file_sha256"]),
                "side": anchor_side,
                "k": g["k_anchor"],
                "norm_offset": g["norm_offset"],
                "signed_shift": g["signed_shift_anchor"],
                "zoom": g["zoom"],
                "zoom_band": zoom_b,
                "latent_shift": abs(g["signed_shift_anchor"]) / C.VAE_SCALE,
            },
            "mirror_unit": {
                "side": mirror_side,
                "k": g["k_mirror"],
                "norm_offset": g["k_mirror"] / g["available"],
                "signed_shift": g["signed_shift_mirror"],
                "latent_shift": abs(g["signed_shift_mirror"]) / C.VAE_SCALE,
            },
            "geometry": {
                "viewport": C.VIEWPORT,
                "full_height": g["full_height"],
                "available": g["available"],
                "k_start": g["k_start"],
                "k_end": g["k_end"],
                "crop_box_top": list(g["crop_box_top"]),
                "crop_box_bottom": list(g["crop_box_bottom"]),
                "signed_shift_top": g["latent_shift_top"],
                "signed_shift_bottom": g["latent_shift_bottom"],
                "abs_shift_equal": abs(g["latent_shift_top"]) == abs(g["latent_shift_bottom"]),
                "anchor_is_top": anchor_is_top,
            },
            "t_values": list(t_vals),
        }
        if not row["geometry"]["abs_shift_equal"]:
            log(f"FATAL: pair {i} |shift| not equal", log_path)
            return 2
        pairs_out.append(row)
    log(f"sources: extracted {n_extracted}, cached {n_cached}", log_path)

    # ---- freeze (spec s13: frozen pair manifest + sha) ----
    doc = {
        "label": "VERTICAL BOTTOM SUPERVISION PAIR MANIFEST",
        "status": "OK",
        "spec": "same-source mirrored vertical viewport pairs (VBS audit)",
        "master_seed": C.MASTER_SEED_VBS,
        "base_sha": C.BASE_SHA,
        "branch": C.BRANCH,
        "worktree": str(C.WORKTREE),
        "stage1_manifest": str(C.STAGE1_MANIFEST),
        "repo_head": str(args.repo),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "target": args.target,
        "minimum": args.minimum,
        "n_eligible": len(eligible),
        "n_unique": len(unique),
        "dedup_removed": dedup_removed,
        "rejection_reasons": reasons,
        "n_selected": len(pairs_out),
        "selection_rule": (
            "dedup (source_shard,sample_id) keep smallest unit id; "
            "strata=(anchor START|END)xzoom(mild|medium|strong); proportional "
            "largest-remainder allocation; sample-key order "
            "(source_shard, sample_id, unit); seed recorded, ordering "
            "seed-independent"
        ),
        "stratum_pool_sizes": sel["stratum_pool_sizes"],
        "allocation": sel["allocation"],
        "t_values": list(t_vals),
        "pairs": pairs_out,
    }
    sha = C.write_frozen(args.out_root / "pair-manifest.json", doc)
    freeze = {
        "manifest_path": str(args.out_root / "pair-manifest.json"),
        "sha256": sha,
        "frozen_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_pairs": len(pairs_out),
    }
    C.write_frozen(args.out_root / "pair-manifest-frozen.json", freeze)
    log(f"FROZEN pair-manifest.json n={len(pairs_out)} sha256={sha[:16]}...", log_path)
    with open(args.out_root / "pair-manifest.json", "r", encoding="utf-8") as fh:
        json.load(fh)
    log("PAIR MANIFEST OK", log_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
