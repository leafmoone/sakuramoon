"""Camera coordinate causal audit - offset balance review (posthoc V2).

Second main task of the posthoc V2 round (sections 26-43):
  Explain - or at least decompose - WHY the universal END / vertical physical
  BOTTOM cohort shows a NEGATIVE adjusted OPPOSITE POST-PRE effect
  (V1: -4.93e-4 [CI < 0] universal END; -6.34e-4 [CI < 0] vertical END) while
  the aggregate primary arms are positive.

  The audit distinguishes, in order of specificity:
    A. PLANNER_COUNT_IMBALANCE          - observed exposure deviates from the
       exact discrete production-sampler expectation (z-scored);
    B. GEOMETRY_DISTRIBUTION_IMBALANCE  - TOP vs BOTTOM geometry differs and
       the BOTTOM negative collapses after geometry balance;
    C. EFFECT_PERSISTS_AFTER_GEOMETRY_BALANCE;
    D. CONTENT_OR_SHARD_INTERACTION_SUSPECTED;
    E. INCONCLUSIVE.

  Sources (all read-only; nothing is re-generated):
    - stage1-manifest.json  : per-unit orientation, norm_offset, zoom,
      latent_shift, pixel_shift, target, source_shard, opp_na, sample_id;
    - results-worker{0,1}.pt: per-unit per-ckpt per-stratum arm losses;
      D_opp_unit = (M_OPP_POST - M_OPP_PRE) - (M_SAME_POST - M_SAME_PRE)
      with M = 4-stratum mean margin (section 36); NO new forwards;
    - units/unit-*.pt       : routing covariates (main_token_lengths,
      condition_mask, use_null_condition) - optional, skipped with --no-bundles;
    - C2 distribution json  : secondary replication context (read-only).

  Per-unit planner reconstruction (verified exact across all 2048 camera
  units against pixel_shift before use):
    r = 256 (target square), F = round(zoom^2 * r) (long canvas edge),
    available = F - r, k = round(norm_offset * available) in [0, available],
    signed_shift = k + r/2 - F/2 must equal the recorded pixel_shift.

Read-only except for the three report files under <repo>/reports/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import posthoc_v2_contracts as v2c

BOOT_SEED = 20260907
N_BOOT = 10000
R_EDGE = 256  # square viewport bucket (target [256, 256] for every unit)

ZOOM_BANDS = (("mild", 1.20), ("medium", 1.35))  # mild <1.20, medium <1.35, strong >=1.35
LATENT_BANDS = (("lt2", 2.0), ("2to4", 4.0))     # <2, 2-4, >=4

C2_DIST_DEFAULT = Path("/sakuramoon-runtime/sakuramoon-camera-v2-c2/reports/camera-viewport-v2-distribution.json")


def log(msg: str) -> None:
    print(msg, flush=True)


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _ci95(samples: np.ndarray) -> list[float]:
    return [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))]


def _cluster_ci(vals: np.ndarray, rng: np.random.Generator) -> list[float]:
    v = np.asarray(vals, dtype=np.float64)
    if v.size < 2:
        return [float(v[0])] * 2 if v.size else [float("nan"), float("nan")]
    idx = rng.integers(0, v.size, size=(N_BOOT, v.size))
    return _ci95(v[idx].mean(axis=1))


def _chi2_pvalue(x: float, df: int) -> float:
    """Right-tail p of chi-square via the Wilson-Hilferty normal
    approximation (diagnostic quality, documented)."""
    if x <= 0:
        return 1.0
    z = ((x / df) ** (1.0 / 3.0) - (1.0 - 2.0 / (9.0 * df))) / math.sqrt(2.0 / (9.0 * df))
    return float(0.5 * math.erfc(z / math.sqrt(2.0)))


def zoom_band(z: float) -> str:
    for name, hi in ZOOM_BANDS:
        if z < hi:
            return name
    return "strong"


def latent_band(ls: float) -> str:
    for name, hi in LATENT_BANDS:
        if ls < hi:
            return name
    return "ge4"


def tertiles(values: np.ndarray) -> tuple[float, float]:
    """Pooled tertile cut points of a reference distribution."""
    v = np.asarray(values, dtype=np.float64)
    return (float(np.percentile(v, 33.3333333)), float(np.percentile(v, 66.6666667)))


def band_of(x: float, cuts: tuple[float, float]) -> int:
    if x < cuts[0]:
        return 0
    if x < cuts[1]:
        return 1
    return 2


def reconstruct_planner(row: dict) -> dict:
    """Exact per-unit planner reconstruction from manifest fields.

    Returns F (long edge), available, k (integer offset), ok flags.
    """
    r = int(row["target"][0])
    if tuple(row["target"]) != (r, r):
        return {"ok": False, "reason": f"target not square: {row['target']}"}
    z = float(row["zoom"])
    F = round(z * z * r)
    if F <= r:
        return {"ok": False, "reason": f"F={F} <= r"}
    available = F - r
    q = float(row["norm_offset"])
    kf = q * available
    k = round(kf)
    if abs(kf - k) > 1e-6:
        return {"ok": False, "reason": f"norm_offset*available not integral: {kf!r}"}
    if not (0 <= k <= available):
        return {"ok": False, "reason": f"k={k} outside [0,{available}]"}
    s = (k + r // 2) - F / 2.0
    ps = float(row["pixel_shift"])
    shift_ok = abs(s - ps) <= 1e-6
    return {
        "ok": shift_ok,
        "reason": "" if shift_ok else f"signed_shift {s} != pixel_shift {ps}",
        "F": F,
        "available": available,
        "k": k,
        "retention": float(r * r) / float(F * r),
        "canvas_aspect": float(F) / r,
    }


def load_deltas(evidence: Path, units: dict[int, dict], camera_units: list[int]) -> dict:
    """Per-unit 4-stratum-mean margins M_OPP / M_SAME per ck and the
    adjusted per-unit delta D_opp_unit (section 36)."""
    done: dict[int, dict] = {}
    for w in (0, 1):
        obj = torch.load(evidence / f"results-worker{w}.pt", map_location="cpu", weights_only=False)
        for u, rows in obj.items():
            done.setdefault(int(u), {}).update(rows)

    def margin(u: int, ck: str, arm: str) -> float:
        row = done.get(u, {}).get(ck, {})
        loss = row.get("loss", {})
        if arm not in loss or "CORRECT" not in loss:
            return float("nan")
        a = np.array(loss[arm], dtype=np.float64)
        c = np.array(loss["CORRECT"], dtype=np.float64)
        if a.shape != c.shape or a.shape[0] != 4:
            return float("nan")
        if not (np.isfinite(a).all() and np.isfinite(c).all()):
            return float("nan")
        return float(np.mean(a - c))

    out: dict[int, dict] = {}
    for u in camera_units:
        m_opp = {ck: margin(u, ck, "OPPOSITE") for ck in v2c.CKS}
        m_same = {ck: margin(u, ck, "SAME") for ck in v2c.CKS}
        need = [m_opp["PRE"], m_opp["POST"], m_same["PRE"], m_same["POST"]]
        valid = all(np.isfinite(x) for x in need) and not bool(units[u].get("opp_na", False))
        d_opp = (m_opp["POST"] - m_opp["PRE"]) - (m_same["POST"] - m_same["PRE"]) if valid else float("nan")
        out[u] = {"m_opp": m_opp, "m_same": m_same, "valid": valid, "d_opp": d_opp}
    return out


def load_routing(evidence: Path, unit_ids: list[int], use_bundles: bool) -> dict[int, dict]:
    """Per-unit routing covariates from the frozen bundles (read-only).
    Prefers mmap tensor access; falls back to full load; if --no-bundles or
    bundles are unreadable, returns {} and the caller marks the covariates
    NOT_AVAILABLE (sections 37-38: never fail the audit on this)."""
    out: dict[int, dict] = {}
    if not use_bundles:
        return out
    for u in unit_ids:
        p = evidence / "units" / f"unit-{u:05d}.pt"
        if not p.is_file():
            continue
        try:
            try:
                b = torch.load(p, map_location="cpu", weights_only=False, mmap=True)
            except (OSError, ValueError, RuntimeError, TypeError):
                b = torch.load(p, map_location="cpu", weights_only=False)
            ml = b.get("main_token_lengths")
            mm = b.get("main_mask")
            cm = b.get("condition_mask")
            nuc = b.get("use_null_condition")
            entry: dict = {}
            if mm is not None and ml is not None:
                entry["main_tokens"] = int(mm.sum().item())
            elif ml is not None:
                entry["main_tokens"] = int(np.asarray(ml, dtype=np.float64).sum())
            if cm is not None:
                entry["condition_tokens"] = int(cm.sum().item())
            if nuc is not None:
                entry["null_condition"] = bool(nuc.flatten()[0].item())
            if entry:
                out[int(u)] = entry
        except Exception as exc:  # noqa: BLE001 - read-only best effort
            log(f"routing bundle unreadable for unit {u}: {exc}")
    return out


def group_stats(vals: np.ndarray) -> dict:
    v = np.asarray(vals, dtype=np.float64)
    if v.size == 0:
        return {"n": 0, "mean": None, "p50": None, "p90": None}
    return {
        "n": int(v.size),
        "mean": float(v.mean()),
        "p50": float(np.percentile(v, 50.0)),
        "p90": float(np.percentile(v, 90.0)),
    }


def shard_table(unit_rows: list[dict], key: str) -> dict:
    from collections import Counter
    c = Counter(row[key] for row in unit_rows)
    total = sum(c.values()) or 1
    return {"n": total, "top10": dict(c.most_common(10)), "n_shards": len(c)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True, help="worktree containing reports/")
    ap.add_argument("--evidence", type=Path, default=Path("/tmp/camera-coordinate-causal"))
    ap.add_argument("--c2-dist", type=Path, default=C2_DIST_DEFAULT)
    ap.add_argument("--no-bundles", action="store_true", help="skip per-unit bundle routing covariates")
    ap.add_argument("--skip-write", action="store_true", help="analyze only; do not write reports")
    args = ap.parse_args()
    t0 = time.time()

    ev = args.evidence
    with open(ev / "stage1-manifest.json", "r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest.get("status") != "OK":
        raise SystemExit("stage1 manifest not OK")
    units: dict[int, dict] = {}
    for row in manifest["units"]:
        row = dict(row)
        row["opp_na"] = bool(row.get("opp_na", False))
        units[int(row["unit"])] = row
    camera_units = [u for u in sorted(units) if units[u]["cohort"] == "camera"]
    n_cam = len(camera_units)
    log(f"camera units: {n_cam}")

    # ---------------- planner reconstruction + validation ----------------
    recon: dict[int, dict] = {}
    n_bad = 0
    for u in camera_units:
        r = reconstruct_planner(units[u])
        recon[u] = r
        if not r["ok"]:
            n_bad += 1
            if n_bad <= 5:
                log(f"reconstruction FAILED unit {u}: {r['reason']}")
    if n_bad:
        raise SystemExit(f"planner reconstruction failed for {n_bad} units - STOP (no fabrication allowed)")
    log("planner reconstruction verified exact for all units (pixel_shift cross-check)")

    # ---------------- per-unit adjusted delta ----------------
    deltas = load_deltas(ev, units, camera_units)
    valid_units = [u for u in camera_units if deltas[u]["valid"]]
    log(f"valid D_opp units (finite OPP+SAME PRE/POST, not opp_na): {len(valid_units)} / {n_cam}")

    # ---------------- routing covariates ----------------
    routing = load_routing(ev, camera_units, use_bundles=not args.no_bundles)
    routing_available = len(routing) > 0
    if not routing_available:
        log("routing covariates NOT_AVAILABLE (bundles skipped or unreadable)")

    # ---------------- planner discrete symmetry (28-30) ----------------
    def tertile_counts(idxs: list[int]) -> dict:
        obs = {t: 0 for t in v2c.TERTS}
        for u in idxs:
            obs[v2c.tertile_of(float(units[u]["norm_offset"]))] += 1
        return obs

    plan_blocks = {}
    for scope, idxs in (
        ("all", camera_units),
        ("horizontal", [u for u in camera_units if units[u]["orientation"] == "horizontal"]),
        ("vertical", [u for u in camera_units if units[u]["orientation"] == "vertical"]),
    ):
        avail = [recon[u]["available"] for u in idxs]
        offs = [recon[u]["k"] for u in idxs]
        exp = v2c.expected_tertile_counts(avail)
        obs = tertile_counts(idxs)
        z = v2c.observed_tertile_z(avail, obs)
        plan_blocks[scope] = {
            "n_units": len(idxs),
            "observed": obs,
            "expected": exp["expected"],
            "sd": exp["sd"],
            "z": z["z"],
            "mirror": v2c.mirror_symmetry(avail, offs),
        }
    exposure_imbalance = any(
        abs(v2c.observed_tertile_z([recon[u]["available"] for u in scope_units], tertile_counts(scope_units))["z"][t]) >= v2c.Z_IMBALANCE
        for scope_units in (camera_units, [u for u in camera_units if units[u]["orientation"] == "vertical"])
        for t in v2c.TERTS
    )
    log(f"exposure count imbalance (|z| >= {v2c.Z_IMBALANCE}): {exposure_imbalance}")

    # ---------------- TOP vs BOTTOM geometry (31-32) ----------------
    v_units = [u for u in camera_units if units[u]["orientation"] == "vertical"]
    top_units = [u for u in v_units if v2c.tertile_of(float(units[u]["norm_offset"])) == "START"]
    bottom_units = [u for u in v_units if v2c.tertile_of(float(units[u]["norm_offset"])) == "END"]

    def feats(idxs: list[int]) -> dict[str, np.ndarray]:
        f: dict[str, np.ndarray] = {}
        f["zoom"] = np.asarray([recon[u]["canvas_aspect"] for u in idxs])  # F/256 = zoom^2, canvas aspect
        f["zoom_val"] = np.asarray([float(units[u]["zoom"]) for u in idxs])
        f["latent_shift"] = np.asarray([float(units[u]["latent_shift"]) for u in idxs])
        f["pixel_shift"] = np.asarray([float(units[u]["pixel_shift"]) for u in idxs])
        f["abs_displacement"] = np.asarray([abs(float(units[u]["pixel_shift"])) for u in idxs])
        f["available"] = np.asarray([recon[u]["available"] for u in idxs])
        f["canvas_aspect"] = f["zoom"].copy()
        f["retention"] = np.asarray([recon[u]["retention"] for u in idxs])
        f["norm_offset"] = np.asarray([float(units[u]["norm_offset"]) for u in idxs])
        f["main_tokens"] = np.asarray([routing.get(u, {}).get("main_tokens", np.nan) for u in idxs])
        f["condition_tokens"] = np.asarray([routing.get(u, {}).get("condition_tokens", np.nan) for u in idxs])
        f["null_condition"] = np.asarray([1.0 if routing.get(u, {}).get("null_condition") else 0.0 for u in idxs] if routing else [np.nan] * len(idxs))
        return f

    ftop, fbot = feats(top_units), feats(bottom_units)
    smd_fields = ("zoom", "zoom_val", "latent_shift", "pixel_shift", "abs_displacement", "available", "canvas_aspect", "retention", "norm_offset")
    smds = {k: v2c.smd(ftop[k], fbot[k]) for k in smd_fields}
    if routing_available:
        smds["main_tokens"] = v2c.smd(ftop["main_tokens"], fbot["main_tokens"])
        smds["condition_tokens"] = v2c.smd(ftop["condition_tokens"], fbot["condition_tokens"])
    # norm_offset / pixel_shift / abs_displacement define the TOP/BOTTOM split
    # itself (spec 42-43): tautological SMDs, excluded from the imbalance gate.
    largest_smd, geometry_imbalanced = v2c.geometry_imbalance_gate(smds)

    geom_table = {}
    for nm, idxs in (("TOP", top_units), ("BOTTOM", bottom_units)):
        f = feats(idxs)
        geom_table[nm] = {
            "n": len(idxs),
            "zoom": group_stats(f["zoom_val"]),
            "latent_shift": group_stats(f["latent_shift"]),
            "pixel_shift": group_stats(f["pixel_shift"]),
            "abs_displacement": group_stats(f["abs_displacement"]),
            "canvas_aspect": group_stats(f["canvas_aspect"]),
            "retention": group_stats(f["retention"]),
            "available": group_stats(f["available"]),
            "norm_offset": group_stats(f["norm_offset"]),
            "main_tokens": group_stats(f["main_tokens"]) if routing_available else {"n": 0, "status": "NOT_AVAILABLE"},
            "condition_tokens": group_stats(f["condition_tokens"]) if routing_available else {"n": 0, "status": "NOT_AVAILABLE"},
            "null_condition_fraction": (float(f["null_condition"].mean()) if routing_available else None),
            "shard": shard_table([units[u] for u in idxs], "source_shard"),
        }

    # ---------------- valid vertical subsets for effects ----------------
    v_valid = [u for u in v_units if deltas[u]["valid"]]
    top_valid = [u for u in v_valid if v2c.tertile_of(float(units[u]["norm_offset"])) == "START"]
    bottom_valid = [u for u in v_valid if v2c.tertile_of(float(units[u]["norm_offset"])) == "END"]
    d_top = np.asarray([deltas[u]["d_opp"] for u in top_valid])
    d_bot = np.asarray([deltas[u]["d_opp"] for u in bottom_valid])

    # cell definitions (pooled over valid vertical units): aspect tertile + available tertile
    fvv = feats(v_valid)
    aspect_cuts = tertiles(fvv["canvas_aspect"])
    avail_cuts = tertiles(fvv["available"])

    def cell_of(u: int) -> str:
        zb = zoom_band(float(units[u]["zoom"]))
        lb = latent_band(float(units[u]["latent_shift"]))
        at = band_of(recon[u]["canvas_aspect"], aspect_cuts)
        av = band_of(recon[u]["available"], avail_cuts)
        return f"{zb}|{lb}|aspect{at}|avail{av}"

    cell_top: dict[str, list[int]] = {}
    cell_bot: dict[str, list[int]] = {}
    for u in top_valid:
        cell_top.setdefault(cell_of(u), []).append(u)
    for u in bottom_valid:
        cell_bot.setdefault(cell_of(u), []).append(u)

    # ---------------- geometry-stratified END effect (33) ----------------
    rng = np.random.default_rng(BOOT_SEED)
    strat_cells = []
    for cname in sorted(set(cell_top) | set(cell_bot)):
        tu = cell_top.get(cname, [])
        bu = cell_bot.get(cname, [])
        td = np.asarray([deltas[u]["d_opp"] for u in tu])
        bd = np.asarray([deltas[u]["d_opp"] for u in bu])
        strat_cells.append({
            "cell": cname,
            "top": {"n": len(tu), "mean": float(td.mean()) if td.size else None, "ci95": _cluster_ci(td, rng) if td.size >= 2 else None},
            "bottom": {"n": len(bu), "mean": float(bd.mean()) if bd.size else None, "ci95": _cluster_ci(bd, rng) if bd.size >= 2 else None},
            "top_small_n": len(tu) < 20,
            "bottom_small_n": len(bu) < 20,
        })
    n_small = sum(1 for c in strat_cells if c["top_small_n"] or c["bottom_small_n"])

    # ---------------- reweighted TOP vs BOTTOM (34) ----------------
    n_top_cell = {c: len(v) for c, v in cell_top.items()}
    n_bot_cell = {c: len(v) for c, v in cell_bot.items()}
    rw = v2c.cell_reweight_weights(n_top_cell, n_bot_cell)
    ret_top = [u for u in top_valid if cell_of(u) in set(rw["shared_cells"])]
    ret_bot = [u for u in bottom_valid if cell_of(u) in set(rw["shared_cells"])]
    rw_res = {"status": "INSUFFICIENT_SUPPORT"}
    if ret_top and ret_bot:
        wt = np.asarray([rw["weight_top_by_cell"][cell_of(u)] for u in ret_top])
        wb = np.asarray([rw["weight_bottom_by_cell"][cell_of(u)] for u in ret_bot])
        dt = np.asarray([deltas[u]["d_opp"] for u in ret_top])
        db = np.asarray([deltas[u]["d_opp"] for u in ret_bot])
        rw_res = {
            "status": "OK",
            "shared_cells": rw["n_shared_cells"],
            "dropped_cells": len(set(n_top_cell) | set(n_bot_cell)) - rw["n_shared_cells"],
            "retained_units": {"top": len(ret_top), "bottom": len(ret_bot)},
            "dropped_units": {"top": rw["n_dropped_top"], "bottom": rw["n_dropped_bottom"]},
            "top": v2c.weighted_mean_ci(dt, wt, N_BOOT, BOOT_SEED, rng),
            "bottom": v2c.weighted_mean_ci(db, wb, N_BOOT, BOOT_SEED, rng),
        }

    # unbalanced reference estimates (same valid vertical sets, plain unit bootstrap)
    ref_top = {"point": float(d_top.mean()) if d_top.size else None, "ci95": _cluster_ci(d_top, rng) if d_top.size >= 2 else None}
    ref_bot = {"point": float(d_bot.mean()) if d_bot.size else None, "ci95": _cluster_ci(d_bot, rng) if d_bot.size >= 2 else None}
    bottom_sig_unbalanced = bool(ref_bot["ci95"] and ref_bot["ci95"][1] < 0.0)
    bottom_sig_reweighted = bool(rw_res["status"] == "OK" and rw_res["bottom"]["ci95"][1] < 0.0)
    bottom_negative_after_reweight = bool(
        (rw_res["status"] == "OK" and rw_res["bottom"]["ci95"][1] < 0.0)
        if bottom_sig_unbalanced
        else (ref_bot["point"] is not None and ref_bot["point"] < 0.0)
    )
    shrink = None
    if rw_res["status"] == "OK" and ref_bot["point"] and ref_bot["point"] < 0:
        shrink = float(1.0 - rw_res["bottom"]["point"] / ref_bot["point"])
    geometry_explains_bottom = bool(
        bottom_sig_unbalanced
        and rw_res["status"] == "OK"
        and (rw_res["bottom"]["ci95"][0] >= 0.0 or (shrink is not None and shrink >= 0.5))
    )

    # ---------------- matched geometry audit (35) ----------------
    if len(top_valid) >= 10 and len(bottom_valid) >= 10:
        match_feats = ("zoom_val", "latent_shift", "canvas_aspect", "available")
        tf = np.stack([ftop[k] for k in match_feats], axis=1)
        bf = np.stack([fbot[k] for k in match_feats], axis=1)
        pooled = np.concatenate([tf, bf], axis=0)
        mu, sd = pooled.mean(axis=0), pooled.std(axis=0, ddof=1)
        sd = np.where(sd <= 0, 1.0, sd)
        match = v2c.match_nearest((bf - mu) / sd, (tf - mu) / sd)
        bi_idx = [p[0] for p in match["pairs"]]
        ti_idx = [p[1] for p in match["pairs"]]
        post_match = {}
        for k_i, k in enumerate(match_feats):
            tt = tf[ti_idx, k_i]
            bt = bf[bi_idx, k_i]
            post_match[k] = v2c.smd((tt - mu[k_i]) / sd[k_i], (bt - mu[k_i]) / sd[k_i])
        pair_diffs = np.asarray([
            deltas[top_valid[p[1]]]["d_opp"] - deltas[bottom_valid[p[0]]]["d_opp"] for p in match["pairs"]
        ])
        match_res = {
            "n_pairs": match["n_pairs"],
            "with_replacement": match["with_replacement"],
            "post_match_smd": post_match,
            "max_post_match_smd": max(post_match.values()),
            "top_minus_bottom": v2c.paired_bootstrap_diff(pair_diffs, N_BOOT, BOOT_SEED, rng),
        }
    else:
        match_res = {"status": "INSUFFICIENT_SUPPORT"}

    # ---------------- continuous offset relationship (40-41) ----------------
    q_v = np.asarray([float(units[u]["norm_offset"]) for u in v_valid])
    d_v = np.asarray([deltas[u]["d_opp"] for u in v_valid])
    if d_v.size >= 10:
        rho, rho_ci = v2c.spearman_ci(q_v, d_v, 4000, BOOT_SEED, rng)
        deciles = v2c.decile_stats(q_v, d_v, 4000, BOOT_SEED, rng)
        mirror = v2c.mirror_bin_stats(q_v, d_v, 0.1, 4000, BOOT_SEED, rng)
        continuous = {
            "spearman_rho": rho,
            "spearman_ci95": rho_ci,
            "deciles": deciles,
            "mirror_bins": mirror,
            "interpretation": (
                "monotone-ish decline from TOP to BOTTOM"
                if rho < -0.15
                else ("only the extreme BOTTOM band is abnormal"
                      if any(m["mirror_diff_low_minus_high"] is not None and m["ci95"] and m["ci95"][1] < 0.0 for m in mirror if m["bin_low"] == "[0.0,0.1)")
                      else "no clear continuous trend")
            ),
        }
    else:
        continuous = {"status": "INSUFFICIENT_SUPPORT"}

    # ---------------- shard / cohort bias (39) ----------------
    from collections import Counter
    c_top = Counter(units[u]["source_shard"] for u in bottom_valid)  # BOTTOM focus
    c_all = Counter(units[u]["source_shard"] for u in v_valid)
    total_b, total_a = sum(c_top.values()), sum(c_all.values())
    shard_dev = []
    for s, nb in c_top.items():
        pb = nb / total_b if total_b else 0.0
        pa = c_all.get(s, 0) / total_a if total_a else 0.0
        shard_dev.append((s, pb - pa, pb, pa))
    shard_dev.sort(key=lambda x: -abs(x[1]))
    # chi-square on BOTTOM vs non-BOTTOM vertical (valid)
    shards_all = set(c_top) | {s for s in c_all if s not in c_top}
    obs = np.zeros((2, len(shards_all)))
    for i, s in enumerate(sorted(shards_all)):
        obs[0, i] = c_top.get(s, 0)
        obs[1, i] = c_all.get(s, 0) - c_top.get(s, 0)
    total = obs.sum()
    if obs.shape[1] > 1 and (obs > 0).any():
        row_m, col_m = obs.sum(axis=1, keepdims=True), obs.sum(axis=0, keepdims=True)
        expm = row_m @ col_m / total
        with np.errstate(divide="ignore", invalid="ignore"):
            chi2 = float(((obs - expm) ** 2 / np.where(expm == 0, np.nan, expm)).sum())
        df = 1 * (obs.shape[1] - 1)
        p_chi2 = _chi2_pvalue(chi2, df) if np.isfinite(chi2) else None
    else:
        chi2, p_chi2, df = None, None, 0
    shard_imbalance = bool(p_chi2 is not None and p_chi2 < 0.05)

    # ---------------- balance classification (42-43) ----------------
    notes: list[str] = []
    if exposure_imbalance:
        classification = "PLANNER_COUNT_IMBALANCE"
        notes.append("observed exposure deviates from exact discrete planner expectation (|z| >= 3)")
    elif geometry_imbalanced and geometry_explains_bottom:
        classification = "GEOMETRY_DISTRIBUTION_IMBALANCE"
        notes.append("TOP/BOTTOM geometry imbalance explains the BOTTOM negative (collapses after balance)")
    elif bottom_negative_after_reweight and not exposure_imbalance and not geometry_imbalanced:
        classification = "EFFECT_PERSISTS_AFTER_GEOMETRY_BALANCE"
        notes.append("counts and geometry balanced; BOTTOM negative persists after balance -> directional/content interaction, NOT 'sampling imbalance'")
    elif bottom_negative_after_reweight and (shard_imbalance or (routing_available and smds.get("main_tokens", 0.0) and smds["main_tokens"] >= v2c.SMD_DIAGNOSTIC)):
        classification = "CONTENT_OR_SHARD_INTERACTION_SUSPECTED"
        notes.append("residual after balance + shard/routing cohort differences")
    else:
        classification = "INCONCLUSIVE"
        notes.append("insufficient discriminating evidence")

    # ---------------- C2 secondary replication context ----------------
    c2 = None
    if args.c2_dist.is_file():
        with open(args.c2_dist, "r", encoding="utf-8") as fh:
            c2 = json.load(fh)
        c2 = {"source": str(args.c2_dist), "sha256": _sha256(args.c2_dist), "data": c2}
    else:
        log(f"C2 distribution json absent at {args.c2_dist} - continuing with 2048 primary (not a FAIL)")

    # ---------------- content / raw caption status (37-38) ----------------
    content = {
        "routing_covariates": "AVAILABLE" if routing_available else "NOT_AVAILABLE",
        "routing_coverage": len(routing),
        "raw_caption_metadata": "NOT_AVAILABLE",
        "raw_caption_note": "validation-prompts.json holds 1000 cases keyed by prompt_id; no sample_id join key exists in the cohort manifests and no WebDataset decode/network is permitted - raw caption/tag covariates cannot be attached to the 2048 units",
    }

    out = {
        "status": "OK",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_s": round(time.time() - t0, 2),
        "present": True,
        "bootstrap_note": "seed=20260907 everywhere; n=10000 for reweighted/matched/reference unit-cluster bootstraps (fixed by the prompt) and for the paired matched bootstrap; decile/mirror/Spearman use 4000 replicates (not prompt-fixed)",
        "inputs": {
            "evidence": str(ev),
            "stage1_manifest_sha256": _sha256(ev / "stage1-manifest.json"),
            "results_worker0_sha256": _sha256(ev / "results-worker0.pt"),
            "results_worker1_sha256": _sha256(ev / "results-worker1.pt"),
            "c2_distribution": (c2 or {}).get("sha256"),
        },
        "n_camera": n_cam,
        "valid_d_opp_units": len(valid_units),
        "d_opp_definition": "D_opp_unit = (M_OPP_POST - M_OPP_PRE) - (M_SAME_POST - M_SAME_PRE); M = 4-stratum mean margin; finite-at-both-endpoints and not opp_na",
        "planner": {
            "sampler": "left/top = random.Random(offset_seed).randrange(available + 1) on the long axis (camera_viewport.py plan_camera_viewport)",
            "reconstruction_verified": n_bad == 0,
            "reconstruction": "F = round(zoom^2 * 256); available = F - 256; k = round(norm_offset * available); signed_shift = k + 128 - F/2 must equal pixel_shift (EXACT for all units)",
            "blocks": plan_blocks,
            "exposure_count_imbalance": bool(exposure_imbalance),
            "z_threshold": v2c.Z_IMBALANCE,
        },
        "vertical_geometry": {
            "TOP": geom_table["TOP"],
            "BOTTOM": geom_table["BOTTOM"],
            "smd": smds,
            "largest_imbalance": largest_smd,
            "geometry_imbalanced": bool(geometry_imbalanced),
            "smd_diagnostic_threshold": v2c.SMD_DIAGNOSTIC,
            "split_definitional_excluded": list(v2c.SPLIT_DEFINITIONAL_COVARIATES),
            "note": "|SMD| < 0.1 is a conventional diagnostic, not a scientific hard truth; split-definitional covariates (the TOP/BOTTOM split variables) are excluded from the imbalance gate",
        },
        "reference_effects_unbalanced": {"TOP": ref_top, "BOTTOM": ref_bot},
        "stratified_cells": {
            "definitions": {"zoom": "mild<1.20 / medium<1.35 / strong", "latent_shift": "<2 / 2-4 / >=4", "aspect": "pooled vertical tertiles of canvas_aspect", "available": "pooled vertical tertiles of available"},
            "aspect_cuts": list(aspect_cuts),
            "available_cuts": list(avail_cuts),
            "cells": strat_cells,
            "n_small_n_cells": n_small,
        },
        "reweighted": rw_res,
        "matched": match_res,
        "continuous": continuous,
        "shard": {
            "chi2": chi2,
            "df": df,
            "p_approx_wilson_hilferty": p_chi2,
            "max_bottom_overrepresentation": shard_dev[:5],
            "shard_cohort_imbalance": bool(shard_imbalance),
        },
        "content": content,
        "bottom_negative_after_reweight": bottom_negative_after_reweight,
        "geometry_explains_bottom": bool(geometry_explains_bottom),
        "effect_shrink_after_reweight": shrink,
        "balance_classification": classification,
        "classification_notes": notes,
    }

    if args.skip_write:
        log("skip-write: analysis complete, no reports written")
        print(json.dumps(out, indent=2, allow_nan=False)[:4000])
        return 0

    reports = args.repo / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "camera-coordinate-causal-offset-balance.json").write_text(json.dumps(out, indent=2, allow_nan=False), encoding="utf-8")

    # CSV: per-cell aggregates (section 46 - no per-forward tensors)
    csv_lines = ["cell,scope,n,mean_d_opp,ci_lo,ci_hi,small_n"]
    for c in strat_cells:
        for scope in ("top", "bottom"):
            s = c[scope]
            if s["mean"] is None:
                csv_lines.append(f"{c['cell']},{scope},0,,,")
            else:
                lo, hi = s["ci95"]
                csv_lines.append(f"{c['cell']},{scope},{s['n']},{s['mean']:.10e},{lo:.10e},{hi:.10e},{str(c['bottom_small_n' if scope == 'bottom' else 'top_small_n']).lower()}")
    (reports / "camera-coordinate-causal-offset-balance.csv").write_text("\n".join(csv_lines) + "\n", encoding="utf-8")

    # Markdown
    def fmt(v, nd=4):
        return "n/a" if v is None else (f"{v:.{nd}e}" if isinstance(v, float) else str(v))

    L = []
    L.append("# Camera Coordinate Causal - Offset Balance Audit (Posthoc V2)")
    L.append("")
    L.append(f"- generated: {out['generated_utc']}")
    L.append(f"- evidence: {ev} (manifest sha {out['inputs']['stage1_manifest_sha256'][:16]}…, worker0 {out['inputs']['results_worker0_sha256'][:16]}…, worker1 {out['inputs']['results_worker1_sha256'][:16]}…)")
    L.append(f"- camera units: {n_cam}; valid D_opp units: {len(valid_units)}")
    L.append("- D_opp_unit = (M_OPP_POST - M_OPP_PRE) - (M_SAME_POST - M_SAME_PRE) (no new forwards)")
    L.append("")
    L.append("## 1. Planner discrete symmetry")
    L.append("")
    L.append(f"sampler: {out['planner']['sampler']}")
    L.append(f"reconstruction verified exact for all units: {out['planner']['reconstruction_verified']}")
    L.append("")
    L.append("| scope | tertile | observed | expected | sd | z |")
    L.append("|---|---|---|---|---|---|")
    for scope in ("all", "horizontal", "vertical"):
        blk = plan_blocks[scope]
        for t in v2c.TERTS:
            L.append(f"| {scope} | {t} | {blk['observed'][t]} | {blk['expected'][t]:.2f} | {blk['sd'][t]:.2f} | {blk['z'][t]:+.3f} |")
    L.append("")
    L.append(f"EXPOSURE_COUNT_IMBALANCE (|z| >= {v2c.Z_IMBALANCE}): **{exposure_imbalance}**")
    L.append("")
    L.append("Mirror symmetry (k vs available-k): uniform sampler => P(k) = 1/(available+1) = P(available-k) EXACTLY (static proof).")
    L.append("")
    L.append("## 2. Vertical TOP vs BOTTOM geometry")
    L.append("")
    L.append(f"TOP n = {geom_table['TOP']['n']}; BOTTOM n = {geom_table['BOTTOM']['n']}")
    L.append("")
    L.append("| covariate | TOP mean | TOP p50 | TOP p90 | BOTTOM mean | BOTTOM p50 | BOTTOM p90 | SMD |")
    L.append("|---|---|---|---|---|---|---|---|")
    for k in ("zoom", "latent_shift", "pixel_shift", "abs_displacement", "canvas_aspect", "retention", "available", "norm_offset"):
        a, b = geom_table["TOP"][k], geom_table["BOTTOM"][k]
        L.append(f"| {k} | {fmt(a['mean'])} | {fmt(a['p50'])} | {fmt(a['p90'])} | {fmt(b['mean'])} | {fmt(b['p50'])} | {fmt(b['p90'])} | {fmt(smds.get(k, float('nan')), 3)} |")
    if routing_available:
        for k in ("main_tokens", "condition_tokens"):
            a, b = geom_table["TOP"][k], geom_table["BOTTOM"][k]
            L.append(f"| {k} | {fmt(a['mean'])} | {fmt(a['p50'])} | {fmt(a['p90'])} | {fmt(b['mean'])} | {fmt(b['p50'])} | {fmt(b['p90'])} | {fmt(smds.get(k, float('nan')), 3)} |")
    L.append("")
    if largest_smd is None:
        L.append("largest non-definitional imbalance: n/a; geometry_imbalanced: **False**")
    else:
        L.append(f"largest non-definitional imbalance: **{largest_smd}** (SMD {fmt(smds[largest_smd], 3)}); geometry_imbalanced (>= {v2c.SMD_DIAGNOSTIC}): **{geometry_imbalanced}**")
    L.append("split-definitional SMDs (tautological, excluded from gate): " + ", ".join(f"{k}={fmt(smds[k], 3)}" for k in v2c.SPLIT_DEFINITIONAL_COVARIATES if k in smds))
    L.append(f"null-condition fraction: TOP {geom_table['TOP']['null_condition_fraction']} / BOTTOM {geom_table['BOTTOM']['null_condition_fraction']}")
    L.append("")
    L.append("## 3. Reference effects (unbalanced valid vertical units)")
    L.append("")
    L.append(f"TOP adjusted D_opp: {fmt(ref_top['point'])} CI {ref_top['ci95']}")
    L.append(f"BOTTOM adjusted D_opp: {fmt(ref_bot['point'])} CI {ref_bot['ci95']} (CI<0: {bottom_sig_unbalanced})")
    L.append("")
    L.append("## 4. Reweighted (exact cell reweighting, shared cells only)")
    L.append("")
    if rw_res["status"] == "OK":
        L.append(f"shared cells: {rw_res['shared_cells']}; retained units TOP {rw_res['retained_units']['top']} / BOTTOM {rw_res['retained_units']['bottom']}; dropped units {rw_res['dropped_units']}")
        L.append(f"TOP ESS = {rw_res['top']['ess']:.1f}; BOTTOM ESS = {rw_res['bottom']['ess']:.1f}")
        L.append(f"TOP reweighted: {fmt(rw_res['top']['point'])} CI {rw_res['top']['ci95']}")
        L.append(f"BOTTOM reweighted: {fmt(rw_res['bottom']['point'])} CI {rw_res['bottom']['ci95']} (negative persists: {bottom_sig_reweighted})")
        L.append(f"effect shrink after reweight: {fmt(shrink, 3) if shrink is not None else 'n/a'}")
    else:
        L.append(f"status: {rw_res['status']}")
    L.append("")
    L.append("## 5. Matched geometry audit (1:1 without replacement)")
    L.append("")
    if match_res.get("status") != "INSUFFICIENT_SUPPORT":
        L.append(f"pairs: {match_res['n_pairs']}; post-match SMD: {json.dumps(match_res['post_match_smd'])}; max = {fmt(match_res['max_post_match_smd'], 3)}")
        L.append(f"TOP - BOTTOM paired difference: {fmt(match_res['top_minus_bottom']['point'])} CI {match_res['top_minus_bottom']['ci95']}")
    else:
        L.append("INSUFFICIENT_SUPPORT")
    L.append("")
    L.append("## 6. Continuous offset relationship (vertical)")
    L.append("")
    if continuous.get("status") != "INSUFFICIENT_SUPPORT":
        L.append(f"Spearman(norm_offset, D_opp) = {fmt(continuous['spearman_rho'], 3)} CI {continuous['spearman_ci95']}")
        L.append("deciles (by normalized offset): n / mean D_opp / CI")
        for d in continuous["deciles"]:
            L.append(f"  [{d['decile'] * 0.1:.1f},{(d['decile'] + 1) * 0.1:.1f}): n={d['n']} mean={fmt(d['mean'])} ci={d['ci95']}")
        L.append("mirror bins (low - high):")
        for m in continuous["mirror_bins"]:
            L.append(f"  {m['bin_low']} vs {m['bin_high']}: n {m['n_low']}/{m['n_high']} diff={fmt(m['mirror_diff_low_minus_high'])} ci={m['ci95']}")
        L.append(f"interpretation: {continuous['interpretation']}")
    else:
        L.append("INSUFFICIENT_SUPPORT")
    L.append("")
    L.append("## 7. Shard / cohort / content")
    L.append("")
    L.append(f"shard chi2 = {chi2} df={df} p≈{p_chi2}; SHARD_COHORT_IMBALANCE: {shard_imbalance}")
    L.append(f"content: {content['routing_covariates']}; raw caption: {content['raw_caption_metadata']}")
    L.append("")
    L.append("## 8. Balance classification")
    L.append("")
    L.append(f"**{classification}**")
    for n_ in notes:
        L.append(f"- {n_}")
    L.append("")
    L.append("Caveat: only a genuine exposure or geometry imbalance is called a 'sampling balance' issue. If counts and geometry are balanced and the BOTTOM effect persists, the correct statement is that a directional/content interaction remains after balance (section 43).")
    L.append("")
    (reports / "camera-coordinate-causal-offset-balance.md").write_text("\n".join(L), encoding="utf-8")
    log(f"reports written under {reports}")
    log(f"CLASSIFICATION = {classification}; bottom_negative_after_reweight={bottom_negative_after_reweight}; geometry_explains_bottom={geometry_explains_bottom}; exposure_imbalance={exposure_imbalance}; geometry_imbalanced={geometry_imbalanced}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
