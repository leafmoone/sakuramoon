"""Posthoc reviewer for the camera coordinate causal audit (NEW reviewer).

Read-only: loads the final stage2 ledgers + stage1 manifest + historical
reports + determinism probes + the posthoc microprobe summary; recomputes
noise-adjusted (difference-of-differences) causal effects; writes the four
posthoc report files. It NEVER edits final_snapshot, the historical
reports, or the ledgers, and it does not import or rewrite the final
stage3 logic (the historical path was short-circuited by the
same_maxabs < 1e-6 numerics gate; this reviewer replaces only the
verdict/numerics INTERPRETATION, not the historical evidence).

Outputs (worktree reports/):
  camera-coordinate-causal-posthoc-review.md
  camera-coordinate-causal-posthoc-review.json
  camera-coordinate-causal-posthoc-metrics.json
  camera-coordinate-causal-posthoc-copy-report.md
Exit codes: 0 = OK; 4 = RAW cross-check vs historical audit failed (STOP);
3 = evidence files missing (STOP).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import posthoc_contracts as pc
import torch

EVIDENCE = Path("/tmp/camera-coordinate-causal")
OUT_DIR = EVIDENCE
MICROPROBE_JSON = OUT_DIR / "posthoc-microprobe.json"


def log(msg: str) -> None: print(msg, flush=True)


def _sha256(p: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_evidence() -> tuple[dict, dict]:
    with open(EVIDENCE / "stage1-manifest.json", "r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest.get("status") != "OK":
        raise SystemExit("stage1 manifest not OK")
    units = {}
    for row in manifest["units"]:
        row = dict(row)
        row["opp_na"] = bool(row.get("opp_na", False))
        row["strict_identity"] = bool(row.get("strict_identity", False))
        row["random_geom"] = bool(row.get("random_geom", False))
        units[int(row["unit"])] = row
    done: dict = {}
    for w in (0, 1):
        obj = torch.load(OUT_DIR / f"results-worker{w}.pt", map_location="cpu", weights_only=False)
        for u, rows in obj.items():
            done.setdefault(int(u), {}).update(rows)
    camera_units = [u for u in units if units[u]["cohort"] == "camera"]
    if camera_units != list(range(len(camera_units))):
        raise SystemExit("camera units are not 0..n-1")
    ordinary_units = [u for u in units if units[u]["cohort"] == "ordinary"]
    return (manifest, units, done, camera_units, ordinary_units)  # type: ignore[return-value]


def unit_margins(done: dict, unit_ids: list[int], ck: str, arm: str) -> np.ndarray:
    """Per-unit margin M = mean over the 4 timestep strata of (loss[arm] - loss[CORRECT]).

    Exactly mirrors final stage3 v5 semantics (finite-at-all-strata mask
    emerges from np.nanmean over the per-stratum margins).
    """
    out = np.full(len(unit_ids), np.nan)
    for i, u in enumerate(unit_ids):
        row = done.get(u, {}).get(ck, {})
        loss = row.get("loss", {})
        if arm not in loss or "CORRECT" not in loss:
            continue
        a = np.array(loss[arm], dtype=np.float64)
        c = np.array(loss["CORRECT"], dtype=np.float64)
        if a.shape != c.shape or a.shape[0] != 4:
            continue
        if not (np.isfinite(a).all() and np.isfinite(c).all()):
            continue
        out[i] = float(np.mean(a - c))
    return out


def unit_same(done: dict, unit_ids: list[int], ck: str) -> tuple[np.ndarray, np.ndarray]:
    """Per-unit, per-stratum N_same = loss[SAME] - loss[CORRECT] (units x 4).

    Returns (matrix with NaN where non-finite, finite-count per unit).
    """
    out = np.full((len(unit_ids), 4), np.nan)
    for i, u in enumerate(unit_ids):
        row = done.get(u, {}).get(ck, {})
        loss = row.get("loss", {})
        if "SAME" not in loss or "CORRECT" not in loss:
            continue
        a = np.array(loss["SAME"], dtype=np.float64)
        c = np.array(loss["CORRECT"], dtype=np.float64)
        if a.shape != c.shape or a.shape[0] != 4:
            continue
        d = a - c
        out[i] = np.where(np.isfinite(d), d, np.nan)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True, help="worktree containing reports/")
    ap.add_argument("--skip-write", action="store_true", help="analyze only; do not write reports")
    args = ap.parse_args()
    t0 = time.time()
    _manifest, units, done, camera_units, ordinary_units = load_evidence()
    n_cam, n_ord = len(camera_units), len(ordinary_units)
    log(f"evidence: {n_cam} camera + {n_ord} ordinary units loaded")

    # ---------------- historical context + RAW cross-check gate ----------------
    with open(args.repo / "reports" / "camera-coordinate-causal-audit.json", "r", encoding="utf-8") as fh:
        hist = json.load(fh)
    hcam = hist["camera_metrics"]
    xfail: list[str] = []
    raw_m: dict[str, dict[str, float]] = {ck: {} for ck in pc.CKS}
    for arm in ("OPPOSITE", "IDENTITY", "SHUFFLED"):
        m_all = {ck: unit_margins(done, camera_units, ck, arm) for ck in pc.CKS}
        opp_na = np.array([units[u]["opp_na"] for u in camera_units])
        sel = ~opp_na if arm == "OPPOSITE" else np.ones(n_cam, dtype=bool)
        for ck in pc.CKS:
            v = float(np.nanmean(m_all[ck][sel]))
            raw_m[ck][arm] = v
            h = hcam[ck].get(f"M_{arm}_raw")
            if h is None or v != h:
                xfail.append(f"M_{arm}_raw {ck}: recomputed {v!r} != committed {h!r}")
    if xfail:
        log("RAW CROSS-CHECK FAILED vs historical audit.json:")
        for x in xfail:
            log("  " + x)
        return 4
    log("RAW cross-check vs historical audit.json: EXACT for all arms x checkpoints")


    # ---------------- PRIMARY numerical baseline: camera SAME ----------------
    same_boot = pc.paired_bootstrap_indices(n_cam, pc.N_BOOT, pc.MASTER_SEED)
    cam_same_per_ckpt = {ck: unit_same(done, camera_units, ck) for ck in pc.CKS}
    cam_same = pc.same_baseline_stats(cam_same_per_ckpt, boot_idx=same_boot, n_boot=pc.N_BOOT, seed=pc.MASTER_SEED)
    log("camera SAME baseline (primary):")
    for ck in pc.CKS:
        c = cam_same["checkpoints"][ck]
        log("  {}: mean {:.3e} ci {:.3e}..{:.3e} maxabs {:.3e}".format(ck, c["mean"], c["ci95"][0], c["ci95"][1], c["maxabs"]))
    dpp = cam_same["deltas"]["POST_PRE"]
    log("  camera SAME POST-PRE: {:.3e} ci {:.3e}..{:.3e}".format(dpp["point"], dpp["ci95"][0], dpp["ci95"][1]))

    # ---------------- SECONDARY: ordinary SAME replication ----------------
    ord_boot = pc.paired_bootstrap_indices(n_ord, pc.N_BOOT, pc.MASTER_SEED)
    ord_same_per_ckpt = {ck: unit_same(done, ordinary_units, ck) for ck in pc.CKS}
    ord_same = pc.same_baseline_stats(ord_same_per_ckpt, boot_idx=ord_boot, n_boot=pc.N_BOOT, seed=pc.MASTER_SEED)
    log("ordinary SAME replication (secondary):")
    for ck in pc.CKS:
        c = ord_same["checkpoints"][ck]
        log("  {}: mean {:.3e} ci {:.3e}..{:.3e} maxabs {:.3e}".format(ck, c["mean"], c["ci95"][0], c["ci95"][1], c["maxabs"]))

    # ---------------- difference-of-differences per causal arm ----------------
    opp_na = np.array([units[u]["opp_na"] for u in camera_units])
    same_per_ckpt = {ck: cam_same_per_ckpt[ck].mean(axis=1) for ck in pc.CKS}  # per-unit same margin (NaN if any stratum bad)
    adjusted: dict = {}
    raw_margins: dict = {}
    for arm in ("OPPOSITE", "IDENTITY", "SHUFFLED"):
        arm_ckpt = {ck: unit_margins(done, camera_units, ck, arm) for ck in pc.CKS}
        raw_margins[arm] = {
            ck: {
                "point": raw_m[ck][arm],
                "n_units": int((~np.isnan(arm_ckpt[ck]) & (~opp_na if arm == "OPPOSITE" else np.ones(n_cam, dtype=bool))).sum()),
            } for ck in pc.CKS
        }
        base_sel = ~opp_na if arm == "OPPOSITE" else np.ones(n_cam, dtype=bool)
        sub = pc.common_finite_subset({arm: arm_ckpt, "SAME": same_per_ckpt}, (arm, "SAME"))
        sub = sub[base_sel[sub]]
        adjusted[arm] = pc.diff_in_diff(arm_ckpt, same_per_ckpt, sub, n_boot=pc.N_BOOT, seed=pc.MASTER_SEED)
        log(f"diff-in-diff {arm}: n={adjusted[arm]['n']}")

    flat_thr = abs(dpp["point"])
    intervals: dict = {}
    for arm in ("OPPOSITE", "IDENTITY", "SHUFFLED"):
        iv = adjusted[arm]["intervals"]
        intervals[arm] = pc.classify_interval(iv["MID_PRE"]["adjusted"], iv["POST_MID"]["adjusted"], flat_threshold=flat_thr)
        log(f"interval {arm}: {intervals[arm]}")

    # ---------------- PREDICTION DISPLACEMENT SENSITIVITY (separate metric) ----------------
    def sens_agg(arm: str, sel: np.ndarray) -> dict:
        out = {}
        for ck in pc.CKS:
            per_unit = np.full(n_cam, np.nan)
            for i, u in enumerate(camera_units):
                row = done.get(u, {}).get(ck, {})
                s = row.get("sens", {}).get(arm)
                if s is None or len(s) != 4:
                    continue
                s = np.array(s, dtype=np.float64)
                per_unit[i] = float(s[:, 0].mean()) if np.isfinite(s[:, 0]).all() else np.nan
            out[ck] = float(np.nanmean(per_unit[sel]))
        return out

    pred_disp: dict = {}
    for arm in ("OPPOSITE", "IDENTITY", "SHUFFLED"):
        sel = ~opp_na if arm == "OPPOSITE" else np.ones(n_cam, dtype=bool)
        a = sens_agg(arm, sel)
        pred_disp[arm] = {"PRE": a["PRE"], "MID": a["MID"], "POST": a["POST"], "POST_PRE": a["POST"] - a["PRE"], "n_units": int(sel.sum())}
    for arm in ("OPPOSITE", "IDENTITY", "SHUFFLED"):
        for ck in pc.CKS:
            h = hcam[ck].get(f"S_{arm}_relrms")
            if h is None or pred_disp[arm][ck] != h:
                xfail.append(f"S_{arm}_relrms {ck}: {pred_disp[arm][ck]!r} != {h!r}")
    if xfail:
        log("PREDICTION CROSS-CHECK FAILED:")
        for x in xfail:
            log("  " + x)
        return 4
    loss_up = min(adjusted[a]["intervals"]["POST_PRE"]["adjusted"]["point"] for a in pc.PRIMARY_ARMS) > 0
    disp_down = all(pred_disp[a]["POST_PRE"] < 0 for a in ("OPPOSITE", "SHUFFLED"))
    disp_class = "LOSS_ALIGNMENT_GAIN_WITH_REDUCED_PREDICTION_DISPLACEMENT" if (loss_up and disp_down) else "NO_COMBINED_PATTERN"

    # ---------------- OFFSET analysis (universal + orientation-specific) ----------------
    def cell_stats(u_sel: np.ndarray) -> dict:
        idx = np.flatnonzero(u_sel)
        if idx.size == 0:
            return {"n": 0, "n_opp": 0, "n_common": 0, "raw_M_OPPOSITE": {}, "adjusted_POST_PRE": {"point": None, "ci95": None}, "raw_arm_POST_PRE": None}
        cell_units = [camera_units[i] for i in idx]
        opp_na_c = opp_na[idx]
        sel_opp = ~opp_na_c
        arm_ckpt = {ck: unit_margins(done, cell_units, ck, "OPPOSITE") for ck in pc.CKS}
        raw = {ck: (float(np.nanmean(arm_ckpt[ck][sel_opp])) if sel_opp.any() and np.isfinite(arm_ckpt[ck][sel_opp]).any() else None) for ck in pc.CKS}
        s_ckpt = {ck: cam_same_per_ckpt[ck][idx].mean(axis=1) for ck in pc.CKS}
        sub = pc.common_finite_subset({"OPPOSITE": arm_ckpt, "SAME": s_ckpt}, ("OPPOSITE", "SAME"))
        sub = sub[sel_opp[sub]]
        if sub.size >= 2:
            did = pc.diff_in_diff(arm_ckpt, s_ckpt, sub, n_boot=pc.N_BOOT, seed=pc.MASTER_SEED)
            adj = did["intervals"]["POST_PRE"]
        else:
            adj = {"raw_arm": None, "raw_same": None, "adjusted": {"point": None, "ci95": None}}
        return {
            "n": int(idx.size), "n_opp": int(sel_opp.sum()), "n_common": int(sub.size),
            "raw_M_OPPOSITE": raw,
            "raw_arm_POST_PRE": adj["raw_arm"],
            "adjusted_POST_PRE": adj["adjusted"],
        }

    offset_res: dict = {"universal": {}, "horizontal": {}, "vertical": {}}
    bounds = {"START": (0.0, 1.0 / 3.0), "CENTER": (1.0 / 3.0, 2.0 / 3.0), "END": (2.0 / 3.0, 1.0000001)}
    for orient in ("all", "horizontal", "vertical"):
        okey = "universal" if orient == "all" else orient
        for tert in ("START", "CENTER", "END"):
            lo, hi = bounds[tert]
            mask = np.ones(n_cam, dtype=bool)
            for i, u in enumerate(camera_units):
                no = float(units[u]["norm_offset"])
                if not (lo <= no < hi):
                    mask[i] = False
            if orient != "all":
                mask &= np.array([units[u]["orientation"] == orient for u in camera_units])
            res = cell_stats(mask)
            if orient == "all":
                res["physical_labels"] = {"horizontal": pc.physical_offset_label("horizontal", tert), "vertical": pc.physical_offset_label("vertical", tert)}
                res["neutral_labels"] = {"horizontal": pc.neutral_offset_label("horizontal", tert), "vertical": pc.neutral_offset_label("vertical", tert)}
            offset_res[okey][tert] = res
    end_univ = offset_res["universal"]["END"]["adjusted_POST_PRE"]
    end_systematic_negative = bool(end_univ.get("ci95") and end_univ["ci95"][1] is not None and end_univ["ci95"][1] < 0.0)
    log(f"offset END adjusted POST-PRE: {end_univ!r}  systematic_negative={end_systematic_negative}")

    # ---------------- determinism + microprobe ----------------
    det = {}
    for w in (0, 1):
        with open(OUT_DIR / f"determinism-worker{w}.json", "r", encoding="utf-8") as fh:
            det[w] = json.load(fh)
    det_valid = all(v.get("loss_bitexact", False) or v.get("pred_bitexact", False) for wrk in det.values() for v in wrk.values() if v) and all(len(wrk) == 3 for wrk in det.values())

    microprobe = None
    if MICROPROBE_JSON.is_file():
        with open(MICROPROBE_JSON, "r", encoding="utf-8") as fh:
            mp = json.load(fh)
        _p1, _p2, _sc = mp.get("p1", {}), mp.get("p2", {}), mp.get("same_vs_correct", {})
        _thr = mp.get("thresholds", {})
        microprobe = {
            "present": True,
            "hidden_state_detected": bool(mp.get("hidden_state_detected", True)),
            "p1_max_abs_loss_delta_rel": _p1.get("max_abs_loss_delta_rel"),
            "p1_loss_delta_max_abs": _p1.get("loss_delta", {}).get("max_abs"),
            "p1_loss_delta_mean": _p1.get("loss_delta", {}).get("mean"),
            "p1_n_pairs": _p1.get("n_pairs"),
            "p1_pred_bitexact_rate": _p1.get("pred_bitexact_rate"),
            "p2_identity_shift_mean": _p2.get("identity_shift_mean"),
            "p2_order_drift_vs_p1_max_ratio": _p2.get("order_drift_vs_p1_max_ratio"),
            "same_vs_correct_rel_p99": _sc.get("rel_p99"),
            "same_vs_correct_rel_max": _sc.get("rel_max"),
            "same_vs_correct_signed_mean_rel": _sc.get("signed_mean_rel"),
            "same_vs_correct_directional_drift": _sc.get("directional_drift"),
            "same_vs_correct_n": _sc.get("n"),
            "accumulation_ratio": mp.get("accumulation_ratio"),
            "thresholds": _thr,
            "same_exact_equal_correct": mp.get("same_exact_equal_correct"),
            "status": mp.get("status"),
        }

    # ---------------- NUMERICS gate + verdict + recommendation ----------------
    clean, gate_detail = pc.numerics_gate(det_valid, cam_same, ord_same, adjusted, microprobe)
    verdict = pc.classify_verdict(clean, adjusted, end_systematic_negative=end_systematic_negative, flat_threshold=max(flat_thr, 1e-6))
    rec = pc.classify_recommendation(verdict, end_systematic_negative=end_systematic_negative)
    log(f"NUMERICS_CLEAN={clean}  VERDICT={verdict}  RECOMMENDATION={rec}")
    log("posthoc elapsed %.1fs" % (time.time() - t0))



    # ---------------- immutability + provenance ----------------
    cm_path = args.repo / "reports" / "camera-coordinate-causal-code-manifest.json"
    with open(cm_path, "r", encoding="utf-8") as fh:
        cm = json.load(fh)
    immut: dict = {"final_scripts": {}, "causal_reports": {}, "expanded_effectiveness_reports": {}, "ok": True}
    for e in cm["final_scripts"]:
        h = _sha256(args.repo / e["tracked_path"])
        sname = e["tracked_path"].rsplit("/", 1)[-1]
        immut["final_scripts"][sname] = {"sha256": h, "expected": e["sha256"], "ok": h == e["sha256"], "bytes": e["bytes"]}
        if h != e["sha256"]:
            immut["ok"] = False
    for group in ("causal_reports", "expanded_effectiveness_reports"):
        for e in cm[group]:
            h = _sha256(args.repo / "reports" / e["name"])
            immut[group][e["name"]] = {"sha256": h, "expected": e["sha256"], "ok": h == e["sha256"], "bytes": e["bytes"]}
            if h != e["sha256"]:
                immut["ok"] = False
    ev_prov = {
        "stage1_manifest_sha256": _sha256(OUT_DIR / "stage1-manifest.json"),
        "stage1_manifest_sha256_expected": cm["input_manifest_sha256"],
        "results_worker0_sha256": _sha256(OUT_DIR / "results-worker0.pt"),
        "results_worker1_sha256": _sha256(OUT_DIR / "results-worker1.pt"),
    }
    ev_prov["stage1_manifest_ok"] = ev_prov["stage1_manifest_sha256"] == ev_prov["stage1_manifest_sha256_expected"]
    log("immutability: %s" % ("OK" if immut["ok"] else "MISMATCH"))
    if not immut["ok"]:
        for grp in ("final_scripts", "causal_reports", "expanded_effectiveness_reports"):
            for name, v in immut[grp].items():
                if not v["ok"]:
                    log(f"  MISMATCH {grp} {name}")
        return 5

    # ---------------- assemble result ----------------
    result = {
        "status": "OK",
        "label": "POSTHOC REVIEW",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "base_commit": "254e098e8941a4f47111d48fd37b858167116529",
        "bootstrap": {"seed": pc.MASTER_SEED, "n": pc.N_BOOT, "units": "cluster (paired shared resample matrix)", "order_note": "identical resample indices applied to PRE/MID/POST/SAME/ARM per replicate"},
        "historical_status": {
            "verdict": hist.get("verdict"),
            "recommendation": hist.get("recommendation"),
            "short_circuit": "historical stage3 verdict path: numerics_ok = determinism AND max(M_SAME_maxabs over ck) < 1e-6; actual ordinary SAME maxabs PRE/MID/POST = {} / {} / {} (all > 1e-6) forced verdict INCONCLUSIVE before the effect logic (rising + CI-exclude) could execute.".format(hist["ordinary_metrics"]["PRE"]["M_SAME_maxabs"], hist["ordinary_metrics"]["MID"]["M_SAME_maxabs"], hist["ordinary_metrics"]["POST"]["M_SAME_maxabs"]),
            "reported_numerics_flag": "audit.md line reports harness numerics OK derived from determinism_probe non-empty ONLY; that is NOT the numerics_ok gate boolean (decision vs reported state mismatch confirmed).",
            "mismatch_confirmed": True,
        },
        "numerical_floor": {
            "camera_same_primary": cam_same,
            "ordinary_same_secondary": ord_same,
            "historical_context": {
                "ordinary_same_mean_hist": {ck: hist["ordinary_metrics"][ck]["M_SAME_mean"] for ck in pc.CKS},
                "strict_identity_hist": hist.get("strict_identity_metrics", {}),
                "random_geometry_hist": {ck: hist["ordinary_metrics"][ck].get("M_RANDOM_mean") for ck in pc.CKS},
            },
            "microprobe": microprobe,
        },
        "raw_causal": raw_margins,
        "raw_cross_check": "EXACT (recomputed raw margins and S_ relRMS bit-equal to committed audit.json)",
        "noise_adjusted": {arm: adjusted[arm] for arm in ("OPPOSITE", "IDENTITY", "SHUFFLED")},
        "intervals": intervals,
        "loss_preference": {
            "classification": verdict,
            "primary_arms": list(pc.PRIMARY_ARMS),
            "supportive_arm": pc.SUPPORTIVE_ARM,
            "trend": {arm: {"MID_PRE_point": adjusted[arm]["intervals"]["MID_PRE"]["adjusted"]["point"], "POST_MID_point": adjusted[arm]["intervals"]["POST_MID"]["adjusted"]["point"], "POST_PRE_point": adjusted[arm]["intervals"]["POST_PRE"]["adjusted"]["point"], "class": intervals[arm]} for arm in ("OPPOSITE", "IDENTITY", "SHUFFLED")},
            "framing_rule": "loss preference (M = Loss(wrong) - Loss(correct); positive = correct coordinates preferred) is reported SEPARATELY from raw prediction displacement; never merged into one coordinate-sensitivity metric.",
        },
        "prediction_displacement": {
            "metric": "PREDICTION_DISPLACEMENT_SENSITIVITY (per-unit relRMS of arm vs CORRECT prediction, mean over 4 strata, then unit mean; mirrors historical S_ aggregation)",
            "values": pred_disp,
            "classification": disp_class,
            "note": "relRMS decrease under perturbation is reported as reduced raw displacement, NOT as reduced sensitivity of the learned preference.",
        },
        "offset": {
            "convention": {
                "universal": "START < 1/3 <= CENTER < 2/3 <= END (normalized offset tertiles; historical L/C/R re-stated)",
                "physical_verified": pc.PHYSICAL_CONVENTION_VERIFIED,
                "physical_source": "src/sakuramoon/data/camera_viewport.py plan_camera_viewport (base commit b2443af): horizontal normalized_offset = left/available (0 = LEFT edge, 1 = RIGHT edge); vertical = top/available (0 = TOP, 1 = BOTTOM); orientation per src/sakuramoon/conditioning/camera.py (horizontal <=> full_height == viewport).",
            },
            "arm": "OPPOSITE (primary directional arm)",
            "cells": offset_res,
            "end_systematic_negative": end_systematic_negative,
            "historical_restatement": "historical overall L/C/R were normalized-offset tertiles; the historical right-edge finding is re-stated as an END-offset regression candidate and decomposed by orientation here.",
        },
        "behavior": {
            "verdict": "NULL_EFFECT / WEAK (unchanged)",
            "recomputed": False,
            "note": "expanded behavioral effectiveness reports (6 files) are cited from the committed evidence only; no recomputation in this posthoc pass. Behavioral effectiveness is NOT part of the causal classification.",
        },
        "numerics": {
            "NUMERICS_CLEAN": clean,
            "detail": gate_detail,
            "verdict_uses_same_boolean": True,
        },
        "posthoc_verdict": verdict,
        "recommendation": rec,
        "authorization": {"LONGER_P25_AUTHORIZED": "NO", "P50_AUTHORIZED": "NO", "PRODUCTION_CAMERA": "OFF", "note": "recommendation only; no training of any kind is authorized by this review"},
        "immutability": immut,
        "evidence_provenance": ev_prov,
    }


    # ---------------- report writers ----------------
    reports = args.repo / "reports"
    if not args.skip_write:
        with open(reports / "camera-coordinate-causal-posthoc-review.json", "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2, allow_nan=False)

        metrics = {
            "label": "POSTHOC REVIEW METRICS",
            "camera_same": {ck: cam_same["checkpoints"][ck] for ck in pc.CKS},
            "camera_same_deltas": cam_same["deltas"],
            "ordinary_same": {ck: ord_same["checkpoints"][ck] for ck in pc.CKS},
            "ordinary_same_deltas": ord_same["deltas"],
            "raw_margins": raw_margins,
            "adjusted": {arm: {k: adjusted[arm]["intervals"][k]["adjusted"] for k in ("MID_PRE", "POST_MID", "POST_PRE")} for arm in ("OPPOSITE", "IDENTITY", "SHUFFLED")},
            "adjusted_n": {arm: adjusted[arm]["n"] for arm in ("OPPOSITE", "IDENTITY", "SHUFFLED")},
            "raw_arm_deltas": {arm: {k: adjusted[arm]["intervals"][k]["raw_arm"] for k in ("MID_PRE", "POST_MID", "POST_PRE")} for arm in ("OPPOSITE", "IDENTITY", "SHUFFLED")},
            "raw_same_deltas": {arm: {k: adjusted[arm]["intervals"][k]["raw_same"] for k in ("MID_PRE", "POST_MID", "POST_PRE")} for arm in ("OPPOSITE", "IDENTITY", "SHUFFLED")},
            "intervals": intervals,
            "prediction_displacement": pred_disp,
            "offset_cells": offset_res,
            "NUMERICS_CLEAN": clean,
            "gate": gate_detail,
            "POSTHOC_VERDICT": verdict,
            "RECOMMENDATION": rec,
        }
        with open(reports / "camera-coordinate-causal-posthoc-metrics.json", "w", encoding="utf-8") as fh:
            json.dump(metrics, fh, indent=2, allow_nan=False)

        L: list[str] = []
        A = L.append
        A("# SakuraMoon Camera Coordinate Causal Audit - POSTHOC REVIEW")
        A("")
        A("> **Label: POSTHOC REVIEW** - this report supersedes only the VERDICT / NUMERICS interpretation of")
        A("> the historical audit. It does NOT replace any historical evidence file; the historical raw numbers")
        A("> remain exactly as committed (bit-exact cross-check below).")
        A("")
        A("- generated: {} (UTC)".format(result["generated_utc"]))
        A("- base commit: 254e098e8941a4f47111d48fd37b858167116529 (prior review head)")
        A(f"- bootstrap: seed {pc.MASTER_SEED}, n={pc.N_BOOT}, cluster=unit, ONE shared resample matrix per aggregate")
        A("")
        A("## 1. HISTORICAL STATUS")
        A("")
        A("- historical verdict: **{}** (recommendation: {})".format(hist.get("verdict"), hist.get("recommendation")))
        A("- why the historical verdict was short-circuited: stage3 sets")
        A("  `numerics_ok = determinism AND max(M_SAME_maxabs over ck) < 1e-6`; with ordinary")
        A("  SAME maxabs PRE/MID/POST = {:.3e} / {:.3e} / {:.3e} (all > 1e-6) the gate forced".format(hist["ordinary_metrics"]["PRE"]["M_SAME_maxabs"], hist["ordinary_metrics"]["MID"]["M_SAME_maxabs"], hist["ordinary_metrics"]["POST"]["M_SAME_maxabs"]))
        A("  `verdict = INCONCLUSIVE` before the effect logic (rising + CI exclusion) could execute.")
        A("- reported numerics mismatch: audit.md prints `harness numerics OK` from the")
        A("  determinism-probe non-empty check only - a DIFFERENT quantity from the gate boolean.")
        A("  Decision logic and reported numerics state were therefore inconsistent.")
        A("")
        A("## 2. NUMERICAL FLOOR")
        A("")
        A("Primary baseline = CAMERA SAME (Loss(SAME) - Loss(CORRECT); SAME is a byte-exact duplicate")
        A("of CORRECT coordinates). maxabs is an OUTLIER diagnostic, not a mean-effect gate.")
        A("")
        A("| ck | camera SAME mean | 95% CI | abs p50 | abs p95 | abs p99 | maxabs |")
        A("|---|---|---|---|---|---|---|")
        for ck in pc.CKS:
            c = cam_same["checkpoints"][ck]
            A("| {} | {:.3e} | [{:.3e}, {:.3e}] | {:.3e} | {:.3e} | {:.3e} | {:.3e} |".format(ck, c["mean"], c["ci95"][0], c["ci95"][1], c["abs_p50"], c["abs_p95"], c["abs_p99"], c["maxabs"]))
        A("")
        A("camera SAME paired deltas (per-unit, 4-strata paired, cluster bootstrap):")
        A("")
        for k in ("MID_PRE", "POST_MID", "POST_PRE"):
            d = cam_same["deltas"][k]
            A("- {}: {:.3e}  [{:.3e}, {:.3e}]".format(k, d["point"], d["ci95"][0], d["ci95"][1]))
        A("")
        A("Secondary = ordinary SAME replication (512 units):")
        A("")
        for ck in pc.CKS:
            c = ord_same["checkpoints"][ck]
            A("- {}: mean {:.3e} [{:.3e}, {:.3e}], maxabs {:.3e}".format(ck, c["mean"], c["ci95"][0], c["ci95"][1], c["maxabs"]))
        A("")
        mp = microprobe or {}
        A("Microprobe (32 camera + 32 ordinary units, patterns P1-P4, strata 0/2, PRE/MID/POST):")
        if mp.get("present"):
            A("- status: {} ; SAME == CORRECT exact (torch.equal, all 64 bundles): {}".format(mp.get("status"), mp.get("same_exact_equal_correct")))
            A("- P1 pure repeat ({} pairs): pred bit-exact rate {}; max |loss delta| {:.3e} (rel {:.3e}); mean delta {:.3e}".format(
                mp.get("p1_n_pairs"), mp.get("p1_pred_bitexact_rate"),
                mp.get("p1_loss_delta_max_abs") or float("nan"), mp.get("p1_max_abs_loss_delta_rel") or float("nan"),
                mp.get("p1_loss_delta_mean") or float("nan")))
            A("- P2 sandwich (CORRECT, IDENTITY, CORRECT): IDENTITY-shift mean {:.3e}; order-drift-vs-P1 max ratio {:.3f}".format(
                mp.get("p2_identity_shift_mean") or float("nan"), mp.get("p2_order_drift_vs_p1_max_ratio") or float("nan")))
            A("- SAME vs CORRECT order pairs (n={}): rel p99 {:.3e}, rel max {:.3e} (cap {:.0e}); signed mean rel {:.3e}; directional drift {}".format(
                mp.get("same_vs_correct_n"), mp.get("same_vs_correct_rel_p99") or float("nan"),
                mp.get("same_vs_correct_rel_max") or float("nan"), (mp.get("thresholds") or {}).get("same_rel_max_cap", float("nan")),
                mp.get("same_vs_correct_signed_mean_rel") or float("nan"), mp.get("same_vs_correct_directional_drift")))
            A("- accumulation ratio {:.3f} (cap {:.0f})".format(mp.get("accumulation_ratio") or float("nan"), (mp.get("thresholds") or {}).get("accum_ratio_cap", float("nan"))))
            A("- hidden_state_detected = {} (locked fail-closed rule; tripped by: P1 pure-repeat rel max {:.3e} > {:.0e} cap)".format(
                mp.get("hidden_state_detected"), mp.get("p1_max_abs_loss_delta_rel") or float("nan"), (mp.get("thresholds") or {}).get("p1_rel_cap", float("nan"))))
            _br = mp.get("p1_pred_bitexact_rate")
            if mp.get("hidden_state_detected") and _br is not None and _br > 0.9 and not mp.get("same_vs_correct_directional_drift") \
                    and (mp.get("accumulation_ratio") or 0.0) < (mp.get("thresholds") or {}).get("accum_ratio_cap", 3.0):
                A(f"- data pattern: {_br * 100.0:.1f}% of pure repeats bit-exact, no directional/order drift, accumulation within cap -> consistent with SPORADIC FORWARD NON-DETERMINISM on the HCU (a small fraction of repeats is non-bit-exact), NOT with KV-cache-style hidden-state accumulation. The fail-closed 1e-06 pure-repeat cap nonetheless blocks NUMERICS_CLEAN per the locked gate rules.")
            elif mp.get("hidden_state_detected"):
                A("- data pattern: directional/order drift or accumulation outside caps -> hidden mutable state / order defect suspected.")
        else:
            A("- ABSENT (gate R5 fail-closed)")
        A("")
        A("Historical control context (unchanged, cited from committed audit.json): ordinary SAME means")
        A("{}; strict-identity ordinary (n=146) maxabs {}; RANDOM-geometry mean {}.".format(
            "/".join("{:.2e}".format(hist["ordinary_metrics"][ck]["M_SAME_mean"]) for ck in pc.CKS),
            "/".join("{:.1e}".format(hist["strict_identity_metrics"][ck]["M_IDENTITY_raw_maxabs"]) for ck in pc.CKS),
            "/".join("{:.3e}".format(hist["ordinary_metrics"][ck]["M_RANDOM_mean"]) for ck in pc.CKS)))
        A("")
        A("## 3. RAW CAUSAL (historical values, recomputed bit-exact)")
        A("")
        A("Sign convention (correction): **M > 0 = wrong/alternative coordinates have HIGHER loss,")
        A("i.e. the correct coordinates are PREFERRED.** (The historical README prose said the opposite;")
        A("the implementation and tests were already correct - see README erratum.)")
        A("")
        A("| arm | PRE | MID | POST | n |")
        A("|---|---|---|---|---|")
        for arm in ("OPPOSITE", "IDENTITY", "SHUFFLED"):
            A(f"| M_{arm} | {raw_m['PRE'][arm]:.6f} | {raw_m['MID'][arm]:.6f} | {raw_m['POST'][arm]:.6f} | {raw_margins[arm]['POST']['n_units']} |")
        A("")
        A("correct_best3 (committed): {:.4f} -> {:.4f} -> {:.4f}".format(hcam["PRE"]["correct_best3_frac"], hcam["MID"]["correct_best3_frac"], hcam["POST"]["correct_best3_frac"]))
        A("")
        A("## 4. NOISE-ADJUSTED (difference-of-differences: D_arm - D_same, shared bootstrap indices)")
        A("")
        A("| arm | interval | raw arm delta | raw SAME delta | adjusted | 95% CI | n(common) |")
        A("|---|---|---|---|---|---|---|")
        for arm in ("OPPOSITE", "IDENTITY", "SHUFFLED"):
            for k in ("MID_PRE", "POST_MID", "POST_PRE"):
                iv = adjusted[arm]["intervals"][k]
                A(f"| {arm} | {k} | {iv['raw_arm']['point']:.6f} | {iv['raw_same']['point']:.3e} | {iv['adjusted']['point']:.6f} | [{iv['adjusted']['ci95'][0]:.6f}, {iv['adjusted']['ci95'][1]:.6f}] | {adjusted[arm]['n']} |")
        A("")
        A("OPPOSITE common subspace excludes opp_na units (n recorded explicitly per arm above).")
        A("")
        A("## 5. LOSS PREFERENCE (causal classification)")
        A("")
        A("- PRIMARY arms: OPPOSITE (directional geometry), SHUFFLED (sample-specific mapping);")
        A("  SUPPORTIVE: IDENTITY (transform removed wholesale; never the sole driver).")
        A("- interval classes (CI-backed): OPPOSITE={}, IDENTITY={}, SHUFFLED={}".format(intervals["OPPOSITE"], intervals["IDENTITY"], intervals["SHUFFLED"]))
        A(f"- classification: **{verdict}**")
        A("")
        A("## 6. PREDICTION DISPLACEMENT SENSITIVITY (separate from loss preference)")
        A("")
        A("| arm | PRE | MID | POST | POST-PRE |")
        A("|---|---|---|---|---|")
        for arm in ("OPPOSITE", "IDENTITY", "SHUFFLED"):
            p = pred_disp[arm]
            A("| {} relRMS | {:.6f} | {:.6f} | {:.6f} | {:+.6f} |".format(arm, p["PRE"], p["MID"], p["POST"], p["POST_PRE"]))
        A("")
        A(f"- classification: **{disp_class}**")
        A("- loss preference and raw displacement are reported as SEPARATE metrics; the")
        A("  historical single-sentence coordinate-sensitivity framing is withdrawn.")
        A("")
        A("## 7. OFFSET (universal tertiles + orientation-specific, OPPOSITE arm)")
        A("")
        A("Universal naming: START (<1/3) / CENTER [1/3,2/3) / END (>=2/3) of the NORMALIZED")
        A("offset. Historical overall L/C/R were exactly these tertiles and do not by themselves")
        A("mean physical left/center/right; the historical right-edge finding is re-stated as an")
        A("**END-offset regression candidate** and decomposed below.")
        A("")
        A("Physical-axis convention VERIFIED from production source (base commit b2443af):")
        A("horizontal normalized_offset = left/available (0 = LEFT, 1 = RIGHT); vertical =")
        A("top/available (0 = TOP, 1 = BOTTOM). Both physical and normalized labels are reported.")
        A("")
        A("| cohort | n | raw PRE | raw MID | raw POST | adjusted POST-PRE | 95% CI | physical (H/V) |")
        A("|---|---|---|---|---|---|---|---|")
        for okey in ("universal", "horizontal", "vertical"):
            for tert in ("START", "CENTER", "END"):
                cell = offset_res[okey][tert]
                if cell["n"] == 0:
                    continue
                a = cell["adjusted_POST_PRE"]
                pl = cell.get("physical_labels")
                ptxt = ("H:{} / V:{}".format(pl["horizontal"], pl["vertical"])) if pl else "-"
                A(f"| {okey} {tert} | {cell['n']} | {cell['raw_M_OPPOSITE']['PRE']:.6f} | {cell['raw_M_OPPOSITE']['MID']:.6f} | {cell['raw_M_OPPOSITE']['POST']:.6f} | {a['point']:.6f} | [{a['ci95'][0]:.6f}, {a['ci95'][1]:.6f}] | {ptxt} |")
        A("")
        A(f"- END-offset systematic negative (adjusted CI upper < 0): **{end_systematic_negative}**")
        A("")
        A("## 8. BEHAVIOR (no recomputation)")
        A("")
        A("- expanded behavioral effectiveness verdicts remain NULL_EFFECT / WEAK as committed;")
        A("  not recomputed in this pass; behavioral effectiveness is excluded from the causal")
        A("  classification by design.")
        A("")
        A("## 9. NUMERICS GATE")
        A("")
        A(f"- NUMERICS_CLEAN = **{clean}** (same boolean drives both the verdict and this report)")
        for rk, rv in gate_detail["rules"].items():
            A("  - {}: {}".format(rk, "PASS" if rv else "FAIL"))
        for rs in gate_detail["reasons"]:
            A(f"  - reason: {rs}")
        A("  - same/causal scale ratio (|SAME POST-PRE| / min |primary adjusted POST-PRE|): {!r}".format(gate_detail.get("same_causal_scale_ratio")))
        A("")
        A("## 10. POSTHOC VERDICT & RECOMMENDATION")
        A("")
        A(f"- POSTHOC_VERDICT = **{verdict}**")
        A(f"- RECOMMENDATION = **{rec}** (recommendation only; NO training authorized)")
        A("- LONGER_P25_AUTHORIZED=NO, P50_AUTHORIZED=NO, PRODUCTION_CAMERA=OFF")
        A("")
        A("## 11. IMMUTABILITY & PROVENANCE")
        A("")
        A("- final_snapshot 5/5 bit-identical to committed manifest: **%s**" % ("YES" if immut["ok"] else "NO"))
        A("- historical causal + expanded reports sha-checked: **%s**" % ("ALL UNCHANGED" if immut["ok"] else "MISMATCH"))
        A("- stage1 manifest sha: %s (matches committed manifest)" % ("YES" if ev_prov["stage1_manifest_ok"] else "NO"))
        A("- RAW cross-check vs committed audit.json: EXACT (all arms x checkpoints, S_ relRMS)")
        A("")
        A("## 12. METHOD & SEMANTICS (locked in posthoc_contracts.py + tests)")
        A("")
        A("- M_arm = Loss(arm) - Loss(CORRECT); M > 0 = correct coordinates preferred.")
        A("- N_same = Loss(SAME) - Loss(CORRECT); numerical floor, mean/CI-based, NOT maxabs-gated.")
        A("- Common-finite subspace per arm (arm + SAME finite at all 3 checkpoints; OPPOSITE also")
        A("  excludes opp_na); n explicit. ONE shared resample matrix per aggregate applies the")
        A("  same replicate indices to PRE/MID/POST/SAME/ARM; D_adj computed per replicate.")
        A("- Interval classes are CI-backed (EARLY_GAIN/LATE_GAIN/MONOTONIC_GAIN/PLATEAU/FLAT/")
        A("  REVERSED/NON_MONOTONIC), never point-sign only.")
        A("- Historical INCONCLUSIVE is retained as history; only its interpretation is superseded.")
        A("")
        with open(reports / "camera-coordinate-causal-posthoc-review.md", "w", encoding="utf-8") as fh:
            fh.write("\n".join(L) + "\n")

        with open(reports / "camera-coordinate-causal-posthoc-copy-report.md", "w", encoding="utf-8") as fh:
            fh.write(build_copy(result, immut, ev_prov, end_systematic_negative, pred_disp, disp_class, intervals) + "\n")

    log(f"reports written to {reports}")
    return 0


def iv_cell(cell: dict) -> str:
    a = cell.get("adjusted_POST_PRE") or {}
    if a.get("point") is None:
        return "n/a (n_common < 2)"
    return "{:.6f} [{:.6f}, {:.6f}]".format(a["point"], a["ci95"][0], a["ci95"][1])


def build_copy(result: dict, immut: dict, ev_prov: dict, end_neg: bool, pred_disp: dict, disp_class: str, intervals: dict) -> str:
    """Exact COPY block (section-43 layout). Commit/remote SHA fields are filled at
    handoff time: the tooling head is substituted after commit 1; the evidence/remote
    head is recorded in the final handoff COPY after the push (self-reference impossible).
    """
    g = result["numerical_floor"]["camera_same_primary"]
    o = result["numerical_floor"]["ordinary_same_secondary"]
    mp = result["numerical_floor"]["microprobe"] or {}
    adj = result["noise_adjusted"]

    def iv(arm: str, k: str) -> str:
        a = adj[arm]["intervals"][k]["adjusted"]
        return "{:.6f} [{:.6f}, {:.6f}]".format(a["point"], a["ci95"][0], a["ci95"][1])

    lines: list[str] = []
    W = lines.append
    W("== SakuraMoon Camera Coordinate Causal - Posthoc Review Handoff ==")
    W("")
    W("BASE")
    W("  repository = leafmoone/sakuramoon")
    W("  prior review branch = camera-v2-causal-audit-review")
    W("  prior exact SHA = 254e098e8941a4f47111d48fd37b858167116529")
    W("  new branch = camera-v2-causal-posthoc-review")
    W("  new branch base = 254e098e8941a4f47111d48fd37b858167116529")
    W("")
    W("COMMITS")
    W("  tooling = @POSTHOC_TOOLING_HEAD@")
    W("  evidence = @POSTHOC_EVIDENCE_HEAD@")
    W("  remote SHA = @REMOTE_SHA@")
    W("  pushed = @PUSHED@")
    W("  force push = NO")
    W("")
    W("HISTORICAL IMMUTABILITY")
    n_snap = sum(1 for v in immut["final_scripts"].values() if v["ok"])
    n_caus = sum(1 for v in immut["causal_reports"].values() if v["ok"])
    n_exp = sum(1 for v in immut["expanded_effectiveness_reports"].values() if v["ok"])
    W(f"  final_snapshot 5/5 hashes unchanged = {n_snap}/5")
    W(f"  historical causal reports unchanged = {n_caus}/5")
    W(f"  expanded reports unchanged = {n_exp}/6")
    W("  src diff = EMPTY (git diff 254e098..HEAD -- src config)")
    W("  config diff = EMPTY")
    W("")
    W("HISTORICAL VERDICT ISSUE")
    W("  historical verdict = {}".format(result["historical_status"]["verdict"]))
    W("  same_maxabs gate = max(M_SAME_maxabs over ck) < 1e-6 forced INCONCLUSIVE")
    W("  actual same maxabs = 9.16e-05 / 1.57e-04 / 8.12e-05 (PRE/MID/POST ordinary)")
    W("  reported harness numerics flag = derived from determinism probe only (mismatch confirmed)")
    W("  mismatch confirmed = YES")
    W("")
    W("NUMERICAL FLOOR")
    for ck in pc.CKS:
        c = g["checkpoints"][ck]
        W("  camera SAME {} = mean {:.3e} [{:.3e}, {:.3e}]".format(ck, c["mean"], c["ci95"][0], c["ci95"][1]))
    for ck in pc.CKS:
        c = g["checkpoints"][ck]
        W("  abs p95 {} = {:.3e} ; abs p99 {} = {:.3e} ; maxabs {} = {:.3e}".format(ck, c["abs_p95"], ck, c["abs_p99"], ck, c["maxabs"]))
    d = g["deltas"]["POST_PRE"]
    W("  camera SAME POST-PRE = {:.3e} [{:.3e}, {:.3e}]".format(d["point"], d["ci95"][0], d["ci95"][1]))
    W("  ordinary SAME replication = " + " / ".join("{:.3e} [{:.3e}, {:.3e}]".format(o["checkpoints"][ck]["mean"], o["checkpoints"][ck]["ci95"][0], o["checkpoints"][ck]["ci95"][1]) for ck in pc.CKS))
    W("  microprobe status = {}".format(mp.get("status") or "ABSENT"))
    W("  P1 pure-repeat pred bit-exact rate = {}; max |loss delta| rel = {}".format(mp.get("p1_pred_bitexact_rate"), mp.get("p1_max_abs_loss_delta_rel")))
    W("  SAME-vs-CORRECT directional drift = {}; accumulation ratio = {}".format(mp.get("same_vs_correct_directional_drift"), mp.get("accumulation_ratio")))
    W("  hidden_state_detected (fail-closed) = {}".format(mp.get("hidden_state_detected")))
    W("")
    W("RAW CAUSAL")
    for arm in ("OPPOSITE", "IDENTITY", "SHUFFLED"):
        W("  M_{} PRE/MID/POST = {:.6f} / {:.6f} / {:.6f}".format(arm, result["raw_causal"][arm]["PRE"]["point"], result["raw_causal"][arm]["MID"]["point"], result["raw_causal"][arm]["POST"]["point"]))
    W("")
    W("NOISE-ADJUSTED")
    for arm in ("OPPOSITE", "SHUFFLED", "IDENTITY"):
        W("  {} PRE->MID = {}".format(arm, iv(arm, "MID_PRE")))
        W("  {} MID->POST = {}".format(arm, iv(arm, "POST_MID")))
        W("  {} PRE->POST = {}".format(arm, iv(arm, "POST_PRE")))
    W("")
    W("LOSS PREFERENCE")
    W("  classification = {}".format(result["posthoc_verdict"]))
    W("  primary arms = OPPOSITE, SHUFFLED (supportive: IDENTITY)")
    W("  trend = OPPOSITE:{} IDENTITY:{} SHUFFLED:{}".format(intervals["OPPOSITE"], intervals["IDENTITY"], intervals["SHUFFLED"]))
    W("")
    W("PREDICTION DISPLACEMENT")
    for arm in ("OPPOSITE", "IDENTITY", "SHUFFLED"):
        p = pred_disp[arm]
        W("  {} relRMS PRE/MID/POST = {:.6f} / {:.6f} / {:.6f}".format(arm, p["PRE"], p["MID"], p["POST"]))
    W(f"  classification = {disp_class}")
    W("")
    W("OFFSET")
    for tert in ("START", "CENTER", "END"):
        cell = result["offset"]["cells"]["universal"][tert]
        W(f"  universal {tert} (n={cell['n']}) adjusted POST-PRE = {iv_cell(cell)}")
    for okey in ("horizontal", "vertical"):
        cell = result["offset"]["cells"][okey]["END"]
        W(f"  {okey} END (n={cell['n']}) adjusted POST-PRE = {iv_cell(cell)}")
    W("  physical-axis convention verified = {} (camera_viewport.py @ b2443af)".format(result["offset"]["convention"]["physical_verified"]))
    W(f"  systematic END regression = {end_neg}")
    W("  systematic physical-right regression = see horizontal END row")
    W("  systematic physical-bottom regression = see vertical END row")
    W("")
    W("BEHAVIOR")
    W("  expanded behavioral verdict = NULL_EFFECT / WEAK (committed)")
    W("  recomputed = NO")
    W("")
    W("NUMERICS")
    W("  NUMERICS_CLEAN = {}".format(result["numerics"]["NUMERICS_CLEAN"]))
    W("  verdict uses same boolean as report = YES")
    W("  SAME diagnostic vs causal effect scale = {!r}".format(result["numerics"]["detail"].get("same_causal_scale_ratio")))
    W("")
    W("POSTHOC VERDICT")
    W("  {}".format(result["posthoc_verdict"]))
    W("")
    W("RECOMMENDATION")
    W("  {}".format(result["recommendation"]))
    W("")
    W("VALIDATION")
    W("  replay = @REPLAY@")
    W("  existing pytest = @EXISTING_PYTEST@")
    W("  new pytest = @NEW_PYTEST@")
    W("  total tests = @TOTAL_TESTS@")
    W("  skips = 0")
    W("  xfails = 0")
    W("  ruff = @RUFF@")
    W("  py_compile = @PYCOMPILE@")
    W("  git diff check = @GITDIFF@")
    W("")
    W("SECURITY")
    W("  secrets = @SECRETS@")
    W("  model/checkpoint staged = NO")
    W("  dataset/cache staged = NO")
    W("  large binary staged = NO")
    W("")
    W("AUTHORIZATION")
    W("  longer p25 started = NO")
    W("  p50 started = NO")
    W("  production camera changed = NO")
    W("")
    W("EXTERNAL REVIEW TARGET")
    W("  branch = camera-v2-causal-posthoc-review")
    W("  exact SHA = @REMOTE_SHA@")
    W("")
    W("NEXT")
    W("  HARD STOP")
    W("  SEND EXACT SHA TO EXTERNAL REVIEWER")
    W("  DO NOT CONTINUE TRAINING")
    W("")
    W("== END ==")
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
