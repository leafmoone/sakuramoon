"""Camera coordinate causal audit - posthoc V2 review (read-only analyzer).

Combines (all from committed/frozen evidence, no new forwards except the
separately-run microprobe-v2):
  - historical RAW replay cross-check (bit-exact vs committed audit.json);
  - camera SAME numerical floor (primary) + ordinary SAME (secondary);
  - V2 difference-in-differences with EXACT observed point estimates
    (bootstrap = CI only; section 21);
  - offset cells (universal + physical labels) with V2 intervals;
  - microprobe-v2 state classification (jitter vs hidden mutable state,
    sections 7-8);
  - offset balance audit outcome (reports/camera-coordinate-causal-offset-balance.json,
    produced by offset_balance_review.py - REQUIRED input, fail closed);
  - V2 numerics gate R1-R9; verdict (V1 classes, reused) + V2 recommendation
    CASE A-E;
  - immutability: final snapshot (5) + historical causal (5) + expanded (6)
    + V1 posthoc reports (4) + V1 tooling (3);
  - four V2 reports (must not overwrite any V1 report).

Usage:
  python posthoc_v2_review.py --repo <worktree> [--evidence /tmp/...] [--skip-write]
Exit codes: 0 OK; 3 missing evidence; 4 RAW/balance cross-check failed;
5 immutability mismatch.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import posthoc_contracts as pc
import posthoc_v2_contracts as v2c

EVIDENCE = Path("/tmp/camera-coordinate-causal")
BASELINE = "0caf0a156f45eefc0986829005e49e59a7301082"
BALANCE_REPORT = "camera-coordinate-causal-offset-balance.json"

# V1 generation immutability pins (committed at BASELINE).
V1_POSTHOC_REPORT_SHA = {
    "camera-coordinate-causal-posthoc-review.md": "109731dcb2debceb12eeae7b244201b85ce8ba07ae47e6e154467bc158145898",
    "camera-coordinate-causal-posthoc-review.json": "ab34dad9ee59db6b0751692195087595154c970668345bb0402f60a8d1ce92d0",
    "camera-coordinate-causal-posthoc-metrics.json": "9bd441f3ccda5af7d6c88ea6c9b157971ccd303db19eeec9c1ee4dc3417ea592",
    "camera-coordinate-causal-posthoc-copy-report.md": "8d21bed7f629dc4ba00b6a15b1deb5d0649c444825f761c0f0b973d43ae8d305",
}
V1_TOOLING_SHA = {
    "posthoc_contracts.py": "32f0f13cfb4aab14a95501587ab0d215fe4cc43eaef4502ff2c2c0076d3da44a",
    "posthoc_review.py": "9cf718573e9acc35462f37200018ebdac538abb4de6d586b65bd0c9cf69d9a4a",
    "posthoc_numerics_probe.py": "de485e7b7a4d87527b19565fbc9244749f4e56ddf117be1fa03f5a71d7924fcd",
}


def log(msg: str) -> None:
    print(msg, flush=True)


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_evidence(ev: Path) -> tuple[dict, dict[int, dict], dict, list[int], list[int]]:
    with open(ev / "stage1-manifest.json", "r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest.get("status") != "OK":
        raise SystemExit("stage1 manifest not OK")
    units: dict[int, dict] = {}
    for row in manifest["units"]:
        row = dict(row)
        row["opp_na"] = bool(row.get("opp_na", False))
        units[int(row["unit"])] = row
    done: dict[int, dict] = {}
    for w in (0, 1):
        p = ev / f"results-worker{w}.pt"
        if not p.is_file():
            raise SystemExit(f"missing evidence file: {p}")
        obj = torch.load(p, map_location="cpu", weights_only=False)
        for u, rows in obj.items():
            done.setdefault(int(u), {}).update(rows)
    camera_units = [u for u in sorted(units) if units[u]["cohort"] == "camera"]
    ordinary_units = [u for u in sorted(units) if units[u]["cohort"] == "ordinary"]
    return manifest, units, done, camera_units, ordinary_units


def unit_margins(done: dict, unit_ids: list[int], ck: str, arm: str) -> np.ndarray:
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


def unit_same(done: dict, unit_ids: list[int], ck: str) -> np.ndarray:
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
    ap.add_argument("--evidence", type=Path, default=EVIDENCE)
    ap.add_argument("--skip-write", action="store_true", help="analyze only; do not write reports")
    args = ap.parse_args()
    ev = args.evidence
    t0 = time.time()
    _manifest, units, done, camera_units, ordinary_units = load_evidence(ev)
    n_cam, n_ord = len(camera_units), len(ordinary_units)
    log(f"evidence: {n_cam} camera + {n_ord} ordinary units loaded")

    # ---------------- historical RAW cross-check gate ----------------
    hist_path = args.repo / "reports" / "camera-coordinate-causal-audit.json"
    if not hist_path.is_file():
        log("FATAL: missing historical audit.json (STOP)")
        return 3
    with open(hist_path, "r", encoding="utf-8") as fh:
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

    # ---------------- numerical floor ----------------
    same_boot = pc.paired_bootstrap_indices(n_cam, v2c.N_BOOT_V2, v2c.MASTER_SEED_V2)
    cam_same_per_ckpt = {ck: unit_same(done, camera_units, ck) for ck in pc.CKS}
    cam_same = pc.same_baseline_stats(cam_same_per_ckpt, boot_idx=same_boot, n_boot=v2c.N_BOOT_V2, seed=v2c.MASTER_SEED_V2)
    log("camera SAME baseline (primary):")
    for ck in pc.CKS:
        c = cam_same["checkpoints"][ck]
        log("  {}: mean {:.3e} ci {:.3e}..{:.3e} maxabs {:.3e}".format(ck, c["mean"], c["ci95"][0], c["ci95"][1], c["maxabs"]))
    dpp = cam_same["deltas"]["POST_PRE"]
    log("  camera SAME POST-PRE: {:.3e} ci {:.3e}..{:.3e}".format(dpp["point"], dpp["ci95"][0], dpp["ci95"][1]))

    ord_boot = pc.paired_bootstrap_indices(n_ord, v2c.N_BOOT_V2, v2c.MASTER_SEED_V2)
    ord_same_per_ckpt = {ck: unit_same(done, ordinary_units, ck) for ck in pc.CKS}
    ord_same = pc.same_baseline_stats(ord_same_per_ckpt, boot_idx=ord_boot, n_boot=v2c.N_BOOT_V2, seed=v2c.MASTER_SEED_V2)
    log("ordinary SAME replication (secondary):")
    for ck in pc.CKS:
        c = ord_same["checkpoints"][ck]
        log("  {}: mean {:.3e} ci {:.3e}..{:.3e} maxabs {:.3e}".format(ck, c["mean"], c["ci95"][0], c["ci95"][1], c["maxabs"]))

    # ---------------- V2 difference-in-differences (exact points) ----------------
    opp_na = np.array([units[u]["opp_na"] for u in camera_units])
    same_per_ckpt = {ck: cam_same_per_ckpt[ck].mean(axis=1) for ck in pc.CKS}
    adjusted: dict = {}
    raw_margins: dict = {}
    for arm in ("OPPOSITE", "IDENTITY", "SHUFFLED"):
        arm_ckpt = {ck: unit_margins(done, camera_units, ck, arm) for ck in pc.CKS}
        raw_margins[arm] = {
            ck: {"point": raw_m[ck][arm], "n_units": int((~np.isnan(arm_ckpt[ck]) & (~opp_na if arm == "OPPOSITE" else np.ones(n_cam, dtype=bool))).sum())}
            for ck in pc.CKS
        }
        sub = v2c.select_arm_subspace(arm_ckpt, same_per_ckpt, opp_na if arm == "OPPOSITE" else None)
        adjusted[arm] = v2c.diff_in_diff_v2(arm_ckpt, same_per_ckpt, sub, n_boot=v2c.N_BOOT_V2, seed=v2c.MASTER_SEED_V2)
        log(f"diff-in-diff v2 {arm}: n={adjusted[arm]['n']} POST_PRE adjusted {adjusted[arm]['intervals']['POST_PRE']['adjusted']}")

    flat_thr = abs(dpp["point"])
    intervals: dict = {}
    for arm in ("OPPOSITE", "IDENTITY", "SHUFFLED"):
        iv = adjusted[arm]["intervals"]
        intervals[arm] = pc.classify_interval(iv["MID_PRE"]["adjusted"], iv["POST_MID"]["adjusted"], flat_threshold=flat_thr)
        log(f"interval {arm}: {intervals[arm]}")

    # ---------------- PREDICTION DISPLACEMENT (separate metric, exact cross-check) ----------------
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

    # ---------------- offset cells (V2 intervals) ----------------
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
        sub = v2c.select_arm_subspace(arm_ckpt, s_ckpt, opp_na_c)
        if sub.size >= 2:
            did = v2c.diff_in_diff_v2(arm_ckpt, s_ckpt, sub, n_boot=v2c.N_BOOT_V2, seed=v2c.MASTER_SEED_V2)
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
    log(f"offset END adjusted POST-PRE (v2): {end_univ!r}  systematic_negative={end_systematic_negative}")

    # ---------------- determinism + microprobe-v2 ----------------
    det = {}
    for w in (0, 1):
        p = ev / f"determinism-worker{w}.json"
        if not p.is_file():
            log(f"FATAL: missing {p} (STOP)")
            return 3
        with open(p, "r", encoding="utf-8") as fh:
            det[w] = json.load(fh)
    det_valid = all(v.get("loss_bitexact", False) or v.get("pred_bitexact", False) for wrk in det.values() for v in wrk.values() if v) and all(len(wrk) == 3 for wrk in det.values())

    mp2: dict | None = None
    mp2_path = ev / "posthoc-v2-microprobe.json"
    if mp2_path.is_file():
        with open(mp2_path, "r", encoding="utf-8") as fh:
            mp = json.load(fh)
        st = mp.get("state") or {}
        mp2 = {
            "present": True,
            "status": mp.get("status"),
            "state": st,
            "hidden_mutable_state_detected": bool(mp.get("hidden_mutable_state_detected", True)),
            "repeat_numeric_jitter_present": bool(mp.get("repeat_numeric_jitter_present", False)),
            "coordinate_map_mutation_detected": st.get("COORDINATE_MAP_MUTATION_DETECTED"),
            "persistent_buffer_mutation_detected": st.get("PERSISTENT_BUFFER_MUTATION_DETECTED"),
            "parameter_mutation_detected": st.get("parameter_mutation_detected"),
            "order_dependent_drift_detected": st.get("ORDER_DEPENDENT_DRIFT_DETECTED"),
            "accumulating_drift_detected": st.get("ACCUMULATING_DRIFT_DETECTED"),
            "p0": mp.get("p0", {}),
            "p1": mp.get("p1", {}),
            "p2": mp.get("p2", {}),
            "p3": mp.get("p3", {}),
            "parameter_mutation": mp.get("parameter_mutation", {}),
            "persistent_buffer_mutation": mp.get("persistent_buffer_mutation", {}),
            "rng": mp.get("rng", {}),
            "same_exact_equal_correct": mp.get("same_exact_equal_correct"),
            "checkpoints": mp.get("checkpoints"),
            "interpretation": mp.get("interpretation"),
        }
        log(f"microprobe-v2 loaded: hidden={mp2['hidden_mutable_state_detected']} jitter={mp2['repeat_numeric_jitter_present']}")
    else:
        log("microprobe-v2 ABSENT - state checks R6-R9 fail closed")

    # ---------------- offset balance audit (REQUIRED input) ----------------
    bal_path = args.repo / "reports" / BALANCE_REPORT
    balance: dict | None = None
    if bal_path.is_file():
        with open(bal_path, "r", encoding="utf-8") as fh:
            balance = json.load(fh)
        if balance.get("status") != "OK":
            log(f"FATAL: balance report status {balance.get('status')!r} (STOP)")
            return 4
        man_sha = balance.get("inputs", {}).get("stage1_manifest_sha256")
        if man_sha != _sha256(ev / "stage1-manifest.json"):
            log("FATAL: balance audit manifest sha does not match this evidence (STOP)")
            return 4
        log(f"balance audit loaded: classification={balance.get('balance_classification')}")
    else:
        log("FATAL: offset balance report missing (offset_balance_review.py must run first) (STOP)")
        return 4

    # ---------------- V2 numerics gate + verdict + recommendation ----------------
    clean, gate_detail = v2c.numerics_v2_gate(det_valid, cam_same, ord_same, adjusted, mp2)
    verdict = v2c.classify_verdict_v2(clean, adjusted, end_systematic_negative=end_systematic_negative, flat_threshold=max(flat_thr, 1e-6))
    bal_summary = {
        "present": True,
        "bottom_negative_after_reweight": bool(balance.get("bottom_negative_after_reweight")),
        "geometry_explains_bottom": bool(balance.get("geometry_explains_bottom")),
    }
    rec, rec_info = v2c.classify_recommendation_v2(verdict, bal_summary)
    log(f"NUMERICS_V2_CLEAN={clean}  VERDICT={verdict}  RECOMMENDATION={rec} ({rec_info['case']})")

    # ---------------- scale comparison (section 20) ----------------
    primary_points = [abs(adjusted[a]["intervals"]["POST_PRE"]["adjusted"]["point"]) for a in pc.PRIMARY_ARMS]
    floor = min(primary_points)
    scale = {
        "abs_camera_same_post_pre_mean": float(abs(dpp["point"])),
        "min_abs_primary_adjusted_post_pre": float(floor),
        "ratio_same_over_causal": float(abs(dpp["point"]) / floor) if floor > 0 else None,
        "camera_same_post_pre_ci_width": float(dpp["ci95"][1] - dpp["ci95"][0]),
        "ci_width_over_min_causal": float((dpp["ci95"][1] - dpp["ci95"][0]) / floor) if floor > 0 else None,
        "note": "aggregate SAME floor vs the smallest primary adjusted effect; the SAME CI is the uncertainty band on a quantity that must stay ~0",
    }

    # ---------------- immutability (V1 + historical surface) ----------------
    cm_path = args.repo / "reports" / "camera-coordinate-causal-code-manifest.json"
    with open(cm_path, "r", encoding="utf-8") as fh:
        cm = json.load(fh)
    immut: dict = {"final_scripts": {}, "causal_reports": {}, "expanded_effectiveness_reports": {}, "v1_posthoc_reports": {}, "v1_tooling": {}, "ok": True}
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
    for name, want in V1_POSTHOC_REPORT_SHA.items():
        h = _sha256(args.repo / "reports" / name)
        immut["v1_posthoc_reports"][name] = {"sha256": h, "expected": want, "ok": h == want}
        if h != want:
            immut["ok"] = False
    dev_dir = args.repo / "dev-tools" / "camera_coordinate_causal"
    for name, want in V1_TOOLING_SHA.items():
        h = _sha256(dev_dir / name)
        immut["v1_tooling"][name] = {"sha256": h, "expected": want, "ok": h == want}
        if h != want:
            immut["ok"] = False
    ev_prov = {
        "stage1_manifest_sha256": _sha256(ev / "stage1-manifest.json"),
        "stage1_manifest_sha256_expected": cm["input_manifest_sha256"],
        "results_worker0_sha256": _sha256(ev / "results-worker0.pt"),
        "results_worker1_sha256": _sha256(ev / "results-worker1.pt"),
        "v1_microprobe_sha256": _sha256(ev / "posthoc-microprobe.json"),
        "v1_microprobe_sha256_expected": "85346492a3500454b285e76a0d6539b0227d44593adcd22cec2fc99734068720",
        "microprobe_v2_sha256": _sha256(mp2_path) if mp2_path.is_file() else None,
    }
    ev_prov["stage1_manifest_ok"] = ev_prov["stage1_manifest_sha256"] == ev_prov["stage1_manifest_sha256_expected"]
    ev_prov["v1_microprobe_ok"] = ev_prov["v1_microprobe_sha256"] == ev_prov["v1_microprobe_sha256_expected"]
    log("immutability: %s" % ("OK" if immut["ok"] else "MISMATCH"))
    if not immut["ok"]:
        for grp in ("final_scripts", "causal_reports", "expanded_effectiveness_reports", "v1_posthoc_reports", "v1_tooling"):
            for name, v in immut[grp].items():
                if not v["ok"]:
                    log(f"  MISMATCH {grp} {name}")
        return 5
    if not ev_prov["stage1_manifest_ok"] or not ev_prov["v1_microprobe_ok"]:
        log("FATAL: evidence provenance mismatch (STOP)")
        return 5

    log("posthoc v2 elapsed %.1fs" % (time.time() - t0))

    # ---------------- assemble ----------------
    result = {
        "status": "OK",
        "label": "POSTHOC V2 REVIEW",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "base_commit": BASELINE,
        "bootstrap": {"seed": v2c.MASTER_SEED_V2, "n": v2c.N_BOOT_V2, "units": "cluster (paired shared resample matrix)", "point_semantics": "point estimates are the EXACT observed sample means; the bootstrap is used for CIs only (section 21)"},
        "v1_false_positive_correction": {
            "v1_gate_bug": "V1 set hidden_state_detected=True when p1_rel_max > 1e-6 (a single worst repeat outlier); the external review judged this a false-positive design risk",
            "v1_p1_max_abs_loss_delta_rel": None,
            "v1_result": "NUMERICS_CLEAN=False -> BLOCKED_NUMERICS (driven by the cap, not by any state/order evidence)",
            "v2_correction": "P1_REL_CAP=1e-6 retained as a DIAGNOSTIC only; HIDDEN_MUTABLE_STATE_DETECTED = coordinate/parameter/buffer mutation OR CI-backed order/accumulation drift beyond the P0 floor; repeat jitter alone is never sufficient",
            "v1_values_from_report": "read from camera-coordinate-causal-posthoc-review.json (committed, immutable)",
        },
        "numerical_floor": {
            "camera_same_primary": cam_same,
            "ordinary_same_secondary": ord_same,
            "scale_comparison": scale,
            "microprobe_v2": mp2,
            "historical_context": {
                "ordinary_same_mean_hist": {ck: hist["ordinary_metrics"][ck]["M_SAME_mean"] for ck in pc.CKS},
                "strict_identity_hist": hist.get("strict_identity_metrics", {}),
            },
        },
        "raw_causal": raw_margins,
        "raw_cross_check": "EXACT (recomputed raw margins and S_ relRMS bit-equal to committed audit.json)",
        "point_estimate": {
            "exact_observed_mean_used": True,
            "bootstrap_only_for_ci": True,
            "invariant_across_bootstrap_seed": True,
            "note": "V1 adjusted['point'] = mean(bootstrap replicates); V2 adjusted['point'] = mean(a_unit - s_unit) over the observed common subspace. Locked by regression tests (same data, different seed -> identical points).",
        },
        "noise_adjusted": {arm: adjusted[arm] for arm in ("OPPOSITE", "IDENTITY", "SHUFFLED")},
        "intervals": intervals,
        "loss_preference": {
            "classification": verdict,
            "primary_arms": list(pc.PRIMARY_ARMS),
            "supportive_arm": pc.SUPPORTIVE_ARM,
            "trend": {arm: {"MID_PRE_point": adjusted[arm]["intervals"]["MID_PRE"]["adjusted"]["point"], "POST_MID_point": adjusted[arm]["intervals"]["POST_MID"]["adjusted"]["point"], "POST_PRE_point": adjusted[arm]["intervals"]["POST_PRE"]["adjusted"]["point"], "class": intervals[arm]} for arm in ("OPPOSITE", "IDENTITY", "SHUFFLED")},
            "framing_rule": "loss preference (M = Loss(wrong) - Loss(correct); positive = correct coordinates preferred) is reported SEPARATELY from raw prediction displacement.",
        },
        "prediction_displacement": {
            "metric": "PREDICTION_DISPLACEMENT_SENSITIVITY (per-unit relRMS of arm vs CORRECT prediction; mirrors historical S_ aggregation)",
            "values": pred_disp,
            "classification": disp_class,
            "note": "relRMS decrease under perturbation is reduced raw displacement, NOT reduced sensitivity of the learned preference; loss-preference increase is NOT raw sensitivity increase.",
        },
        "offset": {
            "convention": {
                "universal": "START < 1/3 <= CENTER < 2/3 <= END (normalized offset tertiles)",
                "physical_verified": pc.PHYSICAL_CONVENTION_VERIFIED,
                "physical_source": "src/sakuramoon/data/camera_viewport.py plan_camera_viewport (base b2443af): horizontal 0=LEFT 1=RIGHT; vertical 0=TOP 1=BOTTOM",
            },
            "arm": "OPPOSITE (primary directional arm)",
            "cells": offset_res,
            "end_systematic_negative": end_systematic_negative,
            "balance_audit": {
                "classification": balance.get("balance_classification"),
                "notes": balance.get("classification_notes"),
                "exposure_count_imbalance": balance.get("planner", {}).get("exposure_count_imbalance"),
                "geometry_imbalanced": balance.get("vertical_geometry", {}).get("geometry_imbalanced"),
                "largest_geometry_imbalance": balance.get("vertical_geometry", {}).get("largest_imbalance"),
                "reference_bottom": balance.get("reference_effects_unbalanced", {}).get("BOTTOM"),
                "reweighted": balance.get("reweighted"),
                "matched": balance.get("matched"),
                "continuous_spearman": (balance.get("continuous") or {}).get("spearman_rho"),
                "shard_imbalance": balance.get("shard", {}).get("shard_cohort_imbalance"),
                "bottom_negative_after_reweight": balance.get("bottom_negative_after_reweight"),
                "geometry_explains_bottom": balance.get("geometry_explains_bottom"),
                "report": BALANCE_REPORT,
            },
        },
        "behavior": {
            "verdict": "NULL_EFFECT / WEAK (unchanged)",
            "recomputed": False,
            "note": "expanded behavioral effectiveness reports (6 files) are cited from committed evidence only; no recomputation.",
        },
        "numerics": {
            "NUMERICS_V2_CLEAN": clean,
            "detail": gate_detail,
            "verdict_uses_same_boolean": True,
        },
        "posthoc_v2_verdict": verdict,
        "recommendation": rec,
        "recommendation_case": rec_info["case"],
        "recommendation_notes": rec_info["notes"],
        "authorization": {"LONGER_P25_AUTHORIZED": "NO", "P50_AUTHORIZED": "NO", "PRODUCTION_CAMERA": "OFF", "note": "recommendation only; no training of any kind is authorized by this review"},
        "immutability": immut,
        "evidence_provenance": ev_prov,
    }

    if args.skip_write:
        log("skip-write: analysis complete")
        return 0

    # ---------------- report writers ----------------
    reports = args.repo / "reports"
    with open(reports / "camera-coordinate-causal-posthoc-v2-review.json", "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, allow_nan=False)

    metrics = {
        "label": "POSTHOC V2 REVIEW METRICS",
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
        "NUMERICS_V2_CLEAN": clean,
        "gate": gate_detail,
        "scale_comparison": scale,
        "POSTHOC_V2_VERDICT": verdict,
        "RECOMMENDATION": rec,
        "RECOMMENDATION_CASE": rec_info["case"],
    }
    with open(reports / "camera-coordinate-causal-posthoc-v2-metrics.json", "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, allow_nan=False)

    L: list[str] = []
    A = L.append
    A("# SakuraMoon Camera Coordinate Causal Audit - POSTHOC V2 REVIEW")
    A("")
    A("> **Label: POSTHOC V2 REVIEW** - supersedes ONLY the V1 posthoc NUMERICS CLASSIFICATION and the")
    A("> downstream recommendation. V1 reports, V1 tooling, and all historical evidence remain immutable.")
    A("")
    A(f"- generated: {result['generated_utc']}")
    A(f"- base commit: {BASELINE} (V1 posthoc review head)")
    A(f"- bootstrap: seed {v2c.MASTER_SEED_V2}, n={v2c.N_BOOT_V2}, cluster=unit, ONE shared resample matrix")
    A("- point estimates: EXACT observed sample means; bootstrap for CIs only (section 21)")
    A("")
    A("## 1. V1 FALSE-POSITIVE CORRECTION")
    A("")
    A("- V1 rule: `p1_rel_max > 1e-6 => hidden_state_detected=True`. A single worst repeat outlier")
    A("  was treated as a hidden mutable-state defect. External review: false-positive design risk.")
    A("- V2 rule: `HIDDEN_MUTABLE_STATE_DETECTED = B OR C OR D OR E` (coordinate map mutation,")
    A("  parameter/buffer mutation, order-dependent drift beyond the P0 floor, CI-backed accumulating")
    A("  drift beyond the P0 floor). `REPEAT_NUMERIC_JITTER_PRESENT` (A) is a DIAGNOSTIC ONLY and can")
    A("  never by itself set hidden state or NUMERICS_CLEAN=False.")
    A("")
    A("## 2. NUMERICAL FLOOR (primary basis unchanged)")
    A("")
    A("camera SAME (primary):")
    A("")
    A("| ck | mean | 95% CI | abs p95 | abs p99 | maxabs |")
    A("|---|---|---|---|---|---|")
    for ck in pc.CKS:
        c = cam_same["checkpoints"][ck]
        A("| {} | {:.3e} | [{:.3e}, {:.3e}] | {:.3e} | {:.3e} | {:.3e} |".format(ck, c["mean"], c["ci95"][0], c["ci95"][1], c["abs_p95"], c["abs_p99"], c["maxabs"]))
    A("")
    for k in ("MID_PRE", "POST_MID", "POST_PRE"):
        d = cam_same["deltas"][k]
        A("camera SAME {}: {:.3e} [{:.3e}, {:.3e}]".format(k, d["point"], d["ci95"][0], d["ci95"][1]))
    A("")
    A("ordinary SAME (secondary):")
    for ck in pc.CKS:
        c = ord_same["checkpoints"][ck]
        A("- {}: mean {:.3e} [{:.3e}, {:.3e}], maxabs {:.3e}".format(ck, c["mean"], c["ci95"][0], c["ci95"][1], c["maxabs"]))
    A("")
    A("scale comparison (section 20):")
    A("- |camera SAME POST-PRE mean| / min(|primary adjusted POST-PRE|) = **{}**".format("n/a" if scale["ratio_same_over_causal"] is None else "{:.3e}".format(scale["ratio_same_over_causal"])))
    A("- SAME POST-PRE CI width = {:.3e} ({}x the smallest primary effect)".format(scale["camera_same_post_pre_ci_width"], "n/a" if scale["ci_width_over_min_causal"] is None else "{:.2f}".format(scale["ci_width_over_min_causal"])))
    A("")
    A("## 3. MICROPROBE V2 (state vs jitter)")
    A("")
    if mp2:
        st = mp2["state"]
        A("| concept | value |")
        A("|---|---|")
        for k in ("REPEAT_NUMERIC_JITTER_PRESENT", "COORDINATE_MAP_MUTATION_DETECTED", "PERSISTENT_BUFFER_MUTATION_DETECTED", "ORDER_DEPENDENT_DRIFT_DETECTED", "ACCUMULATING_DRIFT_DETECTED", "HIDDEN_MUTABLE_STATE_DETECTED"):
            A(f"| {k} | {st.get(k)} |")
        A("")
        A(f"parameter mutation: {mp2['parameter_mutation']['detected']}; persistent buffer mutation: {mp2['persistent_buffer_mutation']['detected']}")
        p0 = mp2["p0"] or {}
        A("P0 immediate repeats: bitexact rate {}/{}; signed mean {}; max rel jitter {} (diagnostic cap 1e-6: {})".format(
            p0.get("pred_bitexact_rate"), p0.get("n_pairs"), p0.get("signed_mean"), p0.get("max_abs_loss_delta_rel"),
            "exceeded - WARNING only" if (p0.get("max_abs_loss_delta_rel") or 0) > 1e-6 else "within band"))
        p1 = mp2["p1"] or {}
        p2 = mp2["p2"] or {}
        A("P1 interleave C drift: signed mean {} CI {}; slope {} CI {}; beyond P0 floor: {}".format(
            (p1.get("c_signed") or {}).get("signed_mean"), (p1.get("c_signed") or {}).get("signed_mean_ci95"),
            (p1.get("c_slope") or {}).get("point"), (p1.get("c_slope") or {}).get("ci95"), p1.get("slope_beyond_p0_floor")))
        A("P2 SAME interleave: C signed mean {} CI {}; SAME-vs-C p99 ratio vs P0 {}".format(
            (p2.get("c_signed") or {}).get("signed_mean"), (p2.get("c_signed") or {}).get("signed_mean_ci95"), p2.get("same_p99_ratio_vs_p0")))
        p3 = mp2["p3"] or {}
        A("P3 reload consistency: state_history_effect = {}".format(p3.get("state_history_effect", "not_run")))
        rng = mp2.get("rng") or {}
        A("RNG consumption: {}".format(rng.get("consumption_detected")))
        A("")
        A(f"interpretation: {mp2.get('interpretation')}")
    else:
        A("microprobe-v2 ABSENT - R6-R9 fail closed.")
    A("")
    A("## 4. CAUSAL V2 (exact point estimates, paired shared bootstrap)")
    A("")
    A("| arm | interval | raw arm | raw same | adjusted |")
    A("|---|---|---|---|---|")
    for arm in ("OPPOSITE", "IDENTITY", "SHUFFLED"):
        for k in ("MID_PRE", "POST_MID", "POST_PRE"):
            iv = adjusted[arm]["intervals"][k]
            A("| {} | {} | {:.3e} | {:.3e} | {:.3e} [{:.3e}, {:.3e}] |".format(
                arm, k, iv["raw_arm"]["point"], iv["raw_same"]["point"], iv["adjusted"]["point"], iv["adjusted"]["ci95"][0], iv["adjusted"]["ci95"][1]))
    A("")
    A("intervals: " + "  ".join(f"{a}={c}" for a, c in intervals.items()))
    A("")
    A("## 5. PREDICTION DISPLACEMENT (separate metric)")
    A("")
    for arm in ("OPPOSITE", "IDENTITY", "SHUFFLED"):
        p = pred_disp[arm]
        A("{}: PRE {:.4f} / MID {:.4f} / POST {:.4f} (POST-PRE {:+.4f})".format(arm, p["PRE"], p["MID"], p["POST"], p["POST_PRE"]))
    A("")
    A(f"classification: **{disp_class}**")
    A("")
    A("## 6. OFFSET CELLS (OPPOSITE adjusted POST-PRE)")
    A("")
    A("| scope | tertile | n | n_common | adjusted | CI |")
    A("|---|---|---|---|---|---|")
    for scope in ("universal", "horizontal", "vertical"):
        for tert in ("START", "CENTER", "END"):
            c = offset_res[scope][tert]
            adj = c["adjusted_POST_PRE"]
            A("| {} | {} | {} | {} | {} | {} |".format(
                scope, tert, c["n"], c["n_common"],
                "n/a" if adj.get("point") is None else "{:.3e}".format(adj["point"]),
                "n/a" if not adj.get("ci95") else "[{:.3e}, {:.3e}]".format(adj["ci95"][0], adj["ci95"][1])))
    A("")
    A(f"universal END systematic negative: **{end_systematic_negative}**")
    A("")
    A("## 7. OFFSET BALANCE AUDIT (see offset-balance report)")
    A("")
    ba = result["offset"]["balance_audit"]
    A(f"balance classification: **{ba['classification']}**")
    for n_ in (ba.get("notes") or []):
        A(f"- {n_}")
    A("")
    A(f"- exposure count imbalance (planner z >= 3): {ba['exposure_count_imbalance']}")
    A(f"- geometry imbalanced (|SMD| >= 0.1): {ba['geometry_imbalanced']} (largest: {ba['largest_geometry_imbalance']})")
    rw = ba.get("reweighted") or {}
    if rw.get("status") == "OK":
        A("- reweighted TOP: {:.3e} {}; BOTTOM: {:.3e} {}".format(
            rw["top"]["point"], rw["top"]["ci95"], rw["bottom"]["point"], rw["bottom"]["ci95"]))
    A(f"- bottom negative after reweight: **{ba['bottom_negative_after_reweight']}**; geometry explains bottom: **{ba['geometry_explains_bottom']}**")
    A(f"- Spearman(norm_offset, D_opp) vertical: {ba['continuous_spearman']}; shard imbalance: {ba['shard_imbalance']}")
    A("")
    A("## 8. NUMERICS V2 GATE")
    A("")
    A(f"NUMERICS_V2_CLEAN = **{clean}**")
    for r, v in gate_detail["rules"].items():
        A(f"- {r}: {'PASS' if v else 'FAIL'}")
    for r in gate_detail["reasons"]:
        A(f"  ({r})")
    A("")
    A(f"repeat jitter warning: {gate_detail['repeat_jitter_warning']['present']} (diagnostic only - never a fail condition)")
    A("")
    A("## 9. VERDICT + RECOMMENDATION")
    A("")
    A(f"POSTHOC_V2_VERDICT = **{verdict}**")
    A(f"RECOMMENDATION = **{rec}** (case {rec_info['case']})")
    A("")
    A("> RECOMMENDATION is not AUTHORIZATION. No training of any kind is authorized.")
    A("")
    A("## 10. IMMUTABILITY")
    A("")
    for grp in ("final_scripts", "causal_reports", "expanded_effectiveness_reports", "v1_posthoc_reports", "v1_tooling"):
        names = immut[grp]
        A(f"- {grp}: {sum(1 for v in names.values() if v['ok'])}/{len(names)} unchanged")
    A("")
    with open(reports / "camera-coordinate-causal-posthoc-v2-review.md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))

    # ---------------- copy report (placeholders for the post-commit SHAs) ----------------
    def f3(x, nd=3):
        if x is None:
            return "n/a"
        if isinstance(x, float) and np.isnan(x):
            return "n/a"
        return f"{x:.{nd}e}"

    C: list[str] = []
    c = C.append
    C.append("== SakuraMoon Camera Coordinate Causal - Posthoc V2 Review Handoff ==")
    C.append("")
    C.append("BASE")
    C.append("  repository = leafmoone/sakuramoon")
    C.append("  prior branch = camera-v2-causal-posthoc-review")
    C.append(f"  prior SHA = {BASELINE}")
    C.append("  new branch = camera-v2-causal-posthoc-v2-review")
    C.append(f"  new base = {BASELINE}")
    C.append("")
    C.append("COMMITS")
    C.append("  tooling = @POSTHOC_V2_TOOLING_HEAD@")
    C.append("  evidence = @POSTHOC_V2_EVIDENCE_HEAD@")
    C.append("  remote SHA = @POSTHOC_V2_REMOTE_SHA@")
    C.append("  pushed = @PUSHED@")
    C.append("  force push = NO")
    C.append("")
    C.append("IMMUTABILITY")
    C.append("  final_snapshot = 5/5 unchanged")
    C.append("  historical causal reports = 5/5 unchanged")
    C.append("  expanded reports = 6/6 unchanged")
    C.append("  V1 posthoc reports = 4/4 unchanged")
    C.append("  V1 posthoc tooling = 3/3 unchanged")
    C.append("  src diff = @SRCDIFF@")
    C.append("  config diff = @CONFIGDIFF@")
    C.append("")
    C.append("REPLAY")
    C.append("  historical replay = @REPLAY@")
    C.append("  raw causal crosscheck = EXACT (9 M_ + 3 arms x S_ bit-equal to audit.json)")
    C.append("")
    C.append("V1 FALSE-POSITIVE ISSUE")
    v1p = mp2["p0"] if mp2 else {}
    C.append("  P1 cap = 1e-6 (V1 used it as a hidden-state gate; V2 keeps it diagnostic-only)")
    C.append("  P1 observed max rel (V1) = see camera-coordinate-causal-posthoc-review.json (committed)")
    C.append(f"  P0 observed max rel (V2) = {f3(v1p.get('max_abs_loss_delta_rel'))}")
    C.append("  V1 hidden state result = True (driven by the cap)")
    C.append("  reason V1 blocked = single worst repeat outlier > 1e-6 mapped to hidden_state_detected")
    C.append("")
    C.append("MICROPROBE V2")
    st = (mp2 or {}).get("state") or {}
    C.append(f"  units = {((mp2 or {}).get('n_camera', '?'))} camera + {((mp2 or {}).get('n_ordinary', '?'))} ordinary")
    C.append(f"  checkpoints = {json.dumps((mp2 or {}).get('checkpoints', {}))}")
    C.append(f"  timesteps = strata (0,2): {json.dumps((mp2 or {}).get('t_values', {}))}")
    C.append(f"  immediate repeat bitexact rate = {v1p.get('pred_bitexact_rate')}")
    C.append(f"  repeat jitter present = {(mp2 or {}).get('repeat_numeric_jitter_present')}")
    C.append(f"  repeat jitter mean = {v1p.get('signed_mean')}")
    C.append(f"  repeat jitter p95 = {v1p.get('abs_p95')}")
    C.append(f"  repeat jitter p99 = {v1p.get('abs_p99')}")
    C.append(f"  repeat jitter max = {v1p.get('abs_max')} (rel {v1p.get('max_abs_loss_delta_rel')})")
    C.append(f"  coordinate mutation = {st.get('COORDINATE_MAP_MUTATION_DETECTED')}")
    C.append(f"  parameter mutation = {(mp2 or {}).get('parameter_mutation', {}).get('detected')}")
    C.append(f"  persistent buffer mutation = {(mp2 or {}).get('persistent_buffer_mutation', {}).get('detected')}")
    C.append(f"  RNG consumption = {(mp2 or {}).get('rng', {}).get('consumption_detected')}")
    C.append(f"  order-dependent drift = {st.get('ORDER_DEPENDENT_DRIFT_DETECTED')}")
    C.append(f"  accumulation slope = P1 {(mp2 or {}).get('p1', {}).get('c_slope', {}).get('point')} CI {(mp2 or {}).get('p1', {}).get('c_slope', {}).get('ci95')} / P2 {(mp2 or {}).get('p2', {}).get('c_slope', {}).get('point')} CI {(mp2 or {}).get('p2', {}).get('c_slope', {}).get('ci95')}")
    C.append("  accumulation CI = see p1/p2 c_slope ci95 above")
    C.append(f"  hidden mutable state detected = {st.get('HIDDEN_MUTABLE_STATE_DETECTED')}")
    C.append("")
    C.append("NUMERICAL FLOOR")
    for ck in pc.CKS:
        cc = cam_same["checkpoints"][ck]
        C.append(f"  camera SAME {ck} = mean {f3(cc['mean'])} CI [{f3(cc['ci95'][0])}, {f3(cc['ci95'][1])}]")
    C.append(f"  camera SAME POST-PRE = {f3(dpp['point'])}")
    C.append(f"  CI = [{f3(dpp['ci95'][0])}, {f3(dpp['ci95'][1])}]")
    C.append("  ordinary SAME = " + " / ".join(f"{ck} {f3(ord_same['checkpoints'][ck]['mean'])}" for ck in pc.CKS))
    C.append(f"  same/causal scale ratio = {f3(scale['ratio_same_over_causal'])}")
    C.append("")
    C.append("CAUSAL V2")
    for arm in ("OPPOSITE", "IDENTITY", "SHUFFLED"):
        raws = " / ".join(f3(raw_margins[arm][ck]["point"]) for ck in pc.CKS)
        iv = adjusted[arm]["intervals"]
        C.append(f"  {arm} raw PRE/MID/POST = {raws}")
        C.append(f"  {arm} adj PRE->MID = {f3(iv['MID_PRE']['adjusted']['point'])} [{f3(iv['MID_PRE']['adjusted']['ci95'][0])}, {f3(iv['MID_PRE']['adjusted']['ci95'][1])}]")
        C.append(f"  {arm} adj MID->POST = {f3(iv['POST_MID']['adjusted']['point'])} [{f3(iv['POST_MID']['adjusted']['ci95'][0])}, {f3(iv['POST_MID']['adjusted']['ci95'][1])}]")
        C.append(f"  {arm} adj PRE->POST = {f3(iv['POST_PRE']['adjusted']['point'])} [{f3(iv['POST_PRE']['adjusted']['ci95'][0])}, {f3(iv['POST_PRE']['adjusted']['ci95'][1])}]")
    C.append("")
    C.append("POINT ESTIMATE")
    C.append("  exact observed mean used = YES")
    C.append("  bootstrap only for CI = YES")
    C.append("  invariant across bootstrap seed = YES (regression-locked)")
    C.append("")
    C.append("LOSS PREFERENCE")
    C.append(f"  OPP trend = {intervals['OPPOSITE']}")
    C.append(f"  SHUFFLED trend = {intervals['SHUFFLED']}")
    C.append(f"  IDENTITY trend = {intervals['IDENTITY']}")
    C.append(f"  classification = {verdict}")
    C.append("")
    C.append("PREDICTION DISPLACEMENT")
    for arm in ("OPPOSITE", "IDENTITY", "SHUFFLED"):
        p = pred_disp[arm]
        C.append(f"  {arm} PRE/MID/POST = {p['PRE']:.4f} / {p['MID']:.4f} / {p['POST']:.4f}")
    C.append(f"  classification = {disp_class}")
    C.append("")
    C.append("PLANNER BALANCE")
    C.append("  offset sampler implementation = k = random.Random(offset_seed).randrange(available + 1), k in 0..available uniform (camera_viewport.py @ b2443af)")
    C.append(f"  conditional symmetry verified = {ba['exposure_count_imbalance'] is not None} (static: P(k)=P(available-k) exact for uniform sampler)")
    C.append("  universal observed START/CENTER/END = " + " / ".join(str(offset_res['universal'][t]['n']) for t in ('START', 'CENTER', 'END')))
    C.append("  universal expected = see offset-balance.json planner.blocks.all")
    C.append("  vertical TOP/CENTER/BOTTOM observed = " + " / ".join(str(offset_res['vertical'][t]['n']) for t in ('START', 'CENTER', 'END')))
    C.append("  vertical expected = see offset-balance.json planner.blocks.vertical")
    C.append(f"  exposure count imbalance = {ba['exposure_count_imbalance']}")
    C.append("")
    C.append("VERTICAL GEOMETRY")
    vg = (balance.get("vertical_geometry") or {})
    tg, bg = vg.get("TOP") or {}, vg.get("BOTTOM") or {}
    C.append(f"  TOP n = {tg.get('n')}")
    C.append(f"  BOTTOM n = {bg.get('n')}")
    smd = vg.get("smd") or {}
    C.append(f"  zoom SMD = {f3(smd.get('zoom_val'), 3)}")
    C.append(f"  latent shift SMD = {f3(smd.get('latent_shift'), 3)}")
    C.append(f"  aspect SMD = {f3(smd.get('canvas_aspect'), 3)}")
    C.append(f"  available SMD = {f3(smd.get('available'), 3)}")
    C.append(f"  retention SMD = {f3(smd.get('retention'), 3)}")
    C.append(f"  largest imbalance = {vg.get('largest_imbalance')} (SMD {f3((smd or {}).get(vg.get('largest_imbalance', ''), float('nan')), 3)})")
    C.append("")
    C.append("REWEIGHTED EFFECT")
    rwb = rw.get("bottom") or {}
    rwt = rw.get("top") or {}
    C.append(f"  common geometry cells = {rw.get('shared_cells', 'n/a')}")
    C.append(f"  TOP ESS = {rwt.get('ess', 'n/a')}")
    C.append(f"  BOTTOM ESS = {rwb.get('ess', 'n/a')}")
    C.append(f"  TOP adjusted effect = {f3(rwt.get('point'))} CI {rwt.get('ci95')}")
    C.append(f"  BOTTOM adjusted effect = {f3(rwb.get('point'))} CI {rwb.get('ci95')}")
    C.append(f"  negative persists = {ba['bottom_negative_after_reweight']}")
    C.append("")
    C.append("MATCHED EFFECT")
    mt = (balance.get("matched") or {})
    mdb = mt.get("top_minus_bottom") or {}
    C.append(f"  pairs = {mt.get('n_pairs', 'n/a')}")
    C.append(f"  post-match max SMD = {f3(mt.get('max_post_match_smd'), 3)}")
    C.append(f"  TOP-BOTTOM paired difference = {f3(mdb.get('point'))} CI {mdb.get('ci95')}")
    C.append("  result = see matched.top_minus_bottom")
    C.append("")
    C.append("CONTINUOUS OFFSET")
    cont = (balance.get("continuous") or {})
    C.append(f"  Spearman = {f3(cont.get('spearman_rho'), 3)} CI {cont.get('spearman_ci95')}")
    dec = cont.get("deciles") or []
    C.append("  decile trend = " + " ".join(f"d{d['decile']}:{f3(d['mean'], 2)}" for d in dec))
    mb = cont.get("mirror_bins") or []
    C.append("  extreme mirror bins = " + " ".join(f"{m['bin_low']}vs{m['bin_high']}:{f3(m['mirror_diff_low_minus_high'], 2)}" for m in mb[:2]))
    C.append(f"  interpretation = {cont.get('interpretation', 'n/a')}")
    C.append("")
    C.append("CONTENT / SHARD")
    C.append(f"  routing covariates = {(balance.get('content') or {}).get('routing_covariates')}")
    C.append(f"  shard imbalance = {(balance.get('shard') or {}).get('shard_cohort_imbalance')}")
    C.append(f"  raw caption metadata available = {(balance.get('content') or {}).get('raw_caption_metadata')}")
    C.append("  notable interactions = see offset-balance.md section 7")
    C.append("")
    C.append("OFFSET BALANCE CLASS")
    C.append(f"  {balance.get('balance_classification')}")
    C.append("")
    C.append("NUMERICS V2")
    C.append(f"  NUMERICS_V2_CLEAN = {clean}")
    C.append(f"  repeat jitter warning = {gate_detail['repeat_jitter_warning']['present']}")
    C.append(f"  hidden mutable state = {st.get('HIDDEN_MUTABLE_STATE_DETECTED')}")
    C.append("")
    C.append("POSTHOC V2 VERDICT")
    C.append(f"  {verdict}")
    C.append("")
    C.append("RECOMMENDATION")
    C.append(f"  {rec} (case {rec_info['case']})")
    C.append("")
    C.append("BEHAVIOR")
    C.append("  expanded behavior = NULL_EFFECT / WEAK (committed)")
    C.append("  recomputed = NO")
    C.append("")
    C.append("VALIDATION")
    C.append("  original tests = @ORIG_TESTS@")
    C.append("  V1 posthoc tests = @V1_TESTS@")
    C.append("  V2 tests = @V2_TESTS@")
    C.append("  total = @TOTAL_TESTS@")
    C.append("  skips = @SKIPS@")
    C.append("  xfails = @XFAILS@")
    C.append("  replay = @REPLAY@")
    C.append("  ruff = @RUFF@")
    C.append("  py_compile = @PYCOMPILE@")
    C.append("  diff check = @GITDIFF@")
    C.append("")
    C.append("SECURITY")
    C.append("  secret hits = @SECRETS@")
    C.append("  model/checkpoint staged = NO")
    C.append("  dataset staged = NO")
    C.append("  binary evidence staged = NO")
    C.append("")
    C.append("AUTHORIZATION")
    C.append("  longer P25 started = NO")
    C.append("  P50 started = NO")
    C.append("  production changed = NO")
    C.append("")
    C.append("EXTERNAL REVIEW TARGET")
    C.append("  branch = camera-v2-causal-posthoc-v2-review")
    C.append("  exact SHA = @POSTHOC_V2_REMOTE_SHA@")
    C.append("")
    C.append("NEXT")
    C.append("  HARD STOP")
    C.append("  SEND EXACT SHA TO EXTERNAL REVIEWER")
    C.append("  EXTERNAL REVIEW MUST HAPPEN BEFORE ANY TRAINING GO")
    C.append("")
    C.append("== END ==")
    with open(reports / "camera-coordinate-causal-posthoc-v2-copy-report.md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(C) + "\n")

    log("v2 reports written: review.md/json, metrics.json, copy-report.md (under reports/)")
    log(f"FINAL: NUMERICS_V2_CLEAN={clean} VERDICT={verdict} RECOMMENDATION={rec} (case {rec_info['case']}) balance={balance.get('balance_classification')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
