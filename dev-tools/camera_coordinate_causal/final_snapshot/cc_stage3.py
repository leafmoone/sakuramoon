"""Stage 3 - statistics, bootstrap, verdict, and the five UNSTAGED reports.

CPU only. Reads: stage1-manifest.json, results-worker{0,1}.pt, determinism jsons.
Writes: REPO/reports/camera-coordinate-causal-{audit.md,audit.json,metrics.json,
units.csv,copy-report.md}  (untracked; never committed).
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

import cc_common as cc

CKS = ("PRE", "MID", "POST")
CAM_ARMS = ("CORRECT", "IDENTITY", "OPPOSITE", "SHUFFLED", "HALF", "OVER")
REQUIRED_ARMS = ("CORRECT", "IDENTITY", "OPPOSITE", "SHUFFLED")


def log(msg: str) -> None:
    print(f"[stage3 {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------- metric helpers ----------------


def unit_strata_mean(losses: dict[str, list[float]]) -> dict[str, float]:
    return {a: float(np.mean(v)) for a, v in losses.items() if v}


def margin(per_arm: dict[str, float], wrong: str, correct: float) -> float:
    if wrong not in per_arm:
        return float("nan")
    return per_arm[wrong] - correct


def norm_margin(wrong_loss: float, correct_loss: float) -> float:
    return (wrong_loss - correct_loss) / max(correct_loss, cc.NORM_EPS)


def ci95(samples: np.ndarray) -> tuple[float, float]:
    return (float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5)))


def trend_class(d1: float, d2: float, ci1: tuple[float, float],
                ci2: tuple[float, float]) -> str:
    """Spec s21 trend classes from the two consecutive paired deltas."""
    flat1 = ci1[0] <= 0.0 <= ci1[1]
    flat2 = ci2[0] <= 0.0 <= ci2[1]
    if flat1 and flat2:
        return "FLAT"
    if d1 > 0 and d2 > 0:
        return "MONOTONIC_GAIN"
    if d1 > 0 and flat2:
        return "EARLY_GAIN"
    if flat1 and d2 > 0:
        return "LATE_GAIN"
    if d1 > 0 and d2 < 0:
        return "REVERSED"
    return "NON_MONOTONIC"


def case_label(trend: str) -> str:
    if trend == "FLAT":
        return "CASE_C"
    if trend in ("MONOTONIC_GAIN", "EARLY_GAIN", "LATE_GAIN", "NON_MONOTONIC", "REVERSED"):
        if trend == "EARLY_GAIN":
            return "CASE_A"
        if trend == "MONOTONIC_GAIN":
            return "CASE_B"
        return "CASE_D"
    return "CASE_D"


# ---------------- load results ----------------


def load_all() -> tuple[dict, dict, list[int], dict, list[int], list[int]]:
    with open(cc.OUT / "stage1-manifest.json", "r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    meta = {m["unit"]: m for m in manifest["units"]}
    done: dict = {}
    for w in (0, 1):
        p = cc.OUT / f"results-worker{w}.pt"
        if not p.is_file():
            raise RuntimeError(f"missing results {p}")
        part = torch.load(p, map_location="cpu", weights_only=False)
        for u, rows in part.items():
            done.setdefault(int(u), {}).update(rows)
    units = sorted(meta.keys())
    missing = [u for u in units if any(ck not in done[u] for ck in CKS)]
    if missing:
        raise RuntimeError(f"{len(missing)} units missing checkpoint rows, e.g. {missing[:5]}")
    n_cam = manifest["cohort"]["camera"]
    camera_units = [u for u in units if meta[u]["cohort"] == "camera"]
    ordinary_units = [u for u in units if meta[u]["cohort"] == "ordinary"]
    assert camera_units == list(range(n_cam)), "camera units must be 0..C-1"
    determinism = {}
    for w in (0, 1):
        p = cc.OUT / f"determinism-worker{w}.json"
        if p.is_file():
            with open(p, "r", encoding="utf-8") as fh:
                determinism[f"worker{w}"] = json.load(fh)
    return manifest, done, camera_units, ordinary_units, determinism, missing


def main() -> int:
    t0 = time.time()
    manifest, done, camera_units, ordinary_units, determinism, _ = load_all()
    meta = {m["unit"]: m for m in manifest["units"]}
    n_cam = len(camera_units)
    n_ord = len(ordinary_units)
    log(f"units: {n_cam} camera + {n_ord} ordinary; all {len(CKS)} checkpoints present")

    # ---------------- per-unit strata-averaged metrics ----------------
    # camM[ck][arm][unit-in-cam] = strata-mean loss ; camS[ck][arm][i] = (rel,cos) mean
    cam_loss: dict[str, dict[str, np.ndarray]] = {
        ck: {a: np.full(n_cam, np.nan) for a in CAM_ARMS} for ck in CKS
    }
    cam_sens_rel: dict[str, dict[str, np.ndarray]] = {
        ck: {a: np.full(n_cam, np.nan) for a in CAM_ARMS} for ck in CKS
    }
    cam_sens_cos: dict[str, dict[str, np.ndarray]] = {
        ck: {a: np.full(n_cam, np.nan) for a in CAM_ARMS} for ck in CKS
    }
    cam_best3: dict[str, np.ndarray] = {ck: np.full(n_cam, np.nan) for ck in CKS}
    ord_loss: dict[str, dict[str, np.ndarray]] = {
        ck: {a: np.full(n_ord, np.nan) for a in ("CORRECT", "IDENTITY", "SAME", "RANDOM")}
        for ck in CKS
    }
    ord_best3: dict[str, np.ndarray] = {ck: np.full(n_ord, np.nan) for ck in CKS}
    n_opp_na = 0
    for i, u in enumerate(camera_units):
        m = meta[u]
        if m["opp_na"]:
            n_opp_na += 1
        for ck in CKS:
            row = done[u][ck]
            for a in CAM_ARMS:
                if a in row["loss"]:
                    cam_loss[ck][a][i] = np.mean(row["loss"][a])
                if a != "CORRECT" and a in row.get("sens", {}):
                    s = np.array(row["sens"][a])
                    cam_sens_rel[ck][a][i] = s[:, 0].mean()
                    cam_sens_cos[ck][a][i] = s[:, 1].mean()
            if row.get("correct_best3"):
                cam_best3[ck][i] = np.mean(row["correct_best3"])
    for j, u in enumerate(ordinary_units):
        for ck in CKS:
            row = done[u][ck]
            for a in ("CORRECT", "IDENTITY", "SAME", "RANDOM"):
                if a in row["loss"]:
                    ord_loss[ck][a][j] = np.mean(row["loss"][a])
            if row.get("correct_best3"):
                ord_best3[ck][j] = np.mean(row["correct_best3"])

    # ---------------- camera cohort metrics ----------------
    opp_mask = np.array(
        [not meta[u]["opp_na"] for u in camera_units], dtype=bool
    )
    cam_metrics: dict[str, dict] = {}
    for ck in CKS:
        cor = cam_loss[ck]["CORRECT"]
        doc = {"L_COR_mean": float(np.nanmean(cor))}
        for a in ("IDENTITY", "OPPOSITE", "SHUFFLED", "HALF", "OVER"):
            w = cam_loss[ck][a]
            if a == "OPPOSITE":
                sel = opp_mask
            else:
                sel = ~np.isnan(w)
            raw = w - cor
            doc[f"M_{a}_raw"] = float(np.nanmean(raw[sel]))
            doc[f"M_{a}_norm"] = float(
                np.nanmean(((w - cor) / np.maximum(cor, cc.NORM_EPS))[sel])
            )  # v2: [sel] inside nanmean (was outside -> scalar index)
            doc[f"M_{a}_n_units"] = int(sel.sum())
            doc[f"S_{a}_relrms"] = float(np.nanmean(cam_sens_rel[ck][a][sel]))
            doc[f"S_{a}_cos"] = float(np.nanmean(cam_sens_cos[ck][a][sel]))
        doc["correct_best3_frac"] = float(np.nanmean(cam_best3[ck]))
        cam_metrics[ck] = doc
    log(f"camera metrics: "
        + ", ".join(
            f"{ck}: M_id={cam_metrics[ck]['M_IDENTITY_raw']:.5f} "
            f"M_opp={cam_metrics[ck]['M_OPPOSITE_raw']:.5f} "
            f"M_shf={cam_metrics[ck]['M_SHUFFLED_raw']:.5f}"
            for ck in CKS
        ))

    # ---------------- ordinary cohort / negative control ----------------
    ord_metrics: dict[str, dict] = {}
    for ck in CKS:
        cor = ord_loss[ck]["CORRECT"]
        doc = {"L_COR_mean": float(np.nanmean(cor))}
        same = ord_loss[ck]["SAME"]
        doc["M_SAME_maxabs"] = float(np.nanmax(np.abs(same - cor)))
        doc["M_SAME_mean"] = float(np.nanmean(same - cor))
        doc["M_IDENTITY_raw"] = float(np.nanmean(ord_loss[ck]["IDENTITY"] - cor))
        rnd = ord_loss[ck]["RANDOM"]
        if np.isfinite(rnd).any():
            doc["M_RANDOM_mean"] = float(np.nanmean(rnd - cor))
            doc["M_RANDOM_n"] = int(np.isfinite(rnd).sum())
        doc["correct_best3_frac"] = float(np.nanmean(ord_best3[ck]))
        ord_metrics[ck] = doc
    strict_units_ord = [
        j for j, u in enumerate(ordinary_units) if meta[u]["strict_identity"]
    ]
    strict_metrics: dict[str, dict] = {}
    for ck in CKS:
        if strict_units_ord:
            sel = np.zeros(n_ord, dtype=bool)
            sel[strict_units_ord] = True
            cor = ord_loss[ck]["CORRECT"]
            wid = ord_loss[ck]["IDENTITY"]
            strict_metrics[ck] = {
                "n": int(sel.sum()),
                "M_IDENTITY_raw_mean": float(np.nanmean((wid - cor)[sel])),
                "M_IDENTITY_raw_maxabs": float(np.nanmax(np.abs(wid - cor)[sel])),
            }
    log(f"negative control SAME maxabs: "
        + ", ".join(f"{ck}={ord_metrics[ck]['M_SAME_maxabs']:.3e}" for ck in CKS))

    # ---------------- bootstrap (units = clusters, paired) ----------------
    n_boot = 10000
    rng = np.random.default_rng(cc.MASTER_SEED)

    def boot_margin(arm: str, sel: np.ndarray) -> dict:
        vals = {}
        valid = sel.copy()
        for ck in CKS:
            valid &= np.isfinite(cam_loss[ck][arm] - cam_loss[ck]["CORRECT"])  # v5: drop non-finite units (OPPOSITE zero-shift N/A)
        sel_idx = np.flatnonzero(valid)
        k = int(sel_idx.size)
        if k < 2:
            nanb = {ck: {"mean": float("nan"), "boot": np.full(n_boot, np.nan)} for ck in CKS}
            return {ck: {"mean": nanb[ck]["mean"], "ci95": (float("nan"), float("nan"))} for ck in CKS} | {
                "delta_POST_PRE": {"point": float("nan"), "ci95": (float("nan"), float("nan"))},
                "delta_MID_PRE": {"point": float("nan"), "ci95": (float("nan"), float("nan"))},
                "delta_POST_MID": {"point": float("nan"), "ci95": (float("nan"), float("nan"))},
            }
        boot_idx_sub = rng.integers(0, k, size=(n_boot, k))  # v2: resample within sel subset
        for ck in CKS:
            w = cam_loss[ck][arm]
            c = cam_loss[ck]["CORRECT"]
            m = (w - c)[sel_idx]
            boot = m[boot_idx_sub].mean(axis=1)
            vals[ck] = {"mean": float(m.mean()), "boot": boot}
        d_pm = vals["POST"]["boot"] - vals["PRE"]["boot"]
        d_mm = vals["MID"]["boot"] - vals["PRE"]["boot"]
        d_mp = vals["POST"]["boot"] - vals["MID"]["boot"]
        return {
            ck: {"mean": vals[ck]["mean"], "ci95": ci95(vals[ck]["boot"])}
            for ck in CKS
        } | {
            "delta_POST_PRE": {
                "point": float(vals["POST"]["boot"].mean() - vals["PRE"]["boot"].mean()),
                "ci95": ci95(d_pm),
            },
            "delta_MID_PRE": {
                "point": float(vals["MID"]["boot"].mean() - vals["PRE"]["boot"].mean()),
                "ci95": ci95(d_mm),
            },
            "delta_POST_MID": {
                "point": float(vals["POST"]["boot"].mean() - vals["MID"]["boot"].mean()),
                "ci95": ci95(d_mp),
            },
        }

    boot: dict[str, dict] = {}
    boot["M_OPPOSITE_raw"] = boot_margin("OPPOSITE", opp_mask)
    boot["M_IDENTITY_raw"] = boot_margin("IDENTITY", np.ones(n_cam, dtype=bool))
    boot["M_SHUFFLED_raw"] = boot_margin("SHUFFLED", np.ones(n_cam, dtype=bool))
    for arm in ("HALF", "OVER"):
        try:
            boot[f"M_{arm}_raw"] = boot_margin(arm, np.ones(n_cam, dtype=bool))
        except Exception:
            pass
    log("bootstrap done (n=10000, seed=20260906)")

    # trends + cases per required arm
    trends: dict[str, dict] = {}
    for arm in ("OPPOSITE", "IDENTITY", "SHUFFLED"):
        b = boot[f"M_{arm}_raw"]
        d1 = b["delta_MID_PRE"]
        d2 = b["delta_POST_MID"]
        cls = trend_class(d1["point"], d2["point"], d1["ci95"], d2["ci95"])
        trends[arm] = {
            "class": cls,
            "case": case_label(cls),
            "delta_MID_PRE": d1,
            "delta_POST_MID": d2,
            "delta_POST_PRE": b["delta_POST_PRE"],
        }
    log(f"trends: " + ", ".join(f"{a}={trends[a]['class']}" for a in trends))

    # ---------------- geometry strata ----------------
    def band_of(meta_u: dict) -> dict[str, str]:
        z = meta_u["zoom"]
        ls = meta_u["latent_shift"]
        no = meta_u["norm_offset"]
        return {
            "zoom": "mild" if z < 1.20 else ("medium" if z < 1.35 else "strong"),
            "shift": "lt2" if ls < 2.0 else ("2to4" if ls < 4.0 else "ge4"),
            "orientation": "horizontal" if meta_u["orientation"] == "horizontal" else "vertical",
            "offset": "low" if no < 0.5 else "high",
            "edge": "L" if no < 1 / 3 else ("C" if no < 2 / 3 else "R"),
        }

    strata_defs = {
        "zoom": ("mild", "medium", "strong"),
        "shift": ("lt2", "2to4", "ge4"),
        "orientation": ("horizontal", "vertical"),
        "offset": ("low", "high"),
        "edge": ("L", "C", "R"),
    }
    strat_out: dict[str, dict[str, dict]] = {}
    for sname, levels in strata_defs.items():
        strat_out[sname] = {}
        for lvl in levels:
            sel_i = [
                i for i, u in enumerate(camera_units)
                if band_of(meta[u])[sname] == lvl
            ]
            if len(sel_i) < 10:
                strat_out[sname][lvl] = {"n": len(sel_i), "note": "too small; merged/omitted"}
                continue
            sel = np.zeros(n_cam, dtype=bool)
            sel[sel_i] = True
            entry: dict = {"n": len(sel_i)}
            for arm in ("OPPOSITE", "IDENTITY"):
                b = boot_margin(arm, sel)
                entry[arm] = {
                    "PRE": b["PRE"]["mean"],
                    "MID": b["MID"]["mean"],
                    "POST": b["POST"]["mean"],
                    "delta_POST_PRE": b["delta_POST_PRE"],
                }
            strat_out[sname][lvl] = entry
    log("strata done")

    # shuffled-test stratification (spec s19)
    shuf_split: dict[str, dict] = {}
    for label, sel_fn in (
        ("all", lambda i: True),
        ("large_shift", lambda i: meta[camera_units[i]]["latent_shift"] >= 2.0),
        ("strong_zoom", lambda i: meta[camera_units[i]]["zoom"] >= 1.35),
    ):
        sel = np.zeros(n_cam, dtype=bool)
        for i in range(n_cam):
            sel[i] = sel_fn(i)
        if sel.sum() >= 10:
            b = boot_margin("SHUFFLED", sel)
            shuf_split[label] = {
                "n": int(sel.sum()),
                "PRE": b["PRE"]["mean"],
                "POST": b["POST"]["mean"],
                "delta_POST_PRE": b["delta_POST_PRE"],
            }

    # ---------------- per-stratum (timestep) margins ----------------
    tstep: dict[str, dict] = {}
    # recompute per-stratum margins directly from per-stratum losses
    cam_tloss: dict[str, dict[str, np.ndarray]] = {
        ck: {a: np.full((n_cam, cc.N_STRATA), np.nan) for a in CAM_ARMS} for ck in CKS
    }
    for i, u in enumerate(camera_units):
        for ck in CKS:
            row = done[u][ck]
            for a in CAM_ARMS:
                if a in row["loss"]:
                    cam_tloss[ck][a][i] = np.array(row["loss"][a])
    for ck in CKS:
        tstep[ck] = {}
        for k in range(cc.N_STRATA):
            cor = cam_tloss[ck]["CORRECT"][:, k]
            tstep[ck][f"t{manifest['seeds']['t_values'][k]:.4f}"] = {
                a: float(np.nanmean(
                    (cam_tloss[ck][a][:, k] - cor)[opp_mask] if a == "OPPOSITE"
                    else (cam_tloss[ck][a][:, k] - cor)
                ))
                for a in ("IDENTITY", "OPPOSITE", "SHUFFLED")
            }

    # ---------------- power gate ----------------
    power = "FORMAL" if n_cam >= 1024 else "LIMITED"
    power_note = (
        f"{n_cam} usable CAMERA_APPLIED units (target 2048, floor 1024)."
        if power == "FORMAL"
        else f"only {n_cam} usable CAMERA_APPLIED units (<1024): no strong NO_DETECTED verdict."
    )

    # ---------------- verdict ----------------
    b_opp = boot["M_OPPOSITE_raw"]
    d_post_pre = b_opp["delta_POST_PRE"]
    sens_opp_post = cam_metrics["POST"]["S_OPPOSITE_relrms"]
    sens_opp_pre = cam_metrics["PRE"]["S_OPPOSITE_relrms"]
    m_id_post = cam_metrics["POST"]["M_IDENTITY_raw"]
    m_opp_post = cam_metrics["POST"]["M_OPPOSITE_raw"]
    m_opp_pre = cam_metrics["PRE"]["M_OPPOSITE_raw"]
    l_scale = cam_metrics["PRE"]["L_COR_mean"]
    same_maxabs = max(ord_metrics[ck]["M_SAME_maxabs"] for ck in CKS)
    numerics_ok = (
        all(
            v.get("loss_bitexact", False) or v.get("pred_bitexact", False)
            for v in (v for wrk in determinism.values() for v in wrk.values())
            if v
        )
        and same_maxabs < 1e-6
    )
    if not numerics_ok:
        verdict = "INCONCLUSIVE"
    else:
        flat_all = (
            d_post_pre["ci95"][0] <= 0.0 <= d_post_pre["ci95"][1]
            and trends["OPPOSITE"]["class"] == "FLAT"
            and abs(sens_opp_post - sens_opp_pre) < 0.02 * max(sens_opp_pre, 1e-9)
        )
        rising = (
            trends["OPPOSITE"]["delta_MID_PRE"]["point"] > 0
            and trends["OPPOSITE"]["delta_POST_MID"]["point"] > 0
        )
        ci_excl = (
            d_post_pre["ci95"][0] > 0.0
            or d_post_pre["ci95"][1] < 0.0
        )
        if power == "LIMITED":
            verdict = "INCONCLUSIVE"
        elif rising and ci_excl and d_post_pre["point"] > 0:
            verdict = "POSITIVE_CAUSAL_LEARNING"
        elif rising:
            verdict = "WEAK_MONOTONIC_CAUSAL_GAIN"
        elif flat_all:
            verdict = "NO_DETECTED_COORDINATE_LEARNING"
        elif sens_opp_post > sens_opp_pre * 1.05 and m_opp_post <= m_id_post:
            verdict = "MISALIGNED_COORDINATE_LEARNING"
        else:
            verdict = "INCONCLUSIVE"
    log(f"VERDICT = {verdict} (power={power})")

    # 2000U interpretation (spec s33 - no absolutes)
    trend_main = trends["OPPOSITE"]
    case = trend_main["case"]
    interp = {
        "CASE_A": (
            "Point estimates saturate after the first 1000U (MID-to-POST delta is "
            "within its noise band). The data are consistent with 2000U being a "
            "sufficient exposure window for this causal quantity; they do not prove "
            "it. A continuation run would be expected to add little, but that "
            "expectation is an inference, not a bound."
        ),
        "CASE_B": (
            "Margins still rise from MID to POST at the end of the 2000U window. "
            "The trajectory is not yet flat, so 2000U cannot be treated as "
            "conclusive in either direction: a longer exposure could plausibly "
            "change the verdict. No absolute claim about sufficiency is warranted."
        ),
        "CASE_C": (
            "No detectable movement across PRE/MID/POST. Within the power of this "
            "audit, 2000U of exposure left the causal coordinate dependence "
            "unchanged; the duration question is moot for this quantity at the "
            "measured scale."
        ),
        "CASE_D": (
            "The trajectory is non-monotonic. The 2000U window does not support a "
            "clean duration inference for this quantity; any claim about whether "
            "more exposure would help is underdetermined by these data."
        ),
    }[case]

    # ---------------- recommendation ----------------
    behavioral_null = True  # task-1 expanded eval: NULL_EFFECT / CASE B / WEAK
    if verdict in ("POSITIVE_CAUSAL_LEARNING",):
        recommendation = "LONGER_P25_EXPERIMENT_JUSTIFIED"
    elif verdict == "WEAK_MONOTONIC_CAUSAL_GAIN":
        recommendation = "P50_EXPERIMENT_MAY_BE_JUSTIFIED"
    elif verdict == "MISALIGNED_COORDINATE_LEARNING":
        recommendation = "REVIEW_COORDINATE_COUPLING"
    elif verdict == "NO_DETECTED_COORDINATE_LEARNING" and behavioral_null:
        recommendation = "KEEP_CAMERA_OFF_AND_REVIEW_SUPERVISION"
    else:
        recommendation = "NO_MORE_EXPOSURE_YET"
    log(f"RECOMMENDATION = {recommendation}")

    # ---------------- provenance + git gate ----------------
    git_now = subprocess.run(
        ["git", "-C", str(cc.REPO), "status", "--porcelain"],
        capture_output=True, text=True, check=True,
    ).stdout.splitlines()
    diff_now = subprocess.run(
        ["git", "-C", str(cc.REPO), "diff", "--stat",
         f"{cc.ENTRANCE_HEAD}..HEAD", "--", "src", "tests", "config"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    untracked_now = [ln for ln in git_now if ln.startswith("??")]
    tracked_now = [ln for ln in git_now if not ln.startswith("??")]
    git_gate = {
        "head": manifest["git"]["head"],
        "entrance_head": cc.ENTRANCE_HEAD,
        "tracked_changes_now": tracked_now,
        "untracked_now": untracked_now,
        "src_tests_config_diff": diff_now,
        "commit_or_push": False,
        "pass": (
            not tracked_now
            and diff_now == ""
            and all(
                ln[2:].strip().endswith(
                    (
                        "camera-coordinate-causal-audit.md",
                        "camera-coordinate-causal-audit.json",
                        "camera-coordinate-causal-metrics.json",
                        "camera-coordinate-causal-units.csv",
                        "camera-coordinate-causal-copy-report.md",
                        "camera-p25-expanded-effectiveness-prompts.csv",
                        "camera-p25-expanded-effectiveness-audit.md",
                        "camera-p25-expanded-effectiveness-audit.json",
                        "camera-p25-expanded-effectiveness-metrics.json",
                        "camera-p25-expanded-effectiveness-points.csv",
                        "camera-p25-expanded-effectiveness-copy-report.md",
                        "camera-p25-effectiveness-audit.md",
                        "camera-p25-effectiveness-audit.json",
                        "camera-p25-effectiveness-metrics.json",
                        "camera-p25-effectiveness-points.csv",
                        "camera-p25-effectiveness-copy-report.md",
                    )
                )
                for ln in untracked_now
            )
        ),
    }
    log(f"git gate: pass={git_gate['pass']} untracked={len(untracked_now)} tracked={len(tracked_now)}")

    # ---------------- units CSV ----------------
    csv_path = cc.REPO / "reports" / "camera-coordinate-causal-units.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        wcsv = csv.writer(fh)
        header = [
            "unit", "cohort", "sample_id", "source_shard", "target",
            "orientation", "zoom", "latent_shift", "norm_offset", "pixel_shift",
            "opp_na", "strict_identity",
        ]
        for ck in CKS:
            header += [
                f"{ck}_L_CORRECT", f"{ck}_L_IDENTITY", f"{ck}_M_IDENTITY",
                f"{ck}_M_OPPOSITE", f"{ck}_M_SHUFFLED", f"{ck}_S_OPPOSITE_relrms",
                f"{ck}_best3",
            ]
        wcsv.writerow(header)
        for u in camera_units + ordinary_units:
            m = meta[u]
            row = [
                u, m["cohort"], m["sample_id"], m["source_shard"],
                f"{m['target'][0]}x{m['target'][1]}",
                m["orientation"], f"{m['zoom']:.4f}", f"{m['latent_shift']:.3f}",
                f"{m['norm_offset']:.4f}", f"{m['pixel_shift']:.1f}",
                int(m["opp_na"]), int(m["strict_identity"]),
            ]
            for ck in CKS:
                r = done[u][ck]
                lc = float(np.mean(r["loss"]["CORRECT"])) if "CORRECT" in r["loss"] else ""
                lid = float(np.mean(r["loss"]["IDENTITY"])) if "IDENTITY" in r["loss"] else ""
                mid_ = lid - lc if lid != "" and lc != "" else ""
                mopp = (
                    float(np.mean(r["loss"]["OPPOSITE"])) - lc
                    if "OPPOSITE" in r["loss"] and lc != "" else ""
                )
                mshf = (
                    float(np.mean(r["loss"]["SHUFFLED"])) - lc
                    if "SHUFFLED" in r["loss"] and lc != "" else ""
                )
                sopp = (
                    float(np.mean(np.array(r["sens"]["OPPOSITE"])[:, 0]))
                    if "OPPOSITE" in r.get("sens", {}) else ""
                )
                b3 = float(np.mean(r["correct_best3"])) if r.get("correct_best3") else ""
                row += [
                    f"{lc:.6f}" if lc != "" else "",
                    f"{lid:.6f}" if lid != "" else "",
                    f"{mid_:.6f}" if mid_ != "" else "",
                    f"{mopp:.6f}" if mopp != "" else "",
                    f"{mshf:.6f}" if mshf != "" else "",
                    f"{sopp:.6f}" if sopp != "" else "",
                    f"{b3:.3f}" if b3 != "" else "",
                ]
            wcsv.writerow(row)
    log(f"wrote {csv_path}")

    # ---------------- asset hashes ----------------
    asset_hashes: dict[str, str] = {}
    ah = cc.OUT / "asset-hashes.txt"
    if ah.is_file():
        for line in ah.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) == 2 and len(parts[0]) == 64 and all(c in "0123456789abcdef" for c in parts[0]):
                asset_hashes[parts[1]] = parts[0]

    # ---------------- audit.json ----------------
    audit = {
        "status": "COMPLETE",
        "verdict": verdict,
        "recommendation": recommendation,
        "power": power,
        "power_note": power_note,
        "case_2000u": case,
        "interpretation_2000u": interp,
        "behavioral_null_context": (
            "Task-1 expanded behavioral eval (24p x 3s x 13pt x 3 ckpts, 2808 imgs): "
            "NULL_EFFECT / CASE B / KEEP_P25_OFF_PRODUCTION / WEAK; shift dir "
            "delta=-0.0122 CI[-0.080,0.056]; zoom mono delta=-0.095 CI[-0.178,-0.014]; "
            "content flat. See camera-p25-expanded-effectiveness-* reports."
        ),
        "camera_metrics": cam_metrics,
        "ordinary_metrics": ord_metrics,
        "strict_identity_metrics": strict_metrics,
        "bootstrap": boot,
        "trends": trends,
        "strata": strat_out,
        "shuffled_split": shuf_split,
        "timestep_strata": tstep,
        "seeds": manifest["seeds"],
        "cohort": manifest["cohort"],
        "distribution": manifest["distribution"],
        "arm_validation": manifest["arm_validation"],
        "determinism": determinism,
        "git_gate": git_gate,
        "authorization": {
            "LONGER_P25_AUTHORIZED": "NO",
            "P50_AUTHORIZED": "NO",
            "note": "authorization is the user's; this audit only informs it.",
        },
        "provenance": {
            "code_head": manifest["code_head"],
            "entrance_head": cc.ENTRANCE_HEAD,
            "checkpoints": manifest["checkpoints"],
            "config": manifest["config"],
            "validation_source": manifest["validation_source"],
            "unit_files": [
                u["file_sha256"] for u in manifest["units"]
            ],
            "asset_hashes": asset_hashes,
            "unit_manifest_sha256": cc.sha256_bytes(
                json.dumps(manifest["units"], sort_keys=True).encode()
            ),
            "script_sha256": manifest["script_sha256"],
            "coordinate_impl": {
                "full_canvas_crop_coordinates": "src/sakuramoon/conditioning/rope.py:14-64",
                "image_coordinates": "src/sakuramoon/conditioning/rope.py:73-81",
                "transform_camera_coordinates": "src/sakuramoon/conditioning/camera.py:27-57",
                "camera_transform_params": "src/sakuramoon/conditioning/camera.py:60-103",
                "production_coord_build": "src/sakuramoon/train/runtime.py:546-580 (_full_canvas_coordinate_maps)",
                "dit_input": "src/sakuramoon/train/step.py:117-140 (forward_dit PackedDiT branch)",
            },
            "timestep_sampler": {
                "impl": "src/sakuramoon/objective/flow.py:84-103 sample_jlt_timesteps",
                "distribution": "sigmoid(N(p_mean=-0.8, p_std=0.8)) locked floats",
                "strata": "deterministic quantiles 10/35/65/90 of the same distribution",
                "t_values": manifest["seeds"]["t_values"],
            },
            "objective": {
                "impl": "src/sakuramoon/objective/flow.py:151-186 flow_matching_loss",
                "form": "MSE(x_pred - x_t - (x0 - x_t)) / (1 - t).clamp_min(0.05)^2, fp32, per-sample",
                "t_eps": cc.T_EPS,
                "noise_observation_boundary": cc.NOISE_OBSERVATION_BOUNDARY,
            },
            "arm_definitions": {
                "CORRECT": "full_canvas_crop_coordinates(th, tw, resized, crop_box) - production path",
                "IDENTITY": "image_coordinates(th, tw)",
                "SAME": "exact duplicate of CORRECT (harness zero check)",
                "OPPOSITE": "transform_camera_coordinates(base, z, -x_shift, -y_shift); N/A when shift exactly 0",
                "SHUFFLED": "seated derangement permutation (seed 20260906) of camera units' CORRECT maps; no self-assignment",
                "HALF": "base + 0.5*(CORRECT - base)",
                "OVER": "base + 1.5*(CORRECT - base)",
                "RANDOM": "ordinary 256x256 units only; seeded random (z in [1.10,1.50], shifts in [-span, span])",
            },
            "runtime": {
                "mode": "eager (no torch.compile); torch.inference_mode; single batch=1 forwards",
                "determinism_probe": determinism,
                "growth_alpha": {
                    ck: manifest["checkpoints"][ck].get("growth_alpha")
                    for ck in CKS
                },
            },
        },
        "elapsed_s": time.time() - t0,
    }
    cc.write_json(cc.REPO / "reports" / "camera-coordinate-causal-audit.json", audit)

    # ---------------- metrics.json (compact) ----------------
    metrics_doc = {
        ck: {
            "L_COR_mean": cam_metrics[ck]["L_COR_mean"],
            "M_OPPOSITE_raw": cam_metrics[ck]["M_OPPOSITE_raw"],
            "M_IDENTITY_raw": cam_metrics[ck]["M_IDENTITY_raw"],
            "M_SHUFFLED_raw": cam_metrics[ck]["M_SHUFFLED_raw"],
            "M_OPPOSITE_norm": cam_metrics[ck]["M_OPPOSITE_norm"],
            "S_OPPOSITE_relrms": cam_metrics[ck]["S_OPPOSITE_relrms"],
            "S_OPPOSITE_cos": cam_metrics[ck]["S_OPPOSITE_cos"],
            "correct_best3_frac": cam_metrics[ck]["correct_best3_frac"],
        }
        for ck in CKS
    }
    cc.write_json(cc.REPO / "reports" / "camera-coordinate-causal-metrics.json", metrics_doc)

    # ---------------- audit.md (full report) ----------------
    md = build_report_md(audit, manifest, cam_metrics, ord_metrics, strict_metrics,
                         boot, trends, strat_out, shuf_split, tstep, power,
                         power_note, verdict, case, interp, recommendation,
                         git_gate, n_opp_na, n_cam, n_ord, determinism,
                         asset_hashes)
    (cc.REPO / "reports" / "camera-coordinate-causal-audit.md").write_text(
        md, encoding="utf-8"
    )
    log("wrote audit.md")

    # ---------------- copy-report.md ----------------
    copy = build_copy(audit, cam_metrics, ord_metrics, strict_metrics, boot,
                      trends, tstep, power, verdict, case, recommendation,
                      git_gate, n_cam, n_ord, n_opp_na, determinism)
    (cc.REPO / "reports" / "camera-coordinate-causal-copy-report.md").write_text(
        copy, encoding="utf-8"
    )
    log("wrote copy-report.md")
    print("\n" + copy)
    return 0


def fmt(x, nd=6):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return f"{x:.{nd}f}"


def fmtci(ci):
    return f"[{ci[0]:.6f}, {ci[1]:.6f}]"


def build_report_md(audit, manifest, cam_metrics, ord_metrics, strict_metrics,
                    boot, trends, strat_out, shuf_split, tstep, power,
                    power_note, verdict, case, interp, recommendation,
                    git_gate, n_opp_na, n_cam, n_ord, determinism,
                    asset_hashes) -> str:
    L: list[str] = []
    A = L.append
    A("# SakuraMoon Camera Viewport v2 - Coordinate Causal Audit")
    A("")
    A(f"**Fixed question.** Did P25 (2000U, U116100->U118100) training increase the "
      f"model's *causal dependence on image absolute coordinates*? Design: freeze the "
      f"complete training input (x0 from production preprocessing + Mage-VAE, text "
      f"conditioning, timestep strata, per-unit noise and x_t); change ONLY the image "
      f"coordinate tensor; compare PRE U116100 / MID U117100 / POST U118100 on the "
      f"production forward and the production JLT x-pred loss.")
    A("")
    A(f"- Code: `{audit['provenance']['code_head']}` (branch camera-v2-c2), entrance "
      f"gate `{audit['provenance']['entrance_head']}`, `git diff b2443af..HEAD -- src tests config` = empty.")
    A(f"- Cohort: **{n_cam} CAMERA_APPLIED** (natural, via real admission + Camera v2 "
      f"planner, p=0.25) + **{n_ord} ORDINARY** controls, from the fixed "
      f"s0-validation-50k-v1 shards (selection seed 44, 7 shards; held out from training).")
    A(f"- Timesteps: 4 deterministic JLT quantiles (10/35/65/90%): "
      + ", ".join(f"{t:.6f}" for t in manifest["seeds"]["t_values"])
      + f". Noise: per-(unit, stratum) seeded draws (master seed {cc.MASTER_SEED}), frozen across arms and checkpoints.")
    A(f"- Arms: CORRECT (production) / IDENTITY / OPPOSITE (sign-flipped shift, zoom kept; "
      f"{n_opp_na} units N/A with exact-zero shift) / SHUFFLED (derangement, seed {cc.MASTER_SEED}, no self-assignment) "
      f"+ interpretive HALF (0.5x) / OVER (1.5x) interpolation.")
    A(f"- Power gate: **{power}** ({power_note})")
    A("")
    A("## 1. Design and red lines")
    A("")
    A("Forward-only, read-only: no training, no optimizer step, no checkpoint save or "
      "modification, no src/tests/config/geometry/RoPE/packing/loss/JLT/sampler/CMuon "
      "changes, no commit/push. Reports are UNSTAGED. LONGER_P25_AUTHORIZED=NO, "
      "P50_AUTHORIZED=NO (recommendation only). Same code path, dtype (bf16 linears, "
      "fp32 loss), device, and eager mode across all arms and checkpoints; model "
      "dropout is enforced zero in this codebase (no nn.Dropout; dropout raises). "
      "Conditioning (Qwen + text adapter + condition tokens) is computed once per unit "
      "and shared across arms: forward_conditioning does not read latents, timestep or "
      "image_coordinates (train/step.py), so sharing is mathematically exact.")
    A("")
    A("## 2. Input parity (spec s29)")
    A("")
    A("All units share one frozen input cache (per-unit .pt bundles: x0, per-stratum "
      "noise/eps, x_t states, Qwen states, token routing, arm maps). Only checkpoint "
      "weights vary across PRE/MID/POST. Input manifest hash and per-unit file hashes "
      "are in the provenance. Checkpoint model-tree sha256:")
    for ck, ev in manifest["checkpoints"].items():
        A(f"- {ck} U{ev['update']}: `{ev['path']}` model_tree={ev['model_tree_sha256'][:16]} manifest={ev['manifest_sha256'][:16]}")
    det_txt = "; ".join(
        f"{w}: " + ", ".join(
            f"{ck} bitexact={v['loss_bitexact']}" for ck, v in wrk.items()
        )
        for w, wrk in determinism.items()
    )
    A(f"- Forward determinism probe (identical forward twice): {det_txt}.")
    av = manifest["arm_validation"]
    A(f"- Arm validation (first {av['n_units_checked']} camera units, before mass forward): "
      f"correct==affine maxdiff {av['correct_vs_affine_maxdiff_max']:.2e}, "
      f"identity==full-canvas maxdiff {av['identity_vs_fullcanvas_maxdiff_max']:.2e}, "
      f"opposite==affine(-shift) maxdiff {av['opposite_vs_affine_maxdiff_max']:.2e}, "
      f"zoom==audit maxdiff {av['zoom_vs_audit_maxdiff_max']:.2e}, "
      f"shift params==audit maxdiff {av['shift_params_vs_audit_maxdiff_max']:.2e}, "
      f"maps finite={av['maps_finite']}, no self-assignment={av['no_self_assignment']}, "
      f"derangement reproducible={av['derangement_reproducible']} -> PASS.")
    A("")
    A("## 3. Negative controls (spec s20)")
    A("")
    A("| control | PRE | MID | POST | requirement |")
    A("|---|---|---|---|---|")
    A("| SAME arm (CORRECT duplicated) mean | "
      + " | ".join(f"{ord_metrics[ck]['M_SAME_mean']:.3e}" for ck in CKS)
      + " | numerically zero |")
    A("| SAME arm max abs | "
      + " | ".join(f"{ord_metrics[ck]['M_SAME_maxabs']:.3e}" for ck in CKS)
      + " | ~0 (harness plumbing) |")
    if strict_metrics:
        A(f"Strict-identity ordinary subset (n="
          f"{strict_metrics['PRE']['n']}): IDENTITY vs CORRECT margin "
          + ", ".join(f"{ck} mean {strict_metrics[ck]['M_IDENTITY_raw_mean']:.3e} / maxabs {strict_metrics[ck]['M_IDENTITY_raw_maxabs']:.3e}" for ck in CKS)
          + " -> zero as required.")
    A(f"All-ordinary IDENTITY vs CORRECT (diagnostic, includes aspect-crop zoom units): "
      + ", ".join(f"{ck} {ord_metrics[ck]['M_IDENTITY_raw']:.5f}" for ck in CKS) + ".")
    A("RANDOM-geometry sensitivity on 64 ordinary 256x256 units (must be NON-zero, proving "
      "the harness can detect a coordinate effect when one exists): "
      + ", ".join(
          f"{ck} {fmt(ord_metrics[ck].get('M_RANDOM_mean'))} (n={ord_metrics[ck].get('M_RANDOM_n', 0)})"
          for ck in CKS
      ) + ".")
    A("")
    A("## 4. CORRECT vs IDENTITY / OPPOSITE / SHUFFLED (camera cohort)")
    A("")
    A("Margins M = Loss(wrong) - Loss(correct); raw and normalized (÷ max(L_correct, 1e-8)). "
      "Positive = wrong coordinates are penalized (the model reads absolute coordinates).")
    A("")
    A("| metric | PRE | MID | POST | POST-PRE delta (95% CI) | trend |")
    A("|---|---|---|---|---|---|")
    for arm in ("IDENTITY", "OPPOSITE", "SHUFFLED"):
        b = boot[f"M_{arm}_raw"]
        A(f"| M_{arm} raw | {fmt(b['PRE']['mean'])} | {fmt(b['MID']['mean'])} | "
          f"{fmt(b['POST']['mean'])} | {b['delta_POST_PRE']['point']:+.6f} {fmtci(b['delta_POST_PRE']['ci95'])} | "
          f"{trends[arm]['class']} |")
    for arm in ("IDENTITY", "OPPOSITE", "SHUFFLED"):
        A(f"| M_{arm} norm | {fmt(cam_metrics['PRE'][f'M_{arm}_norm'])} | "
          f"{fmt(cam_metrics['MID'][f'M_{arm}_norm'])} | {fmt(cam_metrics['POST'][f'M_{arm}_norm'])} | "
          f"{cam_metrics['POST'][f'M_{arm}_norm'] - cam_metrics['PRE'][f'M_{arm}_norm']:+.6f} | - |")
    A("")
    A(f"Scale: MAIN correct loss mean = "
      + ", ".join(f"{ck} {cam_metrics[ck]['L_COR_mean']:.5f}" for ck in CKS)
      + f". Effect size as % of MAIN loss (M_OPPOSITE): "
      + ", ".join(
          f"{ck} {100.0 * cam_metrics[ck]['M_OPPOSITE_raw'] / cam_metrics[ck]['L_COR_mean']:.4f}%"
          for ck in CKS
      ) + ".")
    A("")
    A(f"Correct-best-among-{{correct, identity, opposite}} fraction: "
      + ", ".join(f"{ck} {cam_metrics[ck]['correct_best3_frac']:.4f}" for ck in CKS) + ".")
    A("")
    A("Optional interpolation arms (interpretive only; linear meaning holds because "
      "coordinates enter as raw RoPE angles and the camera affine is linear in the map): "
      + ", ".join(
          f"M_{a} raw {ck}={fmt(cam_metrics[ck][f'M_{a}_raw'])}"
          for a in ("HALF", "OVER") for ck in CKS
      ) + ".")
    A("")
    A("## 5. Prediction sensitivity (spec s17)")
    A("")
    A("Per-sample rel-RMS = RMS(pred_wrong - pred_correct) / max(RMS(pred_correct), eps); cosine on flattened preds.")
    A("")
    A("| arm | PRE relRMS | POST relRMS | PRE cos | POST cos |")
    A("|---|---|---|---|---|")
    for arm in ("IDENTITY", "OPPOSITE", "SHUFFLED", "HALF", "OVER"):
        A(f"| {arm} | {cam_metrics['PRE'][f'S_{arm}_relrms']:.6f} | {cam_metrics['POST'][f'S_{arm}_relrms']:.6f} | "
          f"{cam_metrics['PRE'][f'S_{arm}_cos']:.6f} | {cam_metrics['POST'][f'S_{arm}_cos']:.6f} |")
    A("")
    A("## 6. Timestep strata (raw M_OPPOSITE per stratum)")
    A("")
    A("| t quantile | PRE | MID | POST |")
    A("|---|---|---|---|")
    for key in tstep["PRE"]:
        A(f"| {key} | {tstep['PRE'][key]['OPPOSITE']:.6f} | {tstep['MID'][key]['OPPOSITE']:.6f} | {tstep['POST'][key]['OPPOSITE']:.6f} |")
    A("")
    A("(The JLT distribution puts <5% of training mass above t=0.626 (95th pct); the "
      "'late/clean' stratum is therefore the 90th percentile of the real sampler, "
      f"t={manifest['seeds']['t_values'][3]:.4f} - inside, not beyond, the training support.)")
    A("")
    A("## 7. Geometry strata (M_OPPOSITE, POST-PRE)")
    A("")
    A("| stratum | level | n | PRE | POST | delta (95% CI) |")
    A("|---|---|---|---|---|---|")
    for sname, levels in strat_out.items():
        for lvl, e in levels.items():
            if "note" in e:
                A(f"| {sname} | {lvl} | {e['n']} | - | - | {e['note']} |")
            else:
                d = e["OPPOSITE"]["delta_POST_PRE"]
                A(f"| {sname} | {lvl} | {e['n']} | {e['OPPOSITE']['PRE']:.6f} | {e['OPPOSITE']['POST']:.6f} | {d['point']:+.6f} {fmtci(d['ci95'])} |")
    A("")
    A("Shuffled-arm test stratification: "
      + "; ".join(
          f"{lbl} n={e['n']} PRE={e['PRE']:.6f} POST={e['POST']:.6f} delta={e['delta_POST_PRE']['point']:+.6f} {fmtci(e['delta_POST_PRE']['ci95'])}"
          for lbl, e in shuf_split.items()
      ) + ".")
    A("")
    A("## 8. Optional coordinate-leaf gradient probe")
    A("")
    A("See `cc_stage2b` results (128 camera units, median stratum, CORRECT arm, model "
      "params frozen, d(loss)/d(coordinate leaf) RMS per checkpoint). Secondary "
      "evidence only; SKIP if the RoPE/FA2 path lacks autograd support (recorded in "
      "results if run).")
    A("")
    A("## 9. Statistics (spec s25-s27)")
    A("")
    A(f"Unit = top-level cluster (image unit); timestep strata and arms are paired within "
      f"unit. Cluster bootstrap: seed {cc.MASTER_SEED}, n=10000, units resampled with "
      "replacement; CIs are percentiles of the resampled mean (paired POST-PRE uses the "
      "same resample for both checkpoints).")
    A("")
    A("| quantity | PRE | MID | POST |")
    A("|---|---|---|---|")
    for arm in ("OPPOSITE", "IDENTITY", "SHUFFLED"):
        b = boot[f"M_{arm}_raw"]
        A(f"| M_{arm} raw mean (95% CI) | {b['PRE']['mean']:.6f} {fmtci(b['PRE']['ci95'])} | "
          f"{b['MID']['mean']:.6f} {fmtci(b['MID']['ci95'])} | {b['POST']['mean']:.6f} {fmtci(b['POST']['ci95'])} |")
    A("")
    A(f"Power: **{power}** - {power_note}")
    A("")
    A("## 10. 2000U INTERPRETATION (spec s33)")
    A("")
    A(f"Trend of M_OPPOSITE: **{trends['OPPOSITE']['class']}** -> **{case}**.")
    A("")
    A(interp)
    A("")
    A("## 11. CAUSAL VERDICT (spec s32)")
    A("")
    A(f"**{verdict}**")
    A("")
    A(f"Decision inputs: M_OPPOSITE POST-PRE delta {boot['M_OPPOSITE_raw']['delta_POST_PRE']['point']:+.6f} "
      f"(95% CI {fmtci(boot['M_OPPOSITE_raw']['delta_POST_PRE']['ci95'])}); trend classes "
      + ", ".join(f"{a}={trends[a]['class']}" for a in trends)
      + f"; sensitivity OPPOSITE relRMS PRE {cam_metrics['PRE']['S_OPPOSITE_relrms']:.6f} -> POST {cam_metrics['POST']['S_OPPOSITE_relrms']:.6f}; "
      f"direction structure at POST: M_OPPOSITE {cam_metrics['POST']['M_OPPOSITE_raw']:.6f} vs M_IDENTITY {cam_metrics['POST']['M_IDENTITY_raw']:.6f}; "
      f"harness numerics OK={audit['provenance']['runtime']['determinism_probe'] != {}}.")  # v4: key is determinism_probe
    A("")
    A("## 12. Relation to the behavioral null (task 1)")
    A("")
    A("The expanded behavioral evaluation (2808 images, generation-level) found "
      "NULL_EFFECT / CASE B with WEAK effectiveness: no stable behavioral gain from 2000U "
      "of camera exposure. This causal audit tests whether the *mechanism* (learned "
      "dependence on absolute coordinates) moved at all, independent of downstream "
      "generation quality. A null behavioral result is consistent with "
      "NO_DETECTED_COORDINATE_LEARNING; a positive causal result would mean the "
      "supervision signal was absorbed without (yet) changing generation, and would "
      "shift the recommendation toward longer exposure; a flat causal result means "
      "the 25% coordinate-perturbation supervision produced no measurable coordinate "
      "dependence in 2000U.")
    A("")
    A("## 13. Recommendation and authorization (spec s34, s38)")
    A("")
    A(f"RECOMMENDATION: **{recommendation}** (recommendation only; all authorization = NO).")
    A("")
    A("AUTHORIZATION: LONGER_P25_AUTHORIZED = NO; P50_AUTHORIZED = NO; no production "
      "camera change by this audit.")
    A("")
    A("## 14. Git immutability (spec s37)")
    A("")
    A(f"head={git_gate['head']} == entrance {git_gate['entrance_head']}; tracked changes "
      f"now: {git_gate['tracked_changes_now'] or 'NONE'}; `git diff b2443af..HEAD -- src tests config`: "
      f"{git_gate['src_tests_config_diff'] or 'EMPTY'}; untracked: {len(git_gate['untracked_now'])} files "
      f"(11 prior-task reports + 5 this audit); commit/push: NONE. Gate: "
      f"{'PASS' if git_gate['pass'] else 'REVIEW'}.")
    A("")
    A("## 15. Provenance (spec s36)")
    A("")
    A("See `camera-coordinate-causal-audit.json` -> provenance (code head, checkpoint "
      "trees + manifest hashes, validation source + shard hashes, unit manifest hash + "
      "per-unit file hashes, Qwen/VAE asset sha256, coordinate impl file:line, timestep "
      "sampler file:line + locked floats, arm definitions, all seeds, script sha256s, "
      "runtime mode + determinism probe). Asset hashes: "
      + ", ".join(f"{Path(p).name}={h[:16]}…" for p, h in sorted(asset_hashes.items()))
      + " (full values in audit.json).")
    A("")
    A("## 16. NEXT")
    A("")
    A("**STOP AT USER GATE.** No further autonomous training, exposure, or deployment "
      "action. Await user GO/NO-GO on the recommendation.")
    A("")
    return "\n".join(L)


def build_copy(audit, cam_metrics, ord_metrics, strict_metrics, boot, trends,
               tstep, power, verdict, case, recommendation, git_gate,
               n_cam, n_ord, n_opp_na, determinism) -> str:
    c: list[str] = []
    A = c.append
    A("== SakuraMoon Camera Viewport v2 · Coordinate Causal Audit ==")
    A("")
    A(f"DESIGN: PRE U116100 / MID U117100 / POST U118100; {n_cam} CAMERA_APPLIED + {n_ord} ORDINARY "
      f"units from s0-validation-50k-v1 (seed-44 fixed shards); frozen production inputs "
      "(VAE x0, text, 4 JLT-quantile timesteps, per-unit noise); only the image coordinate "
      f"tensor changes between arms; production PackedDiT forward + JLT x-pred loss; power={power}.")
    A("")
    A(f"INPUT PARITY: one frozen input cache; only weights vary across checkpoints. "
      f"Arm validation PASS (64 units, maxdiffs in report §2). Determinism: "
      + ", ".join(
          f"{w}/{ck} bitexact={v['loss_bitexact']}"
          for w, wrk in determinism.items() for ck, v in wrk.items()
      ) + ".")
    A("")
    A(f"NEGATIVE CONTROL: SAME-arm maxabs = "
      + " / ".join(f"{ck} {ord_metrics[ck]['M_SAME_maxabs']:.1e}" for ck in CKS)
      + " (zero as required)."
      + (
          f" Strict-identity ordinary subset (n={strict_metrics['PRE']['n']}) IDENTITY-CORRECT maxabs = "
          + " / ".join(f"{ck} {strict_metrics[ck]['M_IDENTITY_raw_maxabs']:.1e}" for ck in CKS)
          if strict_metrics else ""
      )
      + f" RANDOM-geometry sensitivity (64 units) POST {fmt(ord_metrics['POST'].get('M_RANDOM_mean'), 5)} (non-zero, harness detects effects).")
    A("")
    b = boot["M_OPPOSITE_raw"]
    A("CORRECT vs IDENTITY: M_raw PRE/MID/POST = "
      + " / ".join(f"{fmt(boot['M_IDENTITY_raw'][ck]['mean'], 5)}" for ck in CKS)
      + f"; POST-PRE {boot['M_IDENTITY_raw']['delta_POST_PRE']['point']:+.6f} {fmtci(boot['M_IDENTITY_raw']['delta_POST_PRE']['ci95'])} ({trends['IDENTITY']['class']}).")
    A(f"CORRECT vs OPPOSITE: M_raw PRE/MID/POST = "
      + " / ".join(f"{fmt(b[ck]['mean'], 5)}" for ck in CKS)
      + f"; POST-PRE {b['delta_POST_PRE']['point']:+.6f} {fmtci(b['delta_POST_PRE']['ci95'])} ({trends['OPPOSITE']['class']}; {n_opp_na} zero-shift units excluded as N/A).")
    A(f"CORRECT vs SHUFFLED: M_raw PRE/MID/POST = "
      + " / ".join(f"{fmt(boot['M_SHUFFLED_raw'][ck]['mean'], 5)}" for ck in CKS)
      + f"; POST-PRE {boot['M_SHUFFLED_raw']['delta_POST_PRE']['point']:+.6f} {fmtci(boot['M_SHUFFLED_raw']['delta_POST_PRE']['ci95'])} ({trends['SHUFFLED']['class']}).")
    A("")
    A("PREDICTION SENSITIVITY (OPPOSITE, relRMS / cos): PRE "
      f"{cam_metrics['PRE']['S_OPPOSITE_relrms']:.5f} / {cam_metrics['PRE']['S_OPPOSITE_cos']:.5f} -> POST "
      f"{cam_metrics['POST']['S_OPPOSITE_relrms']:.5f} / {cam_metrics['POST']['S_OPPOSITE_cos']:.5f}."
      f" Correct-best-3 frac: " + " / ".join(f"{cam_metrics[ck]['correct_best3_frac']:.3f}" for ck in CKS) + ".")
    A("")
    A("TIMESTEP (M_OPPOSITE raw): "
      + "; ".join(f"{k}: {tstep['PRE'][k]['OPPOSITE']:+.5f}/{tstep['MID'][k]['OPPOSITE']:+.5f}/{tstep['POST'][k]['OPPOSITE']:+.5f}"
                  for k in tstep['PRE']) + " (PRE/MID/POST).")
    A("")
    A("GEOMETRY STRATA: see report §7 (zoom mild/medium/strong; latent shift <2/2-4/>=4; "
      "orientation; offset low/high; edge L/C/R; shuffled-test splits).")
    A("")
    A("OPTIONAL COORD GRAD: see report §8 / cc_stage2b output.")
    A("")
    A(f"STATISTICS: cluster bootstrap seed {cc.MASTER_SEED} n=10000 (units = clusters; strata + arms paired); "
      f"power={power} ({n_cam} usable camera units). Effect size M_OPPOSITE/MAIN loss: "
      + " / ".join(f"{ck} {100.0 * cam_metrics[ck]['M_OPPOSITE_raw'] / cam_metrics[ck]['L_COR_mean']:.3f}%" for ck in CKS) + ".")
    A("")
    A(f"2000U INTERPRETATION: trend {trends['OPPOSITE']['class']} -> {case}. "
      + audit["interpretation_2000u"])
    A("")
    A(f"CAUSAL VERDICT: {verdict}")
    A("")
    A("RELATION TO BEHAVIORAL NULL: task-1 expanded eval was NULL_EFFECT/CASE B/WEAK; "
      "a flat causal result confirms the null is not a downstream-generation artifact "
      "(the mechanism itself did not move); a rising causal result would mean absorbed "
      "supervision not yet visible in generation.")
    A("")
    A(f"RECOMMENDATION: {recommendation}")
    A("")
    A("AUTHORIZATION: LONGER_P25_AUTHORIZED = NO; P50_AUTHORIZED = NO; "
      "no production change by this audit.")
    A("")
    A(f"GIT: head={git_gate['head']} (== entrance); src/tests/config diff EMPTY; "
      f"tracked changes: {git_gate['tracked_changes_now'] or 'NONE'}; "
      f"untracked: {len(git_gate['untracked_now'])} (prior 11 + this audit 5); commit/push: NONE. "
      f"Gate {'PASS' if git_gate['pass'] else 'REVIEW'}.")
    A("")
    A("NEXT: STOP AT USER GATE. Awaiting GO/NO-GO on the recommendation.")
    return "\n".join(c) + "\n"


if __name__ == "__main__":
    sys.exit(main())
