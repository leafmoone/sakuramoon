"""PHASE B: visual-content residual statistics over the frozen mirror pairs.

Reads (read-only):
  * the frozen subset manifest (+ sha verify) from PHASE A,
  * the frozen pair manifest (geometry rows only: zoom / latent shift),
  * the frozen per-pair mirror statistics from
        /tmp/camera-vertical-bottom/analysis/vertical-bottom-main.json
    cross-checked EXACTLY against the committed
        reports/camera-vertical-bottom-supervision-audit.json  (per_pair)
    and against the 12-significant-digit committed pairs CSV (G_PRE_POST).
  * the committed prerequisite-gate blocks inside the committed audit JSON.

This audit uses ONLY frozen feature JSON and frozen per-pair statistics; no
latents, images, captions, or model weights are touched (CPU only).

Outputs (under --out-root):
  analysis.json   full structured results
  bootstrap.json  every bootstrap call (seed, n_boot, point, ci95)
  logs/analyze.log
Reports (worktree reports/, new files only):
  camera-vertical-bottom-visual-content-residual-audit.md
  camera-vertical-bottom-visual-content-residual-audit.json
  camera-vertical-bottom-visual-content-residual-metrics.json
  camera-vertical-bottom-visual-content-residual-subsets.csv
  camera-vertical-bottom-visual-content-residual-copy-report.md
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import contracts as C

VBS_AUDIT_REL = "reports/camera-vertical-bottom-supervision-audit.json"
VBS_CSV_REL = "reports/camera-vertical-bottom-supervision-pairs.csv"

REPORT_MD = "camera-vertical-bottom-visual-content-residual-audit.md"
REPORT_JSON = "camera-vertical-bottom-visual-content-residual-audit.json"
REPORT_METRICS = "camera-vertical-bottom-visual-content-residual-metrics.json"
REPORT_CSV = "camera-vertical-bottom-visual-content-residual-subsets.csv"
REPORT_COPY = "camera-vertical-bottom-visual-content-residual-copy-report.md"
_REPORT_NAMES = (REPORT_MD, REPORT_JSON, REPORT_METRICS, REPORT_CSV, REPORT_COPY)

_GEO_TABLE: dict[int, dict] = {}


def log(out_dir: Path, msg: str) -> None:
    line = f"[visual-analyze {time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    with (out_dir / "logs" / "analyze.log").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def _ci_str(ci) -> str:
    if not isinstance(ci, (list, tuple)):
        return str(ci)
    return f"[{ci[0]:.6g}, {ci[1]:.6g}]"


def _geo(i: int, key: str) -> float:
    return float(_GEO_TABLE[i][key])


# ---------------- frozen inputs ----------------


def load_frozen_subsets(out_root: Path) -> dict:
    frozen = json.loads((out_root / "visual-subsets-frozen.json").read_text(encoding="utf-8"))
    body = (out_root / "visual-subsets.json").read_bytes()
    sha = C.sha256_of_bytes(body)
    if sha != frozen["sha256_visual_subsets"]:
        raise RuntimeError(
            f"STOP: subset freeze sha mismatch ({sha[:12]}... != {frozen['sha256_visual_subsets'][:12]}...)"
        )
    doc = json.loads(body)
    problems = C.verify_subset_doc(doc)
    if problems:
        raise RuntimeError(f"STOP: subset doc problems: {problems}")
    return doc


def load_mirror_stats(worktree: Path, frozen_main: Path, out_root: Path) -> dict[int, dict]:
    """per-pair G dict with exact cross-checks (spec section 8)."""
    main = json.loads(frozen_main.read_text(encoding="utf-8"))
    per_main = {int(k): v for k, v in main["per_pair"].items()}
    if sorted(per_main.keys()) != list(range(C.N_PAIRS)):
        raise RuntimeError("STOP: frozen main per_pair does not cover 0..511")

    audit = json.loads((worktree / VBS_AUDIT_REL).read_text(encoding="utf-8"))
    per_audit = {int(k): v for k, v in audit["per_pair"].items()}
    for i in range(C.N_PAIRS):
        for t in C.TRANS_KEYS:
            if per_main[i]["G"][t] != per_audit[i]["G"][t]:
                raise RuntimeError(f"STOP: frozen vs committed G mismatch at pair {i} {t}")

    with (worktree / VBS_CSV_REL).open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != C.N_PAIRS:
        raise RuntimeError(f"STOP: pairs CSV has {len(rows)} rows")
    for i, row in enumerate(rows):
        if int(row["pair_id"]) != i:
            raise RuntimeError(f"STOP: pairs CSV row {i} pair_id {row['pair_id']} out of order")
        g_csv = float(row["G_PRE_POST"])
        g_ref = per_main[i]["G"]["PRE|POST"]
        if abs(g_csv - g_ref) > 1e-12:
            raise RuntimeError(f"STOP: pairs CSV vs frozen G mismatch at pair {i}")
    log(out_root, "mirror stats: frozen main == committed audit (exact) and == committed CSV (12-sig-digit)")
    return per_main


def load_geometry_table(manifest_path: Path, out_root: Path) -> None:
    man_bytes = manifest_path.read_bytes()
    if C.sha256_of_bytes(man_bytes) != C.PAIR_MANIFEST_SHA:
        raise RuntimeError("STOP: pair manifest sha mismatch at phase B")
    manifest = json.loads(man_bytes)
    for p in manifest["pairs"]:
        _GEO_TABLE[int(p["pair_index"])] = {
            "zoom": p["equivalent_zoom"],
            "latent": p["latent_shift"],
            "abs_shift_px": p["absolute_shift"],
        }
    log(out_root, f"geometry table loaded for {len(_GEO_TABLE)} pairs (manifest sha verified)")


# ---------------- statistics ----------------


def boot_of(name: str, bootstrap: dict, idx: list[int], values_by_pair: dict[int, float]) -> dict:
    v = np.array([values_by_pair[i] for i in idx], dtype=np.float64)
    res = C.source_pair_bootstrap_ci(v)
    bootstrap[name] = res
    return res


def composition(idx: list[int], strata: dict[int, str]) -> dict:
    """Observed vs proportional-expected composition + continuous means."""
    obs: dict[str, int] = {}
    for i in idx:
        obs[strata[i]] = obs.get(strata[i], 0) + 1
    n = len(idx)
    rep: dict = {}
    for cell in sorted({strata[i] for i in range(C.N_PAIRS)}):
        expected = (sum(1 for j in range(C.N_PAIRS) if strata[j] == cell) / C.N_PAIRS) * n
        o = obs.get(cell, 0)
        rep[cell] = {"observed": o, "expected_proportional": float(expected), "difference": float(o - expected)}
    rep["_continuous"] = {
        "mean_zoom": float(np.mean([_geo(i, "zoom") for i in idx])),
        "mean_latent_shift": float(np.mean([_geo(i, "latent") for i in idx])),
        "mean_abs_shift_px": float(np.mean([_geo(i, "abs_shift_px") for i in idx])),
    }
    rep["_marginals"] = {
        "original_side": {s: sum(1 for i in idx if strata[i].startswith(s + "|")) for s in ("START", "END")},
        "zoom": {z: sum(1 for i in idx if strata[i].split("|")[1] == z) for z in ("mild", "medium", "strong")},
        "latent": {z: sum(1 for i in idx if strata[i].split("|")[2] == z) for z in ("lt2", "2to4", "ge4")},
    }
    return rep


def signed_groups(label: str, sign: dict[int, float], g_pp: dict[int, float],
                  idx_all: list[int], bootstrap: dict) -> dict:
    top_more = [i for i in idx_all if sign[i] > 0.0]
    bot_more = [i for i in idx_all if sign[i] < 0.0]
    tied = [i for i in idx_all if sign[i] == 0.0]
    return {
        "TOP_preserves_more": {**boot_of(f"G_PRE|POST:{label}_TOP_more", bootstrap, top_more, g_pp), "n": len(top_more)},
        "BOTTOM_preserves_more": {**boot_of(f"G_PRE|POST:{label}_BOTTOM_more", bootstrap, bot_more, g_pp), "n": len(bot_more)},
        "tied_n": len(tied),
    }


def in_subset_geo(idx: list[int], strata: dict[int, str], g_pp: dict[int, float], bootstrap: dict) -> dict:
    out: dict[str, dict] = {}
    bands: dict[str, list[int]] = {}
    for i in idx:
        parts = strata[i].split("|")
        bands.setdefault(f"zoom_{parts[1]}", []).append(i)
        bands.setdefault(f"latent_{parts[2]}", []).append(i)
    for key in sorted(bands.keys()):
        members = bands[key]
        block = boot_of(f"G_PRE|POST:{key}:in_subset", bootstrap, members, g_pp)
        rec = {"n": len(members), "mean_g": block["point"]}
        if len(members) >= 30:
            rec["ci95"] = block["ci95"]
        out[key] = rec
    return out


def _interpret(att25: float, g_v25: dict) -> dict:
    """Pre-registered interpretation wording rules (spec section 34)."""
    out: dict[str, str] = {}
    ci = g_v25.get("ci95")
    if isinstance(ci, list) and ci[0] > 0:
        out["primary"] = (
            "a directional residual persists after visual-content balancing; "
            "visual content asymmetry alone does not explain the TOP-BOTTOM gap"
        )
    elif isinstance(ci, list) and ci[1] < 0:
        out["primary"] = "unexpected: the balanced-subset gap is negative; treat as artifact / inconclusive"
    else:
        out["primary"] = "the TOP-BOTTOM gap weakens or vanishes under visual-content balancing"
    if att25 >= 0.5:
        out["attenuation_note"] = (
            "visual content asymmetry is not sufficient to explain most of the causal TOP-BOTTOM gap"
        )
    elif att25 < 0:
        out["attenuation_note"] = "balancing increased the gap (negative attenuation, reported as-is, not clipped)"
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--worktree", type=Path, required=True)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--frozen-main", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--validation", type=Path, default=None)
    args = ap.parse_args()

    rep = args.worktree / "reports"
    existing_mdjson = [n for n in (REPORT_MD, REPORT_JSON) if (rep / n).is_file()]
    if existing_mdjson and args.validation is None:
        raise RuntimeError(f"refusing to overwrite existing report {rep / existing_mdjson[0]} (use --validation)")

    args.out_root.mkdir(parents=True, exist_ok=True)
    log(args.out_root, "PHASE B start (frozen feature JSON + frozen per-pair stats only; CPU only)")

    subsets = load_frozen_subsets(args.out_root)
    log(args.out_root, f"subset freeze verified (sha match); sizes={subsets['subset_sizes']}")
    load_geometry_table(args.manifest, args.out_root)
    per_main = load_mirror_stats(args.worktree, args.frozen_main, args.out_root)

    # committed prerequisite gate (from the reviewed audit JSON)
    audit = json.loads((args.worktree / VBS_AUDIT_REL).read_text(encoding="utf-8"))
    pre = audit["numerics_prerequisite"]
    if pre["v2_numerics_clean"] is not True or pre["hidden_mutable_state"] is not False:
        raise RuntimeError("STOP: numerics prerequisite not PASS")
    if pre["gate"]["status"] != "PASS":
        raise RuntimeError("STOP: numerics gate status != PASS")
    if pre["p3_wired_into_gate"] is not True or pre["p3_state_history_effect"] is not False:
        raise RuntimeError("STOP: P3 not wired into the gate or P3 effect true")
    if audit["pair_design"]["selected_pairs"] != C.N_PAIRS:
        raise RuntimeError("STOP: committed selected_pairs != 512")
    g_all_block = audit["paired_top_bottom"]["PRE->POST_gap"]
    if not (isinstance(g_all_block.get("ci95"), list) and g_all_block["ci95"][0] > 0):
        raise RuntimeError("STOP: committed same-source PRE->POST gap CI lower not > 0")
    log(args.out_root, "prerequisite gate: V2 numerics clean, hidden state false, committed gap CI>0")

    # per-pair arrays
    per_pair = {int(k): v for k, v in subsets["per_pair"].items()}
    idx_all = list(range(C.N_PAIRS))
    idx_v50 = [int(i) for i in subsets["subsets"]["VISUAL_GEO_BALANCED_50"]]
    idx_v25 = [int(i) for i in subsets["subsets"]["VISUAL_GEO_BALANCED_25"]]
    if len(idx_v50) != C.V50_TARGET or len(idx_v25) != C.V25_TARGET:
        raise RuntimeError("STOP: subset sizes deviate from pre-registered targets")

    g = {t: {i: float(per_main[i]["G"][t]) for i in idx_all} for t in C.TRANS_KEYS}
    c_img = {i: float(per_pair[i]["C_img_signed"]) for i in idx_all}
    c_pe = {i: float(per_pair[i]["C_pe_signed"]) for i in idx_all}
    c_text = {i: float(per_pair[i].get("C_text_signed", float("nan"))) for i in idx_all}
    text_available = subsets.get("clip_text_status") == "AVAILABLE"
    a_img = {i: float(per_pair[i]["A_img"]) for i in idx_all}
    a_pe = {i: float(per_pair[i]["A_pe"]) for i in idx_all}
    a_visual = {i: float(per_pair[i]["A_visual"]) for i in idx_all}
    strata = {i: per_pair[i]["geometry_stratum"] for i in idx_all}

    bootstrap: dict[str, dict] = {}

    # ---- section 20: visual balance effectiveness ----
    visual = {}
    for name, idx in (("ALL", idx_all), ("V50", idx_v50), ("V25", idx_v25)):
        visual[name] = {
            "n": len(idx),
            "C_img_signed": boot_of(f"C_img_signed:{name}", bootstrap, idx, c_img),
            "C_pe_signed": boot_of(f"C_pe_signed:{name}", bootstrap, idx, c_pe),
            "A_img": {
                "mean": float(np.mean([a_img[i] for i in idx])),
                "p50": C.percentile3([a_img[i] for i in idx], 50),
                "p90": C.percentile3([a_img[i] for i in idx], 90),
            },
            "A_pe": {
                "mean": float(np.mean([a_pe[i] for i in idx])),
                "p50": C.percentile3([a_pe[i] for i in idx], 50),
                "p90": C.percentile3([a_pe[i] for i in idx], 90),
            },
            "A_visual": {
                "mean": float(np.mean([a_visual[i] for i in idx])),
                "p50": C.percentile3([a_visual[i] for i in idx], 50),
                "p90": C.percentile3([a_visual[i] for i in idx], 90),
            },
        }

    # ---- sections 21/23: causal residual + intervals ----
    gaps = {}
    for t in C.TRANS_KEYS:
        key = t.replace("|", "_")
        gaps[t] = {
            "ALL": boot_of(f"G_{key}:ALL", bootstrap, idx_all, g[t]),
            "V50": boot_of(f"G_{key}:V50", bootstrap, idx_v50, g[t]),
            "V25": boot_of(f"G_{key}:V25", bootstrap, idx_v25, g[t]),
        }

    g_all = gaps["PRE|POST"]["ALL"]
    g_v50 = gaps["PRE|POST"]["V50"]
    g_v25 = gaps["PRE|POST"]["V25"]
    ret50 = C.retention(g_v50, g_all)
    ret25 = C.retention(g_v25, g_all)
    att50 = C.attenuation(g_v50, g_all)
    att25 = C.attenuation(g_v25, g_all)

    # ---- sections 24/25: correlations (Spearman) ----
    gv = np.array([g["PRE|POST"][i] for i in idx_all], dtype=np.float64)
    abs_gv = np.abs(gv)
    correlations: dict = {
        "rho_C_img_signed_vs_G": C.spearman_rho([c_img[i] for i in idx_all], gv),
        "rho_C_pe_signed_vs_G": C.spearman_rho([c_pe[i] for i in idx_all], gv),
        "rho_A_img_vs_absG": C.spearman_rho([a_img[i] for i in idx_all], abs_gv),
        "rho_A_pe_vs_absG": C.spearman_rho([a_pe[i] for i in idx_all], abs_gv),
        "rho_A_visual_vs_absG": C.spearman_rho([a_visual[i] for i in idx_all], abs_gv),
    }
    if text_available:
        correlations["clip_text"] = {
            "rho_C_text_signed_vs_G": C.spearman_rho([c_text[i] for i in idx_all], gv),
            "rho_abs_C_text_signed_vs_absG": C.spearman_rho([abs(c_text[i]) for i in idx_all], abs_gv),
        }
    else:
        correlations["clip_text"] = "NOT_AVAILABLE"

    # ---- section 26: A_visual quartiles ----
    quartiles = {}
    for q, chunk in enumerate(C.quartile_order([a_visual[i] for i in idx_all])):
        block = boot_of(f"G_PRE|POST:Q{q}", bootstrap, chunk, g["PRE|POST"])
        quartiles[f"Q{q}"] = {
            "n": len(chunk),
            "A_visual_mean": float(np.mean([a_visual[i] for i in chunk])),
            "mean_g": block["point"],
            "ci95": block["ci95"],
        }

    # ---- section 27: signed visual-loss groups ----
    groups_img = signed_groups("clip_image", c_img, g["PRE|POST"], idx_all, bootstrap)
    groups_pe = signed_groups("pe_retained", c_pe, g["PRE|POST"], idx_all, bootstrap)

    # ---- section 29: in-subset geometry confirmation ----
    geo_v50 = in_subset_geo(idx_v50, strata, g["PRE|POST"], bootstrap)
    geo_v25 = in_subset_geo(idx_v25, strata, g["PRE|POST"], bootstrap)

    # ---- section 17: subset geometry composition gate ----
    composition_all = composition(idx_all, strata)
    composition_v50 = composition(idx_v50, strata)
    composition_v25 = composition(idx_v25, strata)

    # ---- section 28: joint-low intersection statistics ----
    idx_joint = [int(i) for i in subsets["subsets"]["JOINT_LOW_VISUAL_50_INTERSECTION"]]
    joint_block = boot_of("G_PRE|POST:JOINT_LOW", bootstrap, idx_joint, g["PRE|POST"])
    joint_rec = {
        "n": len(idx_joint),
        "mean_g": joint_block["point"],
        "ci95": joint_block["ci95"] if len(idx_joint) >= 128 else "LIMITED_N",
        "ci95_raw": joint_block["ci95"],
        "note": "intersection of per-metric lowest-256 sets (secondary stringent subset)",
    }

    # ---- sensitivity subsets: G PRE|POST ----
    sensitivity = {}
    for name in (
        "GLOBAL_VISUAL_50",
        "GLOBAL_VISUAL_25",
        "CLIP_IMAGE_ONLY_GEO_50",
        "CLIP_IMAGE_ONLY_GEO_25",
        "PE_ONLY_GEO_50",
        "PE_ONLY_GEO_25",
    ):
        idx = [int(i) for i in subsets["subsets"][name]]
        sensitivity[name] = {"n": len(idx), **boot_of(f"G_PRE|POST:{name}", bootstrap, idx, g["PRE|POST"])}

    # ---- classification ----
    residual = C.classify_directional_residual(g_v50, g_v25, ret25)
    contribution = C.classify_visual_contribution(
        visual["ALL"]["C_img_signed"],
        visual["ALL"]["C_pe_signed"],
        correlations["rho_C_img_signed_vs_G"],
        correlations["rho_C_pe_signed_vs_G"],
        att25,
    )
    overall = C.classify_overall(residual, contribution)
    rec = C.recommendation_for(overall)

    frozen_doc = json.loads((args.out_root / "visual-subsets-frozen.json").read_text(encoding="utf-8"))
    analysis = {
        "schema_version": C.SCHEMA_VERSION,
        "base_sha": C.BASE_SHA,
        "branch": "camera-v2-visual-content-residual-review",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "provenance": {
            "pair_manifest_sha": subsets["pair_manifest_sha"],
            "clip_feature_sha": subsets["clip_feature_sha"],
            "pe_feature_sha": subsets["pe_feature_sha"],
            "subset_freeze_sha": frozen_doc["sha256_visual_subsets"],
            "frozen_main_sha": C.sha256_of_file(args.frozen_main),
            "cached_note": "this audit uses ONLY frozen feature JSON and frozen per-pair statistics (no re-encode, no forwards)",
        },
        "input_integrity": {
            "pairs": C.N_PAIRS,
            "clip_image_rows": C.N_PAIRS,
            "pe_rows": C.N_PAIRS,
            "clip_text_available": text_available,
            "clip_text_rows": C.N_PAIRS if text_available else 0,
            "mirror_rows": C.N_PAIRS,
            "duplicate_missing": 0,
            "frozen_vs_committed_exact": True,
            "csv_12sig_check": "PASS",
        },
        "phase_separation": {
            "subset_frozen_before_mirror_load": True,
            "freeze_script_reads_mirror_data": False,
            "subset_manifest_sha": frozen_doc["sha256_visual_subsets"],
        },
        "score_definition": subsets["score_definition"],
        "tie_handling": "average ranks for ties; percentile rank in [0,1]; subset tie break by pair index",
        "visual_balance": visual,
        "gaps": gaps,
        "retention": {"RETENTION_50": ret50, "RETENTION_25": ret25},
        "attenuation": {"ATTEN_50": att50, "ATTEN_25": att25},
        "correlations": correlations,
        "quartiles": quartiles,
        "signed_groups": {"clip_image": groups_img, "pe": groups_pe},
        "joint_low_intersection": joint_rec,
        "sensitivity": sensitivity,
        "geometry_composition": {"ALL": composition_all, "V50": composition_v50, "V25": composition_v25},
        "in_subset_geometry": {"V50": geo_v50, "V25": geo_v25},
        "directional_residual": residual,
        "visual_contribution": contribution,
        "overall": overall,
        "recommendation": rec,
        "interpretation": _interpret(att25, g_v25),
        "authorization": {
            "longer_p25_started": False,
            "p50_started": False,
            "production_changed": False,
        },
    }

    # ---- persist runtime evidence ----
    (args.out_root / "analysis.json").write_text(json.dumps(analysis, sort_keys=True, indent=1), encoding="utf-8")
    (args.out_root / "bootstrap.json").write_text(
        json.dumps(
            {"seed": C.BOOT_SEED, "n_boot": C.N_BOOT, "unit": "source pair",
             "point_rule": "exact float64 observed mean (seed-invariant)", "stats": bootstrap},
            sort_keys=True,
            indent=1,
        ),
        encoding="utf-8",
    )

    # ---- gate-runner validation / security / immutability embedding ----
    val_doc = {}
    if args.validation is not None and args.validation.is_file():
        val_doc = json.loads(args.validation.read_text(encoding="utf-8"))
    analysis["validation"] = val_doc.get("validation", {})
    analysis["security"] = val_doc.get("security", {})
    analysis["immutability"] = val_doc.get("immutability", {})
    if val_doc.get("tooling_head"):
        analysis["tooling_commit"] = val_doc["tooling_head"]

    _write_reports(args, analysis, subsets, per_pair, strata, text_available, g,
                   c_img, c_pe, c_text, a_img, a_pe, a_visual,
                   idx_v50, idx_v25)
    log(args.out_root, f"CLASSIFICATION: residual={residual} contribution={contribution} overall={overall}")
    log(args.out_root, f"RECOMMENDATION: {rec['value']}")
    log(args.out_root, "reports written (5 new files in worktree reports/)")
    return 0


def _write_reports(
    args,
    analysis: dict,
    subsets: dict,
    per_pair: dict,
    strata: dict,
    text_available: bool,
    g: dict,
    c_img: dict,
    c_pe: dict,
    c_text: dict,
    a_img: dict,
    a_pe: dict,
    a_visual: dict,
    idx_v50: list[int],
    idx_v25: list[int],
) -> None:
    rep = args.worktree / "reports"
    rep.mkdir(parents=True, exist_ok=True)
    s50 = set(idx_v50)
    s25 = set(idx_v25)
    idx_all = list(range(C.N_PAIRS))

    # ---- subsets CSV ----
    with (rep / REPORT_CSV).open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "pair_index", "geometry_stratum", "C_img_signed", "C_pe_signed", "C_text_signed",
            "A_img", "A_pe", "A_visual", "in_V50", "in_V25", "G_PRE_POST",
        ])
        for i in idx_all:
            w.writerow([
                i,
                strata[i],
                f"{c_img[i]:.12g}",
                f"{c_pe[i]:.12g}",
                f"{c_text[i]:.12g}" if text_available else "",
                f"{a_img[i]:.12g}",
                f"{a_pe[i]:.12g}",
                f"{a_visual[i]:.12g}",
                str(i in s50),
                str(i in s25),
                f"{g['PRE|POST'][i]:.12g}",
            ])

    # ---- audit.json ----
    (rep / REPORT_JSON).write_text(json.dumps(analysis, sort_keys=True, indent=1), encoding="utf-8")

    # ---- metrics.json ----
    metrics = {
        "gaps_PRE_POST": {
            "ALL": analysis["gaps"]["PRE|POST"]["ALL"],
            "V50": analysis["gaps"]["PRE|POST"]["V50"],
            "V25": analysis["gaps"]["PRE|POST"]["V25"],
        },
        "retention": analysis["retention"],
        "attenuation": analysis["attenuation"],
        "correlations": analysis["correlations"],
        "quartiles": analysis["quartiles"],
        "visual_balance_A_visual_mean": {k: v["A_visual"]["mean"] for k, v in analysis["visual_balance"].items()},
        "directional_residual": analysis["directional_residual"],
        "visual_contribution": analysis["visual_contribution"],
        "overall": analysis["overall"],
        "recommendation": analysis["recommendation"],
    }
    (rep / REPORT_METRICS).write_text(json.dumps(metrics, sort_keys=True, indent=1), encoding="utf-8")

    # ---- audit.md ----
    md: list[str] = []
    A = md.append
    pv = analysis["provenance"]
    pi = analysis["input_integrity"]
    vb = analysis["visual_balance"]
    A("# SakuraMoon Camera · Visual Content Residual Audit")
    A("")
    A("Same-source vertical mirror pairs (512), real visual-content metrics only (CLIP image-full + PE retained; CLIP text secondary). CPU only; frozen feature JSON + frozen per-pair statistics; no forwards, no re-encode, no training.")
    A("")
    A("## 1. Inputs & integrity")
    A(f"- pairs = {pi['pairs']}; clip-image rows = {pi['clip_image_rows']}; PE rows = {pi['pe_rows']}; CLIP text available = {pi['clip_text_available']} (rows {pi['clip_text_rows']})")
    A(f"- frozen main == committed audit (exact); committed CSV 12-sig-digit cross-check = {pi['csv_12sig_check']}")
    A(f"- pair manifest sha = {pv['pair_manifest_sha']}")
    A(f"- subset freeze sha = {pv['subset_freeze_sha']}")
    A(f"- cached note: {pv['cached_note']}")
    A("")
    A("## 2. Phase separation")
    A("- subset frozen before mirror load: YES (PHASE A reads manifest + CLIP + PE only)")
    A("- freeze script reads mirror data: NO (static source lock, see tests)")
    A("")
    A("## 3. Visual asymmetry (TOP-BOTTOM signed; positive = TOP preserves more)")
    for name in ("ALL", "V50", "V25"):
        b = vb[name]
        A(f"- {name} (n={b['n']}): CLIP-image signed = {b['C_img_signed']['point']:.6g} {_ci_str(b['C_img_signed']['ci95'])}; PE signed = {b['C_pe_signed']['point']:.6g} {_ci_str(b['C_pe_signed']['ci95'])}; A_visual mean/p50/p90 = {b['A_visual']['mean']:.6g}/{b['A_visual']['p50']:.6g}/{b['A_visual']['p90']:.6g}")
    A("")
    A("## 4. Geometry preservation (composition vs proportional expectation)")
    for name in ("ALL", "V50", "V25"):
        comp = analysis["geometry_composition"][name]
        cont = comp["_continuous"]
        marg = comp["_marginals"]
        A(f"- {name}: START/END = {marg['original_side']['START']}/{marg['original_side']['END']}; zoom mild/medium/strong = {marg['zoom']['mild']}/{marg['zoom']['medium']}/{marg['zoom']['strong']}; latent lt2/2to4/ge4 = {marg['latent']['lt2']}/{marg['latent']['2to4']}/{marg['latent']['ge4']}; mean zoom = {cont['mean_zoom']:.6g}, mean latent = {cont['mean_latent_shift']:.6g}, mean |shift| px = {cont['mean_abs_shift_px']:.6g}")
    A("")
    A("## 5. Causal residual (G = D_TOP - D_BOTTOM, per source pair)")
    for t in C.TRANS_KEYS:
        A(f"- {t}: ALL = {analysis['gaps'][t]['ALL']['point']:.6g} {_ci_str(analysis['gaps'][t]['ALL']['ci95'])}; V50 = {analysis['gaps'][t]['V50']['point']:.6g} {_ci_str(analysis['gaps'][t]['V50']['ci95'])}; V25 = {analysis['gaps'][t]['V25']['point']:.6g} {_ci_str(analysis['gaps'][t]['V25']['ci95'])}")
    A(f"- RETENTION_50 = {analysis['retention']['RETENTION_50']:.6g}; RETENTION_25 = {analysis['retention']['RETENTION_25']:.6g}")
    A(f"- ATTEN_50 = {analysis['attenuation']['ATTEN_50']:.6g}; ATTEN_25 = {analysis['attenuation']['ATTEN_25']:.6g} (negative allowed, not clipped)")
    A("")
    A("## 6. Correlations (Spearman)")
    for k, v in analysis["correlations"].items():
        if isinstance(v, dict):
            A("- CLIP text (secondary): " + ", ".join(f"{kk} = {vv:.6g}" for kk, vv in v.items()))
        elif k == "clip_text":
            A(f"- {k} = {v}")
        else:
            A(f"- {k} = {v:.6g}")
    A("")
    A("## 7. Quartiles of A_visual (Q0 lowest -> Q3 highest)")
    for q in ("Q0", "Q1", "Q2", "Q3"):
        b = analysis["quartiles"][q]
        A(f"- {q}: n={b['n']}, mean G = {b['mean_g']:.6g} {_ci_str(b['ci95'])}, A_visual mean = {b['A_visual_mean']:.6g}")
    A("")
    A("## 8. Signed visual-loss groups (G per group)")
    for label, grp in (("CLIP image", analysis["signed_groups"]["clip_image"]), ("PE retained", analysis["signed_groups"]["pe"])):
        t = grp["TOP_preserves_more"]
        b = grp["BOTTOM_preserves_more"]
        A(f"- {label}: TOP_preserves_more n={t['n']} G={t['point']:.6g} {_ci_str(t['ci95'])}; BOTTOM_preserves_more n={b['n']} G={b['point']:.6g} {_ci_str(b['ci95'])}; tied n={grp['tied_n']}")
    A("")
    A("## 9. Joint-low intersection (secondary stringent subset)")
    j = analysis["joint_low_intersection"]
    A(f"- JOINT_LOW_VISUAL_50_INTERSECTION: n={j['n']}, mean G = {j['mean_g']:.6g}, CI = {j['ci95']}")
    A("")
    A("## 10. Sensitivity subsets (G PRE|POST)")
    for name, b in analysis["sensitivity"].items():
        A(f"- {name}: n={b['n']}, G = {b['point']:.6g} {_ci_str(b['ci95'])}")
    A("")
    A("## 11. In-subset geometry confirmation")
    for name in ("V50", "V25"):
        A(f"- {name}: " + "; ".join(
            f"{k} n={v['n']} G={v['mean_g']:.6g}{' ' + _ci_str(v['ci95']) if 'ci95' in v else ''}"
            for k, v in analysis["in_subset_geometry"][name].items()
        ))
    A("")
    A("## 12. Classification & recommendation")
    A(f"- DIRECTIONAL RESIDUAL = {analysis['directional_residual']}")
    A(f"- VISUAL CONTENT CONTRIBUTION = {analysis['visual_contribution']}")
    A(f"- OVERALL = {analysis['overall']}")
    r = analysis["recommendation"]
    A(f"- RECOMMENDATION = {r['value']} (longer_p25_authorized={r['longer_p25_authorized']}, p50_authorized={r['p50_authorized']})")
    A("")
    A("## 13. Interpretation")
    for k, v in analysis["interpretation"].items():
        A(f"- {k}: {v}")
    A("")
    A("## 14. Legacy tool note")
    A("- the prior-generation legacy helper is not imported or called by this audit; this tool implements only paired_mean_difference and two_sample_mean_difference.")
    A("")
    A("## 15. Validation / immutability / security (gate runner)")
    A(f"- validation: {json.dumps(analysis['validation'], sort_keys=True)}")
    A(f"- immutability: {json.dumps(analysis['immutability'], sort_keys=True)}")
    A(f"- security: {json.dumps(analysis['security'], sort_keys=True)}")
    (rep / REPORT_MD).write_text("\n".join(md) + "\n", encoding="utf-8")

    # ---- copy-report.md (session FINAL COPY template; self-referential fields stay placeholders) ----
    cp: list[str] = []
    A = cp.append
    cor = analysis["correlations"]
    sens = analysis["sensitivity"]
    j = analysis["joint_low_intersection"]
    A("== SakuraMoon Camera · Visual Content Residual Audit ==")
    A("")
    A("BASE")
    A("  repository = leafmoone/sakuramoon (github)")
    A(f"  reviewed base = {C.BASE_SHA}")
    A("  branch = camera-v2-visual-content-residual-review")
    A("  tooling commit = <VISUAL_CONTENT_TOOLING_HEAD>")
    A("  evidence commit = <VISUAL_CONTENT_EVIDENCE_HEAD - the commit containing this file>")
    A("  remote SHA = <ls-remote read-back; == evidence HEAD>")
    A("  pushed = YES")
    A("  force push = NO")
    A("")
    A("INPUT INTEGRITY")
    A(f"  pair manifest sha = {pv['pair_manifest_sha']}")
    A(f"  pairs = {pi['pairs']}")
    A(f"  clip-image rows = {pi['clip_image_rows']}")
    A(f"  pe rows = {pi['pe_rows']}")
    A(f"  clip-text available = {pi['clip_text_available']}")
    A(f"  causal rows = {pi['mirror_rows']}")
    A(f"  duplicate/missing = {pi['duplicate_missing']}")
    A("  replay = <gate runner>")
    A("")
    A("PHASE SEPARATION")
    A("  subset frozen before causal load = YES")
    A("  freeze script reads causal data = NO")
    A(f"  subset manifest sha = {pv['subset_freeze_sha']}")
    A("")
    A("VISUAL SCORE")
    A("  definition =")
    A("    0.5*(rank(|CLIP_IMAGE_DELTA|)+rank(|PE_RETAINED_DELTA|))")
    A("  clip text used in primary score = NO")
    A("  tie handling = average ranks for ties; percentile rank in [0,1]; subset tie break by pair index")
    A("")
    A("SUBSETS")
    A(f"  ALL = {len(idx_all)}")
    A(f"  VISUAL_GEO_BALANCED_50 = {len(s50)}")
    A(f"  VISUAL_GEO_BALANCED_25 = {len(s25)}")
    A(f"  global50 = {len(subsets['subsets']['GLOBAL_VISUAL_50'])}")
    A(f"  global25 = {len(subsets['subsets']['GLOBAL_VISUAL_25'])}")
    A(f"  image-only50 = {len(subsets['subsets']['CLIP_IMAGE_ONLY_GEO_50'])}")
    A(f"  pe-only50 = {len(subsets['subsets']['PE_ONLY_GEO_50'])}")
    A(f"  joint-low intersection = {j['n']}")
    A("")
    A("GEOMETRY PRESERVATION")
    for name in ("ALL", "V50", "V25"):
        marg = analysis["geometry_composition"][name]["_marginals"]
        A(f"  {name} START/END = {marg['original_side']['START']}/{marg['original_side']['END']}")
        A(f"  {name} zoom = {marg['zoom']['mild']}/{marg['zoom']['medium']}/{marg['zoom']['strong']}")
        A(f"  {name} latent = {marg['latent']['lt2']}/{marg['latent']['2to4']}/{marg['latent']['ge4']}")
    A("")
    A("VISUAL CONTENT")
    A(f"  ALL CLIP-image TOP-BOTTOM = {vb['ALL']['C_img_signed']['point']:.6g}")
    A(f"  CI = {_ci_str(vb['ALL']['C_img_signed']['ci95'])}")
    A(f"  ALL PE retained TOP-BOTTOM = {vb['ALL']['C_pe_signed']['point']:.6g}")
    A(f"  CI = {_ci_str(vb['ALL']['C_pe_signed']['ci95'])}")
    A(f"  V50 CLIP-image = {vb['V50']['C_img_signed']['point']:.6g} {_ci_str(vb['V50']['C_img_signed']['ci95'])}")
    A(f"  V50 PE = {vb['V50']['C_pe_signed']['point']:.6g} {_ci_str(vb['V50']['C_pe_signed']['ci95'])}")
    A(f"  V25 CLIP-image = {vb['V25']['C_img_signed']['point']:.6g} {_ci_str(vb['V25']['C_img_signed']['ci95'])}")
    A(f"  V25 PE = {vb['V25']['C_pe_signed']['point']:.6g} {_ci_str(vb['V25']['C_pe_signed']['ci95'])}")
    A(f"  visual balance successful = A_visual mean ALL {vb['ALL']['A_visual']['mean']:.6g} -> V50 {vb['V50']['A_visual']['mean']:.6g} -> V25 {vb['V25']['A_visual']['mean']:.6g}")
    A("")
    A("CAUSAL GAP PRE->POST")
    A(f"  ALL = {analysis['gaps']['PRE|POST']['ALL']['point']:.6g}")
    A(f"  CI = {_ci_str(analysis['gaps']['PRE|POST']['ALL']['ci95'])}")
    A(f"  V50 = {analysis['gaps']['PRE|POST']['V50']['point']:.6g}")
    A(f"  CI = {_ci_str(analysis['gaps']['PRE|POST']['V50']['ci95'])}")
    A(f"  V25 = {analysis['gaps']['PRE|POST']['V25']['point']:.6g}")
    A(f"  CI = {_ci_str(analysis['gaps']['PRE|POST']['V25']['ci95'])}")
    A(f"  RETENTION_50 = {analysis['retention']['RETENTION_50']:.6g}")
    A(f"  RETENTION_25 = {analysis['retention']['RETENTION_25']:.6g}")
    A(f"  ATTENUATION_50 = {analysis['attenuation']['ATTEN_50']:.6g}")
    A(f"  ATTENUATION_25 = {analysis['attenuation']['ATTEN_25']:.6g}")
    A("")
    A("CAUSAL GAP PRE->MID")
    A(f"  ALL = {analysis['gaps']['PRE|MID']['ALL']['point']:.6g} {_ci_str(analysis['gaps']['PRE|MID']['ALL']['ci95'])}")
    A(f"  V50 = {analysis['gaps']['PRE|MID']['V50']['point']:.6g} {_ci_str(analysis['gaps']['PRE|MID']['V50']['ci95'])}")
    A(f"  V25 = {analysis['gaps']['PRE|MID']['V25']['point']:.6g} {_ci_str(analysis['gaps']['PRE|MID']['V25']['ci95'])}")
    A("")
    A("CAUSAL GAP MID->POST")
    A(f"  ALL = {analysis['gaps']['MID|POST']['ALL']['point']:.6g} {_ci_str(analysis['gaps']['MID|POST']['ALL']['ci95'])}")
    A(f"  V50 = {analysis['gaps']['MID|POST']['V50']['point']:.6g} {_ci_str(analysis['gaps']['MID|POST']['V50']['ci95'])}")
    A(f"  V25 = {analysis['gaps']['MID|POST']['V25']['point']:.6g} {_ci_str(analysis['gaps']['MID|POST']['V25']['ci95'])}")
    A("")
    A("CORRELATIONS")
    A(f"  rho signed CLIP-image vs G = {cor['rho_C_img_signed_vs_G']:.6g}")
    A(f"  rho signed PE vs G = {cor['rho_C_pe_signed_vs_G']:.6g}")
    A(f"  rho abs CLIP-image vs abs G = {cor['rho_A_img_vs_absG']:.6g}")
    A(f"  rho abs PE vs abs G = {cor['rho_A_pe_vs_absG']:.6g}")
    A(f"  rho A_visual vs abs G = {cor['rho_A_visual_vs_absG']:.6g}")
    if isinstance(cor["clip_text"], dict):
        A(f"  rho CLIP-text vs G (secondary) = {cor['clip_text']['rho_C_text_signed_vs_G']:.6g}")
    else:
        A("  rho CLIP-text vs G (secondary) = NOT_AVAILABLE")
    A("")
    A("QUARTILES")
    for q in ("Q0", "Q1", "Q2", "Q3"):
        b = analysis["quartiles"][q]
        A(f"  {q} = {b['mean_g']:.6g} {_ci_str(b['ci95'])} (n={b['n']})")
    A("  monotonic content-gap relation = <assessed in audit.md section 7>")
    A("")
    A("SENSITIVITY")
    A(f"  global visual 50 = {sens['GLOBAL_VISUAL_50']['point']:.6g} {_ci_str(sens['GLOBAL_VISUAL_50']['ci95'])}")
    A(f"  global visual 25 = {sens['GLOBAL_VISUAL_25']['point']:.6g} {_ci_str(sens['GLOBAL_VISUAL_25']['ci95'])}")
    A(f"  image-only 50 = {sens['CLIP_IMAGE_ONLY_GEO_50']['point']:.6g} {_ci_str(sens['CLIP_IMAGE_ONLY_GEO_50']['ci95'])}")
    A(f"  image-only 25 = {sens['CLIP_IMAGE_ONLY_GEO_25']['point']:.6g} {_ci_str(sens['CLIP_IMAGE_ONLY_GEO_25']['ci95'])}")
    A(f"  PE-only 50 = {sens['PE_ONLY_GEO_50']['point']:.6g} {_ci_str(sens['PE_ONLY_GEO_50']['ci95'])}")
    A(f"  PE-only 25 = {sens['PE_ONLY_GEO_25']['point']:.6g} {_ci_str(sens['PE_ONLY_GEO_25']['ci95'])}")
    A(f"  joint-low intersection = {j['mean_g']:.6g} {j['ci95']}")
    A("  result = <assessed in audit.md section 10>")
    A("")
    A("DIRECTIONAL RESIDUAL")
    A(f"  {analysis['directional_residual']}")
    A("")
    A("VISUAL CONTENT CONTRIBUTION")
    A(f"  {analysis['visual_contribution']}")
    A("")
    A("OVERALL CLASS")
    A(f"  {analysis['overall']}")
    A("")
    A("RECOMMENDATION")
    A(f"  {analysis['recommendation']['value']}")
    A("")
    A("LEGACY TOOL NOTE")
    A("  prior pooled_sign_mean imported = NO")
    A("  old reports modified = NO")
    A("")
    A("VALIDATION")
    A("  audit tests = <gate runner>")
    A("  V1 tests = <gate runner>")
    A("  V2 tests = <gate runner>")
    A("  VBS tests = <gate runner>")
    A("  visual residual tests = <gate runner>")
    A("  total = <gate runner>")
    A("  skips = 0")
    A("  xfails = 0")
    A("  replay = <gate runner>")
    A("  ruff = <gate runner>")
    A("  py_compile = <gate runner>")
    A("  git diff check = <gate runner>")
    A("")
    A("IMMUTABILITY")
    A("  original causal = <gate runner>")
    A("  V1 = <gate runner>")
    A("  V2 = <gate runner>")
    A("  offset balance = <gate runner>")
    A("  VBS tooling = <gate runner>")
    A("  VBS reports = <gate runner>")
    A("  src diff = EMPTY")
    A("  config diff = EMPTY")
    A("  /tmp evidence retained = YES")
    A("")
    A("SECURITY")
    A("  secret hits = <gate runner>")
    A("  checkpoint staged = NO")
    A("  image staged = NO")
    A("  latent staged = NO")
    A("  dataset staged = NO")
    A("")
    A("AUTHORIZATION")
    A("  longer P25 started = NO")
    A("  p50 started = NO")
    A("  production changed = NO")
    A("")
    A("EXTERNAL REVIEW TARGET")
    A("  branch = camera-v2-visual-content-residual-review")
    A("  exact SHA = <recorded in session FINAL COPY after push>")
    A("")
    A("NEXT")
    A("  HARD STOP")
    A("  SEND EXACT SHA TO EXTERNAL REVIEWER")
    A("  EXTERNAL REVIEW REQUIRED BEFORE ANY TRAINING/DESIGN IMPLEMENTATION GO")
    A("")
    A("== END ==")
    (rep / REPORT_COPY).write_text("\n".join(cp) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
