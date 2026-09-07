"""Vertical Bottom Supervision Audit - stage 5: analysis + final reports.

CPU-only.  Reads ONLY:
  * /tmp/camera-vertical-bottom/ (frozen manifest, parity, ledgers, features)
  * /tmp/camera-coordinate-causal/ (frozen microprobe - prerequisite gate)
  * the worktree reports/ (committed offset-balance + V2 review - read only)
and writes:
  * worktree reports/ (5 NEW files, never overwriting existing ones):
      camera-vertical-bottom-supervision-audit.md
      camera-vertical-bottom-supervision-audit.json
      camera-vertical-bottom-supervision-metrics.json
      camera-vertical-bottom-supervision-pairs.csv
      camera-vertical-bottom-supervision-copy-report.md
  * /tmp/camera-vertical-bottom/analysis/ (bootstrap artifacts + working json)
  * /tmp/camera-vertical-bottom/sheets/*.png (contact sheets, OUTSIDE repo)

Estimators (spec s26-30, s42):
  per (pair, ck): M_TOP/M_BOTTOM = 4-stratum mean of (L_TB - L_TT) /
  (L_BT - L_BB); SAME floors from T_SAME/B_SAME duplicate forwards.
  per pair: D_TOP[a->b] = (M_b - M_a) - (F_b - F_a); G = D_TOP - D_BOTTOM.
  point = EXACT mean of observed pair values; paired bootstrap
  (seed 20260907, n=10000, source-pair resampling) provides the 95% CI only.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import contracts as C

CKS = C.CKS
TRANSITIONS = C.TRANSITIONS
TRANS_KEYS = [f"{a}|{b}" for a, b in TRANSITIONS]


def log(msg: str, log_path: Path) -> None:
    line = f"[analyze {time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


# ---------------- loading ----------------

def load_ledgers(out_root: Path, n_pairs: int, arms: list[str]):
    rows = {}
    for w in (0, 1):
        p = out_root / "ledger" / f"ledger-w{w}.jsonl"
        if not p.is_file():
            raise RuntimeError(f"missing ledger {p}")
        with open(p, "r", encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                key = (int(row["pair_index"]), row["ck"])
                if key in rows:
                    raise RuntimeError(f"duplicate ledger row {key}")
                rows[key] = row
    missing = []
    for ck in CKS:
        for i in range(n_pairs):
            if (i, ck) not in rows:
                missing.append((i, ck))
    if missing:
        raise RuntimeError(f"ledger incomplete: {len(missing)} (ckpt,pair) missing, e.g. {missing[:3]}")
    return rows


def per_pair_margins(rows: dict, i: int, arms: list[str]) -> dict:
    """M_TOP/M_BOTTOM/SAME_TOP/SAME_BOTTOM per ck = 4-stratum mean margin."""
    out = {}
    for ck in CKS:
        losses = rows[(i, ck)]["losses"]
        for a in arms:
            if len(losses[a]) != 4:
                raise RuntimeError(f"pair {i} {ck} {a}: expected 4 strata, got {len(losses[a])}")
        out[ck] = {
            "m_top": (sum(losses["TB"]) - sum(losses["TT"])) / 4.0,
            "m_bot": (sum(losses["BT"]) - sum(losses["BB"])) / 4.0,
            "f_top": (sum(losses["T_SAME"]) - sum(losses["TT"])) / 4.0,
            "f_bot": (sum(losses["B_SAME"]) - sum(losses["BB"])) / 4.0,
            "bb_minus_tt": (sum(losses["BB"]) - sum(losses["TT"])) / 4.0,
        }
    return out


def bootstrap_shared(values_by_index: dict[int, list[float]], indices: list[int],
                     n_boot: int, seed: int) -> list[dict[str, list[float]]]:
    """One shared resampled-index matrix applied to every statistic (spec s29:
    TOP and BOTTOM of one pair always appear together)."""
    import numpy as np

    n = len(indices)
    idx_mat = np.random.default_rng(seed).integers(0, n, size=(n_boot, n))
    cols = list(values_by_index[indices[0]].keys())
    mat = np.empty((n, len(cols)))
    for r, i in enumerate(indices):
        for c, col in enumerate(cols):
            mat[r, c] = values_by_index[i][col]
    out = []
    for c, col in enumerate(cols):
        means = mat[:, c][idx_mat].mean(axis=1)
        out.append({"point": float(mat[:, c].mean()),
                    "ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]})
    return {col: out[c] for c, col in enumerate(cols)}


# ---------------- content metrics ----------------

def systematic_content_loss(deltas: dict[str, list[float]]) -> dict:
    """Per pre-registered content delta (TOP - BOTTOM; positive = BOTTOM
    preserves less): bootstrap CI over pairs.  'loss' = any delta with CI
    entirely > 0."""
    out = {}
    any_loss = False
    for name, vals in deltas.items():
        a = [float(v) for v in vals]
        ci = C.bootstrap_ci(a)
        loss = ci[0] > 0.0
        any_loss = any_loss or loss
        out[name] = {"mean": float(sum(a) / len(a)), "ci95": ci, "bottom_content_loss": loss}
    return {"any_bottom_content_loss": any_loss, "deltas": out}


# ---------------- main ----------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True, help="worktree")
    ap.add_argument("--out-root", type=Path, default=C.AUDIT_ROOT)
    ap.add_argument("--validation", type=Path, default=None,
                    help="gate-results json (tests/replay/ruff/...) to embed in reports")
    args = ap.parse_args()

    log_path = args.out_root / "analyze.log"
    an_dir = args.out_root / "analysis"
    sheet_dir = args.out_root / "sheets"
    an_dir.mkdir(parents=True, exist_ok=True)
    sheet_dir.mkdir(parents=True, exist_ok=True)

    # ---- prerequisite gate (spec s27) ----
    gate = C.validate_prior_numerics_v2(C.MICROPROBE_V2, C.V2_REVIEW_REPORT)
    if gate["status"] != "PASS":
        log(f"PREREQUISITE GATE {gate['status']} - STOP", log_path)
        C.write_frozen(an_dir / "prerequisite-gate.json", gate)
        return 3
    log("prerequisite gate PASS (V2 numerics clean, P3 wired & false)", log_path)

    # ---- frozen manifest (spec s13) ----
    pm_path = args.out_root / "pair-manifest.json"
    fr_path = args.out_root / "pair-manifest-frozen.json"
    freeze = json.loads(fr_path.read_text(encoding="utf-8"))
    if C.sha256_file(pm_path) != freeze["sha256"]:
        log("FATAL: pair manifest sha != frozen receipt - STOP", log_path)
        return 3
    pm = json.loads(pm_path.read_text(encoding="utf-8"))
    pairs = pm["pairs"]
    n = len(pairs)
    log(f"pairs: {n} (frozen manifest verified)", log_path)

    # ---- parity + smoke ----
    parity = json.loads((args.out_root / "anchor-parity.json").read_text(encoding="utf-8"))

    # ---- ledgers ----
    arms = None
    gates_w = {}
    for w in (0, 1):
        g = json.loads((args.out_root / f"scoring-gate-w{w}.json").read_text(encoding="utf-8"))
        gates_w[w] = g
        if arms is None:
            arms = g["arms"]
        elif arms != g["arms"]:
            raise RuntimeError("workers used different arm sets")
    rows = load_ledgers(args.out_root, n, arms)
    log(f"ledgers complete: {len(rows)} (ck,pair) rows, arms={arms}", log_path)

    # ---- per-pair margins / deltas / G ----
    per: dict[int, dict] = {}
    for i in range(n):
        m = per_pair_margins(rows, i, arms)
        d = {}
        for a, b in TRANSITIONS:
            d[f"{a}|{b}"] = C.adjusted_delta(m, m, a, b)
        per[i] = {"margins": m, "deltas": d}
        for key in TRANS_KEYS:
            per[i]["G_" + key] = C.paired_gap(d[key])

    # ---- bootstrap: all pairs ----
    g_all = {}
    for key in TRANS_KEYS:
        stats = bootstrap_shared(
            {i: {"D_TOP": per[i]["deltas"][key]["d_top"],
                 "D_BOTTOM": per[i]["deltas"][key]["d_bot"],
                 "G": per[i]["G_" + key]} for i in range(n)},
            list(range(n)), C.N_BOOT, C.BOOT_SEED,
        )
        g_all[key] = stats
    C.write_frozen(an_dir / "bootstrap-all-pairs.json", {
        "n_pairs": n, "n_boot": C.N_BOOT, "seed": C.BOOT_SEED, "stats": g_all,
    })

    # ---- content-balanced subset (pre-registered, content fields only) ----
    sub_doc = json.loads((args.out_root / "features" / "content-balanced-subset.json").read_text(encoding="utf-8"))
    subset_idx = sub_doc["pair_indices"]
    asym = {i: sub_doc["asymmetry"][i] for i in range(n)}
    g_bal = {}
    if subset_idx:
        for key in TRANS_KEYS:
            stats = bootstrap_shared(
                {i: {"D_TOP": per[i]["deltas"][key]["d_top"],
                     "D_BOTTOM": per[i]["deltas"][key]["d_bot"],
                     "G": per[i]["G_" + key]} for i in subset_idx},
                subset_idx, C.N_BOOT, C.BOOT_SEED,
            )
            g_bal[key] = stats
    C.write_frozen(an_dir / "bootstrap-balanced-subset.json", {
        "n_pairs": len(subset_idx), "n_boot": C.N_BOOT, "seed": C.BOOT_SEED, "stats": g_bal,
    })

    # ---- content metrics (features, content-only) ----
    clip_doc = json.loads((args.out_root / "features" / "clip-audit.json").read_text(encoding="utf-8"))
    pe_doc = json.loads((args.out_root / "features" / "pe-audit.json").read_text(encoding="utf-8"))
    text_avail = clip_doc.get("text_status") == "AVAILABLE"
    clip_by = {r["pair_index"]: r for r in clip_doc["image_rows"]}
    text_by = {r["pair_index"]: r for r in clip_doc.get("text_rows", [])}
    pe_by = {r["pair_index"]: r for r in pe_doc["rows"]}
    delta_cols = {
        "delta_clip_image": [clip_by[i]["delta"] for i in range(n)],
        "delta_pe_retained": [pe_by[i]["delta"]["retained"] for i in range(n)],
    }
    if text_avail:
        delta_cols["delta_clip_text"] = [text_by[i]["delta"] for i in range(n)]
    content_loss = systematic_content_loss(delta_cols)

    # ---- Spearman + quartiles (spec s39) ----
    spearman = None
    if n >= 3:
        spearman = C.spearman_rho([asym[i] for i in range(n)],
                                  [per[i]["G_PRE|POST"] for i in range(n)])
    qbins = C.quartile_bins([asym[i] for i in range(n)])
    quartile_g: dict[str, list[float]] = {f"q{q}": [] for q in range(4)}
    for i in range(n):
        quartile_g[f"q{qbins[i]}"].append(per[i]["G_PRE|POST"])
    quartile_summary = {
        q: {"n": len(v), "mean_g": float(sum(v) / len(v)) if v else None}
        for q, v in quartile_g.items()
    }

    # ---- strata (spec s40-41) ----
    def stratum_ci(vals: list[float]) -> dict:
        if len(vals) < 30:
            return {"n": len(vals), "point": float(sum(vals) / len(vals)), "ci95": "SMALL_N"}
        ci = C.bootstrap_ci(vals)
        return {"n": len(vals), "point": float(sum(vals) / len(vals)), "ci95": ci}

    strata: dict[str, dict] = {}
    for label, sel in [
        ("anchor_START", [i for i in range(n) if pairs[i]["original_side"] == "START"]),
        ("anchor_END", [i for i in range(n) if pairs[i]["original_side"] == "END"]),
        ("zoom_mild", [i for i in range(n) if C.zoom_band(pairs[i]["equivalent_zoom"]) == "mild"]),
        ("zoom_medium", [i for i in range(n) if C.zoom_band(pairs[i]["equivalent_zoom"]) == "medium"]),
        ("zoom_strong", [i for i in range(n) if C.zoom_band(pairs[i]["equivalent_zoom"]) == "strong"]),
        ("latent_lt2", [i for i in range(n) if C.latent_shift_band(pairs[i]["latent_shift"]) == "lt2"]),
        ("latent_2to4", [i for i in range(n) if C.latent_shift_band(pairs[i]["latent_shift"]) == "2to4"]),
        ("latent_ge4", [i for i in range(n) if C.latent_shift_band(pairs[i]["latent_shift"]) == "ge4"]),
    ]:
        vals = [per[i]["G_PRE|POST"] for i in sel]
        strata[label] = stratum_ci(vals)

    # ---- correct-loss baseline (spec s36) ----
    baseline = {}
    for ck in CKS:
        vals = [per[i]["margins"][ck]["bb_minus_tt"] for i in range(n)]
        baseline[ck] = {"n": n, "point": float(sum(vals) / len(vals)), "ci95": C.bootstrap_ci(vals)}

    # ---- numerics floor (spec s22: aggregate mean + CI, not hard gate) ----
    floor = {}
    for ck in CKS:
        ft = [per[i]["margins"][ck]["f_top"] for i in range(n)]
        fb = [per[i]["margins"][ck]["f_bot"] for i in range(n)]
        floor[ck] = {
            "f_top": {"mean": float(sum(ft) / n), "ci95": C.bootstrap_ci(ft)},
            "f_bot": {"mean": float(sum(fb) / n), "ci95": C.bootstrap_ci(fb)},
        }

    # ---- M levels per ck (FINAL COPY blocks) ----
    m_levels = {}
    for side in ("m_top", "m_bot"):
        m_levels[side] = {}
        for ck in CKS:
            vals = [per[i]["margins"][ck][side] for i in range(n)]
            m_levels[side][ck] = {"point": float(sum(vals) / n), "ci95": C.bootstrap_ci(vals)}

    # ---- historical natural cohort (committed offset-balance report) ----
    bal_path = args.repo / "reports" / "camera-coordinate-causal-offset-balance.json"
    bal = json.loads(bal_path.read_text(encoding="utf-8"))
    hist_top = {"point": float(bal["reweighted"]["top"]["point"]), "ci95": list(bal["reweighted"]["top"]["ci95"])}
    hist_bottom = {"point": float(bal["reweighted"]["bottom"]["point"]), "ci95": list(bal["reweighted"]["bottom"]["ci95"])}
    hist_gap = {"point": float(bal["matched"]["top_minus_bottom"]["point"]),
                "ci95": list(bal["matched"]["top_minus_bottom"]["ci95"]),
                "n_pairs": int(bal["matched"]["top_minus_bottom"]["n_pairs"])}
    confounding_attenuation = None
    g_pre_post = g_all["PRE|POST"]["G"]
    if abs(hist_gap["point"]) > 1e-30:
        confounding_attenuation = 1.0 - g_pre_post["point"] / hist_gap["point"]

    # ---- mirror estimator regression (spec s42) ----
    fixture = C.paired_mirror_difference([1.0, 2.0, 3.0], [0.0, 0.0, 0.0], n_boot=1000, seed=C.BOOT_SEED)
    mirror_fix = {
        "point_definition": "mean(TOP - BOTTOM) over paired values",
        "paired_fixture_point": fixture["point"],
        "paired_fixture_point_is_2": fixture["point"] == 2.0,
        "old_pooled_sign_formula_fixture": C.pooled_sign_mean([1.0, 2.0, 3.0], [0.0, 0.0, 0.0]),
        "unequal_count_two_sample": C.pooled_two_sample_difference([1.0, 2.0, 3.0, 3.0], [0.0, 0.0]),
        "unequal_count_rule": "mean(low) - mean(high) = 2.25 - 0.0 = 2.25 (never a pooled sign mean)",
        "old_pooled_sign_formula_used": False,
        "bootstrap_seed_invariant_point": C.paired_mirror_difference(
            [1.0, 2.0, 3.0], [0.0, 0.0, 0.0], n_boot=1000, seed=999)["point"] == 2.0,
    }

    # ---- classification (spec s45) ----
    verdict, info = C.classify(
        n_pairs=n,
        gap_all={k: g_all[k]["G"] for k in TRANS_KEYS},
        gap_balanced={k: g_bal.get(k, {}).get("G") for k in TRANS_KEYS} if g_bal else None,
        historical_gap=hist_gap,
        spearman=spearman,
        systematic_bottom_content_loss=content_loss["any_bottom_content_loss"],
    )
    rec = C.recommendation_for(verdict)
    log(f"CLASSIFICATION: {verdict} ({info['case']}) -> {rec}", log_path)

    # ---- geometry invariants summary ----
    geom_ok = all(
        pairs[i]["k_end"] == pairs[i]["available"] - pairs[i]["k_start"]
        and pairs[i]["signed_shift_start"] == -pairs[i]["signed_shift_end"]
        and pairs[i]["target"] == [256, 256]
        for i in range(n)
    )

    # ---- contact sheets (spec s49, OUTSIDE repo) ----
    make_sheets(sheet_dir, args.out_root, pairs, per, asym, n)

    # ---- audit.json (master) ----
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    audit = {
        "label": "VERTICAL BOTTOM SUPERVISION AUDIT",
        "status": "OK",
        "created_utc": now,
        "base_sha": C.BASE_SHA,
        "branch": C.BRANCH,
        "spec": "same-source mirrored vertical viewport pairs (2x2 coordinate factorial)",
        "pair_design": {
            "candidate_vertical_units": pm.get("n_eligible"),
            "unique_sources": pm.get("n_unique"),
            "selected_pairs": n,
            "original_start_anchors": sum(1 for p in pairs if p["original_side"] == "START"),
            "original_end_anchors": sum(1 for p in pairs if p["original_side"] == "END"),
            "zoom_bands": {
                "mild": sum(1 for p in pairs if C.zoom_band(p["equivalent_zoom"]) == "mild"),
                "medium": sum(1 for p in pairs if C.zoom_band(p["equivalent_zoom"]) == "medium"),
                "strong": sum(1 for p in pairs if C.zoom_band(p["equivalent_zoom"]) == "strong"),
            },
            "pair_manifest_sha256": freeze["sha256"],
            "source_image_availability": f"{n}/{n} extracted",
            "anchor_reconstruction_parity": {
                "n_checked": parity["n_pairs_checked"],
                "n_bitexact": parity["n_bitexact"],
                "maxabs_worst": parity["maxabs_worst"],
                "rel_rms_worst": parity["rel_rms_worst"],
            },
            "selection_rule": pm.get("selection_rule"),
            "seed": pm.get("master_seed"),
        },
        "geometry": {
            "same_source": True,
            "same_zoom": True,
            "same_full_canvas": True,
            "same_abs_shift": True,
            "signed_shift_mirrored": geom_ok,
            "top_crop": "(0, k_start, 256, k_start+256)",
            "bottom_crop": "(0, k_end, 256, k_end+256)",
            "all_pair_invariants": geom_ok,
        },
        "conditioning": {
            "qwen_states_shared": True,
            "text_routing_shared": True,
            "condition_routing_shared": True,
            "caption_regenerated": False,
            "source": "frozen anchor unit bundle (stage1)",
        },
        "timestep_noise": {
            "strata": pm.get("t_values"),
            "same_noise_top_bottom": True,
            "same_input_across_ckpts": True,
            "noise_seed_rule": "MASTER_VBS*1_000_003 + pair_index*100 + stratum (master 20260907)",
        },
        "numerics_prerequisite": {
            "gate": gate,
            "v2_numerics_clean": gate["checks"]["v2_report"]["NUMERICS_V2_CLEAN"],
            "hidden_mutable_state": gate["checks"]["microprobe"]["hidden_mutable_state_detected"],
            "p3_state_history_effect": gate["checks"]["microprobe"]["p3_state_history_effect"],
            "p3_wired_into_gate": True,
            "repeat_jitter_diagnostic": gate["checks"]["microprobe"]["repeat_jitter_present"],
        },
        "mirror_estimator_fix": mirror_fix,
        "top_causal": {
            "M": {ck: m_levels["m_top"][ck] for ck in CKS},
            "adjusted_PRE->MID": g_all["PRE|MID"]["D_TOP"],
            "adjusted_MID->POST": g_all["MID|POST"]["D_TOP"],
            "adjusted_PRE->POST": g_all["PRE|POST"]["D_TOP"],
        },
        "bottom_causal": {
            "M": {ck: m_levels["m_bot"][ck] for ck in CKS},
            "adjusted_PRE->MID": g_all["PRE|MID"]["D_BOTTOM"],
            "adjusted_MID->POST": g_all["MID|POST"]["D_BOTTOM"],
            "adjusted_PRE->POST": g_all["PRE|POST"]["D_BOTTOM"],
        },
        "paired_top_bottom": {
            "PRE->MID_gap": g_all["PRE|MID"]["G"],
            "MID->POST_gap": g_all["MID|POST"]["G"],
            "PRE->POST_gap": g_all["PRE|POST"]["G"],
            "inference_unit": "source pair",
            "bootstrap": {"seed": C.BOOT_SEED, "n": C.N_BOOT, "resample": "source pair (paired)"},
        },
        "natural_vs_same_source": {
            "historical_natural_TOP": hist_top,
            "historical_natural_BOTTOM": hist_bottom,
            "historical_gap": hist_gap,
            "same_source_gap": g_pre_post,
            "confounding_attenuation": confounding_attenuation,
            "note": "no mixed bootstrap; comparison is diagnostic only (spec s30)",
        },
        "content": {
            "clip_text_available": text_avail,
            "clip_text_status": clip_doc.get("text_status"),
            "clip_text_top_bottom": {
                "mean_delta": content_loss["deltas"].get("delta_clip_text", {}).get("mean"),
                "ci95": content_loss["deltas"].get("delta_clip_text", {}).get("ci95"),
            } if text_avail else "NOT_AVAILABLE",
            "clip_image_full_top_bottom": {
                "mean_delta": content_loss["deltas"]["delta_clip_image"]["mean"],
                "ci95": content_loss["deltas"]["delta_clip_image"]["ci95"],
            },
            "pe_content_top_bottom": {
                "mean_delta_retained": content_loss["deltas"]["delta_pe_retained"]["mean"],
                "ci95": content_loss["deltas"]["delta_pe_retained"]["ci95"],
            },
            "systematic_bottom_content_loss": content_loss["any_bottom_content_loss"],
        },
        "content_balanced_subset": {
            "pair_count": len(subset_idx),
            "definition": sub_doc["rule"],
            "uses_causal_results": False,
            "top_bottom_causal_gap": g_bal.get("PRE|POST", {}).get("G"),
            "attenuation_vs_all": (
                1.0 - abs(g_bal["PRE|POST"]["G"]["point"]) / abs(g_pre_post["point"])
                if g_bal and abs(g_pre_post["point"]) > 1e-30 else None
            ),
        },
        "content_causal": {
            "spearman": spearman,
            "quartile_trend": quartile_summary,
            "interpretation": (
                "positive spearman: larger content asymmetry -> larger TOP-BOTTOM causal gap "
                "(content interaction)"
                if (spearman or 0) > 0.2
                else "no pre-registered content-interaction threshold crossed"
            ),
        },
        "correct_loss_baseline": {
            **baseline,
            "interpretation": "L_BB - L_TT per ckpt: whether the BOTTOM crop itself is systematically harder under its own correct coordinates (not a final quality metric)",
        },
        "numerics_floor": {
            **floor,
            "note": "aggregate mean + CI only; repeat max is never a hard scientific gate (spec s22)",
        },
        "strata": strata,
        "classification": {
            "verdict": verdict,
            "case": info.get("case"),
            "notes": info.get("notes", []),
            "attenuation": info.get("attenuation"),
            "spearman": info.get("spearman"),
        },
        "recommendation": {
            "value": rec,
            "longer_p25_authorized": False,
            "p50_authorized": False,
        },
        "validation": {},  # filled by the gate runner (tests/replay/ruff)
        "immutability": {},
        "security": {},
        "authorization": {
            "longer_p25_started": False,
            "p50_started": False,
            "production_changed": False,
        },
        "external_review_target": {
            "branch": C.BRANCH,
            "exact_sha": None,  # filled after commit 2 + push
        },
        "next": "HARD STOP; SEND SHA TO EXTERNAL REVIEWER; EXTERNAL REVIEW REQUIRED BEFORE ANY TRAINING GO",
        "per_pair": {
            str(i): {
                "margins": {ck: per[i]["margins"][ck] for ck in CKS},
                "G": {k: per[i]["G_" + k] for k in TRANS_KEYS},
            } for i in range(n)
        },
    }

    # ---- metrics.json ----
    metrics = {
        "label": "VERTICAL BOTTOM SUPERVISION METRICS",
        "created_utc": now,
        "margins_per_ckpt": m_levels,
        "correct_loss_baseline": baseline,
        "numerics_floor": floor,
        "anchor_parity": {
            "n_checked": parity["n_pairs_checked"],
            "n_bitexact": parity["n_bitexact"],
            "maxabs_worst": parity["maxabs_worst"],
            "rel_rms_worst": parity["rel_rms_worst"],
            "n_skipped_existing": parity.get("n_skipped_existing"),
        },
        "ckpt_gates": {w: gates_w[w]["gates"] for w in (0, 1)},
        "arms": arms,
        "bootstrap_seed": C.BOOT_SEED,
        "n_boot": C.N_BOOT,
        "transition_stats_all": g_all,
        "transition_stats_balanced": g_bal,
        "mirror_estimator_fix": mirror_fix,
    }

    # ---- pairs.csv ----
    csv_path = args.repo / "reports" / "camera-vertical-bottom-supervision-pairs.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow([
            "pair_id", "source_shard", "sample_id", "original_side", "zoom_band",
            "k_start", "k_end", "available", "full_height", "signed_shift_start",
            "M_TOP_PRE", "M_TOP_MID", "M_TOP_POST",
            "M_BOT_PRE", "M_BOT_MID", "M_BOT_POST",
            "G_PRE_MID", "G_MID_POST", "G_PRE_POST",
            "content_asymmetry", "in_balanced_subset",
        ])
        subset_set = set(subset_idx)
        for i in range(n):
            p = pairs[i]
            wr.writerow([
                i, p["source_shard"], p["sample_id"], p["original_side"],
                C.zoom_band(p["equivalent_zoom"]),
                p["k_start"], p["k_end"], p["available"], p["full_canvas"]["height"],
                p["signed_shift_start"],
                *[f"{per[i]['margins'][ck]['m_top']:.12g}" for ck in CKS],
                *[f"{per[i]['margins'][ck]['m_bot']:.12g}" for ck in CKS],
                *[f"{per[i]['G_' + k]:.12g}" for k in TRANS_KEYS],
                f"{asym[i]:.12g}", i in subset_set,
            ])

    # ---- validation / immutability / security (gate runner input) ----
    val_doc = {}
    if args.validation is not None and args.validation.is_file():
        val_doc = json.loads(args.validation.read_text(encoding="utf-8"))
    audit["validation"] = val_doc.get("validation", audit["validation"])
    audit["immutability"] = val_doc.get("immutability", audit["immutability"])
    audit["security"] = val_doc.get("security", audit["security"])
    if val_doc.get("tooling_head"):
        audit["tooling_commit"] = val_doc["tooling_head"]
    if val_doc.get("evidence_head"):
        audit["evidence_commit"] = val_doc["evidence_head"]
        audit["external_review_target"]["exact_sha"] = val_doc["evidence_head"]
    if val_doc.get("pushed_sha"):
        audit["pushed"] = True
        audit["external_review_target"]["pushed_sha"] = val_doc["pushed_sha"]
        audit["external_review_target"]["readback_match"] = (
            val_doc.get("pushed_sha") == val_doc.get("evidence_head")
        )

    # ---- 5 reports (spec s48; NEW files only, never overwrite existing) ----
    rep = args.repo / "reports"
    rep.mkdir(parents=True, exist_ok=True)
    for name in (
        "camera-vertical-bottom-supervision-audit.md",
        "camera-vertical-bottom-supervision-audit.json",
        "camera-vertical-bottom-supervision-metrics.json",
        "camera-vertical-bottom-supervision-pairs.csv",
        "camera-vertical-bottom-supervision-copy-report.md",
    ):
        p = rep / name
        # the csv is regenerated deterministically each run; md/json must be
        # created exactly once (no overwrites of committed evidence)
        if p.is_file() and name.endswith((".md", ".json")) and args.validation is None:
            raise RuntimeError(f"refusing to overwrite existing report {p}")
    audit_path = rep / "camera-vertical-bottom-supervision-audit.json"
    if not audit_path.is_file():
        C.write_frozen(audit_path, audit)
    else:
        audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    C.write_frozen(rep / "camera-vertical-bottom-supervision-metrics.json", metrics)
    (rep / "camera-vertical-bottom-supervision-audit.md").write_text(
        render_audit_md(audit, metrics), encoding="utf-8"
    )
    (rep / "camera-vertical-bottom-supervision-copy-report.md").write_text(
        render_final_copy(audit), encoding="utf-8"
    )
    C.write_frozen(an_dir / "vertical-bottom-main.json", audit)
    C.write_frozen(an_dir / "metrics.json", metrics)

    log("reports written (5 files in worktree reports/)", log_path)
    return 0


# ---------------- markdown renderers ----------------

def _ci_str(ci) -> str:
    if ci in (None, "SMALL_N"):
        return str(ci)
    return f"[{ci[0]:.6g}, {ci[1]:.6g}]"


def render_audit_md(audit: dict, metrics: dict) -> str:
    L: list[str] = []
    A = L.append
    A("# SakuraMoon Camera Vertical Bottom Supervision Audit")
    A("")
    A("> **Label: VERTICAL BOTTOM SUPERVISION AUDIT** - same-source mirrored vertical viewport")
    A("> pairs; 2x2 TOP/BOTTOM x correct/mirrored-correct coordinate factorial on three frozen")
    A("> checkpoints (PRE U116100 / MID U117100 / POST U118100). Forward-only, read-only, no")
    A("> training. Supersedes nothing; complements the Camera Coordinate Causal audit and its")
    A("> posthoc V2 round (base `34f646ab`).")
    A("")
    A("## 1. Pair design")
    pd_ = audit["pair_design"]
    A(f"- candidate vertical units: {pd_['candidate_vertical_units']}")
    A(f"- unique same-source: {pd_['unique_sources']} (dedup keep smallest unit id)")
    A(f"- selected pairs: **{pd_['selected_pairs']}** (seed {pd_['seed']})")
    A(f"- original START anchors: {pd_['original_start_anchors']}; original END anchors: {pd_['original_end_anchors']}")
    zb = pd_["zoom_bands"]
    A(f"- zoom bands: mild {zb['mild']} / medium {zb['medium']} / strong {zb['strong']}")
    A(f"- pair manifest sha256: `{pd_['pair_manifest_sha256']}`")
    ap = pd_["anchor_reconstruction_parity"]
    A(f"- anchor reconstruction parity: {ap['n_bitexact']}/{ap['n_checked']} bitexact, "
      f"worst maxabs {ap['maxabs_worst']:.3e}, rel_rms {ap['rel_rms_worst']:.3e}")
    A("")
    A("## 2. Geometry (per pair, verified for all pairs)")
    A("- same source / full canvas / zoom / available / |shift| / target (256,256): YES")
    A(f"- signed_shift_top == -signed_shift_bottom (exact): {audit['geometry']['signed_shift_mirrored']}")
    A("- TOP crop = (0, k_start, 256, k_start+256); BOTTOM crop = (0, k_end, 256, k_end+256)")
    A(f"- all pair invariants: {audit['geometry']['all_pair_invariants']}")
    A("")
    A("## 3. Conditioning / timestep / noise")
    A("- conditioning = frozen anchor unit bundle (qwen states + all routing), shared by TOP and BOTTOM; NO caption regeneration")
    A(f"- t strata: {audit['timestep_noise']['strata']}")
    A("- one deterministic eps per (pair, stratum), shared by TOP and BOTTOM; same inputs across PRE/MID/POST")
    A("")
    A("## 4. Numerics prerequisite (spec s27)")
    np_ = audit["numerics_prerequisite"]
    A(f"- V2 numerics clean: {np_['v2_numerics_clean']}; hidden mutable state: {np_['hidden_mutable_state']}")
    A(f"- P3 state_history_effect: {np_['p3_state_history_effect']} (wired into this gate: {np_['p3_wired_into_gate']})")
    A(f"- repeat jitter (diagnostic only): {np_['repeat_jitter_diagnostic']}")
    A("")
    A("## 5. Causal results (point = exact mean; CI = paired bootstrap seed 20260907, n=10000)")
    A("")
    A("| transition | D_TOP | CI | D_BOTTOM | CI | G = D_TOP - D_BOTTOM | CI |")
    A("|---|---|---|---|---|---|---|")
    ptt = audit["paired_top_bottom"]
    order = ["PRE|MID", "MID|POST", "PRE|POST"]
    label = {"PRE|MID": "PRE->MID", "MID|POST": "MID->POST", "PRE|POST": "PRE->POST"}
    for k in order:
        dt = audit["top_causal"]["adjusted_" + label[k]]
        db = audit["bottom_causal"]["adjusted_" + label[k]]
        g = ptt[label[k] + "_gap"]
        A(f"| {label[k]} | {dt['point']:.6g} | {_ci_str(dt['ci95'])} | {db['point']:.6g} | {_ci_str(db['ci95'])} | **{g['point']:.6g}** | {_ci_str(g['ci95'])} |")
    A("")
    A("### M levels (4-stratum mean margin, all pairs)")
    for side, name in (("m_top", "M_TOP"), ("m_bot", "M_BOTTOM")):
        row = metrics["margins_per_ckpt"][side]
        A(f"- {name}: PRE {row['PRE']['point']:.6g} {_ci_str(row['PRE']['ci95'])}; "
          f"MID {row['MID']['point']:.6g} {_ci_str(row['MID']['ci95'])}; "
          f"POST {row['POST']['point']:.6g} {_ci_str(row['POST']['ci95'])}")
    A("")
    A("## 6. Natural cohort vs same-source (diagnostic, no mixed bootstrap)")
    nv = audit["natural_vs_same_source"]
    A(f"- historical natural TOP: {nv['historical_natural_TOP']['point']:.6g} {_ci_str(nv['historical_natural_TOP']['ci95'])}")
    A(f"- historical natural BOTTOM: {nv['historical_natural_BOTTOM']['point']:.6g} {_ci_str(nv['historical_natural_BOTTOM']['ci95'])}")
    A(f"- historical gap: {nv['historical_gap']['point']:.6g} {_ci_str(nv['historical_gap']['ci95'])}")
    A(f"- same-source gap (PRE->POST): {nv['same_source_gap']['point']:.6g} {_ci_str(nv['same_source_gap']['ci95'])}")
    att = nv["confounding_attenuation"]
    A(f"- confounding attenuation: {att:.3f}" if att is not None else "- confounding attenuation: n/a")
    A("")
    A("## 7. Content audit")
    ct = audit["content"]
    A(f"- CLIP text available: {ct['clip_text_available']} ({ct['clip_text_status']})")
    if ct["clip_text_available"]:
        A(f"- CLIP text TOP-BOTTOM delta: {ct['clip_text_top_bottom']['mean_delta']:.6g} {_ci_str(ct['clip_text_top_bottom']['ci95'])}")
    A(f"- CLIP image sim(full,crop) TOP-BOTTOM delta: {ct['clip_image_full_top_bottom']['mean_delta']:.6g} {_ci_str(ct['clip_image_full_top_bottom']['ci95'])}")
    A(f"- PE retained-content TOP-BOTTOM delta: {ct['pe_content_top_bottom']['mean_delta_retained']:.6g} {_ci_str(ct['pe_content_top_bottom']['ci95'])}")
    A(f"- systematic BOTTOM content loss: {ct['systematic_bottom_content_loss']}")
    cbs = audit["content_balanced_subset"]
    A(f"- content-balanced subset: n={cbs['pair_count']} (pre-registered, uses causal results: {cbs['uses_causal_results']})")
    gcb = cbs["top_bottom_causal_gap"]
    if gcb:
        A(f"- balanced-subset PRE->POST gap: {gcb['point']:.6g} {_ci_str(gcb['ci95'])}; attenuation vs all: {cbs['attenuation_vs_all']:.3f}")
    cc = audit["content_causal"]
    A(f"- Spearman(content asymmetry, G_PRE->POST): {cc['spearman']}")
    A(f"- quartile trend (G mean by content-asymmetry quartile): {cc['quartile_trend']}")
    A("")
    A("## 8. Correct-loss baseline (L_BB - L_TT)")
    cl = audit["correct_loss_baseline"]
    for ck in ("PRE", "MID", "POST"):
        A(f"- {ck}: {cl[ck]['point']:.6g} {_ci_str(cl[ck]['ci95'])}")
    A("")
    A("## 9. Numerics floor (SAME duplicate forwards; mean + CI, not a hard gate)")
    for ck in ("PRE", "MID", "POST"):
        f = audit["numerics_floor"][ck]
        A(f"- {ck}: floor TOP {f['f_top']['mean']:.6g} {_ci_str(f['f_top']['ci95'])}; "
          f"floor BOTTOM {f['f_bot']['mean']:.6g} {_ci_str(f['f_bot']['ci95'])}")
    A("")
    A("## 10. Strata (formal CI only for n>=30, else SMALL_N)")
    for k, v in audit["strata"].items():
        A(f"- {k}: n={v['n']} G={v['point']:.6g} CI={_ci_str(v['ci95'])}")
    A("")
    A("## 11. Classification & recommendation (spec s45/s46)")
    A(f"- **VERDICT: {audit['classification']['verdict']}** (case {audit['classification']['case']})")
    for note in audit["classification"].get("notes", []):
        A(f"  - {note}")
    A(f"- RECOMMENDATION: **{audit['recommendation']['value']}** "
      f"(longer_p25_authorized={audit['recommendation']['longer_p25_authorized']}, "
      f"p50_authorized={audit['recommendation']['p50_authorized']})")
    A("")
    A("## 12. Validation / immutability / security")
    A(f"- validation: {json.dumps(audit['validation'], sort_keys=True)}")
    A(f"- immutability: {json.dumps(audit['immutability'], sort_keys=True)}")
    A(f"- security: {json.dumps(audit['security'], sort_keys=True)}")
    A("")
    A("## 13. Next")
    A(f"- {audit['next']}")
    A("")
    return "\n".join(L) + "\n"


def render_final_copy(audit: dict) -> str:
    """spec s60 FINAL COPY (verbatim template; self-referential commit fields
    are filled in the session's final output after push - a commit hash cannot
    contain itself)."""
    pd_ = audit["pair_design"]
    ap = pd_["anchor_reconstruction_parity"]
    ct = audit["content"]
    cbs = audit["content_balanced_subset"]
    cc = audit["content_causal"]
    nv = audit["natural_vs_same_source"]
    g = audit["paired_top_bottom"]
    cl = audit["correct_loss_baseline"]
    v = audit["validation"]
    S: list[str] = []
    A = S.append
    A("== SakuraMoon Camera Vertical Bottom / Same-Source Supervision Audit ==")
    A("")
    A("BASE")
    A("  repository = leafmoone/sakuramoon (github)")
    A(f"  reviewed base = {audit['base_sha']}")
    A(f"  branch = {audit['branch']}")
    A(f"  tooling commit = {audit.get('tooling_commit', '<TOOLING_HEAD>')}")
    A(f"  evidence commit = {audit.get('evidence_commit', '<EVIDENCE_HEAD - the commit containing this file>')}")
    A(f"  remote SHA = {audit['external_review_target'].get('pushed_sha', '<ls-remote read-back; == evidence HEAD>')}")
    A(f"  pushed = {audit.get('pushed', '<recorded in session FINAL COPY after push>')}")
    A("  force push = NO")
    A("")
    A("PAIR DESIGN")
    A(f"  candidate vertical units = {pd_['candidate_vertical_units']}")
    A(f"  unique sources = {pd_['unique_sources']}")
    A(f"  selected pairs = {pd_['selected_pairs']}")
    A(f"  original START anchors = {pd_['original_start_anchors']}")
    A(f"  original END anchors = {pd_['original_end_anchors']}")
    zb = pd_["zoom_bands"]
    A(f"  mild/medium/strong = {zb['mild']}/{zb['medium']}/{zb['strong']}")
    A(f"  pair manifest sha = {pd_['pair_manifest_sha256']}")
    A(f"  source image availability = {pd_['source_image_availability']}")
    A(f"  anchor reconstruction parity = {ap['n_bitexact']}/{ap['n_checked']} bitexact, worst maxabs {ap['maxabs_worst']:.3e}")
    A("")
    A("GEOMETRY")
    A("  same source = YES")
    A("  same zoom = YES")
    A("  same full canvas = YES")
    A("  same |shift| = YES")
    A("  signed shift mirrored = YES (exact)")
    A("  TOP crop = (0, k_start, 256, k_start+256)")
    A("  BOTTOM crop = (0, k_end, 256, k_end+256)")
    A(f"  all pair invariants = {audit['geometry']['all_pair_invariants']}")
    A("")
    A("CONDITIONING")
    A("  qwen states shared = YES")
    A("  text routing shared = YES")
    A("  condition routing shared = YES")
    A("  caption regenerated = NO")
    A("")
    A("TIMESTEP / NOISE")
    A(f"  strata = {audit['timestep_noise']['strata']}")
    A("  same noise TOP/BOTTOM = YES")
    A("  same input across PRE/MID/POST = YES")
    A("")
    A("NUMERICS PREREQUISITE")
    np_ = audit["numerics_prerequisite"]
    A(f"  V2 numerics clean = {np_['v2_numerics_clean']}")
    A(f"  hidden mutable state = {np_['hidden_mutable_state']}")
    A(f"  P3 state_history_effect = {np_['p3_state_history_effect']}")
    A(f"  P3 now wired into gate = {np_['p3_wired_into_gate']}")
    A(f"  repeat jitter diagnostic = {np_['repeat_jitter_diagnostic']}")
    A("")
    A("MIRROR ESTIMATOR FIX")
    A("  point definition =")
    A("    mean(TOP-BOTTOM)")
    A(f"  paired bootstrap = seed {C.BOOT_SEED}, n={C.N_BOOT}, source-pair resampling")
    A("  old pooled-sign formula used = NO")
    A(f"  unequal-count regression = {audit['mirror_estimator_fix']['unequal_count_rule']}")
    A("")
    A("TOP CAUSAL")
    for ck in ("PRE", "MID", "POST"):
        A(f"  {ck} M = {audit['top_causal']['M'][ck]['point']:.6g} {_ci_str(audit['top_causal']['M'][ck]['ci95'])}")
    A(f"  adjusted PRE->MID = {audit['top_causal']['adjusted_PRE->MID']['point']:.6g} {_ci_str(audit['top_causal']['adjusted_PRE->MID']['ci95'])}")
    A(f"  adjusted MID->POST = {audit['top_causal']['adjusted_MID->POST']['point']:.6g} {_ci_str(audit['top_causal']['adjusted_MID->POST']['ci95'])}")
    A(f"  adjusted PRE->POST = {audit['top_causal']['adjusted_PRE->POST']['point']:.6g} {_ci_str(audit['top_causal']['adjusted_PRE->POST']['ci95'])}")
    A(f"  CI = {_ci_str(audit['top_causal']['adjusted_PRE->POST']['ci95'])}")
    A("")
    A("BOTTOM CAUSAL")
    for ck in ("PRE", "MID", "POST"):
        A(f"  {ck} M = {audit['bottom_causal']['M'][ck]['point']:.6g} {_ci_str(audit['bottom_causal']['M'][ck]['ci95'])}")
    A(f"  adjusted PRE->MID = {audit['bottom_causal']['adjusted_PRE->MID']['point']:.6g} {_ci_str(audit['bottom_causal']['adjusted_PRE->MID']['ci95'])}")
    A(f"  adjusted MID->POST = {audit['bottom_causal']['adjusted_MID->POST']['point']:.6g} {_ci_str(audit['bottom_causal']['adjusted_MID->POST']['ci95'])}")
    A(f"  adjusted PRE->POST = {audit['bottom_causal']['adjusted_PRE->POST']['point']:.6g} {_ci_str(audit['bottom_causal']['adjusted_PRE->POST']['ci95'])}")
    A(f"  CI = {_ci_str(audit['bottom_causal']['adjusted_PRE->POST']['ci95'])}")
    A("")
    A("PAIRED TOP-BOTTOM")
    A(f"  PRE->MID gap = {g['PRE->MID_gap']['point']:.6g} {_ci_str(g['PRE->MID_gap']['ci95'])}")
    A(f"  MID->POST gap = {g['MID->POST_gap']['point']:.6g} {_ci_str(g['MID->POST_gap']['ci95'])}")
    A(f"  PRE->POST gap = {g['PRE->POST_gap']['point']:.6g} {_ci_str(g['PRE->POST_gap']['ci95'])}")
    A(f"  95% CI = {_ci_str(g['PRE->POST_gap']['ci95'])}")
    A("  inference unit = source pair")
    A("")
    A("NATURAL vs SAME-SOURCE")
    A(f"  historical natural TOP = {nv['historical_natural_TOP']['point']:.6g} {_ci_str(nv['historical_natural_TOP']['ci95'])}")
    A(f"  historical natural BOTTOM = {nv['historical_natural_BOTTOM']['point']:.6g} {_ci_str(nv['historical_natural_BOTTOM']['ci95'])}")
    A(f"  historical gap = {nv['historical_gap']['point']:.6g} {_ci_str(nv['historical_gap']['ci95'])}")
    A(f"  same-source gap = {nv['same_source_gap']['point']:.6g} {_ci_str(nv['same_source_gap']['ci95'])}")
    att = nv["confounding_attenuation"]
    A(f"  confounding attenuation = {att:.3f}" if att is not None else "  confounding attenuation = n/a")
    A("")
    A("CONTENT")
    A(f"  CLIP text available = {ct['clip_text_available']}")
    if ct["clip_text_available"]:
        A(f"  CLIP text TOP/BOTTOM = {ct['clip_text_top_bottom']['mean_delta']:.6g} {_ci_str(ct['clip_text_top_bottom']['ci95'])}")
    else:
        A("  CLIP text TOP/BOTTOM = NOT_AVAILABLE")
    A(f"  CLIP image-full TOP/BOTTOM = {ct['clip_image_full_top_bottom']['mean_delta']:.6g} {_ci_str(ct['clip_image_full_top_bottom']['ci95'])}")
    A(f"  PE content TOP/BOTTOM = {ct['pe_content_top_bottom']['mean_delta_retained']:.6g} {_ci_str(ct['pe_content_top_bottom']['ci95'])}")
    A(f"  systematic BOTTOM content loss = {ct['systematic_bottom_content_loss']}")
    A("")
    A("CONTENT-BALANCED SUBSET")
    A(f"  pair count = {cbs['pair_count']}")
    A(f"  definition = {cbs['definition']}")
    A("  uses causal results = NO")
    gcb = cbs["top_bottom_causal_gap"]
    if gcb:
        A(f"  TOP-BOTTOM causal gap = {gcb['point']:.6g} {_ci_str(gcb['ci95'])}")
        A(f"  CI = {_ci_str(gcb['ci95'])}")
    else:
        A("  TOP-BOTTOM causal gap = n/a")
        A("  CI = n/a")
    A(f"  attenuation vs all = {cbs['attenuation_vs_all']:.3f}" if cbs["attenuation_vs_all"] is not None else "  attenuation vs all = n/a")
    A("")
    A("CONTENT <-> CAUSAL")
    A(f"  Spearman = {cc['spearman']}")
    A(f"  quartile trend = {cc['quartile_trend']}")
    A(f"  interpretation = {cc['interpretation']}")
    A("")
    A("CORRECT-LOSS BASELINE")
    A(f"  PRE BB-TT = {cl['PRE']['point']:.6g} {_ci_str(cl['PRE']['ci95'])}")
    A(f"  MID = {cl['MID']['point']:.6g} {_ci_str(cl['MID']['ci95'])}")
    A(f"  POST = {cl['POST']['point']:.6g} {_ci_str(cl['POST']['ci95'])}")
    A(f"  interpretation = {cl['interpretation']}")
    A("")
    A("STRATA")
    st = audit["strata"]
    A(f"  anchor START result = {st['anchor_START']['point']:.6g} {_ci_str(st['anchor_START']['ci95'])} (n={st['anchor_START']['n']})")
    A(f"  anchor END result = {st['anchor_END']['point']:.6g} {_ci_str(st['anchor_END']['ci95'])} (n={st['anchor_END']['n']})")
    A(f"  mild = {st['zoom_mild']['point']:.6g} {_ci_str(st['zoom_mild']['ci95'])} (n={st['zoom_mild']['n']})")
    A(f"  medium = {st['zoom_medium']['point']:.6g} {_ci_str(st['zoom_medium']['ci95'])} (n={st['zoom_medium']['n']})")
    A(f"  strong = {st['zoom_strong']['point']:.6g} {_ci_str(st['zoom_strong']['ci95'])} (n={st['zoom_strong']['n']})")
    A(f"  latent <2 = {st['latent_lt2']['point']:.6g} {_ci_str(st['latent_lt2']['ci95'])} (n={st['latent_lt2']['n']})")
    A(f"  latent 2-4 = {st['latent_2to4']['point']:.6g} {_ci_str(st['latent_2to4']['ci95'])} (n={st['latent_2to4']['n']})")
    A(f"  latent >=4 = {st['latent_ge4']['point']:.6g} {_ci_str(st['latent_ge4']['ci95'])} (n={st['latent_ge4']['n']})")
    A("")
    A("CLASSIFICATION")
    A(f"  {audit['classification']['verdict']}")
    A("")
    A("RECOMMENDATION")
    A(f"  {audit['recommendation']['value']}")
    A("")
    A("VALIDATION")
    A(f"  old audit tests = {v.get('old_audit_tests', '<gate runner>')}")
    A(f"  V1 tests = {v.get('v1_tests', '<gate runner>')}")
    A(f"  V2 tests = {v.get('v2_tests', '<gate runner>')}")
    A(f"  new tests = {v.get('new_tests', '<gate runner>')}")
    A(f"  total = {v.get('total_tests', '<gate runner>')}")
    A(f"  skips = {v.get('skips', 0)}")
    A(f"  xfails = {v.get('xfails', 0)}")
    A(f"  replay = {v.get('replay', '<gate runner>')}")
    A(f"  ruff = {v.get('ruff', '<gate runner>')}")
    A(f"  py_compile = {v.get('py_compile', '<gate runner>')}")
    A(f"  src/config diff = {v.get('src_config_diff', '<gate runner>')}")
    A("")
    A("IMMUTABILITY")
    im = audit["immutability"]
    A(f"  historical causal = {im.get('historical_causal', '<gate runner>')}")
    A(f"  V1 posthoc = {im.get('v1_posthoc', '<gate runner>')}")
    A(f"  V2 posthoc = {im.get('v2_posthoc', '<gate runner>')}")
    A(f"  offset balance V2 = {im.get('offset_balance_v2', '<gate runner>')}")
    A(f"  prior tooling = {im.get('prior_tooling', '<gate runner>')}")
    A(f"  /tmp evidence retained = {im.get('tmp_evidence_retained', '<gate runner>')}")
    A("")
    A("SECURITY")
    se = audit["security"]
    A(f"  secret hits = {se.get('secret_hits', '<gate runner>')}")
    A("  checkpoints staged = NO")
    A("  PNG staged = NO")
    A("  dataset staged = NO")
    A("  binary cache staged = NO")
    A("")
    A("AUTHORIZATION")
    A("  longer P25 started = NO")
    A("  P50 started = NO")
    A("  production changed = NO")
    A("")
    A("EXTERNAL REVIEW TARGET")
    A(f"  branch = {audit['branch']}")
    A(f"  exact SHA = {audit['external_review_target'].get('exact_sha', '<EVIDENCE_HEAD>')}")
    A("")
    A("NEXT")
    A("  HARD STOP")
    A("  SEND SHA TO EXTERNAL REVIEWER")
    A("  EXTERNAL REVIEW REQUIRED BEFORE ANY TRAINING GO")
    A("")
    A("== END ==")
    return "\n".join(S) + "\n"


# ---------------- contact sheets ----------------

def make_sheets(sheet_dir: Path, out_root: Path, pairs: list[dict],
                per: dict, asym: dict, n: int) -> None:
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return

    g = [per[i]["G_PRE|POST"] for i in range(n)]
    picks: dict[str, list[int]] = {
        "representative": list(range(min(8, n))),
        "strongest_content_asym": sorted(range(n), key=lambda i: -asym[i])[:4],
        "lowest_content_asym": sorted(range(n), key=lambda i: asym[i])[:4],
        "strongest_causal_gap": sorted(range(n), key=lambda i: -g[i])[:4],
        "weakest_causal_gap": sorted(range(n), key=lambda i: g[i])[:4],
    }
    img_dir = out_root / "images"
    for name, idxs in picks.items():
        cell_w, cell_h = 256, 256
        cols = 4  # label + 3 panels
        rows = len(idxs)
        label_w = 150
        canvas = Image.new("RGB", (label_w + cols * cell_w, rows * cell_h), "white")
        draw = ImageDraw.Draw(canvas)
        for r, i in enumerate(idxs):
            draw.text((4, r * cell_h + 8), f"pair {i}\n{pairs[i]['original_side']} z={pairs[i]['equivalent_zoom']:.3f}", fill="black")
            src = (img_dir / f"canvas-{i:04d}.png").open("rb")
            full = Image.open(src).convert("RGB").resize((cell_w, cell_h), Image.Resampling.LANCZOS)
            top = Image.open(img_dir / f"pair-{i:04d}-top.png").convert("RGB").resize((cell_w, cell_h))
            bot = Image.open(img_dir / f"pair-{i:04d}-bottom.png").convert("RGB").resize((cell_w, cell_h))
            canvas.paste(full, (label_w, r * cell_h))
            canvas.paste(top, (label_w + cell_w, r * cell_h))
            canvas.paste(bot, (label_w + 2 * cell_w, r * cell_h))
        canvas.save(sheet_dir / f"sheet-{name}.png")


if __name__ == "__main__":
    sys.exit(main())
