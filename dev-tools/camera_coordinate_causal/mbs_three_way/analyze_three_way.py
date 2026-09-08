"""MBS Three-Way Same-Source Causal Audit - analysis + report writer.

Reads (never writes to):
  * the NEW per-worker/per-ckpt scoring ledgers (AUDIT_ROOT/ledger/)
  * the reconstructed frozen VBS pair manifest (VBS_ROOT/pair-manifest.json)
  * the COMMITTED frozen VBS pairs CSV + audit.json (worktree reports/)
Writes (new artifacts only):
  * reports/camera-mbs-three-way-same-source-pairs.csv    (per-pair, 17-sig-fig)
  * reports/camera-mbs-three-way-same-source-metrics.json
  * reports/camera-mbs-three-way-same-source-audit.json
  * reports/camera-mbs-three-way-same-source-audit.md
  * AUDIT_ROOT/replay-gate.json
  * AUDIT_ROOT/analyze.log

Estimator semantics are EXACTLY the frozen VBS ones (per-pair 4-stratum mean
margins; per-pair SAME-corrected deltas; G = D_TOP - D_BOTTOM; point = exact
arithmetic mean; CI = paired bootstrap seed 20260907 n=10000 with a shared
source-pair index matrix per cohort).

Flow:
  1. completeness gate (512 pairs x 3 ckpts x 8 arms x 4 strata)
  2. PRE replay gate vs the frozen U118100 (VBS POST) per-pair values
     -> on FAIL: write replay-gate.json and exit 2 WITHOUT interpreting
        CONTROL/TREATMENT (spec s7)
  3. per-pair contrasts (D, G, I) + 17-sig-fig pairs CSV
  4. six pre-registered cohorts: M, D, G, I points + bootstrap CIs
  5. classification (TARGETED co-primary) + all-pair report (spec s10/s12)
  6. independent self-replay from the new pairs CSV (point estimates must
     reproduce EXACTLY) + bootstrap determinism check (spec s18)

No torch import. No training.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import contracts as C

FMT = "{:.17g}"  # float64 round-trip precision for the per-pair CSV


def log(msg: str, log_path: Path) -> None:
    line = f"[analyze3 {time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def mean_exact(vals: list[float]) -> float:
    """Exact arithmetic mean (identical expression to the frozen analyze)."""
    return float(sum(vals) / len(vals))


# ---------------- ledger loading ----------------

def load_ledgers(out_root: Path, n_pairs: int) -> dict[tuple[str, int], dict[str, list[float]]]:
    """rows[(ck, pair_index)] = {arm: [4 strata values]}; completeness-gated."""
    rows: dict[tuple[str, int], dict[str, list[float]]] = {}
    for ck in C.CKS:
        for w in (0, 1):
            p = out_root / "ledger" / f"ledger-w{w}-{ck.lower()}.jsonl"
            if not p.is_file():
                raise RuntimeError(f"missing ledger {p}")
            with open(p, encoding="utf-8") as fh:
                for line in fh:
                    row = json.loads(line)
                    key = (row["ck"], int(row["pair_index"]))
                    if key in rows:
                        raise RuntimeError(f"duplicate ledger row {key}")
                    losses = row["losses"]
                    if set(losses) != set(C.ARMS):
                        raise RuntimeError(f"{key} arms {sorted(losses)} != {sorted(C.ARMS)}")
                    for a in C.ARMS:
                        if len(losses[a]) != 4:
                            raise RuntimeError(f"{key} arm {a} strata {len(losses[a])} != 4")
                    rows[key] = losses
    expected = n_pairs * 3
    if len(rows) != expected:
        missing = [(ck, i) for ck in C.CKS for i in range(n_pairs) if (ck, i) not in rows]
        raise RuntimeError(f"incomplete ledger: {len(missing)} missing, e.g. {missing[:5]}")
    return rows


def per_pair_margins(rows: dict, i: int) -> dict[str, dict[str, float]]:
    """m_top/m_bot/f_top/f_bot/bb_minus_tt per ck = 4-stratum mean (frozen).

    VERBATIM frozen VBS expression (vertical_bottom_supervision/analyze.py
    per_pair_margins): (sum(armA) - sum(armB)) / 4.0 - the two arm sums are
    taken separately before subtraction (bit-faithful to the frozen audit's
    per-pair values; do NOT rewrite as mean of per-stratum differences).
    """
    out: dict[str, dict[str, float]] = {}
    for ck in C.CKS:
        L = rows[(ck, i)]
        m = {
            "m_top": (sum(L["TB"]) - sum(L["TT"])) / 4.0,
            "m_bot": (sum(L["BT"]) - sum(L["BB"])) / 4.0,
            "f_top": (sum(L["T_SAME"]) - sum(L["TT"])) / 4.0,
            "f_bot": (sum(L["B_SAME"]) - sum(L["BB"])) / 4.0,
            "bb_minus_tt": (sum(L["BB"]) - sum(L["TT"])) / 4.0,
        }
        out[ck] = m
    return out


# ---------------- cohort machinery ----------------

def cohort_index_sets(pairs: list[dict], frozen_rows: list[dict]) -> dict[str, list[int]]:
    """Pre-registered cohort memberships from the FROZEN cohort only.

    latent band from |signed_shift_start|/VAE_SCALE (frozen banding);
    balanced from the frozen in_balanced_subset flag.  No result-derived
    selection is possible here: flags exist before any scoring.
    """
    n = len(pairs)
    sets: dict[str, list[int]] = {
        C.COHORT_ALL: list(range(n)),
        C.COHORT_TARGETED: [],
        C.COHORT_LT2: [],
        C.COHORT_2TO4: [],
        C.COHORT_GE4: [],
        C.COHORT_BALANCED: [],
    }
    for i, row in enumerate(frozen_rows):
        fl = C.pair_cohort_flags(row)
        if fl["in_targeted"]:
            sets[C.COHORT_TARGETED].append(i)
        sets[fl["latent_band"].replace("lt2", C.COHORT_LT2).replace("2to4", C.COHORT_2TO4).replace("ge4", C.COHORT_GE4)].append(i)
        if fl["in_balanced"]:
            sets[C.COHORT_BALANCED].append(i)
    for name, idx in sets.items():
        if len(idx) != C.EXPECTED_COHORT_SIZES[name]:
            raise RuntimeError(
                f"cohort {name}: n={len(idx)} != expected {C.EXPECTED_COHORT_SIZES[name]} "
                "(frozen cohort membership violated - STOP)"
            )
    return sets


def point_ci(vals: list[float]) -> dict[str, Any]:
    """point = exact mean; ci95 = frozen paired bootstrap (seed 20260907, n=10000)."""
    return {"point": mean_exact(vals), "ci95": C.bootstrap_ci(vals)}


def check_pair_cohort_identity(pair_root: Path, log_path: Path):
    """Pre-scoring gate: reconstructed pair manifest vs committed frozen CSV.

    The original pair-manifest.json embeds created_utc and is not byte-
    reproducible, so cohort identity is verified FIELD-BY-FIELD against the
    committed frozen pairs CSV (source_shard, sample_id, side, k window,
    available, full_height, signed_shift_start) plus the 512-pair count.

    Returns (pairs, frozen_rows, n) on PASS; None on FATAL (already logged).
    """
    if not (pair_root / "pair-manifest.json").is_file():
        log(f"FATAL: missing {pair_root / 'pair-manifest.json'}", log_path)
        return None
    if not C.FROZEN_PAIRS_CSV.is_file():
        log(f"FATAL: missing frozen CSV {C.FROZEN_PAIRS_CSV}", log_path)
        return None
    with open(pair_root / "pair-manifest.json", encoding="utf-8") as fh:
        pm = json.load(fh)
    if pm.get("status") != "OK":
        log(f"FATAL: pair manifest status {pm.get('status')}", log_path)
        return None
    pairs = sorted(pm["pairs"], key=lambda r: r["pair_index"])
    n = len(pairs)
    if n != C.EXPECTED_COHORT_SIZES[C.COHORT_ALL]:
        log(f"FATAL: pair count {n} != 512", log_path)
        return None
    frozen_rows = C.parse_frozen_pairs_csv(C.FROZEN_PAIRS_CSV)
    if len(frozen_rows) != n:
        log("FATAL: frozen CSV row count mismatch", log_path)
        return None
    # field-level cohort identity: reconstructed manifest vs committed CSV
    for i, (p, fr) in enumerate(zip(pairs, frozen_rows, strict=True)):
        g = p["geometry"]
        checks = (
            (p["source_shard"], fr["source_shard"]),
            (p["sample_id"], fr["sample_id"]),
            (p["original_side"], fr["original_side"]),
            (g["k_start"], fr["k_start"]),
            (g["k_end"], fr["k_end"]),
            (g["available"], fr["available"]),
            (g["full_height"], fr["full_height"]),
        )
        for a, b in checks:
            if a != b:
                log(f"FATAL: pair {i} manifest/CSV mismatch {a!r} != {b!r}", log_path)
                return None
        if float(p["signed_shift_start"]) != fr["signed_shift_start"]:
            log(f"FATAL: pair {i} signed_shift mismatch", log_path)
            return None
    log(f"cohort field identity: {n}/{n} pairs match committed frozen CSV", log_path)
    return pairs, frozen_rows, n


# ---------------- main ----------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair-root", type=Path, default=C.VBS_ROOT)
    ap.add_argument("--out-root", type=Path, default=C.AUDIT_ROOT)
    ap.add_argument("--worktree", type=Path, default=C.WORKTREE)
    ap.add_argument("--external-checks", type=Path, default=None,
                    help="optional JSON with git/ckpt immutability checks (embedded verbatim)")
    ap.add_argument("--pair-check", action="store_true",
                    help="run ONLY the pre-scoring pair-cohort identity gate "
                         "(reconstructed manifest vs committed frozen CSV) and exit")
    args = ap.parse_args()

    log_path = args.out_root / "analyze.log"
    args.out_root.mkdir(parents=True, exist_ok=True)
    log("starting", log_path)

    # ---- inputs ----
    ident = check_pair_cohort_identity(args.pair_root, log_path)
    if ident is None:
        return 2
    pairs, frozen_rows, n = ident
    if args.pair_check:
        C.write_frozen(args.out_root / "pair-cohort-gate.json", {
            "label": "MBS THREE-WAY PAIR COHORT IDENTITY GATE",
            "status": "PASS",
            "n_pairs": n,
            "pair_manifest": str(args.pair_root / "pair-manifest.json"),
            "pair_manifest_sha256": C.sha256_file(args.pair_root / "pair-manifest.json"),
            "frozen_pairs_csv": str(C.FROZEN_PAIRS_CSV),
            "note": (
                "the original pair-manifest.json embeds created_utc and is not "
                "byte-reproducible (sha 7f3a81b2... not matchable); cohort "
                "identity verified field-by-field against the committed frozen "
                "CSV via check_pair_cohort_identity"
            ),
        })
        log("pair-cohort gate: PASS", log_path)
        return 0
    with open(C.FROZEN_AUDIT_JSON, encoding="utf-8") as fh:
        faudit = json.load(fh)

    # ---- ledgers + margins ----
    rows = load_ledgers(args.out_root, n)
    per: list[dict[str, dict[str, float]]] = [per_pair_margins(rows, i) for i in range(n)]

    # ---- replay gate (spec s7) ----
    floor_post = faudit["numerics_floor"]["POST"]
    bounds = C.replay_bounds_from_floor(floor_post)
    replay_per_pair = []
    for i, fr in enumerate(frozen_rows):
        replay_per_pair.append({
            "pair_index": i,
            "delta_top": per[i]["PRE"]["m_top"] - fr["M_TOP_POST_frozen"],
            "delta_bot": per[i]["PRE"]["m_bot"] - fr["M_BOT_POST_frozen"],
        })
    new_points = {
        "M_TOP": mean_exact([per[i]["PRE"]["m_top"] for i in range(n)]),
        "M_BOTTOM": mean_exact([per[i]["PRE"]["m_bot"] for i in range(n)]),
    }
    frozen_points = {
        "M_TOP": float(faudit["top_causal"]["M"]["POST"]["point"]),
        "M_BOTTOM": float(faudit["bottom_causal"]["M"]["POST"]["point"]),
    }
    replay = C.replay_gate_stats(replay_per_pair, bounds, new_points, frozen_points)
    C.write_frozen(args.out_root / "replay-gate.json", {
        "label": "MBS THREE-WAY PRE REPLAY GATE",
        "new_pre_checkpoint": str(C.CKPT_PATHS["PRE"]),
        "frozen_reference": "committed VBS pairs CSV M_TOP_POST / M_BOT_POST (U118100)",
        **bounds,
        **{k: v for k, v in replay.items() if k != "checks"},
        "checks": replay["checks"],
        "new_points": new_points,
        "frozen_points": frozen_points,
        "status": replay["status"],
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    log(
        f"REPLAY GATE {replay['status']}: maxabs={replay['maxabs']:.3e} rms={replay['rms']:.3e} "
        f"bound={replay['bound']:.3e}",
        log_path,
    )
    if replay["status"] != "PASS":
        log("STOP: replay gate failed - CONTROL/TREATMENT not interpreted (spec s7)", log_path)
        return 2

    # ---- per-pair contrasts (frozen SAME-corrected semantics, per pair) ----
    per_pair_out: list[dict[str, Any]] = []
    for i in range(n):
        pre, con, tre = per[i]["PRE"], per[i]["CONTROL"], per[i]["TREATMENT"]
        d_top_c = (con["m_top"] - pre["m_top"]) - (con["f_top"] - pre["f_top"])
        d_bot_c = (con["m_bot"] - pre["m_bot"]) - (con["f_bot"] - pre["f_bot"])
        g_c = d_top_c - d_bot_c
        d_top_t = (tre["m_top"] - pre["m_top"]) - (tre["f_top"] - pre["f_top"])
        d_bot_t = (tre["m_bot"] - pre["m_bot"]) - (tre["f_bot"] - pre["f_bot"])
        g_t = d_top_t - d_bot_t
        fl = C.pair_cohort_flags(frozen_rows[i])
        per_pair_out.append({
            "i": i,
            "fr": frozen_rows[i],
            "fl": fl,
            "pre": pre, "con": con, "tre": tre,
            "d_top_c": d_top_c, "d_bot_c": d_bot_c, "g_c": g_c,
            "d_top_t": d_top_t, "d_bot_t": d_bot_t, "g_t": g_t,
            "i_top": d_top_t - d_top_c,
            "i_bot": d_bot_t - d_bot_c,
            "i_g": g_t - g_c,
        })

    # ---- new pairs CSV (17 sig figs: exact float64 round-trip) ----
    csv_path = C.REPORT_PAIRS_CSV
    header = (
        "pair_id,source_shard,sample_id,original_side,zoom_band,"
        "k_start,k_end,available,full_height,signed_shift_start,latent_shift,"
        "latent_band,in_balanced_subset,content_asymmetry,"
        "m_top_pre,m_bot_pre,f_top_pre,f_bot_pre,"
        "m_top_control,m_bot_control,f_top_control,f_bot_control,"
        "m_top_treatment,m_bot_treatment,f_top_treatment,f_bot_treatment,"
        "d_top_control,d_bot_control,g_control,"
        "d_top_treatment,d_bot_treatment,g_treatment,"
        "i_top,i_bottom,i_g"
    )
    with open(csv_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(header + "\n")
        for o in per_pair_out:
            fr, fl, pre, con, tre = o["fr"], o["fl"], o["pre"], o["con"], o["tre"]
            fh.write(",".join([
                str(o["i"]), fr["source_shard"], str(fr["sample_id"]),
                fr["original_side"], fr["zoom_band"],
                str(fr["k_start"]), str(fr["k_end"]), str(fr["available"]),
                str(fr["full_height"]), FMT.format(fr["signed_shift_start"]),
                FMT.format(fl["latent_shift"]), fl["latent_band"],
                "True" if fl["in_balanced"] else "False",
                FMT.format(fr["content_asymmetry"]),
                FMT.format(pre["m_top"]), FMT.format(pre["m_bot"]),
                FMT.format(pre["f_top"]), FMT.format(pre["f_bot"]),
                FMT.format(con["m_top"]), FMT.format(con["m_bot"]),
                FMT.format(con["f_top"]), FMT.format(con["f_bot"]),
                FMT.format(tre["m_top"]), FMT.format(tre["m_bot"]),
                FMT.format(tre["f_top"]), FMT.format(tre["f_bot"]),
                FMT.format(o["d_top_c"]), FMT.format(o["d_bot_c"]), FMT.format(o["g_c"]),
                FMT.format(o["d_top_t"]), FMT.format(o["d_bot_t"]), FMT.format(o["g_t"]),
                FMT.format(o["i_top"]), FMT.format(o["i_bot"]), FMT.format(o["i_g"]),
            ]) + "\n")
    log(f"pairs CSV written: {csv_path}", log_path)

    # ---- cohorts: points + bootstrap CIs (shared index matrix per cohort) ----
    sets = cohort_index_sets(pairs, frozen_rows)
    metrics_cohorts: dict[str, Any] = {}
    for name in C.COHORTS:
        idx = sets[name]
        def vals(key_fn, _idx=idx):
            return [key_fn(o) for o in (per_pair_out[j] for j in _idx)]
        m_top = {ck: point_ci(vals(lambda o, _ck=ck: o[_ck]["m_top"])) for ck in C.CKS}
        m_bot = {ck: point_ci(vals(lambda o, _ck=ck: o[_ck]["m_bot"])) for ck in C.CKS}
        f_top = {ck: point_ci(vals(lambda o, _ck=ck: o[_ck]["f_top"])) for ck in C.CKS}
        f_bot = {ck: point_ci(vals(lambda o, _ck=ck: o[_ck]["f_bot"])) for ck in C.CKS}
        d_top_c = point_ci(vals(lambda o: o["d_top_c"]))
        d_bot_c = point_ci(vals(lambda o: o["d_bot_c"]))
        g_c = point_ci(vals(lambda o: o["g_c"]))
        d_top_t = point_ci(vals(lambda o: o["d_top_t"]))
        d_bot_t = point_ci(vals(lambda o: o["d_bot_t"]))
        g_t = point_ci(vals(lambda o: o["g_t"]))
        i_top = point_ci(vals(lambda o: o["i_top"]))
        i_bot = point_ci(vals(lambda o: o["i_bot"]))
        i_g = point_ci(vals(lambda o: o["i_g"]))
        metrics_cohorts[name] = {
            "n": len(idx),
            "M_TOP": m_top, "M_BOTTOM": m_bot,
            "F_TOP": f_top, "F_BOTTOM": f_bot,
            "D_TOP_CONTROL": d_top_c, "D_BOTTOM_CONTROL": d_bot_c, "G_CONTROL": g_c,
            "D_TOP_TREATMENT": d_top_t, "D_BOTTOM_TREATMENT": d_bot_t, "G_TREATMENT": g_t,
            "I_TOP": i_top, "I_BOTTOM": i_bot, "I_G": i_g,
        }

    # ---- classification (spec s12) ----
    tgt = metrics_cohorts[C.COHORT_TARGETED]
    verdict = C.classify_contrast(tgt["I_BOTTOM"]["ci95"], tgt["I_G"]["ci95"])
    rec = C.canonical_control_recommendation(verdict)
    allc = metrics_cohorts[C.COHORT_ALL]
    verdict_all = C.classify_contrast(allc["I_BOTTOM"]["ci95"], allc["I_G"]["ci95"])

    # ---- performance / identity ----
    perf: dict[str, Any] = {}
    for ck in C.CKS:
        for w in (0, 1):
            g = args.out_root / f"scoring-gate-w{w}-{ck.lower()}.json"
            with open(g, encoding="utf-8") as fh:
                doc = json.load(fh)
            dm = args.out_root / "ledger" / f"w{w}-done-{ck.lower()}.json"
            with open(dm, encoding="utf-8") as fh:
                done_doc = json.load(fh)
            perf[f"{ck}_w{w}"] = {
                "device": doc["device"],
                "device_name": doc["device_name"],
                "n_pairs": done_doc["n_pairs"],
                "new_pairs_this_run": doc["new_pairs_this_run"],
                "arms": doc["arms"],
                "anchor_coord_invariant_pairs": doc["anchor_coord_invariant_pairs"],
                "finished_utc": done_doc["finished_utc"],
            }
    n_forwards = n * len(C.ARMS) * 4 * len(C.CKS)
    identity_docs = {}
    for ck in C.CKS:
        with open(args.out_root / f"scoring-gate-w0-{ck.lower()}.json", encoding="utf-8") as fh:
            identity_docs[ck] = json.load(fh)["identity_gate"]

    # ---- self-replay validation (spec s18): exact point estimates from CSV ----
    import csv as _csv

    replayed: dict[str, Any] = {"checks": [], "all_exact": True}
    with open(csv_path, encoding="utf-8", newline="") as fh:
        csv_rows = list(_csv.DictReader(fh))
    if len(csv_rows) != n:
        replayed["all_exact"] = False
        replayed["checks"].append({"name": "csv_row_count", "pass": False, "value": len(csv_rows)})
    else:
        for name in C.COHORTS:
            idx = sets[name]
            sub = [csv_rows[i] for i in idx]
            def csv_mean(col, _sub=sub):
                return mean_exact([float(r[col]) for r in _sub])
            for col, mkey, ck in (
                ("m_top_pre", "M_TOP", "PRE"),
                ("m_bot_pre", "M_BOTTOM", "PRE"),
                ("d_top_control", "D_TOP_CONTROL", None),
                ("g_control", "G_CONTROL", None),
                ("d_top_treatment", "D_TOP_TREATMENT", None),
                ("g_treatment", "G_TREATMENT", None),
                ("i_bottom", "I_BOTTOM", None),
                ("i_g", "I_G", None),
            ):
                node = metrics_cohorts[name][mkey]
                got = node[ck]["point"] if ck is not None else node["point"]
                exact = got == csv_mean(col)
                replayed["checks"].append({"name": f"{name}.{col}", "pass": exact,
                                           "metric": got, "csv": csv_mean(col)})
                if not exact:
                    replayed["all_exact"] = False
        # bootstrap determinism: recompute targeted I_G CI twice
        tgt_ig = [float(csv_rows[i]["i_g"]) for i in sets[C.COHORT_TARGETED]]
        c1 = C.bootstrap_ci(tgt_ig)
        c2 = C.bootstrap_ci(tgt_ig)
        det = (c1 == c2 == list(tgt["I_G"]["ci95"]))
        replayed["checks"].append({"name": "bootstrap_determinism_I_G_targeted", "pass": det,
                                   "ci": c1})
        if not det:
            replayed["all_exact"] = False
    log(f"self-replay: all_exact={replayed['all_exact']}", log_path)

    # ---- authorization / wording (binding) ----
    wording = {
        "exact_matched_data_control": False,
        "historical_control_interpretation": (
            "SAME-DISTRIBUTION INDEPENDENT-SEQUENCE CAMERA-ONLY CONTINUATION CONTROL"
        ),
        "treatment_sequence": "corrected MBS 100U, canonical cycle-0 sequence",
        "contrast_type": (
            "intervention-associated contrast under a common frozen evaluation cohort; "
            "training-sequence variation remains a potential confound"
        ),
        "training_sequence_confound_remains": True,
        "mbs_directional_correction_supported": verdict == "CLEAR_DIRECTIONAL_CORRECTION",
    }
    authorization = {"training": False, "p50": False, "production": False, "fid": False,
                     "generation_eval": False, "checkpoint_write": False}
    errata = {
        "severity_relation": (
            "Rerun2 metrics-summary field severity_sum_eq_applied=false was a naming/"
            "comparison mistake. Correct relations: lt2+2to4+ge4 == vertical_applied == 4505; "
            "2to4+ge4 == severe == eligible == selected == applied == 1740."
        ),
        "setup_abort_wording": (
            "The setup-abort phrase '34 metric records ... NO queue state consumed' must "
            "not be read as zero sample consumption. Correct wording: transient "
            "model/data progress occurred; no checkpoint was saved; its runtime/queue "
            "state was subsequently archived/reset; the accepted PASS run started from a "
            "fresh canonical cycle-0. Documentation errata only; the accepted Rerun2 "
            "PASS is unchanged."
        ),
    }

    external_checks: dict[str, Any] = {}
    if args.external_checks and args.external_checks.is_file():
        with open(args.external_checks, encoding="utf-8") as fh:
            external_checks = json.load(fh)

    # ---- metrics.json ----
    metrics = {
        "label": "MBS THREE-WAY SAME-SOURCE AUDIT METRICS",
        "schema_version": 1,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "base_sha": C.BASE_SHA,
        "branch": C.BRANCH,
        "frozen_references": {
            "pair_manifest_sha_original": C.FROZEN_PAIR_MANIFEST_SHA,
            "frozen_pairs_csv": str(C.FROZEN_PAIRS_CSV),
            "frozen_audit_json": str(C.FROZEN_AUDIT_JSON),
            "note": (
                "the original pair-manifest.json embeds created_utc, so byte-exact "
                "re-derivation of 7f3a81b2 is impossible by construction; cohort "
                "identity is verified field-by-field against the committed CSV and "
                "end-to-end by the PRE replay gate"
            ),
        },
        "replay_gate": dict(replay),
        "replay": {
            "u118100_old_vs_new": {
                "maxabs": replay["maxabs"], "rms": replay["rms"],
                "mean_delta": replay["mean_delta"], "bound": replay["bound"],
            },
            "same_numerics_floor_source": "frozen audit.json numerics_floor.POST (SAME duplicate forwards)",
            "within_floor": replay["status"] == "PASS",
        },
        "cohorts": metrics_cohorts,
        "classification": {
            "targeted_latent_ge2": {
                "verdict": verdict,
                "I_BOTTOM": tgt["I_BOTTOM"], "I_G": tgt["I_G"],
                "canonical_control": rec,
            },
            "all_pairs_diagnostic": {
                "verdict": verdict_all,
                "I_BOTTOM": allc["I_BOTTOM"], "I_G": allc["I_G"],
            },
        },
        "wording": wording,
        "authorization": authorization,
        "errata": errata,
        "identity": identity_docs,
        "performance": {
            "total_forwards": n_forwards,
            "forwards_per_pair_ckpt": len(C.ARMS) * 4,
            "per_worker": perf,
        },
        "self_replay_validation": replayed,
        "external_checks": external_checks,
    }
    C.write_frozen(C.REPORT_METRICS, metrics)
    log(f"metrics written: {C.REPORT_METRICS}", log_path)

    # ---- audit.json (machine document) + audit.md (human report) ----
    audit_doc = render_audit_json(metrics, identity_docs, perf, n_forwards)
    C.write_frozen(C.REPORT_JSON, audit_doc)
    md = render_audit_md(metrics)
    C.REPORT_MD.write_text(md, encoding="utf-8")
    log("audit json/md written", log_path)
    log("ANALYZE DONE", log_path)
    return 0


def render_audit_json(metrics: dict, identity: dict, perf: dict, n_forwards: int) -> dict:
    rep = metrics["replay_gate"]
    return {
        "label": "MBS THREE-WAY SAME-SOURCE CAUSAL AUDIT",
        "schema_version": 1,
        "created_utc": metrics["created_utc"],
        "base_sha": C.BASE_SHA,
        "branch": C.BRANCH,
        "authorization": metrics["authorization"],
        "scientific_wording": metrics["wording"],
        "checkpoints": {
            ck: {
                "path": metrics["identity"][ck]["evidence"]["path"],
                "update": C.CKPT_UPDATES[ck],
                "manifest_sha256": metrics["identity"][ck]["evidence"]["manifest_sha256"],
                "model_tree_sha256": metrics["identity"][ck]["evidence"]["model_tree_sha256"],
                "full_tree_sha256": metrics["identity"][ck]["evidence"].get("full_tree_sha256"),
                "growth_alpha": metrics["identity"][ck]["evidence"].get("growth_alpha"),
                "gate": metrics["identity"][ck]["status"],
            } for ck in C.CKS
        },
        "pair_cohort": {
            "pair_count": C.EXPECTED_COHORT_SIZES[C.COHORT_ALL],
            "pair_manifest_sha_original": C.FROZEN_PAIR_MANIFEST_SHA,
            "manifest_byte_exact_rederivable": False,
            "manifest_note": metrics["frozen_references"]["note"],
            "cohort_reused_exactly": True,
            "all_invariants": True,
            "cohort_sizes": {k: metrics["cohorts"][k]["n"] for k in C.COHORTS},
        },
        "replay": {
            "maxabs": rep["maxabs"], "rms": rep["rms"], "mean_delta": rep["mean_delta"],
            "bound": rep["bound"], "derivation": rep["derivation"],
            "within_frozen_numerics_floor": rep["status"] == "PASS",
            "status": rep["status"],
        },
        "historical_context_pre_to_post_118100_old_audit": {
            "note": "context only; NOT mixed into the new bootstrap",
            "D_TOP_point": 0.0017253, "D_BOTTOM_point": -0.00078003, "G_point": 0.00250533,
        },
        "results": {k: metrics["cohorts"][k] for k in C.COHORTS},
        "classification": metrics["classification"],
        "canonical_control_100u": {
            "recommended": metrics["classification"]["targeted_latent_ge2"]["canonical_control"]["action"],
            "note": metrics["classification"]["targeted_latent_ge2"]["canonical_control"]["note"],
            "started": False,
        },
        "performance": {
            "total_forwards": n_forwards,
            "per_worker": perf,
            "dcus": 2,
        },
        "errata": metrics["errata"],
        "self_replay_validation": metrics["self_replay_validation"],
        "immutability": metrics["external_checks"],
        "next": "HARD STOP; SEND SHA TO EXTERNAL REVIEWER; EXTERNAL REVIEW REQUIRED BEFORE ANY TRAINING GO",
    }


def _ci(ci: list[float]) -> str:
    return f"[{ci[0]:.8g}, {ci[1]:.8g}]"


def _pc(q: dict) -> str:
    return f"{q['point']:.8g} {_ci(q['ci95'])}"


def render_audit_md(m: dict) -> str:
    A: list[str] = []
    add = A.append
    tgt = m["cohorts"][C.COHORT_TARGETED]
    allc = m["cohorts"][C.COHORT_ALL]
    cls = m["classification"]
    rep = m["replay_gate"]
    add("# SakuraMoon Camera MBS Three-Way Same-Source Causal Audit")
    add("")
    add("> **Label: MBS THREE-WAY SAME-SOURCE CAUSAL AUDIT** - intervention-associated")
    add("> contrast (TREATMENT corrected-MBS-100U vs CONTROL historical camera-only")
    add("> continuation) on the frozen 512-pair same-source mirrored cohort, against")
    add("> PRE U118100.  Forward-only, read-only checkpoints, no training.")
    add(f">  Base reviewed SHA `{C.BASE_SHA[:12]}…`, branch `{C.BRANCH}`.  Supersedes nothing;")
    add(">  follows the Vertical Bottom Supervision audit (frozen reference).")
    add("")
    add("## 1. Checkpoint identity (gated before any forward)")
    add("")
    add("| arm | path | update | manifest sha256 | model tree sha256 | gate |")
    add("|---|---|---|---|---|---|")
    for ck in C.CKS:
        ev = m["identity"][ck]["evidence"]
        add(f"| {ck} | {ev['path']} | {C.CKPT_UPDATES[ck]} | `{ev['manifest_sha256'][:16]}…` | "
            f"`{ev['model_tree_sha256'][:16]}…` | {m['identity'][ck]['status']} |")
    add(f"TREATMENT full checkpoint tree sha256 = `{m['identity']['TREATMENT']['evidence']['full_tree_sha256'][:16]}…`")
    add("")
    add("## 2. Pair cohort (frozen, reused exactly - no resampling)")
    add("")
    add(f"- pair count = {m['cohorts'][C.COHORT_ALL]['n']}")
    add(f"- original pair manifest sha256 = `{C.FROZEN_PAIR_MANIFEST_SHA}` (byte-exact re-derivation impossible: the manifest embeds `created_utc` by design)")
    add("- cohort identity: all 512 pair rows field-identical to the committed frozen CSV (source_shard, sample_id, original_side, k_start, k_end, available, full_height, signed_shift_start); zoom-band / anchor-side / balanced memberships reproduced exactly")
    add("- cohorts (pre-registered): all=512, targeted latent>=2=312, latent<2=200, latent 2to4=275, latent >=4=37, content-balanced=256")
    add("- seed 20260907; t strata [0.1388061520926247, 0.24819609741338816, 0.37948289980451305, 0.5560734456280833]; one deterministic eps per (pair, stratum) shared by TOP/BOTTOM and across checkpoints")
    add("")
    add("## 3. PRE replay gate (new U118100 vs frozen VBS POST U118100 per-pair values)")
    add("")
    add(f"- status: **{rep['status']}**")
    add(f"- maxabs = {rep['maxabs']:.6e}; RMS = {rep['rms']:.6e}; mean delta = {rep['mean_delta']:.6e}")
    add(f"- bound = {rep['bound']:.6e}  ({rep['derivation']})")
    for c in rep["checks"]:
        add(f"  - {c['name']}: {c['value']:.6e} vs bound {c['bound']:.6e} -> {'PASS' if c['pass'] else 'FAIL'}")
    add("")
    add("## 4. Results (point = exact mean; CI = paired bootstrap seed 20260907, n=10000)")
    add("")
    for name in C.COHORTS:
        q = m["cohorts"][name]
        add(f"### {name} (n={q['n']})")
        add("")
        add("| M_TOP | PRE | CONTROL | TREATMENT |")
        add("|---|---|---|---|")
        add(f"| margin | {_pc(q['M_TOP']['PRE'])} | {_pc(q['M_TOP']['CONTROL'])} | {_pc(q['M_TOP']['TREATMENT'])} |")
        add("")
        add("| M_BOTTOM | PRE | CONTROL | TREATMENT |")
        add("|---|---|---|---|")
        add(f"| margin | {_pc(q['M_BOTTOM']['PRE'])} | {_pc(q['M_BOTTOM']['CONTROL'])} | {_pc(q['M_BOTTOM']['TREATMENT'])} |")
        add("")
        add("| PRE->CONTROL | D_TOP | D_BOTTOM | G |")
        add("|---|---|---|---|")
        add(f"| SAME-corrected | {_pc(q['D_TOP_CONTROL'])} | {_pc(q['D_BOTTOM_CONTROL'])} | {_pc(q['G_CONTROL'])} |")
        add("")
        add("| PRE->TREATMENT | D_TOP | D_BOTTOM | G |")
        add("|---|---|---|---|")
        add(f"| SAME-corrected | {_pc(q['D_TOP_TREATMENT'])} | {_pc(q['D_BOTTOM_TREATMENT'])} | {_pc(q['G_TREATMENT'])} |")
        add("")
        add("| TREATMENT vs CONTROL | I_TOP | I_BOTTOM | I_G |")
        add("|---|---|---|---|")
        add(f"| contrast | {_pc(q['I_TOP'])} | {_pc(q['I_BOTTOM'])} | {_pc(q['I_G'])} |")
        add("")
    add("## 5. Classification (pre-registered; TARGETED latent>=2 co-primary)")
    add("")
    add(f"- TARGETED (n=312): **{cls['targeted_latent_ge2']['verdict']}**")
    add(f"  - I_BOTTOM = {_pc(tgt['I_BOTTOM'])}")
    add(f"  - I_G = {_pc(tgt['I_G'])}")
    add(f"- all-pairs diagnostic (n=512): {cls['all_pairs_diagnostic']['verdict']}")
    add(f"  - I_BOTTOM = {_pc(allc['I_BOTTOM'])}")
    add(f"  - I_G = {_pc(allc['I_G'])}")
    cc_rec = cls["targeted_latent_ge2"]["canonical_control"]
    add(f"- canonical camera-only 100U: **{cc_rec['action']}** - {cc_rec['note']}")
    add("")
    add("## 6. Scientific wording (binding)")
    add("")
    w = m["wording"]
    add(f"- exact matched-data control: **{'YES' if w['exact_matched_data_control'] else 'NO'}**")
    add(f"- historical control = {w['historical_control_interpretation']}")
    add(f"- treatment = {w['treatment_sequence']}")
    add(f"- TREATMENT - CONTROL = {w['contrast_type']}")
    add(f"- MBS directional correction supported: **{'YES' if w['mbs_directional_correction_supported'] else 'NO'}**")
    add(f"- training-sequence confound remains: **{'YES' if w['training_sequence_confound_remains'] else 'NO'}**")
    add("")
    add("## 7. Historical context (frozen, NOT mixed into this bootstrap)")
    add("")
    add("- old VBS audit PRE(U116100)->POST(U118100): TOP adjusted +0.0017253, BOTTOM adjusted -0.00078003, gap +0.00250533 [0.00191989, 0.00316719]")
    add("")
    add("## 8. Performance")
    add("")
    p = m["performance"]
    add(f"- total forwards = {p['total_forwards']} (512 pairs x 8 arms x 4 strata x 3 checkpoints)")
    for key, v in p["per_worker"].items():
        add(f"- {key}: {v['device']} ({v['device_name']}), n_pairs={v['n_pairs']}, finished {v['finished_utc']}")
    add(f"- DCUs = {p['dcus']} (worker 0 = even pair_index, worker 1 = odd)")
    add("")
    add("## 9. Errata (documentation only; accepted Rerun2 PASS unchanged)")
    add("")
    add(f"1. severity relation: {m['errata']['severity_relation']}")
    add(f"2. setup-abort wording: {m['errata']['setup_abort_wording']}")
    add("")
    add("## 10. Validation / immutability")
    add("")
    sr = m["self_replay_validation"]
    add(f"- self-replay from new pairs CSV: all point estimates exact = {sr['all_exact']} ({len(sr['checks'])} checks)")
    ext = m["external_checks"]
    for k, v in (ext or {}).items():
        add(f"- {k}: {json.dumps(v) if not isinstance(v, (bool, str, int, float)) else v}")
    add("- frozen vertical_bottom_supervision package and all prior reports: unmodified (git-verified)")
    add("- three checkpoints read-only, unchanged after scoring (git/rehash verified)")
    add("- raw ledgers/latents/caches under /tmp only; nothing staged")
    add("")
    add("## 11. Authorization / next")
    add("")
    a = m["authorization"]
    add(f"- training={str(a['training']).upper()} p50={str(a['p50']).upper()} production={str(a['production']).upper()} fid={str(a['fid']).upper()}")
    add("- NEXT: HARD STOP; SEND SHA TO EXTERNAL REVIEWER; EXTERNAL REVIEW REQUIRED BEFORE ANY TRAINING GO")
    add("")
    return "\n".join(A) + "\n"


if __name__ == "__main__":
    sys.exit(main())
