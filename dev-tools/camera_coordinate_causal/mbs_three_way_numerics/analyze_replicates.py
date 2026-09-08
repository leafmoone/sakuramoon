"""MBS Three-Way Numerics Adjudication — replicate-robustness analysis.

NEW tooling (numerics-v2).  Reuses, READ-ONLY:
  * the reviewed three-way tooling mbs_three_way (pair-cohort identity
    gate, cohort index sets, exact mean, frozen replay constants), and
  * the FROZEN VBS contracts (bootstrap mechanics, classification tree).

Reads exactly six raw replicate ledgers (PRE A/B, CONTROL A/B,
TREATMENT A/B — 2 files of 256 rows each per replicate), verifies the
raw hash freeze, and produces the 4 report files under the worktree
reports/ directory.

NO torch import.  CPU float64 only.  No per-pair maxabs hard gate
anywhere: per-pair tail statistics are written out as diagnostics and
never enter the adjudication.

Exit codes: 0 = analysis complete (PASS or AMBIGUOUS verdict both are
legitimate scientific outcomes and are recorded as such),
2 = protocol violation (freeze mismatch, completeness, PRE-cancellation
implementation bug).
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

_HERE = Path(__file__).resolve().parent
_MBS_DIR = _HERE.parent / "mbs_three_way"
# Register the mbs modules under their OWN bare names first (their code uses
# bare `import contracts`), then load this package's contracts under a
# unique name so the bare name "contracts" never binds to THIS module.
for _p in (str(_MBS_DIR), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import analyze_three_way  # noqa: F401  (mbs's own module)
import contracts  # noqa: F401  (mbs's own module)


def _load_unique(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None, f"missing module file: {path}"
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


Cn = _load_unique("mbs3w_numerics_contracts", _HERE / "contracts.py")

WORKTREE = Path("/sakuramoon-runtime/sakuramoon-camera-mbs-numerics-adjudication")
REPORTS = WORKTREE / "reports"

M_TOP, M_BOT, F_TOP, F_BOT = "m_top", "m_bot", "f_top", "f_bot"
I_METRICS = ("i_top", "i_bot", "i_g")
MARGIN_METRICS = (M_TOP, M_BOT, F_TOP, F_BOT)
CONTRAST_METRICS = ("d_top_c", "d_bot_c", "g_c", "d_top_t", "d_bot_t", "g_t", *I_METRICS)


def log(msg: str) -> None:
    print(f"[numerics {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {msg}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze-manifest", type=Path, default=Cn.FREEZE_MANIFEST)
    ap.add_argument("--reports-dir", type=Path, default=REPORTS)
    args = ap.parse_args()

    log("starting numerics adjudication analysis")
    args.reports_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1) raw hash freeze (protocol s22) — BEFORE reading any science ----
    with open(args.freeze_manifest, encoding="utf-8") as fh:
        manifest = json.load(fh)
    Cn.verify_freeze(manifest)
    log("raw replicate freeze verified (6 ledgers)")

    # ---- 2) pair-cohort identity + cohort membership (frozen, reused) ------
    ident = Cn.MBS_A3.check_pair_cohort_identity(Cn.VBS_ROOT, Path(str(args.reports_dir) + ".pair.log"))
    if ident is None:
        log("STOP: pair-cohort identity gate failed")
        return 2
    pairs, frozen_rows, n = ident
    if n != Cn.N_PAIRS:
        log(f"STOP: pair count {n} != {Cn.N_PAIRS}")
        return 2
    cohorts = Cn.MBS_A3.cohort_index_sets(pairs, frozen_rows)
    for name in Cn.COHORTS:
        if len(cohorts[name]) != Cn.EXPECTED_COHORT_SIZES[name]:
            log(f"STOP: cohort {name} size {len(cohorts[name])} != {Cn.EXPECTED_COHORT_SIZES[name]}")
            return 2
    log(f"pair-cohort identity PASS ({n} pairs, all cohort sizes as frozen)")

    # ---- 3) load 6 raw replicates + per-pair margins (frozen expression) ---
    margins: dict[str, dict[str, list[dict[str, float]]]] = {}
    for ck in Cn.CKS:
        margins[ck] = {}
        for rep in Cn.REPS:
            rows = Cn.load_replicate_ledger(ck, rep, n)
            margins[ck][rep] = [Cn.pair_margins(rows[i]) for i in range(n)]
            log(f"loaded {ck}/{rep}: 512 pairs, 8 arms x 4 strata, all finite")

    # ---- 4) all 8 replicate tuples -> per-pair contrasts -------------------
    tuples = Cn.replicate_tuples()
    assert len(tuples) == 8
    per_tuple: dict[str, dict[str, list[float]]] = {}
    for label, p, c, t in tuples:
        pre = margins["PRE"][p]
        con = margins["CONTROL"][c]
        tre = margins["TREATMENT"][t]
        vals: dict[str, list[float]] = {m: [] for m in CONTRAST_METRICS}
        for i in range(n):
            ctr = Cn.pair_contrast(pre[i], con[i], tre[i])
            for m in CONTRAST_METRICS:
                vals[m].append(ctr[m])
        per_tuple[label] = vals

    # ---- 5) PRE cancellation diagnostic proof (protocol s24) ---------------
    # I computed through the UNFACTORED frozen formulas with PRE_A vs PRE_B
    # must agree within fp64 roundoff for every C/T combination and metric.
    cancel: dict[str, dict[str, float]] = {}
    cancel_ok = True
    for combo_label, c, t in Cn.ct_combinations():
        per_combo: dict[str, float] = {}
        ci = Cn.REPS.index(c)
        ti = Cn.REPS.index(t)
        for m in I_METRICS:
            ia = per_tuple[f"P0 C{ci} T{ti}"][m]
            ib = per_tuple[f"P1 C{ci} T{ti}"][m]
            mx = Cn.pre_cancellation_max_abs([x - y for x, y in zip(ia, ib, strict=True)])
            per_combo[m] = mx
            if mx > Cn.PRE_CANCELLATION_TOL:
                cancel_ok = False
        cancel[combo_label] = per_combo
    if not cancel_ok:
        log(f"STOP: PRE cancellation exceeded {Cn.PRE_CANCELLATION_TOL} — implementation bug (protocol s24)")
        return 2
    log("PRE cancellation proof PASS (I is PRE-invariant within fp64 roundoff)")

    # ---- 6) shared bootstrap matrices + all cohort x tuple x metric CIs ----
    # ONE index matrix per cohort (depends only on seed and cohort size),
    # reused across ALL tuples and combinations (protocol s11).
    idx_by_cohort = {name: Cn.shared_bootstrap_index_matrix(len(idxs)) for name, idxs in cohorts.items()}

    results: dict[str, dict[str, dict[str, Any]]] = {}  # cohort -> tuple -> metric
    for name, idxs in cohorts.items():
        idx = idx_by_cohort[name]
        results[name] = {}
        for label, _, _, _ in tuples:
            row: dict[str, Any] = {}
            for m in I_METRICS:
                arr = np.asarray([per_tuple[label][m][i] for i in idxs], dtype=np.float64)
                lo, hi = Cn.bootstrap_ci_shared(arr, idx)
                row[m] = {
                    "point": Cn.MBS_A3.mean_exact([per_tuple[label][m][i] for i in idxs]),
                    "ci95": [lo, hi],
                }
            results[name][label] = row

    # ---- 7) classification: all 8 tuples (TARGETED) + conservative envelope -
    targeted = results[Cn.COHORT_TARGETED]
    tuple_classes = [Cn.classify_contrast(targeted[lb]["i_bot"]["ci95"], targeted[lb]["i_g"]["ci95"]) for lb, _, _, _ in tuples]
    class_doc = {lb: cl for (lb, _, _, _), cl in zip(tuples, tuple_classes, strict=True)}

    # Envelope over the 4 unique C/T combinations (PRE-invariant by s24;
    # values taken from the P0 variants, which equal P1 within roundoff).
    combos = Cn.ct_combinations()
    env_doc: dict[str, dict[str, Any]] = {}
    for name in Cn.COHORTS:
        env_doc[name] = {}
        for m in I_METRICS:
            points: list[float] = []
            cis: list[tuple[float, float]] = []
            for _, c, t in combos:
                ci = Cn.REPS.index(c)
                ti = Cn.REPS.index(t)
                cell = results[name][f"P0 C{ci} T{ti}"][m]
                points.append(cell["point"])
                cis.append((cell["ci95"][0], cell["ci95"][1]))
            env_doc[name][m] = Cn.envelope(points, cis)
    # Envelope classification: the SAME decision tree applied to BOTH
    # envelope CIs in one decision call:
    ebot = env_doc[Cn.COHORT_TARGETED]["i_bot"]
    eg = env_doc[Cn.COHORT_TARGETED]["i_g"]
    envelope_class = Cn.classify_envelope((ebot["ci_low"], ebot["ci_high"]), (eg["ci_low"], eg["ci_high"]))

    # ---- 8) final adjudication (protocol s15) --------------------------------
    adj = Cn.adjudicate(tuple_classes, envelope_class)

    # ---- 9) diagnostics: A-vs-B per-pair + estimator spread (DIAGNOSTICS) ---
    diag: dict[str, Any] = {}
    for ck in Cn.CKS:
        diag[ck] = {}
        for m in MARGIN_METRICS:
            a = [margins[ck]["A"][i][m] for i in range(n)]
            b = [margins[ck]["B"][i][m] for i in range(n)]
            diag[ck][m] = Cn.replicate_diff_stats(a, b)
    spread: dict[str, Any] = {}
    for m in I_METRICS:
        points = [targeted[lb][m]["point"] for lb, _, _, _ in tuples]
        widths = [targeted[lb][m]["ci95"][1] - targeted[lb][m]["ci95"][0] for lb, _, _, _ in tuples]
        spread[m] = Cn.point_spread_ratio(points, widths)
    diag["estimator_spread_targeted"] = spread

    # ---- 10) new pairs CSV (17 sig figs: exact float64 round-trip) ----------
    csv_path = args.reports_dir / "camera-mbs-three-way-numerics-pairs.csv"
    header = (
        "pair_id,source_shard,sample_id,original_side,zoom_band,"
        "k_start,k_end,available,full_height,signed_shift_start,latent_shift,"
        "latent_band,in_balanced_subset,content_asymmetry,"
        + ",".join(f"{m}_{ck.lower()}_{rep}" for m in MARGIN_METRICS for ck in Cn.CKS for rep in Cn.REPS)
        + ","
        + ",".join(f"{m}_{cl}" for m in I_METRICS for cl, _, _ in combos)
    )
    def fmt(x: float) -> str:
        return f"{x:.17g}"
    with open(csv_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(header + "\n")
        for i, fr in enumerate(frozen_rows):
            fl = Cn.MBS_C.pair_cohort_flags(fr)
            vals_csv: list[str | int | float] = [
                fr["pair_id"], fr["source_shard"], fr["sample_id"], fr["original_side"], fr["zoom_band"],
                fr["k_start"], fr["k_end"], fr["available"], fr["full_height"], fmt(fr["signed_shift_start"]),
                fmt(fl["latent_shift"]), fl["latent_band"], "True" if fl["in_balanced"] else "False",
                fmt(fr["content_asymmetry"]),
            ]
            for m in MARGIN_METRICS:
                for ck in Cn.CKS:
                    for rep in Cn.REPS:
                        vals_csv.append(fmt(margins[ck][rep][i][m]))
            for m in I_METRICS:
                for _, c, t in combos:
                    ci = Cn.REPS.index(c)
                    ti = Cn.REPS.index(t)
                    vals_csv.append(fmt(per_tuple[f"P0 C{ci} T{ti}"][m][i]))
            fh.write(",".join(str(x) for x in vals_csv) + "\n")
    log(f"wrote pairs CSV: {csv_path.name} ({n} rows)")

    # ---- 11) self-replay: recompute ALL point estimates from the new CSV ----
    replay_ok = True
    with open(csv_path, encoding="utf-8") as fh:
        rdr = csv.DictReader(fh)
        rows_csv = list(rdr)
    if len(rows_csv) != n:
        replay_ok = False
    else:
        # per-tuple points: reconstructed from the CSV margin columns via
        # the frozen per-pair formulas (17-sig-fig round-trip), and must
        # match the in-memory reference EXACTLY.
        for name, idxs in cohorts.items():
            for label, p, c, t in tuples:
                for m in I_METRICS:
                    ref = results[name][label][m]["point"]
                    if not _replay_tuple_point(rows_csv, idxs, m, p, c, t, ref):
                        replay_ok = False
                        log(f"REPLAY MISMATCH cohort={name} tuple={label} {m}")
        # per-combination I columns replay directly (combo labels C0/T0 ...
        # map to Cn.REPS letters via the same enumeration used when writing):
        for name, idxs in cohorts.items():
            for m in I_METRICS:
                for cl, c, t in combos:
                    point_csv = Cn.MBS_A3.mean_exact([float(rows_csv[i][f"{m}_{cl}"]) for i in idxs])
                    ref = results[name][f"P0 C{Cn.REPS.index(c)} T{Cn.REPS.index(t)}"][m]["point"]
                    if point_csv != ref:
                        replay_ok = False
                        log(f"REPLAY MISMATCH cohort={name} {m} {cl}: csv {point_csv!r} != ref {ref!r}")
    if not replay_ok:
        log("STOP: CSV self-replay failed to exactly reproduce point estimates")
        return 2
    log("CSV self-replay: all point estimates reproduced exactly")

    # ---- 12) metrics + adjudication + report files --------------------------
    metrics_doc: dict[str, Any] = {
        "label": "MBS THREE-WAY NUMERICS ADJUDICATION — REPLICATE-ROBUSTNESS METRICS",
        "bootstrap": {"seed": Cn.BOOT_SEED, "n_boot": Cn.N_BOOT, "unit": "source pair",
                     "shared_index_matrix": "one per cohort, reused across all tuples/combos"},
        "cohorts": {name: len(idxs) for name, idxs in cohorts.items()},
        "tuples": [lb for lb, _, _, _ in tuples],
        "ct_combinations": [cl for cl, _, _ in combos],
        "pre_cancellation": {"tolerance": Cn.PRE_CANCELLATION_TOL, "max_abs_by_combo": cancel, "status": "PASS"},
        "per_tuple": results,
        "classification_by_tuple_targeted": class_doc,
        "envelope_targeted": {m: env_doc[Cn.COHORT_TARGETED][m] for m in I_METRICS},
        "envelope_classification": envelope_class,
        "adjudication": adj,
        "diagnostics_only": diag,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (args.reports_dir / "camera-mbs-three-way-numerics-metrics.json").write_text(
        json.dumps(metrics_doc, indent=2) + "\n", encoding="utf-8")

    adj_doc = {
        "label": "MBS THREE-WAY NUMERICS ADJUDICATION — FINAL",
        "question": ("Does HCU/DCU forward numerical nondeterminism change the "
                     "intervention-associated estimator classification under a "
                     "same-source frozen evaluation cohort?"),
        "replicate_inventory": manifest,
        "tuple_classifications_targeted": class_doc,
        "adjudication": adj,
        "canonical_control_recommendation": (
            Cn.MBS_C.canonical_control_recommendation(adj["common_classification"] or "")
            if adj["verdict"] == Cn.ADJUDICATION_PASS
            else {"status": "DEFERRED", "reason": "numerical adjudication AMBIGUOUS — no directional release"}
        ),
        "root_cause_statement": ("localized HCU/DCU forward numerical nondeterminism; "
                                  "root operator/kernel family not isolated"),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (args.reports_dir / "camera-mbs-three-way-numerics-adjudication.json").write_text(
        json.dumps(adj_doc, indent=2) + "\n", encoding="utf-8")

    md = render_markdown(adj, class_doc, env_doc, diag, spread, results, cancel)
    (args.reports_dir / "camera-mbs-three-way-numerics-adjudication.md").write_text(md, encoding="utf-8")
    log(f"NUMERICAL_ADJUDICATION = {adj['verdict']} "
        f"(stable={adj['stable_label']}, common={adj['common_classification']}, envelope={envelope_class})")
    return 0


def _replay_tuple_point(
    rows_csv: list[dict[str, str]],
    idxs: list[int],
    m: str,
    p: str,
    c: str,
    t: str,
    ref_point: float,
) -> bool:
    """Recompute one tuple x cohort x metric point estimate from the CSV
    margin columns using the frozen per-pair formulas, and require EXACT
    equality with the in-memory reference (17-sig-fig round-trip)."""

    def g(r: dict[str, str], mm: str, ck: str, rep: str) -> float:
        return float(r[f"{mm}_{ck.lower()}_{rep}"])

    vals: list[float] = []
    for i in idxs:
        r = rows_csv[i]
        d_top_c = (g(r, M_TOP, "CONTROL", c) - g(r, M_TOP, "PRE", p)) - (g(r, F_TOP, "CONTROL", c) - g(r, F_TOP, "PRE", p))
        d_bot_c = (g(r, M_BOT, "CONTROL", c) - g(r, M_BOT, "PRE", p)) - (g(r, F_BOT, "CONTROL", c) - g(r, F_BOT, "PRE", p))
        g_c = d_top_c - d_bot_c
        d_top_t = (g(r, M_TOP, "TREATMENT", t) - g(r, M_TOP, "PRE", p)) - (g(r, F_TOP, "TREATMENT", t) - g(r, F_TOP, "PRE", p))
        d_bot_t = (g(r, M_BOT, "TREATMENT", t) - g(r, M_BOT, "PRE", p)) - (g(r, F_BOT, "TREATMENT", t) - g(r, F_BOT, "PRE", p))
        g_t = d_top_t - d_bot_t
        vals.append({"i_top": d_top_t - d_top_c, "i_bot": d_bot_t - d_bot_c, "i_g": g_t - g_c}[m])
    return Cn.MBS_A3.mean_exact(vals) == ref_point


def render_markdown(
    adj: dict[str, Any],
    class_doc: dict[str, str],
    env_doc: dict[str, Any],
    diag: dict[str, Any],
    spread: dict[str, Any],
    results: dict[str, dict[str, dict[str, Any]]],
    cancel: dict[str, dict[str, float]],
) -> str:
    t = results[Cn.COHORT_TARGETED]
    lines: list[str] = []
    ap = lines.append
    ap("# MBS Three-Way Numerics Adjudication (replicate-robustness, numerics-v2)")
    ap("")
    ap("Pre-registered protocol frozen at H1 (see `mbs_three_way_numerics/README.md`). "
       "NO per-pair maxabs hard gate; per-pair tail statistics are diagnostics only.")
    ap("")
    ap("**Question.** Does HCU/DCU forward numerical nondeterminism change the "
       "intervention-associated estimator classification under a same-source frozen "
       "evaluation cohort?")
    ap("")
    ap(f"**NUMERICAL_ADJUDICATION: {adj['verdict']}** — {adj['stable_label']}, "
       f"common classification: {adj['common_classification']}, "
       f"conservative envelope: {adj['envelope_classification']}.")
    ap("")
    ap("## 1. Replicate tuples (TARGETED latent>=2, n=312)")
    ap("")
    ap("| tuple | I_BOTTOM point [CI] | I_G point [CI] | class |")
    ap("|---|---|---|---|")
    for lb, _, _, _ in Cn.replicate_tuples():
        ib, ig = t[lb]["i_bot"], t[lb]["i_g"]
        ap(f"| {lb} | {ib['point']:.6f} [{ib['ci95'][0]:.6f}, {ib['ci95'][1]:.6f}] "
           f"| {ig['point']:.6f} [{ig['ci95'][0]:.6f}, {ig['ci95'][1]:.6f}] | {class_doc[lb]} |")
    ap("")
    ap("## 2. Conservative envelope (4 unique C/T combinations, TARGETED)")
    ap("")
    for m in I_METRICS:
        e = env_doc[Cn.COHORT_TARGETED][m]
        ap(f"- **{m}**: point in [{e['point_min']:.6f}, {e['point_max']:.6f}], "
           f"envelope CI [{e['ci_low']:.6f}, {e['ci_high']:.6f}]")
    ap("")
    ap("## 3. PRE cancellation (diagnostic proof, protocol s24)")
    ap("")
    ap(f"tolerance {Cn.PRE_CANCELLATION_TOL:.0e}; max |I(P_A)-I(P_B)| per combination: "
       + "; ".join(f"{k}: " + ", ".join(f"{m}={v:.2e}" for m, v in v2.items()) for k, v2 in cancel.items()) + ". **PASS**")
    ap("")
    ap("## 4. Replicate A vs B per-pair diagnostics (DIAGNOSTICS ONLY)")
    ap("")
    for ck in Cn.CKS:
        for m in MARGIN_METRICS:
            s = diag[ck][m]
            ap(f"- {ck}/{m}: bit-exact {s['bitexact_count']}/512 "
               f"({s['bitexact_fraction']*100:.1f}%), maxabs {s['max_abs']:.3e}, rms {s['rms']:.3e}, "
               f">1e-6: {s['count_gt_1e-6']}, >1e-5: {s['count_gt_1e-5']}, >1e-4: {s['count_gt_1e-4']}")
    ap("")
    ap("## 5. Estimator numerical spread (diagnostic only)")
    ap("")
    for m in I_METRICS:
        s = spread[m]
        ap(f"- {m}: point spread {s['point_spread']:.3e} vs median CI width {s['median_ci_width']:.3e} "
           f"(ratio {s['ratio']:.3f})")
    ap("")
    ap("## 6. Other cohorts (points + CIs, all tuples)")
    ap("")
    for name in Cn.COHORTS:
        if name == Cn.COHORT_TARGETED:
            continue
        row = ", ".join(
            f"{m}: {results[name][Cn.replicate_tuples()[0][0]][m]['point']:.6f}"
            for m in I_METRICS)
        ap(f"- {name} (n={Cn.EXPECTED_COHORT_SIZES[name]}), first tuple: {row}")
    ap("")
    ap("## 7. Statement")
    ap("")
    if adj["verdict"] == Cn.ADJUDICATION_PASS:
        ap(f"The numerical nondeterminism is **{adj['stable_label']}** across all 8 replicate "
           f"tuples: the robust classification is **{adj['common_classification']}**, and it "
           f"agrees with the conservative replicate envelope. This is a "
           f"**replicate-robust intervention-associated contrast under a same-source frozen "
           f"evaluation cohort** — NOT an exact causal effect. Root cause wording: localized "
           f"HCU/DCU forward numerical nondeterminism; root operator/kernel family not isolated.")
    else:
        ap("The numerical nondeterminism is **NOT demonstrated stable** across replicate "
           "tuples (or the envelope disagrees with the common class). NO directional "
           "classification is released; further numerical stabilization or an explicit "
           "tolerance review is required before any decision.")
    ap("")
    ap("## 8. All-cohort detail")
    ap("")
    ap("See `camera-mbs-three-way-numerics-metrics.json` (`per_tuple`) for every "
       "cohort x tuple x metric point + CI, and "
       "`camera-mbs-three-way-numerics-pairs.csv` (self-replay verified) for the "
       "per-pair replicate margins and C/T combination contrasts.")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
