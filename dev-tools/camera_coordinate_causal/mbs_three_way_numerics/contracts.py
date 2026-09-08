"""MBS Three-Way Numerics Adjudication — pre-registered protocol contracts.

NEW companion package (numerics-v2) to the reviewed three-way tooling
(dev-tools/camera_coordinate_causal/mbs_three_way/, tooling SHA 9d428faf)
and the FROZEN VBS package (vertical_bottom_supervision/).  NEITHER is
modified; both are imported read-only from explicit paths under distinct
module names (a same-basename `import contracts` would resolve to THIS
module and shadow them).

This file FREEZES the replicate-robustness protocol BEFORE TREATMENT_B is
scored.  README.md holds the binding protocol text (sections 0-17 of the
round prompt); the constants below are the executable form.

Core principle (protocol s7):

  NO per-pair maxabs hard gate.
  NO aggregate SAME CI -> per-pair threshold conversion.
  Per-pair numerical tail statistics are DIAGNOSTICS ONLY.

The hard scientific question is whether replicate choice changes the final
estimator classification (I_TOP / I_BOTTOM / I_G, TARGETED latent>=2).

No torch import at module load.  Pure numpy + stdlib.
"""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Read-only imports of the reviewed three-way tooling.  mbs_three_way's own
# modules use BARE `import contracts` / `import analyze_three_way`, so they
# must be imported under their own names with the mbs directory first on
# sys.path — otherwise the bare `import contracts` inside analyze_three_way
# would bind to THIS package's same-named module (shadowing bug found in the
# previous round; the explicit-name trick that works for mbs's own VBS import
# does NOT work in the reverse direction).
# ---------------------------------------------------------------------------
_MBS_DIR = Path(__file__).resolve().parent.parent / "mbs_three_way"


def _mbs_module(name: str, fname: str):
    """Import one mbs_three_way module under its own bare name, verifying it
    resolved to the mbs directory (never to this package's shadow)."""
    expected = str(_MBS_DIR / fname)
    existing = sys.modules.get(name)
    if existing is not None and getattr(existing, "__file__", None) == expected:
        return existing
    sp = str(_MBS_DIR)
    if sp not in sys.path:
        sys.path.insert(0, sp)
    mod = importlib.import_module(name)
    if getattr(mod, "__file__", None) != expected:
        raise ImportError(f"{name} resolved to {getattr(mod, '__file__', None)!r}, expected {expected!r}")
    return mod


MBS_C = _mbs_module("contracts", "contracts.py")
MBS_A3 = _mbs_module("analyze_three_way", "analyze_three_way.py")
VBS = MBS_C.VBS  # frozen VBS contracts (re-exported by mbs contracts)

# ---------------- frozen, reused (checkpoint-independent) -------------------
ARMS: tuple[str, ...] = MBS_C.ARMS
CKS: tuple[str, str, str] = MBS_C.CKS
BOOT_SEED: int = MBS_C.BOOT_SEED          # 20260907 (frozen)
N_BOOT: int = MBS_C.N_BOOT                # 10000 (frozen)
VBS_ROOT = MBS_C.VBS_ROOT                 # /tmp/camera-vertical-bottom (frozen pair root)
FROZEN_PAIRS_CSV = MBS_C.FROZEN_PAIRS_CSV

# Cohort names / expected sizes: reused verbatim from the reviewed tooling.
COHORT_ALL = MBS_C.COHORT_ALL
COHORT_TARGETED = MBS_C.COHORT_TARGETED
COHORT_LT2 = MBS_C.COHORT_LT2
COHORT_2TO4 = MBS_C.COHORT_2TO4
COHORT_GE4 = MBS_C.COHORT_GE4
COHORT_BALANCED = MBS_C.COHORT_BALANCED
COHORTS: tuple[str, ...] = MBS_C.COHORTS
EXPECTED_COHORT_SIZES: dict[str, int] = MBS_C.EXPECTED_COHORT_SIZES

# ---------------- replicate layout (protocol s8/s21/s22) -------------------
N_PAIRS = 512

# Raw roots per (checkpoint, replicate).  A = original three-way scoring run;
# B = independent fresh-process numerical realization.  PRE/CONTROL B is the
# full re-verification run; TREATMENT B is scored THIS round into a fresh root
# that must not contain previous ledgers.
REP_ROOTS: dict[str, dict[str, Path]] = {
    "PRE": {"A": Path("/tmp/camera-mbs-three-way-audit"), "B": Path("/tmp/mbs3-rerun-full")},
    "CONTROL": {"A": Path("/tmp/camera-mbs-three-way-audit"), "B": Path("/tmp/mbs3-rerun-full")},
    "TREATMENT": {"A": Path("/tmp/camera-mbs-three-way-audit"), "B": Path("/tmp/camera-mbs-three-way-numerics/treatment-b")},
}

NUMERICS_ROOT = Path("/tmp/camera-mbs-three-way-numerics")
FREEZE_MANIFEST = NUMERICS_ROOT / "raw-replicates.json"


def ledger_path(ck: str, rep: str, worker: int) -> Path:
    """Exact ledger file for one (checkpoint, replicate, worker)."""
    return REP_ROOTS[ck][rep] / "ledger" / f"ledger-w{worker}-{ck.lower()}.jsonl"


# ---------------- replicate combinations (protocol s10) ---------------------
REPS: tuple[str, str] = ("A", "B")


def replicate_tuples() -> list[tuple[str, str, str, str]]:
    """ALL 8 replicate tuples (P_i, C_j, T_k), fixed order, no cherry-picking.

    P0/P1 = PRE A/B, C0/C1 = CONTROL A/B, T0/T1 = TREATMENT A/B."""
    out: list[tuple[str, str, str, str]] = []
    for pi, p in enumerate(REPS):
        for ci, c in enumerate(REPS):
            for ti, t in enumerate(REPS):
                out.append((f"P{pi} C{ci} T{ti}", p, c, t))
    return out


def ct_combinations() -> list[tuple[str, str, str]]:
    """The 4 unique C/T combinations for I-only endpoints and the envelope
    (PRE cancels from I; the 8-tuple analysis proves that fact)."""
    out: list[tuple[str, str, str]] = []
    for ci, c in enumerate(REPS):
        for ti, t in enumerate(REPS):
            out.append((f"C{ci}/T{ti}", c, t))
    return out


# ---------------- per-pair margins (frozen verbatim expression) -------------
def pair_margins(losses: dict[str, list[float]]) -> dict[str, float]:
    """m_top/m_bot/f_top/f_bot/bb_minus_tt = 4-stratum mean, FROZEN VBS
    expression verbatim (identical to mbs_three_way.analyze_three_way
    .per_pair_margins and to vertical_bottom_supervision/analyze.py):
    (sum(armA) - sum(armB)) / 4.0 — the two arm sums are taken separately
    before subtraction (bit-faithful; do NOT rewrite)."""
    return {
        "m_top": (sum(losses["TB"]) - sum(losses["TT"])) / 4.0,
        "m_bot": (sum(losses["BT"]) - sum(losses["BB"])) / 4.0,
        "f_top": (sum(losses["T_SAME"]) - sum(losses["TT"])) / 4.0,
        "f_bot": (sum(losses["B_SAME"]) - sum(losses["BB"])) / 4.0,
        "bb_minus_tt": (sum(losses["BB"]) - sum(losses["TT"])) / 4.0,
    }


def pair_contrast(pre: dict[str, float], con: dict[str, float], tre: dict[str, float]) -> dict[str, float]:
    """Per-pair D/G/I via the ORIGINAL frozen formulas (identical operation
    order to mbs_three_way.analyze_three_way.main, which itself reuses the
    frozen VBS adjusted-delta semantics):

      d_top_c = (con.m_top - pre.m_top) - (con.f_top - pre.f_top)
      d_bot_c = (con.m_bot - pre.m_bot) - (con.f_bot - pre.f_bot)
      g_c     = d_top_c - d_bot_c
      (same with tre for treatment),  i_* = treatment - control.
    """
    d_top_c = (con["m_top"] - pre["m_top"]) - (con["f_top"] - pre["f_top"])
    d_bot_c = (con["m_bot"] - pre["m_bot"]) - (con["f_bot"] - pre["f_bot"])
    g_c = d_top_c - d_bot_c
    d_top_t = (tre["m_top"] - pre["m_top"]) - (tre["f_top"] - pre["f_top"])
    d_bot_t = (tre["m_bot"] - pre["m_bot"]) - (tre["f_bot"] - pre["f_bot"])
    g_t = d_top_t - d_bot_t
    return {
        "d_top_c": d_top_c, "d_bot_c": d_bot_c, "g_c": g_c,
        "d_top_t": d_top_t, "d_bot_t": d_bot_t, "g_t": g_t,
        "i_top": d_top_t - d_top_c,
        "i_bot": d_bot_t - d_bot_c,
        "i_g": g_t - g_c,
    }


# ---------------- bootstrap (protocol s11) ----------------------------------
def shared_bootstrap_index_matrix(n: int, seed: int = BOOT_SEED, n_boot: int = N_BOOT) -> np.ndarray:
    """ONE deterministic source-pair bootstrap index matrix per cohort size.

    Mechanically identical to the frozen VBS.bootstrap_ci internals
    (rng.integers(0, n, (n_boot, n)) under np.random.default_rng(seed)); it
    depends on (seed, n) ONLY — never on the values — so the exact same
    matrix is reused across all checkpoint replicates, all replicate tuples,
    and all C/T combinations within a cohort.
    """
    rng = np.random.default_rng(seed)
    return rng.integers(0, n, size=(n_boot, n))


def bootstrap_ci_shared(values: np.ndarray, idx: np.ndarray) -> tuple[float, float]:
    """Percentile 95% CI of the bootstrap means using the SHARED index
    matrix.  Identical arithmetic to VBS.bootstrap_ci for the same values."""
    a = np.asarray(values, dtype=np.float64)
    means = a[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# ---------------- classification (protocol s13 — unchanged tree) ------------
CLASS_CORRECTION = "CLEAR_DIRECTIONAL_CORRECTION"
CLASS_HARM = "CLEAR_HARM"
CLASS_BORDERLINE = "BORDERLINE"

# The per-combination decision tree is the EXACT already pre-registered one:
classify_contrast = MBS_C.classify_contrast


def classify_envelope(i_bottom_ci: tuple[float, float], i_g_ci: tuple[float, float]) -> str:
    """Conservative-envelope classification: the SAME decision tree applied
    to the conservative envelope CI bounds (protocol s15)."""
    return classify_contrast(i_bottom_ci, i_g_ci)


# ---------------- envelope (protocol s14) ------------------------------------
def envelope(
    points: list[float], cis: list[tuple[float, float]]
) -> dict[str, float]:
    """Conservative replicate envelope over ALL unique C/T numerical
    combinations for one cohort+metric: worst low, worst high, min/max
    point.  Caller passes exactly len(ct_combinations()) entries."""
    assert len(points) == len(cis) and len(points) > 0
    return {
        "point_min": min(points),
        "point_max": max(points),
        "ci_low": min(lo for lo, _ in cis),
        "ci_high": max(hi for _, hi in cis),
    }


# ---------------- final adjudication (protocol s15) --------------------------
STABLE = "NUMERICALLY_STABLE"
AMBIGUOUS_STABLE = "NUMERICALLY_AMBIGUOUS"
ADJUDICATION_PASS = "PASS"
ADJUDICATION_AMBIGUOUS = "AMBIGUOUS"


def adjudicate(
    tuple_classes: list[str],
    envelope_class: str,
) -> dict[str, Any]:
    """NUMERICALLY_STABLE iff ALL 8 TARGETED replicate tuples produce the
    SAME classification.  If stable: robust classification = the common
    class; if common class == conservative envelope class then
    NUMERICAL_ADJUDICATION = PASS, else AMBIGUOUS (no directional release).
    If not stable: NUMERICAL_ADJUDICATION = AMBIGUOUS.

    NOTE (protocol s7): this function consumes ONLY the combination-level
    classification labels — it never consults per-pair maxabs/RMS or any
    other per-pair numerical tail statistic.  No per-pair hard gate exists
    in the final adjudication.
    """
    if len(set(tuple_classes)) == 1:
        stable = True
        common = tuple_classes[0]
    else:
        stable = False
        common = None
    if stable and common == envelope_class:
        verdict = ADJUDICATION_PASS
    else:
        verdict = ADJUDICATION_AMBIGUOUS
    return {
        "stable": stable,
        "stable_label": STABLE if stable else AMBIGUOUS_STABLE,
        "common_classification": common,
        "envelope_classification": envelope_class,
        "verdict": verdict,
    }


# ---------------- PRE cancellation (protocol s24) ----------------------------
# Pre-registered tolerance: the I contrast computed through the UNFACTORED
# frozen formulas with PRE_A vs PRE_B must agree within ordinary CPU float64
# arithmetic roundoff.  Values are O(1e-3); fp64 eps at that scale is ~1e-19.
# 1e-12 is 7 orders of magnitude above roundoff and ~6 orders below any
# scientifically meaningful contrast scale.  Exceeding it = implementation
# bug = STOP.
PRE_CANCELLATION_TOL = 1e-12


def pre_cancellation_max_abs(diff: list[float]) -> float:
    """Max |I(P_A) - I(P_B)| over pairs for one metric+combination."""
    return max((abs(x) for x in diff), default=0.0)


# ---------------- A-vs-B diagnostics (protocol s17 — DIAGNOSTICS ONLY) -------
def replicate_diff_stats(a: list[float], b: list[float]) -> dict[str, Any]:
    """Per-pair A-vs-B difference statistics for one margin/floor metric.
    DIAGNOSTICS ONLY — these numbers never enter the adjudication."""
    assert len(a) == len(b)
    n = len(a)
    d = [x - y for x, y in zip(a, b, strict=True)]
    ad = sorted(abs(x) for x in d)
    bitexact = sum(1 for x in d if x == 0.0)
    return {
        "n": n,
        "mean_signed": sum(d) / n,
        "mean_abs": sum(ad) / n,
        "rms": float(np.sqrt(np.mean(np.square(np.asarray(d, dtype=np.float64))))),
        "median_abs": ad[n // 2],
        "p95_abs": ad[int(n * 0.95)],
        "p99_abs": ad[int(n * 0.99)],
        "max_abs": ad[-1],
        "bitexact_count": bitexact,
        "bitexact_fraction": bitexact / n,
        "count_gt_1e-6": sum(1 for x in ad if x > 1e-6),
        "count_gt_1e-5": sum(1 for x in ad if x > 1e-5),
        "count_gt_1e-4": sum(1 for x in ad if x > 1e-4),
    }


# ---------------- estimator spread diagnostics (protocol s17) -----------------
def point_spread_ratio(points: list[float], ci_widths: list[float]) -> dict[str, float]:
    """numerical point spread / median bootstrap CI width — diagnostic only,
    no post-hoc hard threshold."""
    spread = max(points) - min(points)
    med_w = sorted(ci_widths)[len(ci_widths) // 2]
    return {"point_spread": spread, "median_ci_width": med_w, "ratio": spread / med_w if med_w > 0 else float("inf")}


# ---------------- ledger loading (completeness-gated) -------------------------
def load_replicate_ledger(ck: str, rep: str, n_pairs: int = N_PAIRS) -> dict[int, dict[str, list[float]]]:
    """rows[pair_index] = {arm: [4 strata values]} for one (checkpoint,
    replicate).  Same completeness gating as the reviewed scorer loader:
    both worker files present, no duplicates, all 8 arms, 4 strata each,
    exactly n_pairs unique indices."""
    out: dict[int, dict[str, list[float]]] = {}
    for w in (0, 1):
        p = ledger_path(ck, rep, w)
        if not p.is_file():
            raise RuntimeError(f"missing ledger {p}")
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                if row["ck"] != ck:
                    raise RuntimeError(f"{p}: row ck {row['ck']} != {ck}")
                i = int(row["pair_index"])
                if i in out:
                    raise RuntimeError(f"duplicate ledger row {ck}/{rep}/pair {i}")
                losses = row["losses"]
                if set(losses) != set(ARMS):
                    raise RuntimeError(f"{ck}/{rep}/pair {i} arms {sorted(losses)} != {sorted(ARMS)}")
                for a in ARMS:
                    if len(losses[a]) != 4:
                        raise RuntimeError(f"{ck}/{rep}/pair {i} arm {a} strata {len(losses[a])} != 4")
                for a in ARMS:
                    for v in losses[a]:
                        if not np.isfinite(v):
                            raise RuntimeError(f"{ck}/{rep}/pair {i} arm {a} nonfinite loss")
                out[i] = losses
    if len(out) != n_pairs:
        missing = [i for i in range(n_pairs) if i not in out]
        raise RuntimeError(f"incomplete {ck}/{rep} ledger: {len(missing)} missing, e.g. {missing[:5]}")
    return out


# ---------------- hash freeze (protocol s22) ---------------------------------
def freeze_manifest_doc() -> dict[str, Any]:
    """Shape of the raw-replicates freeze manifest (written by the run
    script after TREATMENT_B, verified by analyze_replicates before any
    scientific value is read)."""
    doc: dict[str, Any] = {"label": "MBS THREE-WAY NUMERICS — RAW REPLICATE HASH FREEZE"}
    for ck in CKS:
        doc[ck] = {}
        for rep in REPS:
            doc[ck][rep] = {
                "root": str(REP_ROOTS[ck][rep]),
                "files": {
                    f"w{w}": str(ledger_path(ck, rep, w)) for w in (0, 1)
                },
            }
    return doc


def verify_freeze(manifest: dict[str, Any]) -> None:
    """Refuse to analyze unless every frozen ledger still matches its
    recorded sha256 and row count."""
    import hashlib

    for ck in CKS:
        for rep in REPS:
            spec = manifest[ck][rep]
            for wk in (0, 1):
                key = f"w{wk}"
                path = Path(spec["files"][key])
                h = hashlib.sha256()
                rows_seen = 0
                with open(path, "rb") as fh:
                    for line in fh:
                        h.update(line)
                        rows_seen += 1
                if h.hexdigest() != spec["sha256"][key]:
                    raise RuntimeError(f"freeze mismatch: {path}")
                if rows_seen != spec["rows"][key]:
                    raise RuntimeError(f"freeze row-count mismatch: {path}")
    print("freeze verify: PASS (6 ledgers, 3072 rows total)")


__all__ = [
    "ADJUDICATION_AMBIGUOUS",
    "ADJUDICATION_PASS",
    "AMBIGUOUS_STABLE",
    "ARMS",
    "BOOT_SEED",
    "CKS",
    "CLASS_BORDERLINE",
    "CLASS_CORRECTION",
    "CLASS_HARM",
    "COHORTS",
    "COHORT_2TO4",
    "COHORT_ALL",
    "COHORT_BALANCED",
    "COHORT_GE4",
    "COHORT_LT2",
    "COHORT_TARGETED",
    "EXPECTED_COHORT_SIZES",
    "FREEZE_MANIFEST",
    "FROZEN_PAIRS_CSV",
    "MBS_A3",
    "MBS_C",
    "NUMERICS_ROOT",
    "N_BOOT",
    "N_PAIRS",
    "PRE_CANCELLATION_TOL",
    "REPS",
    "REP_ROOTS",
    "STABLE",
    "VBS",
    "VBS_ROOT",
    "adjudicate",
    "bootstrap_ci_shared",
    "classify_contrast",
    "classify_envelope",
    "ct_combinations",
    "envelope",
    "freeze_manifest_doc",
    "ledger_path",
    "load_replicate_ledger",
    "pair_contrast",
    "pair_margins",
    "point_spread_ratio",
    "pre_cancellation_max_abs",
    "replicate_diff_stats",
    "replicate_tuples",
    "shared_bootstrap_index_matrix",
    "verify_freeze",
]
