"""Vertical Bottom Supervision Audit - frozen contracts (pure numpy + stdlib).

Read-only, forward-only audit companion to the Camera Coordinate Causal
audit (final_snapshot/ + posthoc V2).  This module defines the PRE-REGISTERED
geometry reconstruction, pair-eligibility, selection, seed, arm, margin and
inference contracts for the SAME-SOURCE MIRRORED VIEWPORT PAIR audit.

Design rules (task spec sections 9-12, 19-21, 28-30, 42-44):
  * No torch import at module load: everything here is testable on CPU with
    numpy + stdlib only.
  * All thresholds and seeds are constants below; nothing is fitted or chosen
    after seeing results.
  * The historical offset-balance TOP/BOTTOM tertile split is reused exactly
    (posthoc_v2_contracts.tertile_of semantics: START < 1/3, CENTER [1/3, 2/3),
    END >= 2/3 on the normalized offset).
  * Production geometry reconstruction follows the V2 offset-balance verified
    formula (EXACT for all 2048 camera units at audit time):
        F = round(zoom**2 * 256);  available = F - 256;
        k = round(norm_offset * available);
        signed_shift = k + 128 - F/2   (== manifest pixel_shift, EXACT)
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

# ---------------- fixed audit constants ----------------

MASTER_SEED_VBS: int = 20260907
BOOT_SEED: int = 20260907
N_BOOT: int = 10_000

T_QUANTILES: tuple[float, float, float, float] = (0.10, 0.35, 0.65, 0.90)
P_MEAN: float = -0.8
P_STD: float = 0.8
NOISE_SCALE: float = 1.0
T_EPS: float = 0.05
NOISE_OBSERVATION_BOUNDARY: float = 0.95

VIEWPORT: int = 256
LATENT_TOKENS: int = 16  # 256 / (VAE 8x * patch 2x)
VAE_SCALE: int = 16       # latent_center_shift = abs(pixel_shift) / 16 (production)

N_PAIRS_TARGET: int = 512
N_PAIRS_MIN: int = 256
N_SMOKE_PAIRS: int = 32

# Pre-registered ID-arm timing gate (spec s23 "if the extra forward cost is
# reasonable"): include T_ID/B_ID only if the 32-pair smoke measures
# <= ID_ARMS_MAX_S_PER_FORWARD seconds per DiT forward (keeps the full 512-pair
# 8-arm 3-checkpoint run within ~8 h wall on 2 workers).
ID_ARMS_MAX_S_PER_FORWARD: float = 1.18

# Content classification thresholds (spec s42; pre-registered, no tuning).
SPEARMAN_EXPLAIN: float = 0.30     # |rho| >= 0.30  -> content explains (B)
SPEARMAN_NONE: float = 0.20        # |rho| <  0.20  -> content absent (C)
ATTENUATE_NONE: float = 0.25       # balanced-subset attenuation <= 25% -> C
ATTENUATE_ALL: float = 0.50        # attenuation >= 50% + rho -> B
# (25% < attenuation < 50% or mixed signals -> D)

# ---------------- paths (come3 audit host) ----------------

BASE_SHA = "34f646abdb1e64f45dfddd47cf7e8b9247a7a979"
BRANCH = "camera-v2-vertical-bottom-supervision-review"
WORKTREE = Path("/sakuramoon-runtime/sakuramoon-camera-vertical-bottom-review")
RUNTIME_ROOT = Path("/sakuramoon-runtime")
AUDIT_ROOT = Path("/tmp/camera-vertical-bottom")
EVIDENCE_ROOT = Path("/tmp/camera-coordinate-causal")
SHARD_ROOT = RUNTIME_ROOT / "data" / "validation-cohorts" / "s0-validation-50k-v1" / "shards"
STAGE1_MANIFEST = EVIDENCE_ROOT / "stage1-manifest.json"
MICROPROBE_V2 = EVIDENCE_ROOT / "posthoc-v2-microprobe.json"

CKPT_PATHS: dict[str, Path] = {
    "PRE": RUNTIME_ROOT / "output_model" / "g1" / "ckpt_116100_raw-116100-update-cadence",
    "MID": EVIDENCE_ROOT / "ckpts" / "MID" / "ckpt_117100_raw-117100-update-cadence",
    "POST": RUNTIME_ROOT / "output_model" / "g1_camera_v2_p25" / "ckpt_118100_raw-118100-update-cadence",
}
CKPT_UPDATES: dict[str, int] = {"PRE": 116100, "MID": 117100, "POST": 118100}
CKS: tuple[str, str, str] = ("PRE", "MID", "POST")
TRANSITIONS: tuple[tuple[str, str], ...] = (("PRE", "MID"), ("MID", "POST"), ("PRE", "POST"))

PE_WEIGHTS = RUNTIME_ROOT / "model" / "pe_spatial_b16_512" / "PE-Spatial-B16-512.pt"
PE_ASSET_JSON = RUNTIME_ROOT / "model" / "pe_spatial_b16_512" / "asset.json"
PE_APPROVED_SHA = "86217607f0bb28c0adb5ac3f9b0608ae22f6fb634bf1c16b2316847e8148a2a5"
PE_APPROVED_SIZE: int = 345_783_707
CLIP_DIR = RUNTIME_ROOT / "model" / "clip-vit-large-patch14-336"
QWEN_DIR = RUNTIME_ROOT / "model" / "qwen_3.5_2B"
VAE_DIR = RUNTIME_ROOT / "model" / "vae"

# ---------------- tertile / band contracts (spec s10, s12, s40-41) ----------------

def tertile_of(norm_offset: float) -> str:
    """START < 1/3, CENTER [1/3, 2/3), END >= 2/3 (identical to posthoc V2)."""
    q = float(norm_offset)
    if q < 1.0 / 3.0:
        return "START"
    if q < 2.0 / 3.0:
        return "CENTER"
    return "END"


def zoom_band(zoom: float) -> str:
    """mild < 1.20 <= medium < 1.35 <= strong (stage1 camera_zoom_band)."""
    z = float(zoom)
    if z < 1.20:
        return "mild"
    if z < 1.35:
        return "medium"
    return "strong"


def latent_shift_band(latent_shift: float) -> str:
    """<2, [2,4), >=4 (spec s41)."""
    v = float(latent_shift)
    if v < 2.0:
        return "lt2"
    if v < 4.0:
        return "2to4"
    return "ge4"


# ---------------- pair geometry (spec s10) ----------------

class PairGeometryError(RuntimeError):
    pass


def full_height_from_zoom(zoom: float) -> int:
    """F = round(zoom**2 * 256) - EXACT reconstruction (V2 offset balance)."""
    return round(float(zoom) ** 2 * float(VIEWPORT))


def signed_shift_of(k: int, full_height: int) -> float:
    """Production formula: (k + r//2) - full_height/2 with r = 256."""
    return float(k + VIEWPORT // 2) - float(full_height) / 2.0


def _tertile_int(k: int, available: int) -> str:
    """Exact integer tertile (no rounding): 3k vs available boundaries."""
    if 3 * k < available:
        return "START"
    if 3 * k < 2 * available:
        return "CENTER"
    return "END"


def pair_geometry(unit: dict[str, Any]) -> dict[str, Any]:
    """Reconstruct the full mirrored-pair geometry from one stage1 manifest row.

    The manifest stores ROUNDED zoom/norm_offset; the EXACT invariant (verified
    for all camera units in the V2 offset-balance round) is the production
    signed-shift formula: k = round(norm_offset*available) must satisfy
    signed_shift(k, F) == pixel_shift exactly.  Mirror landing is tested with
    exact integer arithmetic on k.  Raises PairGeometryError on any violation.
    """
    zoom = float(unit["zoom"])
    norm_offset = float(unit["norm_offset"])
    pixel_shift = float(unit["pixel_shift"])
    target = unit.get("target")
    if not (isinstance(target, (list, tuple)) and len(target) == 2
            and all(int(v) == VIEWPORT for v in target)):
        raise PairGeometryError(f"target {target!r} != [{VIEWPORT},{VIEWPORT}]")
    F = full_height_from_zoom(zoom)
    available = F - VIEWPORT
    if available <= 0:
        raise PairGeometryError(f"available {available} <= 0")
    k = round(norm_offset * available)
    if k < 0 or k > available:
        raise PairGeometryError(f"k {k} out of [0,{available}]")
    s = signed_shift_of(k, F)
    if s != pixel_shift:
        raise PairGeometryError(f"signed_shift {s} != manifest pixel_shift {pixel_shift}")
    # cross-check against the manifest latent_shift (abs/16), rounded-stored
    latent_manifest = float(unit.get("latent_shift", abs(s) / VAE_SCALE))
    if abs(latent_manifest - abs(s) / VAE_SCALE) > 1e-6:
        raise PairGeometryError(f"latent_shift {latent_manifest} != {abs(s)/VAE_SCALE}")
    k_mirror = available - k
    s_anchor = s
    s_mirror = signed_shift_of(k_mirror, F)
    if s_mirror != -s_anchor:
        raise PairGeometryError("mirror shift not exactly antisymmetric")
    # anchor side: same tertile semantics as the V2 historical split (stored
    # norm_offset); mirror landing: exact integer test on k_mirror
    anchor_side = tertile_of(norm_offset)
    mirror_side = _tertile_int(k_mirror, available)
    k_start, k_end = sorted((k, k_mirror))
    if k_start + k_end != available:
        raise PairGeometryError("k_start + k_end != available")
    return {
        "full_height": F,
        "available": available,
        "k_anchor": k,
        "k_mirror": k_mirror,
        "k_start": k_start,
        "k_end": k_end,
        "signed_shift_anchor": s_anchor,
        "signed_shift_mirror": s_mirror,
        "anchor_tertile": anchor_side,
        "mirror_tertile": mirror_side,
        "zoom": zoom,
        "norm_offset": norm_offset,
        "norm_start": k_start / available,
        "norm_end": k_end / available,
        "crop_box_top": (0, k_start, VIEWPORT, k_start + VIEWPORT),
        "crop_box_bottom": (0, k_end, VIEWPORT, k_end + VIEWPORT),
        "latent_shift_top": k_start + VIEWPORT // 2 - F / 2.0,
        "latent_shift_bottom": k_end + VIEWPORT // 2 - F / 2.0,
    }


def pair_eligible(unit: dict[str, Any]) -> tuple[bool, str]:
    """Spec s10 eligibility. Returns (ok, reason); ok=False reasons are exact."""
    if unit.get("cohort") != "camera":
        return False, "not_camera"
    if unit.get("orientation") != "vertical":
        return False, "not_vertical"
    if bool(unit.get("opp_na", False)):
        return False, "opposite_not_applicable"
    try:
        g = pair_geometry(unit)
    except PairGeometryError as e:
        return False, f"geometry:{e}"
    if g["available"] <= 0:
        return False, "available_zero"
    if g["anchor_tertile"] == "CENTER":
        return False, "center_anchor"
    # mirror must land on the opposite side; an END anchor at exactly 2/3
    # mirrors to CENTER and is excluded (START mirrors always land in END)
    if g["anchor_tertile"] == "END" and g["mirror_tertile"] != "START":
        return False, "mirror_not_start"
    if g["anchor_tertile"] == "START" and g["mirror_tertile"] != "END":
        return False, "mirror_not_end"
    return True, "ok"


# ---------------- pair selection (spec s11-12) ----------------

def dedup_key(unit: dict[str, Any]) -> tuple[str, int]:
    return (str(unit["source_shard"]), int(unit["sample_id"]))


def select_unique_pairs(units: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Dedup by (source_shard, sample_id), keep smallest unit id (spec s11).

    Returns (unique units, dedup_count_removed).
    """
    best: dict[tuple[str, int], dict[str, Any]] = {}
    for u in units:
        key = dedup_key(u)
        cur = best.get(key)
        if cur is None or int(u["unit"]) < int(cur["unit"]):
            best[key] = u
    removed = len(units) - len(best)
    return list(best.values()), removed


def stratified_selection(
    candidates: Sequence[dict[str, Any]],
    target: int = N_PAIRS_TARGET,
    minimum: int = N_PAIRS_MIN,
    seed: int = MASTER_SEED_VBS,
) -> dict[str, Any]:
    """Deterministic stratified selection (spec s12).

    Strata = (anchor_side START|END) x (zoom band mild|medium|strong).
    Allocation: proportional to stratum pool size (largest remainder), then
    within each stratum take the first n by deterministic sample-key order
    (source_shard, sample_id, unit).  If the pool < target use the whole pool;
    if the pool < minimum the caller must STOP.  No redraw after seeing
    results: the seed is recorded for auditability but the ordering is
    sample-key deterministic (seed-independent by construction).
    """
    pool = list(candidates)
    if len(pool) < minimum:
        return {
            "status": "STOP_BELOW_MINIMUM",
            "n_pool": len(pool),
            "minimum": minimum,
            "selected": [],
            "allocation": {},
            "seed": seed,
        }
    n_take = min(target, len(pool))
    strata: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for u in pool:
        g = pair_geometry(u)
        side = "START" if g["anchor_tertile"] == "START" else "END"
        band = zoom_band(g["zoom"])
        strata.setdefault((side, band), []).append(u)
    for items in strata.values():
        items.sort(key=lambda u: (str(u["source_shard"]), int(u["sample_id"]), int(u["unit"])))
    # proportional largest-remainder allocation
    sizes = {s: len(v) for s, v in strata.items()}
    total = sum(sizes.values())
    raw = {s: n_take * sz / total for s, sz in sizes.items()}
    alloc = {s: math.floor(raw[s]) for s in sizes}
    leftover = n_take - sum(alloc.values())
    # distribute leftover by largest fractional remainder, ties by stratum key
    order = sorted(raw, key=lambda s: (-(raw[s] - math.floor(raw[s])), s))
    i = 0
    while leftover > 0:
        s = order[i % len(order)]
        if alloc[s] < sizes[s]:
            alloc[s] += 1
            leftover -= 1
        i += 1
    selected: list[dict[str, Any]] = []
    allocation: dict[str, int] = {}
    for s in sorted(strata):
        take = alloc[s]
        selected.extend(strata[s][:take])
        allocation["|".join(s)] = take
    selected.sort(key=lambda u: (str(u["source_shard"]), int(u["sample_id"]), int(u["unit"])))
    return {
        "status": "OK",
        "n_pool": len(pool),
        "n_selected": len(selected),
        "target": target,
        "minimum": minimum,
        "seed": seed,
        "allocation": allocation,
        "stratum_pool_sizes": {"|".join(k): v for k, v in sorted(sizes.items(), key=lambda kv: str(kv[0]))},
        "selected": selected,
    }


# ---------------- noise / timestep contract (spec s19) ----------------

# ---------------- JLT timestep strata (production-locked) ----------------

# Acklam's rational approximation of the inverse normal CDF (same approach as
# final_snapshot/cc_common.py; stage1 verified the quantiles against 2M draws
# from the production sampler with tolerance 0.005).
_A = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
      1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
_B = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
      6.680131188771972e+01, -1.328068155288572e+01)
_C = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
      -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
_D = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
      3.754408661907416e+00)
_P_LOW = 0.02425
_P_HIGH = 1.0 - 0.02425


def normal_cdf_inv(p: float) -> float:
    """Acklam rational approximation - verbatim copy of cc_common.normal_cdf_inv."""
    if not (0.0 < p < 1.0):
        raise ValueError("p must be in (0,1)")
    if p < _P_LOW:
        q = math.sqrt(-2.0 * math.log(p))
        num = (((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5])
        den = (((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1.0
        return num / den
    if p > 1.0 - _P_LOW:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        num = (((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5])
        den = (((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1.0
        return -num / den
    q = p - 0.5
    r = q * q
    num = (((((_A[0] * r + _A[1]) * r + _A[2]) * r + _A[3]) * r + _A[4]) * r + _A[5])
    den = (((((_B[0] * r + _B[1]) * r + _B[2]) * r + _B[3]) * r + _B[4]) * r + 1.0)
    return num * q / den


def jlt_quantile(q: float, p_mean: float = P_MEAN, p_std: float = P_STD) -> float:
    """t-quantile of the production JLT sampler: sigmoid(N(p_mean, p_std)).

    Bit-identical expression to final_snapshot/cc_common.jlt_quantile (the
    frozen audit reference): a = p_mean + p_std * z; 1/(1+exp(-a)).
    """
    z = normal_cdf_inv(q)
    a = p_mean + p_std * z
    return 1.0 / (1.0 + math.exp(-a))


def t_values() -> tuple[float, float, float, float]:
    """Frozen JLT quantile timesteps (production P_MEAN/P_STD locked floats)."""
    return tuple(jlt_quantile(q) for q in T_QUANTILES)


def noise_seed(pair_index: int, stratum: int) -> int:
    """Pair-shared noise seed: MASTER_VBS*1_000_003 + pair_index*100 + stratum.

    Mirrors the eps_seed form of final_snapshot/cc_common.py (MASTER_SEED
    20260906) with the VBS master 20260907.  One eps tensor per (pair,
    stratum) is drawn once and shared by the TOP and BOTTOM arms.
    """
    return MASTER_SEED_VBS * 1_000_003 + int(pair_index) * 100 + int(stratum)


# ---------------- arm / margin contract (spec s21, s29) ----------------

ARMS: tuple[str, ...] = ("TT", "TB", "BT", "BB", "T_SAME", "B_SAME", "T_ID", "B_ID")
ARMS_NO_ID: tuple[str, ...] = ("TT", "TB", "BT", "BB", "T_SAME", "B_SAME")

def arm_side(arm: str) -> str:
    if arm in ("TT", "TB", "T_SAME", "T_ID"):
        return "top"
    if arm in ("BT", "BB", "B_SAME", "B_ID"):
        return "bottom"
    raise ValueError(f"unknown arm {arm!r}")

def arm_coord(arm: str) -> str:
    """Which coordinate map the arm receives: top_correct | bottom_correct | id."""
    if arm in ("TT", "T_SAME", "T_ID"):
        return "top_correct" if arm != "T_ID" else "id"
    if arm in ("TB",):
        return "bottom_correct"
    if arm in ("BT",):
        return "top_correct"
    if arm in ("BB", "B_SAME", "B_ID"):
        return "bottom_correct" if arm != "B_ID" else "id"
    raise ValueError(f"unknown arm {arm!r}")

def margin_from_losses(losses: dict[str, float]) -> dict[str, float]:
    """m_top = L_TB - L_TT ; m_bot = L_BT - L_BB (spec s29)."""
    return {
        "m_top": losses["TB"] - losses["TT"],
        "m_bot": losses["BT"] - losses["BB"],
    }

def floor_from_losses(losses: dict[str, float]) -> dict[str, float]:
    """Numerics floor from SAME duplicates: f_top = L_T_SAME - L_TT etc."""
    return {
        "f_top": losses["T_SAME"] - losses["TT"],
        "f_bot": losses["B_SAME"] - losses["BB"],
    }

def adjusted_delta(m: dict[str, float], f: dict[str, float], a: str, b: str) -> dict[str, float]:
    """D_top[a->b] = (M_b - M_a) - (F_b - F_a) and same for bottom.

    m/f are 4-stratum mean margins/floors per checkpoint (offset-balance
    convention: M = mean over strata).
    """
    return {
        "d_top": (m[b]["m_top"] - m[a]["m_top"]) - (f[b]["f_top"] - f[a]["f_top"]),
        "d_bot": (m[b]["m_bot"] - m[a]["m_bot"]) - (f[b]["f_bot"] - f[a]["f_bot"]),
    }

def paired_gap(d: dict[str, float]) -> float:
    """G = D_top - D_bottom (inference unit = source pair)."""
    return d["d_top"] - d["d_bot"]


# ---------------- bootstrap inference (spec s29) ----------------

def bootstrap_ci(values: Sequence[float], n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> list[float]:
    """Paired bootstrap over source pairs: percentile CI of the mean.

    point estimate = EXACT mean of the observed pair values (float64);
    the bootstrap is used for the 95% CI only.  Identical mechanics to the
    posthoc V2 probe _block_mean_ci (rng.integers(0,n,(n_boot,n))).
    """
    a = np.asarray(values, dtype=np.float64)
    if a.size == 0:
        raise ValueError("empty value set")
    n = a.size
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = a[idx].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def ci_excl0(ci: Sequence[float]) -> bool:
    lo, hi = float(ci[0]), float(ci[1])
    return lo > 0.0 or hi < 0.0


def paired_mirror_difference(
    top_values: Sequence[float],
    bottom_values: Sequence[float],
    n_boot: int = N_BOOT,
    seed: int = BOOT_SEED,
) -> dict[str, Any]:
    """Correct mirror estimator (spec s42): point = mean(top - bottom) over
    PAIRED values; bootstrap resamples shared pair indices (TOP and BOTTOM of
    one pair always appear together in a replicate).

    Regression reference (the OLD pooled-sign formula that this replaces):
    see pooled_two_sample_difference.  On the spec fixture
    top=[1,2,3], bottom=[0,0,0] this returns point == 2 (never 1).
    """
    t = np.asarray(top_values, dtype=np.float64)
    b = np.asarray(bottom_values, dtype=np.float64)
    if t.shape != b.shape or t.size == 0:
        raise ValueError("paired values must be equal-length and nonempty")
    diffs = t - b
    n = t.size
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = diffs[idx].mean(axis=1)
    return {
        "point": float(diffs.mean()),
        "ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
        "n": int(n),
        "n_boot": int(n_boot),
        "seed": int(seed),
    }


def pooled_two_sample_difference(
    low_values: Sequence[float],
    high_values: Sequence[float],
) -> float:
    """Unequal-count two-sample difference (spec s42, helper contract):
    mean(low) - mean(high), i.e. the difference of the two group means.
    This is the correct form when the two arms have UNEQUAL counts (as in the
    historical natural cohort); it must NOT be replaced by a pooled-sign mean.
    """
    t = np.asarray(low_values, dtype=np.float64)
    b = np.asarray(high_values, dtype=np.float64)
    if t.size == 0 or b.size == 0:
        raise ValueError("empty pool")
    return float(t.mean() - b.mean())


def pooled_sign_mean(
    top_values: Sequence[float],
    bottom_values: Sequence[float],
) -> float:
    """OLD (incorrect) pooled-sign formula, retained ONLY as a regression
    reference: a single mean over the concatenated signed pool.  On the spec
    fixture top=[1,2,3], bottom=[0,0,0] it returns 1.0, while the correct
    paired mirror point is 2.0 (spec s42: point must be 2, never 1).
    """
    t = np.asarray(top_values, dtype=np.float64)
    b = np.asarray(bottom_values, dtype=np.float64)
    if t.size == 0 or b.size == 0:
        raise ValueError("empty pool")
    return float(np.concatenate([t, b]).mean())


def ci_sign(ci: Sequence[float]) -> int:
    lo, hi = float(ci[0]), float(ci[1])
    if lo > 0.0:
        return 1
    if hi < 0.0:
        return -1
    return 0


def point_ci(values: Sequence[float]) -> dict[str, Any]:
    a = np.asarray(values, dtype=np.float64)
    return {
        "n": int(a.size),
        "point": float(a.mean()),
        "ci95": bootstrap_ci(a),
    }


# ---------------- Spearman / quartiles (spec s39) ----------------

def _rankdata(a: np.ndarray) -> np.ndarray:
    order = a.argsort(kind="mergesort")
    ranks = np.empty(a.size, dtype=np.float64)
    ranks[order] = np.arange(1, a.size + 1)
    # average ranks for ties
    sorted_a = a[order]
    i = 0
    while i < a.size:
        j = i
        while j + 1 < a.size and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        if j > i:
            avg = (i + j + 2) / 2.0  # ranks i+1..j+1 average
            ranks[order[i:j + 1]] = avg
        i = j + 1
    return ranks


def spearman_rho(x: Sequence[float], y: Sequence[float]) -> float:
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    if xa.size != ya.size or xa.size < 3:
        raise ValueError("need equal lengths >= 3")
    rx = _rankdata(xa)
    ry = _rankdata(ya)
    if rx.std() == 0.0 or ry.std() == 0.0:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def quartile_bins(values: Sequence[float]) -> list[int]:
    """0..3 quartile assignment (quantile cuts at 25/50/75)."""
    a = np.asarray(values, dtype=np.float64)
    cuts = np.quantile(a, [0.25, 0.5, 0.75])
    return [int(v) for v in np.digitize(a, cuts)]


# ---------------- classification (spec s42-44) ----------------

VERDICTS: tuple[str, ...] = (
    "NATURAL_COHORT_CONFOUNDING",
    "CONTENT_CROP_ASYMMETRY_SUPPORTED",
    "DIRECTIONAL_MECHANISM_ASYMMETRY_CONFIRMED",
    "MIXED_CONTENT_AND_DIRECTIONAL_ASYMMETRY",
    "NO_PAIRED_ASYMMETRY",
    "INCONCLUSIVE",
)
RECOMMENDATIONS: tuple[str, ...] = (
    "REVIEW_EVALUATION_COHORT",
    "REVIEW_CROP_CONTENT_SUPERVISION",
    "REVIEW_VERTICAL_COORDINATE_SUPERVISION",
    "REVIEW_CROP_AND_VERTICAL_SUPERVISION",
    "LONGER_P25_REVIEW_CANDIDATE",
    "NO_MORE_EXPOSURE_YET",
)

# spec s46 (training recommendation mapping; recommendation only, never
# authorization: LONGER_P25_AUTHORIZED = NO, P50_AUTHORIZED = NO)
_VERDICT_RECOMMENDATION: dict[str, str] = {
    "NATURAL_COHORT_CONFOUNDING": "REVIEW_EVALUATION_COHORT",
    "CONTENT_CROP_ASYMMETRY_SUPPORTED": "REVIEW_CROP_CONTENT_SUPERVISION",
    "DIRECTIONAL_MECHANISM_ASYMMETRY_CONFIRMED": "REVIEW_VERTICAL_COORDINATE_SUPERVISION",
    "MIXED_CONTENT_AND_DIRECTIONAL_ASYMMETRY": "REVIEW_CROP_AND_VERTICAL_SUPERVISION",
    "NO_PAIRED_ASYMMETRY": "LONGER_P25_REVIEW_CANDIDATE",
    "INCONCLUSIVE": "NO_MORE_EXPOSURE_YET",
}

def classify(
    *,
    n_pairs: int,
    gap_all: dict[str, dict[str, Any]],
    gap_balanced: dict[str, dict[str, Any]] | None,
    historical_gap: dict[str, Any] | None,
    spearman: float | None,
    systematic_bottom_content_loss: bool = False,
    numerics_floor_ci_ok: bool = True,
) -> tuple[str, dict[str, Any]]:
    """A-F classification (spec s45, pre-registered decision tree).

    Inputs (observed statistics only, nothing fitted after seeing results):
      gap_* : transition key "PRE|POST" -> {point, ci95} for G = D_TOP - D_BOTTOM
      historical_gap: {point, ci95} natural-cohort matched TOP-BOTTOM gap
      spearman: pair-level Spearman(content asymmetry, G_PRE->POST) or None
      systematic_bottom_content_loss: True when a pre-registered content delta
        (TOP-BOTTOM; positive = BOTTOM preserves less) has a bootstrap CI
        excluding 0 on the positive side
    """
    notes: list[str] = []
    pre = gap_all.get("PRE|POST")
    if n_pairs < N_PAIRS_MIN or pre is None:
        return "INCONCLUSIVE", {"case": "F", "notes": ["insufficient pairs or missing PRE->POST"]}
    if not numerics_floor_ci_ok:
        notes.append("numerics floor CI overlaps the signal magnitude; interpret with caution")
    gap_excl0 = ci_excl0(pre["ci95"])
    hist_excl0 = historical_gap is not None and ci_excl0(historical_gap["ci95"])
    if gap_excl0 and historical_gap is not None and ci_sign(pre["ci95"]) != ci_sign(historical_gap["ci95"]):
        notes.append("same-source gap sign flips vs historical natural-cohort gap")
    # A (s45): same-source gap collapses / CI includes 0, natural cohort gap clear
    if not gap_excl0 and hist_excl0:
        notes.append("same-source paired gap CI contains 0 while natural-cohort gap excluded 0")
        return "NATURAL_COHORT_CONFOUNDING", {"case": "A", "notes": notes}
    # E (s45): no same-source gap and the natural result does not reproduce
    if not gap_excl0:
        return "NO_PAIRED_ASYMMETRY", {"case": "E", "notes": notes}
    # gap persists (CI excludes 0) -> content vs directional decomposition
    if gap_balanced is None:
        notes.append("content-balanced subset unavailable; directional/content split inconclusive")
        return "INCONCLUSIVE", {"case": "F", "notes": notes}
    pre_bal = gap_balanced.get("PRE|POST")
    if pre_bal is None:
        notes.append("balanced-subset PRE->POST gap missing")
        return "INCONCLUSIVE", {"case": "F", "notes": notes}
    balanced_persists = ci_excl0(pre_bal["ci95"])
    attenuation = 1.0 - abs(pre_bal["point"]) / max(abs(pre["point"]), 1e-30)
    rho = spearman
    rho_abs = abs(rho) if rho is not None else None
    content_diff_present = (
        (rho_abs is not None and rho_abs >= SPEARMAN_NONE)
        or systematic_bottom_content_loss
    )
    # B (s45): gap persists AND (balanced gap collapses/shrinks OR consistent
    # content relationship OR systematic BOTTOM content loss)
    if not balanced_persists or attenuation >= ATTENUATE_ALL:
        notes.append(f"balanced-subset gap collapsed (persists={balanced_persists}, attenuation={attenuation:.3f})")
        return "CONTENT_CROP_ASYMMETRY_SUPPORTED", {
            "case": "B", "notes": notes, "attenuation": attenuation, "spearman": rho,
            "content_diff_present": content_diff_present,
        }
    # balanced gap still clearly > 0
    if content_diff_present:
        return "MIXED_CONTENT_AND_DIRECTIONAL_ASYMMETRY", {
            "case": "D", "notes": notes, "attenuation": attenuation, "spearman": rho,
            "content_diff_present": True,
        }
    return "DIRECTIONAL_MECHANISM_ASYMMETRY_CONFIRMED", {
        "case": "C", "notes": notes, "attenuation": attenuation, "spearman": rho,
        "content_diff_present": False,
    }


def recommendation_for(verdict: str) -> str:
    return _VERDICT_RECOMMENDATION[verdict]


# ---------------- content-balanced subset (spec s37) ----------------

def content_balanced_subset(
    pairs: Sequence[dict[str, Any]],
    asym: Sequence[float],
    fraction: float = 0.5,
) -> list[int]:
    """Pre-registered lowest-asymmetry fraction of pairs.

    Takes ONLY pair rows and a per-pair content-asymmetry value computed from
    content fields exclusively (no causal field may flow in).  Deterministic:
    sort by (asym, pair_index); take floor(n*fraction) pairs.
    """
    if len(pairs) != len(asym):
        raise ValueError("pairs/asym length mismatch")
    n_take = math.floor(len(pairs) * fraction)
    order = sorted(range(len(pairs)), key=lambda i: (float(asym[i]), i))
    return [int(i) for i in order[:n_take]]


# ---------------- prerequisite gate (spec s27) ----------------

MICROPROBE_V2_SHA = "ab5bc0634631f38388e8c5b5c4d6fd216ac5254afcd171f04ef05282d77e30f9"
V2_REVIEW_REPORT = WORKTREE / "reports" / "camera-coordinate-causal-posthoc-v2-review.json"


def validate_prior_numerics_v2(
    microprobe_path: Path,
    v2_report_path: Path,
    *,
    expected_microprobe_sha: str = MICROPROBE_V2_SHA,
) -> dict[str, Any]:
    """spec s43/s27: the prior-V2 numerics prerequisite gate.

    Reads (never re-runs) the frozen posthoc-v2-microprobe.json and the V2
    review report.  P3 (state-history probe) is wired INTO this gate:
    p3.state_history_effect == true  =>  status BLOCKED_NUMERICS.
    Returns a dict; caller must STOP on any status other than "PASS".
    """
    out: dict[str, Any] = {"status": "PASS", "checks": {}}
    if not microprobe_path.is_file():
        out["status"] = "STOP_MICROPROBE_MISSING"
        return out
    actual_sha = sha256_file(microprobe_path)
    out["microprobe_sha256"] = actual_sha
    if expected_microprobe_sha and actual_sha != expected_microprobe_sha:
        out["status"] = "STOP_MICROPROBE_SHA_MISMATCH"
        return out
    with open(microprobe_path, "r", encoding="utf-8") as fh:
        mp = json.load(fh)
    gate = microprobe_gate(mp)
    out["checks"]["microprobe"] = gate
    if gate["hidden_mutable_state_detected"] is not False:
        out["status"] = "BLOCKED_NUMERICS"
        return out
    if not (gate["p3_present"] and gate["p3_state_history_effect"] is False):
        # P3 must be present AND false; a true P3 blocks, a missing P3 stops
        out["status"] = "BLOCKED_NUMERICS" if gate["p3_state_history_effect"] is True else "STOP_P3_MISSING"
        return out
    if not v2_report_path.is_file():
        out["status"] = "STOP_V2_REPORT_MISSING"
        return out
    with open(v2_report_path, "r", encoding="utf-8") as fh:
        rep = json.load(fh)
    v2 = v2_report_gate(rep)
    out["checks"]["v2_report"] = v2
    if not v2["pass"]:
        out["status"] = "BLOCKED_NUMERICS"
    return out


def microprobe_gate(doc: dict[str, Any]) -> dict[str, Any]:
    """STATE_HISTORY_GATE + numerics-clean gate over the V2 microprobe json.

    REQUIRE (spec s27):
      state.HIDDEN_MUTABLE_STATE_DETECTED == false
      p3.state_history_effect == false   (and p3 present - the field must be
      wired into the gate, i.e. read here, not merely recorded)
      repeat jitter is diagnostic only (never a hidden-state implication)
    """
    out: dict[str, Any] = {
        "hidden_mutable_state_detected": None,
        "p3_state_history_effect": None,
        "p3_present": False,
        "repeat_jitter_present": None,
        "pass": False,
    }
    state = doc.get("state")
    if isinstance(state, dict):
        hms = state.get("HIDDEN_MUTABLE_STATE_DETECTED")
        out["hidden_mutable_state_detected"] = hms
        out["repeat_jitter_present"] = state.get("REPEAT_NUMERIC_JITTER_PRESENT")
    p3 = doc.get("p3")
    if isinstance(p3, dict):
        out["p3_present"] = True
        out["p3_state_history_effect"] = p3.get("state_history_effect")
    out["pass"] = (
        out["hidden_mutable_state_detected"] is False
        and out["p3_present"]
        and out["p3_state_history_effect"] is False
    )
    return out


def v2_report_gate(doc: dict[str, Any]) -> dict[str, Any]:
    """NUMERICS_V2_CLEAN == true and verdict == POSITIVE_LOSS_PREFERENCE_LEARNING.

    Committed V2 review report schema: the clean flag lives under
    ``doc["numerics"]["NUMERICS_V2_CLEAN"]`` and the verdict under
    ``doc["posthoc_v2_verdict"]``; the flat legacy keys are also accepted.
    """
    num = doc.get("numerics")
    clean = num.get("NUMERICS_V2_CLEAN") if isinstance(num, dict) else doc.get("NUMERICS_V2_CLEAN")
    verdict = doc.get("posthoc_v2_verdict", doc.get("POSTHOC_V2_VERDICT"))
    return {
        "NUMERICS_V2_CLEAN": clean,
        "POSTHOC_V2_VERDICT": verdict,
        "pass": clean is True and verdict == "POSITIVE_LOSS_PREFERENCE_LEARNING",
    }


# ---------------- hashing helpers ----------------

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_frozen(path: Path, doc: dict[str, Any]) -> str:
    """Write JSON deterministically, return its sha256 (freeze = hash-locked)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(doc, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.write_text(text, encoding="utf-8")
    return sha256_file(path)


__all__ = [
    "ARMS",
    "ARMS_NO_ID",
    "ATTENUATE_ALL",
    "ATTENUATE_NONE",
    "AUDIT_ROOT",
    "BASE_SHA",
    "BOOT_SEED",
    "BRANCH",
    "CKPT_PATHS",
    "CKPT_UPDATES",
    "CKS",
    "CLIP_DIR",
    "EVIDENCE_ROOT",
    "ID_ARMS_MAX_S_PER_FORWARD",
    "LATENT_TOKENS",
    "MASTER_SEED_VBS",
    "MICROPROBE_V2",
    "MICROPROBE_V2_SHA",
    "NOISE_OBSERVATION_BOUNDARY",
    "NOISE_SCALE",
    "N_BOOT",
    "N_PAIRS_MIN",
    "N_PAIRS_TARGET",
    "N_SMOKE_PAIRS",
    "PE_APPROVED_SHA",
    "PE_APPROVED_SIZE",
    "PE_ASSET_JSON",
    "PE_WEIGHTS",
    "P_MEAN",
    "P_STD",
    "QWEN_DIR",
    "RECOMMENDATIONS",
    "RUNTIME_ROOT",
    "SHARD_ROOT",
    "SPEARMAN_EXPLAIN",
    "SPEARMAN_NONE",
    "STAGE1_MANIFEST",
    "TRANSITIONS",
    "T_EPS",
    "T_QUANTILES",
    "V2_REVIEW_REPORT",
    "VAE_DIR",
    "VAE_SCALE",
    "VERDICTS",
    "VIEWPORT",
    "WORKTREE",
    "PairGeometryError",
    "_tertile_int",
    "adjusted_delta",
    "arm_coord",
    "arm_side",
    "bootstrap_ci",
    "ci_excl0",
    "ci_sign",
    "classify",
    "content_balanced_subset",
    "dedup_key",
    "floor_from_losses",
    "full_height_from_zoom",
    "jlt_quantile",
    "latent_shift_band",
    "margin_from_losses",
    "microprobe_gate",
    "noise_seed",
    "normal_cdf_inv",
    "pair_eligible",
    "pair_geometry",
    "paired_gap",
    "paired_mirror_difference",
    "point_ci",
    "pooled_sign_mean",
    "pooled_two_sample_difference",
    "quartile_bins",
    "recommendation_for",
    "select_unique_pairs",
    "sha256_file",
    "signed_shift_of",
    "spearman_rho",
    "stratified_selection",
    "t_values",
    "tertile_of",
    "v2_report_gate",
    "validate_prior_numerics_v2",
    "write_frozen",
    "zoom_band",
]
