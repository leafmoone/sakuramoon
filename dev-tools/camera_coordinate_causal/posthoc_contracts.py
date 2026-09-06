"""Posthoc statistical contracts for the camera coordinate causal audit review.

Pure functions (stdlib + numpy only). No I/O, no torch, no final_snapshot
imports: posthoc_review.py calls these, and the regression tests exercise
these directly without the 34 GB evidence store.

Semantics locked here (see test_camera_coordinate_causal_posthoc.py):
- SAME margin sign: N_same = Loss(SAME) - Loss(CORRECT); SAME is an exact
  duplicate of CORRECT, so a zero-mean N_same is the numerical-floor signal.
- Cluster bootstrap: units are the clusters; ONE shared resample matrix is
  applied to every checkpoint/arm/quantity that is paired (never two
  independent bootstrap summaries subtracted).
- Common-finite subspace: a unit enters an aggregate only if every required
  arm is finite at every required checkpoint; the subset size n is explicit.
- maxabs is an OUTLIER diagnostic, never a mean-effect validity gate.
"""
from __future__ import annotations

import numpy as np

MASTER_SEED = 20260906
N_BOOT = 10000
CKS = ("PRE", "MID", "POST")
PRIMARY_ARMS = ("OPPOSITE", "SHUFFLED")
SUPPORTIVE_ARM = "IDENTITY"

# Physical-axis convention for normalized_offset, verified from production
# source at base commit b2443af (src/sakuramoon/data/camera_viewport.py
# lines 425-447: horizontal -> left = randrange(available+1),
# normalized_offset = left/available, so 0 = LEFT edge, 1 = RIGHT edge;
# vertical -> top/available, so 0 = TOP edge, 1 = BOTTOM edge; orientation
# definition in src/sakuramoon/conditioning/camera.py lines 70-75: horizontal
# <=> full_height == viewport).
PHYSICAL_CONVENTION_VERIFIED = True


def _ci95(samples: np.ndarray) -> tuple[float, float]:
    return (float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5)))


def common_finite_subset(
    values: dict[str, dict[str, np.ndarray]],
    arms: tuple[str, ...],
    cks: tuple[str, ...] = CKS,
) -> np.ndarray:
    """Indices of units whose required arms are finite at every checkpoint.

    values[arm][ck] must be 1-D arrays of identical length (units).
    Returns a 1-D int64 index array (sorted, no duplicates).
    """
    n = None
    for arm in arms:
        for ck in cks:
            a = np.asarray(values[arm][ck], dtype=np.float64)
            if n is None:
                n = a.shape[0]
            elif a.shape[0] != n:
                raise ValueError(f"arm {arm} ck {ck}: length {a.shape[0]} != {n}")
    valid = np.ones(n, dtype=bool)
    for arm in arms:
        for ck in cks:
            valid &= np.isfinite(np.asarray(values[arm][ck], dtype=np.float64))
    return np.flatnonzero(valid)


def paired_bootstrap_indices(
    k: int, n_boot: int = N_BOOT, seed: int = MASTER_SEED, rng: np.random.Generator | None = None,
) -> np.ndarray:
    """ONE shared (n_boot, k) resample-with-replacement index matrix.

    Resampling is within the selected subspace (indices 0..k-1), mirroring
    the final stage3 v5 semantics. The caller applies the SAME matrix to all
    paired quantities (checkpoints, arms, SAME and causal).
    """
    if k < 2:
        raise ValueError(f"k={k} < 2: cannot bootstrap a single unit")
    if rng is None:
        rng = np.random.default_rng(seed)
    return rng.integers(0, k, size=(n_boot, k))


def cluster_mean_ci(values: np.ndarray, boot_idx: np.ndarray) -> tuple[float, tuple[float, float]]:
    """Mean + 95% CI of a per-unit vector under a shared resample matrix.

    values: (n_units,) ; boot_idx: (n_boot, n_units) with columns in 0..n_units-1
    (the shared subspace coordinates). Returns (mean, (lo, hi)).
    """
    if values.shape != boot_idx.shape[1]:
        raise ValueError("values length must equal the subspace size of boot_idx")
    means = values[boot_idx].mean(axis=1)
    return float(values.mean()), _ci95(means)


def same_baseline_stats(
    per_unit_stratum: dict[str, np.ndarray],
    boot_idx: np.ndarray | None = None, n_boot: int = N_BOOT, seed: int = MASTER_SEED,
) -> dict:
    """SAME-baseline numerical floor from per-unit, per-stratum N_same.

    per_unit_stratum[ck] has shape (n_units, n_strata). The per-checkpoint
    statistic pools all (unit, stratum) pairs but bootstraps at UNIT level
    (one shared matrix across checkpoints). Deltas are paired per unit
    (mean over strata, then unit-cluster bootstrap with the same matrix).

    Returns per ck: mean, ci95, abs_p50/p90/p95/p99, maxabs, n_units; and
    per interval (POST_PRE/MID_PRE/POST_MID): point, ci95, n.
    """
    n_units = None
    for ck in CKS:
        a = np.asarray(per_unit_stratum[ck], dtype=np.float64)
        if a.ndim != 2:
            raise ValueError(f"per_unit_stratum[{ck}] must be 2-D (units, strata)")
        if n_units is None:
            n_units = a.shape[0]
        elif a.shape[0] != n_units:
            raise ValueError(f"per_unit_stratum[{ck}] units mismatch")
    if boot_idx is None:
        boot_idx = paired_bootstrap_indices(n_units, n_boot, seed)
    out: dict = {"checkpoints": {}, "deltas": {}, "n_units": n_units, "n_boot": N_BOOT if boot_idx is None else int(boot_idx.shape[0]), "seed": seed}
    unit_mean = {}
    for ck in CKS:
        a = np.asarray(per_unit_stratum[ck], dtype=np.float64)
        pooled = a.ravel()
        abs_vals = np.abs(pooled)
        means = a[boot_idx].mean(axis=(1, 2))  # per replicate: mean over (resampled units x strata)
        unit_mean[ck] = a.mean(axis=1)
        out["checkpoints"][ck] = {
            "mean": float(a.mean()),
            "ci95": _ci95(means),
            "abs_p50": float(np.percentile(abs_vals, 50.0)),
            "abs_p90": float(np.percentile(abs_vals, 90.0)),
            "abs_p95": float(np.percentile(abs_vals, 95.0)),
            "abs_p99": float(np.percentile(abs_vals, 99.0)),
            "maxabs": float(abs_vals.max()),
            "n_units": int(a.shape[0]),
            "n_strata": int(a.shape[1]),
        }
    for dkey, (post_ck, pre_ck) in (
        ("POST_PRE", ("POST", "PRE")), ("MID_PRE", ("MID", "PRE")), ("POST_MID", ("POST", "MID")),
    ):
        d_unit = unit_mean[post_ck] - unit_mean[pre_ck]
        d_means = d_unit[boot_idx].mean(axis=1)
        out["deltas"][dkey] = {"point": float(d_unit.mean()), "ci95": _ci95(d_means), "n": int(d_unit.shape[0])}
    return out


def diff_in_diff(
    arm_m: dict[str, np.ndarray],
    same_m: dict[str, np.ndarray],
    common_idx: np.ndarray,
    n_boot: int = N_BOOT, seed: int = MASTER_SEED, rng: np.random.Generator | None = None,
) -> dict:
    """Difference-in-differences for one causal arm vs the SAME baseline.

    arm_m[ck] / same_m[ck] are per-unit margins over ALL units (1-D, same
    length). common_idx selects the common-finite subspace (explicit n).
    ONE shared bootstrap matrix is applied to the same subspace; per
    replicate r: D_adj[r] = mean_r(arm delta) - mean_r(same delta), i.e.
    SAME and the causal arm use the same replicate indices. Returns per
    interval: raw_arm delta (point, ci95), raw_same delta, adjusted
    (point, ci95), and n.
    """
    n_all = None
    for nm, v in (("arm", arm_m), ("same", same_m)):
        for ck in CKS:
            a = np.asarray(v[ck], dtype=np.float64)
            if n_all is None:
                n_all = a.shape[0]
            elif a.shape[0] != n_all:
                raise ValueError(f"{nm}[{ck}] length mismatch")
    if common_idx.min() < 0 or common_idx.max() >= n_all:
        raise ValueError("common_idx outside the unit space")
    k = int(common_idx.size)
    if k < 2:
        raise ValueError(f"common subspace k={k} < 2")
    if rng is None:
        rng = np.random.default_rng(seed)
    sub_idx = rng.integers(0, k, size=(n_boot, k))
    out: dict = {"n": k, "n_all": n_all, "n_boot": n_boot, "seed": seed, "intervals": {}}
    for dkey, (post_ck, pre_ck) in (
        ("POST_PRE", ("POST", "PRE")), ("MID_PRE", ("MID", "PRE")), ("POST_MID", ("POST", "MID")),
    ):
        a_unit = np.asarray(arm_m[post_ck], dtype=np.float64)[common_idx] - np.asarray(arm_m[pre_ck], dtype=np.float64)[common_idx]
        s_unit = np.asarray(same_m[post_ck], dtype=np.float64)[common_idx] - np.asarray(same_m[pre_ck], dtype=np.float64)[common_idx]
        a_boot = a_unit[sub_idx].mean(axis=1)
        s_boot = s_unit[sub_idx].mean(axis=1)
        adj_boot = a_boot - s_boot
        out["intervals"][dkey] = {
            "raw_arm": {"point": float(a_unit.mean()), "ci95": _ci95(a_boot)},
            "raw_same": {"point": float(s_unit.mean()), "ci95": _ci95(s_boot)},
            "adjusted": {"point": float(adj_boot.mean()), "ci95": _ci95(adj_boot)},
        }
    return out


INTERVAL_CLASSES = (
    "EARLY_GAIN", "LATE_GAIN", "MONOTONIC_GAIN", "PLATEAU", "FLAT", "REVERSED", "NON_MONOTONIC",
)


def _sig_plus(d: dict) -> bool:
    return d["ci95"][0] > 0.0


def _sig_minus(d: dict) -> bool:
    return d["ci95"][1] < 0.0


def _significant(d: dict) -> bool:
    return _sig_plus(d) or _sig_minus(d)


def classify_interval(
    delta_early: dict, delta_late: dict, flat_threshold: float | None = None,
) -> str:
    """CI-backed interval classification for one arm (MID-PRE = early, POST-MID = late).

    Operational rules (locked by tests):
    MONOTONIC_GAIN  both segments CI-exclude 0 on the positive side
    REVERSED        both segments CI-exclude 0 on the negative side
    PLATEAU         early significant positive, late not significant and
                    |late point| <= 0.25 * |early point| (early jump + flat)
    EARLY_GAIN      early significant positive, late not significant
    LATE_GAIN       late significant positive, early not significant and
                    |early point| <= 0.25 * |late point| (flat + late gain)
    FLAT            neither significant and both |points| <= flat_threshold
                    (default 1e-4 = the historical noise-scale band)
    REVERSED        a single significant-negative segment with the other not
                    significantly positive
    NON_MONOTONIC   everything else (mixed signs, non-negligible non-sig drift,
                    significant negative + significant positive mix)
    """
    eps = 1e-12
    early_sig_p, late_sig_p = _sig_plus(delta_early), _sig_plus(delta_late)
    early_sig_m, late_sig_m = _sig_minus(delta_early), _sig_minus(delta_late)
    early_any, late_any = _significant(delta_early), _significant(delta_late)
    if early_sig_p and late_sig_p:
        return "MONOTONIC_GAIN"
    if early_sig_m and late_sig_m:
        return "REVERSED"
    if early_sig_p and not late_any:
        if abs(delta_late["point"]) <= 0.25 * max(abs(delta_early["point"]), eps):
            return "PLATEAU"
        return "EARLY_GAIN"
    if late_sig_p and not early_any:
        if abs(delta_early["point"]) <= 0.25 * max(abs(delta_late["point"]), eps):
            return "LATE_GAIN"
        return "NON_MONOTONIC"
    if early_sig_m and not late_sig_p and not late_sig_m:
        return "REVERSED"
    if late_sig_m and not early_sig_p and not early_sig_m:
        return "REVERSED"
    if not early_any and not late_any:
        thr = 1e-4 if flat_threshold is None else flat_threshold
        if abs(delta_early["point"]) <= thr and abs(delta_late["point"]) <= thr:
            return "FLAT"
        return "NON_MONOTONIC"
    return "NON_MONOTONIC"


def offset_tertile(norm_offset: float) -> str:
    """Universal cohort naming: normalized offset tertiles.

    START  < 1/3 ; CENTER [1/3, 2/3) ; END >= 2/3.
    Never L/C/R: the historical L/C/R labels were normalized-offset
    tertiles and do NOT automatically mean physical left/center/right.
    """
    if not np.isfinite(norm_offset) or not (0.0 <= norm_offset <= 1.0):
        raise ValueError(f"norm_offset out of [0,1]: {norm_offset}")
    if norm_offset < 1.0 / 3.0:
        return "START"
    if norm_offset < 2.0 / 3.0:
        return "CENTER"
    return "END"


PHYSICAL_LABELS: dict[str, dict[str, str]] = {
    "horizontal": {"START": "LEFT", "CENTER": "CENTER", "END": "RIGHT"},
    "vertical": {"START": "TOP", "CENTER": "CENTER", "END": "BOTTOM"},
}


def physical_offset_label(
    orientation: str, tertile: str, convention_verified: bool = PHYSICAL_CONVENTION_VERIFIED,
) -> str | None:
    """Orientation-specific physical label, FAIL CLOSED.

    Returns the physical label only when the production convention is
    verified (horizontal: 0=LEFT edge/1=RIGHT edge; vertical: 0=TOP/
    1=BOTTOM, from camera_viewport.py plan_camera_viewport); otherwise
    returns None and callers must fall back to H_*/V_* neutral labels.
    """
    if orientation not in ("horizontal", "vertical"):
        raise ValueError(f"unknown orientation: {orientation}")
    if tertile not in ("START", "CENTER", "END"):
        raise ValueError(f"unknown tertile: {tertile}")
    if not convention_verified:
        return None
    return PHYSICAL_LABELS[orientation][tertile]


def neutral_offset_label(orientation: str, tertile: str) -> str:
    """H_START/H_CENTER/H_END / V_START/V_CENTER/V_END (always safe)."""
    if orientation not in ("horizontal", "vertical"):
        raise ValueError(f"unknown orientation: {orientation}")
    return ("H_" if orientation == "horizontal" else "V_") + tertile


def same_margin(loss_same: np.ndarray, loss_correct: np.ndarray) -> np.ndarray:
    """N_same = Loss(SAME) - Loss(CORRECT) (exact-duplicate arm).

    Sign convention: positive = SAME (identical coordinates) has HIGHER
    loss. SAME is a byte/float-exact duplicate of CORRECT, so any
    non-zero N_same is a harness/runtime numeric artifact, not a causal
    effect; its mean/CI/quantiles define the numerical floor.
    """
    a = np.asarray(loss_same, dtype=np.float64)
    b = np.asarray(loss_correct, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("loss_same and loss_correct must share shape")
    return a - b


def arm_margin(loss_arm: np.ndarray, loss_correct: np.ndarray) -> np.ndarray:
    """M_arm = Loss(arm) - Loss(CORRECT).

    Sign convention (correction of the historical README prose error):
    M > 0 means the WRONG/alternative coordinates receive HIGHER loss,
    i.e. the correct coordinates are PREFERRED. The historical
    implementation and tests were already correct; only the explanatory
    prose was inverted.
    """
    a = np.asarray(loss_arm, dtype=np.float64)
    b = np.asarray(loss_correct, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("loss_arm and loss_correct must share shape")
    return a - b


def numerics_gate(
    determinism_valid: bool, camera_same: dict, ordinary_same: dict, causal_adjusted: dict,
    microprobe: dict | None, mean_bias_cap: float = 1e-5, drift_frac_cap: float = 0.5,
) -> tuple[bool, dict]:
    """Posthoc NUMERICS_CLEAN gate with a full detail record (not just a bool).

    camera_same / ordinary_same: same_baseline_stats() output (or None for
    ordinary when unavailable -> fail closed with a reason).
    causal_adjusted: {arm: diff_in_diff() output} for the PRIMARY arms
    (OPPOSITE/SHUFFLED); used only to scale the drift comparison.
    microprobe: posthoc_numerics_probe.py summary dict, or None (fail closed).

    Rules (each failure recorded in "reasons"):
    R1 determinism probes remain bitexact/valid;
    R2 every camera SAME per-ck mean CI contains 0 AND |mean| <= mean_bias_cap;
    R3 camera SAME POST-PRE delta CI contains 0 AND
       |delta point| <= drift_frac_cap * min(|adjusted primary POST-PRE point|);
    R4 ordinary SAME replication: per-ck mean CI contains 0 AND |mean| <= cap;
    R5 microprobe present and no hidden-state defect flagged.

    Required detail fields: camera_same_mean_ci, camera_same_delta_ci,
    ordinary_same_mean_ci, same_causal_scale_ratio, plus per-rule flags.
    """
    reasons: list[str] = []
    det = bool(determinism_valid)
    if not det:
        reasons.append("R1: determinism probes not all valid")
    cam_mean_ci = {ck: camera_same["checkpoints"][ck]["ci95"] for ck in CKS} if camera_same else None
    cam_mean_ok = bool(camera_same)
    if cam_mean_ok:
        for ck in CKS:
            c = camera_same["checkpoints"][ck]
            if not (c["ci95"][0] <= 0.0 <= c["ci95"][1]) or abs(c["mean"]) > mean_bias_cap:
                cam_mean_ok = False
                reasons.append("R2: camera SAME {} mean {:.3e} ci {} exceeds systematic-bias band".format(ck, c["mean"], c["ci95"]))
    else:
        reasons.append("R2: camera SAME baseline unavailable")
    cam_delta = camera_same["deltas"]["POST_PRE"] if camera_same else None
    cam_delta_ok = cam_delta is not None
    primary_points = [abs(causal_adjusted[a]["intervals"]["POST_PRE"]["adjusted"]["point"]) for a in PRIMARY_ARMS if a in causal_adjusted]
    scale_ratio = None
    if cam_delta is not None and primary_points:
        floor = min(primary_points)
        scale_ratio = float(abs(cam_delta["point"]) / floor) if floor > 0 else float("inf")
        if not (cam_delta["ci95"][0] <= 0.0 <= cam_delta["ci95"][1]):
            cam_delta_ok = False
            reasons.append("R3: camera SAME POST-PRE delta CI excludes 0 (systematic drift)")
        elif floor > 0 and abs(cam_delta["point"]) > drift_frac_cap * floor:
            cam_delta_ok = False
            reasons.append("R3: camera SAME POST-PRE drift {:.3e} > {}x smallest primary adjusted delta".format(cam_delta["point"], drift_frac_cap))
    elif cam_delta is not None:
        reasons.append("R3: no primary adjusted deltas to scale against")
        cam_delta_ok = False
    ord_mean_ci = {ck: ordinary_same["checkpoints"][ck]["ci95"] for ck in CKS} if ordinary_same else None
    ord_ok = bool(ordinary_same)
    if ord_ok:
        for ck in CKS:
            c = ordinary_same["checkpoints"][ck]
            if not (c["ci95"][0] <= 0.0 <= c["ci95"][1]) or abs(c["mean"]) > mean_bias_cap:
                ord_ok = False
                reasons.append("R4: ordinary SAME {} replication off: mean {:.3e} ci {}".format(ck, c["mean"], c["ci95"]))
    else:
        reasons.append("R4: ordinary SAME replication unavailable")
    mp = microprobe or {}
    mp_ok = bool(mp.get("present", False)) and mp.get("hidden_state_detected", True) is False
    if not mp_ok:
        reasons.append("R5: microprobe absent or flagged hidden-state/order defect")
    clean = det and cam_mean_ok and cam_delta_ok and ord_ok and mp_ok
    detail = {
        "clean": bool(clean),
        "determinism_valid": det,
        "camera_same_mean_ci": cam_mean_ci,
        "camera_same_delta_ci": {k: v["ci95"] for k, v in (camera_same["deltas"] if camera_same else {}).items()},
        "camera_same_means": {ck: camera_same["checkpoints"][ck]["mean"] for ck in CKS} if camera_same else None,
        "ordinary_same_mean_ci": ord_mean_ci,
        "same_causal_scale_ratio": scale_ratio,
        "microprobe": {"present": bool(mp.get("present", False)), "hidden_state_detected": mp.get("hidden_state_detected") if mp else None, "p1_max_abs_loss_delta_rel": mp.get("p1_max_abs_loss_delta_rel"), "same_vs_correct_rel_p99": mp.get("same_vs_correct_rel_p99")},
        "rules": {"R1": det, "R2": cam_mean_ok, "R3": cam_delta_ok, "R4": ord_ok, "R5": mp_ok},
        "reasons": reasons,
    }
    return clean, detail


VERDICT_CLASSES = (
    "POSITIVE_LOSS_PREFERENCE_LEARNING",
    "WEAK_POSITIVE_LOSS_PREFERENCE",
    "NO_DETECTED_LOSS_PREFERENCE_GAIN",
    "MISALIGNED_LOSS_PREFERENCE",
    "BLOCKED_NUMERICS",
)


def classify_verdict(
    numerics_clean: bool, adjusted: dict, end_systematic_negative: bool = False, flat_threshold: float | None = None,
) -> str:
    """Posthoc causal classification (A-E). NEVER reuses the historical
    stage3 verdict codes (that path was short-circuited by the
    same_maxabs < 1e-6 gate before the effect logic ever ran).

    A POSITIVE_LOSS_PREFERENCE_LEARNING: clean + both PRIMARY arms
    (OPPOSITE, SHUFFLED) adjusted POST-PRE CI lower > 0, no inversion.
    IDENTITY is supportive and reported separately, never the driver.
    B WEAK_POSITIVE_LOSS_PREFERENCE: clean + both primary point estimates
    positive but at least one CI overlaps 0.
    C NO_DETECTED_LOSS_PREFERENCE_GAIN: clean + primary adjusted deltas
    near zero (|point| <= flat_threshold; 1e-4 when omitted).
    D MISALIGNED_LOSS_PREFERENCE: clean + a primary adjusted POST-PRE CI
    upper < 0 (systematic wrong direction) or a systematic negative
    END-offset structure.
    E BLOCKED_NUMERICS: numerics gate says NO.
    """
    if not numerics_clean:
        return "BLOCKED_NUMERICS"
    pp = {a: adjusted[a]["intervals"]["POST_PRE"]["adjusted"] for a in PRIMARY_ARMS if a in adjusted}
    if "OPPOSITE" not in pp or "SHUFFLED" not in pp:
        return "BLOCKED_NUMERICS"
    inv = any(pp[a]["ci95"][1] < 0.0 for a in ("OPPOSITE", "SHUFFLED"))
    id_pp = adjusted["IDENTITY"]["intervals"]["POST_PRE"]["adjusted"] if "IDENTITY" in adjusted else None
    if id_pp is not None and id_pp["ci95"][1] < 0.0:
        inv = True
    if pp["OPPOSITE"]["ci95"][0] > 0.0 and pp["SHUFFLED"]["ci95"][0] > 0.0 and not inv:
        return "POSITIVE_LOSS_PREFERENCE_LEARNING"
    if pp["OPPOSITE"]["point"] > 0.0 and pp["SHUFFLED"]["point"] > 0.0 and not inv:
        return "WEAK_POSITIVE_LOSS_PREFERENCE"
    if end_systematic_negative or any(pp[a]["ci95"][1] < 0.0 for a in ("OPPOSITE", "SHUFFLED")):
        return "MISALIGNED_LOSS_PREFERENCE"
    thr = 1e-4 if flat_threshold is None else flat_threshold
    if all(abs(pp[a]["point"]) <= thr for a in ("OPPOSITE", "SHUFFLED")):
        return "NO_DETECTED_LOSS_PREFERENCE_GAIN"
    return "NO_DETECTED_LOSS_PREFERENCE_GAIN"


RECOMMENDATIONS = (
    "LONGER_P25_REVIEW_CANDIDATE",
    "REVIEW_CAMERA_SAMPLING_BALANCE_FIRST",
    "FIX_AUDIT_HARNESS_BEFORE_TRAINING",
    "NO_MORE_EXPOSURE",
)


def classify_recommendation(
    verdict: str, end_systematic_negative: bool = False, identity_point: float | None = None,
) -> str:
    """Training recommendation mapping (authorization is NEVER granted here).

    BLOCKED_NUMERICS                        -> FIX_AUDIT_HARNESS_BEFORE_TRAINING
    POSITIVE + no END problem               -> LONGER_P25_REVIEW_CANDIDATE
    POSITIVE/WEAK + systematic negative END -> REVIEW_CAMERA_SAMPLING_BALANCE_FIRST
    MISALIGNED / NO_DETECTED                -> NO_MORE_EXPOSURE
    """
    if verdict not in VERDICT_CLASSES:
        raise ValueError("unknown verdict class: " + str(verdict))
    if verdict == "BLOCKED_NUMERICS":
        return "FIX_AUDIT_HARNESS_BEFORE_TRAINING"
    if verdict == "POSITIVE_LOSS_PREFERENCE_LEARNING":
        return "REVIEW_CAMERA_SAMPLING_BALANCE_FIRST" if end_systematic_negative else "LONGER_P25_REVIEW_CANDIDATE"
    if verdict == "WEAK_POSITIVE_LOSS_PREFERENCE" and end_systematic_negative:
        return "REVIEW_CAMERA_SAMPLING_BALANCE_FIRST"
    return "NO_MORE_EXPOSURE"
