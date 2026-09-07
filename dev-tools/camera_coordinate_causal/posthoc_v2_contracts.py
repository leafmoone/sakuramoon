"""Camera coordinate causal audit - posthoc V2 contracts.

Pure stdlib + numpy. This module implements the V2 correction layer on top of
the immutable V1 contracts (posthoc_contracts.py, imported read-only and never
modified):

  1. Terminology split (sections 7-8): REPEAT_NUMERIC_JITTER_PRESENT (A) is a
     DIAGNOSTIC only and can NEVER by itself set HIDDEN_MUTABLE_STATE_DETECTED
     (F). F is B OR C OR D OR E, where B = coordinate map mutation,
     C = persistent state (parameter or buffer) mutation, D = order-dependent
     interleave drift beyond the P0 runtime floor, E = CI-backed accumulating
     drift beyond the P0 floor.
  2. Exact point estimates (section 21): diff_in_diff_v2() reports point
     estimates as the EXACT observed sample means (raw arm, raw same, and
     adjusted = mean(a_unit - s_unit)); the bootstrap is used for CIs only and
     its seed can never move a point estimate.
  3. V2 numerics gate (section 19): R1-R9. The primary basis stays the
     aggregate camera SAME floor (mean + cluster CI), checkpoint drift,
     ordinary SAME replication, determinism probes, and the V2 state/order
     checks. maxabs/p99/jitter are diagnostic-only.
  4. Offset balance statistics (sections 28-42): exact discrete planner
     probabilities, SMDs, cell reweighting, matched-pair bootstrap, deciles,
     Spearman, mirror bins.

The V1 VERDICT logic (A-E classes) is reused unchanged via
posthoc_contracts.classify_verdict; the V2 RECOMMENDATION mapping (CASE A-E)
is new and combines the verdict with the offset balance audit outcome.
"""
from __future__ import annotations

import numpy as np
import posthoc_contracts as pc

# ---------------------------------------------------------------------------
# Locked constants
# ---------------------------------------------------------------------------

MASTER_SEED_V2 = 20260907
N_BOOT_V2 = pc.N_BOOT  # 10000
CKS = pc.CKS
PRIMARY_ARMS = pc.PRIMARY_ARMS
SUPPORTIVE_ARM = pc.SUPPORTIVE_ARM
TERTS = ("START", "CENTER", "END")

# V1 cap retained for DIAGNOSTIC use only (section 8): never a gate input.
P1_REL_CAP_DIAGNOSTIC = 1e-6
MEAN_BIAS_CAP = 1e-5
DRIFT_FRAC_CAP = 0.5
FLAT_THRESHOLD = 1e-4
SMD_DIAGNOSTIC = 0.1
Z_IMBALANCE = 3.0

BALANCE_CLASSES = (
    "PLANNER_COUNT_IMBALANCE",
    "GEOMETRY_DISTRIBUTION_IMBALANCE",
    "EFFECT_PERSISTS_AFTER_GEOMETRY_BALANCE",
    "CONTENT_OR_SHARD_INTERACTION_SUSPECTED",
    "INCONCLUSIVE",
)

RECOMMENDATIONS_V2 = (
    "LONGER_P25_REVIEW_CANDIDATE",
    "REVIEW_VERTICAL_END_SUPERVISION_FIRST",
    "REVIEW_CAMERA_SAMPLING_BALANCE_FIRST",
    "FIX_AUDIT_HARNESS_BEFORE_TRAINING",
    "NO_MORE_EXPOSURE",
)


def _ci95(samples: np.ndarray) -> tuple[float, float]:
    return (float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5)))


# ---------------------------------------------------------------------------
# (1) Terminology: jitter vs hidden mutable state (sections 7, 18)
# ---------------------------------------------------------------------------

CONCEPT_A = "REPEAT_NUMERIC_JITTER_PRESENT"
CONCEPT_B = "COORDINATE_MAP_MUTATION_DETECTED"
CONCEPT_C = "PERSISTENT_BUFFER_MUTATION_DETECTED"
CONCEPT_D = "ORDER_DEPENDENT_DRIFT_DETECTED"
CONCEPT_E = "ACCUMULATING_DRIFT_DETECTED"
CONCEPT_F = "HIDDEN_MUTABLE_STATE_DETECTED"


def classify_microprobe_state(
    *,
    coordinate_map_mutated: bool,
    persistent_buffer_mutated: bool,
    parameter_mutated: bool = False,
    order_dependent_drift: bool,
    accumulating_drift: bool,
    repeat_numeric_jitter_present: bool,
) -> dict:
    """V2 hidden-state classification (section 18).

    F = B OR C OR D OR E. Concept C covers persistent model state (module
    buffers) and model parameters. A alone NEVER implies F (sections 7-8):
    pure repeat numerical jitter is a diagnostic, not a state defect.
    """
    b = bool(coordinate_map_mutated)
    c = bool(persistent_buffer_mutated) or bool(parameter_mutated)
    d = bool(order_dependent_drift)
    e = bool(accumulating_drift)
    f = b or c or d or e
    a = bool(repeat_numeric_jitter_present)
    reasons: list[str] = []
    if b:
        reasons.append("B: coordinate map mutated")
    if c:
        reasons.append("C: persistent state mutated (buffer or parameter)")
    if d:
        reasons.append("D: order-dependent interleave drift beyond P0 floor")
    if e:
        reasons.append("E: accumulating drift with CI-backed positive slope beyond P0 floor")
    if a and not f:
        reasons.append("A: repeat numeric jitter present (DIAGNOSTIC ONLY - does not imply hidden state)")
    return {
        CONCEPT_A: a,
        CONCEPT_B: b,
        CONCEPT_C: c,
        CONCEPT_D: d,
        CONCEPT_E: e,
        CONCEPT_F: f,
        # flat aliases for gate/report consumption
        "coordinate_map_mutation_detected": b,
        "persistent_buffer_mutation_detected": c,
        "parameter_mutation_detected": bool(parameter_mutated),
        "order_dependent_drift_detected": d,
        "accumulating_drift_detected": e,
        "hidden_mutable_state_detected": f,
        "repeat_jitter_alone_insufficient": bool(a and not f),
        "reasons": reasons,
    }


# ---------------------------------------------------------------------------
# (2) Difference-in-differences with EXACT point estimates (section 21)
# ---------------------------------------------------------------------------

def diff_in_diff_v2(
    arm_m: dict[str, np.ndarray],
    same_m: dict[str, np.ndarray],
    common_idx: np.ndarray,
    n_boot: int = N_BOOT_V2,
    seed: int = MASTER_SEED_V2,
    rng: np.random.Generator | None = None,
) -> dict:
    """V2 diff-in-differences: point estimates are the EXACT observed sample
    means; the paired shared bootstrap matrix is used for CIs ONLY.

    raw_arm point   = mean(a_unit)          where a_unit = arm[post]-arm[pre]
    raw_same point  = mean(s_unit)          where s_unit = same[post]-same[pre]
    adjusted point  = mean(a_unit - s_unit) = raw_arm point - raw_same point
    (exact, by construction). A different bootstrap seed can change the CIs
    but never the points (locked by regression test).
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
        sub_idx = pc.paired_bootstrap_indices(k, n_boot, seed)
    else:
        sub_idx = rng.integers(0, k, size=(n_boot, k))
    out: dict = {"n": k, "n_all": n_all, "n_boot": int(n_boot), "seed": int(seed), "intervals": {}}
    for dkey, (post_ck, pre_ck) in (
        ("POST_PRE", ("POST", "PRE")),
        ("MID_PRE", ("MID", "PRE")),
        ("POST_MID", ("POST", "MID")),
    ):
        a_unit = np.asarray(arm_m[post_ck], dtype=np.float64)[common_idx] - np.asarray(arm_m[pre_ck], dtype=np.float64)[common_idx]
        s_unit = np.asarray(same_m[post_ck], dtype=np.float64)[common_idx] - np.asarray(same_m[pre_ck], dtype=np.float64)[common_idx]
        adj_unit = a_unit - s_unit
        a_boot = a_unit[sub_idx].mean(axis=1)
        s_boot = s_unit[sub_idx].mean(axis=1)
        adj_boot = adj_unit[sub_idx].mean(axis=1)
        out["intervals"][dkey] = {
            "raw_arm": {"point": float(a_unit.mean()), "ci95": _ci95(a_boot)},
            "raw_same": {"point": float(s_unit.mean()), "ci95": _ci95(s_boot)},
            "adjusted": {"point": float(adj_unit.mean()), "ci95": _ci95(adj_boot)},
        }
    return out


def select_arm_subspace(
    arm_m: dict[str, np.ndarray],
    same_m: dict[str, np.ndarray],
    opp_na: np.ndarray | None = None,
) -> np.ndarray:
    """Common-finite subspace for one causal arm vs SAME, with the
    OPPOSITE-specific opp_na exclusion applied when opp_na flags are given.

    arm_m / same_m: {ck: per-unit 1-D array over ALL units}. Returns the final
    index array used by diff_in_diff_v2: units finite at every checkpoint for
    BOTH the arm and SAME, and (if opp_na is provided and non-None) not
    flagged opp_na. opp_na is a boolean array over ALL units.
    """
    arm_name = next(iter(arm_m))
    sub = pc.common_finite_subset({arm_name: arm_m, "SAME": same_m}, (arm_name, "SAME"))
    if opp_na is not None:
        flags = np.asarray(opp_na, dtype=bool)
        sub = sub[~flags[sub]]
    return sub


# ---------------------------------------------------------------------------
# (3) V2 numerics gate (section 19)
# ---------------------------------------------------------------------------

def numerics_v2_gate(
    determinism_valid: bool,
    camera_same: dict | None,
    ordinary_same: dict | None,
    causal_adjusted: dict,
    microprobe_v2: dict | None,
    mean_bias_cap: float = MEAN_BIAS_CAP,
    drift_frac_cap: float = DRIFT_FRAC_CAP,
) -> tuple[bool, dict]:
    """V2 NUMERICS_CLEAN gate with a full detail record.

    R1 determinism probes valid;
    R2 camera SAME per-ck mean CI contains 0 AND |mean| <= mean_bias_cap;
    R3 camera SAME POST-PRE delta CI contains 0;
    R4 camera SAME drift tiny relative to the primary causal effect:
       |POST-PRE point| <= drift_frac_cap * min(|adjusted OPP|, |adjusted SHUF|);
    R5 ordinary SAME replication compatible with zero (per-ck CI contains 0
       AND |mean| <= mean_bias_cap);
    R6 coordinate map mutation = False;
    R7 persistent buffer/parameter mutation = False;
    R8 order-dependent drift = False;
    R9 accumulating drift = False.

    REPEAT_NUMERIC_JITTER_PRESENT is NOT a fail condition; it is preserved as
    a warning in the detail record (section 8).
    """
    reasons: list[str] = []
    det = bool(determinism_valid)
    if not det:
        reasons.append("R1: determinism probes not all valid")

    cam_mean_ok = bool(camera_same)
    cam_mean_ci: dict[str, tuple[float, float]] | None = None
    if cam_mean_ok:
        cam_mean_ci = {}
        for ck in CKS:
            c = camera_same["checkpoints"][ck]
            cam_mean_ci[ck] = tuple(c["ci95"])
            if not (c["ci95"][0] <= 0.0 <= c["ci95"][1]) or abs(c["mean"]) > mean_bias_cap:
                cam_mean_ok = False
                reasons.append("R2: camera SAME {} mean {:.3e} ci {} exceeds systematic-bias band".format(ck, c["mean"], c["ci95"]))
    else:
        reasons.append("R2: camera SAME baseline unavailable")

    cam_delta = camera_same["deltas"]["POST_PRE"] if camera_same else None
    cam_delta_ok = cam_delta is not None
    primary_points = [
        abs(causal_adjusted[a]["intervals"]["POST_PRE"]["adjusted"]["point"])
        for a in PRIMARY_ARMS if a in causal_adjusted
    ]
    scale_ratio: float | None = None
    ci_width_ratio: float | None = None
    if cam_delta is not None and primary_points:
        floor = min(primary_points)
        scale_ratio = float(abs(cam_delta["point"]) / floor) if floor > 0 else float("inf")
        width = cam_delta["ci95"][1] - cam_delta["ci95"][0]
        ci_width_ratio = float(width / floor) if floor > 0 else None
        if not (cam_delta["ci95"][0] <= 0.0 <= cam_delta["ci95"][1]):
            cam_delta_ok = False
            reasons.append("R3: camera SAME POST-PRE delta CI excludes 0 (systematic drift)")
        elif floor > 0 and abs(cam_delta["point"]) > drift_frac_cap * floor:
            cam_delta_ok = False
            reasons.append("R4: camera SAME POST-PRE drift {:.3e} > {}x smallest primary adjusted delta".format(cam_delta["point"], drift_frac_cap))
    elif cam_delta is not None:
        cam_delta_ok = False
        reasons.append("R3/R4: no primary adjusted deltas to scale against")
    else:
        cam_delta_ok = False
        reasons.append("R3: camera SAME POST-PRE delta unavailable")

    ord_ok = bool(ordinary_same)
    ord_mean_ci: dict[str, tuple[float, float]] | None = None
    if ord_ok:
        ord_mean_ci = {}
        for ck in CKS:
            c = ordinary_same["checkpoints"][ck]
            ord_mean_ci[ck] = tuple(c["ci95"])
            if not (c["ci95"][0] <= 0.0 <= c["ci95"][1]) or abs(c["mean"]) > mean_bias_cap:
                ord_ok = False
                reasons.append("R5: ordinary SAME {} replication off: mean {:.3e} ci {}".format(ck, c["mean"], c["ci95"]))
    else:
        ord_ok = False
        reasons.append("R5: ordinary SAME replication unavailable")

    mp = microprobe_v2 or {}
    state = mp.get("state") if mp else None
    if mp.get("present") and isinstance(state, dict):
        r6 = state.get(CONCEPT_B) is False
        r7 = (state.get(CONCEPT_C) is False) and (state.get("parameter_mutation_detected") is False)
        r8 = state.get(CONCEPT_D) is False
        r9 = state.get(CONCEPT_E) is False
        if not r6:
            reasons.append("R6: coordinate map mutation detected")
        if not r7:
            reasons.append("R7: persistent buffer/parameter mutation detected")
        if not r8:
            reasons.append("R8: order-dependent drift detected beyond P0 floor")
        if not r9:
            reasons.append("R9: accumulating drift detected beyond P0 floor")
    else:
        r6 = r7 = r8 = r9 = False
        reasons.append("R6-R9: microprobe-v2 absent or state block missing (fail closed)")

    jitter_present = bool(mp.get("repeat_numeric_jitter_present", False)) if mp else False
    clean = det and cam_mean_ok and cam_delta_ok and ord_ok and r6 and r7 and r8 and r9
    detail = {
        "clean": bool(clean),
        "determinism_valid": det,
        "camera_same_mean_ci": cam_mean_ci,
        "camera_same_delta_ci": {k: tuple(v["ci95"]) for k, v in (camera_same["deltas"] if camera_same else {}).items()},
        "camera_same_means": {ck: camera_same["checkpoints"][ck]["mean"] for ck in CKS} if camera_same else None,
        "ordinary_same_mean_ci": ord_mean_ci,
        "same_causal_scale_ratio": scale_ratio,
        "same_delta_ci_width": (cam_delta["ci95"][1] - cam_delta["ci95"][0]) if cam_delta else None,
        "same_delta_ci_width_vs_primary": ci_width_ratio,
        "microprobe_v2": {
            "present": bool(mp.get("present", False)),
            "hidden_mutable_state_detected": (state or {}).get(CONCEPT_F),
            "repeat_numeric_jitter_present": jitter_present,
            "p0_bitexact_rate": (mp.get("p0") or {}).get("pred_bitexact_rate"),
            "p0_max_abs_loss_delta_rel": (mp.get("p0") or {}).get("max_abs_loss_delta_rel"),
        },
        "repeat_jitter_warning": {
            "present": jitter_present,
            "note": "repeat numeric jitter is a diagnostic only; it is NOT a numerics fail condition in V2",
        },
        "rules": {"R1": det, "R2": cam_mean_ok, "R3": cam_delta_ok, "R4": cam_delta_ok, "R5": ord_ok, "R6": r6, "R7": r7, "R8": r8, "R9": r9},
        "reasons": reasons,
    }
    return clean, detail


# ---------------------------------------------------------------------------
# (4) Verdict + V2 recommendation (sections 23, 44)
# ---------------------------------------------------------------------------

def classify_verdict_v2(
    numerics_clean: bool, adjusted: dict, end_systematic_negative: bool = False, flat_threshold: float | None = None,
) -> str:
    """Reuse the V1 verdict classes (A-E) with the exact-point adjusted dict."""
    return pc.classify_verdict(numerics_clean, adjusted, end_systematic_negative, flat_threshold)


def classify_recommendation_v2(
    verdict: str,
    balance: dict | None,
) -> tuple[str, dict]:
    """V2 recommendation CASE A-E (section 44). RECOMMENDATION, never
    AUTHORIZATION.

    CASE D: verdict BLOCKED_NUMERICS -> FIX_AUDIT_HARNESS_BEFORE_TRAINING
    CASE E: causal null / no positive direction -> NO_MORE_EXPOSURE
    CASE B: positive causal + BOTTOM negative persists after geometry balance
            -> REVIEW_VERTICAL_END_SUPERVISION_FIRST
    CASE C: positive causal + geometry imbalance explains the BOTTOM negative
            (it collapses after balance) -> REVIEW_CAMERA_SAMPLING_BALANCE_FIRST
    CASE A: positive causal + numerics clean + no systematic directional
            negative after balance -> LONGER_P25_REVIEW_CANDIDATE
    """
    if verdict not in pc.VERDICT_CLASSES:
        raise ValueError("unknown verdict class: " + str(verdict))
    notes: list[str] = []
    if verdict == "BLOCKED_NUMERICS":
        notes.append("CASE D: numerics blocked")
        return "FIX_AUDIT_HARNESS_BEFORE_TRAINING", {"case": "D", "notes": notes}
    positive = verdict in ("POSITIVE_LOSS_PREFERENCE_LEARNING", "WEAK_POSITIVE_LOSS_PREFERENCE")
    if not positive:
        notes.append(f"CASE E: causal null or non-positive (verdict {verdict})")
        return "NO_MORE_EXPOSURE", {"case": "E", "notes": notes}
    bal = balance or {}
    if bal.get("present") is False or "bottom_negative_after_reweight" not in bal:
        notes.append("balance audit unavailable: END negative cannot be cleared - conservative vertical-end review")
        return "REVIEW_VERTICAL_END_SUPERVISION_FIRST", {"case": "B(unverified)", "notes": notes}
    if bal["bottom_negative_after_reweight"]:
        if bal.get("geometry_explains_bottom"):
            notes.append("CASE C: geometry imbalance explains the BOTTOM negative (collapses after balance)")
            return "REVIEW_CAMERA_SAMPLING_BALANCE_FIRST", {"case": "C", "notes": notes}
        notes.append("CASE B: BOTTOM negative persists after geometry balance")
        return "REVIEW_VERTICAL_END_SUPERVISION_FIRST", {"case": "B", "notes": notes}
    notes.append("CASE A: no systematic directional negative after balance")
    return "LONGER_P25_REVIEW_CANDIDATE", {"case": "A", "notes": notes}


# ---------------------------------------------------------------------------
# (5) Offset balance statistics (sections 28-42)
# ---------------------------------------------------------------------------

def tertile_of(norm_offset: float) -> str:
    return pc.offset_tertile(norm_offset)


def discrete_tertile_probs(available: int) -> dict[str, float]:
    """Exact theoretical per-tertyle probabilities for one unit.

    Production planner: k ~ Uniform{0..available} (randrange(available + 1)),
    normalized_offset = k / available (0.0 when available == 0). Tertile bands
    on the normalized offset: START < 1/3, CENTER [1/3, 2/3), END >= 2/3.
    The probabilities are computed by exact integer counting over all legal
    k; they sum to EXACTLY 1. No 1/3 assumption is made (section 28).
    """
    a = int(available)
    if a < 0:
        raise ValueError("available must be >= 0")
    if a == 0:
        return {"START": 1.0, "CENTER": 0.0, "END": 0.0}
    counts = [0, 0, 0]
    for k in range(a + 1):
        q = k / a
        counts[0 if q < 1.0 / 3.0 else (1 if q < 2.0 / 3.0 else 2)] += 1
    n = a + 1
    return {t: float(counts[i]) / n for i, t in enumerate(TERTS)}


def expected_tertile_counts(availables: list[int] | np.ndarray) -> dict:
    """Expected tertile counts and marginal standard deviations over
    independent units with per-unit discrete probabilities (section 28)."""
    avail = [int(a) for a in np.asarray(availables).ravel()]
    exp = [0.0, 0.0, 0.0]
    var = [0.0, 0.0, 0.0]
    for a in avail:
        p = discrete_tertile_probs(a)
        for i, t in enumerate(TERTS):
            exp[i] += p[t]
            var[i] += p[t] * (1.0 - p[t])
    return {
        "n_units": len(avail),
        "expected": {t: float(exp[i]) for i, t in enumerate(TERTS)},
        "sd": {t: float(np.sqrt(var[i])) for i, t in enumerate(TERTS)},
    }


def observed_tertile_z(availables: list[int] | np.ndarray, observed: dict[str, int]) -> dict:
    """z-scores (observed - expected) / sd per tertile."""
    e = expected_tertile_counts(availables)
    z = {}
    for t in TERTS:
        sd = e["sd"][t]
        z[t] = float((observed[t] - e["expected"][t]) / sd) if sd > 0 else 0.0
    return {"z": z, "observed": {t: int(observed[t]) for t in TERTS}, **e}


def mirror_symmetry(availables: list[int] | np.ndarray, offsets: list[int] | np.ndarray | None = None) -> dict:
    """k vs available-k symmetry (section 29).

    Static (code-level): with k ~ Uniform{0..available}, P(k) = 1/(available+1)
    = P(available - k) exactly, so the mirror probabilities are exactly
    symmetric for every available.
    Empirical (optional): when per-unit integer offsets are provided, compare
    the observed count of k with the count of available-k per available bucket.
    """
    avail = [int(a) for a in np.asarray(availables).ravel()]
    buckets: dict[int, list[int]] = {}
    if offsets is not None:
        off = [int(o) for o in np.asarray(offsets).ravel()]
        if len(off) != len(avail):
            raise ValueError("offsets length must match availables")
        for a, o in zip(avail, off):
            if not (0 <= o <= a):
                raise ValueError(f"offset {o} outside [0, {a}]")
            buckets.setdefault(a, []).append(o)
    empirical = {}
    for a, offs in sorted(buckets.items()):
        n_k = len(offs)
        n_mirror = sum(1 for o in offs if o == a - o)
        dev = max(abs(offs.count(o) - offs.count(a - o)) for o in range(a + 1)) if n_k else 0
        empirical[str(a)] = {
            "n_units": n_k,
            "units_on_mirror": n_mirror,
            "max_count_deviation_k_vs_mirror": int(dev),
        }
    return {
        "uniform_sampler": True,
        "static_exact_symmetry": True,
        "static_proof": "k = random.Random(offset_seed).randrange(available + 1): P(k) = 1/(available+1) = P(available-k) for all k",
        "n_available_values": len(set(avail)),
        "empirical_by_available": empirical,
    }


# Covariates that DEFINE the TOP/BOTTOM split itself: their SMD is tautologically
# large for any such split and must not drive the geometry-imbalance gate
# (spec 42-43: only a genuine geometry imbalance counts).
SPLIT_DEFINITIONAL_COVARIATES = ("norm_offset", "pixel_shift", "abs_displacement")


def geometry_imbalance_gate(
    smds: dict[str, float], threshold: float = SMD_DIAGNOSTIC
) -> tuple[str | None, bool]:
    """(largest non-definitional covariate, imbalance flag) from an SMD dict.

    Split-definitional covariates and NaN SMDs are excluded; returns
    (None, False) when no non-definitional finite SMD exists.
    """
    candidates = {
        k: float(v) for k, v in smds.items()
        if k not in SPLIT_DEFINITIONAL_COVARIATES and not np.isnan(v)
    }
    if not candidates:
        return None, False
    largest = max(candidates, key=lambda k: candidates[k])
    return largest, bool(candidates[largest] >= threshold)


def smd(a: np.ndarray, b: np.ndarray) -> float:
    """Standardized mean difference with pooled sd (section 32). |SMD| < 0.1
    is a conventional diagnostic, not a scientific hard truth."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size == 0 or b.size == 0:
        return float("nan")
    sa, sb = float(a.std(ddof=1)) if a.size > 1 else 0.0, float(b.std(ddof=1)) if b.size > 1 else 0.0
    pooled = float(np.sqrt((sa * sa + sb * sb) / 2.0))
    if pooled <= 0.0:
        return 0.0
    return float(abs(a.mean() - b.mean()) / pooled)


def cell_reweight_weights(n_top: dict[str, int], n_bottom: dict[str, int]) -> dict:
    """Exact cell reweighting (section 34).

    Only cells where BOTH groups have units (count > 0) get weight. A cell
    present in one dict with count 0 is treated as absent. Within shared
    cell c:
      w_top(u)  = n_c / (2 * n_top_c)   and   w_bottom(u) = n_c / (2 * n_bottom_c)
    so the weighted cell count of EACH group in cell c equals n_c / 2, i.e.
    TOP and BOTTOM share the identical target cell distribution. Units in
    non-shared cells are dropped (support mismatch).
    """
    top_pos = {c: int(n) for c, n in n_top.items() if n > 0}
    bot_pos = {c: int(n) for c, n in n_bottom.items() if n > 0}
    shared = sorted(set(top_pos) & set(bot_pos))
    dropped_top = {c: top_pos[c] for c in sorted(set(top_pos) - set(bot_pos))}
    dropped_bottom = {c: bot_pos[c] for c in sorted(set(bot_pos) - set(top_pos))}
    n_top = top_pos
    n_bottom = bot_pos
    w_top = {c: (n_top[c] + n_bottom[c]) / (2.0 * n_top[c]) for c in shared}
    w_bottom = {c: (n_top[c] + n_bottom[c]) / (2.0 * n_bottom[c]) for c in shared}
    return {
        "shared_cells": shared,
        "n_shared_cells": len(shared),
        "weight_top_by_cell": w_top,
        "weight_bottom_by_cell": w_bottom,
        "dropped_top_cells": dropped_top,
        "dropped_bottom_cells": dropped_bottom,
        "n_dropped_top": int(sum(dropped_top.values())),
        "n_dropped_bottom": int(sum(dropped_bottom.values())),
        "n_retained_top": int(sum(n_top[c] for c in shared)),
        "n_retained_bottom": int(sum(n_bottom[c] for c in shared)),
    }


def effective_sample_size(weights: np.ndarray) -> float:
    w = np.asarray(weights, dtype=np.float64)
    s = float(w.sum())
    if s <= 0:
        return 0.0
    return float(s * s / float((w * w).sum()))


def weighted_mean_ci(
    values: np.ndarray, weights: np.ndarray, n_boot: int = N_BOOT_V2, seed: int = MASTER_SEED_V2,
    rng: np.random.Generator | None = None,
) -> dict:
    """Weighted mean + 95% CI via weighted cluster bootstrap (units are the
    cluster; resample units with probability proportional to weight, then take
    the weighted mean of each replicate)."""
    v = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if v.shape != w.shape:
        raise ValueError("values/weights shape mismatch")
    if v.size == 0:
        raise ValueError("no units")
    if (w <= 0).any():
        raise ValueError("weights must be positive")
    if rng is None:
        rng = np.random.default_rng(seed)
    p = w / w.sum()
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    p_rep = p[idx]
    v_rep = v[idx]
    denom = p_rep.sum(axis=1)
    reps = (p_rep * v_rep).sum(axis=1) / denom
    point = float((w * v).sum() / w.sum())
    return {"point": point, "ci95": _ci95(reps), "ess": effective_sample_size(w), "n_units": int(v.size)}


def match_nearest(
    bottom_feats: np.ndarray, top_feats: np.ndarray,
) -> dict:
    """Greedy 1:1 nearest-neighbor matching without replacement (section 35).

    Features must be standardized (z-scores) before calling. Each BOTTOM unit
    is matched to its closest still-available TOP unit by Euclidean distance.
    Returns pair indices, distances, and whether replacement was needed
    (it never is, given the 1:1 constraint and n_bottom <= n_top).
    """
    bf = np.asarray(bottom_feats, dtype=np.float64)
    tf = np.asarray(top_feats, dtype=np.float64)
    nb, nt = bf.shape[0], tf.shape[0]
    if nb == 0:
        return {"pairs": [], "distances": [], "n_pairs": 0, "with_replacement": False, "unmatched_bottom": 0}
    if nt < nb:
        raise ValueError("need at least as many TOP as BOTTOM units")
    dist = np.sqrt(((bf[:, None, :] - tf[None, :, :]) ** 2).sum(axis=2))
    used = np.zeros(nt, dtype=bool)
    pairs: list[tuple[int, int]] = []
    ds: list[float] = []
    order = np.argsort(dist.min(axis=1), kind="stable")
    for bi in order:
        avail = np.flatnonzero(~used)
        j = int(avail[np.argmin(dist[bi, avail])])
        pairs.append((int(bi), j))
        ds.append(float(dist[bi, j]))
        used[j] = True
    return {"pairs": pairs, "distances": ds, "n_pairs": len(pairs), "with_replacement": False, "unmatched_bottom": 0}


def paired_bootstrap_diff(
    diffs: np.ndarray, n_boot: int = N_BOOT_V2, seed: int = MASTER_SEED_V2,
    rng: np.random.Generator | None = None,
) -> dict:
    """Mean + 95% CI of per-pair differences; PAIRS are the bootstrap cluster
    (section 35)."""
    d = np.asarray(diffs, dtype=np.float64)
    if d.size < 2:
        raise ValueError("need >= 2 pairs")
    if rng is None:
        rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n_boot, d.size))
    reps = d[idx].mean(axis=1)
    return {"point": float(d.mean()), "ci95": _ci95(reps), "n_pairs": int(d.size)}


def _ranks(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    order = np.argsort(v, kind="stable")
    ranks = np.empty(v.size, dtype=np.float64)
    vs = v[order]
    i = 0
    while i < v.size:
        j = i
        while j + 1 < v.size and vs[j + 1] == vs[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0  # average 1-based ranks
        i = j + 1
    return ranks


def spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 3:
        raise ValueError("need >= 3 observations")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("non-finite values")
    rx, ry = _ranks(x), _ranks(y)
    cx, cy = rx - rx.mean(), ry - ry.mean()
    denom = float(np.sqrt((cx * cx).sum() * (cy * cy).sum()))
    if denom <= 0:
        return 0.0
    return float((cx * cy).sum() / denom)


def spearman_ci(
    x: np.ndarray, y: np.ndarray, n_boot: int = N_BOOT_V2, seed: int = MASTER_SEED_V2,
    rng: np.random.Generator | None = None,
) -> tuple[float, tuple[float, float]]:
    if rng is None:
        rng = np.random.default_rng(seed)
    n = x.size
    idx = rng.integers(0, n, size=(n_boot, n))
    reps = np.empty(n_boot)
    for r in range(n_boot):
        reps[r] = spearman_rho(x[idx[r]], y[idx[r]])
    return float(spearman_rho(x, y)), _ci95(reps)


def decile_stats(
    q: np.ndarray, d: np.ndarray, n_boot: int = N_BOOT_V2, seed: int = MASTER_SEED_V2,
    rng: np.random.Generator | None = None,
) -> list[dict]:
    """Per-decile (by q) n / mean / 95% CI of d; unit bootstrap within decile."""
    q = np.asarray(q, dtype=np.float64)
    d = np.asarray(d, dtype=np.float64)
    if q.size != d.size or q.size == 0:
        raise ValueError("q/d mismatch or empty")
    edges = np.linspace(0.0, 1.0, 11)
    qs = np.clip((q - q.min()) / max(float(q.max() - q.min()), 1e-30), 0.0, 1.0)
    if rng is None:
        rng = np.random.default_rng(seed)
    out = []
    for i in range(10):
        if i < 9:
            sel = np.flatnonzero((qs >= edges[i]) & (qs < edges[i + 1]))
        else:
            sel = np.flatnonzero((qs >= edges[i]) & (qs <= edges[i + 1]))  # last bin closed at 1.0
        if sel.size == 0:
            out.append({"decile": i, "n": 0, "mean": None, "ci95": None})
            continue
        dv = d[sel]
        if sel.size >= 2:
            idx = rng.integers(0, sel.size, size=(min(n_boot, 4000), sel.size))
            ci = _ci95(dv[idx].mean(axis=1))
        else:
            ci = (float(dv[0]), float(dv[0]))
        out.append({"decile": i, "n": int(sel.size), "mean": float(dv.mean()), "ci95": ci})
    return out


def mirror_bin_stats(
    q: np.ndarray, d: np.ndarray, width: float = 0.1, n_boot: int = N_BOOT_V2,
    seed: int = MASTER_SEED_V2, rng: np.random.Generator | None = None,
) -> list[dict]:
    """Mirror offset bins q vs 1-q (section 41): [0,.1] vs [.9,1], [.1,.2] vs
    [.8,.9], ... report the per-pair mirror difference (lower - upper) with a
    cluster bootstrap CI on the pooled two-side units."""
    q = np.asarray(q, dtype=np.float64)
    d = np.asarray(d, dtype=np.float64)
    if q.size != d.size:
        raise ValueError("q/d mismatch")
    if rng is None:
        rng = np.random.default_rng(seed)
    m = round(1.0 / width)
    out = []
    for i in range(m // 2):
        lo_sel = np.flatnonzero((q >= i * width) & (q < (i + 1) * width))
        hi_upper = 1.0 - i * width
        if i == 0:
            hi_sel = np.flatnonzero((q >= 1.0 - (i + 1) * width) & (q <= hi_upper))  # closed at 1.0
        else:
            hi_sel = np.flatnonzero((q >= 1.0 - (i + 1) * width) & (q < hi_upper))
        lo, hi = d[lo_sel], d[hi_sel]
        both = np.concatenate([lo, hi])
        sign = np.concatenate([np.ones(lo.size), -np.ones(hi.size)])
        point = float((sign * both).mean()) if both.size else float("nan")
        ci = None
        if both.size >= 2:
            idx = rng.integers(0, both.size, size=(min(n_boot, 4000), both.size))
            ci = _ci95((sign[idx] * both[idx]).mean(axis=1))
        out.append({
            "bin_low": f"[{i * width:.1f},{(i + 1) * width:.1f})",
            "bin_high": f"[{1.0 - (i + 1) * width:.1f},{1.0 - i * width:.1f})",
            "n_low": int(lo.size),
            "n_high": int(hi.size),
            "mean_low": float(lo.mean()) if lo.size else None,
            "mean_high": float(hi.mean()) if hi.size else None,
            "mirror_diff_low_minus_high": point,
            "ci95": ci,
        })
    return out


def slope_ci(
    slopes: np.ndarray, n_boot: int = N_BOOT_V2, seed: int = MASTER_SEED_V2,
    rng: np.random.Generator | None = None,
) -> dict:
    """Per-unit/block slopes pooled: mean slope + 95% CI via cluster bootstrap
    (the block is the cluster). Section 17."""
    s = np.asarray(slopes, dtype=np.float64)
    if s.size < 2:
        raise ValueError("need >= 2 blocks")
    if rng is None:
        rng = np.random.default_rng(seed)
    idx = rng.integers(0, s.size, size=(n_boot, s.size))
    reps = s[idx].mean(axis=1)
    return {"point": float(s.mean()), "ci95": _ci95(reps), "n_blocks": int(s.size)}


def ols_slope(y: np.ndarray, x: np.ndarray | None = None) -> float:
    """Least-squares slope of y on x (default x = 0..n-1)."""
    y = np.asarray(y, dtype=np.float64)
    if x is None:
        x = np.arange(y.size, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    cx, cy = x - x.mean(), y - y.mean()
    denom = float((cx * cx).sum())
    if denom <= 0:
        return 0.0
    return float((cx * cy).sum() / denom)
