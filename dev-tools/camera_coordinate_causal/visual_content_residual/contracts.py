"""Shared contracts for the Visual Content Residual Audit (Camera v2, phase B inputs).

This audit re-scores the already-audited 512 same-source vertical mirror pairs
with real visual-content metrics (CLIP image-full preservation + PE-Spatial
retained content). It is CPU-only: no model forwards, no encodes, no training.

Phase separation (pre-registered):
  PHASE A  freeze_visual_subsets.py  reads ONLY the frozen pair manifest and the
             frozen CLIP/PE feature JSONs. It emits visual-subsets.json (+ sha
             freeze) and must never look at per-pair mirror statistics.
  PHASE B  analyze.py  verifies the freeze, then combines the frozen subset
             manifest with the frozen per-pair mirror statistics.

All random quantities use the pre-registered seed 20260907 with n=10000
source-pair bootstrap resamples. Points are always the exact float64 observed
mean; the bootstrap only supplies the CI.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

# ---------------- frozen identities ----------------

BASE_SHA = "f3d183a35b3123e0b9c2a301fbc68ff5a276d345"
PAIR_MANIFEST_SHA = "7f3a81b248473d0b0591377c1bcd9f7df353066c9716836dabe90d9861f56297"

BOOT_SEED = 20260907
N_BOOT = 10000

N_PAIRS = 512
V50_TARGET = 256
V25_TARGET = 128

SCHEMA_VERSION = "vcr-1"

SUBSET_NAMES = (
    "ALL",
    "VISUAL_GEO_BALANCED_50",
    "VISUAL_GEO_BALANCED_25",
    "GLOBAL_VISUAL_50",
    "GLOBAL_VISUAL_25",
    "CLIP_IMAGE_ONLY_GEO_50",
    "CLIP_IMAGE_ONLY_GEO_25",
    "PE_ONLY_GEO_50",
    "PE_ONLY_GEO_25",
    "JOINT_LOW_VISUAL_50_INTERSECTION",
)

TRANS_KEYS = ("PRE|MID", "MID|POST", "PRE|POST")

# ---------------- geometry strata (frozen VBS definitions) ----------------


def zoom_band(zoom: float) -> str:
    """mild < 1.20 <= medium < 1.35 <= strong (stage1 camera_zoom_band)."""
    z = float(zoom)
    if z < 1.20:
        return "mild"
    if z < 1.35:
        return "medium"
    return "strong"


def latent_shift_band(latent_shift: float) -> str:
    """<2, [2,4), >=4 (latent units; manifest stores abs/16, non-negative)."""
    v = float(latent_shift)
    if v < 2.0:
        return "lt2"
    if v < 4.0:
        return "2to4"
    return "ge4"


def geometry_stratum(pair: dict) -> str:
    """cell = original_side x zoom band x latent-shift band (<= 18 cells)."""
    return "|".join(
        (
            str(pair["original_side"]),
            zoom_band(pair["equivalent_zoom"]),
            latent_shift_band(pair["latent_shift"]),
        )
    )


# ---------------- ranks / composite visual asymmetry ----------------


def average_ranks(values: np.ndarray) -> np.ndarray:
    """1-based ranks with average ranks for ties; stable, deterministic."""
    v = np.asarray(values, dtype=np.float64)
    n = v.size
    order = np.argsort(v, kind="stable")
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and v[order[j + 1]] == v[order[i]]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def percentile_rank_average_ties(values) -> np.ndarray:
    """Rank-normalized to [0, 1]: lowest -> 0, highest -> 1, ties averaged."""
    v = np.asarray(values, dtype=np.float64)
    n = v.size
    if n == 0:
        return np.empty(0, dtype=np.float64)
    if n == 1:
        return np.zeros(1, dtype=np.float64)
    ranks = average_ranks(v)
    return (ranks - 1.0) / (n - 1.0)


def composite_visual_score(r_img: np.ndarray, r_pe: np.ndarray) -> np.ndarray:
    """A_visual = 0.5 * (R_img + R_pe). CLIP text never enters this score."""
    a = 0.5 * (np.asarray(r_img, dtype=np.float64) + np.asarray(r_pe, dtype=np.float64))
    return a


# ---------------- subset construction (no mirror-statistic inputs) ----------------


def proportional_largest_remainder(cell_sizes: dict[str, int], target: int) -> dict[str, int]:
    """Exact `target` quota per cell by proportional largest remainder.

    Deterministic: ties on the fractional remainder break by cell name.
    """
    sizes = {str(k): int(v) for k, v in cell_sizes.items() if int(v) > 0}
    total = sum(sizes.values())
    if target < 0 or target > total:
        raise ValueError(f"target {target} outside [0, {total}]")
    raw = {c: sizes[c] * target / total for c in sizes}
    base = {c: int(np.floor(raw[c])) for c in sizes}
    remainder = target - sum(base.values())
    by_frac = sorted(sizes.keys(), key=lambda c: (-(raw[c] - base[c]), c))
    for c in by_frac[:remainder]:
        base[c] += 1
    return base


def geometry_stratified_select(
    scores: np.ndarray,
    strata: list[str],
    target: int,
) -> list[int]:
    """Select `target` pair indices preserving the geometry-cell proportions.

    Within a cell, selection is ascending by (score, pair_index). The selector
    accepts only scores + strata: it has no mirror-statistic parameter.
    """
    s = np.asarray(scores, dtype=np.float64)
    n = s.size
    if n != len(strata):
        raise ValueError("scores/strata length mismatch")
    cells: dict[str, list[int]] = {}
    for i, cell in enumerate(strata):
        cells.setdefault(str(cell), []).append(i)
    quotas = proportional_largest_remainder({c: len(v) for c, v in cells.items()}, target)
    chosen: list[int] = []
    for cell in sorted(cells.keys()):
        idx = sorted(cells[cell], key=lambda i: (s[i], i))
        chosen.extend(idx[: quotas[cell]])
    return sorted(chosen)


def global_lowest_select(scores: np.ndarray, target: int) -> list[int]:
    """The globally lowest-`target` scores, tie-broken by pair index."""
    s = np.asarray(scores, dtype=np.float64)
    order = sorted(range(s.size), key=lambda i: (s[i], i))
    return sorted(order[:target])


def lowest_index_set(values, k: int) -> set[int]:
    v = np.asarray(values, dtype=np.float64)
    order = sorted(range(v.size), key=lambda i: (v[i], i))
    return set(order[:k])


def joint_lowest_intersection(values_a, values_b, k: int) -> set[int]:
    """Intersection of the two per-metric lowest-`k` index sets."""
    return lowest_index_set(values_a, k) & lowest_index_set(values_b, k)


# ---------------- statistics ----------------


def paired_mean_difference(low_side, high_side) -> float:
    """mean(low - high) over paired values (the mirror point estimator)."""
    lo = np.asarray(low_side, dtype=np.float64)
    hi = np.asarray(high_side, dtype=np.float64)
    if lo.shape != hi.shape:
        raise ValueError("paired inputs must have equal shape")
    return float(np.mean(lo - hi))


def two_sample_mean_difference(group_a, group_b) -> float:
    """mean(group_a) - mean(group_b); unequal counts allowed (never pooled signs)."""
    a = np.asarray(group_a, dtype=np.float64)
    b = np.asarray(group_b, dtype=np.float64)
    if a.size == 0 or b.size == 0:
        raise ValueError("empty group")
    return float(np.mean(a) - np.mean(b))


def source_pair_bootstrap_ci(values, n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> dict:
    """Source-pair bootstrap CI. point = exact float64 observed mean.

    The resample matrix is derived from (seed, n) alone, so every statistic
    with the same n and seed shares one index matrix; the point is
    seed-invariant by construction.
    """
    v = np.asarray(values, dtype=np.float64)
    n = v.size
    if n == 0:
        raise ValueError("empty values")
    point = float(v.mean())
    if n < 2:
        return {
            "n": n,
            "n_boot": 0,
            "seed": int(seed),
            "point": point,
            "ci95": "SMALL_N",
            "note": "n<2: point only, no CI",
        }
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(int(n_boot), n))
    means = v[idx].mean(axis=1)
    lo, hi = (float(x) for x in np.percentile(means, [2.5, 97.5]))
    return {
        "n": n,
        "n_boot": int(n_boot),
        "seed": int(seed),
        "point": point,
        "ci95": [lo, hi],
    }


def percentile3(values, q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def spearman_rho(x, y) -> float:
    """Spearman rank correlation with average-rank tie handling."""
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    if a.shape != b.shape or a.size < 2:
        raise ValueError("x/y must be equal-length, n>=2")
    ra, rb = average_ranks(a), average_ranks(b)
    if ra.std() == 0.0 or rb.std() == 0.0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def quartile_order(scores, n_quartiles: int = 4) -> list[list[int]]:
    """Deterministic quartiles: sorted by (score, pair_index), consecutive chunks.

    With n divisible by 4 each chunk has exactly n/4 members; otherwise the
    surplus goes to the first chunks (stable, no randomness).
    """
    s = np.asarray(scores, dtype=np.float64)
    n = s.size
    order = sorted(range(n), key=lambda i: (s[i], i))
    base, extra = divmod(n, n_quartiles)
    chunks: list[list[int]] = []
    pos = 0
    for q in range(n_quartiles):
        size = base + (1 if q < extra else 0)
        chunks.append(order[pos : pos + size])
        pos += size
    return chunks


# ---------------- retention / attenuation (no clipping) ----------------


def retention(g_subset: dict, g_all: dict) -> float:
    """|point_subset| / |point_all| (0.0 when the reference point is 0)."""
    a = abs(float(g_all["point"]))
    if a == 0.0:
        return 0.0
    return abs(float(g_subset["point"])) / a


def attenuation(g_subset: dict, g_all: dict) -> float:
    """1 - retention; may be negative (never clipped)."""
    return 1.0 - retention(g_subset, g_all)


def ci_lower_above_zero(block: dict) -> bool:
    ci = block.get("ci95")
    if not isinstance(ci, (list, tuple)) or len(ci) != 2:
        return False
    return bool(ci[0] > 0.0)


# ---------------- classification (pre-registered decision tree) ----------------


def classify_directional_residual(g_v50: dict, g_v25: dict, retention_25: float) -> str:
    """CONFIRMED / WEAK / NOT_DETECTED / INCONCLUSIVE (spec section 30)."""
    ok50 = ci_lower_above_zero(g_v50)
    ok25 = ci_lower_above_zero(g_v25)
    if ok50 and ok25 and retention_25 >= 0.50:
        return "CONFIRMED"
    if ok50:
        return "WEAK"

    def _contains_zero(block: dict) -> bool:
        ci = block.get("ci95")
        return isinstance(ci, (list, tuple)) and len(ci) == 2 and ci[0] <= 0.0 <= ci[1]

    near0 = abs(float(g_v50["point"])) < 1e-4 and abs(float(g_v25["point"])) < 1e-4
    if (_contains_zero(g_v50) and _contains_zero(g_v25)) or near0:
        return "NOT_DETECTED"
    return "INCONCLUSIVE"


def classify_visual_contribution(
    ci_img: dict,
    ci_pe: dict,
    rho_img: float,
    rho_pe: float,
    atten_25: float,
) -> str:
    """STRONG_SUPPORTED / SUPPORTED / WEAK / NOT_SUPPORTED (spec section 31)."""
    a = ci_lower_above_zero(ci_img)
    b = ci_lower_above_zero(ci_pe)
    c = abs(float(rho_img)) >= 0.30
    d = abs(float(rho_pe)) >= 0.30
    e = float(atten_25) >= 0.50
    strong_hits = sum((a, b, c, d, e))
    if strong_hits >= 2:
        return "STRONG_SUPPORTED"
    max_rho = max(abs(float(rho_img)), abs(float(rho_pe)))
    if (a or b) or max_rho >= 0.20 or float(atten_25) >= 0.25:
        return "SUPPORTED"
    if (a or b):
        return "WEAK"
    return "NOT_SUPPORTED"


def classify_overall(residual: str, contribution: str) -> str:
    """Overall class A-E (spec section 32)."""
    if residual == "CONFIRMED":
        if contribution in ("STRONG_SUPPORTED", "SUPPORTED"):
            return "MIXED_VISUAL_CONTENT_AND_DIRECTIONAL"
        return "DIRECTIONAL_RESIDUAL_CONFIRMED"
    if residual != "CONFIRMED" and contribution == "STRONG_SUPPORTED":
        return "VISUAL_CONTENT_EXPLAINS_SUBSTANTIAL_GAP"
    if residual == "NOT_DETECTED" and contribution in ("WEAK", "NOT_SUPPORTED"):
        return "NO_DIRECTIONAL_RESIDUAL_DETECTED"
    return "INCONCLUSIVE"


def recommendation_for(overall: str) -> dict:
    """Recommendation mapping (spec section 33). Suggestion only; authorizes nothing."""
    mapping = {
        "DIRECTIONAL_RESIDUAL_CONFIRMED": "DESIGN_VERTICAL_MIRROR_BALANCED_SUPERVISION",
        "MIXED_VISUAL_CONTENT_AND_DIRECTIONAL": "DESIGN_VERTICAL_MIRROR_BALANCED_SUPERVISION_WITH_CONTENT_GUARDS",
        "VISUAL_CONTENT_EXPLAINS_SUBSTANTIAL_GAP": "REVIEW_CROP_CONTENT_SUPERVISION",
        "NO_DIRECTIONAL_RESIDUAL_DETECTED": "LONGER_P25_REVIEW_CANDIDATE",
        "INCONCLUSIVE": "NO_MORE_EXPOSURE_YET",
    }
    if overall not in mapping:
        raise ValueError(f"unknown overall class {overall}")
    return {
        "value": mapping[overall],
        "longer_p25_authorized": False,
        "p50_authorized": False,
    }


# ---------------- json / hashing helpers ----------------


def canon_json(obj) -> str:
    """Canonical JSON (sorted keys, compact separators) for stable hashing."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_of_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_of_file(path) -> str:
    return sha256_of_bytes(Path(path).read_bytes())


def verify_subset_doc(doc: dict) -> list[str]:
    """Structural checks for a frozen subset document (phase B entry gate)."""
    problems: list[str] = []
    if doc.get("schema_version") != SCHEMA_VERSION:
        problems.append(f"schema_version {doc.get('schema_version')!r} != {SCHEMA_VERSION!r}")
    if doc.get("pair_manifest_sha") != PAIR_MANIFEST_SHA:
        problems.append("pair_manifest_sha mismatch")
    if doc.get("n_pairs") != N_PAIRS:
        problems.append(f"n_pairs {doc.get('n_pairs')} != {N_PAIRS}")
    per = doc.get("per_pair")
    if not isinstance(per, dict) or sorted(int(k) for k in per) != list(range(N_PAIRS)):
        problems.append("per_pair must cover exactly 0..511")
    subs = doc.get("subsets")
    if not isinstance(subs, dict):
        problems.append("subsets missing")
        return problems
    for name in SUBSET_NAMES:
        idx = subs.get(name)
        if not isinstance(idx, list):
            problems.append(f"subset {name} missing")
            continue
        if len(set(idx)) != len(idx):
            problems.append(f"subset {name} has duplicate indices")
        if any(int(i) < 0 or int(i) >= N_PAIRS for i in idx):
            problems.append(f"subset {name} out of range")
    return problems


__all__ = [
    "BASE_SHA",
    "BOOT_SEED",
    "N_BOOT",
    "N_PAIRS",
    "PAIR_MANIFEST_SHA",
    "SCHEMA_VERSION",
    "SUBSET_NAMES",
    "TRANS_KEYS",
    "V25_TARGET",
    "V50_TARGET",
    "attenuation",
    "average_ranks",
    "canon_json",
    "ci_lower_above_zero",
    "classify_directional_residual",
    "classify_overall",
    "classify_visual_contribution",
    "composite_visual_score",
    "geometry_stratified_select",
    "geometry_stratum",
    "global_lowest_select",
    "joint_lowest_intersection",
    "latent_shift_band",
    "lowest_index_set",
    "paired_mean_difference",
    "percentile3",
    "percentile_rank_average_ties",
    "proportional_largest_remainder",
    "quartile_order",
    "recommendation_for",
    "retention",
    "sha256_of_bytes",
    "sha256_of_file",
    "source_pair_bootstrap_ci",
    "spearman_rho",
    "two_sample_mean_difference",
    "verify_subset_doc",
    "zoom_band",
]
