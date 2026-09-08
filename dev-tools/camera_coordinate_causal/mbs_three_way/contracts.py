"""MBS Three-Way Same-Source Causal Audit — new contracts (pure numpy + stdlib).

NEW companion package to the FROZEN Vertical Bottom Supervision (VBS) audit
package (dev-tools/camera_coordinate_causal/vertical_bottom_supervision/).

Why this package exists
-----------------------
The frozen VBS scoring tool hard-codes the OLD audit's checkpoints
(PRE U116100 / MID U117100 / POST U118100).  The three-way audit requires
PRE U118100 / CONTROL U118200 (g1_camera_v2_p25_mirror) / TREATMENT U118200
(g1_camera_v2_p25_mirror_v2).  The frozen package is NOT edited; it is
imported and reused for every checkpoint-independent contract (arms,
margins, floors, SAME correction, bootstrap, noise/timestep seeds,
geometry helpers).  No torch import at module load.

Binding scientific wording (audit spec s13)
-------------------------------------------
Historical CONTROL:
    SAME-DISTRIBUTION INDEPENDENT-SEQUENCE CAMERA-ONLY CONTINUATION CONTROL
    (hash-randomized order, zero mirror pairs).  It is NOT exact
    matched-data.
Treatment:
    corrected MBS 100U, canonical cycle-0 sequence.
Therefore TREATMENT - CONTROL is an intervention-associated contrast under
a common frozen evaluation cohort; training-sequence variation remains a
potential confound.  Never report an "exact matched-data effect" unless a
future canonical camera-only control is actually run.

Frozen references (committed on base 1b8fce4)
---------------------------------------------
  reports/camera-vertical-bottom-supervision-audit.md/.json
  reports/camera-vertical-bottom-supervision-pairs.csv
  reports/camera-vertical-bottom-supervision-metrics.json
  frozen pair manifest sha256 (original): 7f3a81b2...
  frozen t strata / noise rule / seed 20260907 / n_boot 10000
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Frozen VBS package import (read-only reuse; the file is never modified).
# Loaded from an explicit path under a DISTINCT module name: a plain
# `import contracts` here would resolve to THIS module (same base name,
# present in sys.modules mid-load) and silently shadow the frozen file.
# ---------------------------------------------------------------------------
_VBS_DIR = Path(__file__).resolve().parent.parent / "vertical_bottom_supervision"
_VBS_CONTRACTS = _VBS_DIR / "contracts.py"
_spec = importlib.util.spec_from_file_location("vbs_contracts_frozen", _VBS_CONTRACTS)
assert _spec is not None and _spec.loader is not None, f"frozen VBS contracts missing: {_VBS_CONTRACTS}"
VBS = importlib.util.module_from_spec(_spec)  # type: ignore[assignment]
sys.modules["vbs_contracts_frozen"] = VBS  # type: ignore[assignment]
_spec.loader.exec_module(VBS)  # type: ignore[union-attr]
if not hasattr(VBS, "MASTER_SEED_VBS"):
    raise RuntimeError("frozen VBS contracts failed to load: MASTER_SEED_VBS missing")

# ---------------- re-exported frozen contracts (checkpoint-independent) ----
MASTER_SEED_VBS: int = VBS.MASTER_SEED_VBS
BOOT_SEED: int = VBS.BOOT_SEED
N_BOOT: int = VBS.N_BOOT
T_QUANTILES: tuple[float, float, float, float] = VBS.T_QUANTILES
VIEWPORT: int = VBS.VIEWPORT
LATENT_TOKENS: int = VBS.LATENT_TOKENS
VAE_SCALE: int = VBS.VAE_SCALE
T_EPS: float = VBS.T_EPS
NOISE_OBSERVATION_BOUNDARY: float = VBS.NOISE_OBSERVATION_BOUNDARY
NOISE_SCALE: float = VBS.NOISE_SCALE
ARMS: tuple[str, ...] = VBS.ARMS
ARMS_NO_ID: tuple[str, ...] = VBS.ARMS_NO_ID
ID_ARMS_MAX_S_PER_FORWARD: float = VBS.ID_ARMS_MAX_S_PER_FORWARD

t_values = VBS.t_values
noise_seed = VBS.noise_seed
arm_side = VBS.arm_side
arm_coord = VBS.arm_coord
margin_from_losses = VBS.margin_from_losses
floor_from_losses = VBS.floor_from_losses
adjusted_delta = VBS.adjusted_delta
paired_gap = VBS.paired_gap
bootstrap_ci = VBS.bootstrap_ci
latent_shift_band = VBS.latent_shift_band
zoom_band = VBS.zoom_band
tertile_of = VBS.tertile_of
write_frozen = VBS.write_frozen
sha256_file = VBS.sha256_file
full_height_from_zoom = VBS.full_height_from_zoom
signed_shift_of = VBS.signed_shift_of

# ---------------- audit identity (this round) ------------------------------

BASE_SHA = "1b8fce49031b854b835d9f47ea65a3f77b83a2f9"
BRANCH = "camera-v2-mbs-three-way-same-source-audit-review"
WORKTREE = Path("/sakuramoon-runtime/sakuramoon-camera-mbs-three-way-audit")
# Original causal-audit code+config worktree (created at 34f646ab, whose
# src/tests/config content is byte-identical to the prior audit's entrance
# head b2443af4 for src/ and config/; only tests were added).
C2_REPO = Path("/sakuramoon-runtime/sakuramoon-camera-v2-c2")
C2_HEAD = "34f646abdb1e64f45dfddd47cf7e8b9247a7a979"
RUNTIME_ROOT = Path("/sakuramoon-runtime")
VENV_PY = "/sakuramoon-runtime/sakuramoon-dtk-venv/bin/python"
# Spec-mandated stable runtime roots (raw artifacts live under /tmp by
# design and are NEVER committed); S108 is not applicable to these.
ENV_SH = Path("/tmp/camera-mbs-three-way-audit/env-cc.sh")

# Raw artifact roots (never committed):
AUDIT_ROOT = Path("/tmp/camera-mbs-three-way-audit")   # NEW raw audit artifacts
VBS_ROOT = Path("/tmp/camera-vertical-bottom")         # frozen VBS runtime artifacts (pair manifest, latents, sources, anchor-parity)
CC_ROOT = Path("/tmp/camera-coordinate-causal")        # stage1 units (reconstructed; canonical frozen location)
SHARD_ROOT = RUNTIME_ROOT / "data" / "validation-cohorts" / "s0-validation-50k-v1" / "shards"

CKPT_PATHS: dict[str, Path] = {
    "PRE": RUNTIME_ROOT / "output_model" / "g1_camera_v2_p25" / "ckpt_118100_raw-118100-update-cadence",
    "CONTROL": RUNTIME_ROOT / "output_model" / "g1_camera_v2_p25_mirror" / "ckpt_118200_raw-118200-update-cadence",
    "TREATMENT": RUNTIME_ROOT / "output_model" / "g1_camera_v2_p25_mirror_v2" / "ckpt_118200_raw-118200-update-cadence",
}
CKPT_UPDATES: dict[str, int] = {"PRE": 118100, "CONTROL": 118200, "TREATMENT": 118200}
CKS: tuple[str, str, str] = ("PRE", "CONTROL", "TREATMENT")

# Frozen committed references
FROZEN_PAIR_MANIFEST_SHA = "7f3a81b248473d0b0591377c1bcd9f7df353066c9716836dabe90d9861f56297"
FROZEN_PAIRS_CSV = WORKTREE / "reports" / "camera-vertical-bottom-supervision-pairs.csv"
FROZEN_AUDIT_JSON = WORKTREE / "reports" / "camera-vertical-bottom-supervision-audit.json"
FROZEN_AUDIT_MD = WORKTREE / "reports" / "camera-vertical-bottom-supervision-audit.md"
FROZEN_METRICS_JSON = WORKTREE / "reports" / "camera-vertical-bottom-supervision-metrics.json"

# New-output report paths (the ONLY new files committed, besides tooling)
REPORT_MD = WORKTREE / "reports" / "camera-mbs-three-way-same-source-audit.md"
REPORT_JSON = WORKTREE / "reports" / "camera-mbs-three-way-same-source-audit.json"
REPORT_METRICS = WORKTREE / "reports" / "camera-mbs-three-way-same-source-metrics.json"
REPORT_PAIRS_CSV = WORKTREE / "reports" / "camera-mbs-three-way-same-source-pairs.csv"

# ---------------- frozen checkpoint identities (verified 09-08 pre-scoring) --
# PRE and CONTROL: manifest sha256 + model tree sha256.  TREATMENT: manifest +
# model tree plus the FULL checkpoint-tree sha256 (model + train_state +
# COMPLETE).  Tree hashes use the evidence-lineage canonical algorithm (see
# _tree_sha256): sha256 over the sorted sha256sum lines with ABSOLUTE path
# strings, i.e. `find <abs-root> -type f | sort | xargs sha256sum | sha256sum`.
EXPECTED_IDENTITY: dict[str, dict[str, str]] = {
    "PRE": {
        "manifest_sha256": "821a8c12546a094865972fef0c23152e9754e01f82eed8353f7248b92076b892",
        "model_tree_sha256": "32b76aff83a92b26fa1313cf100afeb8ec920b2b11920bf9319c14e2293da370",
    },
    "CONTROL": {
        "manifest_sha256": "f014827fb441aca8780a5bb710eeb2631a40aac112ed3aa6b07b6e3b65d657f7",
        "model_tree_sha256": "5347def81c6c8627b66861159890aaf23eb8fca61a87ed3c544c1a3b07d3c241",
    },
    "TREATMENT": {
        "manifest_sha256": "11deada823da8217a08cd15f4c0365c391f01c60af8335d89c07a9ea61d69824",
        "model_tree_sha256": "22a802f22987a6797db2886a8fd9db717ee2ce92cc0dbc699f00adc5a780fb98",
        "full_tree_sha256": "d2673914eba014ca389178f9bff191df4db31431b5de373ecabbfdfed4206c74",
    },
}

# ---------------- cohorts (pre-registered BEFORE seeing results) -----------
# Memberships come ONLY from the frozen pair cohort (no post-selection):
#   latent_shift = |signed_shift_start| / VAE_SCALE (production scale)
#   latent band via frozen latent_shift_band (lt2 / 2to4 / ge4)
#   targeted (co-primary B) = latent_shift >= 2  ==  bands {2to4, ge4}
#   content-balanced = frozen in_balanced_subset flag (checkpoint-independent,
#     pre-registered in the prior audit; re-derived here only for the record)
COHORT_ALL = "all"
COHORT_TARGETED = "latent_ge2"
COHORT_LT2 = "latent_lt2"
COHORT_2TO4 = "latent_2to4"
COHORT_GE4 = "latent_ge4"
COHORT_BALANCED = "content_balanced"
COHORTS: tuple[str, ...] = (
    COHORT_ALL, COHORT_TARGETED, COHORT_LT2, COHORT_2TO4, COHORT_GE4, COHORT_BALANCED,
)
EXPECTED_COHORT_SIZES = {
    COHORT_ALL: 512,
    COHORT_TARGETED: 312,
    COHORT_LT2: 200,
    COHORT_2TO4: 275,
    COHORT_GE4: 37,
    COHORT_BALANCED: 256,
}


def latent_shift_of(signed_shift_start: float) -> float:
    """Production latent shift = |pixel_shift| / VAE_SCALE (16)."""
    return abs(float(signed_shift_start)) / float(VAE_SCALE)


def pair_cohort_flags(row: dict[str, Any]) -> dict[str, Any]:
    """Cohort membership for one pair row (CSV or manifest geometry)."""
    ls = latent_shift_of(row["signed_shift_start"])
    band = latent_shift_band(ls)
    in_balanced = bool(row.get("in_balanced_subset", False))
    return {
        "latent_shift": ls,
        "latent_band": band,
        "in_targeted": band in ("2to4", "ge4"),
        "in_balanced": in_balanced,
    }


# ---------------- frozen CSV helpers ---------------------------------------
CSV_NUMERIC_COLS = (
    "zoom_band", "k_start", "k_end", "available", "full_height",
    "signed_shift_start", "content_asymmetry",
)


def parse_frozen_pairs_csv(path: Path) -> list[dict[str, Any]]:
    """Parse the committed frozen VBS pairs CSV (512 rows) with typed values."""
    import csv

    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8", newline="") as fh:
        for rec in csv.DictReader(fh):
            row: dict[str, Any] = {
                "pair_id": int(rec["pair_id"]),
                "source_shard": rec["source_shard"],
                "sample_id": int(rec["sample_id"]),
                "original_side": rec["original_side"],
                "zoom_band": rec["zoom_band"],
                "k_start": int(rec["k_start"]),
                "k_end": int(rec["k_end"]),
                "available": int(rec["available"]),
                "full_height": int(rec["full_height"]),
                "signed_shift_start": float(rec["signed_shift_start"]),
                "M_TOP_PRE_frozen": float(rec["M_TOP_PRE"]),
                "M_TOP_MID_frozen": float(rec["M_TOP_MID"]),
                "M_TOP_POST_frozen": float(rec["M_TOP_POST"]),
                "M_BOT_PRE_frozen": float(rec["M_BOT_PRE"]),
                "M_BOT_MID_frozen": float(rec["M_BOT_MID"]),
                "M_BOT_POST_frozen": float(rec["M_BOT_POST"]),
                "G_PRE_MID_frozen": float(rec["G_PRE_MID"]),
                "G_MID_POST_frozen": float(rec["G_MID_POST"]),
                "G_PRE_POST_frozen": float(rec["G_PRE_POST"]),
                "content_asymmetry": float(rec["content_asymmetry"]),
                "in_balanced_subset": rec["in_balanced_subset"].strip().lower() == "true",
            }
            rows.append(row)
    rows.sort(key=lambda r: r["pair_id"])
    return rows


# ---------------- replay gate (pre-score; spec s7) --------------------------
# The NEW PRE checkpoint (U118100) was the POST checkpoint of the frozen VBS
# audit, whose per-pair M values are committed in the frozen pairs CSV.
# Derivation of the allowable replay scale (PRE-REGISTERED, before any
# CONTROL/TREATMENT interpretation):
#   * The frozen audit's committed numerics floor (SAME duplicate forwards,
#     POST checkpoint) is aggregate mean +/- bootstrap CI95:
#         f_top mean 1.3151e-07 CI95 [-3.10e-07, +6.06e-07]
#         f_bot mean 1.4765e-07 CI95 [-3.79e-07, +6.74e-07]
#   * bound = REPLAY_MULTIPLIER * max(ci95_upper_top, ci95_upper_bot)
#     = 5.0 * 6.7397e-07 ~= 3.37e-06.
#   * Rationale: a 5x margin over the aggregate floor CI95 upper bound covers
#     the per-pair spread of the SAME-duplicate (numerics floor) noise while
#     remaining more than two orders of magnitude below the smallest reported
#     directional effect in the frozen audit (|G| stratum points >= 2.9e-04).
#     No looser or arbitrary tolerance is used; the bound is computed from
#     the committed floor evidence at runtime (not hard-coded to a number).
REPLAY_MULTIPLIER: float = 5.0


def replay_bounds_from_floor(floor_doc: dict[str, Any]) -> dict[str, float]:
    """Compute the replay gate bound from the frozen SAME-floor evidence.

    floor_doc = frozen audit.json["numerics_floor"]["POST"]:
      {"f_top": {"mean":..., "ci95":[lo,hi]}, "f_bot": {...}}
    """
    hi_top = float(floor_doc["f_top"]["ci95"][1])
    hi_bot = float(floor_doc["f_bot"]["ci95"][1])
    return {
        "multiplier": REPLAY_MULTIPLIER,
        "floor_ci95_upper_top": hi_top,
        "floor_ci95_upper_bot": hi_bot,
        "bound": REPLAY_MULTIPLIER * max(hi_top, hi_bot),
        "floor_mean_top": float(floor_doc["f_top"]["mean"]),
        "floor_mean_bot": float(floor_doc["f_bot"]["mean"]),
    }


def replay_gate_stats(
    per_pair: Sequence[dict[str, float]],
    bounds: dict[str, float],
    new_points: dict[str, float] | None = None,
    frozen_points: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Replay gate statistics + PASS/FAIL against the pre-registered bound.

    per_pair: [{"pair_index": i, "delta_top": d, "delta_bot": d}, ...]
      where delta = new PRE (U118100) per-pair 4-stratum mean margin minus
      the frozen VBS POST (U118100) per-pair value from the committed CSV.
    new_points / frozen_points: optional {"M_TOP":..., "M_BOTTOM":...}
      aggregate point comparison (same bound applies).
    PASS iff maxabs <= bound AND every aggregate point delta <= bound.
    """
    deltas = [
        float(p["delta_top"]) for p in per_pair
    ] + [float(p["delta_bot"]) for p in per_pair]
    a = np.asarray(deltas, dtype=np.float64)
    if a.size == 0:
        raise ValueError("empty replay delta set")
    maxabs = float(np.max(np.abs(a)))
    rms = float(np.sqrt(np.mean(a * a)))
    mean_delta = float(np.mean(a))
    bound = float(bounds["bound"])
    checks: list[dict[str, Any]] = [
        {"name": "maxabs_per_pair", "value": maxabs, "bound": bound, "pass": maxabs <= bound},
        {"name": "rms_per_pair", "value": rms, "bound": bound, "pass": rms <= bound},
    ]
    if new_points is not None and frozen_points is not None:
        for key in ("M_TOP", "M_BOTTOM"):
            d = float(new_points[key]) - float(frozen_points[key])
            checks.append(
                {"name": f"point_delta_{key}", "value": d, "bound": bound, "pass": abs(d) <= bound}
            )
    ok = all(c["pass"] for c in checks)
    return {
        "status": "PASS" if ok else "FAIL",
        "maxabs": maxabs,
        "rms": rms,
        "mean_delta": mean_delta,
        "bound": bound,
        "derivation": (
            f"{REPLAY_MULTIPLIER:.1f} x max(frozen POST SAME-floor CI95 upper "
            "bounds) computed from committed audit.json numerics_floor"
        ),
        "checks": checks,
        "n_pairs": len(per_pair),
    }


# ---------------- classification (spec s12, pre-registered) -----------------
def classify_contrast(
    i_bottom_ci: Sequence[float], i_g_ci: Sequence[float]
) -> str:
    """TARGETED-cohort classification from the two 95% CIs.

    CLEAR_DIRECTIONAL_CORRECTION iff CI95(I_BOTTOM).lower > 0 AND CI95(I_G).upper < 0.
    CLEAR_HARM iff CI95(I_BOTTOM).upper < 0 OR CI95(I_G).lower > 0.
    Otherwise BORDERLINE.
    """
    if float(i_bottom_ci[0]) > 0.0 and float(i_g_ci[1]) < 0.0:
        return "CLEAR_DIRECTIONAL_CORRECTION"
    if float(i_bottom_ci[1]) < 0.0 or float(i_g_ci[0]) > 0.0:
        return "CLEAR_HARM"
    return "BORDERLINE"


def canonical_control_recommendation(verdict: str) -> dict[str, str]:
    """Spec s12 mapping of the verdict to the canonical-control decision.

    The recommendation is a LATER USER REVIEW item only; this audit MUST NOT
    start any training.
    """
    if verdict == "CLEAR_DIRECTIONAL_CORRECTION":
        return {
            "action": "NOT_REQUIRED_NOW",
            "note": "canonical camera-only 100U NOT REQUIRED now (supported directional correction)",
        }
    if verdict == "CLEAR_HARM":
        return {
            "action": "NOT_REQUIRED_NOW",
            "note": "canonical camera-only 100U NOT REQUIRED now; MBS strategy requires review",
        }
    if verdict == "BORDERLINE":
        return {
            "action": "RECOMMENDED_FOR_LATER_USER_REVIEW",
            "note": (
                "canonical camera-only matched-sequence 100U RECOMMENDED FOR "
                "LATER USER REVIEW (this audit does not start it)"
            ),
        }
    raise ValueError(f"unknown verdict {verdict!r}")


# ---------------- checkpoint identity gate (this audit's own gate) ----------
def _tree_sha256(root: Path) -> str:
    """Canonical audit tree hash (evidence-lineage algorithm).

    sha256 over the sorted GNU ``sha256sum`` output lines for every file under
    ``root``, using ABSOLUTE path strings and the text-mode two-space
    separator.  Byte-identical to the shell pipeline used for the RERUN #2
    terminal full-tree digest d2673914...:

        find <abs-root> -type f | sort | xargs sha256sum | sha256sum

    Note: the frozen score_pairs._tree_sha256 is a DIFFERENT algorithm
    (relative-path JSON payload); it is not used for checkpoint identity.
    """
    root = root.resolve()
    files = sorted((p for p in root.rglob("*") if p.is_file()), key=str)
    lines = bytearray()
    for p in files:
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        lines += f"{digest}  {p}\n".encode()
    return hashlib.sha256(bytes(lines)).hexdigest()


def verify_checkpoint_identity(ck: str, *, alpha: float | None = None) -> dict[str, Any]:
    """Gate one frozen checkpoint against the audit's expected identities.

    Checks (all must pass):
      * COMPLETE marker == b"complete\\n"
      * manifest sha256 == EXPECTED_IDENTITY[ck]["manifest_sha256"]
      * model tree sha256 == EXPECTED_IDENTITY[ck]["model_tree_sha256"]
      * (TREATMENT only) full checkpoint tree sha256 == expected
      * train_state update identity == CKPT_UPDATES[ck]
      * growth alpha == 1.0 when provided
    Returns {"status": "PASS"/"FAIL", "problems": [...], "evidence": {...}}.
    """
    ckpt = CKPT_PATHS[ck]
    expected = EXPECTED_IDENTITY[ck]
    problems: list[str] = []
    evidence: dict[str, Any] = {"path": str(ckpt)}
    if not ckpt.is_dir():
        return {"status": "FAIL", "problems": [f"checkpoint dir missing: {ckpt}"], "evidence": evidence}
    complete = (ckpt / "COMPLETE").read_bytes()
    evidence["complete_marker"] = complete.decode("utf-8", "replace").strip()
    if complete != b"complete\n":
        problems.append("COMPLETE marker invalid")
    manifest_sha = sha256_file(ckpt / "manifest.json")
    evidence["manifest_sha256"] = manifest_sha
    if manifest_sha != expected["manifest_sha256"]:
        problems.append("manifest sha mismatch")
    tree = _tree_sha256(ckpt / "model")
    evidence["model_tree_sha256"] = tree
    if tree != expected["model_tree_sha256"]:
        problems.append("model tree sha mismatch")
    if "full_tree_sha256" in expected:
        full = _tree_sha256(ckpt)
        evidence["full_tree_sha256"] = full
        if full != expected["full_tree_sha256"]:
            problems.append("full checkpoint tree sha mismatch")
    try:
        ts = json.loads((ckpt / "train_state" / "trainer_state.json").read_text(encoding="utf-8"))
        upd = int(ts.get("successful_updates", -1))
        evidence["successful_updates"] = upd
        if upd != CKPT_UPDATES[ck]:
            problems.append(f"update {upd} != {CKPT_UPDATES[ck]}")
        gs = json.loads((ckpt / "train_state" / "growth_state.json").read_text(encoding="utf-8"))
        evidence["growth_alpha"] = gs.get("alpha")
        if gs.get("alpha") != 1.0:
            problems.append(f"growth alpha {gs.get('alpha')} != 1.0")
        if alpha is not None and float(alpha) != 1.0:
            problems.append(f"resolved inference alpha {alpha} != 1.0")
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        problems.append(f"train_state unreadable: {exc!r}")
    return {
        "status": "PASS" if not problems else "FAIL",
        "problems": problems,
        "evidence": evidence,
    }


__all__ = [
    "ARMS",
    "ARMS_NO_ID",
    "AUDIT_ROOT",
    "BASE_SHA",
    "BOOT_SEED",
    "BRANCH",
    "C2_HEAD",
    "C2_REPO",
    "CC_ROOT",
    "CKPT_PATHS",
    "CKPT_UPDATES",
    "CKS",
    "COHORTS",
    "COHORT_2TO4",
    "COHORT_ALL",
    "COHORT_BALANCED",
    "COHORT_GE4",
    "COHORT_LT2",
    "COHORT_TARGETED",
    "ENV_SH",
    "EXPECTED_COHORT_SIZES",
    "EXPECTED_IDENTITY",
    "FROZEN_AUDIT_JSON",
    "FROZEN_AUDIT_MD",
    "FROZEN_METRICS_JSON",
    "FROZEN_PAIRS_CSV",
    "FROZEN_PAIR_MANIFEST_SHA",
    "ID_ARMS_MAX_S_PER_FORWARD",
    "LATENT_TOKENS",
    "MASTER_SEED_VBS",
    "NOISE_OBSERVATION_BOUNDARY",
    "NOISE_SCALE",
    "N_BOOT",
    "REPLAY_MULTIPLIER",
    "REPORT_JSON",
    "REPORT_MD",
    "REPORT_METRICS",
    "REPORT_PAIRS_CSV",
    "RUNTIME_ROOT",
    "SHARD_ROOT",
    "T_EPS",
    "T_QUANTILES",
    "VAE_SCALE",
    "VBS",
    "VBS_ROOT",
    "VENV_PY",
    "WORKTREE",
    "adjusted_delta",
    "arm_coord",
    "arm_side",
    "bootstrap_ci",
    "canonical_control_recommendation",
    "classify_contrast",
    "floor_from_losses",
    "full_height_from_zoom",
    "latent_shift_band",
    "margin_from_losses",
    "noise_seed",
    "pair_cohort_flags",
    "paired_gap",
    "parse_frozen_pairs_csv",
    "replay_bounds_from_floor",
    "replay_gate_stats",
    "sha256_file",
    "signed_shift_of",
    "t_values",
    "tertile_of",
    "verify_checkpoint_identity",
    "write_frozen",
    "zoom_band",
]
