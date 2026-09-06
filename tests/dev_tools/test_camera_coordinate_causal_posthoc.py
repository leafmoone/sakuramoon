"""Posthoc review contracts - regression tests.

Locks the posthoc reviewer semantics (section 27 of the posthoc review
prompt) against dev-tools/camera_coordinate_causal/posthoc_contracts.py
and pins the historical forensic surface (final snapshot + historical
reports) so the posthoc pass can never drift the evidence it corrects.

All tests are pure (numpy + stdlib); none of them need the 34 GB evidence
store or the HCU. 0 skip / 0 xfail by design.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DEV_DIR = REPO_ROOT / "dev-tools" / "camera_coordinate_causal"
SNAPSHOT_DIR = DEV_DIR / "final_snapshot"
REPORTS_DIR = REPO_ROOT / "reports"
sys.path.insert(0, str(DEV_DIR))

import posthoc_contracts as pc


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------- forensic surface pinning (J, K) ----------------------------

EXPECTED_SNAPSHOT_SHA256 = {
    "cc_common.py": "63edc8ef5a62fcb1e64c202f9f12d562713512e44b618e49a719fe0527aa437b",
    "cc_stage1.py": "384d58e49b3c08b345525aded9485c1bb38c310d426d399aa07acf6182aa65b0",
    "cc_stage2.py": "e5b7ae910df0db7847fcb96d9ecfddf12ce394e383045750da695c62c7c664b9",
    "cc_stage2b.py": "aea987d6505338b81a7e9d72084e5bef6a458b04cf41454750f347c779da2177",
    "cc_stage3.py": "4a855b3134b78f4478003cbdd611945b9a4e1c8e53e95827ec0cc28ea27657a6",
}

EXPECTED_CAUSAL_REPORT_SHA256 = {
    "camera-coordinate-causal-audit.md": "1a3f299c43ed1a2ef32d8ec651a4bd0d7201f3b004542777ecda1dc706965b70",
    "camera-coordinate-causal-audit.json": "ccd33d348eac1ce8c66a7eedcdc726aee6fa07a3fe947f0b169014c438631ca0",
    "camera-coordinate-causal-metrics.json": "93aac18d30cb14c3fb9db464a81103407402cf0f1198920baa45d0e2dfcc7a99",
    "camera-coordinate-causal-units.csv": "763bc3df684ac1df409a1dac91eb1ce0f9d026a39e16ea171c5bbe2b18ffffd0",
    "camera-coordinate-causal-copy-report.md": "d55280c4e6c6d9deefc02e530a529282b348110357a28871a149c179207f93da",
}

EXPECTED_EXPANDED_REPORT_SHA256 = {
    "camera-p25-expanded-effectiveness-audit.md": "2cf75a17cef217085c1f3bd1af58c53cbd2dcbd27c5f8d7a47fca6e8f6e4efd1",
    "camera-p25-expanded-effectiveness-audit.json": "267088a5db5c88b35a0ced746045d94f76cbbfee464e4c65b4b602f175aa4e88",
    "camera-p25-expanded-effectiveness-metrics.json": "6de050858a1625c0344600a337c5d8c1cbb591d92bd6a09e658b6ac48c6f401e",
    "camera-p25-expanded-effectiveness-points.csv": "6da4d3e21a2cccb8094b97dc464c2f70b97c9757d62eaec8c7fee10eac4d8af5",
    "camera-p25-expanded-effectiveness-prompts.csv": "0ba1b77245eb5285b619429765d0131b720b8ba8f73a2d183ba46bdeb9ed5113",
    "camera-p25-expanded-effectiveness-copy-report.md": "b3540d840d1a97de61e0efe836900b84d762505b4230a8eedcde540b367045f1",
}

CODE_MANIFEST_SHA256 = "68787b40b4add71fd014e00b6341ba3bb35be689903e91027179c467f877982b"


def test_final_snapshot_hashes_unchanged():
    for name, want in EXPECTED_SNAPSHOT_SHA256.items():
        p = SNAPSHOT_DIR / name
        assert p.is_file(), f"missing final snapshot: {name}"
        assert _sha256(p) == want, f"final_snapshot/{name} drifted"


def test_historical_causal_report_hashes_unchanged():
    for name, want in EXPECTED_CAUSAL_REPORT_SHA256.items():
        p = REPORTS_DIR / name
        assert p.is_file(), f"missing historical report: {name}"
        assert _sha256(p) == want, f"historical report {name} drifted"


def test_historical_expanded_report_hashes_unchanged():
    for name, want in EXPECTED_EXPANDED_REPORT_SHA256.items():
        p = REPORTS_DIR / name
        assert p.is_file(), f"missing expanded report: {name}"
        assert _sha256(p) == want, f"expanded report {name} drifted"


def test_code_manifest_pinned_and_consistent():
    p = REPORTS_DIR / "camera-coordinate-causal-code-manifest.json"
    assert _sha256(p) == CODE_MANIFEST_SHA256, "code manifest drifted"
    m = json.loads(p.read_text(encoding="utf-8"))
    for e in m["final_scripts"]:
        name = e["tracked_path"].rsplit("/", 1)[-1]
        assert e["sha256"] == EXPECTED_SNAPSHOT_SHA256[name]
    for e in m["causal_reports"]:
        assert e["sha256"] == EXPECTED_CAUSAL_REPORT_SHA256[e["name"]]
    for e in m["expanded_effectiveness_reports"]:
        assert e["sha256"] == EXPECTED_EXPANDED_REPORT_SHA256[e["name"]]


def test_readme_erratum_allowed_but_reports_locked():
    """README.md is the ONE file allowed to change (erratum); it must not be
    treated as a locked historical report."""
    assert "README.md" not in EXPECTED_CAUSAL_REPORT_SHA256
    assert "README.md" not in EXPECTED_EXPANDED_REPORT_SHA256


# ---------------- (A) SAME margin sign ---------------------------------------


def test_same_margin_sign_semantics():
    same = np.array([1.000001, 2.0, 3.0])
    correct = np.array([1.0, 2.000001, 3.0])
    d = pc.same_margin(same, correct)
    assert d[0] > 0  # SAME higher loss -> positive
    assert d[1] < 0  # SAME lower loss -> negative
    assert np.isclose(d[2], 0.0)
    with pytest.raises(ValueError):
        pc.same_margin(same, correct[:2])


def test_arm_margin_sign_convention():
    """M > 0 means the wrong/alternative coordinates have HIGHER loss, i.e.
    the correct coordinates are preferred (the historical README prose said
    the opposite; the implementation was already correct)."""
    arm = np.array([1.001, 1.0])
    correct = np.array([1.0, 1.001])
    d = pc.arm_margin(arm, correct)
    assert d[0] > 0 and d[1] < 0
    assert np.isclose(d[0], 1e-3)


# ---------------- (B) difference-of-differences ------------------------------


def test_diff_in_diff_exact():
    n = 12
    s = np.linspace(-2e-4, 3e-4, n)  # per-unit level, identical across checkpoints
    d0 = 1.234e-4
    same_m = {"PRE": s, "MID": s + 5e-5, "POST": s + 1.5e-4}
    # arm margins: SAME + d0 starting at MID, so every arm delta exceeds the
    # same delta by EXACTLY d0 per unit (the d0 must survive the subtraction
    # because it enters only at the arm side, not per-checkpoint symmetrically)
    arm_m = {"PRE": same_m["PRE"], "MID": same_m["MID"] + d0, "POST": same_m["POST"] + d0}
    out = pc.diff_in_diff(arm_m, same_m, np.arange(n), n_boot=2000, seed=7)
    # intervals telescope: POST_PRE = MID_PRE + POST_MID. d0 enters at MID and
    # POST on the arm side, so it cancels inside POST_MID (both endpoints carry
    # d0) and survives in MID_PRE and POST_PRE.
    for k, same_delta, adj_want in (
        ("MID_PRE", 5e-5, d0),
        ("POST_MID", 1e-4, 0.0),
        ("POST_PRE", 1.5e-4, d0),
    ):
        adj = out["intervals"][k]["adjusted"]
        assert np.isclose(adj["point"], adj_want, atol=1e-15)
        assert np.isclose(adj["ci95"][0], adj_want, atol=1e-15)
        assert np.isclose(adj["ci95"][1], adj_want, atol=1e-15)
        rs = out["intervals"][k]["raw_same"]["point"]
        assert np.isclose(rs, same_delta)
        ra = out["intervals"][k]["raw_arm"]["point"]
        assert np.isclose(ra, same_delta + adj_want)
    assert out["n"] == n


# ---------------- (C) paired bootstrap shared indices ------------------------


def test_paired_bootstrap_same_replicate_indices():
    """If the SAME and arm per-unit POST-PRE deltas differ by an EXACT
    per-unit-constant D (arm_delta = same_delta + D), the adjusted effect
    must collapse to a POINT (CI width 0) - only possible when one resample
    matrix drives both sides of the subtraction on every replicate."""
    n = 16
    s_pre = np.linspace(-1e-4, 2e-4, n)
    s_post = s_pre + np.linspace(1e-5, 3e-5, n)
    D = 5e-4
    same_m = {"PRE": s_pre, "MID": s_pre, "POST": s_post}
    arm_m = {"PRE": s_pre, "MID": s_pre, "POST": s_post + D}
    out = pc.diff_in_diff(arm_m, same_m, np.arange(n), n_boot=3000, seed=11)
    adj = out["intervals"]["POST_PRE"]["adjusted"]
    assert np.isclose(adj["point"], D, atol=1e-15)
    assert np.isclose(adj["ci95"][0], D, atol=1e-15)
    assert np.isclose(adj["ci95"][1], D, atol=1e-15)
    # raw SAME delta must be the plain mean of the per-unit same deltas
    want_raw = float(np.mean(s_post - s_pre))
    assert np.isclose(out["intervals"]["POST_PRE"]["raw_same"]["point"], want_raw)
    assert np.isclose(out["intervals"]["POST_PRE"]["raw_arm"]["point"], want_raw + D)


def test_bootstrap_indices_deterministic_and_bounded():
    a = pc.paired_bootstrap_indices(10, 500, seed=42)
    b = pc.paired_bootstrap_indices(10, 500, seed=42)
    assert a.shape == (500, 10)
    assert np.array_equal(a, b)
    assert a.min() >= 0 and a.max() < 10
    with pytest.raises(ValueError):
        pc.paired_bootstrap_indices(1, 10, seed=42)


# ---------------- (D) common-finite subspace ---------------------------------


def test_common_finite_subspace():
    n = 8
    vals = {
        "OPPOSITE": {
            "PRE": np.array([1.0, np.nan, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
            "MID": np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0]),
            "POST": np.array([1.0, 2.0, 3.0, np.nan, 5.0, 6.0, 7.0, 8.0]),
        },
        "SAME": {ck: np.ones(n) for ck in pc.CKS},
    }
    sub = pc.common_finite_subset(vals, ("OPPOSITE", "SAME"))
    assert sub.tolist() == [0, 4, 5, 6, 7]
    with pytest.raises(ValueError):
        pc.common_finite_subset(
            {"A": {ck: np.ones(3) for ck in pc.CKS}, "B": {ck: np.ones(4) for ck in pc.CKS}},
            ("A", "B"),
        )


# ---------------- (E) OPPOSITE excludes opp_na --------------------------------


def test_opposite_opp_na_exclusion():
    n = 6
    m_arm = {ck: np.linspace(1e-4, 2e-4, n) for ck in pc.CKS}
    m_same = {ck: np.zeros(n) for ck in pc.CKS}
    opp_na = np.array([True, False, True, False, False, False])
    sub = pc.common_finite_subset({ "OPPOSITE": m_arm, "SAME": m_same }, ("OPPOSITE", "SAME"))
    sub = sub[~opp_na[sub]]
    assert sub.tolist() == [1, 3, 4, 5]
    out = pc.diff_in_diff(m_arm, m_same, sub, n_boot=500, seed=3)
    assert out["n"] == 4
    assert out["n_all"] == 6


# ---------------- (F) universal offset boundaries -----------------------------


def test_offset_tertile_boundaries():
    assert pc.offset_tertile(0.0) == "START"
    assert pc.offset_tertile(1.0 / 3.0 - 1e-9) == "START"
    assert pc.offset_tertile(1.0 / 3.0) == "CENTER"
    assert pc.offset_tertile(2.0 / 3.0 - 1e-9) == "CENTER"
    assert pc.offset_tertile(2.0 / 3.0) == "END"
    assert pc.offset_tertile(1.0) == "END"
    for bad in (-0.1, 1.2, float("nan")):
        with pytest.raises(ValueError):
            pc.offset_tertile(bad)


# ---------------- (G) orientation labels fail closed --------------------------


def test_physical_labels_fail_closed():
    for tert in ("START", "CENTER", "END"):
        assert pc.physical_offset_label("horizontal", tert, convention_verified=False) is None
        assert pc.physical_offset_label("vertical", tert, convention_verified=False) is None
    assert pc.physical_offset_label("horizontal", "START") == "LEFT"
    assert pc.physical_offset_label("horizontal", "CENTER") == "CENTER"
    assert pc.physical_offset_label("horizontal", "END") == "RIGHT"
    assert pc.physical_offset_label("vertical", "START") == "TOP"
    assert pc.physical_offset_label("vertical", "END") == "BOTTOM"
    assert pc.neutral_offset_label("horizontal", "START") == "H_START"
    assert pc.neutral_offset_label("vertical", "END") == "V_END"
    with pytest.raises(ValueError):
        pc.physical_offset_label("diagonal", "START")
    with pytest.raises(ValueError):
        pc.physical_offset_label("horizontal", "EDGE")


# ---------------- (H) loss vs displacement separation -------------------------


def test_loss_and_displacement_separate_fields():
    """The posthoc result must carry loss preference and prediction
    displacement as SEPARATE top-level fields; neither block may embed the
    other's data source (source-level lock on posthoc_review.py)."""
    src = (DEV_DIR / "posthoc_review.py").read_text(encoding="utf-8")
    lines = src.splitlines()
    assert '"loss_preference": {' in src
    assert '"prediction_displacement": {' in src
    # locate the two blocks and ensure no cross-embedding
    start_lp = next(i for i, ln in enumerate(lines) if ln.strip().startswith('"loss_preference": {'))
    start_pd = next(i for i, ln in enumerate(lines) if ln.strip().startswith('"prediction_displacement": {'))
    block_lp = "\n".join(lines[start_lp : start_lp + 8])
    block_pd = "\n".join(lines[start_pd : start_pd + 8])
    assert "pred_disp" not in block_lp
    assert "adjusted" not in block_pd
    assert "PREDICTION_DISPLACEMENT_SENSITIVITY" in src
    assert "never merged" in src or "SEPARATE" in src


# ---------------- (I) verdict boolean == report boolean -----------------------


def test_verdict_and_report_share_numerics_boolean():
    n = 10
    clean_same = {
        "checkpoints": {
            ck: {"mean": 1e-9, "ci95": [-1e-7, 1e-7], "abs_p50": 1e-8, "abs_p90": 1e-8, "abs_p95": 1e-8, "abs_p99": 1e-8, "maxabs": 1e-7, "n_units": n, "n_strata": 4}
            for ck in pc.CKS
        },
        "deltas": {k: {"point": 0.0, "ci95": [-1e-7, 1e-7], "n": n} for k in ("POST_PRE", "MID_PRE", "POST_MID")},
        "n_units": n,
    }
    adjusted = {}
    for arm in ("OPPOSITE", "SHUFFLED", "IDENTITY"):
        adjusted[arm] = {
            "n": n,
            "intervals": {
                k: {
                    "raw_arm": {"point": 1e-3, "ci95": [5e-4, 1.5e-3]},
                    "raw_same": {"point": 0.0, "ci95": [-1e-7, 1e-7]},
                    "adjusted": {"point": 1e-3, "ci95": [5e-4, 1.5e-3]},
                }
                for k in ("MID_PRE", "POST_MID", "POST_PRE")
            },
        }
    mp = {"present": True, "hidden_state_detected": False, "p1_max_abs_loss_delta_rel": 0.0, "same_vs_correct_rel_p99": 1e-5}
    clean, detail = pc.numerics_gate(True, clean_same, clean_same, adjusted, mp)
    assert clean is True
    v = pc.classify_verdict(detail["clean"], adjusted)  # the REPORT boolean drives the verdict
    assert v == "POSITIVE_LOSS_PREFERENCE_LEARNING"
    # now break R3: SAME POST-PRE CI excludes 0 -> the SAME boolean must flip the verdict
    bad = json.loads(json.dumps(clean_same))
    bad["deltas"]["POST_PRE"] = {"point": 5e-3, "ci95": [4e-3, 6e-3], "n": n}
    clean2, detail2 = pc.numerics_gate(True, bad, clean_same, adjusted, mp)
    assert clean2 is False
    v2 = pc.classify_verdict(detail2["clean"], adjusted)
    assert v2 == "BLOCKED_NUMERICS"
    assert any("R3" in r for r in detail2["reasons"])


# ---------------- (L) interval classification ---------------------------------


def test_interval_classification_all_classes():
    up = {"point": 1e-3, "ci95": [5e-4, 1.5e-3]}
    dn = {"point": -1e-3, "ci95": [-1.5e-3, -5e-4]}
    flat_sig = {"point": 1e-5, "ci95": [-5e-5, 7e-5]}
    assert pc.classify_interval(up, up) == "MONOTONIC_GAIN"
    assert pc.classify_interval(up, flat_sig) == "PLATEAU"
    late_notiny = {"point": 5e-4, "ci95": [-2e-4, 1.2e-3]}
    assert pc.classify_interval(up, late_notiny) == "EARLY_GAIN"
    early_tiny = {"point": 5e-6, "ci95": [-2e-4, 2.5e-4]}
    assert pc.classify_interval(early_tiny, up) == "LATE_GAIN"
    z = {"point": 1e-6, "ci95": [-2e-4, 2e-4]}
    assert pc.classify_interval(z, z, flat_threshold=1e-4) == "FLAT"
    assert pc.classify_interval(dn, dn) == "REVERSED"
    assert pc.classify_interval(dn, up) == "NON_MONOTONIC"
    big_flat = {"point": 5e-3, "ci95": [-2e-2, 2.5e-2]}
    assert pc.classify_interval(big_flat, big_flat, flat_threshold=1e-4) == "NON_MONOTONIC"


# ---------------- (M) verdict classification paths ----------------------------


def _adj(point, lo, hi):
    n = 10
    return {
        "n": n,
        "intervals": {
            k: {"raw_arm": {"point": point, "ci95": [lo, hi]}, "raw_same": {"point": 0.0, "ci95": [-1e-9, 1e-9]}, "adjusted": {"point": point, "ci95": [lo, hi]}}
            for k in ("MID_PRE", "POST_MID", "POST_PRE")
        },
    }


def test_verdict_classification_paths():
    pos = {
        "OPPOSITE": _adj(1e-3, 5e-4, 1.5e-3),
        "SHUFFLED": _adj(8e-4, 3e-4, 1.3e-3),
        "IDENTITY": _adj(1e-3, -2e-4, 2.2e-3),
    }
    assert pc.classify_verdict(True, pos) == "POSITIVE_LOSS_PREFERENCE_LEARNING"
    weak = {
        "OPPOSITE": _adj(1e-3, -1e-4, 2e-3),
        "SHUFFLED": _adj(5e-4, 1e-4, 9e-4),
    }
    assert pc.classify_verdict(True, weak) == "WEAK_POSITIVE_LOSS_PREFERENCE"
    null = {
        "OPPOSITE": _adj(1e-5, -2e-4, 2e-4),
        "SHUFFLED": _adj(-5e-6, -2e-4, 2e-4),
    }
    assert pc.classify_verdict(True, null, flat_threshold=1e-4) == "NO_DETECTED_LOSS_PREFERENCE_GAIN"
    mis = {
        "OPPOSITE": _adj(-1e-3, -1.5e-3, -5e-4),
        "SHUFFLED": _adj(1e-5, -2e-4, 2e-4),
    }
    assert pc.classify_verdict(True, mis) == "MISALIGNED_LOSS_PREFERENCE"
    assert pc.classify_verdict(False, pos) == "BLOCKED_NUMERICS"
    # END systematic negative does NOT demote a positive verdict (it is a
    # recommendation-level concern), per section 16/32 separation
    assert pc.classify_verdict(True, pos, end_systematic_negative=True) == "POSITIVE_LOSS_PREFERENCE_LEARNING"
    # but a null-ish primary with systematic negative END -> misaligned
    assert pc.classify_verdict(True, null, end_systematic_negative=True) == "MISALIGNED_LOSS_PREFERENCE"


# ---------------- (N) numerics gate detail fields ------------------------------


def test_numerics_gate_detail_fields():
    n = 10
    same = {
        "checkpoints": {
            ck: {"mean": 1e-9, "ci95": [-1e-7, 1e-7], "abs_p50": 1e-8, "abs_p90": 1e-8, "abs_p95": 1e-8, "abs_p99": 1e-8, "maxabs": 1e-7, "n_units": n, "n_strata": 4}
            for ck in pc.CKS
        },
        "deltas": {k: {"point": 0.0, "ci95": [-1e-7, 1e-7], "n": n} for k in ("POST_PRE", "MID_PRE", "POST_MID")},
        "n_units": n,
    }
    adjusted = {
        "OPPOSITE": _adj(1e-3, 5e-4, 1.5e-3),
        "SHUFFLED": _adj(8e-4, 3e-4, 1.3e-3),
    }
    mp = {"present": True, "hidden_state_detected": False, "p1_max_abs_loss_delta_rel": 0.0, "same_vs_correct_rel_p99": 1e-5}
    clean, detail = pc.numerics_gate(True, same, same, adjusted, mp)
    assert clean is True
    for key in ("camera_same_mean_ci", "camera_same_delta_ci", "ordinary_same_mean_ci", "same_causal_scale_ratio", "rules", "reasons", "determinism_valid"):
        assert key in detail
    assert set(detail["camera_same_mean_ci"]) == set(pc.CKS)
    assert set(detail["camera_same_delta_ci"]) == {"POST_PRE", "MID_PRE", "POST_MID"}
    assert detail["same_causal_scale_ratio"] == 0.0
    assert detail["rules"] == {"R1": True, "R2": True, "R3": True, "R4": True, "R5": True}
    # fail-closed paths
    c2, d2 = pc.numerics_gate(False, same, same, adjusted, mp)
    assert c2 is False and any("R1" in r for r in d2["reasons"])
    c3, d3 = pc.numerics_gate(True, same, same, adjusted, None)
    assert c3 is False and any("R5" in r for r in d3["reasons"])
    bias = json.loads(json.dumps(same))
    bias["checkpoints"]["POST"]["mean"] = 1e-4
    c4, d4 = pc.numerics_gate(True, bias, same, adjusted, mp)
    assert c4 is False and any("R2" in r for r in d4["reasons"])


# ---------------- (O) recommendation mapping -----------------------------------


def test_recommendation_mapping():
    assert pc.classify_recommendation("BLOCKED_NUMERICS") == "FIX_AUDIT_HARNESS_BEFORE_TRAINING"
    assert pc.classify_recommendation("POSITIVE_LOSS_PREFERENCE_LEARNING") == "LONGER_P25_REVIEW_CANDIDATE"
    assert pc.classify_recommendation("POSITIVE_LOSS_PREFERENCE_LEARNING", end_systematic_negative=True) == "REVIEW_CAMERA_SAMPLING_BALANCE_FIRST"
    assert pc.classify_recommendation("WEAK_POSITIVE_LOSS_PREFERENCE") == "NO_MORE_EXPOSURE"
    assert pc.classify_recommendation("WEAK_POSITIVE_LOSS_PREFERENCE", end_systematic_negative=True) == "REVIEW_CAMERA_SAMPLING_BALANCE_FIRST"
    assert pc.classify_recommendation("NO_DETECTED_LOSS_PREFERENCE_GAIN") == "NO_MORE_EXPOSURE"
    assert pc.classify_recommendation("MISALIGNED_LOSS_PREFERENCE") == "NO_MORE_EXPOSURE"
    with pytest.raises(ValueError):
        pc.classify_recommendation("NOT_A_VERDICT")


# ---------------- extra: SAME baseline stats -----------------------------------


def test_same_baseline_stats_quantiles_and_deltas():
    n = 6
    rng = np.random.default_rng(0)
    mat = {ck: rng.normal(0, 1e-5, size=(n, 4)) for ck in pc.CKS}
    mat["POST"] = mat["PRE"] + 3e-6  # known per-unit drift
    out = pc.same_baseline_stats(mat, n_boot=1000, seed=5)
    for ck in pc.CKS:
        pooled = np.abs(mat[ck].ravel())
        c = out["checkpoints"][ck]
        assert np.isclose(c["maxabs"], pooled.max())
        assert np.isclose(c["abs_p99"], np.percentile(pooled, 99.0))
        assert np.isclose(c["mean"], mat[ck].mean())
    d = out["deltas"]["POST_PRE"]
    assert np.isclose(d["point"], 3e-6)
    assert d["ci95"][0] <= 3e-6 <= d["ci95"][1]
    assert out["n_units"] == n
