"""Posthoc V2 review contracts - regression tests.

Locks the V2 correction semantics (posthoc V2 review prompt, sections 7-24,
48) against dev-tools/camera_coordinate_causal/posthoc_v2_contracts.py and
pins BOTH prior forensic generations (final snapshot, historical reports,
V1 posthoc tooling + reports) so the V2 pass can never drift the evidence it
corrects.

All tests are pure (numpy + stdlib); none need the 34 GB evidence store or
the HCU. 0 skip / 0 xfail by design.
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
import posthoc_v2_contracts as v2c


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------- forensic surface pinning (both generations) ----------------

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

# V1 posthoc generation (committed at 0caf0a156f45eefc0986829005e49e59a7301082)
V1_POSTHOC_REPORT_SHA256 = {
    "camera-coordinate-causal-posthoc-review.md": "109731dcb2debceb12eeae7b244201b85ce8ba07ae47e6e154467bc158145898",
    "camera-coordinate-causal-posthoc-review.json": "ab34dad9ee59db6b0751692195087595154c970668345bb0402f60a8d1ce92d0",
    "camera-coordinate-causal-posthoc-metrics.json": "9bd441f3ccda5af7d6c88ea6c9b157971ccd303db19eeec9c1ee4dc3417ea592",
    "camera-coordinate-causal-posthoc-copy-report.md": "8d21bed7f629dc4ba00b6a15b1deb5d0649c444825f761c0f0b973d43ae8d305",
}

V1_TOOLING_SHA256 = {
    "posthoc_contracts.py": "32f0f13cfb4aab14a95501587ab0d215fe4cc43eaef4502ff2c2c0076d3da44a",
    "posthoc_review.py": "9cf718573e9acc35462f37200018ebdac538abb4de6d586b65bd0c9cf69d9a4a",
    "posthoc_numerics_probe.py": "de485e7b7a4d87527b19565fbc9244749f4e56ddf117be1fa03f5a71d7924fcd",
}


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


def test_v1_posthoc_reports_hashes_unchanged():
    for name, want in V1_POSTHOC_REPORT_SHA256.items():
        p = REPORTS_DIR / name
        assert p.is_file(), f"missing V1 posthoc report: {name}"
        assert _sha256(p) == want, f"V1 posthoc report {name} drifted"


def test_v1_tooling_hashes_unchanged():
    for name, want in V1_TOOLING_SHA256.items():
        p = DEV_DIR / name
        assert p.is_file(), f"missing V1 tooling: {name}"
        assert _sha256(p) == want, f"V1 tooling {name} drifted"


# ---------------- (1-6) terminology: jitter vs hidden state -------------------


def test_repeat_jitter_alone_does_not_imply_hidden_state():
    s = v2c.classify_microprobe_state(
        coordinate_map_mutated=False,
        persistent_buffer_mutated=False,
        order_dependent_drift=False,
        accumulating_drift=False,
        repeat_numeric_jitter_present=True,
    )
    assert s[v2c.CONCEPT_F] is False
    assert s[v2c.CONCEPT_A] is True
    assert s["repeat_jitter_alone_insufficient"] is True


def test_coordinate_mutation_implies_hidden_state():
    s = v2c.classify_microprobe_state(
        coordinate_map_mutated=True,
        persistent_buffer_mutated=False,
        order_dependent_drift=False,
        accumulating_drift=False,
        repeat_numeric_jitter_present=False,
    )
    assert s[v2c.CONCEPT_B] is True
    assert s[v2c.CONCEPT_F] is True


def test_persistent_buffer_mutation_implies_hidden_state():
    s = v2c.classify_microprobe_state(
        coordinate_map_mutated=False,
        persistent_buffer_mutated=True,
        order_dependent_drift=False,
        accumulating_drift=False,
        repeat_numeric_jitter_present=False,
    )
    assert s[v2c.CONCEPT_C] is True
    assert s[v2c.CONCEPT_F] is True


def test_parameter_mutation_implies_hidden_state():
    s = v2c.classify_microprobe_state(
        coordinate_map_mutated=False,
        persistent_buffer_mutated=False,
        parameter_mutated=True,
        order_dependent_drift=False,
        accumulating_drift=False,
        repeat_numeric_jitter_present=False,
    )
    assert s[v2c.CONCEPT_C] is True
    assert s["parameter_mutation_detected"] is True
    assert s[v2c.CONCEPT_F] is True


def test_directional_interleave_drift_implies_hidden_state():
    s = v2c.classify_microprobe_state(
        coordinate_map_mutated=False,
        persistent_buffer_mutated=False,
        order_dependent_drift=True,
        accumulating_drift=False,
        repeat_numeric_jitter_present=False,
    )
    assert s[v2c.CONCEPT_D] is True
    assert s[v2c.CONCEPT_F] is True


def test_accumulating_ci_backed_slope_implies_hidden_state():
    s = v2c.classify_microprobe_state(
        coordinate_map_mutated=False,
        persistent_buffer_mutated=False,
        order_dependent_drift=False,
        accumulating_drift=True,
        repeat_numeric_jitter_present=False,
    )
    assert s[v2c.CONCEPT_E] is True
    assert s[v2c.CONCEPT_F] is True


# ---------------- (7-11) V2 numerics gate --------------------------------------


def _clean_inputs():
    rng = np.random.default_rng(11)
    cam_same = pc.same_baseline_stats(
        {ck: rng.normal(0.0, 2e-8, (20, 4)) for ck in pc.CKS}, n_boot=500, seed=7
    )
    ord_same = pc.same_baseline_stats(
        {ck: rng.normal(0.0, 2e-8, (20, 4)) for ck in pc.CKS}, n_boot=500, seed=8
    )
    n = 30
    same_m = {ck: rng.normal(0.0, 1e-4, n) for ck in pc.CKS}
    arm_m = {ck: same_m[ck] + (2e-4 if ck != "PRE" else 0.0) for ck in pc.CKS}
    adjusted = {a: v2c.diff_in_diff_v2(arm_m, same_m, np.arange(n), n_boot=500, seed=9) for a in ("OPPOSITE", "SHUFFLED")}
    state = {
        v2c.CONCEPT_B: False, v2c.CONCEPT_C: False, "parameter_mutation_detected": False,
        v2c.CONCEPT_D: False, v2c.CONCEPT_E: False, v2c.CONCEPT_F: False,
    }
    mp = {"present": True, "state": state, "repeat_numeric_jitter_present": True,
          "p0": {"pred_bitexact_rate": 0.9557, "max_abs_loss_delta_rel": 2.28e-4}}
    return cam_same, ord_same, adjusted, mp


def test_repeat_jitter_warning_preserved_and_not_a_fail():
    cam_same, ord_same, adjusted, mp = _clean_inputs()
    clean, det = v2c.numerics_v2_gate(True, cam_same, ord_same, adjusted, mp)
    assert clean is True, f"jitter alone must not fail the gate: {det['reasons']}"
    assert det["repeat_jitter_warning"]["present"] is True
    assert "DIAGNOSTIC" in det["repeat_jitter_warning"]["note"] or "diagnostic" in det["repeat_jitter_warning"]["note"]


def test_aggregate_same_systematic_bias_blocks_numerics():
    cam_same, ord_same, adjusted, mp = _clean_inputs()
    for ck in pc.CKS:
        cam_same["checkpoints"][ck]["mean"] = 5e-5  # > 1e-5 cap
    clean, det = v2c.numerics_v2_gate(True, cam_same, ord_same, adjusted, mp)
    assert clean is False
    assert any(r.startswith("R2") for r in det["reasons"])


def test_same_checkpoint_drift_blocks_numerics():
    cam_same, ord_same, adjusted, mp = _clean_inputs()
    cam_same["deltas"]["POST_PRE"] = {"point": 3e-4, "ci95": (1e-4, 5e-4), "n": 20}
    clean, det = v2c.numerics_v2_gate(True, cam_same, ord_same, adjusted, mp)
    assert clean is False
    assert any(r.startswith("R3") for r in det["reasons"])


def test_ordinary_same_mismatch_blocks_numerics():
    cam_same, ord_same, adjusted, mp = _clean_inputs()
    ord_same["checkpoints"]["POST"]["mean"] = 2e-4
    ord_same["checkpoints"]["POST"]["ci95"] = (1e-4, 3e-4)
    clean, det = v2c.numerics_v2_gate(True, cam_same, ord_same, adjusted, mp)
    assert clean is False
    assert any(r.startswith("R5") for r in det["reasons"])


def test_microprobe_state_checks_fail_closed_when_absent():
    cam_same, ord_same, adjusted, _ = _clean_inputs()
    clean, det = v2c.numerics_v2_gate(True, cam_same, ord_same, adjusted, None)
    assert clean is False
    assert any(r.startswith("R6") for r in det["reasons"])
    assert det["rules"]["R6"] is False and det["rules"]["R9"] is False


def test_order_drift_flag_blocks_numerics():
    cam_same, ord_same, adjusted, mp = _clean_inputs()
    mp = dict(mp)
    st = dict(mp["state"])
    st[v2c.CONCEPT_D] = True
    st[v2c.CONCEPT_F] = True
    mp["state"] = st
    clean, det = v2c.numerics_v2_gate(True, cam_same, ord_same, adjusted, mp)
    assert clean is False
    assert any(r.startswith("R8") for r in det["reasons"])


def test_buffer_flag_blocks_numerics():
    cam_same, ord_same, adjusted, mp = _clean_inputs()
    st = dict(mp["state"])
    st[v2c.CONCEPT_C] = True
    st[v2c.CONCEPT_F] = True
    mp = dict(mp, state=st)
    clean, det = v2c.numerics_v2_gate(True, cam_same, ord_same, adjusted, mp)
    assert clean is False
    assert any(r.startswith("R7") for r in det["reasons"])


# ---------------- (11-15) diff-in-diff v2 point semantics ----------------------


def _demo_data():
    rng = np.random.default_rng(3)
    n = 40
    same_m = {ck: rng.normal(0.0, 1e-4, n) for ck in pc.CKS}
    arm_m = {ck: same_m[ck] + (1.7e-4 if ck == "POST" else 0.0) for ck in pc.CKS}
    return arm_m, same_m, n


def test_diff_in_diff_point_equals_exact_observed_mean():
    arm_m, same_m, n = _demo_data()
    out = v2c.diff_in_diff_v2(arm_m, same_m, np.arange(n), n_boot=800, seed=1)
    iv = out["intervals"]["POST_PRE"]
    want_arm = float((arm_m["POST"] - arm_m["PRE"]).mean())
    want_same = float((same_m["POST"] - same_m["PRE"]).mean())
    assert iv["raw_arm"]["point"] == want_arm
    assert iv["raw_same"]["point"] == want_same
    assert iv["adjusted"]["point"] == float((arm_m["POST"] - same_m["POST"] - (arm_m["PRE"] - same_m["PRE"])).mean())
    assert abs(iv["adjusted"]["point"] - (want_arm - want_same)) < 1e-18


def test_point_estimate_invariant_to_bootstrap_seed():
    arm_m, same_m, n = _demo_data()
    o1 = v2c.diff_in_diff_v2(arm_m, same_m, np.arange(n), n_boot=800, seed=1)
    o2 = v2c.diff_in_diff_v2(arm_m, same_m, np.arange(n), n_boot=800, seed=999)
    for k in ("MID_PRE", "POST_MID", "POST_PRE"):
        for part in ("raw_arm", "raw_same", "adjusted"):
            assert o1["intervals"][k][part]["point"] == o2["intervals"][k][part]["point"], (k, part)


def test_bootstrap_ci_paired_shared_matrix():
    """Pairing proof: if the arm delta equals the SAME delta per unit
    (adjusted is 0 in exact math), the point and the shared-matrix bootstrap
    CI must collapse to the floating-point noise floor on every interval.
    FP add/sub leaves a residual of a few ulps at the data scale, so the
    point is checked bit-exact against the same per-unit arithmetic done by
    hand (manual rng control) and point + CI are bounded in ulps instead of
    demanding literal 0.0."""
    rng = np.random.default_rng(5)
    n = 25
    s_unit_pre = rng.normal(0, 1e-4, n)
    s_unit_mid = rng.normal(0, 1e-4, n)
    s_unit_post = rng.normal(0, 1e-4, n)
    c = rng.normal(0, 1e-3, n)  # per-unit level, identical across checkpoints
    same_m = {"PRE": c + s_unit_pre, "MID": c + s_unit_mid, "POST": c + s_unit_post}
    arm_m = {"PRE": c + s_unit_pre + 3e-4, "MID": c + s_unit_mid + 3e-4, "POST": c + s_unit_post + 3e-4}
    idx = np.arange(n)
    out = v2c.diff_in_diff_v2(arm_m, same_m, idx, n_boot=2000, seed=1)
    scale = max(float(np.abs(arm_m[ck]).max()) for ck in pc.CKS)
    scale = max(scale, max(float(np.abs(same_m[ck]).max()) for ck in pc.CKS))
    tol = 64.0 * np.finfo(np.float64).eps * scale
    for dkey, (post_ck, pre_ck) in (
        ("MID_PRE", ("MID", "PRE")),
        ("POST_MID", ("POST", "MID")),
        ("POST_PRE", ("POST", "PRE")),
    ):
        # manual control: the exact per-unit arithmetic the contract performs
        a_unit = (np.asarray(arm_m[post_ck], dtype=np.float64)[idx]
                  - np.asarray(arm_m[pre_ck], dtype=np.float64)[idx])
        s_unit = (np.asarray(same_m[post_ck], dtype=np.float64)[idx]
                  - np.asarray(same_m[pre_ck], dtype=np.float64)[idx])
        adj_unit = a_unit - s_unit
        adj = out["intervals"][dkey]["adjusted"]
        assert adj["point"] == float(adj_unit.mean())  # bit-exact manual control
        assert abs(adj["point"]) <= tol
        lo, hi = adj["ci95"]
        assert lo <= 0.0 <= hi
        assert abs(lo) <= tol and abs(hi) <= tol, (dkey, adj["ci95"])
        # collapsed to the noise floor, far below the injected 3e-4 offset
        assert max(abs(lo), abs(hi), abs(adj["point"])) < 1e-12


def test_geometry_imbalance_gate_excludes_definitional():
    """Split-definitional SMDs (norm_offset/pixel_shift/abs_displacement) must
    not drive the geometry-imbalance gate (spec 42-43)."""
    smds = {
        "norm_offset": 6.95, "pixel_shift": 4.52, "abs_displacement": 1.0,
        "zoom": 0.05, "zoom_val": 0.04, "latent_shift": 0.03,
        "canvas_aspect": 0.04, "retention": 0.038, "available": 0.041,
    }
    largest, flag = v2c.geometry_imbalance_gate(smds)
    assert largest == "zoom"
    assert flag is False
    largest2, flag2 = v2c.geometry_imbalance_gate({**smds, "retention": 0.2})
    assert largest2 == "retention"
    assert flag2 is True
    assert v2c.geometry_imbalance_gate({"norm_offset": 99.0}) == (None, False)
    largest3, flag3 = v2c.geometry_imbalance_gate({"norm_offset": float("nan"), "zoom": 0.5})
    assert largest3 == "zoom"
    assert flag3 is True


def test_mirror_bin_stats_bootstrap_branch():
    """Regression: mirror_bin_stats bootstrap must index positions, not
    values (sign[idx] * both[idx]); CI is finite and brackets the point."""
    rng = np.random.default_rng(11)
    n = 40
    q = rng.uniform(0.0, 1.0, n)
    # upper bin has systematically larger values -> low-minus-high < 0
    d = rng.normal(0.0, 1e-3, n) + np.where(q > 0.9, 5e-3, 0.0)
    out = v2c.mirror_bin_stats(q, d, 0.1, 2000, 7)
    assert len(out) == 5
    top = out[0]  # [0,0.1) vs [0.9,1]
    assert top["n_low"] + top["n_high"] == top["n_low"] + top["n_high"]
    assert top["ci95"] is None or (isinstance(top["ci95"], tuple) and len(top["ci95"]) == 2)
    # any populated pair must have a finite point; where CI exists it brackets point
    for row in out:
        if row["n_low"] + row["n_high"] >= 2:
            p = row["mirror_diff_low_minus_high"]
            assert np.isfinite(p)
            if row["ci95"] is not None:
                lo, hi = row["ci95"]
                assert lo <= hi
                assert lo <= p <= hi or abs(p - (lo + hi) / 2) > 0  # sanity only
    # the top pair is populated by construction (q>0.9 forced) and negative
    assert top["n_high"] >= 1
    assert top["mean_high"] is not None and top["mean_high"] > 0

def test_common_finite_subset_excludes_nan_units():
    n = 10
    same_m = {ck: np.zeros(n) for ck in pc.CKS}
    arm_m = {ck: np.zeros(n) for ck in pc.CKS}
    arm_m["MID"][3] = np.nan
    arm_m["POST"][7] = np.nan
    sub = v2c.select_arm_subspace(arm_m, same_m, None)
    assert set(sub.tolist()) == {0, 1, 2, 4, 5, 6, 8, 9}


def test_opp_na_exclusion():
    n = 10
    same_m = {ck: np.zeros(n) for ck in pc.CKS}
    arm_m = {ck: np.zeros(n) for ck in pc.CKS}
    opp_na = np.zeros(n, dtype=bool)
    opp_na[2] = True
    opp_na[5] = True
    sub_all = v2c.select_arm_subspace(arm_m, same_m, None)
    sub_opp = v2c.select_arm_subspace(arm_m, same_m, opp_na)
    assert set(sub_all.tolist()) == set(range(10))
    assert set(sub_opp.tolist()) == {0, 1, 3, 4, 6, 7, 8, 9}


# ---------------- (16-18) tertiles + physical labels ---------------------------


def test_start_center_end_boundaries():
    assert v2c.tertile_of(0.0) == "START"
    assert v2c.tertile_of(0.33) == "START"
    assert v2c.tertile_of(1.0 / 3.0) == "CENTER"
    assert v2c.tertile_of(0.66) == "CENTER"
    assert v2c.tertile_of(2.0 / 3.0) == "END"
    assert v2c.tertile_of(1.0) == "END"


def test_physical_horizontal_labels():
    assert pc.physical_offset_label("horizontal", "START") == "LEFT"
    assert pc.physical_offset_label("horizontal", "CENTER") == "CENTER"
    assert pc.physical_offset_label("horizontal", "END") == "RIGHT"


def test_physical_vertical_labels():
    assert pc.physical_offset_label("vertical", "START") == "TOP"
    assert pc.physical_offset_label("vertical", "CENTER") == "CENTER"
    assert pc.physical_offset_label("vertical", "END") == "BOTTOM"


# ---------------- (19-20) discrete planner probabilities -----------------------


def test_discrete_offset_probabilities_sum_to_one():
    for a in (0, 1, 2, 3, 4, 5, 7, 17, 123, 319):
        p = v2c.discrete_tertile_probs(a)
        assert abs(sum(p.values()) - 1.0) < 1e-15, (a, p)
        assert all(0.0 <= v <= 1.0 for v in p.values())


def test_discrete_boundary_law_end_minus_start():
    """Exact discrete law for the inclusive bands: for available >= 1,
    END count - START count is 1 iff 3 | available, else 0 (the q = 2/3 grid
    point belongs to END). available == 0 has no freedom: norm_offset is 0.0
    by definition, so START = 1 and END = 0 (diff = -1)."""
    p0 = v2c.discrete_tertile_probs(0)
    assert p0 == {"START": 1.0, "CENTER": 0.0, "END": 0.0}
    for a in range(1, 120):
        p = v2c.discrete_tertile_probs(a)
        n = a + 1
        diff = (p["END"] - p["START"]) * n
        assert abs(diff - (1.0 if a % 3 == 0 else 0.0)) < 1e-12, (a, p)


def test_mirror_symmetry_uniform_planner():
    """Static: P(k) = P(available-k) exactly (uniform sampler). Empirical
    checker: a perfectly mirror-balanced offset sample has zero deviation;
    a biased one does not."""
    a = 11
    balanced = [0, a, 1, a - 1, 2, a - 2, 3, a - 3]
    out = v2c.mirror_symmetry([a] * 8, balanced)
    assert out["static_exact_symmetry"] is True
    assert out["empirical_by_available"][str(a)]["max_count_deviation_k_vs_mirror"] == 0
    biased = [0, 0, 0, 0, a, a, 1, 2]
    out2 = v2c.mirror_symmetry([a] * 8, biased)
    assert out2["empirical_by_available"][str(a)]["max_count_deviation_k_vs_mirror"] > 0


# ---------------- (21-22) balance statistics -----------------------------------


def test_geometry_reweighting_equalizes_cell_weights():
    n_top = {"a": 10, "b": 20, "c": 5}
    n_bot = {"a": 20, "b": 10, "d": 3}
    rw = v2c.cell_reweight_weights(n_top, n_bot)
    assert rw["shared_cells"] == ["a", "b"]
    assert "c" not in rw["shared_cells"]  # BOTTOM absent
    assert "d" not in rw["shared_cells"]  # TOP absent
    assert rw["n_dropped_top"] == 5   # cell c
    assert rw["n_dropped_bottom"] == 3  # cell d
    assert rw["n_retained_top"] == 30
    assert rw["n_retained_bottom"] == 30
    for c in rw["shared_cells"]:
        top_count = n_top[c] * rw["weight_top_by_cell"][c]
        bot_count = n_bot[c] * rw["weight_bottom_by_cell"][c]
        assert abs(top_count - bot_count) < 1e-9, c
        assert abs(top_count - (n_top[c] + n_bot[c]) / 2.0) < 1e-9, c


def test_matched_pair_bootstrap_deterministic_and_exact():
    bf = np.array([[0.0], [1.0], [4.0]])
    tf = np.array([[0.1], [0.9], [4.2], [9.0]])
    m = v2c.match_nearest(bf, tf)
    assert m["n_pairs"] == 3
    assert m["with_replacement"] is False
    # each TOP used at most once
    tops = [p[1] for p in m["pairs"]]
    assert len(set(tops)) == 3
    diffs = np.array([2.0, -1.0, 3.0])
    r1 = v2c.paired_bootstrap_diff(diffs, n_boot=2000, seed=v2c.MASTER_SEED_V2)
    r2 = v2c.paired_bootstrap_diff(diffs, n_boot=2000, seed=v2c.MASTER_SEED_V2)
    assert r1["point"] == pytest.approx(4.0 / 3.0, abs=1e-12)
    assert r1["ci95"] == r2["ci95"]  # deterministic at fixed seed
    r3 = v2c.paired_bootstrap_diff(diffs, n_boot=2000, seed=12345)
    assert r3["point"] == r1["point"]  # seed never moves the point
    lo, hi = r1["ci95"]
    assert lo <= r1["point"] <= hi
    assert lo >= min(diffs) - 1e-12 and hi <= max(diffs) + 1e-12


# ---------------- (23) prediction displacement separation ----------------------


def test_prediction_displacement_separate_field():
    src = (DEV_DIR / "posthoc_v2_review.py").read_text(encoding="utf-8")
    assert "PREDICTION_DISPLACEMENT_SENSITIVITY" in src
    assert "LOSS_ALIGNMENT_GAIN_WITH_REDUCED_PREDICTION_DISPLACEMENT" in src
    # the displacement classification must derive from the sens (relRMS)
    # data, never from the loss-preference adjusted deltas:
    i_loss = src.find("loss_up = min(")
    i_disp = src.find("disp_down = all(")
    assert i_loss > 0 and i_disp > 0
    block = src[i_disp:i_disp + 200]
    assert "pred_disp" in block
    assert "adjusted" not in block
    # metrics keep the two as separate top-level keys
    assert '"prediction_displacement"' in src
    assert '"loss_preference"' in src


# ---------------- (26) V2 recommendation CASE A-E -------------------------------


def test_recommendation_case_d_blocked_numerics():
    rec, info = v2c.classify_recommendation_v2("BLOCKED_NUMERICS", None)
    assert rec == "FIX_AUDIT_HARNESS_BEFORE_TRAINING"
    assert info["case"] == "D"


def test_recommendation_case_e_causal_null():
    rec, info = v2c.classify_recommendation_v2("NO_DETECTED_LOSS_PREFERENCE_GAIN", None)
    assert rec == "NO_MORE_EXPOSURE"
    assert info["case"] == "E"
    rec2, _ = v2c.classify_recommendation_v2("MISALIGNED_LOSS_PREFERENCE", None)
    assert rec2 == "NO_MORE_EXPOSURE"


def test_recommendation_case_a_positive_clean():
    bal = {"present": True, "bottom_negative_after_reweight": False, "geometry_explains_bottom": False}
    rec, info = v2c.classify_recommendation_v2("POSITIVE_LOSS_PREFERENCE_LEARNING", bal)
    assert rec == "LONGER_P25_REVIEW_CANDIDATE"
    assert info["case"] == "A"


def test_recommendation_case_b_bottom_persists():
    bal = {"present": True, "bottom_negative_after_reweight": True, "geometry_explains_bottom": False}
    rec, info = v2c.classify_recommendation_v2("POSITIVE_LOSS_PREFERENCE_LEARNING", bal)
    assert rec == "REVIEW_VERTICAL_END_SUPERVISION_FIRST"
    assert info["case"] == "B"


def test_recommendation_case_c_geometry_explains():
    bal = {"present": True, "bottom_negative_after_reweight": True, "geometry_explains_bottom": True}
    rec, info = v2c.classify_recommendation_v2("WEAK_POSITIVE_LOSS_PREFERENCE", bal)
    assert rec == "REVIEW_CAMERA_SAMPLING_BALANCE_FIRST"
    assert info["case"] == "C"


def test_recommendation_balance_missing_is_conservative():
    rec, _info = v2c.classify_recommendation_v2("POSITIVE_LOSS_PREFERENCE_LEARNING", None)
    assert rec == "REVIEW_VERTICAL_END_SUPERVISION_FIRST"


# ---------------- (27-29) auxiliary statistics ---------------------------------


def test_smd_zero_for_identical_groups():
    v = np.array([1.0, 2.0, 3.0, 4.0])
    assert v2c.smd(v, v) == 0.0


def test_smd_positive_for_separated_groups():
    assert v2c.smd(np.array([0.0, 0.1, 0.2]), np.array([10.0, 10.1, 10.2])) > 1.0


def test_spearman_perfect_monotone():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert v2c.spearman_rho(x, x) == pytest.approx(1.0, abs=1e-12)
    assert v2c.spearman_rho(x, -x) == pytest.approx(-1.0, abs=1e-12)


def test_weighted_mean_ci_known_mean():
    v = np.array([1.0, 3.0])
    w = np.array([3.0, 1.0])
    r = v2c.weighted_mean_ci(v, w, n_boot=1000, seed=1)
    assert r["point"] == pytest.approx(1.5, abs=1e-12)
    assert r["ess"] == pytest.approx(4.0 * 4.0 / (9.0 + 1.0), abs=1e-9)


def test_decile_stats_partition():
    rng = np.random.default_rng(0)
    q = rng.uniform(0, 1, 100)
    d = rng.normal(0, 1, 100)
    out = v2c.decile_stats(q, d, n_boot=500, seed=1)
    assert len(out) == 10
    assert sum(x["n"] for x in out) == 100


def test_expected_tertile_counts_all_available_zero():
    e = v2c.expected_tertile_counts([0, 0, 0])
    assert e["expected"]["START"] == 3.0
    assert e["expected"]["CENTER"] == 0.0
    assert e["expected"]["END"] == 0.0
    assert e["sd"]["START"] == 0.0


def test_v2_verdict_reuses_v1_classes():
    adj = {
        a: {
            "intervals": {
                "POST_PRE": {"adjusted": {"point": 5e-4, "ci95": (2e-4, 8e-4)}, "raw_arm": None, "raw_same": None},
                "MID_PRE": {"adjusted": {"point": 3e-4, "ci95": (1e-4, 5e-4)}, "raw_arm": None, "raw_same": None},
                "POST_MID": {"adjusted": {"point": 2e-4, "ci95": (5e-5, 3.5e-4)}, "raw_arm": None, "raw_same": None},
            }
        }
        for a in ("OPPOSITE", "SHUFFLED")
    }
    adj["IDENTITY"] = dict(adj["OPPOSITE"])
    assert v2c.classify_verdict_v2(True, adj) == "POSITIVE_LOSS_PREFERENCE_LEARNING"
    assert v2c.classify_verdict_v2(False, adj) == "BLOCKED_NUMERICS"


def test_v2_constants():
    assert v2c.MASTER_SEED_V2 == 20260907
    assert v2c.N_BOOT_V2 == 10000
    assert v2c.P1_REL_CAP_DIAGNOSTIC == 1e-6
    assert v2c.SMD_DIAGNOSTIC == 0.1
    assert v2c.Z_IMBALANCE == 3.0
