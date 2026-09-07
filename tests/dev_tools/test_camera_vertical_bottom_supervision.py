"""Vertical Bottom Supervision Audit - regression tests.

Locks the pre-registered same-source mirrored-pair semantics (task spec
sections 10-12, 21-23, 26-30, 37, 39, 41-43, 45-46, 50) against
dev-tools/camera_coordinate_causal/vertical_bottom_supervision/contracts.py.

All tests are pure (numpy + stdlib; the PE MNN contract test imports torch,
available in the audit venv).  No HCU, no 34 GB evidence store, no network.
0 skip / 0 xfail by design.
"""
from __future__ import annotations

import inspect
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
VBS_DIR = REPO_ROOT / "dev-tools" / "camera_coordinate_causal" / "vertical_bottom_supervision"
sys.path.insert(0, str(VBS_DIR))

import contracts as C

# frozen production JLT t-strata (reports/camera-coordinate-causal-audit.json,
# verified bit-identical by the V2 numerics probe)
FROZEN_T_VALUES = (0.1388061520926247, 0.24819609741338816,
                   0.37948289980451305, 0.5560734456280833)


def make_unit(unit: int = 1, shard: str = "shard0", sample_id: int = 100,
              F: int = 384, k: int = 16, cohort: str = "camera",
              orientation: str = "vertical", opp_na: bool = False) -> dict:
    """Synthetic stage1-manifest unit row consistent with the production
    geometry formulas (F=round(zoom^2*256); s=k+128-F/2)."""
    available = F - 256
    s = C.signed_shift_of(k, F)
    return {
        "unit": unit,
        "source_shard": shard,
        "sample_id": sample_id,
        "cohort": cohort,
        "orientation": orientation,
        "opp_na": opp_na,
        "zoom": math.sqrt(F / 256.0),
        "norm_offset": k / available,
        "pixel_shift": s,
        "target": [256, 256],
        "latent_shift": abs(s) / 16.0,
    }


# ---------------- geometry (spec s10, s14) ----------------

def test_mirrored_k_is_available_minus_k():
    g = C.pair_geometry(make_unit(F=384, k=16))
    assert g["available"] == 128
    assert g["k_mirror"] == g["available"] - g["k_anchor"]  # k2 = available - k
    assert g["k_mirror"] == 112


def test_signed_shifts_exact_opposite():
    g = C.pair_geometry(make_unit(F=384, k=16))
    assert g["signed_shift_anchor"] == -48.0
    assert g["signed_shift_mirror"] == 48.0
    assert g["signed_shift_anchor"] == -g["signed_shift_mirror"]  # exact float


def test_same_zoom_and_full_canvas():
    u = make_unit(F=384, k=16)
    g = C.pair_geometry(u)
    assert g["full_height"] == C.full_height_from_zoom(u["zoom"]) == 384
    assert g["zoom"] == u["zoom"]  # same zoom on both sides by construction
    assert g["crop_box_top"][0] == g["crop_box_bottom"][0] == 0
    assert g["crop_box_top"][2] == g["crop_box_bottom"][2] == 256
    assert C.VIEWPORT == 256  # both crops are 256x256


def test_start_anchor_mirrors_to_end():
    g = C.pair_geometry(make_unit(F=384, k=16))
    assert g["anchor_tertile"] == "START"
    assert g["mirror_tertile"] == "END"
    assert g["k_start"] == 16 and g["k_end"] == 112


def test_end_anchor_mirrors_to_start():
    g = C.pair_geometry(make_unit(F=384, k=112))
    assert g["anchor_tertile"] == "END"
    assert g["mirror_tertile"] == "START"
    assert g["signed_shift_anchor"] == 48.0 and g["signed_shift_mirror"] == -48.0


def test_end_anchor_exactly_two_thirds_excluded():
    # available=126, k=84 -> mirror k=42 sits exactly at 1/3 of available
    # (3*42 == available) which is the CENTER/END boundary from the START
    # side -> mirror lands in CENTER, not START -> excluded from primary cohort
    u = make_unit(F=382, k=84)
    ok, reason = C.pair_eligible(u)
    assert ok is False
    assert reason == "mirror_not_start"


def test_pair_geometry_invariants():
    g = C.pair_geometry(make_unit(F=384, k=16))
    assert g["k_start"] + g["k_end"] == g["available"]
    assert abs(g["latent_shift_top"]) == abs(g["latent_shift_bottom"])
    assert g["crop_box_top"] == (0, 16, 256, 272)
    assert g["crop_box_bottom"] == (0, 112, 256, 368)
    assert g["norm_start"] + g["norm_end"] == pytest.approx(1.0)


def test_signed_shift_production_formula_real_unit():
    # V2 offset-balance verified real camera unit: F=380, k=15, shift=-47.0
    assert C.signed_shift_of(15, 380) == -47.0
    u = {
        "zoom": math.sqrt(380.0 / 256.0),
        "norm_offset": 0.121,  # manifest stores ROUNDED norm_offset
        "pixel_shift": -47.0,
        "target": [256, 256],
        "latent_shift": 47.0 / 16.0,
    }
    g = C.pair_geometry(u)
    assert g["k_anchor"] == 15
    assert g["full_height"] == 380
    assert g["anchor_tertile"] == "START" and g["mirror_tertile"] == "END"


def test_eligibility_exclusions():
    assert C.pair_eligible(make_unit(orientation="horizontal")) == (False, "not_vertical")
    assert C.pair_eligible(make_unit(opp_na=True)) == (False, "opposite_not_applicable")
    assert C.pair_eligible(make_unit(cohort="control")) == (False, "not_camera")
    assert C.pair_eligible(make_unit(F=384, k=64)) == (False, "center_anchor")
    ok, reason = C.pair_eligible(make_unit(F=384, k=16))
    assert ok is True and reason == "ok"


# ---------------- selection (spec s11-12) ----------------

def test_duplicate_source_dedup_deterministic():
    units = [
        make_unit(unit=9, sample_id=7),
        make_unit(unit=2, sample_id=7),
        make_unit(unit=5, sample_id=7),
        make_unit(unit=3, sample_id=8),
    ]
    uniq, removed = C.select_unique_pairs(units)
    assert removed == 2
    assert [u["unit"] for u in uniq] == [2, 3]  # smallest unit id kept


def test_stratified_selection_deterministic_whole_pool():
    units = [make_unit(unit=i, sample_id=100 + i) for i in range(30)]
    units += [make_unit(unit=100 + i, sample_id=200 + i, F=384, k=112) for i in range(10)]
    out = C.stratified_selection(units, target=512, minimum=8)
    assert out["status"] == "OK"
    assert out["n_selected"] == 40  # pool < target -> use the whole pool
    again = C.stratified_selection(units, target=512, minimum=8)
    assert [u["unit"] for u in again["selected"]] == [u["unit"] for u in out["selected"]]
    assert sum(out["allocation"].values()) == 40


def test_stratified_selection_allocation_largest_remainder():
    units = [make_unit(unit=i, sample_id=1000 + i) for i in range(400)]
    units += [make_unit(unit=1000 + i, sample_id=2000 + i, F=384, k=112) for i in range(200)]
    out = C.stratified_selection(units, target=512, minimum=256)
    assert out["status"] == "OK"
    # raw: 512*400/600 = 341.33, 512*200/600 = 170.67 -> floors 341/170,
    # the single leftover goes to the larger remainder (END)
    assert out["allocation"]["START|medium"] == 341
    assert out["allocation"]["END|medium"] == 171
    assert out["n_selected"] == 512


def test_pair_count_power_gate():
    units = [make_unit(unit=i, sample_id=1000 + i) for i in range(200)]
    out = C.stratified_selection(units)
    assert out["status"] == "STOP_BELOW_MINIMUM"
    assert out["n_pool"] == 200
    assert out["selected"] == []


# ---------------- band boundaries (spec s12, s41) ----------------

def test_tertile_boundaries():
    assert C.tertile_of(0.3333) == "START"
    assert C.tertile_of(1.0 / 3.0) == "CENTER"
    assert C.tertile_of(0.6) == "CENTER"
    assert C.tertile_of(2.0 / 3.0) == "END"
    assert C.tertile_of(0.9) == "END"


def test_zoom_band_boundaries():
    assert C.zoom_band(1.1999) == "mild"
    assert C.zoom_band(1.20) == "medium"
    assert C.zoom_band(1.3499) == "medium"
    assert C.zoom_band(1.35) == "strong"
    assert C.zoom_band(2.0) == "strong"


def test_latent_shift_bands():
    assert C.latent_shift_band(1.99) == "lt2"
    assert C.latent_shift_band(2.0) == "2to4"
    assert C.latent_shift_band(3.99) == "2to4"
    assert C.latent_shift_band(4.0) == "ge4"


# ---------------- mirror estimator fix (spec s42) ----------------

def test_mirrored_point_is_exact_mean_not_pooled():
    # spec fixture: top=[1,2,3], bottom=[0,0,0] -> point MUST be 2 (never 1)
    out = C.paired_mirror_difference([1.0, 2.0, 3.0], [0.0, 0.0, 0.0], n_boot=500, seed=C.BOOT_SEED)
    assert out["point"] == 2.0
    assert out["point"] != 1.0


def test_mirror_point_bootstrap_seed_invariant():
    a = C.paired_mirror_difference([1.0, 2.0, 3.0], [0.0, 0.0, 0.0], n_boot=2000, seed=20260907)
    b = C.paired_mirror_difference([1.0, 2.0, 3.0], [0.0, 0.0, 0.0], n_boot=2000, seed=999)
    assert a["point"] == b["point"] == 2.0  # exact observed mean, seed-blind


def test_old_pooled_sign_formula_intentionally_differs():
    # regression proof: the old pooled-sign formula gives 1.0 on the same
    # fixture where the correct paired point is 2.0
    old = C.pooled_sign_mean([1.0, 2.0, 3.0], [0.0, 0.0, 0.0])
    new = C.paired_mirror_difference([1.0, 2.0, 3.0], [0.0, 0.0, 0.0], n_boot=10, seed=1)
    assert old == 1.0
    assert new["point"] == 2.0
    assert old != new["point"]


def test_unequal_count_two_sample_mean_low_minus_high():
    # mean(low) - mean(high) = 2.25 - 0.0, not a pooled sign mean (which is 1.5)
    assert C.pooled_two_sample_difference([1.0, 2.0, 3.0, 3.0], [0.0, 0.0]) == 2.25
    assert C.pooled_sign_mean([1.0, 2.0, 3.0, 3.0], [0.0, 0.0]) == pytest.approx(1.5)


def test_paired_bootstrap_uses_same_source_index():
    top = [1.0, 5.0, 2.0, 9.0, 3.0]
    bot = [0.0, 1.0, 0.5, 0.0, 1.5]
    out = C.paired_mirror_difference(top, bot, n_boot=2000, seed=20260907)
    direct = C.bootstrap_ci([t - b for t, b in zip(top, bot)], n_boot=2000, seed=20260907)
    assert out["ci95"] == pytest.approx(direct)  # one shared resample matrix
    # pairing invariant: constant per-pair difference -> degenerate CI at c
    bot2 = [0.1, 0.2, 0.3, 0.4, 0.5]
    top2 = [b + 2.0 for b in bot2]
    deg = C.paired_mirror_difference(top2, bot2, n_boot=5000, seed=20260907)
    assert deg["ci95"][0] == pytest.approx(2.0, abs=1e-12)
    assert deg["ci95"][1] == pytest.approx(2.0, abs=1e-12)


# ---------------- margins / D / G (spec s21, s22, s26, s27) ----------------

def test_top_margin_sign():
    m = C.margin_from_losses({"TT": 1.0, "TB": 1.2, "BT": 0.9, "BB": 1.1,
                              "T_SAME": 1.0, "B_SAME": 1.1})
    # positive M_TOP = mirrored (wrong) coordinate costs more ->
    # correct view coordinate preferred
    assert m["m_top"] == pytest.approx(0.2)


def test_bottom_margin_sign():
    m = C.margin_from_losses({"TT": 1.0, "TB": 1.2, "BT": 0.9, "BB": 1.1,
                              "T_SAME": 1.0, "B_SAME": 1.1})
    assert m["m_bot"] == pytest.approx(-0.2)  # L_BT - L_BB = 0.9 - 1.1


def test_same_numerical_correction():
    m = {"PRE": {"m_top": 0.20, "m_bot": 0.05}, "MID": {"m_top": 0.30, "m_bot": 0.10}}
    f = {"PRE": {"f_top": 0.05, "f_bot": 0.01}, "MID": {"f_top": 0.15, "f_bot": 0.02}}
    d = C.adjusted_delta(m, f, "PRE", "MID")
    assert d["d_top"] == pytest.approx(0.0, abs=1e-15)  # 0.10 drift fully removed
    assert d["d_bot"] == pytest.approx(0.04)


def test_d_top_formula():
    m = {"a": {"m_top": 0.1, "m_bot": 0.0}, "b": {"m_top": 0.4, "m_bot": 0.0}}
    f = {"a": {"f_top": 0.0, "f_bot": 0.0}, "b": {"f_top": 0.0, "f_bot": 0.0}}
    d = C.adjusted_delta(m, f, "a", "b")
    assert d["d_top"] == pytest.approx(0.3)  # (M_b - M_a) - (F_b - F_a)


def test_d_bot_formula():
    m = {"a": {"m_top": 0.0, "m_bot": 0.05}, "b": {"m_top": 0.0, "m_bot": 0.11}}
    f = {"a": {"f_top": 0.0, "f_bot": 0.02}, "b": {"f_top": 0.0, "f_bot": 0.05}}
    d = C.adjusted_delta(m, f, "a", "b")
    assert d["d_bot"] == pytest.approx(0.03)  # (0.06) - (0.03)


def test_g_pair_formula():
    assert C.paired_gap({"d_top": 0.5, "d_bot": 0.2}) == pytest.approx(0.3)


def test_point_estimate_exact_observed_mean():
    vals = [1.1, -2.2, 3.3, 4.4, -5.5]
    out = C.paired_mirror_difference(vals, [0.0] * 5, n_boot=500, seed=20260907)
    assert out["point"] == float(np.mean(np.asarray(vals, dtype=np.float64)))
    assert C.point_ci(vals)["point"] == pytest.approx(float(np.mean(np.asarray(vals, np.float64))))


# ---------------- content-balanced subset (spec s37) ----------------

def test_content_balanced_subset_depends_only_on_content_fields():
    n = 10
    pairs = [{"pair_index": i, "zoom": 1.2 + 0.001 * i} for i in range(n)]
    asym = [float(i % 3) for i in range(n)]
    base = C.content_balanced_subset(pairs, asym)
    # causal-looking fields added to the rows must not change the subset
    pairs2 = [dict(p, G_pre_post=99.0, M_TOP_POST=5.0, d_top=-1.0) for p in pairs]
    assert C.content_balanced_subset(pairs2, asym) == base


def test_subset_cannot_read_causal_fields():
    sig = inspect.signature(C.content_balanced_subset)
    assert list(sig.parameters) == ["pairs", "asym", "fraction"]
    # the ONLY per-pair signal is the asymmetry scalar: equal asym -> index order
    pairs = [{"pair_index": i} for i in range(8)]
    idx = C.content_balanced_subset(pairs, [1.0] * 8)
    assert idx == [0, 1, 2, 3]


def test_clip_unavailable_fail_soft():
    # CLIP_TEXT = NOT_AVAILABLE -> asymmetry = |clip image delta| + |pe delta|
    # (binary-exact values so the ordering is deterministic across platforms)
    n = 8
    clip_img = [0.5, 0.125, 0.25, 0.125, 0.375, 0.25, 0.875, 0.125]
    pe_ret = [0.125, 0.125, 0.25, 0.625, 0.125, 0.25, 0.125, 0.625]
    asym = [abs(c) + abs(p) for c, p in zip(clip_img, pe_ret)]
    # asym: i0 0.625, i1 0.25, i2 0.5, i3 0.75, i4 0.5, i5 0.5, i6 1.0, i7 0.75
    pairs = [{"pair_index": i} for i in range(n)]
    idx = C.content_balanced_subset(pairs, asym, fraction=0.5)
    # lowest 4 by (asym, index): i1, then i2/i4/i5 (all 0.5, index order)
    assert idx == [1, 2, 4, 5]
    assert len(idx) == 4  # floor(8 * 0.5)


# ---------------- PE feature finiteness contract (spec s34) ----------------

def test_pe_feature_finite_contract():
    import feature_audit as FA
    import torch

    g = torch.Generator().manual_seed(7)
    crop = torch.randn(256, 768, generator=g)
    full = torch.randn(256, 768, generator=g)
    out = FA.mnn_match(crop, full)
    for key in ("n", "matched_cos", "inlier_ratio", "retained"):
        assert math.isfinite(out[key])
    assert 0.0 <= out["inlier_ratio"] <= 1.0
    assert 0.0 <= out["matched_cos"] <= 1.0 + 1e-6
    assert out["retained"] == pytest.approx(out["inlier_ratio"] * out["matched_cos"])
    # identity grid -> perfect mutual matching
    same = FA.mnn_match(crop, crop)
    assert same["inlier_ratio"] == 1.0
    assert same["matched_cos"] == pytest.approx(1.0, abs=1e-5)


# ---------------- gates (spec s27, s43) ----------------

def test_p3_false_passes_state_history_gate():
    doc = {
        "state": {"HIDDEN_MUTABLE_STATE_DETECTED": False,
                  "REPEAT_NUMERIC_JITTER_PRESENT": True},
        "p3": {"state_history_effect": False},
    }
    out = C.microprobe_gate(doc)
    assert out["pass"] is True
    assert out["p3_present"] is True
    assert out["repeat_jitter_present"] is True  # diagnostic only, never blocks


def test_p3_true_blocks_gate():
    doc = {
        "state": {"HIDDEN_MUTABLE_STATE_DETECTED": False,
                  "REPEAT_NUMERIC_JITTER_PRESENT": False},
        "p3": {"state_history_effect": True},
    }
    out = C.microprobe_gate(doc)
    assert out["pass"] is False
    assert out["p3_state_history_effect"] is True


def test_v2_hidden_state_false_required():
    doc = {
        "state": {"HIDDEN_MUTABLE_STATE_DETECTED": True,
                  "REPEAT_NUMERIC_JITTER_PRESENT": False},
        "p3": {"state_history_effect": False},
    }
    assert C.microprobe_gate(doc)["pass"] is False
    # committed report schema (nested) + flat legacy keys
    good = {"numerics": {"NUMERICS_V2_CLEAN": True},
            "posthoc_v2_verdict": "POSITIVE_LOSS_PREFERENCE_LEARNING"}
    bad = {"numerics": {"NUMERICS_V2_CLEAN": False},
           "posthoc_v2_verdict": "POSITIVE_LOSS_PREFERENCE_LEARNING"}
    assert C.v2_report_gate(good)["pass"] is True
    assert C.v2_report_gate(bad)["pass"] is False
    legacy = {"NUMERICS_V2_CLEAN": True,
              "POSTHOC_V2_VERDICT": "POSITIVE_LOSS_PREFERENCE_LEARNING"}
    assert C.v2_report_gate(legacy)["pass"] is True


def test_prior_sha_report_immutability(tmp_path):
    mp = {
        "state": {"HIDDEN_MUTABLE_STATE_DETECTED": False,
                  "REPEAT_NUMERIC_JITTER_PRESENT": True},
        "p3": {"state_history_effect": False},
    }
    mp_path = tmp_path / "posthoc-v2-microprobe.json"
    C.write_frozen(mp_path, mp)
    sha = C.sha256_file(mp_path)
    rep = {"NUMERICS_V2_CLEAN": True,
           "POSTHOC_V2_VERDICT": "POSITIVE_LOSS_PREFERENCE_LEARNING"}
    rep_path = tmp_path / "v2-review.json"
    C.write_frozen(rep_path, rep)
    out = C.validate_prior_numerics_v2(mp_path, rep_path, expected_microprobe_sha=sha)
    assert out["status"] == "PASS"
    # tamper the frozen microprobe -> sha mismatch -> STOP (never re-run)
    mp_path.write_text(mp_path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    out2 = C.validate_prior_numerics_v2(mp_path, rep_path, expected_microprobe_sha=sha)
    assert out2["status"] == "STOP_MICROPROBE_SHA_MISMATCH"
    # P3 true -> BLOCKED_NUMERICS (P3 is wired into the gate)
    mp3_path = tmp_path / "mp3.json"
    mp3 = {"state": {"HIDDEN_MUTABLE_STATE_DETECTED": False},
           "p3": {"state_history_effect": True}}
    C.write_frozen(mp3_path, mp3)
    out3 = C.validate_prior_numerics_v2(mp3_path, rep_path,
                                        expected_microprobe_sha=C.sha256_file(mp3_path))
    assert out3["status"] == "BLOCKED_NUMERICS"


# ---------------- classification / recommendation (spec s45-46) ----------------

def _gap(point, lo, hi):
    return {"point": point, "ci95": [lo, hi]}


def test_classify_scenarios_A_F():
    hist = _gap(2.4e-3, 1.9e-3, 3.0e-3)
    # A: same-source collapses (CI has 0), natural gap clear
    v, info = C.classify(n_pairs=512,
                         gap_all={"PRE|POST": _gap(1e-4, -1e-3, 1e-3)},
                         gap_balanced=None, historical_gap=hist, spearman=None)
    assert v == "NATURAL_COHORT_CONFOUNDING" and info["case"] == "A"
    # E: no gap anywhere
    v, info = C.classify(n_pairs=512,
                         gap_all={"PRE|POST": _gap(1e-4, -1e-3, 1e-3)},
                         gap_balanced=None, historical_gap=_gap(0.0, -1e-3, 1e-3),
                         spearman=None)
    assert v == "NO_PAIRED_ASYMMETRY" and info["case"] == "E"
    # B: gap persists, balanced subset collapses
    v, info = C.classify(n_pairs=512,
                         gap_all={"PRE|POST": _gap(2e-3, 1e-3, 3e-3)},
                         gap_balanced={"PRE|POST": _gap(5e-4, -1e-4, 1e-3)},
                         historical_gap=hist, spearman=0.1)
    assert v == "CONTENT_CROP_ASYMMETRY_SUPPORTED" and info["case"] == "B"
    # B (attenuation >= 0.5 with balanced gap still > 0)
    v, info = C.classify(n_pairs=512,
                         gap_all={"PRE|POST": _gap(2e-3, 1e-3, 3e-3)},
                         gap_balanced={"PRE|POST": _gap(1e-3, 5e-4, 1.5e-3)},
                         historical_gap=hist, spearman=0.1)
    assert v == "CONTENT_CROP_ASYMMETRY_SUPPORTED" and info["case"] == "B"
    assert info["attenuation"] == pytest.approx(0.5)
    # C: gap persists, balanced persists, no content explanation
    v, info = C.classify(n_pairs=512,
                         gap_all={"PRE|POST": _gap(2e-3, 1e-3, 3e-3)},
                         gap_balanced={"PRE|POST": _gap(1.9e-3, 1.2e-3, 2.6e-3)},
                         historical_gap=hist, spearman=0.1)
    assert v == "DIRECTIONAL_MECHANISM_ASYMMETRY_CONFIRMED" and info["case"] == "C"
    # D: same but a consistent content relationship
    v, info = C.classify(n_pairs=512,
                         gap_all={"PRE|POST": _gap(2e-3, 1e-3, 3e-3)},
                         gap_balanced={"PRE|POST": _gap(1.9e-3, 1.2e-3, 2.6e-3)},
                         historical_gap=hist, spearman=0.35)
    assert v == "MIXED_CONTENT_AND_DIRECTIONAL_ASYMMETRY" and info["case"] == "D"
    # F: insufficient power
    v, info = C.classify(n_pairs=200,
                         gap_all={"PRE|POST": _gap(2e-3, 1e-3, 3e-3)},
                         gap_balanced=None, historical_gap=hist, spearman=None)
    assert v == "INCONCLUSIVE" and info["case"] == "F"
    # F: gap persists but no balanced subset available
    v, info = C.classify(n_pairs=512,
                         gap_all={"PRE|POST": _gap(2e-3, 1e-3, 3e-3)},
                         gap_balanced=None, historical_gap=hist, spearman=None)
    assert v == "INCONCLUSIVE" and info["case"] == "F"


def test_recommendation_mapping():
    assert C.recommendation_for("NATURAL_COHORT_CONFOUNDING") == "REVIEW_EVALUATION_COHORT"
    assert C.recommendation_for("CONTENT_CROP_ASYMMETRY_SUPPORTED") == "REVIEW_CROP_CONTENT_SUPERVISION"
    assert C.recommendation_for("DIRECTIONAL_MECHANISM_ASYMMETRY_CONFIRMED") == "REVIEW_VERTICAL_COORDINATE_SUPERVISION"
    assert C.recommendation_for("MIXED_CONTENT_AND_DIRECTIONAL_ASYMMETRY") == "REVIEW_CROP_AND_VERTICAL_SUPERVISION"
    assert C.recommendation_for("NO_PAIRED_ASYMMETRY") == "LONGER_P25_REVIEW_CANDIDATE"
    assert C.recommendation_for("INCONCLUSIVE") == "NO_MORE_EXPOSURE_YET"


# ---------------- statistics helpers ----------------

def test_t_values_exact_frozen():
    assert C.t_values() == FROZEN_T_VALUES


def test_noise_seed_structure():
    assert C.noise_seed(0, 0) == C.MASTER_SEED_VBS * 1_000_003
    assert C.noise_seed(1, 0) - C.noise_seed(0, 0) == 100
    assert C.noise_seed(0, 1) - C.noise_seed(0, 0) == 1
    seeds = {C.noise_seed(i, k) for i in range(512) for k in range(4)}
    assert len(seeds) == 512 * 4


def test_bootstrap_ci_determinism():
    a = C.bootstrap_ci([1.0, 2.0, 3.0, 4.0, 5.0], n_boot=1000, seed=20260907)
    b = C.bootstrap_ci([1.0, 2.0, 3.0, 4.0, 5.0], n_boot=1000, seed=20260907)
    assert a == b
    assert C.point_ci([1.0, 2.0, 3.0])["point"] == 2.0


def test_ci_excl0_and_sign():
    assert C.ci_excl0([0.1, 0.2]) is True
    assert C.ci_excl0([-0.1, 0.1]) is False
    assert C.ci_excl0([-0.2, -0.1]) is True
    assert C.ci_sign([0.1, 0.2]) == 1
    assert C.ci_sign([-0.2, -0.1]) == -1
    assert C.ci_sign([-0.1, 0.1]) == 0


def test_spearman_known():
    assert C.spearman_rho([1, 2, 3, 4, 5], [2, 4, 6, 8, 10]) == pytest.approx(1.0)
    assert C.spearman_rho([1, 2, 3, 4, 5], [10, 8, 6, 4, 2]) == pytest.approx(-1.0)
    assert C.spearman_rho([1, 1, 1, 1, 1], [1, 2, 3, 4, 5]) == 0.0
    # tie averaging: perfectly anti-aligned ranks -> 0 with ties
    assert C.spearman_rho([1, 1, 2, 2], [1, 2, 1, 2]) == pytest.approx(0.0)


def test_quartile_bins():
    bins = C.quartile_bins([float(i) for i in range(100)])
    assert bins.count(0) == 25
    assert bins.count(1) == 25
    assert bins.count(2) == 25
    assert bins.count(3) == 25


def test_write_frozen_sha_stability(tmp_path):
    p1 = tmp_path / "a.json"
    p2 = tmp_path / "b.json"
    doc = {"x": 1, "y": [1, 2, 3], "z": "s"}
    s1 = C.write_frozen(p1, doc)
    s2 = C.write_frozen(p2, doc)
    assert s1 == s2 == C.sha256_file(p1)
    assert json.loads(p1.read_text(encoding="utf-8")) == doc
