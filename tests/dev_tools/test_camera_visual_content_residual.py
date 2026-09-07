"""Visual Content Residual Audit - regression tests (Camera Viewport v2).

Locks the pre-registered semantics (task spec sections 10-12, 14-19, 21-27,
29-33, 35, 37, 39-41):

  * percentile-rank / average-tie semantics and the composite A_visual
    (CLIP image + PE only; CLIP text never enters the primary score),
  * geometry-stratified subset construction with exact proportional
    largest-remainder quotas and deterministic (score, pair_index) tie
    breaks; the subset builders accept no mirror-statistic input,
  * static phase separation: the PHASE A freeze source must not reference
    any per-pair mirror statistic or prior report,
  * source-pair bootstrap: exact observed-mean point, seed-invariant,
  * retention / attenuation (negative allowed, never clipped),
  * the classification tree and the recommendation mapping,
  * exclusion of the prior-generation legacy helper,
  * immutability pins: every prior report / test / VBS tooling file is
    hash-pinned to the reviewed base tree,
  * the frozen pair-manifest identity.

Pure tests: numpy + stdlib only.  No HCU, no network, no model assets.
0 skip / 0 xfail by design (>= 37 cases).
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
VCR_DIR = REPO_ROOT / "dev-tools" / "camera_coordinate_causal" / "visual_content_residual"

# ---------------------------------------------------------------------------
# module loading: the prior VBS suite also imports a bare module named
# "contracts" from its own directory; drop any cached copy so this suite
# binds the VCR contracts (this test file is collected last alphabetically).
# ---------------------------------------------------------------------------
sys.modules.pop("contracts", None)
sys.modules.pop("freeze_visual_subsets", None)
VCR_PATH = str(VCR_DIR)
if VCR_PATH not in sys.path:
    sys.path.insert(0, VCR_PATH)

import contracts as C
import freeze_visual_subsets as F

PINNED_PRIOR_FILES = {
    'reports/camera-vertical-bottom-supervision-audit.md': 'cd8501005b9512df617f7e70394ea8ac03aca89a39da7c5b688b38695b15d6a3',
    'reports/camera-vertical-bottom-supervision-audit.json': 'edaf361fa0b3df40d29acdf6ce45f602e842f71a681d1b8f231c6de274a18c1b',
    'reports/camera-vertical-bottom-supervision-metrics.json': 'c31e988e887138c5085e78dfc8841e060281c6936e3cca99cc6d33bdf1e5f392',
    'reports/camera-vertical-bottom-supervision-pairs.csv': '1a08b68de84db872e67b3fd3044929884d89c1ae2b6e8b317611be4f1221c4a8',
    'reports/camera-vertical-bottom-supervision-copy-report.md': 'd88f531ae20f79ddd65cb53be6d25a71f9c1218b7406ac13d39404fbc2804a55',
    'tests/dev_tools/test_camera_coordinate_causal_audit_contract.py': '77041fa1cf1d29e368dc47410fbbb6bb412597337952e66175a5c832886907a2',
    'tests/dev_tools/test_camera_coordinate_causal_posthoc.py': '62e34cf6f6f162097aa3a256701e7c0f010520a8ccdb7b2086e5e021a08ec67f',
    'tests/dev_tools/test_camera_coordinate_causal_posthoc_v2.py': '2d84b3766d70179372d8f73038e8ab5ae63b04e92eb9cdac278c546d7022f0ba',
    'tests/dev_tools/test_camera_vertical_bottom_supervision.py': '58e2f7076369b2e0ca64e08cc45fcfc9d0b1badf48ba1277919ddb27b4e4fbef',
    'dev-tools/camera_coordinate_causal/vertical_bottom_supervision/README.md': '76ab70d5c966776b51b98c8c8af5538398b1ee19b12ae0cbcad08e81d87392f8',
    'dev-tools/camera_coordinate_causal/vertical_bottom_supervision/analyze.py': 'a58097ee641da669eba23ef6bcb0cf1e61b50eba41d564b6f1a65ea9bdfd35a4',
    'dev-tools/camera_coordinate_causal/vertical_bottom_supervision/build_pairs.py': '8ac1e432cb4aca66290ae0007154fd80a1eea4111fb361ee0714c2deac271e7c',
    'dev-tools/camera_coordinate_causal/vertical_bottom_supervision/contracts.py': '527d865b6508f7a1cebbb5e6d2fb3d11d096128fca9142af1ee1a62872b84341',
    'dev-tools/camera_coordinate_causal/vertical_bottom_supervision/encode_pairs.py': '7429b7d653ab0d22a8dd6b5d851c6f512630600881d4f24c9cbcf1b6d25fe891',
    'dev-tools/camera_coordinate_causal/vertical_bottom_supervision/feature_audit.py': '248b427ebe0f49fa959fd04a6bac4c06ba50bc75546e462ddaa9425b75e8b2aa',
    'dev-tools/camera_coordinate_causal/vertical_bottom_supervision/pe_ref/LICENSE.PE': 'c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4',
    'dev-tools/camera_coordinate_causal/vertical_bottom_supervision/pe_ref/NOTICE': 'd45b8b66b694f3d00f177d080923a2eb6c8fd987b8c0c091a0dfd37c367e899b',
    'dev-tools/camera_coordinate_causal/vertical_bottom_supervision/pe_ref/__init__.py': '1ef7b351b39361d83fbbc3a6cf472498ad4c1c7f17f4a0dd9a72f90ce19551d1',
    'dev-tools/camera_coordinate_causal/vertical_bottom_supervision/pe_ref/config.py': '7581f627a1dda3eab6c1103c201150ab025fd1c8c871333e986a4ba64c7c584f',
    'dev-tools/camera_coordinate_causal/vertical_bottom_supervision/pe_ref/drop_path.py': '062823621c86d74424095fbba9a6b6ff9139a7df0d31fc00f5038ec290166d09',
    'dev-tools/camera_coordinate_causal/vertical_bottom_supervision/pe_ref/pe_vision.py': '226f4b94e2391bcda901e6187b4bb1e939e0bb1b9039fc1bd5c763edaee31c2a',
    'dev-tools/camera_coordinate_causal/vertical_bottom_supervision/pe_ref/rope.py': '55c41f553834f210c08454fc2f0c8a231d2b28f8dc3e5c56dbef0b8f74763571',
    'dev-tools/camera_coordinate_causal/vertical_bottom_supervision/score_pairs.py': '32446b1bc4f478a6b090499b22481739948f874cf474184aea98e8c01f11db8a'
}

FORBIDDEN_IN_FREEZE_SOURCE = (
    "causal",
    "vertical-bottom",
    "supervision-audit",
    "pairs.csv",
    "ledger",
    "margins",
    "losses",
    "G_PRE",
    "G_MID",
)

# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

ZOOM_VAL = {"mild": 1.15, "medium": 1.28, "strong": 1.40}
LATENT_VAL = {"lt2": 1.0, "2to4": 3.0, "ge4": 5.0}


def fixture_pairs(n: int = 512):
    """512 pairs spread deterministically over the 18 geometry cells."""
    sides = ("START", "END")
    zooms = ("mild", "medium", "strong")
    lats = ("lt2", "2to4", "ge4")
    cells = [(s, z, l) for s in sides for z in zooms for l in lats]
    base = n // len(cells)  # 28
    extra = n - base * len(cells)  # 8 -> first `extra` cells get +1
    pairs = []
    i = 0
    for c, cell in enumerate(cells):
        cnt = base + (1 if c < extra else 0)
        for _ in range(cnt):
            pairs.append(
                {
                    "pair_index": i,
                    "original_side": cell[0],
                    "equivalent_zoom": ZOOM_VAL[cell[1]],
                    "latent_shift": LATENT_VAL[cell[2]],
                }
            )
            i += 1
    return pairs


def fixture_manifest(pairs=None) -> dict:
    return {"pairs": pairs if pairs is not None else fixture_pairs()}


def make_clip_doc(n=512, text_values=None):
    rng = np.random.default_rng(11)
    top = 0.90 + rng.random(n) * 0.05
    bot = 0.85 + rng.random(n) * 0.05
    img_rows = [
        {
            "pair_index": i,
            "sim_full_top": float(top[i]),
            "sim_full_bot": float(bot[i]),
            "delta": float(top[i] - bot[i]),
        }
        for i in range(n)
    ]
    doc = {"text_status": "AVAILABLE", "image_rows": img_rows}
    tv = text_values if text_values is not None else rng.random(n)
    doc["text_rows"] = [
        {
            "pair_index": i,
            "sim_text_top": float(0.1 + tv[i]),
            "sim_text_bot": float(0.08 + 0.5 * tv[i]),
            "delta": float(0.1 + tv[i] - 0.08 - 0.5 * tv[i]),
            "text_len": 768,
        }
        for i in range(n)
    ]
    return doc


def make_pe_doc(n=512):
    rng = np.random.default_rng(22)
    ret_top = 0.40 + rng.random(n) * 0.10
    ret_bot = 0.38 + rng.random(n) * 0.10
    rows = [
        {
            "pair_index": i,
            "top": {"retained": float(ret_top[i]), "matched_cos": 0.9, "inlier_ratio": 0.5, "n": 100},
            "bottom": {"retained": float(ret_bot[i]), "matched_cos": 0.9, "inlier_ratio": 0.5, "n": 100},
            "delta": {
                "retained": float(ret_top[i] - ret_bot[i]),
                "matched_cos": 0.0,
                "inlier_ratio": 0.0,
            },
        }
        for i in range(n)
    ]
    return {"rows": rows}


def fixture_scores(n=512, seed=7):
    rng = np.random.default_rng(seed)
    return np.sort(rng.random(n))


def fixture_strata(pairs=None) -> list:
    pairs = pairs if pairs is not None else fixture_pairs()
    return [C.geometry_stratum(p) for p in pairs]


# ---------------------------------------------------------------------------
# 1-4, 38-39: ranks / composite / strata
# ---------------------------------------------------------------------------


def test_average_rank_ties():
    # 1-based average ranks 1, 2.5, 2.5, 4 -> percentile (rank-1)/(n-1)
    ranks = C.percentile_rank_average_ties([1.0, 2.0, 2.0, 3.0])
    assert ranks[0] == pytest.approx(0.0)
    assert ranks[1] == pytest.approx(0.5)
    assert ranks[2] == pytest.approx(0.5)
    assert ranks[3] == pytest.approx(1.0)


def test_percentile_rank_endpoints():
    v = [3.0, 1.0, 2.0, 4.0]  # unique min and max
    r = C.percentile_rank_average_ties(v)
    assert r.min() == pytest.approx(0.0)
    assert r.max() == pytest.approx(1.0)
    # tied minima share the (non-zero) average-tie rank, strictly inside (0, 1)
    r2 = C.percentile_rank_average_ties([3.0, 1.0, 2.0, 1.0])
    assert r2[1] == r2[3]
    assert 0.0 < r2[1] < 1.0


def test_a_visual_only_clip_image_and_pe():
    """A_visual must depend only on the two visual metrics (spec 2/12)."""
    manifest = fixture_manifest()
    clip_a = make_clip_doc(text_values=np.arange(512) * 0.001)
    clip_b = make_clip_doc(text_values=np.ones(512) * 0.7)
    pe = make_pe_doc()
    pp_a = F.build_per_pair(manifest, clip_a, pe, 512)
    pp_b = F.build_per_pair(manifest, clip_b, pe, 512)
    assert np.array_equal(pp_a["a_visual"], pp_b["a_visual"])
    assert np.array_equal(pp_a["r_img"], pp_b["r_img"])
    assert np.array_equal(pp_a["r_pe"], pp_b["r_pe"])
    for i in (0, 127, 511):
        rec = pp_a["per_pair"][str(i)]
        assert rec["A_visual"] == pytest.approx(0.5 * (rec["R_img"] + rec["R_pe"]))


def test_clip_text_excluded_from_primary_score():
    """Changing only the CLIP text features must leave A_visual untouched."""
    manifest = fixture_manifest()
    clip = make_clip_doc()
    clip2 = dict(clip)
    clip2["text_rows"] = [dict(r, sim_text_top=0.0, sim_text_bot=0.0, delta=0.0) for r in clip["text_rows"]]
    pe = make_pe_doc()
    a1 = F.build_per_pair(manifest, clip, pe, 512)["a_visual"]
    a2 = F.build_per_pair(manifest, clip2, pe, 512)["a_visual"]
    assert np.array_equal(a1, a2)


def test_geometry_stratum_definitions():
    assert (
        C.geometry_stratum({"original_side": "START", "equivalent_zoom": 1.15, "latent_shift": 1.0})
        == "START|mild|lt2"
    )
    assert (
        C.geometry_stratum({"original_side": "END", "equivalent_zoom": 1.25, "latent_shift": 2.5})
        == "END|medium|2to4"
    )
    assert (
        C.geometry_stratum({"original_side": "END", "equivalent_zoom": 1.5, "latent_shift": 4.0})
        == "END|strong|ge4"
    )
    # frozen boundary semantics
    assert C.zoom_band(1.20) == "medium"
    assert C.zoom_band(1.35) == "strong"
    assert C.latent_shift_band(2.0) == "2to4"
    assert C.latent_shift_band(4.0) == "ge4"


# ---------------------------------------------------------------------------
# 4-5, 8-11: subset construction
# ---------------------------------------------------------------------------


def test_subset_builder_interface_has_no_mirror_statistic():
    for fn in (
        C.geometry_stratified_select,
        C.global_lowest_select,
        C.proportional_largest_remainder,
        C.lowest_index_set,
        C.joint_lowest_intersection,
        F.build_subsets,
    ):
        params = inspect.signature(fn).parameters
        for name in params:
            assert name not in ("g", "g_pre_post", "causal", "mirror", "losses", "margins"), (
                fn.__name__,
                name,
            )


def test_freeze_source_has_no_mirror_statistic_dependency():
    src = (VCR_DIR / "freeze_visual_subsets.py").read_text(encoding="utf-8")
    for token in FORBIDDEN_IN_FREEZE_SOURCE:
        assert token not in src, f"forbidden token {token!r} in freeze source"


def test_largest_remainder_quotas_proportional():
    sizes = {"A": 10, "B": 10}
    q = C.proportional_largest_remainder(sizes, 3)
    assert sum(q.values()) == 3
    assert q == {"A": 2, "B": 1}  # equal remainders break by cell name
    sizes2 = {"A": 28, "B": 29, "C": 30, "D": 31}
    q2 = C.proportional_largest_remainder(sizes2, 20)
    assert sum(q2.values()) == 20
    raw = {c: s * 20 / 118 for c, s in sizes2.items()}
    for c in sizes2:
        assert q2[c] in (int(np.floor(raw[c])), int(np.floor(raw[c])) + 1)


def test_largest_remainder_deterministic_under_key_order():
    sizes = {"a|b": 51, "c|d": 52, "e|f": 49}
    q1 = C.proportional_largest_remainder(sizes, 40)
    q2 = C.proportional_largest_remainder(dict(reversed(list(sizes.items()))), 40)
    assert q1 == q2
    assert sum(q1.values()) == 40


def test_v50_exact_256_and_v25_exact_128():
    scores = fixture_scores()
    strata = fixture_strata()
    v50 = C.geometry_stratified_select(scores, strata, C.V50_TARGET)
    v25 = C.geometry_stratified_select(scores, strata, C.V25_TARGET)
    assert len(v50) == 256
    assert len(v25) == 128
    assert all(0 <= i < 512 for i in v50 + v25)
    assert set(v25) <= set(v50)  # the 25% set is the strictest core


def test_geometry_stratum_composition_matches_quota():
    scores = fixture_scores()
    pairs = fixture_pairs()
    strata = fixture_strata(pairs)
    v50 = C.geometry_stratified_select(scores, strata, 256)
    cells: dict = {}
    for i, s in enumerate(strata):
        cells.setdefault(s, []).append(i)
    quotas = C.proportional_largest_remainder({c: len(v) for c, v in cells.items()}, 256)
    for cell, members in cells.items():
        member_set = set(members)
        got = sum(1 for i in v50 if i in member_set)
        assert got == quotas[cell], (cell, got, quotas[cell])


def test_subset_tie_break_by_pair_index():
    scores = np.zeros(512)
    strata = fixture_strata()
    v50 = C.geometry_stratified_select(scores, strata, 256)
    cells: dict = {}
    for i, s in enumerate(strata):
        cells.setdefault(s, []).append(i)
    quotas = C.proportional_largest_remainder({c: len(v) for c, v in cells.items()}, 256)
    for cell, members in cells.items():
        member_set = set(members)
        expected = sorted(members)[: quotas[cell]]
        got = [i for i in v50 if i in member_set]
        assert got == expected, cell
    g = C.global_lowest_select(np.ones(512), 256)
    assert g == list(range(256))


# ---------------------------------------------------------------------------
# 12-14, 15-17: bootstrap / retention / attenuation
# ---------------------------------------------------------------------------


def test_bootstrap_point_is_exact_observed_mean():
    v = [0.1, 0.2, 0.3, 0.4, 0.5]
    res = C.source_pair_bootstrap_ci(v, n_boot=200, seed=20260907)
    assert res["point"] == float(np.mean(v))


def test_bootstrap_seed_does_not_change_point():
    v = [0.1, 0.2, 0.3, 0.4, 0.5]
    r1 = C.source_pair_bootstrap_ci(v, n_boot=200, seed=20260907)
    r2 = C.source_pair_bootstrap_ci(v, n_boot=200, seed=999)
    assert r1["point"] == r2["point"]
    assert r1["n_boot"] == r2["n_boot"] == 200


def test_source_pair_bootstrap_bounds():
    v = [0.0, 0.0, 2.0, 2.0]
    res = C.source_pair_bootstrap_ci(v, n_boot=500, seed=20260907)
    lo, hi = res["ci95"]
    assert 0.0 <= lo <= hi <= 2.0
    assert res["n_boot"] == 500
    assert res["seed"] == 20260907


def test_attenuation_formula():
    g_all = {"point": 1.0}
    g_sub = {"point": 0.4}
    assert C.attenuation(g_sub, g_all) == pytest.approx(0.6)


def test_negative_attenuation_allowed_not_clipped():
    g_all = {"point": 1.0}
    g_sub = {"point": 1.5}
    assert C.attenuation(g_sub, g_all) == pytest.approx(-0.5)


def test_retention_formula():
    g_all = {"point": -2.0}
    g_sub = {"point": 0.8}
    assert C.retention(g_sub, g_all) == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# 18-19: signed content semantics
# ---------------------------------------------------------------------------


def test_signed_clip_image_semantics():
    manifest = fixture_manifest()
    clip = make_clip_doc()
    pe = make_pe_doc()
    pp = F.build_per_pair(manifest, clip, pe, 512)
    for i in (0, 1, 2):
        row = clip["image_rows"][i]
        assert pp["c_img"][i] == pytest.approx(row["sim_full_top"] - row["sim_full_bot"])


def test_signed_pe_semantics():
    manifest = fixture_manifest()
    clip = make_clip_doc()
    pe = make_pe_doc()
    pp = F.build_per_pair(manifest, clip, pe, 512)
    for i in (0, 1, 2):
        row = pe["rows"][i]
        assert pp["c_pe"][i] == pytest.approx(row["top"]["retained"] - row["bottom"]["retained"])


# ---------------------------------------------------------------------------
# 20-24: balance / correlations / quartiles / joint intersection
# ---------------------------------------------------------------------------


def test_visual_balance_lowers_a_visual():
    manifest = fixture_manifest()
    clip = make_clip_doc()
    pe = make_pe_doc()
    pp = F.build_per_pair(manifest, clip, pe, 512)
    v50 = C.geometry_stratified_select(pp["a_visual"], pp["strata"], 256)
    mean_all = float(pp["a_visual"].mean())
    mean_v50 = float(pp["a_visual"][np.asarray(v50, dtype=int)].mean())
    assert mean_v50 < mean_all


def test_spearman_known_positive():
    x = np.arange(1, 11, dtype=float)
    assert C.spearman_rho(x, x ** 3) == pytest.approx(1.0, abs=1e-12)


def test_spearman_known_negative():
    x = np.arange(1, 11, dtype=float)
    assert C.spearman_rho(x, -x ** 2) == pytest.approx(-1.0, abs=1e-12)


def test_quartile_assignment_deterministic():
    scores = [float((i * 13) % 50) for i in range(512)]
    q1 = C.quartile_order(scores)
    q2 = C.quartile_order(list(scores))
    assert [list(c) for c in q1] == [list(c) for c in q2]
    assert all(len(c) == 128 for c in q1)
    assert max(scores[i] for i in q1[0]) <= min(scores[i] for i in q1[3])
    flat = [i for c in q1 for i in c]
    assert sorted(flat) == list(range(512))


def test_joint_low_intersection():
    n = 512
    a = np.arange(n, dtype=float)  # lowest 256 -> indices 0..255
    b = np.array([float(i - 10) if i >= 10 else float(i + 1000) for i in range(n)])
    inter = C.joint_lowest_intersection(a, b, 256)
    assert inter == set(range(10, 256))


# ---------------------------------------------------------------------------
# 25-33: classification / recommendation
# ---------------------------------------------------------------------------


def _block(point, lo, hi):
    return {"point": float(point), "ci95": [float(lo), float(hi)]}


def test_residual_confirmed():
    v50 = _block(0.002, 0.001, 0.003)
    v25 = _block(0.0016, 0.001, 0.002)
    assert C.classify_directional_residual(v50, v25, 0.8) == "CONFIRMED"


def test_residual_weak():
    v50 = _block(0.002, 0.001, 0.003)
    v25_cross = _block(0.001, -0.0005, 0.0025)
    assert C.classify_directional_residual(v50, v25_cross, 0.8) == "WEAK"
    v25_pos = _block(0.001, 0.0005, 0.002)
    assert C.classify_directional_residual(v50, v25_pos, 0.4) == "WEAK"


def test_residual_not_detected():
    v50 = _block(1e-5, -0.001, 0.001)
    v25 = _block(2e-5, -0.001, 0.001)
    assert C.classify_directional_residual(v50, v25, 0.5) == "NOT_DETECTED"


def test_visual_strong_supported():
    ci_img = _block(0.02, 0.01, 0.03)
    ci_pe = _block(0.009, 0.005, 0.013)
    assert C.classify_visual_contribution(ci_img, ci_pe, 0.5, 0.4, 0.6) == "STRONG_SUPPORTED"


def test_visual_supported():
    ci_img = _block(0.02, 0.01, 0.03)
    ci_pe = _block(0.0, -0.01, 0.01)
    assert C.classify_visual_contribution(ci_img, ci_pe, 0.25, 0.1, 0.3) == "SUPPORTED"


def test_overall_mixed():
    assert C.classify_overall("CONFIRMED", "STRONG_SUPPORTED") == "MIXED_VISUAL_CONTENT_AND_DIRECTIONAL"
    assert C.classify_overall("CONFIRMED", "SUPPORTED") == "MIXED_VISUAL_CONTENT_AND_DIRECTIONAL"


def test_overall_directional():
    assert C.classify_overall("CONFIRMED", "WEAK") == "DIRECTIONAL_RESIDUAL_CONFIRMED"
    assert C.classify_overall("CONFIRMED", "NOT_SUPPORTED") == "DIRECTIONAL_RESIDUAL_CONFIRMED"


def test_overall_visual_explains():
    assert C.classify_overall("NOT_DETECTED", "STRONG_SUPPORTED") == "VISUAL_CONTENT_EXPLAINS_SUBSTANTIAL_GAP"
    assert C.classify_overall("WEAK", "STRONG_SUPPORTED") == "VISUAL_CONTENT_EXPLAINS_SUBSTANTIAL_GAP"


def test_recommendation_mapping():
    expected = {
        "DIRECTIONAL_RESIDUAL_CONFIRMED": "DESIGN_VERTICAL_MIRROR_BALANCED_SUPERVISION",
        "MIXED_VISUAL_CONTENT_AND_DIRECTIONAL": "DESIGN_VERTICAL_MIRROR_BALANCED_SUPERVISION_WITH_CONTENT_GUARDS",
        "VISUAL_CONTENT_EXPLAINS_SUBSTANTIAL_GAP": "REVIEW_CROP_CONTENT_SUPERVISION",
        "NO_DIRECTIONAL_RESIDUAL_DETECTED": "LONGER_P25_REVIEW_CANDIDATE",
        "INCONCLUSIVE": "NO_MORE_EXPOSURE_YET",
    }
    for overall, value in expected.items():
        rec = C.recommendation_for(overall)
        assert rec["value"] == value
        assert rec["longer_p25_authorized"] is False
        assert rec["p50_authorized"] is False


# ---------------------------------------------------------------------------
# 34-37: legacy exclusion / immutability pins / identity pins
# ---------------------------------------------------------------------------


def test_legacy_helper_not_imported():
    import re

    for name in ("contracts.py", "freeze_visual_subsets.py", "analyze.py"):
        src = (VCR_DIR / name).read_text(encoding="utf-8")
        assert not re.search(r"^(from\s+\S+\s+)?import\s+.*pooled_sign_mean", src, flags=re.MULTILINE), (
            f"legacy helper imported in {name}"
        )
        assert not re.search(r"pooled_sign_mean\s*\(", src), f"legacy helper called in {name}"
    for name in ("contracts.py", "freeze_visual_subsets.py"):
        assert "pooled_sign_mean" not in (VCR_DIR / name).read_text(encoding="utf-8")
    assert hasattr(C, "paired_mean_difference")
    assert hasattr(C, "two_sample_mean_difference")
    assert C.paired_mean_difference([1.0, 2.0, 3.0], [0.0, 0.0, 0.0]) == pytest.approx(2.0)
    assert C.two_sample_mean_difference([1.0, 2.0, 3.0, 3.0], [0.0, 0.0]) == pytest.approx(2.25)


def test_prior_reports_untouched_hashes():
    report_files = [(rel, sha) for rel, sha in PINNED_PRIOR_FILES.items() if rel.startswith("reports/")]
    assert len(report_files) == 5
    for rel, sha in report_files:
        assert C.sha256_of_file(REPO_ROOT / rel) == sha, rel


def test_prior_tooling_untouched_hashes():
    tool_files = [
        (rel, sha)
        for rel, sha in PINNED_PRIOR_FILES.items()
        if rel.startswith(("dev-tools/", "tests/"))
    ]
    assert len(tool_files) == 18  # 14 VBS tooling files + 4 prior test suites
    for rel, sha in tool_files:
        assert C.sha256_of_file(REPO_ROOT / rel) == sha, rel


def test_pair_manifest_sha_pin():
    assert C.PAIR_MANIFEST_SHA == "7f3a81b248473d0b0591377c1bcd9f7df353066c9716836dabe90d9861f56297"
    assert C.BASE_SHA == "f3d183a35b3123e0b9c2a301fbc68ff5a276d345"
    assert len(C.PAIR_MANIFEST_SHA) == 64
    assert len(C.BASE_SHA) == 40


def test_subset_doc_verification():
    manifest = fixture_manifest()
    clip = make_clip_doc()
    pe = make_pe_doc()
    pp = F.build_per_pair(manifest, clip, pe, 512)
    subsets = F.build_subsets(pp, 512)
    doc = {
        "schema_version": C.SCHEMA_VERSION,
        "base_sha": C.BASE_SHA,
        "pair_manifest_sha": C.PAIR_MANIFEST_SHA,
        "n_pairs": 512,
        "subsets": subsets,
        "per_pair": pp["per_pair"],
    }
    assert C.verify_subset_doc(doc) == []
    bad = dict(doc)
    bad["subsets"] = dict(subsets)
    bad["subsets"]["ALL"] = list(range(255)) + [255, 255]
    assert any("duplicate" in p for p in C.verify_subset_doc(bad))


def test_v50_and_v25_nested_and_cell_coverage():
    scores = fixture_scores()
    strata = fixture_strata()
    v50 = set(C.geometry_stratified_select(scores, strata, 256))
    v25 = set(C.geometry_stratified_select(scores, strata, 128))
    assert v25 < v50
    present = {strata[i] for i in v50}
    assert present == set(strata)
