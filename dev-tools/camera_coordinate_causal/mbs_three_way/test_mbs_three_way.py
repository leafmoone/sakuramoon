"""Focused tests for the MBS three-way same-source audit tooling.

Scope (no torch, no GPU - pure stdlib + numpy, the numeric contract only):
  * per-pair margin estimator bit-fidelity to the FROZEN VBS expression
    ((sum(armA) - sum(armB)) / 4.0, arm sums taken separately)
  * SAME-corrected D / G / I construction (frozen adjusted_delta/paired_gap)
  * bootstrap determinism, exact mechanics, shared index matrix property
  * classification boundaries (spec s12, inclusive-zero edges)
  * replay-gate bound derivation from the committed frozen numerics floor
  * cohort flags + frozen pairs CSV parse with the pre-registered sizes
  * frozen t strata / noise-seed / arm constants
  * 17-significant-digit CSV round-trip (exact float64)
  * tree hash + checkpoint identity gate (temp fixtures)

Run:  python -m pytest test_mbs_three_way.py -q
The frozen reports (pairs CSV, audit JSON) are located via C.FROZEN_PAIRS_CSV
/ C.FROZEN_AUDIT_JSON by default; set MBS3_TEST_FROZEN_DIR to a directory
holding both files to run the tests off-host (e.g. on the dev laptop).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import analyze_three_way as A3
import contracts as C

_frozen_dir_env = os.environ.get("MBS3_TEST_FROZEN_DIR", "").strip()
FROZEN_DIR = Path(_frozen_dir_env) if _frozen_dir_env else None


def _frozen_csv() -> Path:
    if FROZEN_DIR is not None:
        return FROZEN_DIR / "camera-vertical-bottom-supervision-pairs.csv"
    return C.FROZEN_PAIRS_CSV


def _frozen_audit_json() -> Path:
    if FROZEN_DIR is not None:
        return FROZEN_DIR / "camera-vertical-bottom-supervision-audit.json"
    return C.FROZEN_AUDIT_JSON


# ---------------------------------------------------------------------------
# module import guards
# ---------------------------------------------------------------------------

def test_no_torch_imported_at_module_load():
    """contracts / score / analyze must be importable without torch (the
    scoring worker imports torch only inside main())."""
    import importlib

    for name in ("contracts", "score_three_way", "analyze_three_way"):
        mod = importlib.import_module(name)
        assert mod is not None


# ---------------------------------------------------------------------------
# 1. per-pair margin estimator: bit-fidelity to the frozen expression
# ---------------------------------------------------------------------------

def test_margin_expression_fidelity():
    """per_pair_margins must equal the FROZEN expression
    (sum(armA) - sum(armB)) / 4.0  with the two arm sums taken SEPARATELY
    (not the mean of per-stratum differences - different float64 rounding)."""
    from analyze_three_way import per_pair_margins

    # values mixing scales so evaluation order can matter in the last ulp
    base = {
        "TT": [0.001234567890123, 0.000987654321098, 0.002100000000001, 0.001500000000002],
        "TB": [0.001334567890123, 0.001087654321098, 0.002110000000001, 0.001510000000002],
        "BT": [0.002200000000003, 0.001900000000001, 0.002500000000009, 0.002100000000007],
        "BB": [0.002100000000003, 0.001800000000001, 0.002400000000009, 0.002000000000007],
        "T_SAME": [0.001234567890123, 0.000987654321098, 0.002100000000001, 0.001500000000002],
        "B_SAME": [0.002100000000003, 0.001800000000001, 0.002400000000009, 0.002000000000007],
        "T_ID": [0.0011] * 4,
        "B_ID": [0.0022] * 4,
    }
    rows = {(ck, 0): dict(base) for ck in C.CKS}
    got = per_pair_margins(rows, 0)[C.CKS[0]]

    L = base
    assert got["m_top"] == (sum(L["TB"]) - sum(L["TT"])) / 4.0
    assert got["m_bot"] == (sum(L["BT"]) - sum(L["BB"])) / 4.0
    assert got["f_top"] == (sum(L["T_SAME"]) - sum(L["TT"])) / 4.0
    assert got["f_bot"] == (sum(L["B_SAME"]) - sum(L["BB"])) / 4.0
    assert got["bb_minus_tt"] == (sum(L["BB"]) - sum(L["TT"])) / 4.0


def test_margin_frozen_contract_helper_matches():
    """The frozen per-loss margin/floor helpers agree with the 4-stratum
    aggregation used by the analyzer for exact binary values."""
    losses = {a: [0.25, 0.5, 0.75, 1.0] for a in C.ARMS}
    for k in range(4):
        losses["TB"][k] += 0.25
        losses["BT"][k] += 0.5
    m = C.margin_from_losses({a: sum(losses[a]) / 4.0 for a in C.ARMS})
    f = C.floor_from_losses({a: sum(losses[a]) / 4.0 for a in C.ARMS})
    assert m["m_top"] == 0.25
    assert m["m_bot"] == 0.5
    assert f["f_top"] == 0.0
    assert f["f_bot"] == 0.0


# ---------------------------------------------------------------------------
# 2. SAME-corrected D / G / I construction
# ---------------------------------------------------------------------------

def test_adjusted_delta_and_gap_semantics():
    # adjusted_delta(m_by_ckpt, f_by_ckpt, a, b): each ck entry holds
    # m_top/m_bot AND f_top/f_bot (the frozen analyze calls it as (m, m, a, b)).
    # All values binary-exact (multiples of 1/128) so == is meaningful.
    m_by = {
        "a": {"m_top": 0.25, "m_bot": 0.5, "f_top": 1 / 128, "f_bot": 1 / 64},
        "b": {"m_top": 0.25 + 1 / 32, "m_bot": 0.5 + 1 / 16, "f_top": 2 / 128, "f_bot": 3 / 64},
    }
    d = C.adjusted_delta(m_by, m_by, "a", "b")
    assert d["d_top"] == (1 / 32) - (1 / 128)
    assert d["d_bot"] == (1 / 16) - (1 / 32)
    assert C.paired_gap(d) == d["d_top"] - d["d_bot"] == (1 / 32 - 1 / 128) - (1 / 16 - 1 / 32)


def test_three_way_contrast_formulas():
    """I_TOP/I_BOTTOM/I_G = TREATMENT minus CONTROL contrasts (spec s9)."""
    from analyze_three_way import per_pair_margins

    def mk(mtop, mbot, ftop, fbot):
        # one stratum value repeated 4x keeps the 4-stratum mean exact
        return {
            "TT": [0.25] * 4, "TB": [0.25 + mtop] * 4,
            "BT": [0.50 + mbot] * 4, "BB": [0.50] * 4,
            "T_SAME": [0.25 + ftop] * 4, "B_SAME": [0.50 + fbot] * 4,
            "T_ID": [0.3] * 4, "B_ID": [0.4] * 4,
        }

    # binary-exact increments so every derived value is exactly representable
    rows = {}
    for ck, vals in zip(
        C.CKS,
        [
            (0.0, 0.0, 0.0, 0.0),                                   # PRE
            (1 / 128, 1 / 64, 1 / 1024, 1 / 512),                    # CONTROL
            (1 / 32, 1 / 16, 1 / 256, 1 / 128),                      # TREATMENT
        ],
        strict=True,
    ):
        rows[(ck, 0)] = mk(*vals)
    per = per_pair_margins(rows, 0)
    pre, con, tre = per["PRE"], per["CONTROL"], per["TREATMENT"]

    d_top_c = (con["m_top"] - pre["m_top"]) - (con["f_top"] - pre["f_top"])
    d_bot_c = (con["m_bot"] - pre["m_bot"]) - (con["f_bot"] - pre["f_bot"])
    d_top_t = (tre["m_top"] - pre["m_top"]) - (tre["f_top"] - pre["f_top"])
    d_bot_t = (tre["m_bot"] - pre["m_bot"]) - (tre["f_bot"] - pre["f_bot"])
    assert d_top_c == 1 / 128 - 1 / 1024
    assert d_bot_c == 1 / 64 - 1 / 512
    assert d_top_t == 1 / 32 - 1 / 256
    assert d_bot_t == 1 / 16 - 1 / 128
    g_c = d_top_c - d_bot_c
    g_t = d_top_t - d_bot_t
    # I = TREATMENT - CONTROL (exact)
    assert d_top_t - d_top_c == (1 / 32 - 1 / 256) - (1 / 128 - 1 / 1024)
    assert d_bot_t - d_bot_c == (1 / 16 - 1 / 128) - (1 / 64 - 1 / 512)
    assert g_t - g_c == ((1 / 32 - 1 / 256) - (1 / 16 - 1 / 128)) - ((1 / 128 - 1 / 1024) - (1 / 64 - 1 / 512))


# ---------------------------------------------------------------------------
# 3. bootstrap: exact mechanics, determinism, shared index matrix
# ---------------------------------------------------------------------------

def test_bootstrap_exact_mechanics():
    """bootstrap_ci == percentile(2.5/97.5) of the bootstrap means of
    rng.integers(0, n, (n_boot, n)) resamples, seeded with BOOT_SEED."""
    vals = [0.1, -0.25, 0.375, -0.5, 0.75, 0.125]
    got = C.bootstrap_ci(vals)
    a = np.asarray(vals, dtype=np.float64)
    n = a.size
    rng = np.random.default_rng(C.BOOT_SEED)
    idx = rng.integers(0, n, size=(C.N_BOOT, n))
    means = a[idx].mean(axis=1)
    assert got[0] == float(np.percentile(means, 2.5))
    assert got[1] == float(np.percentile(means, 97.5))


def test_bootstrap_determinism_and_seed():
    vals = [1.5, -2.25, 0.125, 3.0, -0.375, 0.75, -1.125, 2.5]
    assert C.bootstrap_ci(vals) == C.bootstrap_ci(vals)
    assert C.bootstrap_ci(vals, seed=C.BOOT_SEED + 1) != C.bootstrap_ci(vals)
    assert C.BOOT_SEED == 20260907
    assert C.N_BOOT == 10_000


def test_bootstrap_shared_index_matrix_property():
    """Two statistics of equal length are resampled with the SAME index
    matrix: doubling every value doubles both CI endpoints EXACTLY
    (x2 is exact in binary; same summation order)."""
    rng = np.random.default_rng(7)
    x = list(rng.uniform(-1.0, 1.0, size=40))
    x2 = [2.0 * v for v in x]
    c1 = C.bootstrap_ci(x)
    c2 = C.bootstrap_ci(x2)
    assert c2[0] == 2.0 * c1[0]
    assert c2[1] == 2.0 * c1[1]


def test_point_estimate_is_exact_mean():
    vals = [0.25, 0.5, 0.75, 1.0]
    from analyze_three_way import mean_exact

    assert mean_exact(vals) == float(sum(vals) / len(vals)) == 0.625


# ---------------------------------------------------------------------------
# 4. classification boundaries (spec s12)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "ci_ib, ci_ig, expected",
    [
        ([1e-9, 2e-9], [-2e-9, -1e-9], "CLEAR_DIRECTIONAL_CORRECTION"),
        # lower bound exactly 0 -> NOT clear (strict inequality)
        ([0.0, 2e-9], [-2e-9, -1e-9], "BORDERLINE"),
        # I_G upper exactly 0 -> NOT clear
        ([1e-9, 2e-9], [-2e-9, 0.0], "BORDERLINE"),
        # I_BOTTOM entirely below 0 -> harm
        ([-2e-9, -1e-9], [-1e-9, 1e-9], "CLEAR_HARM"),
        # I_G entirely above 0 -> harm
        ([-1e-9, 1e-9], [1e-9, 2e-9], "CLEAR_HARM"),
        # neither
        ([-1e-9, 1e-9], [-1e-9, 1e-9], "BORDERLINE"),
    ],
)
def test_classification_boundaries(ci_ib, ci_ig, expected):
    assert C.classify_contrast(ci_ib, ci_ig) == expected


def test_canonical_control_recommendation_mapping():
    assert C.canonical_control_recommendation("CLEAR_DIRECTIONAL_CORRECTION")["action"] == "NOT_REQUIRED_NOW"
    assert C.canonical_control_recommendation("CLEAR_HARM")["action"] == "NOT_REQUIRED_NOW"
    assert (
        C.canonical_control_recommendation("BORDERLINE")["action"]
        == "RECOMMENDED_FOR_LATER_USER_REVIEW"
    )
    with pytest.raises(ValueError):
        C.canonical_control_recommendation("NOPE")


# ---------------------------------------------------------------------------
# 5. replay gate: bound derived from committed evidence, PASS/FAIL logic
# ---------------------------------------------------------------------------

def test_replay_bound_derivation_from_floor():
    floor_doc = {
        "f_top": {"mean": 1.3151e-07, "ci95": [-3.10e-07, 6.06e-07]},
        "f_bot": {"mean": 1.4765e-07, "ci95": [-3.79e-07, 6.74e-07]},
    }
    b = C.replay_bounds_from_floor(floor_doc)
    assert b["multiplier"] == 5.0
    assert b["bound"] == 5.0 * 6.74e-07
    assert b["bound"] == pytest.approx(3.37e-06, rel=1e-3)


def test_replay_bound_tied_to_committed_frozen_audit_json():
    """The pre-registered bound must be reproducible from the COMMITTED
    frozen audit.json numerics floor (not an invented constant)."""
    path = _frozen_audit_json()
    if not path.is_file():
        pytest.skip(f"frozen audit.json not available at {path}")
    faudit = json.loads(path.read_text(encoding="utf-8"))
    floor_post = faudit["numerics_floor"]["POST"]
    b = C.replay_bounds_from_floor(floor_post)
    # contracts.py documents the bound as 5.0 * max(ci95 upper) ~= 3.37e-06
    assert b["bound"] == pytest.approx(3.37e-06, rel=1e-3)
    # and the frozen aggregate points exist for the point-delta checks
    faudit["top_causal"]["M"]["POST"]["point"]
    faudit["bottom_causal"]["M"]["POST"]["point"]


def test_replay_gate_pass_fail():
    b = C.replay_bounds_from_floor(
        {
            "f_top": {"mean": 1e-07, "ci95": [0.0, 6.06e-07]},
            "f_bot": {"mean": 1e-07, "ci95": [0.0, 6.74e-07]},
        }
    )
    bound = b["bound"]
    ok_pairs = [
        {"pair_index": i, "delta_top": 1e-7 * (i % 7 - 3), "delta_bot": -1e-7 * (i % 5 - 2)}
        for i in range(16)
    ]
    r = C.replay_gate_stats(ok_pairs, b)
    assert r["status"] == "PASS"
    assert r["maxabs"] <= bound and r["rms"] <= bound
    assert r["n_pairs"] == 16

    bad = ok_pairs + [{"pair_index": 999, "delta_top": bound * 1.1, "delta_bot": 0.0}]
    r2 = C.replay_gate_stats(bad, b)
    assert r2["status"] == "FAIL"
    assert not all(c["pass"] for c in r2["checks"])

    # aggregate point deltas are gated by the same bound
    r3 = C.replay_gate_stats(
        ok_pairs, b,
        new_points={"M_TOP": 0.00211501, "M_BOTTOM": 0.00354669},
        frozen_points={"M_TOP": 0.00211501, "M_BOTTOM": 0.00354669 + bound * 2},
    )
    assert r3["status"] == "FAIL"
    with pytest.raises(ValueError):
        C.replay_gate_stats([], b)


# ---------------------------------------------------------------------------
# 6. cohort flags + frozen CSV parse
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "pixel_shift, band, targeted",
    [
        (15.99, "lt2", False),
        (31.99, "lt2", False),
        (32.0, "2to4", True),      # exactly 2.0 -> 2to4 (inclusive lower)
        (63.99, "2to4", True),
        (64.0, "ge4", True),       # exactly 4.0 -> ge4
        (128.0, "ge4", True),
        (0.0, "lt2", False),
    ],
)
def test_cohort_flags(pixel_shift, band, targeted):
    fl = C.pair_cohort_flags({"signed_shift_start": -pixel_shift, "in_balanced_subset": False})
    assert fl["latent_band"] == band
    assert fl["in_targeted"] is targeted
    assert fl["latent_shift"] == pixel_shift / 16.0


def test_frozen_pairs_csv_parse_and_cohort_sizes():
    path = _frozen_csv()
    if not path.is_file():
        pytest.skip(f"frozen pairs CSV not available at {path}")
    rows = C.parse_frozen_pairs_csv(path)
    assert len(rows) == C.EXPECTED_COHORT_SIZES[C.COHORT_ALL] == 512
    assert [r["pair_id"] for r in rows] == list(range(512))

    # pre-registered cohort sizes (spec s10) - any miss is a STOP condition
    sizes = dict.fromkeys(C.COHORTS, 0)
    for r in rows:
        fl = C.pair_cohort_flags(r)
        sizes[C.COHORT_ALL] += 1
        if fl["in_targeted"]:
            sizes[C.COHORT_TARGETED] += 1
        sizes[{"lt2": C.COHORT_LT2, "2to4": C.COHORT_2TO4, "ge4": C.COHORT_GE4}[fl["latent_band"]]] += 1
        if fl["in_balanced"]:
            sizes[C.COHORT_BALANCED] += 1
    assert sizes == C.EXPECTED_COHORT_SIZES

    # typed values + a known row (pair 0 of the committed CSV)
    r0 = rows[0]
    assert isinstance(r0["k_start"], int)
    assert isinstance(r0["in_balanced_subset"], bool)
    assert r0["signed_shift_start"] == pytest.approx(-41.0)
    assert r0["M_TOP_POST_frozen"] == pytest.approx(0.000403687357903, rel=1e-12)


# ---------------------------------------------------------------------------
# 7. frozen constants: t strata, noise seed, arms
# ---------------------------------------------------------------------------

def test_frozen_t_strata_exact():
    assert C.t_values() == (
        0.1388061520926247,
        0.24819609741338816,
        0.37948289980451305,
        0.5560734456280833,
    )


def test_noise_seed_contract():
    master = C.MASTER_SEED_VBS
    assert master == 20260907
    assert C.noise_seed(0, 0) == master * 1_000_003
    for i, k in ((0, 1), (1, 0), (511, 3), (256, 2)):
        assert C.noise_seed(i, k) == master * 1_000_003 + i * 100 + k


def test_arms_contract():
    assert C.ARMS == ("TT", "TB", "BT", "BB", "T_SAME", "B_SAME", "T_ID", "B_ID")
    assert C.ARMS_NO_ID == ("TT", "TB", "BT", "BB", "T_SAME", "B_SAME")
    for arm in C.ARMS:
        assert C.arm_side(arm) in ("top", "bottom")
        assert C.arm_coord(arm) in ("top_correct", "bottom_correct", "id")
    # ID arms use identity coordinates; SAME arms duplicate their side
    assert C.arm_coord("T_ID") == "id"
    assert C.arm_coord("B_ID") == "id"
    assert C.arm_coord("T_SAME") == "top_correct"
    assert C.arm_coord("B_SAME") == "bottom_correct"
    assert C.arm_side("T_SAME") == "top"
    assert C.arm_side("B_SAME") == "bottom"


# ---------------------------------------------------------------------------
# 8. CSV round-trip precision
# ---------------------------------------------------------------------------

def test_17g_roundtrip_exact():
    from analyze_three_way import FMT

    vals = [
        0.00211501, -0.00354669, 1.3151e-07, -3.79e-07, 0.0017253,
        -41.0, 2.5625, 1.0, -1.0, 0.1 + 0.2, 1e-300, -1e-300,
        3.141592653589793, 2.718281828459045,
    ]
    for v in vals:
        assert float(FMT.format(v)) == v


# ---------------------------------------------------------------------------
# 9. tree hash + checkpoint identity gate (temp fixtures)
# ---------------------------------------------------------------------------

def test_tree_sha256_matches_reference_pipeline(tmp_path):
    """Canonical audit tree hash == sha256 over sorted sha256sum lines.

    The reference re-implementation (below) independently pins the
    evidence-lineage algorithm used for the RERUN #2 digest d2673914...
    (`find <abs-root> -type f | sort | xargs sha256sum | sha256sum`):
    sorted ABSOLUTE path strings, text-mode two-space separator, one line
    per file (trailing newline per line), outer sha256 over the stream.
    """
    import hashlib as _h

    root = tmp_path / "tree"
    (root / "model").mkdir(parents=True)
    (root / "model" / "w2.bin").write_bytes(b"22")
    (root / "model" / "w1.bin").write_bytes(b"11")
    (root / "state.bin").write_bytes(b"s")

    def reference(base: Path) -> str:
        lines = []
        for name in sorted(str(p) for p in base.rglob("*") if p.is_file()):
            d = _h.sha256(Path(name).read_bytes()).hexdigest()
            lines.append(f"{d}  {name}")
        return _h.sha256(("\n".join(lines) + "\n").encode("utf-8")).hexdigest()

    assert C._tree_sha256(root) == reference(root)
    assert C._tree_sha256(root) == reference(root)  # deterministic
    before = C._tree_sha256(root)
    (root / "state.bin").write_bytes(b"S")
    assert C._tree_sha256(root) != before  # content sensitivity


def _make_fake_ckpt(root: Path, *, complete=b"complete\n", update=118100, alpha=1.0,
                    manifest=b'{"identity": {}}') -> Path:
    ck = root / "ckpt"
    (ck / "model").mkdir(parents=True)
    (ck / "model" / "weights.bin").write_bytes(b"w" * 64)
    (ck / "train_state").mkdir()
    (ck / "COMPLETE").write_bytes(complete)
    (ck / "manifest.json").write_bytes(manifest)
    (ck / "train_state" / "trainer_state.json").write_text(
        json.dumps({"successful_updates": update}), encoding="utf-8")
    (ck / "train_state" / "growth_state.json").write_text(
        json.dumps({"alpha": alpha}), encoding="utf-8")
    return ck


def test_checkpoint_identity_gate(tmp_path, monkeypatch):
    ck = _make_fake_ckpt(tmp_path)
    expected = {
        "manifest_sha256": C.sha256_file(ck / "manifest.json"),
        "model_tree_sha256": C._tree_sha256(ck / "model"),
    }
    monkeypatch.setitem(C.CKPT_PATHS, "PRE", ck)
    monkeypatch.setitem(C.CKPT_UPDATES, "PRE", 118100)
    monkeypatch.setitem(C.EXPECTED_IDENTITY, "PRE", expected)

    r = C.verify_checkpoint_identity("PRE")
    assert r["status"] == "PASS", r["problems"]
    assert r["evidence"]["successful_updates"] == 118100

    # corrupt COMPLETE marker
    (ck / "COMPLETE").write_bytes(b"not complete")
    r2 = C.verify_checkpoint_identity("PRE")
    assert r2["status"] == "FAIL"
    assert "COMPLETE marker invalid" in r2["problems"]

    # wrong update identity
    (ck / "COMPLETE").write_bytes(b"complete\n")
    (ck / "train_state" / "trainer_state.json").write_text(
        json.dumps({"successful_updates": 118200}), encoding="utf-8")
    r3 = C.verify_checkpoint_identity("PRE")
    assert r3["status"] == "FAIL"
    assert any("update" in p for p in r3["problems"])

    # wrong growth alpha
    (ck / "train_state" / "trainer_state.json").write_text(
        json.dumps({"successful_updates": 118100}), encoding="utf-8")
    (ck / "train_state" / "growth_state.json").write_text(
        json.dumps({"alpha": 0.5}), encoding="utf-8")
    r4 = C.verify_checkpoint_identity("PRE")
    assert r4["status"] == "FAIL"
    assert any("alpha" in p for p in r4["problems"])


def test_checkpoint_identity_missing_dir(tmp_path, monkeypatch):
    monkeypatch.setitem(C.CKPT_PATHS, "PRE", tmp_path / "absent")
    monkeypatch.setitem(C.EXPECTED_IDENTITY, "PRE", {
        "manifest_sha256": "0" * 64, "model_tree_sha256": "0" * 64})
    r = C.verify_checkpoint_identity("PRE")
    assert r["status"] == "FAIL"
    assert r["problems"]


# ---------------------------------------------------------------------------
# pair-cohort identity gate (analyze --pair-check; pre-scoring)
# ---------------------------------------------------------------------------

_FROZEN_CSV_HEADER = (
    "pair_id,source_shard,sample_id,original_side,zoom_band,k_start,k_end,"
    "available,full_height,signed_shift_start,"
    "M_TOP_PRE,M_TOP_MID,M_TOP_POST,M_BOT_PRE,M_BOT_MID,M_BOT_POST,"
    "G_PRE_MID,G_MID_POST,G_PRE_POST,content_asymmetry,in_balanced_subset"
)


def _build_pair_fixtures(tmp_path: Path, n: int = 2) -> None:
    """Write a consistent (frozen CSV, pair-manifest.json) fixture for n pairs."""
    lines = [_FROZEN_CSV_HEADER]
    pairs = []
    for i in range(n):
        side = "top" if i % 2 == 0 else "bottom"
        shift = -41.0 + i
        shard = "data/1_2025.9/shard-000001.tar"
        lines.append(
            f"{i},{shard},{1000 + i},{side},lt2,{i},{i + 4},32,2048,"
            f"{shift:.17g},1.0,1.0,1.0,2.0,2.0,2.0,0.1,0.1,0.1,0.0,true"
        )
        pairs.append({
            "pair_index": i,
            "source_shard": shard,
            "sample_id": 1000 + i,
            "original_side": side,
            "signed_shift_start": shift,
            "geometry": {
                "k_start": i,
                "k_end": i + 4,
                "available": 32,
                "full_height": 2048,
                "anchor_is_top": side == "top",
            },
        })
    (tmp_path / "frozen-pairs.csv").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    (tmp_path / "pair-manifest.json").write_text(
        json.dumps({"status": "OK", "pairs": pairs}), encoding="utf-8")


def _manifest_mut(tmp_path: Path, *, pair: dict | None = None,
                  geom: dict | None = None) -> None:
    pm = json.loads((tmp_path / "pair-manifest.json").read_text(encoding="utf-8"))
    if pair:
        pm["pairs"][0].update(pair)
    if geom:
        pm["pairs"][0]["geometry"].update(geom)
    (tmp_path / "pair-manifest.json").write_text(json.dumps(pm), encoding="utf-8")


def test_pair_cohort_gate(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "FROZEN_PAIRS_CSV", tmp_path / "frozen-pairs.csv")
    monkeypatch.setitem(C.EXPECTED_COHORT_SIZES, C.COHORT_ALL, 2)
    log_path = tmp_path / "gate.log"
    log_path.write_text("", encoding="utf-8")

    _build_pair_fixtures(tmp_path, n=2)
    out = A3.check_pair_cohort_identity(tmp_path, log_path)
    assert out is not None
    pairs, frozen_rows, n = out
    assert n == 2 and len(pairs) == 2 and len(frozen_rows) == 2

    # signed_shift mismatch -> FATAL (None)
    _build_pair_fixtures(tmp_path, n=2)
    _manifest_mut(tmp_path, pair={"signed_shift_start": -40.5})
    assert A3.check_pair_cohort_identity(tmp_path, log_path) is None

    # count mismatch (3 pairs vs expected 2) -> FATAL
    _build_pair_fixtures(tmp_path, n=3)
    assert A3.check_pair_cohort_identity(tmp_path, log_path) is None

    # manifest status != OK -> FATAL
    _build_pair_fixtures(tmp_path, n=2)
    pm = json.loads((tmp_path / "pair-manifest.json").read_text(encoding="utf-8"))
    pm["status"] = "STOP_BELOW_MINIMUM"
    (tmp_path / "pair-manifest.json").write_text(json.dumps(pm), encoding="utf-8")
    assert A3.check_pair_cohort_identity(tmp_path, log_path) is None

    # side mismatch -> FATAL
    _build_pair_fixtures(tmp_path, n=2)
    _manifest_mut(tmp_path, pair={"original_side": "bottom"})  # pair 0 is top
    assert A3.check_pair_cohort_identity(tmp_path, log_path) is None

    # geometry-level mismatch (k_start) -> FATAL
    _build_pair_fixtures(tmp_path, n=2)
    _manifest_mut(tmp_path, geom={"k_start": 99})
    assert A3.check_pair_cohort_identity(tmp_path, log_path) is None

    # missing manifest -> clean FATAL (no traceback)
    (tmp_path / "pair-manifest.json").unlink()
    assert A3.check_pair_cohort_identity(tmp_path, log_path) is None

    # missing frozen CSV -> clean FATAL
    _build_pair_fixtures(tmp_path, n=2)
    (tmp_path / "frozen-pairs.csv").unlink()
    assert A3.check_pair_cohort_identity(tmp_path, log_path) is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
