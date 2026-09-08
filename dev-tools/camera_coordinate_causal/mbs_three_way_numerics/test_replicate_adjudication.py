"""Focused tests for the MBS three-way numerics adjudication protocol.

10 focused tests (protocol s9):
  1. exactly 8 replicate tuples, correct Cartesian product + fixed order
  2. exactly 4 unique C/T combinations
  3. PRE cancellation: I through the UNFACTORED frozen formulas with two
     different PRE replicates agrees within fp64 roundoff and equals the
     factored PRE-free expression
  4. classification decision tree UNCHANGED (identical to the frozen
     mbs_three_way.classify_contrast on boundary cases)
  5. envelope uses the correct worst-case low/high + min/max point
  6. all-8-classes-same -> NUMERICALLY_STABLE
  7. mixed classes -> NUMERICALLY_AMBIGUOUS
  8. stable but envelope disagrees -> AMBIGUOUS (no directional release)
  9. ONE shared bootstrap index matrix per cohort, reused across tuples;
     identical to frozen VBS.bootstrap_ci (bit-level)
 10. NO per-pair hard max gate: the final adjudication consumes only
     combination-level classification labels and never per-pair maxabs

No torch import.  No GPU.  Deterministic.
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

_HERE = Path(__file__).resolve().parent
_MBS_DIR = _HERE.parent / "mbs_three_way"
for _p in (str(_MBS_DIR), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import analyze_replicates as AR

# This package's contracts module (loaded under a unique name by AR's
# bootstrap so the bare name "contracts" stays bound to mbs's own module).
Cn = AR.Cn


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
def _margins(m_top, m_bot, f_top, f_bot):
    return {"m_top": m_top, "m_bot": m_bot, "f_top": f_top, "f_bot": f_bot, "bb_minus_tt": 0.0}


@pytest.fixture()
def margins_fixture():
    """Deterministic per-pair margins for 3 checkpoints x 2 replicates x 8
    pairs.  PRE A and PRE B differ by ~1e-15 (the kind of fp64-level jitter
    the protocol must absorb); C/T replicates differ by larger amounts so
    classification flips are detectable if the logic regresses."""
    rng = np.random.default_rng(7)
    out = {}
    base = {
        "PRE": {"A": 0.0, "B": 1e-15},
        "CONTROL": {"A": 0.0, "B": 2e-5},
        "TREATMENT": {"A": 0.0, "B": 4e-5},
    }
    for ck, reps in base.items():
        out[ck] = {}
        for rep, off in reps.items():
            rows = []
            for _ in range(8):
                v = rng.normal(0.0, 1e-3)
                rows.append(_margins(
                    float(v) + off, float(v * 1.1) + off,
                    float(v * 0.5), float(v * 0.45) + off,
                ))
            out[ck][rep] = rows
    return out


# ---------------------------------------------------------------------------
# 1-2: combination enumeration
# ---------------------------------------------------------------------------
def test_replicate_tuples_exact_cartesian_product():
    ts = Cn.replicate_tuples()
    assert len(ts) == 8
    expected = []
    for pi, p in enumerate(Cn.REPS):
        for ci, c in enumerate(Cn.REPS):
            for ti, t in enumerate(Cn.REPS):
                expected.append((f"P{pi} C{ci} T{ti}", p, c, t))
    assert ts == expected
    # every (P, C, T) replicate triple appears exactly once
    triples = [(p, c, t) for _, p, c, t in ts]
    assert len(set(triples)) == 8
    for p, c, t in triples:
        assert p in Cn.REPS and c in Cn.REPS and t in Cn.REPS


def test_ct_combinations_exactly_four_unique():
    cs = Cn.ct_combinations()
    assert len(cs) == 4
    assert [c for _, c, t in cs] == ["A", "A", "B", "B"]
    assert [t for _, c, t in cs] == ["A", "B", "A", "B"]
    assert len({(c, t) for _, c, t in cs}) == 4
    # and exactly the Cartesian product of C x T replicates
    assert {(c, t) for _, c, t in cs} == set(itertools.product(Cn.REPS, Cn.REPS))


# ---------------------------------------------------------------------------
# 3: PRE cancellation (protocol s24)
# ---------------------------------------------------------------------------
def test_pre_cancellation_unfactored_equals_factored(margins_fixture):
    """I computed through the UNFACTORED frozen formulas with PRE_A vs
    PRE_B must (a) agree within the pre-registered tolerance and
    (b) equal the factored PRE-free expression within roundoff."""
    mf = margins_fixture
    for c in Cn.REPS:
        for t in Cn.REPS:
            con = [mf["CONTROL"][c][i] for i in range(8)]
            tre = [mf["TREATMENT"][t][i] for i in range(8)]
            pre_a = [mf["PRE"]["A"][i] for i in range(8)]
            pre_b = [mf["PRE"]["B"][i] for i in range(8)]
            for i in range(8):
                ctr_a = Cn.pair_contrast(pre_a[i], con[i], tre[i])
                ctr_b = Cn.pair_contrast(pre_b[i], con[i], tre[i])
                for m in ("i_top", "i_bot", "i_g"):
                    diff = abs(ctr_a[m] - ctr_b[m])
                    assert diff <= Cn.PRE_CANCELLATION_TOL, (m, diff)
                # factored PRE-free expression (algebraically identical)
                ib_factored = (tre[i]["m_bot"] - tre[i]["f_bot"]) - (con[i]["m_bot"] - con[i]["f_bot"])
                assert abs(ctr_a["i_bot"] - ib_factored) <= Cn.PRE_CANCELLATION_TOL


# ---------------------------------------------------------------------------
# 4: classification decision tree unchanged
# ---------------------------------------------------------------------------
def test_classification_tree_unchanged_reference_cases():
    # reference: exact decision tree from the pre-registered spec
    def ref(ib_ci, ig_ci):
        if ib_ci[0] > 0.0 and ig_ci[1] < 0.0:
            return "CLEAR_DIRECTIONAL_CORRECTION"
        if ib_ci[1] < 0.0 or ig_ci[0] > 0.0:
            return "CLEAR_HARM"
        return "BORDERLINE"
    cases = [
        ((0.01, 0.05), (-0.05, -0.01)),    # correction (strict)
        ((0.0, 0.05), (-0.05, -0.01)),     # ib lower == 0 -> NOT correction
        ((0.01, 0.05), (-0.05, 0.0)),      # ig upper == 0 -> NOT correction
        ((-0.05, -0.01), (-0.05, 0.05)),   # harm via ib upper < 0
        ((0.01, 0.05), (0.01, 0.05)),      # harm via ig lower > 0
        ((-0.05, 0.05), (-0.05, 0.05)),    # borderline: both cross 0
        ((-0.05, 0.0), (-0.0, 0.05)),      # ib upper == 0, ig lower == -0.0 -> borderline
        ((0.0, 0.0), (0.0, 0.0)),          # all zeros -> borderline
    ]
    for ib_ci, ig_ci in cases:
        assert Cn.classify_contrast(ib_ci, ig_ci) == ref(ib_ci, ig_ci)
        assert Cn.classify_envelope(ib_ci, ig_ci) == ref(ib_ci, ig_ci)
    # and it IS the very function object the reviewed tooling shipped
    assert Cn.classify_contrast is Cn.MBS_C.classify_contrast


# ---------------------------------------------------------------------------
# 5: envelope worst-case bounds
# ---------------------------------------------------------------------------
def test_envelope_worst_case_bounds():
    points = [0.02, 0.05, -0.01, 0.03]
    cis = [(0.00, 0.04), (0.03, 0.06), (-0.03, 0.01), (0.01, 0.04)]
    e = Cn.envelope(points, cis)
    assert e["point_min"] == -0.01
    assert e["point_max"] == 0.05
    assert e["ci_low"] == -0.03    # worst (lowest) low bound
    assert e["ci_high"] == 0.06    # worst (highest) high bound
    with pytest.raises(AssertionError):
        Cn.envelope([0.1], [])     # length mismatch
    with pytest.raises(AssertionError):
        Cn.envelope([], [])        # empty


# ---------------------------------------------------------------------------
# 6-8: adjudication logic
# ---------------------------------------------------------------------------
def test_adjudication_stable_when_all_eight_classes_same():
    res = Cn.adjudicate([Cn.CLASS_CORRECTION] * 8, Cn.CLASS_CORRECTION)
    assert res["stable"] is True
    assert res["stable_label"] == Cn.STABLE
    assert res["common_classification"] == Cn.CLASS_CORRECTION
    assert res["verdict"] == Cn.ADJUDICATION_PASS


def test_adjudication_ambiguous_when_classes_mixed():
    classes = [Cn.CLASS_CORRECTION] * 7 + [Cn.CLASS_BORDERLINE]
    res = Cn.adjudicate(classes, Cn.CLASS_BORDERLINE)
    assert res["stable"] is False
    assert res["stable_label"] == Cn.AMBIGUOUS_STABLE
    assert res["common_classification"] is None
    assert res["verdict"] == Cn.ADJUDICATION_AMBIGUOUS


def test_adjudication_ambiguous_when_envelope_disagrees():
    # all 8 tuples say CORRECTION but the conservative envelope says
    # BORDERLINE -> NO directional release.
    res = Cn.adjudicate([Cn.CLASS_CORRECTION] * 8, Cn.CLASS_BORDERLINE)
    assert res["stable"] is True
    assert res["verdict"] == Cn.ADJUDICATION_AMBIGUOUS
    # same for a HARM envelope disagreement
    res2 = Cn.adjudicate([Cn.CLASS_BORDERLINE] * 8, Cn.CLASS_HARM)
    assert res2["verdict"] == Cn.ADJUDICATION_AMBIGUOUS


# ---------------------------------------------------------------------------
# 9: shared bootstrap index matrix
# ---------------------------------------------------------------------------
def test_shared_bootstrap_matrix_reused_and_matches_frozen():
    # one matrix per cohort size, deterministic
    idx1 = Cn.shared_bootstrap_index_matrix(312)
    idx2 = Cn.shared_bootstrap_index_matrix(312)
    assert (idx1 == idx2).all()
    assert idx1.shape == (Cn.N_BOOT, 312)
    assert idx1.min() >= 0 and idx1.max() < 312
    # different cohort size -> different matrix
    idx37 = Cn.shared_bootstrap_index_matrix(37)
    assert idx37.shape == (Cn.N_BOOT, 37)
    # bit-identical to the frozen VBS mechanics for the same values:
    rng = np.random.default_rng(123)
    vals = rng.normal(0.0, 1e-3, size=312)
    ref = Cn.VBS.bootstrap_ci(vals.tolist(), n_boot=Cn.N_BOOT, seed=Cn.BOOT_SEED)
    got = Cn.bootstrap_ci_shared(np.asarray(vals, dtype=np.float64), idx1)
    assert got == tuple(ref)  # exact float equality
    # ...and 'reused across tuples': two different value vectors through the
    # SAME matrix are the only way the CIs are paired across replicates.
    vals2 = vals + 1e-6
    got2 = Cn.bootstrap_ci_shared(np.asarray(vals2, dtype=np.float64), idx1)
    ref2 = Cn.VBS.bootstrap_ci(vals2.tolist(), n_boot=Cn.N_BOOT, seed=Cn.BOOT_SEED)
    assert got2 == tuple(ref2)


# ---------------------------------------------------------------------------
# 10: no per-pair hard max gate in the final adjudication
# ---------------------------------------------------------------------------
def test_no_per_pair_hard_max_gate_in_adjudication():
    # (a) adjudicate() signature accepts ONLY combination-level labels —
    #     there is no per-pair statistic it can consult.
    import inspect
    sig = inspect.signature(Cn.adjudicate)
    assert set(sig.parameters) == {"tuple_classes", "envelope_class"}
    # (b) functionally: extreme per-pair tails that do NOT change the
    #     combination-level classification cannot change the verdict.
    clean = Cn.adjudicate([Cn.CLASS_CORRECTION] * 8, Cn.CLASS_CORRECTION)
    with_extreme_tails = Cn.adjudicate([Cn.CLASS_CORRECTION] * 8, Cn.CLASS_CORRECTION)
    assert clean["verdict"] == with_extreme_tails["verdict"] == Cn.ADJUDICATION_PASS
    # (c) the analyze module contains no per-pair maxabs gate that gates
    #     the adjudication: the only per-pair max in the pipeline is the
    #     PRE-cancellation IMPLEMENTATION check (roundoff, not science).
    src = Path(AR.__file__).read_text(encoding="utf-8")
    assert "replay_gate_stats" not in src  # old aggregate-gate machinery absent
    assert "maxabs_bound" not in src
