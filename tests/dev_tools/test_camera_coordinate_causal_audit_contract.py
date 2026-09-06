"""Regression contracts for the Camera Coordinate Causal audit tooling.

These tests lock the bug classes found while producing the final causal
audit numbers (see dev-tools/camera_coordinate_causal/README.md, "Known bug
history"). They exercise the pure contract helpers in
dev-tools/camera_coordinate_causal/review_contracts.py and pin the forensic
snapshot to the exact executed script hashes.

They do NOT import production src and do NOT recompute the audit numbers.
"""
from __future__ import annotations

import hashlib
import py_compile
import sys
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT_DIR = REPO_ROOT / "dev-tools" / "camera_coordinate_causal" / "final_snapshot"
sys.path.insert(0, str(REPO_ROOT / "dev-tools" / "camera_coordinate_causal"))

import review_contracts as rc

IN_CHANNELS = 128  # Mage-VAE latent channels of the audited model


# ---------------- forensic snapshot pinning ---------------------------------

EXPECTED_SNAPSHOT_SHA256 = {
    "cc_common.py": "63edc8ef5a62fcb1e64c202f9f12d562713512e44b618e49a719fe0527aa437b",
    "cc_stage1.py": "384d58e49b3c08b345525aded9485c1bb38c310d426d399aa07acf6182aa65b0",
    "cc_stage2.py": "e5b7ae910df0db7847fcb96d9ecfddf12ce394e383045750da695c62c7c664b9",
    "cc_stage2b.py": "aea987d6505338b81a7e9d72084e5bef6a458b04cf41454750f347c779da2177",
    "cc_stage3.py": "4a855b3134b78f4478003cbdd611945b9a4e1c8e53e95827ec0cc28ea27657a6",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_forensic_snapshot_pinned_to_executed_hashes():
    for name, want in EXPECTED_SNAPSHOT_SHA256.items():
        p = SNAPSHOT_DIR / name
        assert p.is_file(), f"missing final snapshot: {name}"
        assert _sha256(p) == want, f"final_snapshot/{name} drifted from the executed version"


def test_forensic_snapshot_syntax_compiles():
    for name in EXPECTED_SNAPSHOT_SHA256:
        p = SNAPSHOT_DIR / name
        with tempfile.TemporaryDirectory() as td:
            py_compile.compile(str(p), cfile=str(Path(td) / f"{name}c"), doraise=True)


# ---------------- 10A: checkpoint-key separation ----------------------------


def test_checkpoint_key_separation_three_checkpoints():
    """Same sample/timestep/arm under PRE/MID/POST must aggregate to 3 units."""
    ledger = {
        7: {
            "PRE": {"sample_id": "s1", "timestep": 0.248, "arm": "OPPOSITE", "loss": 1.0},
            "MID": {"sample_id": "s1", "timestep": 0.248, "arm": "OPPOSITE", "loss": 2.0},
            "POST": {"sample_id": "s1", "timestep": 0.248, "arm": "OPPOSITE", "loss": 3.0},
        }
    }
    keys = rc.aggregate_ledger_units(ledger)
    assert keys == [(7, "MID"), (7, "POST"), (7, "PRE")]
    assert len(keys) == 3
    vals = {ck: ledger[7][ck]["loss"] for ck in ("PRE", "MID", "POST")}
    assert vals == {"PRE": 1.0, "MID": 2.0, "POST": 3.0}
    assert len(set(vals.values())) == 3


def test_checkpoint_key_requires_tag():
    with pytest.raises(ValueError):
        rc.ledger_key(7, "")


# ---------------- 10B: 3-seed / variant cardinality --------------------------


def test_variant_cardinality_three_seeds_no_collapse():
    rows = ["case-X", "case-X-v1_derived", "case-X-v2_derived"]
    out = rc.variant_cardinality(rows)
    assert out["logical_base_count"] == 1
    assert out["observed_variant_rows"] == 3
    assert out["all_rows_preserved"] is True
    # hard assertion: expected variants == observed variants
    expected = {("case-X", None), ("case-X", 1), ("case-X", 2)}
    observed = {rc.split_base_variant(n) for n in rows}
    assert expected == observed


def test_base_id_filter_keeps_all_variant_rows():
    rows = ["case-X", "case-X-v1_derived", "case-X-v2_derived", "case-Y", "case-Y-v1_derived"]
    out = rc.variant_cardinality(rows)
    assert out["logical_base_count"] == 2
    assert out["observed_variant_rows"] == 5
    assert out["all_rows_preserved"] is True


# ---------------- 10C: image/noise identity uniqueness ----------------------


def test_identity_hashes_distinct_pass():
    rc.assert_distinct_identity_hashes(["aa01", "bb02", "cc03"])


def test_identity_hash_duplicate_fails_closed():
    with pytest.raises(ValueError, match="duplicate"):
        rc.assert_distinct_identity_hashes(["aa01", "bb02", "aa01"])


# ---------------- 10D: latent 3-D contract ----------------------------------


def test_latent_3d_valid_accepted():
    rc.validate_latent_tuple([(128, 16, 16)], IN_CHANNELS)
    rc.validate_latent_tuple([(128, 32, 20)], IN_CHANNELS)  # non-square OK


def test_latent_4d_batched_rejected():
    """The frozen-bundle shape (1, C, H, W) that crashed stage2/2b."""
    with pytest.raises(ValueError, match="3-D"):
        rc.validate_latent_tuple([(1, 128, 16, 16)], IN_CHANNELS)


def test_latent_transposed_rejected():
    with pytest.raises(ValueError, match="channel dim"):
        rc.validate_latent_tuple([(16, 16, 128)], IN_CHANNELS)


def test_squeeze_batch_matches_final_transform():
    assert rc.squeeze_batch((1, 128, 16, 16)) == (128, 16, 16)
    assert rc.squeeze_batch((128, 16, 16)) == (128, 16, 16)
    # the final script pipeline: frozen 4-D bundle -> squeeze(0) -> valid latent
    rc.validate_latent_tuple([rc.squeeze_batch((1, 128, 16, 16))], IN_CHANNELS)


# ---------------- 10E: bootstrap subspace ------------------------------------


def test_bootstrap_subspace_checkpoint_isolation():
    """2 checkpoints x 2 base prompts x 3 seeds = 6 units; sentinel rows.

    PRE rows are all 1.0, POST rows are sentinel 1000.0: any cross-checkpoint
    read would explode the mean and fail the assertion.
    """
    sel = ["p1-s0", "p1-s1", "p1-s2", "p2-s0", "p2-s1", "p2-s2"]
    pre = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    post = [1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0]
    resample = [[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0]]
    out = rc.bootstrap_checkpoint_means(
        {"PRE": pre, "POST": post}, sel, {"PRE": resample, "POST": resample}
    )
    assert out["PRE"] == [1.0, 1.0]
    assert out["POST"] == [1000.0, 1000.0]


def test_bootstrap_out_of_subspace_index_fails_closed():
    """The v3 bug: a global-space index (5) into a 3-row selected subspace."""
    with pytest.raises(ValueError, match="outside subspace"):
        rc.bootstrap_checkpoint_means(
            {"PRE": [1.0, 2.0, 3.0]}, ["u0", "u1", "u2"], {"PRE": [[0, 1, 5]]}
        )


def test_bootstrap_paired_structure_preserved():
    """Identical resample matrices -> identically paired replicates."""
    sel = ["a", "b", "c", "d"]
    resample = [[0, 1, 2, 3], [3, 2, 1, 0], [1, 1, 2, 2]]
    out = rc.bootstrap_checkpoint_means(
        {"PRE": [1.0, 2.0, 3.0, 4.0], "MID": [10.0, 20.0, 30.0, 40.0]},
        sel, {"PRE": resample, "MID": resample},
    )
    assert len(out["PRE"]) == len(out["MID"]) == 3
    for a, b in zip(out["PRE"], out["MID"]):
        assert b == 10.0 * a


def test_bootstrap_row_count_mismatch_fails_closed():
    with pytest.raises(ValueError, match="rows !="):
        rc.bootstrap_checkpoint_means(
            {"PRE": [1.0, 2.0]}, ["u0", "u1", "u2"], {"PRE": [[0, 1, 2]]}
        )


def test_stratum_rows_never_mixed():
    m = {
        "PRE": {0: [1.0, 2.0, 3.0], 1: [100.0, 200.0, 300.0]},
        "POST": {0: [1.1, 2.1, 3.1], 1: [100.1, 200.1, 300.1]},
    }
    assert rc.per_stratum_rows(m, "PRE", 0) == [1.0, 2.0, 3.0]
    assert rc.per_stratum_rows(m, "PRE", 1) == [100.0, 200.0, 300.0]
    assert rc.per_stratum_rows(m, "POST", 1) == [100.1, 200.1, 300.1]
    with pytest.raises(KeyError):
        rc.per_stratum_rows(m, "POST", 2)
    with pytest.raises(KeyError):
        rc.per_stratum_rows(m, "WRONG_CK", 0)


# ---------------- 10F: strata masks ------------------------------------------


def test_strata_mask_exact_rows():
    bands = [
        rc.band_of(zoom=1.10, latent_shift=1.0, orientation="horizontal", norm_offset=0.2),
        rc.band_of(zoom=1.30, latent_shift=3.0, orientation="vertical", norm_offset=0.5),
        rc.band_of(zoom=1.40, latent_shift=5.0, orientation="vertical", norm_offset=0.9),
    ]
    assert rc.stratum_mask(bands, "zoom", "mild") == [0]
    assert rc.stratum_mask(bands, "zoom", "medium") == [1]
    assert rc.stratum_mask(bands, "zoom", "strong") == [2]
    assert rc.stratum_mask(bands, "shift", "lt2") == [0]
    assert rc.stratum_mask(bands, "shift", "2to4") == [1]
    assert rc.stratum_mask(bands, "shift", "ge4") == [2]
    assert rc.stratum_mask(bands, "orientation", "horizontal") == [0]
    assert rc.stratum_mask(bands, "orientation", "vertical") == [1, 2]
    assert rc.stratum_mask(bands, "offset", "low") == [0]
    assert rc.stratum_mask(bands, "offset", "high") == [1, 2]
    assert rc.stratum_mask(bands, "edge", "L") == [0]
    assert rc.stratum_mask(bands, "edge", "C") == [1]
    assert rc.stratum_mask(bands, "edge", "R") == [2]


def test_strata_mask_wrong_key_fails_closed():
    bands = [rc.band_of(zoom=1.1, latent_shift=1.0, orientation="horizontal", norm_offset=0.2)]
    with pytest.raises(KeyError):
        rc.stratum_mask(bands, "nonexistent_stratum", "mild")


# ---------------- 10G: causal-margin sign ------------------------------------


def test_causal_margin_sign_convention():
    assert rc.causal_margin(wrong_loss=2.0, correct_loss=1.0) == 1.0
    assert rc.causal_margin(wrong_loss=1.0, correct_loss=2.0) == -1.0
