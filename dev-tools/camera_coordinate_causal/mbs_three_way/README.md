# MBS Three-Way Same-Source Causal Audit — new tooling

Forward-only, evaluation-only tooling for the MBS three-way same-source
causal audit on the SakuraMoon Camera model.  Companion to the FROZEN
Vertical Bottom Supervision (VBS) audit package, which is never edited:

    dev-tools/camera_coordinate_causal/vertical_bottom_supervision/   (FROZEN)
    dev-tools/camera_coordinate_causal/final_snapshot/                (FROZEN)

## Why this package exists

The frozen VBS scoring tool hard-codes the OLD audit's checkpoints
(PRE U116100 / MID U117100 / POST U118100).  This audit needs three
different arms:

    PRE       = U118100   (g1_camera_v2_p25)
    CONTROL   = U118200   (g1_camera_v2_p25_mirror,          camera-only)
    TREATMENT = U118200   (g1_camera_v2_p25_mirror_v2,       corrected MBS)

Every checkpoint-independent numeric contract (arms, 4-stratum mean margins,
SAME-correction, paired bootstrap seed 20260907 / n 10000, deterministic
noise/timestep seeds, geometry, latent bands) is imported from the frozen VBS
package and reused verbatim.  Only the checkpoint selection, the
checkpoint-identity gate, the pair-root location, and the report targets
differ.  No torch import at module load; no monkeypatching of the frozen
package at runtime.

## Files

    contracts.py         new contracts: checkpoint identity, cohorts, replay
                         gate, classification; re-exports the frozen VBS
                         numeric cores
    reconstruct_stage1.py verbatim copy of frozen final_snapshot/cc_stage1.py
                         with exactly three documented deviations (see below)
    score_three_way.py   stage 3: paired 2x2 scoring worker, one process per
                         DCU, one checkpoint per invocation
    analyze_three_way.py stage 4: replay gate, per-pair contrasts, six
                         pre-registered cohorts, bootstrap, classification,
                         four report writers, self-replay validation
    test_mbs_three_way.py focused tests (no torch, no GPU)

## Provenance of the FROZEN references (never modified)

Worktree base reviewed SHA:
    1b8fce49031b854b835d9f47ea65a3f77b83a2f9
Branch:
    camera-v2-mbs-three-way-same-source-audit-review

Frozen VBS package sha256 (dev-tools/camera_coordinate_causal/vertical_bottom_supervision/):

    a58097ee641da669eba23ef6bcb0cf1e61b50eba41d564b6f1a65ea9bdfd35a4  analyze.py
    8ac1e432cb4aca66290ae0007154fd80a1eea4111fb361ee0714c2deac271e7c  build_pairs.py
    527d865b6508f7a1cebbb5e6d2fb3d11d096128fca9142af1ee1a62872b84341  contracts.py
    7429b7d653ab0d22a8dd6b5d851c6f512630600881d4f24c9cbcf1b6d25fe891  encode_pairs.py
    248b427ebe0f49fa959fd04a6bac4c06ba50bc75546e462ddaa9425b75e8b2aa  feature_audit.py
    32446b1bc4f478a6b090499b22481739948f874cf474184aea98e8c01f11db8a  score_pairs.py

Frozen final_snapshot sha256 (dev-tools/camera_coordinate_causal/final_snapshot/):

    63edc8ef5a62fcb1e64c202f9f12d562713512e44b618e49a719fe0527aa437b  cc_common.py
    384d58e49b3c08b345525aded9485c1bb38c310d426d399aa07acf6182aa65b0  cc_stage1.py
    e5b7ae910df0db7847fcb96d9ecfddf12ce394e383045750da695c62c7c664b9  cc_stage2.py
    aea987d6505338b81a7e9d72084e5bef6a458b04cf41454750f347c779da2177  cc_stage2b.py
    4a855b3134b78f4478003cbdd611945b9a4e1c8e53e95827ec0cc28ea27657a6  cc_stage3.py

Committed frozen reports (worktree reports/):

    1a08b68de84db872e67b3fd3044929884d89c1ae2b6e8b317611be4f1221c4a8  camera-vertical-bottom-supervision-pairs.csv
    edaf361fa0b3df40d29acdf6ce45f602e842f71a681d1b8f231c6de274a18c1b  camera-vertical-bottom-supervision-audit.json

Original frozen pair-manifest sha256 (historical, from the VBS audit):

    7f3a81b248473d0b0591377c1bcd9f7df353066c9716836dabe90d9861f56297

NOTE on the pair-manifest sha: the manifest embeds `created_utc` by design, so
a byte-exact re-derivation of 7f3a81b2 is impossible.  Cohort identity is
instead verified field-by-field against the committed pairs CSV (all 512 rows:
source_shard, sample_id, original_side, k_start, k_end, available,
full_height, signed_shift_start) and end-to-end by the PRE replay gate.

## Deviations of reconstruct_stage1.py from the frozen cc_stage1.py

The file is a verbatim copy of frozen final_snapshot/cc_stage1.py
(sha256 384d58e49b3c08b345525aded9485c1bb38c310d426d399aa07acf6182aa65b0)
except for a provenance docstring header (replacing the one-line frozen
docstring) and exactly three documented functional deviations:

    1. sys.path insert targets the frozen final_snapshot/ directory (the copy
       lives in mbs_three_way/).
    2. The prior-audit checkpoint-evidence loop is reduced to PRE (U116100,
       recorded verbatim) plus an explicit UNAVAILABLE note for MID (U117100,
       which was a /tmp copy on the original host).  Provenance-only; zero
       effect on pair inputs.
    3. The manifest script_sha256 block hashes the frozen scripts from their
       final_snapshot/ location (same frozen files; semantics unchanged).

Every other line — pipeline wiring, quota, freeze, SHUFFLED derangement,
RANDOM arms, arm validation, manifest schema — is byte-identical to the frozen
tool (verified by `diff`: header + the three deviations only).

## The one import-fidelity fix vs the previous draft

`contracts.py` loads the frozen VBS contracts from an explicit file path under
a DISTINCT module name (`vbs_contracts_frozen`).  A plain `import contracts`
would resolve to this module itself (same base name, already present in
sys.modules mid-load) and silently shadow the frozen file — a latent
self-import bug that would have failed at scoring time.  The frozen file is
never copied or modified.

## Run order (on come7, DTK venv, 2 DCUs)

    # 1. reconstruct the frozen stage-1 units (VAE + Qwen freeze, ~4096 units)
    python reconstruct_stage1.py

    # 2. frozen VBS pair build + latent encode (read-only scripts, run as-is)
    python ../vertical_bottom_supervision/build_pairs.py \
        --repo /sakuramoon-runtime/sakuramoon-camera-mbs-three-way-audit \
        --out-root /tmp/camera-vertical-bottom
    python ../vertical_bottom_supervision/encode_pairs.py \
        --repo /sakuramoon-runtime/sakuramoon-camera-mbs-three-way-audit \
        --out-root /tmp/camera-vertical-bottom --device cuda:0

    # 3. PRE-SCORING GATE: reconstructed manifest must be field-identical to
    #    the committed frozen pairs CSV (the original manifest sha is not
    #    byte-reproducible because build_pairs embeds created_utc)
    python analyze_three_way.py --pair-check

    # 4. score, one checkpoint per invocation, one process per DCU (parity split)
    python score_three_way.py --worker 0 --ckpt PRE        # replay gate first
    python score_three_way.py --worker 1 --ckpt PRE
    python score_three_way.py --worker 0 --ckpt CONTROL
    python score_three_way.py --worker 1 --ckpt CONTROL
    python score_three_way.py --worker 0 --ckpt TREATMENT
    python score_three_way.py --worker 1 --ckpt TREATMENT

    # 5. analyze + reports (enforces the replay gate before interpreting)
    python analyze_three_way.py

Scoring is resume-safe by completed (ckpt, pair) key.  Raw artifacts stay
under /tmp/camera-mbs-three-way-audit and are never committed.

## Ruff status (repo config, ruff 0.16)

    contracts.py, score_three_way.py, analyze_three_way.py, test_mbs_three_way.py
        -> ruff-clean (verified on come7)
    reconstruct_stage1.py
        -> intentionally inherits the 11 ruff findings of the frozen
           final_snapshot/cc_stage1.py (verbatim provenance wins over lint;
           the byte-diff vs the frozen source shows ONLY the 3 documented
           deviations).  The frozen cc_stage1.py in this repo carries the
           same 11 findings.

## Focused tests

    python -m pytest test_mbs_three_way.py -q

No torch, no GPU.  Set MBS3_TEST_FROZEN_DIR to a directory holding the frozen
pairs CSV + audit JSON to run off-host (otherwise the 2 frozen-report tests
skip; on come7 all 35 run).

## Immutability (binding)

    * vertical_bottom_supervision/ and final_snapshot/ : never modified
    * src/, config/, scripts/, existing tests, prior reports : never modified
    * three checkpoints : read-only, re-hashed after scoring
    * /tmp artifacts, checkpoints, weights, images, latents, cache, secrets :
      never staged
