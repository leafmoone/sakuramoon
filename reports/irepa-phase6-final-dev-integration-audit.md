# iREPA Phase 6A — Final Dev Integration + Revalidation Audit

Date: 09-04 (machine: salt13, 2x BW DCU, DTK torch 2.9.0+das)
Branch: `iprea-phase6-integration` (worktree `/sakuramoon-runtime/sakuramoon-iprea-phase6`)
Anchors:
- P5 head: `4571d75b006c77e628a693fa7038dece47556db2` (origin/iprea)
- P5.5 evidence head: `87cc153803362fc6068c42b356fc02e614cb6c5a`
- final dev: `3a341c0efa8aa6c82b41e508cf5fa2730e20fddb` (origin/dev)
- merge-base (verified): `f045f87830913a0220d66a60408b856508d68910`
- merge commit: `174c52a076e403ef98b002f70a9a5dc550ffacf6`
  parents: [87cc1538…, 3a341c0…] (order verified via `git cat-file -p HEAD`)
- PHASE6_FIX_HEAD: NONE (no adaptation commit required)

## Verdict: **PASS — ready for user gate (STOP; Phase 6B NOT started)**

## 1. Entrance gate (deviations documented)

The salt13 worktree was found at `00bd795` with the P5 work uncommitted
(7 modified + 9 untracked files = exactly the 16 files of commit 4571d75).
GitHub remote verified first (via ai_proxy 10.16.1.51:3128): `iprea=4571d75…`,
`dev=3a341c0…` exactly. Fetch + byte-level reconciliation of worktree vs
4571d75 showed: (a) 3 P5 report files present only in the commit,
(b) the 6 P5 untracked files identical, (c) the ONLY content drift =
CRLF line endings in 2 files (`checkpoint/artifact.py` 294 lines,
`optim/groups.py` 167 lines; commit = LF). `--ignore-all-space` diff = 0.
Backup taken (stash `p5-worktree-backup-20260904` + durable branch
`p5-worktree-backup-20260904`), then `git reset --hard 4571d75` (content
no-op). Post: branch=iprea, HEAD=4571d75, clean.
`reports/irepa-phase5-copy-report.md` lives in the Windows clone
(`D:\sakruamoon\sakuramoon`, branch iprea @ 4571d75, untracked there —
the only untracked file); it is absent from the salt13 worktree, which is
a deviation from the spec's "single known untracked" expectation, not a
violation (the file stays untracked and unstaged in its own clone).

## 2. P5.5 reconciliation (evidence commit 87cc153)

- Tracked `reports/irepa-phase5-checkpoint-migration-audit.md`: 8 targeted
  count corrections (287→289, 146→148; migrated 289/141/148 →
  291/141/150) + short Phase 5.5 note. Residual "287/146" mentions exist
  only inside the note (historical reference).
- Untracked `reports/irepa-phase5-copy-report.md` (Windows clone): 2
  targeted corrections; verified 0 residual; still the only untracked
  file; never staged/committed.
- New evidence: `reports/irepa-phase5-5-parameter-reconciliation.{md,json}`
  (topologies 289/141/148 ×3 + 291/141/150; COMMON 289 / REAL_ONLY 0 /
  S18_ONLY 0; six mismatch classes all 0; migration count-hardcodes 0,
  FQN-driven; synthetic dry-run PASS; source update anchor 111500;
  added projector FQNs; stale-doc root cause; Phase5 conclusion NOT
  weakened; no live checkpoint claim).
- Evidence commit stages exactly 3 files (`--name-status` verified +
  `--check` clean). No push (forbidden).

## 3. Pre-merge semantic audit (/tmp/irepa-phase6-overlap-audit.txt)

- DEV side: 34 files (+8866/−49); IREPA side: 68 files (+12539/−67).
- OVERLAP = exactly 3 files: `train/production.py`, `train/runtime.py`,
  `train/step.py`.
- Hunk-level audit: no hunk region overlap in any of the 3 files:
  - production.py — DEV: `checkpoint_source` param + `run_id` into
    HybridCMuon build (F2 telemetry identity), terminal fix in
    `_resume_state_for_config`, checkpoint_source at the
    `_build_optimizer` call site. IREPA: `require_production_irepa_readiness`
    gate, `_runtime` irepa teacher/schedule wiring, lifecycle call sites.
    Disjoint regions.
  - runtime.py — DEV: 1 hunk (terminal check
    `terminal_successful_update != config.stage.planned_updates`). IREPA:
    iREPA wiring (imports, PreparedTrainingBatch.irepa_targets,
    RuntimeMeasurement irepa splits, measure/_loss, runtime
    set_irepa_weight). DEV hunk at base L963 is outside all IREPA hunk
    regions.
  - step.py — DEV: `_note_forensic_update` (new fn) + 1 call line. IREPA:
    imports, `TrainableCompositeIRepaOutput`, composite
    init/bind/tapped-forward, projector grad-norm, `__all__`. Disjoint.

## 4. Merge (git merge --no-ff --no-commit 3a341c0)

Automatic merge, ZERO conflicts, all 3 overlap files auto-merged.
Staged-diff audit (both directions):
- HEAD vs final-dev: exactly the IREPA file set + 1 documented lint fix.
- HEAD vs P5.5: exactly the DEV file set + 1 documented lint fix.
- Union markers present (checkpoint_source ×3, require_production_irepa_readiness ×3,
  terminal fix ×1, irepa_teacher ×10, _note_forecin_update ×2, irepa_projector_grad_norm ×3).

### Invariants checklist

DEV invariants (all verified):
- CMuon implementation verbatim: `git diff 3a341c0..HEAD -- optim/cmuon.py
  optim/guarded_canonical.py optim/fp32_rescue.py optim/cmuon_hardfail.py
  optim/cmuon_ns_trace.py` = 0 lines.
- F3 soft-below-floor semantics: fp32_rescue.py byte-identical to final
  dev; `stats_log_every_n` default = 100 (dev cadence); per-sample data
  skip prints absent (0 matches) while `_trace_sample` + `rejection_observer`
  retained.
- F2 hard-fail capsule identity: `checkpoint_source` plumbed (production.py).
- terminal/planned-updates fix: both call sites (production.py, runtime.py).
- MemoryError data-skip: pipeline.py dev change present (M in merge).
- `config/train_g1_cmuon_production.toml` present with final-dev semantics
  (no `[irepa]` section → iREPA disabled by default in production config);
  NO iprea-added config file exists (grep of IREPA file list = NONE).
- Planned-updates terminal test: `tests/unit/train/test_planned_updates_terminal.py`
  PASS (unit batch).

IREPA invariants (all verified):
- IRepaConfig, frozen PE-Spatial teacher asset contract, IRepaAlignment
  (MixedPrecisionConv2d 3x3 2560→768), slot-08 tap binding, z-score FP32
  target, lambda schedule + binding, no-skip lambda=0 contract, telemetry
  v11 metric splits, explicit migration CLI + irepa_state sidecar,
  FQN-driven routing, projector in AdamW only (CMuon set 141 invariant),
  preflight fail-closed gate — all present, all on the IREPA file set
  (untouched by dev), all exercised by the PASSing gates below.

Forbidden-semantic-change list: zero violations (no CMuon NS/momentum/
allowlist/ceiling/safety/FP32-rescue-acceptance change; no teacher/gamma/
eps/projector/tap/cosine/lambda/JLT change; no threshold change to pass a
gate; no new production-looking TOML; no push/force/rebase/amend/squash;
no whole-file conflict resolution (no conflicts at all); the untracked
copy-report remains untracked/unstaged/uncommitted).

### Merge-commit hygiene (documented deviations)

Two pre-existing lint defects were fixed inside the merge commit so the
merged tree satisfies the pre-commit ruff gate (spec §25); both were
proven pre-existing by running ruff on both parents (dev parent: 1 error;
iprea parent: 1 error; merge adds 0):
1. I001 import order in `tests/gpu/checkpoint/test_irepa_checkpoint_save.py`
   (ruff --fix; non-semantic, test file).
2. EXE001 shebang-without-exec-bit in `tests/gpu/optim/cmuon_capsule_teardown.py`
   (chmod +x; dev-committed the file 100644 despite the shebang).
Additionally `git diff --cached --check` flags one pre-existing whitespace
artifact in the dev-added `config/train_g1_cmuon_production.toml`
(blank line at EOF); left byte-identical to final dev deliberately
(production-looking file; cosmetic; documented).

## 5. Static gates (merged tree)

- ruff `check src tests`: **All checks passed** (after the 2 documented fixes).
- pyright: NEW-in-src = **0**, proven against the UNION of both parents'
  error sets (merged 599 signatures; dev 600; iprea 573; merged ⊆ union;
  the 1 signature present in a parent but gone from the merged tree is
  `checkpoint/load.py reportUnnecessaryContains` — removed by the P5
  load.py rewrite, matches the P5 audit "load 10→9").
- Targeted unit (merged tree): 162 passed / 4 failed — the 4 failures
  (`tests/unit/optim/test_cmuon_fp32_forensic.py`: test_bcd_hard_fail
  [below_floor], test_e_writer_failure…, test_f_serialization_failure…,
  test_m2_mirror_failure…) reproduce **exactly** on the DEV parent tree
  (4 failed / 8 passed) = BASELINE_SHARED (F2 forensic tests expecting
  pre-F3 hard-fail semantics vs final-dev F3 soft-below-floor);
  integration-only failures = 0.
- Targeted GPU (merged tree): `tests/gpu/checkpoint/test_irepa_checkpoint_save.py`
  = 6 passed.

## 6. CMuon persistent-state contract (§11-13)

Machine-readable key tree: `/tmp/irepa-phase6-final-cmuon-state.json`.
- Production class GuardedCanonicalHybridCMuon top-level keys: optimizer
  (AdamW int-id state {step, exp_avg, exp_avg_sq} + param_groups ×2),
  sr_rng, cmuon {momenta(141 bf16), ns_steps, ns_coefficients, momentum,
  nesterov, eps, momentum_dtype, chunk_rescale_sqrt_n,
  qkv_group_rescale}, routing {cmuon, adamw, counts}, transition
  {from_adamw8bit, preserved_adamw_params, dropped_cmuon_params, note},
  hybrid_cmuon_schema_version=1, guard {schema_version=1, config(6 keys),
  references, skip_total, skip_by_role, skip_by_fqn, observations,
  bootstrap_mode, owner_mapping_version, world_size, canonical_ns_mode,
  ns_map}, guarded_canonical_schema_version=1.
- Sidecar optimizer_schema.json v2: groups + hybrid_cmuon +
  guarded_canonical blocks; iREPA-only addition irepa_state.json (v4).
- Migration transparency: **class A (literal transparent)** —
  `migrated_hybrid = {**hybrid, optimizer: <re-indexed>, routing:
  <target manifest>}`: every other top-level key copied verbatim by
  reference; AdamW state values copied by reference with ids re-assigned
  in canonical FQN order (1:1 by name); CMuon routing FQN-set-verified
  unchanged; schema algorithm blocks verbatim. drop/reset/rewrite/
  normalize/rebuild: none.
- Recursive exact-state test on a synthetic FINAL-DEV (guarded-canonical)
  fixture: NEW test
  `tests/unit/checkpoint/test_irepa_migration_finaldev_state.py` — PASS
  (top-level key set unchanged; cmuon/sr_rng/transition/guard recursive
  bit-exact; AdamW per-FQN state bit-exact across the id remap; additions
  exactly the 2 projector FQNs, stateless until first step; schema v2
  algorithm blocks deep-equal). The test file is an untracked Phase6A
  audit artifact (not committed; PHASE6_FIX_HEAD=NONE).

## 7. HCU revalidation legs (merged tree)

- §14 Topology: no-iREPA = **289/141/148**; iREPA = **291/141/150**;
  CMuon FQN set identical (141); AdamW delta = exactly the 2 projector
  FQNs — **PASS** (/tmp/irepa-phase6-topology-dryrun.json).
- §15 Migration dry-run on FINAL-DEV schema: real production-shape source
  chain (2 real updates, production optimizer) → `migrate_irepa_checkpoint
  (dry_run=True)`: source_update=2, anchor=3, projector_in_channels=2560,
  added_fqns = exactly the 2 projector FQNs, added_optimizer_parameters=2,
  destination not written — **PASS** (same JSON).
- §16 S18 deterministic bit-exact parity:
  `test_lambda_zero_one_update_parity` **PASS** (merged tree; all 289 old
  params torch.equal, 141 CMuon momenta exact, 148 AdamW exact, SR RNG
  exact, guard/rescue exact, MAIN/TOTAL losses bit-exact, TOTAL==MAIN).
- §17 Production NS HCU leg: `test_lambda_zero_production_ns_behavior`
  **PASS**; worst CMuon rel-rms = **4.106e-3** (worst param
  `dit.conditioner.shared_block_projection.weight`); Phase5 measured
  4.781e-3 → hard gate 5e-2 = **12.2x margin**; rescue counters identical
  across arms (A=B: bf16_attempts=498, safety_failures=0, fp32_attempts=0,
  fp32_rescues=0, fp32_rescue_failures=0); zero nonfinite, zero partial
  commits; deterministic components all bit-exact.
- §18 F3 below-floor regression: `tests/unit/optim/test_fp32_rescue_f3.py`
  PASS (finite-below-floor ACCEPT; nonfinite/above-ceiling FAIL) + 2-rank
  HCU `fp32_rescue_f3_2rank.py`: **PASS** (verdict=PASS; healthy ok;
  soft_rescue ok — 1 fp32_low_delta rescue at attention_content_gate;
  mixed_step ok — 2 low-delta rescues; above_ceiling → CMuonSafetyError;
  nonfinite → CMuonSafetyError; all rank spreads 0.0).
- §19 Hard-fail forensic isolation: fp32_rescue.py/guarded_canonical.py
  byte-identical to final dev (zero semantic delta); forensic unit suite:
  8/12 PASS, 4 = BASELINE_SHARED (pre-F3 expectations, repro on dev
  parent); above-ceiling/nonfinite hard-fail paths among the passing set.
- §20 2-rank DDP full migration/resume chain: `irepa_ddp_lambda_zero_smoke.py`
  (torchrun ws=2, deterministic NS): **PASS** — chain 89.8s,
  source_update=2 → next_update=3, lambda_zero_parity="bit_exact (primary
  gate)", cross_rank_fingerprints=864 with 0 mismatches, guard/rescue
  identical (249 bf16 attempts, 0 failures). Matches the P5 2-rank baseline.
- §21 Compile/activation-checkpoint matrix at production shape (d=2560,
  depth 20, 20Q/5KV, head_dim 128, bf16, slot08): 5/5 legs PASS
  (eager×{none,alternating,all}, compile×{none,all}); MAIN loss and
  grad-norm identical across all legs; slot08 capture bit-identical for
  act-ckpt legs, 1-bf16-ulp max abs for compiled legs (allclose gate);
  test-only dynamo compiled-namespace cleanup between compile legs —
  **PASS** (/tmp/irepa-phase6-compile-matrix.json).
- §31 Performance smoke (production-shape step, 2 warmup + 5 timed
  updates, 1 batch, real production optimizer from each tree's own
  `train_g1_fp32_rescue_r1.toml`):
  | leg | tree | tensors | median s/update |
  |---|---|---|---|
  | A final-dev no-iREPA | /tmp/rt-dev (3a341c0) | 289 | 0.9406 |
  | B integrated no-iREPA | merged v3 | 289 | 0.9284 (−1.3% vs A) |
  | C integrated λ=0 | merged v4 + teacher | 291 | 0.9571 (+1.8% vs A) |
  Conclusion: B/A = 0.987 → **zero-impact guarantee holds** (no measurable
  overhead on the no-iREPA path); C/A = 1.018 → λ=0 full iREPA graph
  (frozen teacher encode + projector + cosine) costs **+1.8%** per update;
  per-leg spread < 1% (min/max). — **PASS** (JSON: /sakuramoon-runtime/p6a-perf-{A,B,C}.json).

## 8. Telemetry / config / preflight (§22-24)

- Telemetry superset: merged observer carries all dev fields + iREPA v11
  metric splits (`_irepa_metric_splits` → main, irepa, irepa_weighted,
  irepa_cosine_mean, irepa_lambda); t-bin and noise-bucket telemetry stay
  strictly MAIN-JLT-only (buckets derive from the main flow-matching loss
  branch); dev cadence semantics retained (observations summary every 100,
  no per-sample skip prints).
- Config: `config/train_g1_cmuon_production.toml` = final-dev byte
  (dev addition, untouched); no iprea-added config file; production
  TOML has no `[irepa]` section (iREPA off by default; Phase 6B owns the
  canary config).
- Preflight gate matrix (unit, all PASS in the merged tree):
  - irepa absent → pass (test_absent_irepa_passes_production_readiness)
  - irepa disabled → pass (test_disabled_irepa_passes_production_readiness)
  - enabled + no resume → fail-closed
    (test_enabled_irepa_without_resume_fails_production_readiness)
  - enabled + unmigrated checkpoint → fail-closed
    (test_enabled_irepa_with_unmigrated_checkpoint_fails)
  - enabled + migrated checkpoint → pass
    (test_enabled_irepa_with_migrated_checkpoint_passes)
  - enabled + invalid sidecar → fail-closed
    (test_enabled_irepa_with_invalid_state_sidecar_fails)
  - tap slot not active at current depth (depth-16 fail / depth-20 pass
    semantics) → test_irepa_composite.py tap tests PASS
  - teacher not installed → fail (test_set_irepa_weight_requires_an_installed_teacher)
  - projector FQN delta ≠ exactly 2 → migration fail-closed (unit migration suite)
  - CMuon guard schema/world-size/owner-mapping mismatch → fail-closed at
    load (guarded_canonical load_state_dict; exercised by S18 resume chain)

## 9. Dual-worktree full pytest (§29-30)

Both runs: `pytest tests/ -q -p no:cacheprovider -rf`, salt13 HCU, one
visible DCU per run (parallel on the two DCUs).

| tree | result |
|---|---|
| merged `iprea-phase6-integration` @174c52a | **7 failed, 1040 passed, 3 skipped** (2132s) |
| dev parent 3a341c0 | **7 failed, 810 passed, 2 skipped** (1268s) |

Delta = exactly the 230 new iREPA tests — **all pass**; failure set
byte-identical (same 7 test IDs in both trees):
1. `test_pipeline_encoders.py::test_real_pipeline_qwen_and_mage_encode_one_batch`
   — BASELINE_SHARED (Qwen asset absent on salt13, machine condition).
2. `test_varlen_attention.py::...[host_metadata]` — BASELINE_SHARED
   (machine-independent baseline bug, present on dev parent).
3-6. 4× `test_cmuon_fp32_forensic.py` (bcd below-floor / e writer /
   f serialization / m2 mirror) — BASELINE_SHARED (F2-expected pre-F3
   hard-fail semantics vs final-dev F3 soft-below-floor; reproduce
   exactly on the dev parent).
7. `test_fp32_rescue.py::test_DE_two_rank_owner_only_and_consistency`
   — identical in both trees (2-rank DDP test under a single-visible-DCU
   pytest env; the real 2-rank legs ran green as dedicated torchrun runs).

Skips: dev 2 = merged first 2 (the two real-ModelScope asset tests,
skipped on salt13 by asset absence, identical in both trees). Merged's
3rd skip = `test_pe_spatial_geometry.py::test_teacher_device_mismatch_rejected`
— a NEW iREPA test with an explicit `pytest.skip("requires two
accelerators")` (the run exposed one DCU); by-design conditional skip,
not a regression, and the 2-accelerator path is covered by the
dedicated 2-rank DDP leg (PASS, 0/864).

**§29/30: integration-only failures = 0; no new skip/xfail — PASS.**

## 10. Immutability / scope audit (§33)

- `git diff 3a341c0..HEAD -- optim/cmuon.py optim/guarded_canonical.py
  optim/fp32_rescue.py optim/cmuon_hardfail.py optim/cmuon_ns_trace.py`
  = 0 lines (all dev-verbatim; no iREPA-ancestry modification of dev CMuon
  files).
- `git diff --name-only 3a341c0..HEAD` ⊆ IREPA file set ∪ {1 documented
  lint fix}; `git diff --name-only 87cc153..HEAD` ⊆ DEV file set ∪
  {1 documented lint fix}.
- Merge commit parents verified in order [87cc153, 3a341c0].
- No push of any ref; no production checkpoint touched; no training
  started; salt11/salt14 untouched.

## 11. PASS conditions (28-item) — summary

| # | condition | status | evidence |
|---|---|---|---|
| 1 | Entrance: remote verified, worktree == P5 head 4571d75 (clean) | PASS | §1 (CRLF-only drift, stash+branch backup, reset) |
| 2 | P5.5 evidence commit = exactly 3 files, no push | PASS | 87cc153 `--name-status`; push count 0 |
| 3 | Tracked P5 audit counts corrected (287→289 / 146→148) | PASS | 8 targeted edits + Phase5.5 NOTE |
| 4 | Untracked P5 copy report corrected, still sole untracked (Windows clone) | PASS | 2 edits, 0 residual, `git status` = 1 untracked |
| 5 | Overlap = exactly 3 train files, hunk-disjoint, per-hunk semantic union | PASS | /tmp/irepa-phase6-overlap-audit.txt |
| 6 | Merge commit parents in order [87cc153, 3a341c0] | PASS | `git cat-file -p HEAD` |
| 7 | Scope both directions: dev set ∪ 1 lint / irepa set ∪ 1 lint | PASS | §4 staged-diff audit |
| 8 | 5 CMuon core files byte-identical to final dev | PASS | `git diff 3a341c0..HEAD` = 0 lines |
| 9 | F3 semantics preserved (cadence 100, trace/rejection kept, no spam) | PASS | fp32_rescue.py identical; pipeline hunks present |
| 10 | F2 hard-fail capsule identity (checkpoint_source) | PASS | 3 hits in production.py |
| 11 | terminal/planned-updates fix present (both call sites) | PASS | runtime.py + production.py |
| 12 | Production TOML = final-dev byte; no [irepa]; no new iprea config | PASS | byte compare; config file lists |
| 13 | ruff PASS (merged tree) | PASS | 2 pre-existing fixes documented |
| 14 | pyright NEW-in-src = 0 (union-of-parents baseline) | PASS | 599 ⊆ (600 ∪ 573); GONE=1 explained |
| 15 | Targeted unit: integration-only failures = 0 | PASS | 162P/4F, 4F reproduce on dev parent |
| 16 | Targeted GPU save tests | PASS | 6 passed |
| 17 | §13 final-dev schema recursive-exact migration test | PASS | new test, untracked artifact, 1 passed |
| 18 | Topology 289/141/148 & 291/141/150, CMuon set invariant, AdamW delta exact | PASS | /tmp/irepa-phase6-topology-dryrun.json |
| 19 | Migration dry-run on final-dev schema (real source chain) | PASS | anchor=3=src+1, delta=exactly 2 projector FQNs |
| 20 | λ=0 deterministic S18 bit-exact (all classes, 0 mismatches) | PASS | test PASSED; 289/141/148 bit-exact; TOTAL==MAIN |
| 21 | Production NS HCU worst rel RMS ≤ 5e-2 | PASS | 4.106e-3 (12.2x margin; P5: 4.781e-3) |
| 22 | F3 below-floor regression (unit + 2-rank HCU) | PASS | f3 unit PASS; 2-rank verdict=PASS |
| 23 | Hard-fail forensic isolation (no semantic delta) | PASS | 0-line diff; 4F = BASELINE_SHARED |
| 24 | 2-rank DDP full migration/resume chain | PASS | 0/864 mismatches; bit_exact primary gate |
| 25 | Compile/act-ckpt matrix at production shape | PASS | 5/5 legs; loss/grad identical; capture stable |
| 26 | Performance smoke (zero-impact + λ=0 overhead) | PASS | B/A=0.987; C/A=1.018 (+1.8%) |
| 27 | Telemetry superset + t-bin MAIN-JLT only + config/preflight matrix | PASS | §8 (6 gate tests + tap + teacher PASS) |
| 28 | Dual full pytest: integration-only failures=0, no new skip/xfail | PASS | merged 1040P/7F/3S vs dev 810P/7F/2S; failure set byte-identical; +230 iREPA tests all pass; 1 extra skip = new iREPA 2-accel test (by design) |

Immutability (cross-cutting, all PASS): no push of any ref; no production
checkpoint touched (salt11/salt14 untouched); no training started;
`reports/irepa-phase5-copy-report.md` untracked/unstaged/uncommitted;
REAL_PRODUCTION_CHECKPOINT_TEST = PENDING (no user-copied COMPLETE RAW
checkpoint on the experiment machine — not a FAIL per spec).

## 12. Artifacts

- /tmp/irepa-phase6-overlap-audit.txt (946 lines, 3-file hunk audit)
- /tmp/irepa-phase6-final-cmuon-state.json (key tree + transparency)
- /tmp/irepa-phase6-topology-dryrun.json (§14+15)
- /tmp/irepa-phase6-compile-matrix.json (§21)
- /sakuramoon-runtime/p6a-*.log (HCU leg logs)
- reports/irepa-phase6-final-dev-integration-audit.{md,json} (this file, unstaged)
- reports/irepa-phase6-state-parity.json (unstaged)
- reports/irepa-phase6-copy-report.md (unstaged)
- Untracked Phase6A audit test: tests/unit/checkpoint/test_irepa_migration_finaldev_state.py
