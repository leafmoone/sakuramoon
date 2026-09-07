# SakuraMoon Camera — MBS 100U Canary Launch Readiness

Round: **MBS 100U CANARY LAUNCH-READINESS — SAFE STOP-CAP + COMPILE-ON HCU SMOKE**
NO TRAINING / NO BACKWARD / NO OPTIMIZER STEP. Review branch only.

- reviewed base (exact SHA): `18a5e87f3c092065f0ce50c9f9c94df4fd64d201`
- branch: `camera-v2-mbs-canary-launch-readiness`
- worktree: `/sakuramoon-runtime/sakuramoon-camera-mbs-canary-launch-readiness`
- MBS_CANARY_TOOLING_HEAD: `dec876b8b685a8cf8f3301958f7dd1056c49fee0`
- MBS_CANARY_EVIDENCE_HEAD: (this commit — recorded in the round transcript / FINAL COPY)
- round mode: **NO TRAINING / NO BACKWARD / NO OPTIMIZER STEP / HARD STOP**

## BASE

| field | value |
|---|---|
| repository | leafmoone/sakuramoon |
| base SHA | 18a5e87f3c092065f0ce50c9f9c94df4fd64d201 |
| base review status | externally reviewed: MBS core / CUDA condition routing / logical 0.5/0.5 pair loss / same timestep+noise / Qwen logical reuse / telemetry schema v12 / eager HCU forward smoke = PASS; resume source U118100 = READY |
| branch | camera-v2-mbs-canary-launch-readiness |
| worktree | /sakuramoon-runtime/sakuramoon-camera-mbs-canary-launch-readiness (come3) |
| force push | NO |

## SOURCE (read-only recheck, section 18)

| field | value |
|---|---|
| checkpoint | /sakuramoon-runtime/output_model/g1_camera_v2_p25/ckpt_118100_raw-118100-update-cadence |
| COMPLETE | YES (marker "complete") |
| successful updates | 118100 (attempted 118100) |
| stage budget start | 58000 |
| stage budget terminal | 168000 |
| checkpoint cadence | 100 (last_successful_update 118100) |
| fingerprint (manifest sha256) | 821a8c12546a094865972fef0c23152e9754e01f82eed8353f7248b92076b892 — identical to the fix-review round audit |
| modifications | none (read-only) |

## SECTION 4 — CURRENT-LOOP FACTS (confirmed before coding)

- **A.** `resolve_single_gpu_stop_cap_target` with cap absent returns the restored
  `stage_budget.terminal_successful_update` — the runtime invocation target is the
  stage terminal (unchanged production behavior).
- **B.** `SingleGpuTrainingLoop.run` stops on
  `while self.state.successful_updates < self.target_successful_updates`; the loop
  constructor additionally rejects `target <= restored successful_updates`
  (fail-closed) — this is what makes a second invocation at the stop impossible.
- **C.** Post-update handling (update → scheduler step → checkpoint/observer)
  completes inside one while-body iteration, before the termination condition is
  re-checked — so the terminal cadence checkpoint and observer fire at the stop
  update, then the loop exits.
- **D.** U118200 is a normal cadence point under cadence=100
  (118200 % 100 == 0, restored last_successful_update = 118100 committed).
- Extra fail-closed fact: `_resume_state_for_config` rejects any configured
  terminal below the restored terminal — which is why §1's warning applies:
  `planned_updates = 118200` would fail as an illegal stage-budget shrink. The
  canary therefore keeps the stage budget untouched and uses a runtime-only
  invocation stop cap.

## STOP CAP (governed, runtime-only)

| field | value |
|---|---|
| config field | `stage.canary_stop_successful_update: int \| None` (StageConfig; default None = behavior unchanged) |
| STAGE TERMINAL (checkpoint budget) | 168000 — inherited, never mutated |
| SOURCE update | 118100 |
| CANARY INVOCATION STOP | 118200 |
| CANARY LENGTH | exactly 100 successful updates |
| runtime target | min(168000, 118200) = 118200 |
| stage budget mutated | NO (`RawCheckpointState.stage_budget` never written) |
| cadence alignment | 118200 % 100 == 0 (required for this canary; proven by test) |
| fail-closed validation | rejects: non-int, <= 0, <= restored successful update, > stage terminal, `automatic_transition != false` (MBS-specific) |
| resume-from-stop blocked | YES — restored 118200 with cap 118200 → `stop <= restored` rejected before any update (regression test); loop constructor independently rejects target <= restored |
| terminal checkpoint reason | UPDATE_CADENCE (never STAGE_FINALIZE) — loop-order test proves update → scheduler → cadence checkpoint → observer → exit, with no U118201 |

Resolved-config proof (real files, come3):

```
P25 stage:    planned_updates = 168000, automatic_transition = False
CANARY stage: planned_updates = 168000, automatic_transition = False,
              canary_stop_successful_update = 118200   (only added leaf)
```

## COMPILE-ON HCU SMOKE (sections 11-14)

Real HCU/DCU forward-only smoke through the production compile path
(`compile_packed_dit_blocks`, same call as `production.py`; no invented wrapper).
Script: /sakuramoon-runtime/mbs-smoke/hcu_mbs_compile_smoke.py
Evidence: /sakuramoon-runtime/mbs-smoke/compile-smoke-evidence.json

| field | value |
|---|---|
| device | BW (Hygon DCU) cuda:0, 2 available |
| torch | 2.9.0 (DTK venv /sakuramoon-runtime/sakuramoon-dtk-venv) |
| compile enabled / dynamic / mode | true / true / max-autotune-no-cudagraphs — resolved from the real canary config, byte-identical to the U118100 production resolved config |
| backend | inductor |
| compiled block count | 20 PackedDiTBlock (== active slots) |
| recompile-limit policy | `fail_on_recompile_limit_hit = True` (set by the production install); `suppress_errors` verified disabled — no silent eager fallback is possible |
| layout sequence | A → B → A through the SAME compiled model/runtime |
| CASE A | logical 2, mirror pairs 0, physical 2 (two sub-threshold vertical null-condition rows) |
| CASE B | logical 2, mirror pairs 1, physical 3 (row 0: applied vertical shift 5.1875, mirror selected, ACTIVE artist condition; row 1: sub-threshold shift 1.75, null condition) |
| logical view counts (Qwen input) | [2, 98] in all three measures (1 Qwen call per measure) |
| physical view counts (VAE input) | [2,3,256,256] / [3,3,256,256] / [2,3,256,256] (1 VAE call per measure) |
| coordinate map counts | 2 / 3 / 2 (matches physical views) |
| mirror applied counts | 0 / 1 / 0 |
| mirror table (B) | logical 2, vertical 2, eligible 1, selected 1, applied 1, extra 1, physical 3 |
| finite losses | A1 [0.0791, 0.0826], B [0.0687, 0.0686], A2 [0.0911, 0.1029] — all finite, positive, logical-domain shape (2,) |
| same pair timestep | bitwise equal (B) |
| same pair noise | bitwise equal (B); rows distinct |
| active condition device parity | prepared state on cuda:0 — the Fix-A device-local remap + fail-closed guard path completed without tripping |
| measure times | A1 32.69 s (first forward performs the max-autotune compilation), B 3.20 s (second physical shape accepted by the dynamic compiled model), A2 0.14 s (warm re-entry) |
| compile exception | none |
| recompile-limit failure | none (guard armed; B/A2 succeeded after the first compiled shape) |
| eager fallback | none possible (suppress_errors disabled; all three forwards completed on the compiled path) |
| parameter/buffer mutation | none — sha256 + data_ptr + grad-None across all composite/Qwen/VAE tensors before vs after all three measures |
| optimizer object usage | none (no optimizer/scheduler constructed) |
| checkpoint write | none — checkpoint dir file list + manifest sha256 unchanged |
| peak allocation | 6.992 GiB |
| result | **PASS — 41/41 checks** |

## MBS / CAMERA POLICY (unchanged, section 10)

- mirror: vertical only (`vertical_mirror_pair_v1`), min latent shift 2.0,
  pair probability 1.0, pair weight 1.0 — no horizontal mirror
- camera: `hdm_shifted_square_v2`, probability 0.25, zoom 1.10-1.50
- no content rejection, no CLIP/PE in hot path

## IDENTITY (section 17/21)

- default no-cap: invocation target == stage terminal (tests 1, 12)
- production config unchanged, no cap (test 12)
- P25 config unchanged, no cap (test 13)
- MBS canary resolved: planned_updates stays 168000 (test 14), stop cap
  resolves exactly 118200 cadence-aligned (test 15)
- optimizer / scheduler / RoPE / coordinate math / DiT / conditioner /
  Qwen / VAE / camera geometry / mirror geometry / objective: unchanged
  (diff scope is exactly the 5 whitelisted files)

## VALIDATION (section 22/23)

| gate | result |
|---|---|
| new deterministic tests | 18 (0 skip, 0 xfail), 18/18 PASS on come3 (21.26 s) |
| affected suites | test_camera_mirror_balance.py + test_mbs_canary_stop_cap.py: 53/53 PASS |
| full unit — new tree | **955 passed / 4 failed** (251.65 s) |
| full unit — base 18a5e87 tree | 937 passed / 4 failed (255.94 s) |
| failure comparison | FAILED sets byte-identical (diff of sorted lists empty): 4 pre-existing `test_cmuon_fp32_forensic` failures; **0 new failures**; 955 = 937 + 18 new tests |
| environment note | both trees run with the identical command, `--ignore`-ing the 2 deepghs tests (DTK venv lacks onnxruntime — pre-existing environment gap, out of diff scope) |
| ruff (changed scope, repo py311 config) | clean: schema.py, runtime.py, both test files |
| py_compile | clean: all 4 changed/new Python files |
| git diff --check | clean |
| secrets scan (diff + changed files) | 0 hits |
| staged surface | text only — no binary/model/checkpoint/dataset/image, no /tmp |

### New tests (section 15/16/8 mapping)

1. `test_no_cap_target_is_stage_terminal`
2. `test_cap_shortens_target_to_stop`
3. `test_cap_does_not_mutate_stage_budget`
4. `test_cap_equal_to_source_rejected`
5. `test_cap_below_source_rejected`
6. `test_cap_above_terminal_rejected`
7. `test_cap_zero_or_negative_rejected`
8. `test_cap_non_int_rejected_at_runtime`
9. `test_cap_with_automatic_transition_rejected`
10. `test_cap_aligned_to_cadence_accepted`
11. `test_resume_at_stop_blocked_before_training`
12. `test_loop_source3_cap5_runs_exactly_updates_4_and_5`
13. `test_terminal_checkpoint_reason_is_update_cadence_not_stage_finalize`
14. `test_terminal_order_update_scheduler_checkpoint_observer_exit`
15. `test_default_production_config_has_no_cap_and_target_is_terminal`
16. `test_p25_config_unchanged_no_cap`
17. `test_mbs_canary_resolved_planned_updates_stays_168000`
18. `test_mbs_canary_stop_cap_resolves_118200_and_cadence_aligned`

### One pre-existing test updated (legitimate consequence of section 9)

`tests/unit/data/test_camera_mirror_balance.py::
test_canary_config_changes_only_mirror_and_identity` previously asserted that the
canary may only touch identity + the mirror table. Section 9 explicitly adds the
stop cap to the canary config, so the allowlist now also admits
`stage.canary_stop_successful_update` — and the test was **strengthened**, not
weakened: it now pins the cap value to exactly 118200 and asserts
`planned_updates` inheritance is unchanged. No other assertion was relaxed.

## SECURITY

- secrets = 0
- checkpoint staged = NO
- dataset staged = NO
- binary staged = NO (text reports only)

## READINESS (section 21)

| condition | status |
|---|---|
| stop-cap semantics correct | PASS (18 tests) |
| no stage-budget shrink | PASS (test 3 + resolved config proof) |
| resolved canary planned_updates still 168000 | PASS (test 17 + probe) |
| runtime stop exactly 118200 | PASS (test 18 + probe) |
| resume-at-118200 blocked | PASS (test 11 + loop-constructor fail-closed) |
| cadence checkpoint at 118200 proven | PASS (tests 13/14) |
| compile-on real HCU smoke PASS | PASS (41/41) |
| dynamic physical 2/3 layouts PASS | PASS (A→B→A) |
| no new relevant test failure | PASS (failure sets identical) |
| security clean | PASS |
| U118100 provenance PASS | PASS (fingerprint match) |

**CANARY_LAUNCH_READY = TRUE**

Even TRUE, this does NOT authorize training.

## AUTHORIZATION (section 29)

| action | status |
|---|---|
| training started | NO |
| backward | NO |
| optimizer step | NO |
| resume started | NO |
| P50 started | NO |
| production changed | NO |

## NEXT

HARD STOP. Send the exact pushed evidence SHA to the external reviewer.
NO TRAINING UNTIL EXTERNAL REVIEW.
