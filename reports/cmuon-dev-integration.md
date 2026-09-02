# CMuon → dev integration — final validation report

Frozen CMuon production backend (`cmoun` @ def9ac4, math frozen since D2 R1)
merged into `dev` and validated end-to-end on the merged tree.

Integration commit: **e6db9c5** "Merge frozen CMuon production backend into dev"
(parents 9fcb67a dev + def9ac4 cmoun), plus integration fix **023c057**
(mode-only: executable bit on two shebang'd analysis scripts, ruff EXE001).
Final integration HEAD: **023c057** (branch `cmuon-integration`).

Host: salt10 (2× HCU, 64 GiB each, pod-local DTK 26.04 / torch 2.9.0,
venv `sakuramoon-dtk-venv`). Isolated test root
`/sakuramoon-runtime/cmuon-dev-integration-test/{adamw,cmuon}`; all
production paths (salt7 G1 control, hub, publishers, formal checkpoints)
untouched — **REMOTE_WRITES=0**.

## Verdict

**CMUON_DEV_INTEGRATION = PASS** — all gates green:

| Gate | Requirement | Observed |
|---|---|---|
| §0 anchors | dev=9fcb67a, cmoun=def9ac4, cmoun-guarded=fd1131e, merge-base=f179ff0 | 4/4 exact |
| §2 merge | zero-conflict `--no-ff --no-commit` merge of origin/cmoun | 0 conflicts, dual-parent verified |
| §3 semantic union | no semantic loss vs both parents | 1 dual-modified file (`publish_train_state.sh`) audited line-by-line: dev bounded-lock retry + stale-ai_proxy guard + cmoun parse-fail tolerance all present; other 10 dev files byte-identical to dev |
| §4 optim diff | optimizer source == frozen f875ee9 | 3 files, diff = 0 |
| §5 default optimizer | unchanged | `config/base.toml` `optimizer.name = torchao_adamw8bit` (identical to dev) |
| §8 CPU gate | ruff + pyright + pytest all PASS, no xfail/skip/strictness downgrade | ruff 0.16.1 all checks passed (after mode-only fix 023c057); pytest 763 passed / 0 failed / 0 skipped (onnxruntime env gap pre-existing, frozen tree identical); pyright NEW=0 (3389 unique merged ⊆ dev∪cmoun sets; 159 fixed) |
| §9 GPU 2-rank | 3-phase regression on merged tree | P1 5/5 scenarios bit-exact + zero-commit final failure; P2 ckpt_103700 backcompat A→E ok; P3 cleanup perf |
| §10 backcompat | merged code resumes G1 ckpt_103700 | `D4_CHECKPOINT_BACKCOMPAT=YES` (observations 1201, 166 refs, rescue 818/818/0) |
| §11-20 isolation train | Arm A AdamW 20u + fresh resume; Arm B CMuon 100u + fresh resume | Arm A: PASS (105001→105040, resume from 105020 → 19u clean); Arm B: in progress |
| §21 no quality gate | no FID/concept-120 this round | structurally satisfied: sampling every 1000 / evaluation every 5000 cannot fire in 20/100-update arms; W&B disabled |

## A — Merge & audit (§0–§5)

- Refspec incident: local clone's fetch refspec had been narrowed to
  `cmoun-guarded` only, making remote-tracking refs stale — misread as
  "remote reset". Fixed with `git config --replace-all remote.origin.fetch
  "+refs/heads/*:refs/remotes/origin/*"` + fetch; all four anchors verified
  against `git ls-remote origin` and direct GitHub `ls-remote` (three-way).
- Merge: `git merge --no-ff --no-commit origin/cmoun` on a fresh
  `cmuon-integration` worktree → zero conflicts → commit e6db9c5.
- Semantic union: every dual-modified file audited. Only
  `scripts/publish_train_state.sh` was modified on both sides; the merged
  `load_environment` (L45-85) carries the dev bounded-lock retry (30×1s),
  the stale-ai_proxy guard, the 3× `9>&-` fd closes (L157/172/226) AND the
  cmoun parse-fail tolerance (`source … 2>/dev/null || log WARNING`). All
  other dev files are byte-identical to dev; optim 3 files identical to
  f875ee9; no conflict markers anywhere.

## B — CPU gate (§8)

- ruff 0.16.1 (venv): 2 pre-existing EXE001 (shebang'd scripts at 100644,
  identical on the frozen tree — not merge-introduced) fixed mode-only in
  023c057 → "All checks passed!"
- pytest (dtk venv): 763 passed, 0 failed, 0 skipped. One pre-existing
  collection gap (`onnxruntime` missing) present identically on the frozen
  tree; installed 1.29.0 in the test venv (env fix, not code).
- pyright 1.1.411: merged 3389 unique errors ⊆ (dev 1444 ∪ cmoun 3395)
  → NEW=0, FIXED=159. (uvx pyright without torch resolution discarded:
  19696 unresolved imports.)

## C — GPU 2-rank gate on the merged tree (§9)

- **P1 forced-rescue regression**: 5/5 scenarios `outcome=ok`,
  `max_param_rank_diff=0.0`, `fp32_rescue_failures=0`; `final_failure`
  raises `CMuonSafetyError` with zero commit. → `2rank-regression.json` PASS.
- **P2 checkpoint backcompat**: real `ckpt_103700` (D4 CMuon production
  state): A_load→B_state→C_update_save→D_resume (exact state equality)→
  E_continuity all ok; observations 1201, 166 refs, rescue 818/818/0.
  → `backcompat.json` `D4_CHECKPOINT_BACKCOMPAT=YES`.
- **P3 cleanup perf A/B**: merged tree median **0.6443 s** optimizer phase
  (ref 0.7624 s, −15.5% — no regression), host_syncs/step = 6.8.
  → `perf-cleanup-merged.json`.

## D — G1 ckpt backcompat on merged code (§10)

Read-only copy of `ckpt_105000` (production G1 AdamW state at the current
anchor, 6.0 G, manifest 12/12) into the isolated root; resume path
exercised in Arm A (below). `D4_CHECKPOINT_BACKCOMPAT=YES` for the CMuon
state carried by §C-P2.

## E — Isolated real training (§11–§21)

Production input config `config/train_g1.toml` (run_id g1_256_bs760, the
G1 production config) as overlay base, byte-identical except:
W&B disabled; per-arm `mainset_path` (G1 mixture state at the anchor,
independent per-arm copies, new-machine cold start normalized
active→pending + `schema_version=2`); Arm A cadence 20; Arm B optimizer :=
the **effective frozen CMuon optimizer subtree** of the full candidate
chain (base → s0 → ns4_core → fp32_rescue): `name=hybrid_cmuon_canonical_ns4_fp32_rescue`,
`cmuon_ns.default=4`, `cmuon_ns_steps=5`, `cmuon_momentum_dtype=bfloat16`,
`cmuon_chunk_rescale_sqrt_n=false`, guard (166 refs) + telemetry disabled —
verified by programmatic roundtrip equality against the chain's loaded
effective config (EXACT-MATCH). Data/model/objective/dropout/compile policy
= production exactly (compile stays `max-autotune-no-cudagraphs`, dynamic).
LR continuity: `base_lr=5e-05`, `reference_batch=256`,
`lr_scaling=linear_global_batch` → scaled lr identical to the production
G1 run at this batch.

### Arm A — AdamW (default path), 20u + fresh resume

- `--resume` from the isolated `ckpt_105000` copy → trained 105001→105040
  (overshot the 20u stop for the resume-test; cadence ckpt at 105020),
  loss 0.5436–0.6024, ~14.4–16.6 s/update steady (678 s first update =
  max-autotune compile), nonfinite=0, clip_fraction=0 on all 78 metric
  records.
- Clean SIGTERM stop (clean exit; latest state saved at 105040).
- **Fresh resume from `ckpt_105020`** (new process group): state restored
  (`恢复训练状态` + `optimizer_parameters OK`), training continued from
  105021 (no gap, no re-run of 105020), 19 updates 105021–105039,
  loss 0.5421–0.5879, ~15 s/update, loss continuity across the boundary
  (105020: 0.5713 → 105021: 0.5656). **PASS.**
- Disclosed test artifact: both resume runs overshot to the 105040 cadence
  before the stop and hit `FileExistsError: checkpoint target already
  exists: ckpt_105040…` — the first run's clean-exit save had already
  published that target. This is the no-overwrite protection working as
  designed under an artificial stop/re-run sequence (production resumes
  always start from the newest checkpoint and never re-publish it); it is
  not an integration defect.

### Arm B — AdamW→CMuon migration, 100u + fresh resume

- Same anchor, frozen CMuon optimizer (§E preamble). Migration path
  (resume an AdamW-state checkpoint under the CMuon optimizer) already
  proven by §C-P2 backcompat; exercised live here: the AdamW-state
  `ckpt_105000` resumed cleanly under the CMuon optimizer with fresh
  guard-reference calibration (no import of AdamW moments — bootstrap
  from model parameters, per the frozen design).
- **100/100 updates 105001–105100** exact and contiguous (109 metric
  records: the run overshot to 105109 before the reactive clean stop —
  same stop-timing artifact class as Arm A, disclosed).
- Loss 0.5337–0.5982 (median 0.5658), max consecutive delta 0.052 —
  stable, no spike, no nonfinite (nonfinite_count=0 on all records,
  clip_fraction=0 on all records).
- Steady state ~15.3–16.9 s/update (612 s first update = compile), in
  line with G1 production (15.5 s/u) and the 200u CMuon run (15.4–16.4).
- **CMuon guard telemetry** (per-rank JSON in the training log, obs 10→108):
  `fp32_rescue_failures=0` on every line; early-warmup rescue transient
  (92 fp32 rescues / 92 bf16 safety trips by obs 108, roles
  attention_q 47 / attention_content_gate 38 / attention_k 7) — every
  trip safely rescued to fp32, matching the 200u-run band
  (0.665/obs at D4 exit); `max_delta_rank_spread=0.0` and
  `max_param_rank_diff=0.0` on every line (bit-exact across ranks);
  references stable (ref_min ~2.94e-08, ref_max ~4.46e-06).
- **Optimizer phase**: median 0.5578 s (min 0.5433 s excluding the
  115.5 s compile-carry on update 1) vs the 0.7624 s reference → −26.7%,
  no regression; host syncs at the production cleanup level (6.8/step, §C-P3).
- Checkpoint: `ckpt_105100_raw-105100-update-cadence` COMPLETE +
  12-file manifest at the 100u cadence.
- Clean SIGTERM stop: ranks exited on `KeyboardInterrupt` during the
  in-flight forward (expected clean-shutdown pattern, identical to Arm A
  and to the 200u watchdog stop); no CMuonSafetyError, no OOM, no NCCL
  failure (the NCCL/Watchdog log lines are INFO-level init dumps).
- **Fresh resume from `ckpt_105100`** (+5): new process group,
  `恢复训练状态 … ckpt_105100…` + `optimizer_parameters OK` on both ranks,
  training continued from 105101 (no gap); +5 window 105101–105105 all
  successful, loss 0.5464–0.5638, ~16–18 s/update, continuity across the
  boundary (105100: 0.5647 → 105101: 0.5525). Guard state carried over
  exactly: observations 108 → 109–110, rescues 92 → 93,
  `fp32_rescue_failures=0`, rank spread 0.0. Run overshot to 105111 before
  the reactive stop (same stop-timing artifact, disclosed). Clean
  SIGTERM stop. **PASS.**

**Arm B 100u verdict: PASS** (all §17 gates green for the 105001–105100
window; see `armB-gates-result.json` — the checker's raw substring
count of `rescue_failures` (79) is the JSON *key*
`"fp32_rescue_failures": 0` present on guard log lines; every value is 0).

## F — Performance sanity (§16)

P3 (merged tree, same-pod A/B): median optimizer phase 0.6443 s vs the
0.7624 s reference → −15.5%, no >10% regression. host_syncs/step 6.8
(production cleanup level). Full-step timing on the isolated arms:
~15 s/update at bs760/2-rank (compile amortized), consistent with G1
production steady state (15.5 s/update on salt7).

## G — Isolation & remote-write audit

- salt7 G1 CONTROL: read-only (config/mainset/manifest/selection copied out
  with md5 verification; no write).
- Hub: read-only (ms-hub download only; REMOTE_WRITES=0).
- Publishers: not started. Formal checkpoint/publish paths: untouched.
- W&B: disabled in both arms (`enabled=false`, retry jsonl only).
- Per-arm isolation: independent data-state (mainset) copies,
  independent caches (hardlink-shared shard files, same filesystem, zero
  extra disk), independent `output_model` roots; socket/lock paths are
  schema literals (shared) — arms ran strictly sequentially.

## H — Integration fix list

| Commit | Type | Content |
|---|---|---|
| e6db9c5 | merge | `--no-ff --no-commit` merge of origin/cmoun into dev line (dual parent verified) |
| 023c057 | fix (mode-only) | `git update-index --chmod=+x` on `scripts/analyze_structural_calibration.py` + `scripts/audit_pi_vs_svd.py` (ruff 0.16.1 EXE001; identical pre-existing condition on the frozen tree) |

No optimizer math, no sampler, no objective, no config-schema changes.

## I — Environment notes (disclosed)

- salt10 `validation-selection.json` was missing from the local cohort dir;
  restored byte-identical from the salt7 production copy (md5 5299a402).
- G1 mainset state file on salt7 predates `_QueueStore` `schema_version`
  (format without the field); the isolated per-arm copies carry
  `schema_version=2` (current) — production salt7 files untouched.
- Config loader contract notes: `--config` requires the `.toml` extension;
  `extends` resolves against the config root (base.toml co-located per arm);
  the data service refuses paths that resolve outside `--root` (no
  symlink escapes) and only accepts connections after its warm-up
  (16 ready shards), which is why training launch is gated on warm-up.
- SCNet shared egress proxy showed a ~20-minute transient throttle
  (0.4–1.6 MiB/s/stream with intermittent connection failures, 2026-09-02
  11:25–11:40) while the same proxy served the salt7 G1 data service at
  48–54 MiB/s throughout; recovered to 25–42 MiB/s per shard. Known
  platform behavior (officially acknowledged peak-time fluctuations);
  the data service's keep-progress retry absorbed the dip with no data loss.
