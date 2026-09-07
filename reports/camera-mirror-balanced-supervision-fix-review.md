# SakuraMoon Camera — MBS Canary Readiness Fix Review

- repository: `leafmoone/sakuramoon`
- reviewed base (exact SHA): `f814e1cd285d2f4bc86775af525ade2a699f52a5`
- branch: `camera-v2-mirror-balanced-supervision-fix-review`
- worktree: `/sakuramoon-runtime/sakuramoon-camera-mirror-balance-fix-review`
- MIRROR_FIX_TOOLING_HEAD: `98d9a5839f2fbc12b23a9da1a8e6cf9d6f09186a`
- MIRROR_FIX_EVIDENCE_HEAD: (this commit — recorded in the round transcript / FINAL COPY)
- round mode: **NO TRAINING / NO BACKWARD / NO OPTIMIZER STEP / HARD STOP**

This report is the fix-review deliverable required by the external review of
the MBS (vertical mirror-balanced camera supervision) design round. It
records the four review fixes (A–D), the HCU forward-only smoke evidence,
the corrected compute estimate, the RETENTION_25 erratum, the disabled-path
identity wording, and the read-only resume-budget audit of the intended
source checkpoint. The original design report
(`reports/camera-mirror-balanced-supervision-design.{md,json}`) is immutable
and was NOT modified by this round.

---

## 1. External review result and scope of this round

Accepted by the external reviewer (carried over, unchanged):
MIRROR GEOMETRY, LOGICAL PAIR WEIGHTING, SAME TIMESTEP / SAME NOISE, QWEN
LOGICAL-ROW DESIGN, EXISTING COORDINATE MATH, VERTICAL-ONLY V1.

Blocked (fixed in this round):

| ID | Blocker | Fix location |
|----|---------|--------------|
| A | CUDA condition-routing device bug (CPU index tensor reaches the device-local condition-token path) | `src/sakuramoon/train/runtime.py` + `tests/unit/train/test_mirror_pair_training.py` |
| B | Mirror counts not persisted to durable telemetry | `src/sakuramoon/telemetry/metrics.py`, `src/sakuramoon/telemetry/observer.py` + 5 test files (incl. new `tests/unit/test_telemetry_camera_mirror.py`) |
| C | Degrade counters hide failures (decision history erased) | `src/sakuramoon/data/pipeline.py` + `tests/unit/data/test_camera_mirror_balance.py` |
| D | Design report stat erratum (RETENTION_25) | Text-only erratum in THIS report (section 9). The reviewed design report is not modified. |

## 2. Fix A — device routing (the canary-critical bug)

Root cause: in the mirror branch of `SingleGpuBatchRuntime.prepare()`, the
remapped active-condition indices were re-derived from
`batch.active_condition_sample_indices`, which is still a **CPU** tensor,
while earlier in `prepare()` the correct **device-local** copy had already
been H2D-transferred into the local variable
`active_condition_sample_indices`. The downstream condition-token path
requires `active_condition_sample_indices.device == qwen_states.device`, so
every real mirror CUDA/HCU batch with at least one active-condition row
would have failed.

Fix (production diff, `src/sakuramoon/train/runtime.py`):

1. `_physical_active_condition_indices(...)` now receives the ALREADY
   DEVICE-LOCAL tensor (no needless copy added). The helper preserves the
   input device; output is 1-D `torch.long` on `qwen_states.device`; an
   empty active set also stays on the device.
2. A fail-closed guard is added before `TrainableCompositeInputs` is
   constructed: if `mirror_layout is not None` and the remapped indices are
   not a 1-D `torch.long` tensor on `qwen_states.device`, a `ValueError` is
   raised immediately instead of surfacing later as an opaque downstream
   device error.

Regression tests (deterministic, 0 skip / 0 xfail, added to
`tests/unit/train/test_mirror_pair_training.py`):

- CPU input → CPU output (device preservation, no copy).
- Mirror logical routing: logical active rows `{0, 2}` with mirror flags
  `{True, False, True}` map to physical rows `{0, 1, 3, 4}` with the device
  unchanged.
- The fail-closed guard trips on a device mismatch (and a correct device
  passes).

Live confirmation: the HCU smoke (section 4) ran a real cuda:0 mirror batch
with one active-condition row through the exact production path; the guard
did not trip and the forward completed (evidence
`smoke-evidence.json`, check `fail_closed_guard_device_local_path`).

## 3. Fix B — durable mirror telemetry (schema v12)

Implementation choice: extend the existing strict `TrainingMetric` schema
(the same fixed-field durable JSONL/W&B path already used by the camera and
spatial tables) — no ungoverned side channel was invented.

- `TRAINING_METRIC_SCHEMA_VERSION`: **11 → 12** (deliberate bump; strict
  JSON mapping and W&B mapping updated for every new field; tests updated).
- 16 new fixed fields on `TrainingMetric` (all `camera_mirror_*`, no
  free-form labels):

  ```
  camera_mirror_logical_samples, camera_mirror_vertical_applied,
  camera_mirror_eligible, camera_mirror_selected, camera_mirror_applied,
  camera_mirror_severity_lt2, camera_mirror_severity_2to4,
  camera_mirror_severity_ge4, camera_mirror_original_start,
  camera_mirror_original_center, camera_mirror_original_end,
  camera_mirror_mirror_start, camera_mirror_mirror_center,
  camera_mirror_mirror_end, camera_mirror_extra_views,
  camera_mirror_physical_views
  ```

- Feature-absent default (strict spec default, filled in `__post_init__`):
  `logical_samples == physical_views == effective_batch`, every activity
  counter 0.
- Update-level aggregation in `observer.py` (typed helper, across
  microbatches) enforces, per update:
  - `logical_samples == effective_batch`
  - `applied <= selected <= eligible <= vertical_applied <= logical_samples`
  - `severity_lt2 + severity_2to4 + severity_ge4 == vertical_applied`
  - `original_start + original_center + original_end == applied`
  - `mirror_start + mirror_center + mirror_end == applied`
  - `extra_views == applied`
  - `physical_views == logical_samples + extra_views`
  A dropped counter is a schema violation, not a silent zero.
- Durable in local `metrics.jsonl` and in the W&B mapping (both pinned by
  tests).

## 4. HCU real forward-only mirror smoke (section 5 of the review spec)

**Executed on come3** (Hygon DCU, device name `BW`), one of two available
HCU devices. This is a real runtime smoke, not a CPU simulation.

- script (untracked, outside the worktree):
  `/sakuramoon-runtime/mbs-smoke/hcu_mbs_forward_smoke.py`
- script sha256: `e5b81c11b9e74e764e9f395e65769e623d90b0ff1beac77daca77280cad77fd6`
- evidence: `/sakuramoon-runtime/mbs-smoke/smoke-evidence.json`
  (24/24 checks PASS, verdict PASS, process exit 0)
- models: real frozen assets, read-only —
  `output_model/g1_camera_v2_p25/ckpt_118100_raw-118100-update-cadence`
  (U118100 DiT composite, `TrainableComposite`, 289 parameter tensors),
  `model/qwen_3.5_2B` (QwenRuntime encoder, sdpa, 320 tensors),
  `model/vae` (Mage-VAE, 644 tensors). torch 2.9.0 (DTK venv).
- runtime hyperparameters exactly as the frozen production resolved config:
  `p_mean=-0.8, p_std=0.8, noise_scale=1.0, t_eps=0.05,
  noise_observation_boundary=0.95, growth_alpha=1.0`, `no_grad=True`,
  torch_compile off.
- scope: 2 logical samples built by the REAL production pipeline + collate
  (stage G1, square bucket 256, shifted-bucket vocabulary, HDM shifted-square
  camera viewport p=1.0 zoom 1.10–1.50, mirror policy
  `vertical_mirror_pair_v1` min_latent_shift=2.0 pair_probability=1.0):
  - row 0 (sample_id 1): applied vertical camera, latent center shift
    5.1875 (>= 2.0, eligible), mirror pair SELECTED + APPLIED (payload
    present, exact vertical mirror crop of the same full canvas), ACTIVE
    artist condition (dense 98);
  - row 1 (sample_id 2): applied vertical camera, latent center shift
    1.75 (< 2.0, not eligible, no mirror payload), NULL condition (dense 98).

Measured facts from the single `SingleGpuBatchRuntime.measure(batch)` call
(no backward, no optimizer, no scheduler, no checkpoint write):

| quantity | value |
|---|---|
| Qwen forward calls | **1** (wrapper counter; input shape `[2, 98]` = logical batch) |
| VAE encode calls | **1** (input shape `[3, 3, 256, 256]` = physical views) |
| physical views | 3 = logical 2 + mirror applied 1 |
| pair timestep | bit-equal across the pair (`torch.equal`, t=0.46219295263290405); row 1 t=0.5086652636528015 distinct |
| pair noise | bit-equal across the pair (`torch.equal`, bf16); max abs cross-row diff 6.3125 |
| qwen_states device vs active-condition-index device | equal (`cuda:0`); fail-closed guard did not trip |
| per-sample loss | logical domain, shape `(2,)`: pair 0.08668571710586548, ordinary 0.09503582119941711 — both finite and > 0 |
| high/low noise sums | finite (high_sum 0.1817215383052826 = pair+ordinary, high_count 2) |
| observed timesteps | logical domain, shape `(2,)` |
| image tokens | 768 = 3 physical views x 256 tokens |
| camera zoom bands | per-logical, len 2 |
| mirror-view latent | differs from the original-view latent (max abs diff 1.5703125 in bf16) — the pair is a real geometric mirror, not a duplicate tensor |
| camera_mirror table | logical 2 / vertical_applied 2 / eligible 1 / selected 1 / applied 1 / extra 1 / physical 3; severity lt2=1, ge4=1; original side start=1, mirror side end=1; all conservation identities hold |
| parameter/buffer mutation | NONE: sha256 + data_ptr + grad-None identical before/after for all 289 composite + 320 Qwen + 644 VAE tensors |
| walltime | measure() 14.72 s; total script 65.3 s; peak allocation 6.99 GiB |

**HCU_SMOKE = PASS** (not NOT_RUN; not faked — every number above is taken
from the recorded evidence file).

Qwen claim (minimum accepted wording, now backed by an actual invocation
counter): "Qwen is executed once over the logical batch before physical-row
expansion; source inspection + runtime smoke confirms no second mirror Qwen
call." The smoke measured exactly 1 encoder call for the 2-logical /
3-physical batch.

## 5. Fix C — degrade semantics

`WebDatasetPipeline._process`: after eligibility + selection, if mirror
payload construction raises, the pipeline now KEEPS `mirror_eligible=True`
and `mirror_selected=True`, sets `mirror=None`, and lets the original sample
continue (the `mirror_degraded` trace is preserved). It no longer resets both
flags to False. Consequence: `selected - applied` is the exact
deterministic degraded-pair count in the fixed `camera_mirror` counters, so
a canary cannot silently lose the decision history.

Test (added to `tests/unit/data/test_camera_mirror_balance.py`): a sample
with eligible=1, selected=1, payload=None produces eligible 1 / selected 1 /
applied 0 — NOT 0/0/0.

## 6. Fix D — RETENTION_25 erratum (text only, this report)

WRONG historical wording (in the reviewed design report, left unchanged):

> RETENTION_25 = 0.908927... interpreted as canvas/crop retention.

CORRECT meaning:

> RETENTION_25 is **CAUSAL GAP RETENTION** after `VISUAL_GEO_BALANCED_25`
> selection: `|G_VISUAL_GEO_BALANCED_25| / |G_ALL|`, i.e. the fraction of
> the top-bottom causal gap retained by the visual-content balancing.

It is NOT image pixel/canvas retention. Do not claim "91% canvas
retained". The actual camera crop retention is separately defined by the
camera geometry as `1 / zoom^2` and is never mixed with causal-gap
retention.

## 7. Compute estimate correction (section 11 of the review spec)

The previous design report mixed the 2048 camera audit units + 512 ordinary
controls as if that 80/20 audit construction were the production P25
population. It is not. Recomputed from real P25 evidence:

- known historical P25 aggregate: camera applied overall ≈ **19.777%**
- frozen camera audit (frozen P25, 2048 camera units): severe vertical
  eligible conditional on applied camera ≈ 630/2048 ≈ **30.76%**
- no better full-run joint orientation x latent-shift distribution exists
  locally, so the joint figure is an **ESTIMATE**:

  expected extra physical-view fraction over all training logical samples
  ≈ 0.19777 x 0.3076 ≈ **0.0608**

  - whole-training estimated physical inflation ≈ **1.061x** (NOT 1.246x)
  - camera-conditional physical inflation ≈ **1.308x**
  - measured walltime multiplier = **UNKNOWN** until the canary runs
    (no automatic conversion; no new long training/forward benchmark was
    run this round)

## 8. Disabled-path identity wording (section 13)

Because this design adds fields to dataclasses (RngIdentity,
TrainingBatch, ...), literal Python object byte identity is NOT claimed.

What holds with the mirror config absent: caption RNG stream unchanged, crop
RNG stream unchanged, camera policy RNG unchanged, camera offset RNG
unchanged, emitted ordinary/camera image tensor unchanged, coordinate tensor
unchanged, Qwen/VAE/DiT numerical training path unchanged, optimizer/update
semantics unchanged.

Allowed wording used in this round: **NUMERICAL / DATA-PATH IDENTITY**.

## 9. Resume budget read-only audit (section 14)

Source checkpoint (read-only inspected, NOT modified):

- path: `/sakuramoon-runtime/output_model/g1_camera_v2_p25/ckpt_118100_raw-118100-update-cadence`
- marker: `COMPLETE`
- manifest sha256: `821a8c12546a094865972fef0c23152e9754e01f82eed8353f7248b92076b892` (schema_version 4, kind `raw`, identity `raw-118100-update-cadence` / update 118100)
- `trainer_state.json`: `successful_updates = 118100`;
  `stage_budget = {start_successful_update: 58000, terminal_successful_update: 168000}`;
  `checkpoint_cadence.every_successful_updates = 100`
- resolved config `[stage]`: name G1, `planned_updates = 168000`
  (inherited/resolved)

Remaining headroom: 168000 - 118100 = **49,900 updates**.

**RESUME_BUDGET_STATUS = READY** (the source stage budget is not exhausted;
a canary terminal inside the inherited stage budget is possible, but per
the review spec NO canary terminal update is invented in this round).

## 10. Scope confirmations (sections 15–16)

- v1 remains **vertical-only**; horizontal is NOT generalized in this fix
  (current causal evidence is vertical-specific; the first intervention
  stays minimal).
- Original design report `reports/camera-mirror-balanced-supervision-design.{md,json}`
  NOT modified; f814 history NOT rewritten (this branch adds exactly two new
  commits on top of f814e1cd, no amend/squash/reset).
- No RoPE, coordinate-math, DiT architecture, Qwen architecture, CMuon, or
  scheduler/LR changes. No checkpoint write, no training, no resume, no
  deploy, no merge, no PR, no force push.

## 11. Validation record

### New / relevant suites (deterministic, 0 skip / 0 xfail)

Covering: device-preserving active-condition remap, mirror active-condition
device equality, telemetry fixed fields, telemetry JSON mapping, telemetry
W&B mapping, telemetry conservation, degraded selected/applied semantics,
disabled numerical-path identity, logical weighting, same t/noise, mirror
geometry, config schema.

- relevant fixed file set (7 files) on the fix-review tree: **120 passed / 0
  failed / 0 skipped / 0 xfailed** (recorded pre-final-gate; the full-suite
  gate below re-runs them in the same environment).

### Full `tests/unit` gate (exact counts from this round)

Command (identical on both trees; two deepghs script tests excluded — see
environment note):

```
python -m pytest tests/unit -q --no-header -p no:cacheprovider \
  --ignore=tests/unit/data/test_deepghs_quality_pipeline.py \
  --ignore=tests/unit/data/test_run_deepghs_quality_pipeline.py
```

- fix-review tree (`camera-v2-mirror-balanced-supervision-fix-review`
  working tree, base f814e1cd + this round's uncommitted fixes):
  ****937 passed / 4 failed / 0 skipped / 0 xfailed** (253.46 s)**
- pristine base tree (`camera-v2-mirror-balanced-supervision-design`
  worktree @ f814e1cd, clean): ****905 passed / 4 failed / 0 skipped / 0 xfailed** (253.53 s)**
- baseline failures comparison: **the 4 failures are the SAME 4 pre-existing cmuon_fp32_forensic environment failures on both trees (test_bcd_hard_fail_captures_fp32_reason[below_floor-below_floor], test_e_writer_failure_still_raises_original_error, test_f_serialization_failure_still_raises_and_leaves_no_capsule, test_m2_mirror_failure_keeps_local_capsule_and_original_error); fix-review additionally passes +32 tests (this round's new deterministic tests); 0 new failures, 0 new skips, 0 new xfails**

Environment note (affects both trees identically, recorded honestly): the
DTK venv on come3 lacks `onnxruntime`, so the two deepghs script tests
(`tests/unit/data/test_deepghs_quality_pipeline.py`,
`tests/unit/data/test_run_deepghs_quality_pipeline.py`) fail at COLLECTION
in both trees; both are outside this fix's diff scope and are excluded by
the identical `--ignore` flag on both runs. The pristine-base comparison
below is therefore exact over the identical collection set.

### Static gates

- `py_compile` on all 11 modified/new python files: **PASS**
- `ruff check` on the changed scope: **All checks passed** (0 new findings)
- `ruff check src tests` (full scope): exactly 1 pre-existing baseline
  finding (EXE001, `tests/gpu/optim/cmuon_capsule_teardown.py`) — identical
  on the pristine base tree; 0 introduced by this round
- `git diff --check f814e1cd`: **clean**
- security scan (secret patterns over the diff scope): **0 hits**
- staged content: only the 11 whitelisted source/test files (commit 1) and
  the 2 report files (commit 2); no checkpoints, datasets, images, *.pt,
  safetensors, /tmp paths, or logs

## 12. Round verdict

- device routing fixed + regression-tested + live-HCU-confirmed
- HCU real mirror forward smoke: **PASS** (24/24 checks, real U118100 DiT +
  Qwen3.5-2B + Mage-VAE on a Hygon DCU)
- mirror telemetry durable JSONL/W&B path (schema v12): implemented + tested
- degraded decision semantics: implemented + tested (selected-applied
  accounting)
- logical loss weighting, same t / same noise: re-verified (CPU tests +
  live bit-equality in the smoke)
- source checkpoint resume budget: **READY** (49,900 updates remaining)
- no new test regression (pristine-base comparison recorded in section 11)
- security clean

**CANARY_READY = TRUE** — this does NOT authorize training. The canary
launch remains gated on external review of the pushed branch
(spec sections 23/24).
