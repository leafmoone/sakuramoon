# SakuraMoon Camera — Corrected MBS 100U Treatment: 0-Update FAIL Evidence

- **Status**: FAILED TREATMENT ATTEMPT (frozen evidence — do not overwrite with a later successful run)
- **Date**: 2026-09-08 (UTC+8)
- **Host**: come7 (crdnotebook-…-come7-21657), 2× Hygon BW DCU
- **Purpose**: record the first real corrected-MBS 100U treatment attempt, which failed before the first successful update due to a torch.compile recompile-limit hard failure exposed by the (correct) mirror fix.

---

## VERDICT

```
FAILED TREATMENT ATTEMPT
successful updates              = 0
source damage                   = NONE
control damage                  = NONE
treatment checkpoint            = NONE

ROOT CAUSE
variable mirror physical rows
→ Python tuple-length specialization
→ Dynamo recompiles
→ recompile_limit 8
→ FailOnRecompileLimitHit

ROOT CAUSE CONFIDENCE            = HIGH

OPTION 1                         = ACCEPTED WITH MODIFICATION
global unconditional 64          = NO
config-governed default8/v2=64   = YES

OPTION 2 padding                 = HOLD
OPTION 3 structural refactor     = HOLD
```

---

## IDENTITY

| field | value |
|---|---|
| reviewed code SHA | `73d419c6cede6d6bf76ee86b706e4dd881bc9a6d` |
| evidence branch | `camera-v2-mbs-corrected-100u-treatment-evidence` |
| worktree | `/sakuramoon-runtime/sakuramoon-camera-mbs-corrected-100u` |
| config | `config/train_g1_camera_v2_p25_mirror_v2_canary.toml` |
| run_id | `g1_camera_v2_p25_mirror_v2` |
| source checkpoint | `output_model/g1_camera_v2_p25/ckpt_118100_raw-118100-update-cadence` (U118100, immutable) |
| control checkpoint | `output_model/g1_camera_v2_p25_mirror/ckpt_118200_raw-118200-update-cadence` (U118200 camera-only, immutable) |
| treatment output root | `output_model/g1_camera_v2_p25_mirror_v2` (created empty; no checkpoint written) |
| world size | 2 |
| optimizer | `hybrid_cmuon_canonical_ns4_fp32_rescue` |
| stage global batch | 800 (local_batch 20 × accum 20 × world 2) |
| telemetry effective_batch | rank-0 cohort = 400 (NOT the global batch) |
| stop cap | `canary_stop_successful_update = 118200` |
| MAIN_PROCESS_PORT | 29500 |
| LOG_ROOT | `/sakuramoon-runtime/logs/mbs-corrected-treatment-118100-118200` |
| RUN_ROOT | `/run/sakuramoon-mbs-corrected-treatment` |
| PUBLISH_STATE_ROOT | `/sakuramoon-runtime/.sm-train-state-publisher-mbs-corrected-118100-118200` |
| REPO_PATH | `experiments/camera-v2-mbs-corrected-treatment-118100-118200` (never targeted production `s0`) |
| INTERVAL_SECONDS | 86400 |

## SUPERVISOR (machine-side, mandatory)

| field | value |
|---|---|
| path | `/sakuramoon-runtime/mbs-corrected-treatment-supervisor.py` (UNTRACKED, outside git worktree) |
| sha256 | `1c83455955792913fe80a6762486a30b7aac81d1b72a0c0d25a03dde8abd8a72` |
| PID | 8682 (nohup + setsid, independent of DSH/session) |
| result path | `/sakuramoon-runtime/mbs-corrected-treatment-supervisor-result.json` |
| result sha256 | `fe8cd9bd0dd5cf4302c60a4d3f4bd3c61a115617d81a95c18e0e1535031ee80d` |
| final status | `FAIL` |
| failure reason | `fatal pattern in train.log: 'Traceback'` |
| last_successful_update | 118100 (no new successful update) |
| early exposure gate | `not_reached` |

### What this run VALIDATED about the supervisor
- **Independent survival**: launched first (WAITING_FOR_START), stayed alive through the DSH session, detected the failure on its own. ✅
- **Fatal-log gate works**: correctly detected the `Traceback` fatal pattern in train.log at the launch byte offset and transitioned RUNNING→FAIL. ✅
- **Fail-closed / no auto-restart**: wrote the FAIL result atomically and did NOT restart training. ✅
- **Startup gate**: observed the training process appear (pid 9300) within the 10-min startup deadline and set T_stack. ✅

### SUPERVISOR BUG FOUND (must fix next round)
The supervisor's `stack_stop()` invocation of `scripts/training_stack.sh stop` **omitted `VENV_ROOT`** from the environment it passes. `training_stack.sh`'s `load_config_contract` then resolved `MANAGEMENT_PYTHON` to the default `${PROJECT_ROOT}/.venv/bin/python`, which does not exist on this host (the real env is the DTK venv at `/sakuramoon-runtime/sakuramoon-dtk-venv`), so the stop command FAST-FAILED:

```
[2026-09-08T13:25:11] INVOKING stack stop: bash …/scripts/training_stack.sh stop
[2026-09-08T13:25:11] stack stop rc=1 … FAST-FAIL: Python is not executable:
  /sakuramoon-runtime/sakuramoon-camera-mbs-corrected-100u/.venv/bin/python
```

Consequence: the training processes had already self-terminated (rank1 crash → elastic launcher SIGINT to rank0), but the **data-service (pid 8883) and publisher (pid 9150) were orphaned** and had to be stopped manually with a correct env (including `VENV_ROOT`). No state damage resulted.

Required next-round fix:
1. Supervisor must include `VENV_ROOT` in the environment it stores and passes to `training_stack.sh stop` (alongside all other operational vars).
2. Add a **pre-launch self-test** of the stop environment (e.g. `supervisor --self-test-stop-env`, a side-effect-free check) that confirms every variable the stop command requires is present and the resolved python is executable — so a non-executable stop path is caught before training, not at FAIL time.

---

## TIMELINE (UTC+8)

| time | event |
|---|---|
| 13:15:05 | supervisor launched (nohup/setsid), state=WAITING_FOR_START, pid 8682 |
| 13:17:21 | `training_stack.sh start` (attempt 1): data-service started (pid 8883, fresh deterministic cycle-0 queue), publisher FAST-FAILED — treatment output root did not exist (known v1-era preflight requirement) |
| 13:18:4x | created empty `output_model/g1_camera_v2_p25_mirror_v2` |
| 13:18:48 | `start` (attempt 2): data reused (8883), publisher started (9150), train started (9300, ranks 2/2); supervisor → RUNNING, T_stack set, train.log offset=0 |
| 13:22 | trainer entered loop: `开始训练: update 118101 -> 168000`; wandb run `g1_camera_v2_p25_mirror_v2` |
| 13:22–13:25 | torch.compile (inductor, max-autotune) first-forward autotune running |
| 13:25:00 | **rank1**: `torch._dynamo hit config.recompile_limit (8)` → `FailOnRecompileLimitHit` (HARD failure) |
| 13:25:11 | supervisor detected `Traceback` → wrote FAIL result → invoked stop (stop FAST-FAILED on missing VENV_ROOT) |
| 13:25:21 | elastic launcher: rank1 exitcode 1 → SIGINT to rank0 → all train processes dead |
| 13:3x | manual `training_stack.sh stop` with correct env cleaned orphaned data-service + publisher |

## ROOT CAUSE (confidence: HIGH)

The worker-clone fix (branch `camera-v2-mbs-worker-clone-fix-review`, 1-line `mirror_policy=self.mirror_policy`) made the mirror **actually fire**. With the mirror active, each batch appends a **variable number of mirror extra views** (0..20, one per eligible logical sample). The PackedDiT varlen attention receives `AcceptedCuSeqlens.sequence_lengths` as a **Python tuple**, and `batch_size = len(sequence_lengths)`. Even though `compile_packed_dit_blocks()` sets `dynamo_config.fail_on_recompile_limit_hit = True` **and** requires `dynamic=True`, Dynamo still builds a guard on the **Python container length** (`len(boundaries.sequence_lengths)`), which is a Python-level value not made dynamic by `dynamic=True`. As the physical row count varies batch-to-batch, Dynamo recompiles the `forward` frame for each new distinct length; after 8 distinct lengths it hits `config.recompile_limit` (default 8) and, because `fail_on_recompile_limit_hit = True`, this is a HARD failure.

Exact Dynamo diagnostic (train.log, lines 402–406):
```
[rank1]:W0908 13:25:00.509000 9463 torch/_dynamo/convert_frame.py:1358] [0/8] torch._dynamo hit config.recompile_limit (8)
[rank1]:    function: 'forward' (…/src/sakuramoon/model/block.py:216)
[rank1]:    last reason: 0/6: len(boundaries.sequence_lengths) == 22   # self.attention(  # block.py:250 in forward
[rank1]:    To log all recompilation reasons, use TORCH_LOGS="recompiles".
```

Exception chain:
```
torch._dynamo.exc.FailOnRecompileLimitHit: recompile_limit reached, because
  fail_on_recompile_limit_hit = True this is a HARD failure
→ sakuramoon.train.production.ProductionTrainingError: accepted production training failed
```

Why the control (v1) never hit this: v1's mirror was broken (eligible/selected/applied ≡ 0), so the physical row count was constant and `sequence_lengths` never changed length → no recompiles → v1 completed its 100U.

Why the prior round's smoke missed this: the real-data exposure smoke ran only ~160 logical samples (1–2 updates), too short to observe 8 distinct `sequence_lengths` lengths.

### Bounded-distinct-length argument (for the fix)
- local logical per rank = 20
- mirror applied per batch = 0..20
- physical sequences = 20..40 → **distinct lengths ≤ 21**
- so a per-code `recompile_limit = 64` gives ~3× headroom for the *pure physical batch count* dimension.
- Caveat (why a saturation smoke is still required): `sequence_lengths` is the whole Python tuple, not only its length; Dynamo may specialize on additional properties. A forward-only smoke over a sustained run of 32–64 real persistent-worker batches with `TORCH_LOGS=recompiles` must confirm: recompile-limit failure = NO, accumulated-limit failure = NO, eager fallback = NO. `accumulated_recompile_limit` (default 256) must NOT be raised in the same change.

## STATE INTEGRITY (verified)

| artifact | before sha256 | after sha256 | immutable |
|---|---|---|---|
| source U118100 `manifest.json` | `821a8c12546a094865972fef0c23152e9754e01f82eed8353f7248b92076b892` | `821a8c12546a094865972fef0c23152e9754e01f82eed8353f7248b92076b892` | YES |
| control U118200 `manifest.json` | `f014827fb441aca8780a5bb710eeb2631a40aac112ed3aa6b07b6e3b65d657f7` | `f014827fb441aca8780a5bb710eeb2631a40aac112ed3aa6b07b6e3b65d657f7` | YES |
| source U118100 full tree (pre) | `f022ff07d29a6865a96bfa0fb1515fd9cb51878bc5fcde68ebd4b952fb8debbf` | (re-hash at final) | — |
| control U118200 full tree (pre) | `f01ad3ff7f0f30460650408ed4d0a79fd39d20738e2ee6a7197a8bdee4b977f4` | (re-hash at final) | — |
| treatment output root | empty (created) | empty, **no checkpoint** | — |
| treatment `metrics.jsonl` | — | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` (0 bytes) | 0 records |
| data queue state (v1 leftover, reset for matched data) | backup sha `a33e2557417d791f5729d6b5eacfd96efed4974998844eaf5689986f50c66cbf` | reset to fresh deterministic cycle-0 (backup retained) | — |

No successful update occurred, so no treatment checkpoint was written and the governed stop cap was never exercised. Data cache accumulated ~112 GiB of (re)downloaded shards during the attempt — reusable by the next round.

## EVIDENCE HASHES

| artifact | sha256 |
|---|---|
| reviewed code SHA | `73d419c6cede6d6bf76ee86b706e4dd881bc9a6d` |
| supervisor script | `1c83455955792913fe80a6762486a30b7aac81d1b72a0c0d25a03dde8abd8a72` |
| supervisor result | `fe8cd9bd0dd5cf4302c60a4d3f4bd3c61a115617d81a95c18e0e1535031ee80d` |
| supervisor log | `c944076438bbd39115e656368c8556c884b08b986d5cf3cf9f7a2553a551c7fe` |
| train.log | `a0a9d825b60f17e2525524309888c52e649f8d1d086b9d01e2d0e2b5a9d5069c` |
| treatment metrics.jsonl (0 bytes) | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| source manifest | `821a8c12546a094865972fef0c23152e9754e01f82eed8353f7248b92076b892` |
| control manifest | `f014827fb441aca8780a5bb710eeb2631a40aac112ed3aa6b07b6e3b65d657f7` |
| treatment manifest | N/A (no checkpoint written) |

## ACCEPTED FIX DIRECTION (next round — NOT implemented in this commit)

**Option 1, accepted WITH MODIFICATION.** Do NOT set a global unconditional `dynamo_config.recompile_limit = 64` (that would lower the production compiler safety ceiling for all runs). Instead add a **governed kernel config**:

```toml
[kernels]
torch_compile_recompile_limit = 8   # default: all existing runs unchanged
```

Only the corrected MBS v2 canary opts in:

```toml
[kernels]
torch_compile_recompile_limit = 64
```

and pass it through:

```python
compile_packed_dit_blocks(
    ...,
    recompile_limit=config.kernels.torch_compile_recompile_limit,
)
```

which internally sets (fail-closed):

```python
dynamo_config.recompile_limit = recompile_limit
dynamo_config.fail_on_recompile_limit_hit = True
```

So the MBS canary explicitly opts in to 64 rather than the whole production compiler safety ceiling being relaxed by one experiment. `KernelsConfig` is already the configuration boundary for compile parameters, so this field belongs there.

`accumulated_recompile_limit` (default 256) is left unchanged in this fix; if the per-code 64 passes but accumulated 256 is hit, that is a separate decision.

## NEXT (ordered)

1. **(B — this commit)** freeze + push this 0-update FAIL evidence. ✅
2. external review of this evidence (exact SHA).
3. **(A)** open a NEW fix branch implementing the config-governed `recompile_limit` (default 8 / v2 canary 64).
4. **long forward-only compile saturation smoke**: real persistent-worker batches, Qwen/VAE/compiled DiT forward-only, sustained 32–64 distinct mirror-count batches, `TORCH_LOGS=recompiles`; require recompile-limit failure = NO, accumulated-limit failure = NO, eager fallback = NO.
5. **supervisor stop-env fix**: include `VENV_ROOT` in the stored/passed stop environment + pre-launch `--self-test-stop-env` (side-effect-free) confirming the stop command's required variables are present.
6. external code review of the fix.
7. **only then** rerun U118100 → U118200.

## CODE IMMUTABILITY

`git diff 73d419c6cede6d6bf76ee86b706e4dd881bc9a6d..HEAD -- src config scripts tests` = **EMPTY** (this commit adds report files only; no code/config/test change).
