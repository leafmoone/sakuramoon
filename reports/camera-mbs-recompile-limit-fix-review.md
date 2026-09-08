# SakuraMoon Camera — MBS Recompile-Limit Fix: Config-Governed Ceiling + Saturation Evidence

- **Status**: FIX IMPLEMENTED + SATURATION VERIFIED (review branch — no training authorized)
- **Date**: 2026-09-08 (UTC+8)
- **Host**: come7 (crdnotebook-…-come7-21657), 2× Hygon BW DCU
- **Purpose**: implement and evidence the accepted Phase-B fix direction — a **config-governed** `torch.compile` recompile ceiling (default 8, v2 canary 64) that survives the corrected mirror's variable packed physical row count — plus supervisor stop-path hardening (VENV_ROOT). NO training, NO backward, NO optimizer step in this round.

---

## VERDICT

```
RECOMPILE LIMIT                     = config-governed (KernelsConfig)
default (all existing runs)         = 8 (UNCHANGED)
v2 canary override                  = 64 (only non-identity behavioral diff)
fail_on_recompile_limit_hit         = True (KEPT, fail-closed)
env override / magic constants      = NO (production passes the config value)
accumulated_recompile_limit         = UNCHANGED (default 256, not touched)

SATURATION SMOKE (forward-only)     = 64+ real persistent-worker batches
distinct physical sequence counts   = SEE § SATURATION (hard gate >= 9)
recompile-limit failure             = NO
accumulated-limit failure           = NO
eager fallback                      = NO
traceback / nonfinite loss          = NO
parameter mutation                  = NONE (bit-identical checksums)

SUPERVISOR STOP PATH                = VENV_ROOT included (15-key stop env)
--self-test-stop-env                = side-effect-free, live PASS + planted-FAIL verified

CANARY_RERUN_READY                  = SEE FINAL VERDICT
training authorization THIS ROUND   = NONE (rerun requires a later GO)
```

---

## IDENTITY

| field | value |
|---|---|
| base (frozen evidence) SHA | `e6d7b10310c3e32408d312add788b44545da0c71` |
| evidence branch | `camera-v2-mbs-recompile-limit-fix-review` |
| worktree | `/sakuramoon-runtime/sakuramoon-camera-mbs-recompile-limit-fix` |
| prior evidence report | `reports/camera-mbs-corrected-100u-treatment.md` (FROZEN, not edited) |
| config (treatment) | `config/train_g1_camera_v2_p25_mirror_v2_canary.toml` |
| run_id (treatment, later) | `g1_camera_v2_p25_mirror_v2` |
| source checkpoint | `output_model/g1_camera_v2_p25/ckpt_118100_raw-118100-update-cadence` (U118100, immutable) |
| control checkpoint | `output_model/g1_camera_v2_p25_mirror/ckpt_118200_raw-118200-update-cadence` (U118200 camera-only, immutable) |
| treatment output root | `output_model/g1_camera_v2_p25_mirror_v2` (stayed EMPTY this round) |
| optimizer | `hybrid_cmuon_canonical_ns4_fp32_rescue` (loaded weights read-only; no optimizer object constructed) |

## CODE CHANGES (exact scope)

### 1. `src/sakuramoon/config/schema.py` — governed field

```python
# KernelsConfig, after torch_compile_dynamic:
# torch.compile recompile ceiling for the packed DiT blocks (fail-closed:
# production passes this value verbatim; see compile_packed_dit_blocks).
# Default 8 keeps every existing run byte-identical; the corrected MBS
# v2 canary opts into 64 (bounded-distinct physical row count <= 21).
torch_compile_recompile_limit: Annotated[int, Field(ge=1, le=256)] = 8
```

Strict-int semantics come from `StrictModel` (`ConfigDict(extra="forbid", strict=True, ...)`): `True/False`, floats, strings, `None` are all rejected at parse time (12 rejection cases pinned in `tests/unit/config/test_kernels_recompile_limit.py`); bounds 1 and 256 accepted.

### 2. `src/sakuramoon/train/runtime.py` — fail-closed passthrough

```python
def compile_packed_dit_blocks(
    composite, *, backend, mode, dynamic, recompile_limit,
) -> list:
    ...
    if not dynamic:
        raise ValueError(...)
    if type(recompile_limit) is not int or recompile_limit <= 0:
        raise ValueError("recompile_limit must be a positive integer")
    ...
    dynamo_config.recompile_limit = recompile_limit
    dynamo_config.fail_on_recompile_limit_hit = True   # KEPT
```

The kwarg is REQUIRED (no default) so no call site can silently inherit a hidden constant; `type(...) is int` rejects `bool` (a subclass of int) even though the config layer already blocks it — defense in depth.

### 3. `src/sakuramoon/train/production.py` — the single production binding

```python
compile_packed_dit_blocks(
    composite,
    backend=config.kernels.torch_compile_backend,
    mode=config.kernels.torch_compile_mode,
    dynamic=config.kernels.torch_compile_dynamic,
    recompile_limit=config.kernels.torch_compile_recompile_limit,  # exact config value
)
```

No env override, no magic number. A source-pin test (`tests/unit/train/test_production_compile_binding.py`) asserts this exact expression exists in `production.py` and that no `recompile_limit=8/64/...` constant appears at the call site; a functional test proves the config value (37) reaches `dynamo_config.recompile_limit` through the real kwarg expressions.

### 4. `config/train_g1_camera_v2_p25_mirror_v2_canary.toml` — the only behavioral diff

```toml
[kernels]
torch_compile_recompile_limit = 64
```

Everything else the v2 canary inherits from the v1 canary stays byte-identical (pinned, updated: `tests/unit/data/test_camera_mirror_worker_clone.py::test_v2_canary_config_differs_only_in_identity` now expects exactly `identity_keys | {"kernels.torch_compile_recompile_limit"}`, values 8→64).

### 5. Resolved-TOML byte-identity anchor regenerated

`tests/unit/config/test_camera_viewport_config.py::test_production_resolved_toml_is_byte_identical_to_baseline` freezes the resolved production TOML as a sha256. Adding the governed field changes every resolved TOML by exactly **one** line — verified by byte-diff (base e6d7b10 vs fix tree):

```
584a585
> torch_compile_recompile_limit = 8
```

New anchor: `bac747fd88ae9d276ba3347ea0831940` + `1f869a6cf00a5d80cc8885776dde7f72` (split halves, as before, so display-layer hex masking cannot corrupt the file). Every other resolved leaf is unchanged; all 5 other configs resolve the new field to 8.

## SUPERVISOR STOP-PATH HARDENING

`scripts/mbs_corrected_treatment_supervisor.py` (now TRACKED; copied from the validated untracked `/sakuramoon-runtime/mbs-corrected-treatment-supervisor.py` used in the Phase B run, then fixed):

- **`STOP_ENV_KEYS`** = 15 keys: the 14 operational vars (PROJECT_ROOT, RUNTIME_ROOT, CONFIG_ROOT, CONFIG_NAME, LOG_ROOT, RUN_ROOT, WORKLOAD_ENV_FILE, RESUME_CHECKPOINT, PUBLISH_STATE_ROOT, PUBLISH_LAST_PUBLISHED, REPO_PATH, INTERVAL_SECONDS, REQUIRED_HOST_SUBSTRING, MAIN_PROCESS_PORT) **+ `VENV_ROOT`** — the exact key whose absence fast-failed the Phase B stop.
- **`build_stop_env(a)`** (pure): copies the process env and sets all 15 explicitly; `PYTHON_BIN`/`ACCELERATE_BIN`/`MANAGEMENT_PYTHON` are preserved if present, otherwise derived from `VENV_ROOT` exactly as `training_stack.sh` lines 13–16 do.
- **`--self-test-stop-env`** (pure, side-effect-free, runs before any Supervisor construction): every key non-empty; absolute-path check on all path keys except the 5 conventionally-relative ones (REPO_PATH, CONFIG_NAME, INTERVAL_SECONDS, REQUIRED_HOST_SUBSTRING, MAIN_PROCESS_PORT); VENV_ROOT is a directory; PYTHON_BIN/ACCELERATE_BIN are executable files; the workload env file is a regular non-symlink file with mode exactly 0600; interval/port parse as ints in range; config name ends in `.toml`. Prints `SELF_TEST_STOP_ENV: PASS/FAIL` with per-key PROBLEM lines; exit 0/1.
- **No auto-restart**: the supervisor's only stack interaction on failure is `scripts/training_stack.sh stop` (source-pinned in a test; the `"start"` variant is asserted absent).

Live verification on come7 (real values, corrected-100u worktree + its NUL env file + DTK venv): `SELF_TEST_STOP_ENV: PASS` (rc 0). Planted problems (missing VENV_ROOT, mode-644 env file, symlinked env file, port 70000, interval 0) each flip it to FAIL (rc 1) with the correct PROBLEM line.

## SATURATION SMOKE (forward-only, real production path)

`scripts/mbs_compile_saturation_smoke.py` (new tracked tooling, forward-only by construction):

- Exact v2 canary config (`load_config`), real U118100 composite via `load_inference_artifact` (read-only), local Qwen3.5-2B (`flash_attention_2`, the v2 kernel value) + local Mage-VAE.
- Production compile install: `compile_packed_dit_blocks(..., recompile_limit=config.kernels.torch_compile_recompile_limit)` — then runtime readback gates: `dynamo_config.recompile_limit == 64`, `fail_on_recompile_limit_hit is True`, `suppress_errors is False`, `accumulated_recompile_limit` recorded and unchanged.
- Real production data path: `ProductionPipelineFactory.from_config` → `.batches(lease client)` → spawned persistent DataLoader workers (8) streaming 8 local real danbooru-v2 shards → collate → accepted batches with `camera_mirror` tables.
- `SingleGpuBatchRuntime.measure` under `torch.no_grad()` over **64 real batches** (bounded extend to 128 if distinct counts < 9), same compiled module throughout; per-batch: mirror invariants, finite loss, grad-None scan, all-blocks-still-compiled check, dynamo counter deltas, DiT-sequence capture (`len(latents)` and `len(main_token_lengths)` both must equal the mirror table's `physical_views`).
- Telemetry: `TORCH_LOGS=recompiles` set in the process environment **before** python import (fail-closed check at script start); recompiles parsed from the redirected stderr (`[__recompiles]` entries, guard reasons, per-frame indices); production inductor cache used (`TORCHINDUCTOR_CACHE_DIR=/sakuramoon-runtime/torchinductor-cache`, the same dir `cli/train.py` maps from `SAKURAMOON_TORCHINDUCTOR_CACHE_DIR`, 931 MiB warm from the Phase B run) with `TORCHINDUCTOR_COMPILE_THREADS=8` as production sets.
- Mutation audit: parameter/buffer sha256 checksums (composite, Qwen, VAE) before/after; both checkpoint `manifest.json` shas and the full source-ckpt file-size fingerprint before/after; treatment output root must stay empty.
- growth_alpha = 1.0: the U118100 stage-relative update count (60100) far exceeds the growth schedule's `max_updates` (5000), so the half-cosine schedule has converged to 1.0 (same value as the launch-readiness compile smoke).

### Telemetry captured (root-cause confirmation)

The redirected stderr contains the exact Phase-B guard, now observed under the governed ceiling:

```
[0/2] [__recompiles] Recompiling function forward in …/model/block.py:216
    triggered by the following guard failure(s):
    - 0/0: len(boundaries.sequence_lengths) == 26   # self.attention(  # block.py:250
[1/1] [__recompiles] Recompiling function forward in …/model/attention.py:510
    - 1/0: len(boundaries.sequence_lengths) == 26   # fa4_varlen_attention(  # attention.py:532
```

Recompile taxonomy observed: (a) one-off type/growth-refinement recompiles at startup (`___check_type_id(mlp_growth, …)`, `attention_growth` float refinement), (b) one recompile per distinct physical row count via `len(boundaries.sequence_lengths)` — the Phase B root cause, now bounded.

### Long-run results (64 real batches, no extension needed)

| gate | result |
|---|---|
| batches run | 64 (target met at 64; bounded extend to 128 NOT triggered) |
| distinct physical sequence counts | **10** — `[20, 21, 22, 23, 24, 25, 26, 27, 28, 30]` |
| hard gate (>= 9) | **MET** (prefer >= 12: not reached — real data produced 10 of the 21 possible values in 64 batches; no shapes fabricated) |
| SATURATION_COVERAGE | **SUFFICIENT** |
| recompile-limit failure (FailOnRecompileLimitHit / "recompile_limit reached") | **NO** |
| accumulated-limit failure | **NO** (`accumulated_recompile_limit` 256 -> 256, untouched) |
| eager fallback | **NO** (every packed block retained its compiled impl after every batch; no fallback markers in stderr) |
| traceback in stderr | **NO** |
| finite losses | **YES**, all 64 (per-batch mean-loss range 0.4507 .. 0.6186) |
| per-batch MBS invariants | **PASS** (severe_vertical = sev2to4 + ge4; eligible = severe; selected = eligible; applied = selected; degraded = 0; extra_views = applied; physical = logical + applied) |
| DiT sequence capture | **PASS** every batch: `len(latents) == len(main_token_lengths) == physical_views` |
| grad None before/after | **PASS** (scanned after every batch; before and after the run) |
| parameter/buffer mutation | **NONE** (sha256 of every param+buffer: composite, Qwen, VAE) |
| source + control manifests | **UNCHANGED** (re-hashed inside the run before/after) |
| treatment output root | **EMPTY** (no checkpoint, no file) |

Aggregate mirror table (1280 logical rows, 64 batches): `eligible = selected = applied = 138` (severity 2-4: 111, severity >= 4: 27); `extra_views = 138`; `physical_views = 1418 = 1280 + 138`.

Recompile telemetry (`TORCH_LOGS=recompiles`, authoritative re-parse of the redirected stderr):

- **29 recompile events**, 236 guard-reason lines, **11 distinct guard reasons**:
  - 2 one-off startup refinements (`___check_type_id(mlp_growth, …)`, `attention_growth` float refinement);
  - **9 × `len(boundaries.sequence_lengths) == N`** for N ∈ {20, 21, 22, 23, 24, 26, 27, 28, 30} — the Phase B root-cause guard, now bounded by the governed ceiling.
- Max per-frame recompile index: **19** (well under the governed 64). Dynamo counters: `unique_graphs = 33`, `calls_captured = 1695`.
- Runtime readback after the production install: `recompile_limit = 64`, `fail_on_recompile_limit_hit = True`, `suppress_errors = False`, `accumulated_recompile_limit = 256` (torch default, unchanged).

Wall/memory: total **360.2 s** (model load + 64 batches); first (coldest) shape 38.5 s, new-but-cached shapes 3-8 s, warm shapes 0.33-0.5 s; peak **8.835 GiB** on cuda:0 (64 GiB DCU).

## TEST EVIDENCE

| suite | fix tree | base (e6d7b10) |
|---|---|---|
| focused (10 files, this change) | **58/58 PASS** (22.7 s) | n/a |
| full `tests/unit` | **4 failed, 1008 passed** (271.6 s) | **4 failed, 962 passed** (277.6 s) |

- The 4 failures on BOTH trees are the identical pre-existing `tests/unit/optim/test_cmuon_fp32_forensic.py` set (present at base e6d7b10 before any change this round): `test_bcd_hard_fail_captures_fp32_reason[below_floor-below_floor]`, `test_e_writer_failure_still_raises_original_error`, `test_f_serialization_failure_still_raises_and_leaves_no_capsule`, `test_m2_mirror_failure_keeps_local_capsule_and_original_error`.
- **No new failures vs base.** Both trees run with identical `--ignore`s for the two `deepghs` test modules (environmental: `onnxruntime` not installed in the come7 DTK venv; both trees fail identically without the ignores).
- New deterministic tests: 0 skip, 0 xfail.
- New/updated tests: `test_kernels_recompile_limit.py` (default/bounds/12 rejections/resolved per-config/sibling-clobber), `test_production_compile_binding.py` (source pin + functional 37), `test_distributed_compile.py` (3 existing call sites pinned at 8 + 4 new: configured-limit binding [8,64] with teardown, invalid-value rejection with global-config-untouched, accumulated-limit preservation, suppress-errors-disabled requirement), `test_mbs_corrected_treatment_supervisor.py` (9: 15-key stop env, bin derivation, bin preservation, self-test PASS, missing-VENV_ROOT FAIL, mode-644 FAIL, symlink FAIL, port/interval FAIL, never-restarts pin), plus the 2 contract updates documented above.

## STATIC / HYGIENE

- `ruff check` (0.16.1, repo config) over all 12 changed files: **All checks passed**.
- `python -m py_compile` on both scripts: OK.
- `git diff --check`: clean.
- Staging: explicit `git add` of exactly the 12 whitelisted files; no `git add .`; no secrets, env files, checkpoints, or binaries staged. Both scripts carry shebang + 100755 (repo convention for executable scripts).

## STATE INTEGRITY (verified this round)

| artifact | sha256 | immutable |
|---|---|---|
| source U118100 `manifest.json` | `821a8c12546a094865972fef0c23152e9754e01f82eed8353f7248b92076b892` | YES |
| control U118200 `manifest.json` | `f014827fb441aca8780a5bb710eeb2631a40aac112ed3aa6b07b6e3b65d657f7` | YES |
| treatment output root | empty (no checkpoint, no file) | — |

Both shas re-verified on come7 immediately before staging (exact match with the Phase B frozen hashes); the saturation smoke itself re-hashes both before and after the run and gates on equality.

No backward, no optimizer object, no scheduler step, no checkpoint write, no publisher, no U118101, no P50, no production stack started this round.

## COMMITS (exact)

```
H2  docs: record MBS compile saturation fix        (ONLY the two report files)
H1  fix: govern packed compile recompile limit     (4 code/config + 2 supervisor/tests + 2 contract-test updates + 4 new test files/dirs)
    e6d7b10  (frozen Phase B evidence)
```

`git rev-list --count e6d7b10..HEAD` = 2. Push: branch only, no force; readback verified via `git ls-remote origin`.

## EVIDENCE HASHES

| artifact | sha256 | location |
|---|---|---|
| saturation smoke script | `ffb7885888cf33e4805328e62e5184a23024c1ba22bab16e6b4609d50a89cd4f` | `scripts/mbs_compile_saturation_smoke.py` (this branch) |
| supervisor script | `e86a209d7fa4b8e31aa3d54d5eb7be65d642cb201a80b68bc9437a8ba4066e51` | `scripts/mbs_corrected_treatment_supervisor.py` (this branch) |
| long-run stderr (recompiles sink) | `f2b30730ce9f3a9fb565fb9eb978d21085c004c7fe5edcb29d565883f8f61974` | `/sakuramoon-runtime/mbs-recompile-fix/long/stderr.log` (come7) |
| long-run stdout | `85f8c91d31ce30e6daccda1dc61445273f65cb9813c85790d11079ff702e03ee` | `/sakuramoon-runtime/mbs-recompile-fix/long/stdout.log` (come7) |
| long-run evidence JSON | `3cbadaa376cb14266cf77a5292f93ca443658e9162ddcdcac632755a0430c0dc` | `/sakuramoon-runtime/mbs-recompile-fix/long/evidence.json` (come7) |
| source U118100 `manifest.json` | `821a8c12546a094865972fef0c23152e9754e01f82eed8353f7248b92076b892` | immutable |
| control U118200 `manifest.json` | `f014827fb441aca8780a5bb710eeb2631a40aac112ed3aa6b07b6e3b65d657f7` | immutable |

Probe history (same smoke, earlier invocations, for provenance): probe-1 (default inductor cache, capture bug found: DiT input is the per-view latents tuple, not a `main_token_lengths` carrier), probe-2 (production cache; all gates green except the capture), probe-3 (capture verified: physical 26/24/23 real values). Probe artifacts under `/sakuramoon-runtime/mbs-recompile-fix/probe*/`.

Teardown artifact (no result impact): at stream close (15:28:49, kernel `core_pattern=core`) one spawned DataLoader worker wrote a 2.5 GiB core (`core.67129`, `multiprocessing.spawn` child) — a DTK/HIP process-teardown race. The main process had already completed all 64 batches, every gate, and the evidence write; no mid-run worker failure occurred. The untracked core binary was removed from the worktree (never staged).

## FINAL

```
CANARY_RERUN_READY                  = TRUE
training authorized THIS ROUND      = NONE

The corrected v2 canary (U118100 -> U118200, 2 ranks) may be started ONLY on a
later explicit GO. Launch with the tracked supervisor and run
`--self-test-stop-env` before `training_stack.sh start`.

Expected compile behavior on the real run: one recompile per distinct physical
row count (<= 21 possible values), well under the governed ceiling of 64;
FailOnRecompileLimitHit stays armed (fail-closed). No further code change is
required to start the rerun.
```
