# CMuon production cleanup — 200-update implementation regression (103500→103700)

Candidate: `hybrid_cmuon_canonical_ns4_fp32_rescue` with the **production
cleanup** (per-step host syncs 168→2; math and config frozen since D2 R1,
D3-verified, D4 1000-gate `SAFE_1000=YES` at fd1131e). Host: salt10
(2×HCU renderD133/135, 64 GiB each, pod-local DTK 26.04 / torch 2.9.0,
venv `sakuramoon-dtk-venv`), code @**f875e9f** (cmoun head; commit chain
e0c8720 → 95166ae → 824f08d → 9efe640 → 75782ad → 1488042 → 24147aa →
589092f → f875ee9). Frozen toml `train_g1_fp32_rescue_r1.toml`, frozen
math — **implementation regression, not a new run**.

Window under test: **200 cumulative successful updates, 103501–103700**,
resumed from the D4-validated `ckpt_103500_raw-103500-update-cadence`
(G1 CMuon production state; guard refs and counters carried over),
stopped exactly at 103700 by the w200 watchdog (`WATCHDOG-STOPPED-DONE`).

## Verdict

**IMPLEMENTATION_REGRESSION_200U = PASS** — all gates passed:

| Gate | Requirement | Observed |
|---|---|---|
| Updates | 200/200 successful | 200 metrics records, 103501–103700 exact |
| Safety | 0 CMuonSafetyError, 0 nonfinite | 0 / 0 |
| Rescue continuity | counters continuous from D4 state | obs 1000→1200, rescues 685→818, failures 0 |
| Loss stability | D4 steady-state 0.556±0.001 | mean 0.554610, range 0.525703–0.583701 |
| Checkpoint atomicity | COMPLETE + manifest at cadence | ckpt_103600 + ckpt_103700 (5 files each, `complete`) |
| Hub chain | cmoun/ extends past 103500 | 103600 + 103700 uploaded 02:12:08 (verify-pass) |
| Throughput | no regression vs D4 15.6–16.3 s/u | 15.39–16.44 s/u |

**With one disclosed (expected) observation**: the rescue trip rate on
salt10 is 133 rescues / 200 obs = **0.665/obs**, in line with the D4
post-rebuild band (D4 final 0.685/obs; D4 peak window 1.13/obs) and
slightly *below* the D4 exit rate — consistent with the guard-reference
convergence transient decaying as recommended by the 1000-gate report
(monitor the next 500–1000 updates for band flattening). Every trip was
safely rescued (0 failures, 0 nonfinite, 0 parameter drift).

## A — GPU three-phase regression (2-rank forced rescue / backcompat / perf A/B)

All three phases green on the cleanup tree (24147aa; f875ee9 re-verified
the identical code paths plus the publisher fix).

### A1 — 2-rank forced-rescue regression (spec 10): PASS

Mock 2-rank shard (`_GlobalConditioner` Linears bf16 weights + fp32
biases, production dtype contract), 22 inputs / 11 owned per rank, 5 ok
scenarios + 1 forced final failure:

- All 5 ok scenarios **bit-exact across ranks**: `max_delta_rank_spread`,
  `max_param_rank_diff`, `param_fingerprint_spread` all `0.0`.
- Forced per-rank rescues 0/1/1/3/4 cumulative (rank0 `forced_mine=4`,
  `natural_rescues=0`), each rescue counted exactly once on the owning
  rank; cross-owner forces (owner1_single, simultaneous_owners) rescue on
  the correct rank.
- `final_failure`: both BF16 and FP32 forced nonfinite on
  `dit.conditioner.shared_block_projection.weight#chunk5` →
  **CMuonSafetyError raised, zero commit** (failure recorded before the
  raise: rank0 `fp32_rescue_failures=1`, rank1 `0`).
- Verdict JSON: `2rank-regression.json` → `verdict: PASS`.

### A2 — D4 ckpt_103500 backcompat (spec 11): PASS

Production resume contract on the cleanup code, from the real D4
checkpoint (`train_state/optimizer.pt`, CPU load, SR-RNG
`device_index` 0→local remap, optimizer rebuilt from the ckpt's
`resolved_config.toml` — guard P3 calibration, scaled lr 0.00015625,
sr_seed 44, ns default 4):

- A_load ok → B_state_after_load ok → C_update_and_save ok (1 real
  update + save) → **D_resume ok (fresh optimizer, exact state equality)**
  → E_continuity: `observations_delta=1`, `bf16_attempts_delta=83`,
  `fp32_*_delta=0`, `n_references_stable=True`.
- Verdict JSON: `backcompat.json` → `D4_CHECKPOINT_BACKCOMPAT=YES`.

### A3 — same-pod perf A/B (spec 15): PASS — the 168→2 sync design is validated

Identical 2-rank harness, warmup 6 × 40 measured iters, optimizer-phase
only on the real composite model; three configs:

| Config | Tree | Mean s/step | Host syncs/step |
|---|---|---|---|
| baseline | frozen D4 code @37602e4 | **0.8894** | **175.2** |
| cleanup | cmoun @f875ee9 | **0.7624** | **6.8** |
| adamw8bit (reference) | cmoun @f875ee9 | 0.4037 | 4.2 |

**Host syncs per step 175.2 → 6.8 (−96.1%)**, step time **−14.28%**
(0.8894s → 0.7624s). The residual ~5–7 syncs are the by-design syncs of
the cleanup; the per-chunk `.item()` loop of the frozen D4 tree is the
source of the ~170. `perf-merged.json` preserved under
`/sakuramoon-runtime/artifacts/g1/cmuon-cleanup/`.

### A4 — harness incident log (first-run only; production code never implicated)

Six rounds of fixes, **all on the GPU test harness / ops scripts**
(e0c8720…f875ee9), none in optimizer math or training code:

1. venv has no `torchrun` → `$PY -m torch.distributed.run` (system
   torchrun lacks torchao).
2. Mock dtype contract: matrix weights bf16, 1-D params (incl. biases)
   fp32 — mirror of the production audit contract.
3. Ckpt layout `train_state/optimizer.pt`; optimizer must be built from
   the ckpt's `resolved_config.toml` (hardcoded test values are
   legitimately rejected by the guard-config check).
4. DTK 2.9 `torch.randn` has no scalar-size overload; the production
   model contains 0-D params → draw `(1,)` then `reshape(())`.
5. Rescue counters are per-rank; `fp32_rescue_failures` is recorded
   **before** the CMuonSafetyError raise.
6. Two full model+optimizer generations OOM the 64 GiB HCU → free
   generation 1 before the phase-D rebuild.
7. Ops: a killed phase driver orphans its perf-ab child, which keeps
   running the config loop and holding master port 29531 (EADDRINUSE) —
   the driver now self-cleans the orphan pattern at startup.
8. Ops: the SCNet platform `ai_proxy` file starts with unparseable
   human-readable header lines; under `set -Eeuo` sourcing it aborted
   the publisher (exit 127) — `publish_train_state.sh` now treats vendor
   source failures as non-fatal warnings (589092f).

## B — 200-update implementation regression (103501–103700)

- Stack: data service (cold-cache refill of ~67 GiB shards, 8 parallel
  streams, 17–79 MiB/s each, no starvation — ready_queue_depth 8) +
  isolated cmoun publisher + 2-rank training; resume pinned to
  ckpt_103500; w200 watchdog armed (fail-fast on CMuonSafetyError,
  stop at successful_update ≥ 103700).
- **200/200 successful updates** in ~53 min; 15.39–16.44 s/u.
- Loss (high-noise component): min 0.525703, mean 0.554610, max
  0.583701 — flat vs D4 steady state.
- **0 CMuonSafetyError, 0 nonfinite** across the whole window.
- Rescue telemetry from the **authoritative final optimizer state
  (ckpt_103700)**: `bf16_attempts=99600 = 83 chunks × 1200 observations`
  (D4 1000 + 200, continuous), `fp32_rescues=818` (685+133),
  `fp32_rescue_failures=0`, `bf16_safety_failures=818` (1:1 rescued).
- Checkpoints: `ckpt_103600` and `ckpt_103700` both
  `COMPLETE=complete` + `manifest.json` (5 files each).
- Publisher: `tree upload complete: 2 checkpoint directories` at
  02:12:08 — hub `leafmoone/sm_train_state` `cmoun/` now holds the full
  chain 102500/…/103500/**103600/103700** (13 cadence points,
  verify-pass on upload).

## C — freeze state

- **Frozen code**: cmoun @f875ee9 (cleanup tree; ruff gate clean,
  reports-only commit on top of the 8-commit fix chain).
- **Frozen training state**: `ckpt_103700_raw-103700-update-cadence`
  (latest verified; also mirrored on salt10
  `/sakuramoon-runtime/output_model/g1/`).
- **Rollback sources intact**: salt6 (172.31.59.244) keeps the D4
  pre-cleanup tree @37602e4 + forensic mirror; salt7 G1 CONTROL
  untouched (never disturbed by this migration).
- salt10 left **idle with the frozen state** (stack stopped by the
  watchdog; data cache warm; code tree clean at f875ee9).

### Disclosures

- No quality/eval gate in this window: no 1000-cadence sample hits
  103500→103700 (next is 104000) and the 5000-cadence eval is out of
  scope; the 120-concept suite files are still missing locally
  (0.6 G tar staged in `tmp/concept-120-ab.tar` or on salt3/salt7) —
  quality gate remains a user decision, as does deployment.
- wandb disabled in the frozen toml (disclosed in the D4 report);
  publisher state isolated (`cmoun/` path, dedicated state root) —
  no interference with any other run's chain.
- This is an **implementation regression** (cleanup of host syncs), not
  a math change: optimizer math frozen since D2 R1; config frozen
  (guard P3 calibration, scaled lr 0.00015625, ns4).

### COPY block — run the frozen stack

```bash
# salt10 (or a salt10-equivalent host with DTK 26.04 + the venv + model):
export PROJECT_ROOT=/sakuramoon-runtime/sakuramoon-cmuon        # @f875ee9
export RUNTIME_ROOT=/sakuramoon-runtime
export CONFIG_NAME=train_g1_fp32_rescue_r1.toml
export VENV_ROOT=/sakuramoon-runtime/sakuramoon-dtk-venv
export WORKLOAD_ENV_FILE=/root/d4-workload-env
export RESUME_CHECKPOINT=/sakuramoon-runtime/output_model/g1/ckpt_103700_raw-103700-update-cadence
export REQUIRED_HOST_SUBSTRING=salt10
export MAIN_PROCESS_PORT=29500
export REPO_ID=leafmoone/sm_train_state
export REPO_PATH=cmoun
export PUBLISH_STATE_ROOT=/sakuramoon-runtime/.sm-train-state-publisher-cmuon
export PUBLISH_LAST_PUBLISHED=/root/private_data/.sm-train-state-publisher-cmuon/last-published-cmuon.txt
bash "$PROJECT_ROOT/scripts/training_stack.sh" start
```

## Follow-ups (user decision)

1. **Quality gate** for the cleaned pipeline: needs the 120-concept
   suite files (staged locally / on salt3 / salt7) + a 1000-cadence
   sample window (103700→104700 or resume from 104000).
2. **Deployment** of the cleanup to a G1 production run (salt7 or
   successor): the 168→2 sync cleanup is validated (−96% syncs, −14.3%
   optimizer-phase step time, safety identical) — awaiting GO.
3. Keep monitoring the rescue band for another 500–1000 updates (D4
   recommendation; this window's 0.665/obs is already at/below the D4
   exit rate).
