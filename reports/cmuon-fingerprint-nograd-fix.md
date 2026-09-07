# CMuon F3 parameter fingerprint no-grad fix — final report

Date: 2026-09-07 · Machine: come6 (2× DCU 64 GiB, DTK torch 2.9.0) ·
Branch: `fix/cmuon-fingerprint-no-grad` (parent a88c20c) ·
Implementation commit: `38f877812b4ff81642585ca78bf729b86dc8baf7`
(`fix: avoid autograd graphs in CMuon parameter fingerprints`)

## IDENTITY

- Fix: `src/sakuramoon/optim/fp32_rescue.py`, ONE hunk (24 ins / 17 del, net +7)
  in `HybridCMuonCanonicalNS4FP32Rescue.step()` — the post-commit parameter
  fingerprint invariant block (BASE lines 700-718) re-indented under
  `with torch.no_grad():` + 6-line explanatory comment.
- No decorator, no inference_mode, no `.data`, no requires_grad changes;
  statement order, dtypes and the error message are verbatim.
- File sha256: BASE `cf26f316ce43e92f3038701c82ff33689c62c27a28a40b9073347bfebc940dd7`
  → FIX `7d9b4ccc57bd192fab939d0ed6f44f11cd4efc3e5f4897f40b4d31119dff1b65`.
- Committed files in 38f8778 (exactly 5): src fix, unit test, 2-rank HCU
  test, memprobe tool, this report (v1).
- This report's measurement-scope and number corrections (v2) are appended by
  a SEPARATE documentation commit on top of 38f8778. Integration target is
  FINAL_HEAD (the branch tip at merge time); the implementation code is
  38f8778's `src/` blob and is byte-identical in FINAL_HEAD.

## ROOT_CAUSE

The post-commit parameter fingerprint block executed with grad mode ENABLED
(it sits outside the PHASE-2 commit no_grad scope). For every CMuon spec:
`pf = spec.parameter.float()` produces an fp32 tensor requiring grad (leaf
parameter as source); `pf.pow(2).mean().sqrt()` and `pf.abs().max()` then
build autograd graphs whose backward nodes retain FULL-SIZE fp32 copies of
`pf` as saved tensors until freed. At production scale this retains ~8 bytes
per bf16 parameter element per step (≈11.39 GiB at the GO-estimated
1.5288e9-element scale) with no diagnostic value — the fingerprint is a
read-only safety check.

## PROOF

RED (BASE, real F3 path, explicit `torch.enable_grad()`,
`saved_tensors_hooks` metadata probe): param-sized saved tensor appears with
`grad_fn=CloneBackward0` → contract test FAILS.
`1 failed, 3 passed in 27.29s` (evidence: red-run-base-a88c20c.txt).

GREEN (FIX, same env/file): `4 passed in 27.68s`
(evidence: green-run-fixed.txt).

Zero-semantic-change evidence (identical-process, same env):

- C2 bit-exact parity BASE-vs-FIX: fp lo/hi values, parameters, grads, guard
  refs, counters (`fp32_low_delta_by_role`, `_steps_this_process`,
  `max_param_rank_diff`), `sr_rng` state, torch RNG bytes, and the FULL
  `state_dict` (incl. `OptimState8bit` codes/scale/qmap constituents) —
  identical.
- Branch matrix: invariant+ws2 → 2 param-fp all_reduces; invariant off →
  delta fp only; ws1 → zero collectives; outer no_grad → inner mode unchanged.
- Controlled mismatch: same `CMuonSafetyError` type + message prefix as
  pre-fix; `max_param_rank_diff` still updated before the raise
  (≈ message diff within %.3e rounding, rel=1e-3); grad mode restored after
  the raise.

2-rank HCU safety (real NCCL, real HCU Newton-Schulz, 2 DCU, torchrun;
evidence: 2rank-result.json / 2rank-run.log):

- S1 (3 clean steps): `max_param_rank_diff == 0.0` on every step AND an
  independent post-step cross-rank fingerprint (own all_reduce, outside the
  optimizer) reports spread 0.0; observations 1→3.
- S2 (rank-0 controlled mismatch, 1e-3 perturbation of the param-lo MIN
  reduce): the MIN/MAX reduce synchronizes the mismatch to BOTH ranks —
  a consensus failure, as designed — and BOTH ranks raise
  `CMuonSafetyError` with the EXACT same message:
  `fp32-rescue rank invariant violated: cross-rank parameter fingerprint
  diff 1.000e-03 after commit`; step returned unsuccessfully on neither rank
  (see failure-timing note in SAFETY).
- `torchrun` exit 0 (2RANK_PASS), re-verified on the final lint-clean test
  file (sha 800e1be6…).

## SAFETY

- Runtime diff = fingerprint block ONLY (single hunk); `src/config` diff
  EMPTY; secret scan 0 hits; `git diff a88c20c..HEAD` limited to the 5
  committed files; worktree clean after commit.
- No push/merge to any remote, no deploy, no canary, no main-training
  change: G1 (come1) and iREPA production (come2) untouched; Camera HARD
  STOP respected; sync to the come2 LOCAL clone only (ref-only fast-forward,
  detached worktree and running job unaffected).
- The fix only narrows autograd scope around a read-only diagnostic; commit
  order, collective order, dtypes and messages are unchanged.
- Failure timing: the fingerprint check is POST-COMMIT. When it raises on a
  mismatch, the step does not return successfully, but the parameter update
  for that step has ALREADY been applied on every rank — the mismatch path
  is NOT a zero-commit path. The pre-commit hard-fail contracts (nonfinite
  gradients / above-ceiling delta) keep their original semantics unchanged
  and are separate from this check.

## MEASUREMENT

MEASUREMENT SCOPE: SYNTHETIC_PARAMETER_F3_OPTIMIZER_STEP — the timed/peak
window wraps the REAL F3 `opt.step()` call (the full optimizer step,
including Newton-Schulz) run on a synthetic parameter set. Because the
window covers the full step, this is NOT a fingerprint-block-only
measurement, even though the code change is in the fingerprint block.
Isolated: NO production trainer, NO full Qwen/VAE/DiT, NO production ckpt
read. Separate process per version (BASE: PYTHONPATH = main clone src
@a88c20c on cuda:0; FIX: worktree src on cuda:1; the two processes ran in
parallel on the two separate DCUs).
Temporary parameter set at the GO-estimate scale: 1,547,567,104 bf16
elements (3.145 GB params, 36 cmuon specs, production FQN layout) — a
synthetic set, not the full production training model; real HCU NS;
world_size=2 with identity collective mocks (the fingerprint block runs only
for ws>1; the mock exists so a single process can exercise the ws=2 path —
its performance does NOT represent real NCCL); 2 warm-up + 3 measured steps
each; `synchronize` before/after; `reset_peak_memory_stats` per measured
step; no `empty_cache` and no heavy hooks in the timed windows.

| metric (3/3 runs bit-stable) | BASE | FIX | delta |
|---|---|---|---|
| peak_allocated | 28,215,760,384 B (≈26.278 GiB) | 17,238,217,728 B (≈16.054 GiB) | **−10,977,542,656 B (≈10.224 GiB)** |
| post-step allocated | 9,508,552,704 B (≈8.856 GiB) | 9,508,552,704 B (≈8.856 GiB) | 0 (identical on BASE and FIX; no steady-state change) |
| step wall time (3 raw runs) | 0.733 / 0.732 / 0.742 s | 0.734 / 0.733 / 0.745 s | no obvious change observed (see limitations) |

Timing limitations: BASE and FIX were measured in parallel on DIFFERENT
DCUs (cuda:0 vs cuda:1); only 3 measured steps per side; no obvious time
change was observed in these windows. This is not a strict proof of zero
time cost, and no training-throughput improvement is claimed.

Expected retention: 8 B/element × 1,547,567,104 = 12,380,536,832 B
(≈11.53 GiB at this run's parameter scale). The measured peak reduction
(≈10.224 GiB) is ≈88.7% of THIS run's ≈11.53 GiB estimate — the percentage
is stated only to indicate order of magnitude; it is not a correctness
threshold, and no additional measurement is required to match the theoretical
value. (The GO's 11.39 GiB corresponds to the 1.5288e9-element scale, not
this run's 1,547,567,104-element set.) The measured peak difference and the
theoretical saved-tensor retention are not the same metric; peak timing and
other transient allocations may affect the difference; no further
attribution was made in this round.

WHOLE_TRAINING_MEMORY_DELTA = NOT_MEASURED ·
WHOLE_TRAINING_SPEEDUP = NOT_MEASURED.

## REGRESSION

- Targeted 4-file suite, same env, BASE vs FIX: FAILURE_SETS_IDENTICAL —
  `4 failed, 44 passed` on BOTH sides (BASE 176.96 s / FIX 152.67 s). The 4
  failures pre-exist on BASE (forensic `below_floor` stale fixture +
  writer/serialization/mirror failure-injection tests, root-privilege
  artifacts). No new failure, no changed failure.
- ruff 0.16.1: changed files "All checks passed"; repo-wide 8 == BASE 8
  (0 new). New files (unit/2-rank/memprobe) clean after fix.
- pyright 1.1.411: 88 errors == BASE 88 (diff is the pure +7-line shift).

## BOUNDARIES

- Scope: one optimizer method's diagnostic block. Other optimizers and code
  paths unchanged.
- The 36-spec synthetic parameter benchmark is NOT the full production
  training model; the identity collective mock's performance does NOT
  represent real NCCL; real 2-rank correctness is provided by the separate
  NCCL test (S1/S2 above).
- The 2-rank test uses the production-layout mock model (15 cmuon specs) —
  it verifies the safety semantics on real NCCL/HCU, not production scale.
- No production-scale end-to-end training measurement (see NOT_MEASURED
  above).

## VERDICT

**PASS.** The post-commit parameter fingerprint no longer builds autograd
graphs (RED→GREEN), with bit-exact semantic parity, identical regression
behavior, verified 2-rank NCCL safety (including the consensus-failure
path), and a measured peak-memory reduction of ≈10.224 GiB on the synthetic
full-step window (≈88.7% of this run's ≈11.53 GiB estimate), with no obvious
time change observed. Implementation commit 38f8778 made; report corrections
appended by a documentation commit; branch synced to the come2 local clone
(ref-only fast-forward). STOP.

## NEXT

1. User review of the commits + evidence directory
   (`/sakuramoon-runtime/cmuon-fp-nograd-evidence/`).
2. Deployment (canary → production) is a SEPARATE GO — not performed here.
3. Unrelated pending: iREPA 256-resume decision awaits user GO (ckpt_115500
   safe recovery point on come2); 12 NEEDS_REPROVISION dsh-ssh hosts need
   user re-provisioning (esp. come1 = G1).
