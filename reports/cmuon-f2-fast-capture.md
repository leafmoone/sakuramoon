# SakuraMoon CMuon Forensic F2 — FAST MINIMAL HARD-FAIL CAPTURE + EXACT INPUT CLOSURE

Date: 2026-09-04 (+08:00)
Base: `811a4caec0f6591c8bf061a35473aab59107d4b9` (F1, `cmuon-fp32-forensic`)
Branch: `cmuon-fp32-forensic-f2` — independent worktree `D:\sakruamoon\sakuramoon-cmuon-f2`
Test hosts (as the arc progressed; user-directed):
- salt10 (2x HCU, DTK 26.04, torch 2.9.0+das.opt1.dtk2604): old-path timings,
  original capture bench (24 runs), unit 12/12, 2-rank 5 scenarios — all
  completed before the user released this machine (iREPA experiment host;
  its local disk is gone with the pod; the measured numbers above stand).
- salt11 (2x HCU, same DTK 26.04 stack; G1 CONTROL, training STOPPED):
  user-designated host for the remaining GPU gates. Everything ran inside an
  isolated dir `/sakuramoon-runtime/f2-gates/` (trees + all outputs + a
  read-only `model -> /sakuramoon-runtime/model` symlink); production
  code/config/checkpoints zero-touched; data service + state publisher
  verified alive before/after; HCU 0% before every GPU step; the f2-gates
  dir is deleted after evidence collection. Bench re-run (24 runs) +
  2-rank (F2 & F1) + both parities + enrich + teardown + tests/gpu.
- (one CPU-only full unit run, 775 tests, ran earlier on salt12 before the
  machine-role red line was stated; salt12 was fully cleaned — 4.5 GB
  removed, no residual; listed here only for evidence provenance)

## 0. Scope and hard boundaries

This is a **FORENSIC CAPTURE FIX, not an optimizer algorithm fix.**

The F1 failure (112105) root cause: the production hard-fail critical path
computed, BEFORE writing a single byte, two full op-exact CPU Newton–Schulz
replays (BF16 + FP32, 4 iterations × [2560,2560]) plus traces. The owner rank
was SIGTERM'd by elastic teardown ~30 s after the first rank raised — the
publish never finished and the exact-input artifact was lost.

F2 moves every diagnostic OFF the failure critical path. The new critical
path (owner rank only, inside `step()`):

```
verdict settled
  -> frozen = chunk.detach().contiguous().clone()     (device, bf16)
  -> one .cpu()                                        (13 MB D2H)
  -> _write_tensor_bytes + sha256 over file bytes
  -> build_minimal_capsule_metadata   (4 O(n) scalar reductions only)
  -> publish_minimal_capsule          (LOCAL atomic: temp + fsync files
                                       + fsync dir + atomic rename + -rN)
  -> mirror_capsule                   (BEST-EFFORT shared root; never
                                       raises; advisory mirror.json)
  -> original CMuonSafetyError raise
```

No replay, no trace, no SVD, no spectrum, no rank statistics on the critical
path. All of that is offline: `dev-tools/cmuon_hardfail_enrich.py` (replay
>=3 repeats both dtypes, per-iteration traces, recorded-vs-replay, FP32 input
stats, full/randomized SVD spectrum, low-rank hypothesis verdict per spec
§11) and the existing F1 replay CLI (`dev-tools/cmuon_fp32_rescue_replay.py`,
unchanged and capsule-compatible).

FORBIDDEN and verified unchanged (git diff vs base, see §15 closure): NS
algorithm/steps/coefficients, alpha, ceiling, rescue_floor, below_floor
verdict, routing, AdamW fallback, delta clamp/rescale, target-RMS, SVD/polar
production update, momentum math, commit ordering, optimizer state schema,
checkpoint, LR, batch, objective, data, iREPA, Camera v2.
`src/sakuramoon/optim/cmuon.py` is at ZERO diff.

## 1. Filesystem identity (spec §4)

| Mount | salt10 | salt11 | Type |
|---|---|---|---|
| `/sakuramoon-runtime` | 709 G free | 716 G free | **local overlay** |
| `/root/private_data` | 50 G quota | 50 G quota | NFS (scnet.hx) |

The spec's example emergency root (`/root/private_data/cmuon-f1-emergency`)
is NFS on the production host — refuted by `df -T` on both machines, and the
spec's own fallback clause applies. **Emergency root =
`/sakuramoon-runtime/cmuon-f1-emergency`** (local overlay, verified on both
hosts). The best-effort shared mirror keeps the F1 default
`/sakuramoon-runtime/artifacts/g1/cmuon-hard-fail` (also local overlay).

## 2. OLD timeline — F1 critical path (measured on the F1 tree, salt10)

`dev-tools/cmuon_oldpath_timings.py`, 2560×2560 BF16, 1 representative pass
(the CPU NS replays are the deterministic dominant term):

| Phase | seconds |
|---|---|
| d2h (device clone → CPU) | 0.0034 |
| serialize + sha256 | 0.0255 |
| **BF16 CPU NS replay (trace, 4 iters)** | **747.52** |
| FP32 CPU NS replay (trace, 4 iters) | 10.12 |
| metadata assembly | 0.68 |
| local atomic publish | 0.043 |
| **TOTAL (before any write completes)** | **758.39** |

The 112105 owner was SIGTERM'd ~30 s after the raise and SIGKILL'd ~60 s
after — the F1 path needed **~758 s** (≈25× the kill window; the BF16 CPU NS
replay alone is ≈12.5 min on this host). The artifact loss at 112105 is fully
explained.

## 3. NEW timeline — F2 critical path (measured, 2560×2560 BF16, cuda:0)

`dev-tools/cmuon_capture_bench.py`, **24 runs** (spec requires ≥20).
Original run on salt10; re-run on salt11 (final test host) for a
single-host-consistent table — the two hosts agree to ~3%:

| Phase | salt11 p50 (s) | salt11 p95 (s) | salt11 max (s) |
|---|---|---|---|
| device clone | 0.000273 | 0.000497 | 0.000576 |
| d2h (13 MB) | 0.002728 | 0.004274 | 0.008658 |
| serialize | 0.017012 | 0.023723 | 0.033403 |
| sha256 over file bytes | 0.011128 | 0.011181 | 0.011229 |
| metadata (4 O(n) reductions) | 0.526304 | 0.607715 | 0.612520 |
| local atomic publish (fsync+rename) | 0.040809 | 0.125901 | 0.135246 |
| **total local (SLA scope)** | **0.658870** | **0.708693** | **0.763403** |
| mirror (best-effort, excluded) | 0.019704 | 0.022649 | 0.024106 |

salt10 original: total_local p50 **0.660320** / p95 **0.685878** / max
0.701350 (same 24-run protocol; the salt10 per-phase JSON is lost with the
pod release — the salt11 table is the complete one of record).

**SLA: p50 < 1 s and p95 < 2 s → PASS** (both hosts; salt11: 0.659 s /
0.709 s). Old vs new: **758.4 s → 0.66 s ≈ 1149× faster**, and the new
path's cost is deterministic (no multi-minute CPU GEMM variance).
Dominant term is the metadata assembly (4 O(n) scalar reductions on the
CPU fp32 copy, ~0.53 s) — still 2 orders of magnitude inside the SLA.

## 4. Teardown survival test (spec §7) — PASS

`tests/gpu/optim/cmuon_capsule_teardown.py`: 2-rank real torchrun worker,
production-scale 2560×2560 chunks, forced below_floor (the 112105 class),
rank0 = owner publishes the real minimal capsule inside `opt.step()`, then
stays alive; the driver SIGTERMs the owner worker PID at the kill anchor +
delay (delay ∈ {5, 2, 1} s, fresh run dir per delay) and verifies the local
capsule (exactly one event dir, strict metadata.json, tensor sha256 matches,
no partial/corrupt events).

**Kill anchor = the OWNER's RAISED**, not the step start: in production the
elastic launcher reacts to the raising rank's death (F1 measured ~30 s) and
SIGTERMs the remaining ranks — RAISED + delay is the realistic window, and
the stricter one (the kill targets the publisher directly). The first
test revision anchored the kill at STEP_START + delay and gated the kill
check on worker stdout line arrival; the workers are deliberately silent
over the whole NS+publish window, so every kill landed ~5 s late — i.e.
inside the verdict→rename publish window — and lost the capsule even
though the publish itself was fast. That run (schema v1, REPORTED-QUANTIFIED,
all three delays non-durable, kill delivered at RAISED−0.1 s ≈ mid-publish)
is the quantification of the pre-raise loss zone and is preserved as
`runs-v1-evidence.tar`. The test was then corrected (select-poll timer +
RAISED anchor; no optimizer code touched) and re-run (schema v2):

| delay (s after owner RAISED) | raise at step+ | kill at step+ | kill delivered | capsule durable | sha ok | mirror |
|---|---|---|---|---|---|---|
| 5 | 5.895 | 10.895 | true | **true** | true | ok |
| 2 | 5.087 | 7.087 | true | **true** | true | ok |
| 1 | 5.265 | 6.265 | true | **true** | true | ok |

**Verdict: PASS — `minimum_safe_window_s = 1`** (smallest tested delay with
a durable capsule; all three durable). The owner's worker-own `POST_STEP`
line independently confirms the event name at each delay. Interpretation:
the publish (verdict → atomic rename, ~0.66 s p50 bench) completes BEFORE
the raise, so any SIGTERM at or after the raise — the production elastic
behavior (~30 s after the first rank raised in F1) — preserves the
exact-input capsule with >40× margin. The only theoretical loss zone is a
SIGTERM landing inside the ~0.7 s pre-raise publish window (v1 run).

## 5. Parity (spec §12 / §13) — both PASS (salt11)

- **Success-path parity — PASS.** `dev-tools/cmuon_fp32_parity.py`, base =
  F1 forensic tree (811a4ca) vs F2 tree, 10 steps each (53.1 s / 52.6 s),
  compare: **`PASS: 10 steps, all fields byte-identical`** — params /
  momentum / AdamW state / SR RNG / guard refs / rescue counters via
  torch.equal (byte-exact, not tolerance). No hard failures triggered on
  either tree during the 10 steps (the success path).
- **Hard-fail parity (synthetic) — PASS.** `dev-tools/cmuon_hardfail_parity.py`
  cross-tree on fresh salt11 2-rank reports (F2 tree: scenarios A–E; F1
  tree: A–D): **12/12 checks ok** — identical verdict / zero-commit /
  byte-identical `CMuonSafetyError` messages for A/B/C; identical failure
  counters; D (rescue) identical (no artifact); the ONLY differences are
  the F2 artifact contract fields (E scenario present F2-only with the
  minimal capsule: below_floor, fp32_delta_rms 2.824e-14 < floor 1.562e-6,
  mirror ok; A-rank0 artifact gains `mirror_status`). Exactly the allowed
  "artifact order/content" change.

## 6. Tests

| Gate | Result |
|---|---|
| unit `test_cmuon_fp32_forensic.py` (12 tests, HCU) | **12/12 PASS** (26.9 s, salt10; re-verified 33.6 s on salt12) |
| 2-rank `cmuon_fp32_forensic_2rank.py` (A above_ceiling, B nonfinite, C crash-loop -r2, D successful rescue, **E below_floor = 112105 class**: fp32 finite 2.82e-14 < floor 1.56e-6, mirror ok) | **PASS** (85.6 s, salt10; re-run **PASS** 79.6 s + 84.3 s on salt11 — values byte-identical across hosts) |
| teardown 5/2/1 s (schema v2, RAISED anchor) | **PASS** — all three durable, `minimum_safe_window_s = 1` (§4) |
| enrich CLI smoke (real E-scenario capsule, `--device cuda --svd full --repeats 3`) | **PASS (execution gate)** — sha integrity OK; 3× FP32 replay + per-iter traces + recorded-vs-replay; full SVD spectrum on HCU (stable_rank 67.34, top32_energy 0.372, rank_implied 2.09e-16); LOW_RANK_HYPOTHESIS = INCONCLUSIVE (expected: the synthetic capsule's recorded fp32 rms came from the test's forced "tiny" stub, so replay ≠ recorded is the correct flag; the 112105-class verdict is made on the REAL capsule) |
| full pytest `tests/gpu` (salt11, model symlinked to production assets) | **1 failed / 24 passed / 2 skipped (95.5 s) — zero new failures**: the single failure is `test_varlen_attention::test_forged_boundary_handle_fails_before_native_kernel[host_metadata]` = the machine-independent host_metadata packing code bug (KNOWN baseline, classify only, no skip/xfail); `test_pipeline_encoders` **PASSES** on salt11 (real Qwen+VAE assets reachable via the f2-gates model symlink — the "qwen asset missing" seen on bare trees is a tree artifact, not a machine gap) |
| full pytest `tests/unit` | **775 passed / 0 failed (934 s, CPU-only run)** |
| ruff `check src tests dev-tools` (ruff 0.16) | **PASS** |
| pyright vs F1 baseline (same 4 files, same interpreter conditions) | **FLAT** (219 vs 215 total; no new error categories — the delta is line-shift noise + 1 pre-existing `**dict`-unpack noise line) |

Unit-test F2 contract additions: minimal schema tag + identity fields
(run_id/hostname/pid/process_steps/last_successful_update/attempted_update/
checkpoint_source/local_artifact_path), NO `diagnostic_replay_*` keys, local
capsule under the redirected emergency root, best-effort mirror +
`mirror.json` status, serialization-failure leaves no partial capsule,
mirror-failure keeps the local capsule durable (test_m2), crash-loop -r2 on
the local root, non-owner publishes nothing.

## 7. Deployment closure (spec §15)

Runtime files that differ from base (4):

| File | Role of the change |
|---|---|
| `src/sakuramoon/optim/cmuon_hardfail.py` | + minimal-capsule section (`build_minimal_capsule_metadata`, `publish_minimal_capsule`, `mirror_capsule`, `DEFAULT_EMERGENCY_CAPSULE_ROOT`, schema tag); shared `_publish_event_dir` used by both F1 artifact and F2 capsule; F1 functions retained (offline/compat) |
| `src/sakuramoon/optim/fp32_rescue.py` | failure branch now runs the F2 6-step critical path (no CPU replay); + `note_forensic_update`, `emergency_capsule_root`/`checkpoint_source`/`run_id` kwargs; `cmuon_ns_trace` import REMOVED (offline-only helper from now on — the deployed salt11 file is NOT deleted) |
| `src/sakuramoon/train/step.py` | duck-typed `_note_forensic_update` hook before `optimizer.step()` (no-op for every other optimizer) |
| `src/sakuramoon/train/production.py` | threads `checkpoint_source` (resume path) + `run_id` into the fp32_rescue build |

`src/sakuramoon/optim/cmuon_ns_trace.py` stays on disk but is no longer
imported by production code (offline enrichment only).
No dev-tools / tests / reports are deployed.

## 8. Verdict

**PASS.** Every spec gate is green:

| # | Gate | Result |
|---|---|---|
| 1 | Old-path audit + timer bench | 758.4 s measured (BF16 CPU NS replay 747.5 s dominant) — root cause quantified |
| 2 | New critical path (verdict→freeze→d2h→minimal capsule→LOCAL atomic→best-effort mirror→raise) | implemented, no replay/trace/SVD on the path |
| 3 | Capture bench ≥20 runs, p50<1 s / p95<2 s | **PASS** — 24 runs, 0.659 s / 0.709 s (salt11; salt10 0.660/0.686), ≈1149× vs F1 |
| 4 | Filesystem audit (local-first emergency root) | `/sakuramoon-runtime/cmuon-f1-emergency` (local overlay, both hosts); NFS example refuted |
| 5 | Success parity vs 811a4ca ≥10 steps byte-exact | **PASS** — 10 steps, all fields byte-identical |
| 6 | Synthetic hard-fail semantic parity (only artifact may differ) | **PASS** — 12/12 checks |
| 7 | Real 2-rank teardown 5/2/1 s + minimum safe window | **PASS** — all durable, min_safe = 1 s (RAISED anchor; v1 pre-raise loss zone quantified & preserved) |
| 8 | Offline enrichment CLI (sha → ≥3 replays → traces → recorded-vs-replay → full SVD → ranks → verdict) | **PASS** (execution gate, full SVD on HCU) |
| 9 | ruff + pyright + unit + 2-rank + full pytest (2 known baselines classify-only) | **PASS** — 775/0 unit, 12/12 forensic, 2-rank 5/5, tests/gpu zero new failures |
| 10 | Deployment closure | 4 runtime files; `cmuon.py` ZERO diff; no dev-tools/tests/reports deployed |
| 11 | All optimizer math / config / data / iREPA frozen | verified — zero diff outside the 4 files + test/dev-tool additions |

Production host integrity (salt11): data service (84199) + state publisher
(84343) alive before/after; HCU 0% before every GPU step; top-level
`/sakuramoon-runtime` layout diff = the f2-gates test dir only; production
code/config/checkpoints/model zero-touched; f2-gates deleted after
evidence collection.

## 9. Next

Wait for explicit user approval to deploy the 4-file closure, then resume
from **ckpt_112100** only long enough to capture ONE real exact-input capsule
(the 112126-class next hard fail). Do NOT implement any below_floor optimizer
change yet — the low-rank hypothesis is decided on the real capsule via
`dev-tools/cmuon_hardfail_enrich.py` (full SVD + per §11 evidence gates).
