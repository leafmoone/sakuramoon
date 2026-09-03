# iREPA Phase 4 — representation-alignment training graph audit

- branch: `iprea`, parent `14c9cee` (iREPA phase 3 evidence)
- generated: 2026-09-03
- host: salt10 (2x BW DCU, DTK 26.04, system torch 2.9.0+das.opt1.dtk2604, Python 3.11.9)
- scope: slot_08 stable capture + projector (DDP TrainableComposite) + FP32 spatial
  z-score + FP32 per-sample cosine alignment + successful-update λ schedule +
  JLT/iREPA combined objective + telemetry v11 + full regression parity.

## Verdict

```
IPREA_P4_TEST_PARITY = PASS
iREPA P4 = ZERO NEW FAILURES vs dev (iprea-only failures = 0)
PYRIGHT NEW-in-src = 0
```

## Gate results (salt10, DTK HCU)

| step | result |
|------|--------|
| [1] ruff `src tests` | All checks passed |
| [2] pyright NEW-in-src (vs dev baseline) | **0** (production src clean) |
| [3] frozen check (flow/cmuon/guarded/fp32/save) | CLEAN (untouched) |
| [4] new Phase 4 unit tests | 59 passed |
| [5] new Phase 4 GPU training-graph tests | 13 passed |
| [6] full `pytest tests/` (iprea-P4) | 2 failed, 992 passed, 2 skipped (2 = known baseline, shared with dev) |
| [7] full `pytest tests/` (dev BASE) | 2 failed, 786 passed, 2 skipped |

iprea-only failures = 0, base-only failures = 0. The 2 shared failures are
pre-existing on dev (non-blocking):

1. `tests/gpu/data/test_pipeline_encoders.py::test_real_pipeline_qwen_and_mage_encode_one_batch`
   — Qwen weights missing on salt10 (asset gap, not a code defect).
2. `tests/gpu/fa4/test_varlen_attention.py::test_forged_boundary_handle_fails_before_native_kernel[host_metadata]`
   — dev's own exception-contract bug (`ValueError` where `TypeError` is expected),
   classified DEV BUG / NON-BLOCKING (fixed separately on `fix/fa4-host-metadata-exception`).

---

## A. Slot-08 stable capture

The slot_08 capture is **exactly the image span of the tapped (slot_08) stable
block's output**, read at an **eager outer-loop point** inside
`DenseDiT.forward_features` / `PackedDiT.forward_packed_features` — never a
forward hook, never inside a compiled or activation-checkpointed block. It is
the raw joint hidden state, image tokens only, row-major raster (`T = H*W`),
no pad-to-square, no resize.

Key constraint discovered: slot 8 is a **G1 growth slot** (active only at depth
20/24, not the base depth 16). The capture validates the tap slot against the
active slots at the current depth.

Verified by:
- `test_slot08_capture_is_tapped_block_output` — capture is bitwise the image
  span of the block output (cross-checked against a test-only forward hook).
- `test_slot08_capture_token_count_is_row_major_grid` — `T == H*W` for square
  and rectangular grids at depth 20/24.
- `test_slot08_capture_dense_vs_packed_parity` — the **full production model**
  (PackedDiT is locked to d=2560, 20Q/5KV, head_dim=128, bfloat16) at depth 20:
  identical weights + inputs yield the same slot_08 capture through the dense
  (SDPA) and packed (FA4 varlen) backends.

## B. Projector (IRepaAlignment)

`IRepaAlignment(in_channels)` is a single locked `MixedPrecisionConv2d`
(bfloat16 weights / float32 bias): `out_channels=768`, `kernel=3`, `stride=1`,
`padding=1`, `dilation=1`, `groups=1`, `bias=True`. It consumes a bfloat16
image hidden state `[B,T,D]` plus `image_shape (H,W)` (with `T == H*W`),
reshapes to a spatial tensor, convolves, and flattens to `[B,T,768]`. 768
matches the frozen PE-Spatial teacher feature width. Guards: `TypeError` for a
non-bfloat16 hidden state, `ValueError` for `H*W != T` or a width mismatch.

Verified by: `tests/unit/model/test_irepa_alignment.py` (locked config, dtype
guards, shape contract, bf16-in → 768-out).

## C. FP32 spatial z-score teacher target

`spatial_zscore_target(teacher_features, *, gamma, eps)` computes a per-channel
spatial z-score of the frozen PE-Spatial teacher's `[B,T,768]` bfloat16 patch
features **in float32** (mean/variance over the spatial/token axis, per
channel; `(x - mean) / sqrt(var + eps) * gamma`). Output is float32
`[B,T,768]`.

Verified by: `tests/unit/objective/test_irepa_objective.py` (z-score math,
float32 output, gamma/eps) and the GPU teacher chain (real teacher → z-score).

## D. FP32 per-sample cosine alignment loss

`irepa_alignment_loss(student_features, target)` upcasts both to float32,
computes `F.cosine_similarity(dim=-1)` per token, and returns
`IRepaAlignmentLossOutput(per_sample, cosine_per_sample)` with
`per_sample = (1 - cosine).mean(1)` (float32, one value per sample).

Verified by: `tests/unit/objective/test_irepa_objective.py` (cosine math,
per-sample reduction, float32) and the GPU teacher chain (finite loss, cosine
in `[-1, 1]`).

## E. Successful-update λ schedule

`IRepaLambdaSchedule(start_successful_update, target_weight, ramp_in_updates,
ramp_out_after_updates, ramp_out_updates)` + `irepa_weight_for_update(*,
successful_update, ...)`: a **stateless, deterministic** function of the
successful-update count — ramps 0 → `target_weight` over `ramp_in_updates`
starting at `start_successful_update`, holds, then ramps → 0 over
`ramp_out_updates` after `ramp_out_after_updates`. No internal state;
recomputed per update.

Verified by: `tests/unit/objective/test_irepa_objective.py` (schedule math,
determinism, statelessness, ramp boundaries).

## F. JLT/iREPA combined objective (no λ=0 skip)

Per-sample combined loss = `main_per_sample + λ · irepa_per_sample` (float32).
**§14: no λ=0 skip** — at λ=0 the full iREPA graph (capture → projector →
z-score → cosine) still executes and contributes an **exact-zero** tensor to
the loss and an **exact-zero** gradient; it is never skipped.

Verified by: `tests/unit/train/test_irepa_zero_impact.py` (disabled =
byte-identical legacy; enabled λ=0 = exact-zero contribution, grad present but
zero) and `test_lambda_zero_is_exact_zero_contribution` (λ=0 on DCU).

## G. Telemetry v11

`TRAINING_METRIC_SCHEMA_VERSION = 11`. Two new timing phases
(`irepa_teacher`, `irepa_projector`) in `CORE_TIMING_PHASES`/`TIMING_PHASES`.
Six new iREPA fields in `TrainingMetric` (main_loss split + `irepa_loss`,
`irepa_weighted_loss`, `irepa_cosine_mean`, `irepa_lambda`,
`irepa_projector_grad_norm`). The t-bin histogram stays **MAIN-JLT-only**
(uses `main_per_sample_loss`, not the combined). The observer re-derives
`main + weighted == combined` **bit-exactly in float32** and enforces
one-update λ uniformity and finiteness. The json + wandb payloads publish the
six iREPA fields.

Verified by: `tests/unit/test_irepa_metrics_v11.py` (11 tests: schema, phases,
t-bin main-only, split re-derivation, non-uniform-λ rejection, shape-mismatch
rejection, payload, impossible-value rejection) and the schema-11 updates in
`test_telemetry_t_bins` / `test_telemetry_spatial` / `test_g1_telemetry`.

## H. Zero-impact contract

Disabled (`irepa_alignment is None`): `TrainableComposite.forward` returns the
exact legacy tuple (byte-identical, no iREPA graph). Enabled with λ=0: the iREPA
graph is present but contributes exactly zero to the loss and gradient. Legacy
gradients are `torch.equal` to the no-iREPA run (§15).

Verified by: `tests/unit/train/test_irepa_zero_impact.py` (4 tests).

## I. Production gate (fail-closed)

The production gate is **fail-closed** (§26): when iREPA is enabled but a
prerequisite is missing/inconsistent (teacher asset, projector, tap slot,
...), the gate raises a blocker instead of silently degrading.

Verified by: `tests/unit/train/test_irepa_production_gate.py`.

## J. Frozen optimizer + full regression parity

Frozen (zero P4 diff): `objective/flow.py`, `optim/cmuon.py`,
`optim/guarded_canonical.py`, `optim/fp32_rescue.py`, `checkpoint/save.py`
(v4 gate), `train/preflight.py`. The gate frozen check confirms none are
modified by P4.

Full regression parity (salt10 DCU, DTK): iprea-P4 `pytest tests/` vs the dev
baseline → **zero new failures** (iprea-only = 0, base-only = 0). The 2 shared
baseline failures (qwen asset gap; dev host_metadata exception bug) are
pre-existing on dev and non-blocking. No skip/xfail was added; the 2 shared
failures are unchanged.

## Test-isolation note

The compile-stability GPU test (`torch.compile(dit.forward_tapped)`) traces
through the attention module and injects torch.dynamo `__compiled_fn_*`
namespace artifacts that a later unit test
(`test_distributed_compile.test_fa2_is_the_only_explicit_eager_compiler_
boundary`) asserts on. The GPU test file therefore carries an autouse fixture
that removes `__compiled_fn_*` artifacts from `sakuramoon` modules before and
after each test, keeping the session isolated (no production-code change).

## Files (Phase 4 functional commit)

- new: `src/sakuramoon/objective/irepa.py`, `src/sakuramoon/train/irepa_diagnostics.py`
- new tests: `tests/gpu/irepa/test_irepa_training_graph.py`,
  `tests/unit/objective/test_irepa_objective.py`, `tests/unit/test_irepa_metrics_v11.py`,
  `tests/unit/train/test_irepa_lambda_binding.py`, `tests/unit/train/test_irepa_shadow_audit.py`,
  `tests/unit/train/test_irepa_zero_impact.py`
- modified: `src/sakuramoon/config/assembly.py`, `src/sakuramoon/model/dit.py`,
  `src/sakuramoon/telemetry/metrics.py`, `src/sakuramoon/telemetry/observer.py`,
  `src/sakuramoon/train/loop.py`, `src/sakuramoon/train/production.py`,
  `src/sakuramoon/train/runtime.py`, `src/sakuramoon/train/step.py`,
  `tests/unit/test_g1_telemetry.py`, `tests/unit/test_telemetry_spatial.py`,
  `tests/unit/test_telemetry_t_bins.py`, `tests/unit/train/test_irepa_composite.py`,
  `tests/unit/train/test_irepa_production_gate.py`
