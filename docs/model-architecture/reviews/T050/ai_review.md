# T050 independent AI/model correctness final review

Reviewer: independent agent `/root/t050_ai_final_review`.

Scope: T050 after stage/growth/checkpoint binding, attention-backend binding, fresh
resume RNG-order, and exact two-worker remediation. The reviewer made no
implementation or evidence edits and did not run a formal stage or multi-GPU job.

## Verdict

**PASS** for the implemented CPU and bounded single-GPU scope. No blocking
AI/model-correctness finding remains.

## Verified controls

- Strict JLT remains FP32 x-to-v with a per-sample element mean and one global sample
  mean across unequal microbatches.
- Nonfinite/backward failures poison the attempt, clear gradients, preserve cleanup
  errors, and never advance successful updates.
- Accepted preflight is process-local and bound to the exact config, D025 stream,
  runtime, Qwen, VAE, trainable composite, optimizer, restored RAW handle, and
  checkpoint publisher.
- Enabled S0 topology and restored stage, world size, resolution, active slots, ramp
  presence, alpha, zero budget origin, budget length, remaining work, and cadence are
  validated before batch consumption.
- Resolved `fa4_varlen` or `dense_sdpa_reference` selection is bound to the assembled
  DiT artifact; the bounded GPU fixture explicitly selects FA4.
- Fresh resume loads Qwen and Mage-VAE before RAW restore, leaving restore as the last
  model-assembly operation that mutates training RNG state before service/update.
- The final GPU artifacts restore successful update 1 and advance to update 2 with
  exact worker IDs `{0,1}` and four distinct child PIDs across initial/resume runs.

## Independent Verification

The reviewer ran the focused runtime and preflight selections: **36 runtime tests and
28 preflight tests passed**. It inspected the bounded GPU artifacts and recorded no
single-GPU model-correctness blocker.

## Residual Gates

- DataLoader worker startup currently has no explicit generator, so worker creation
  advances global CPU RNG. This does not change the checkpointed CUDA training RNG or
  the T050 verdict, but production assembly/formal S0 review must keep it visible.
- The production CLI remains fail-closed pending C002's binding of the
  already-confirmed Text and Style decisions into strict config, overlays, and
  assembly.
- Formal 1,000-update S0, sustained throughput, FID/IS, and every four-GPU DDP/NCCL
  gate remain pending or blocked. The bounded one-GPU result closes none of them.
