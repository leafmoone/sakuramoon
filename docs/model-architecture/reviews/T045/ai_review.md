# T045 AI/model-correctness rereview

Verdict: **PASS** for the T045 implementation scope.

The raw schema remains internally coherent: raw manifests, trainer state and
growth state use schema v3; raw v1/v2 artifacts are rejected; model-only, PMA
and release artifacts remain schema v1. `RawCheckpointState` binds trainer
update to durable cadence, constrains the update to the absolute stage
interval, binds an active growth ramp to the stage origin, and recomputes the
exact half-cosine alpha. The 16-to-20 and 20-to-24 migration tests preserve old
model/optimizer state and exercise post-transition, midpoint and end round
trips.

## Rereviewed Controls

- `run_single_gpu_training` now forwards the three-argument
  `checkpoint_cadence_event`; runtime tests verify that the exact proposed
  cadence is exposed before commit and that a failed publication leaves the
  restored anchor unchanged (`src/sakuramoon/train/runtime.py:416-482`,
  `tests/unit/train/test_runtime.py:278-374`).
- The runtime consumes `StageBudgetCheckpointState`, uses its absolute terminal
  update, and rejects trainer-state or resolved-config drift before constructing
  the loop (`src/sakuramoon/train/runtime.py:443-459`). Negative tests cover both
  drift directions.
- `checkpoint_reason` provides an exhaustive, exact-type mapping from every
  `ForcedCheckpoint` value to `CheckpointReason`, and tests reject a foreign
  equal-string enum (`src/sakuramoon/train/stage.py:51-64`,
  `tests/unit/train/test_stage.py:107-127`). Scheduler tests cover all mapped
  transition reasons and pre-decay ordering.

## Verification

The reviewer ran the targeted training CPU selection and obtained 38 passed,
then the checkpoint/transition CPU selection and obtained 52 passed. The full
targeted CUDA checkpoint directory previously obtained 25 passed on the visible
RTX 5090, including production-size raw save/load and both growth migrations.
Ruff, strict Pyright, `git diff --check`, and the traceability verifier passed.
No four-GPU or formal-stage gate is claimed.
