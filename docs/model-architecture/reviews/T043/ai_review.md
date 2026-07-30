# T043 AI review

Status: PASS for the implemented single-GPU scope after remediation acceptance. The
independent review below originally found two blockers and one evidence gap. Direct
reviewer restart did not return a task name, so the main agent performed the required
remediation validation under the user's continue-without-agents instruction.

## Remediation acceptance

- Raw growth state now persists stage/world-size/resolution plus ramp origin and
  duration, and strictly recomputes alpha from successful updates. Both growth paths
  prove midpoint fresh-load next-update equality with uninterrupted scheduling.
- The CLI reads checksum-verified raw continuation state, while migration validates
  the same canonical stage axes. S0-as-S1 and G1-as-S2 are rejected.
- The dense alpha-zero test is parametrized over both 16-to-20 and 20-to-24.
- Final validation: 46 targeted CPU tests, 340 full CPU tests, 14 real one-GPU
  checkpoint tests, Ruff, strict Pyright, traceability and import-order all passed.

## Initial blocking findings (remediated)

1. Ramp checkpoints do not contain enough state to resume the half-cosine schedule. `GrowthProgress` requires both `start_successful_update` and `ramp_updates` (`src/sakuramoon/train/stage.py:142-171`), but `GrowthCheckpointState` persists only `active_slot_ids` and the current floating-point `alpha` (`src/sakuramoon/checkpoint/schema.py:47-55`, `242-260`). No production code reconstructs `GrowthProgress` from a loaded raw checkpoint or binds the external transition plan to that checkpoint. The GPU helper at `tests/gpu/checkpoint/test_growth_migration.py:128-147` merely writes an arbitrary alpha and checks that the same value loads; it never compares the first resumed alpha/update with an uninterrupted schedule. A midpoint raw checkpoint therefore cannot independently determine the next alpha, so the claimed midpoint/end restore contract is not implemented. Persist and strictly validate the ramp origin and duration (or an equivalent exact progress representation) in the raw continuation state, then prove fresh-load next-successful-update equality at midpoint for both 16-to-20 and 20-to-24. This arithmetic/restore test does not require a 1,000-update long run.

2. The source checkpoint is not bound to the requested predecessor stage. `StageTransitionRequest` validates only the caller-provided source/target names (`src/sakuramoon/train/stage.py:57-73`), while `transition_plan` reads only raw kind and checkpoint ID and then copies the caller's stage label (`src/sakuramoon/cli/transition.py:17-49`). The raw continuation schema has no stage, resolution or world-size identity. `migrate_loaded_growth` verifies only the actual depth pair (`src/sakuramoon/checkpoint/migrate.py:82-90`), which cannot distinguish S0 from S1 at depth 16 or G1 from S2 at depth 20. Consequently a G1 20-layer/256 raw can be labelled S2 and accepted for G2, bypassing the mandatory S2 resolution stage. Bind a trusted canonical stage identity (including the primary axes, or a stage-resolved config identity that is validated rather than supplied as an opaque matching value) to the raw checkpoint and reject mismatches before mutating the target. Add negative tests for S0-as-S1 and G1-as-S2. Four-rank execution may remain pending, but the CPU validation contract must already reject the wrong predecessor.

3. The alpha-zero functional-equivalence evidence covers only 16-to-20. `tests/unit/model/test_dit.py:135-152` compares a 16-layer source with a 20-layer target; there is no corresponding 20-to-24 comparison. The parametrized GPU migration test covers parameter/state copying for both transitions but performs no forward equivalence check. This is insufficient for `OPEN-075`, which is currently marked implemented. Parameterize the functional test over both growth transitions and compare the complete applicable outputs/features after exact preserved-FQN migration.

## Verified behavior

The stable slot sets and new-slot allowlists are consistent with 16-to-20-to-24 growth. The migration validates the model and optimizer FQN deltas before applying them, preserves initialized old TorchAO moments and optimizer SR RNG, and leaves target new-slot optimizer state lazy. The pure ramp function is monotonic with exact endpoints, and stage readiness has no automatic transition side effect. Existing single-GPU migration evidence for both depth changes was inspected without rerunning GPU work concurrently with the Infra reviewer.

The independent reviewer ran the 18 targeted CPU tests, Ruff, strict Pyright, traceability verification and `git diff --check`; all passed. These machine checks do not close the findings above.

Formal 1,000-5,000-update G1/G2 ramps, post-ramp stability and four-rank model/optimizer equality remain blocked by the no-long-run scope and unavailable 4x RTX 5090 hardware. No single-GPU evidence is used to close those gates.
