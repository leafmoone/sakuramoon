# T043 Infra review

Status: PASS for the implemented single-GPU scope after remediation acceptance. The
independent review below originally found three blockers. Direct reviewer restart did
not return a task name, so the main agent performed the required remediation
validation under the user's continue-without-agents instruction.

## Remediation acceptance

- Migration now applies strict optimizer-to-module object coverage checks to both
  source and target before target mutation; detached source and target cases fail.
- New block subtrees use delimited prefix matching and conditioner biases use exact
  FQN matching; `slot_02evil` is rejected without mutation.
- Plan publication uses an atomic hard-link no-clobber protocol. It retains an open
  descriptor through both directory fsyncs so rollback removes only the publishing
  writer's inode. Concurrent writers and foreign-inode replacement are tested.
- Final validation: 46 targeted CPU tests, 340 full CPU tests, 14 real one-GPU
  checkpoint tests, Ruff, strict Pyright, traceability and import-order all passed.

## Initial blocking findings (remediated)

### INFRA-001: migration does not bind either optimizer to its model

`migrate_loaded_growth()` compares optimizer audit rows by canonical name, group,
shape, and dtype, but never verifies that the audited `Parameter` objects are the
exact parameters owned by `source_module` and `target_module`. A real CUDA
reproduction passed a 20-layer optimizer built for a different 20-layer
composite: migration returned successfully, copied model tensors into the
requested target, and installed optimizer state on the detached composite. A
subsequent optimizer step would therefore not update the migrated target.

Before any target mutation, call the existing strict optimizer-coverage check
for both source and target module/optimizer pairs. Add negative tests for a
same-topology source optimizer and target optimizer owned by different module
instances, and prove model parameters, optimizer state, and SR RNG remain
unchanged on rejection. This applies at
`src/sakuramoon/checkpoint/migrate.py:91-134`; the existing loader protects its
locally constructed source pair, but it does not protect the caller-supplied
target pair or direct callers of `migrate_loaded_growth()`.

### INFRA-002: the new-slot allowlist accepts prefix collisions

`new_slot_fqn_prefixes()` returns the exact conditioner-bias FQN without a
delimiter, while `_is_new_fqn()` applies `str.startswith()` to every entry. A
real CUDA reproduction added
`dit.conditioner.block_biases.slot_02evil`; the malformed parameter survived
the canonical-composite checks, was accepted as a new slot by both the model
and optimizer delta checks, and migration returned success. This violates the
task's predefined new-slot-only boundary.

Represent block subtrees as delimited prefixes and conditioner biases as exact
FQNs, or otherwise use an exact/prefix-aware matcher. Add a negative test for
the collision and retain positive coverage for every legitimate block tensor
and exact bias FQN. The affected logic is
`src/sakuramoon/model/growth.py:66-76` and
`src/sakuramoon/checkpoint/migrate.py:53-54`.

### INFRA-003: transition-plan publication can overwrite another writer

`write_transition_plan()` checks `path.exists()` and later publishes with
unconditional `os.replace()`. The check and replace are not one atomic
no-clobber operation. A synchronized two-writer reproduction made both calls
return successfully while the second replace silently discarded one valid
plan. The `published` flag also does not establish inode ownership, so the
parent-fsync exception path can unlink a plan that is no longer this writer's
file.

Publish with a same-directory atomic no-replace primitive (for example, a
hard-link based no-clobber protocol with inode-aware rollback, or
`RENAME_NOREPLACE`) and fsync the directory after the final namespace state.
Add a deterministic concurrent-writer test asserting exactly one success, one
`FileExistsError`, no overwrite, no leaked temporary, and rollback limited to
the failing writer's inode. The affected code is
`src/sakuramoon/cli/transition.py:52-75`.

## Verified behavior

Apart from the findings above, the implemented scope is fail-closed and
consistent with the current single-GPU boundary. The stage graph has a unique
predecessor and changes exactly one primary axis; H1/H2 remain disabled;
transition state preserves global successful-update counters while resetting
shard progress; readiness reporting has no transition side effects. Growth is
successful-update based, uses the specified 2% clamp and half cosine, and
reports the post-transition, midpoint, and end checkpoint points.

The exact raw checkpoint path, identity, kind, checksums, topology, growth
completion state, model FQN delta, optimizer policy, and TorchAO state schema
are validated before the normal migration mutation path. The positive GPU
tests preserve old model tensors and initialized optimizer moments bitwise,
reconstruct `OptimState8bit` without dequantization, preserve optimizer SR RNG,
leave legitimate new slots randomly initialized with empty state, clear shard
progress, and restore alpha 0.0/0.5/1.0 checkpoints for both 16-to-20 and
20-to-24 transitions. The invalid-alpha test leaves the target model,
optimizer, and SR RNG unchanged.

## Independent verification

Commands run by this reviewer:

```text
uv run pytest -q tests/unit/model/test_growth.py tests/unit/train/test_stage.py tests/unit/cli/test_transition.py tests/unit/model/test_dit.py
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.580.105.08 uv run pytest -q tests/gpu/checkpoint/test_growth_migration.py
uv run ruff check <T043 implementation and test paths>
uv run pyright <T043 implementation and test paths>
uv run python tools/verify_traceability.py
git diff --check
```

Results: 18 targeted CPU tests passed in 4.15 s; 2 real one-GPU growth
migrations passed in 11.13 s; Ruff passed; strict Pyright reported 0 errors;
traceability verification passed for 221 requirements, 67 production modules,
and 234 runtime config keys; `git diff --check` passed. Separate CUDA
reproductions confirmed INFRA-001 and INFRA-002, and a synchronized CPU
two-writer reproduction confirmed INFRA-003.

The GPU process used the installed NVML 580.105.08 library as an explicit
review-time preload because the default userspace NVML points to 610.43 while
the loaded driver is 580.105.08. This is test-only and does not waive the
production preflight blocker.

## Pending boundaries

`OPEN-076` correctly remains blocked by `FOUR-GPU-AVAILABLE` and
`NO-LONG-RUN`. Formal 1,000-5,000-update G1/G2 ramps, post-ramp stability,
production-size migration peak memory/RSS, four-rank state equality,
DDP/NCCL behavior, multi-rank publication, checkpoint scheduling/retention,
diagnostic bundles, and formal stage canaries were not run and are not closed
by this review. The downsized one-GPU migration tests establish migration
semantics, not production four-GPU capacity or throughput.
