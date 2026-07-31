# Data package Infra/performance review

Reviewer: `/root/roadmap_inventory` (independent package reviewer)

Scope: `D010-D016` current CPU durability, boundedness, failure behavior, and the
existing bounded one-GPU engineering evidence. Overall verdict: **CHANGES_REQUIRED**.

## Findings

1. **D012 can report state publication failure after making the new state visible.**
   `src/sakuramoon/data/state.py:121-133` performs `os.replace` before fsyncing the
   parent directory. If that parent fsync fails, the method raises
   `ShardStateError` but neither restores the previous state nor removes the new one.
   A focused fault probe observed `state_visible_after_parent_fsync_failure True` with
   active shard `r/1.tar`. The existing ENOSPC test at
   `tests/fault_injection/test_data_failures.py:221-235` fails the first file fsync, so
   it does not cover the post-replace branch. D012 is **CHANGES_REQUIRED** because a
   caller cannot distinguish a rejected transition from a visible published state.

2. **D010's failed shard publication rollback is not durably published.**
   `src/sakuramoon/data/modelscope.py:344-355` replaces the final shard, fsyncs the
   parent, and unlinks the final file if that fsync fails. It does not fsync the parent
   after the rollback unlink. A crash may therefore preserve either namespace state,
   while the API reports failure. Full shard/cache state is explicitly subject to the
   repository's atomic publication protocol, so D010 is **CHANGES_REQUIRED** with a
   post-replace parent-fsync fault contract.

3. **D013 uses the obsolete no-clobber protocol for a regenerable scan report.**
   `src/sakuramoon/data/image_ops.py:213-270` publishes by hard link and rejects an
   existing destination. `tests/unit/data/test_image_ops.py:203-228` locks that
   behavior. Current repository policy requires regenerable image/metric reports to
   use a same-directory temporary file, file fsync, and `os.replace`. D013 is
   **CHANGES_REQUIRED**; replacement and parent-fsync failure behavior need direct
   contracts.

4. **D012/D015's durable path is serial despite the locked two-worker production
   topology.** `src/sakuramoon/data/collate.py:285-300` hard-fails at any worker count
   other than one. This is a correct fail-closed response to the current single-active-
   shard state model, but it is not completion of the initial two-persistent-worker
   contract in `current/confirmed-decisions.md:151-158`. D012 and D015 remain
   **CHANGES_REQUIRED** until a bounded durable multi-worker state design and focused
   resume/failure tests exist. No throughput conclusion may be inferred from the
   one-worker correctness path.

## Per-task verdicts

| Task | Infra/performance verdict | Evidence boundary |
|---|---|---|
| D010 | CHANGES_REQUIRED | Streamed file verification is bounded, but rollback after parent-fsync failure is not durable; live network evidence remains pending. |
| D011 | PASS | CPU selection/bundle publication contracts are bounded; production 11M scan and real bundle publication remain pending. |
| D012 | CHANGES_REQUIRED | Post-replace state failure is ambiguous and durable two-worker coordination is missing; cold-cache/NVMe failure evidence remains pending. |
| D013 | CHANGES_REQUIRED | Streaming counters are bounded, but regenerable report publication violates current policy; production scans remain pending. |
| D014 | PASS | CPU caption/serializer work is bounded and outside the model hot path; production 100k component distribution remains pending evidence, not a throughput pass. |
| D015 | CHANGES_REQUIRED | Queue/fragments are bounded, but the durable production path cannot run the locked two-worker topology; cold-cache throughput/RSS/ready-wait evidence remains pending. |
| D016 | PASS | Strict config binding has no material runtime performance risk; it makes no production scan claim. |

## Validation and boundaries

The independent CPU package suite passed 285 tests with 8 warnings in 23.47 seconds
under `CUDA_VISIBLE_DEVICES=` and `uv run --frozen`. Ruff passed; Pyright reported
0 errors and 0 warnings. Static success does not discharge the missing post-replace
fault branches or production topology.

No production dataset/network/NVMe sweep, GPU run, long training, DDP, NCCL,
multi-GPU validation, 1,000-step canary, or formal stage was performed. The existing
D015 one-GPU Qwen/VAE engineering smoke remains component/pipeline evidence only; it
does not establish Data throughput, durable multi-worker recovery, four-GPU behavior,
or any formal stage gate. Required production scans, cold-cache throughput, ready-wait,
RSS/swap, disk-full, concurrent coordinator, and replay audit evidence remain pending.
This review did not read `.env` or `reference/` and made no network call.
