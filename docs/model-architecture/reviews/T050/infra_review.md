# T050 independent Infra/performance review

Status: **CHANGES REQUIRED**

Scope: current uncommitted T050 implementation and
`docs/model-architecture/reviews/T050/test_report.json`. No GPU workload was run.

## Findings

1. **High - the accepted-preflight capability does not prove that production checks
   ran or bind the checked identity to the training invocation.**
   `src/sakuramoon/train/preflight.py:123` accepts any public mapping with the expected
   names, registers the resulting handle at line 142, and exports that runner through
   `src/sakuramoon/train/__init__.py:63`. `run_single_gpu_training` only checks WeakSet
   membership at `src/sakuramoon/train/runtime.py:388`; the handle/report contains no
   resolved-config, service, device, or module identity. Consequently a caller can run
   thirteen no-op callbacks and then train with a different config/runtime. The test
   helper at `tests/unit/train/test_runtime.py:51` does exactly this and its accepted
   handle reaches successful updates. Fixed callback order is therefore not a
   non-bypassable production preflight. Require an unforgeable builder-produced plan
   and bind the accepted handle to the exact resolved config and checked runtime/data
   identities before accepting it at the training boundary.

2. **High - normal completion and failure do not deterministically stop D025's two
   persistent workers.** `SingleGpuTrainingLoop.run` creates an iterator at
   `src/sakuramoon/train/loop.py:125` but has no outer `finally` that closes a closeable
   iterator. D025's service iterator performs worker stop and DataLoader shutdown only
   in its `finally` at `src/sakuramoon/data/collate.py:705` and
   `src/sakuramoon/data/collate.py:725`. If the caller retains the iterator, reaching
   the stage target or raising from update/post-update leaves workers and active
   leases alive until nondeterministic destruction. Close the owned batch iterator on
   every exit and test both target completion and failure; early close must retain
   outstanding service leases as active and must not ACK them.

3. **Medium - backward/nonfinite failure can be replaced by cleanup failure.** At
   `src/sakuramoon/train/step.py:196` and `src/sakuramoon/train/step.py:203`, the
   original failure calls `_abort_pending_update`; that method marks the step failed
   before calling `zero_grad` at line 223. If `zero_grad` raises, the original
   nonfinite/autograd exception is never raised, and the loop's subsequent `abort()`
   is a no-op. Preserve both errors in an `ExceptionGroup`, as the optimizer failure
   path already does, and add targeted tests for nonfinite and backward failure plus
   cleanup failure.

4. **Medium - the production loop does not connect the implemented phase timers.**
   `SingleGpuBatchRuntime.measure` can time Qwen, VAE, conditioning, DiT and loss, and
   `SingleGpuStep.finish_update` can time clip/optimizer/zero-grad, but
   `src/sakuramoon/train/runtime.py:415` calls `measure` without a timer and
   `src/sakuramoon/train/loop.py:139` calls `finish_update` without one. The benchmark
   adapter has timing, but the T050 training entry cannot emit mandatory per-update
   phase observations. Wire a caller-owned `PhaseTimer`/telemetry sink through the
   production loop without boundary synchronizations, or keep T050 explicitly open
   until that integration is provided.

## Verified

- Targeted CPU selection: 53 passed, 17 warnings in 21.57 seconds.
- Ruff passed for the changed T050 production, unit, CLI, and GPU-test files.
- Strict Pyright passed for the same selection: 0 errors and 0 warnings.
- The D025-only loader-control surface is otherwise respected: T050 has no
  `service_batches` helper and exposes no batch/worker/queue/pin/drop-last controls.
- Exact hardware checks reject topology, GPU name, compute capability, CUDA 12.8,
  driver drift, memory floor, CPU/RAM/swap drift, and force bypass. The capacity check
  requires a measured positive raw-checkpoint size and three checkpoint copies.
- Read-only mount verification resolved the repository to NFSv3
  (`cs1.vast1.bz1.paratera.com:...`), while `_require_nvme` accepts only ext4/xfs on a
  `/dev/nvme...` source. This environment therefore correctly remains a hard NVMe
  preflight failure; package-level NVML repair cannot change that storage result.
- Checkpoint cadence tests show the proposed cadence is passed to durable publication
  and committed only after callback success. The CLI remains fail-closed and does not
  invent the missing Text/Style constructor values.

## Residual blocked gates

The recorded bounded RTX 5090 compute smoke was not rerun in this independent review.
It covers local Qwen/Mage-VAE, synthetic batch input, 16-layer DiT, loss, backward,
clip, and one optimizer update; it is not a production CLI, preflight, real-service,
or checkpoint round-trip run. Production CLI assembly, healthy exact driver/userspace
identity, real local NVMe capacity, formal S0/1,000 updates, sustained throughput and
memory, DDP/NCCL, four-GPU correctness, and formal stage claims remain blocked.
