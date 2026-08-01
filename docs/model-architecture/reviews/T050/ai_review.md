# T050 independent AI/model correctness review

Reviewer: independent agent `/root/t050_ai_review`.

Scope: frozen uncommitted T050 work on committed base
`016b5263a0add818a1a9bd5efae5b3cdbf406e35`. The unrelated roadmap edit and
concurrent Infra-review workspace were excluded. No implementation, test, task,
trace, archive, reference, or environment-secret file was modified by this review.

## Verdict

**CHANGES REQUIRED.** The strict JLT sample-mean math, optimizer-success accounting,
checkpoint cadence commit ordering, local Qwen/Mage-VAE use, and packed DiT contracts
are coherent for the exercised component path. T050 cannot close because the accepted
preflight can be legitimately issued for caller-supplied no-op checks, the claimed
D025-only batch provenance is not enforced or exercised, and the required DATA
package dependency currently has a reproduced persistent-worker deadlock blocker.

## Findings

### High: the accepted preflight is bypassable through its public issuer

`run_single_gpu_preflight` accepts an arbitrary public mapping and issues an
`AcceptedPreflight` whenever callbacks with the thirteen expected names return
success (`src/sakuramoon/train/preflight.py:123-142`). It does not require the mapping
to come from `build_single_gpu_preflight_checks`, and the accepted report contains no
resolved-config, data-session, encoder, trainable-module, or checkpoint identity.
Both the arbitrary-map runner and concrete builder are publicly exported
(`src/sakuramoon/train/__init__.py:9-17`, `:60-64`).

The runtime tests demonstrate the bypass directly: `_accepted` obtains a registered
handle from thirteen no-op lambdas, and that handle authorizes training-loop tests
(`tests/unit/train/test_runtime.py:51-55`, `:168-183`). Rejecting
`object.__new__(AcceptedPreflight)` only prevents one forgery mechanism; it does not
make the mandatory checks non-bypassable. A handle from a real run is also transferable
to different config/runtime objects because `run_single_gpu_training` checks only
WeakSet membership, S0 topology, and `module is runtime.composite`
(`src/sakuramoon/train/runtime.py:386-411`).

Required remediation: make production acceptance issuable only from a capability
created by the concrete builder, bind it to the exact resolved identity and checked
runtime/resources, and reject arbitrary callback maps at the training boundary. Keep
callback injection, if needed, behind a test-only/internal orchestrator rather than the
public production issuer. Add negative tests for no-op issuance and cross-config,
cross-client, cross-encoder, and cross-module handle reuse.

### High: the documented D025-only input boundary is not established

`run_single_gpu_training` accepts a plain `Iterator[TrainingBatch]`; it receives no
factory-issued identity or accepted iterator handle (`src/sakuramoon/train/runtime.py:363-383`).
`TrainingBatch` is a publicly constructible dataclass, and `_require_batch` validates
its type and selected shapes but not D025 provenance (`src/sakuramoon/data/collate.py:42-62`,
`src/sakuramoon/train/runtime.py:113-131`). The boundary test checks only that loader
control parameter names are absent from the function signature
(`tests/unit/train/test_runtime.py:77-89`).

The one-GPU test constructs a synthetic `TrainingBatch` directly and calls
`runtime.measure` plus `SingleGpuStep`; it does not call
`ProductionPipelineFactory.batches`, accepted preflight, `run_single_gpu_training`,
the scheduler/cadence path, or checkpoint publication
(`tests/gpu/train/test_single_gpu_full_chain.py:91-112`, `:147-160`). This is valid
real-Qwen/Mage/DiT/optimizer component evidence, but it does not prove the task/report
claim that T050 accepts only batches assembled by D025 or that a production full chain
ran (`docs/model-architecture/progress/tasks/T050.md:50-60`,
`docs/model-architecture/reviews/T050/implementation_report.md:16-30`).

Required remediation: either enforce a factory-issued accepted batch-stream boundary,
or narrow the contract to a typed batch boundary and prove that the only production
entry constructs it through D025. Then run a bounded governed factory -> two-worker
service -> runtime -> loop -> durable checkpoint integration after the DATA blocker
below is fixed. Label the current GPU result as a synthetic-batch component smoke, not
a full-chain smoke.

### High dependency blocker: DATA package review is not closed

The current independent DATA Infra review reproduced the implicit-default-`fork`
persistent-worker hang twice and marks D023 failed, with D024/D025 inheriting the
blocker (`docs/model-architecture/reviews/DATA/d025_infra_review.md:9-49`, `:67-70`).
The DATA AI review therefore explicitly withholds package acceptance and also records
the exact governed parser/Qwen/Mage real-shard run as pending
(`docs/model-architecture/reviews/DATA/d025_ai_review.md:10-12`, `:32-36`, `:78-79`).
T050's cited adjacent worker smoke cannot override this later independent failure.

This is upstream remediation rather than a T050 numerical defect, but the roadmap
requires the DATA review to close before T050. Fix D023 with one explicit compatible
multiprocessing context shared by DataLoader and all queues, add bounded startup and
progress failure tests, and obtain the required DATA rereview before rerunning the
T050 integration evidence.

### Medium: cleanup failure can erase the original update failure

For a nonfinite per-sample loss or a backward exception, `SingleGpuStep.backward`
calls `_abort_pending_update`; that method marks the step failed and then calls
`zero_grad` without preserving a cleanup exception alongside the original failure
(`src/sakuramoon/train/step.py:196-205`, `:215-223`). If `zero_grad` raises, the
original nonfinite/backward error is replaced. The loop's later `step.abort()` is a
no-op because the step is already failed, so diagnostics record only the cleanup error
(`src/sakuramoon/train/loop.py:140-152`). The optimizer/clip path correctly uses an
`ExceptionGroup` for the analogous collision (`src/sakuramoon/train/step.py:255-265`).

Preserve both errors consistently and add nonfinite-plus-cleanup and
backward-plus-cleanup tests. Attempted/successful/effective-sample counters must remain
exact in both cases.

## Accepted controls

- `flow_matching_loss` computes FP32 velocity error, takes a per-sample element mean,
  and only then a batch mean (`src/sakuramoon/objective/flow.py:168-185`). Runtime
  preserves one loss scalar per sample for homogeneous and heterogeneous shape paths
  (`src/sakuramoon/train/runtime.py:320-349`).
- `SingleGpuStep` sums per-sample losses across microbatches, scales gradients once by
  total samples, applies FP32 finite/global clip, and records optimizer success before
  handling post-step `zero_grad` failure (`src/sakuramoon/train/step.py:183-213`,
  `:238-285`). Unequal microbatch tests match a merged sample-mean update.
- Scheduler and checkpoint decisions occur only after optimizer success. Pre-decay is
  durably published before scheduler mutation; other cadence points publish the exact
  proposed cadence and commit the in-memory anchor only after callback success
  (`src/sakuramoon/train/loop.py:153-245`). Stage-budget and cadence drift fail before
  loop construction (`src/sakuramoon/train/runtime.py:392-411`).
- The runtime uses the T021 raw block-24 Qwen output, frozen local Qwen and Mage-VAE,
  target-canvas size/aspect, default checkpointed CUDA RNG, T024 accepted packed
  boundary, and the full trainable composite. The recorded real one-GPU component
  smoke reaches text/style gradients and one TorchAO update.

## Independent verification

The reviewer ran:

```text
uv run --frozen pytest -q \
  tests/unit/train/test_step.py tests/unit/train/test_loop.py \
  tests/unit/train/test_preflight_failures.py tests/unit/train/test_runtime.py \
  tests/unit/train/test_scheduler.py tests/unit/cli/test_train.py \
  --basetemp=docs/model-architecture/reviews/T050/.pytest-ai-review \
  -p no:cacheprovider
```

Result: **53 passed** in 21.65 seconds, with 17 dependency deprecation warnings and no
test failure. The task-owned basetemp was precisely removed. GPU work was not rerun;
the recorded RTX 5090 evidence was inspected as required, and no four-GPU or formal
stage conclusion is inferred from it.

## Residual blocked gates

- The production CLI intentionally always fails because strict config still lacks
  explicit Text/Style constructor choices (`src/sakuramoon/cli/train.py:44-56`). No
  production preflight or training executable exists yet.
- The configured repository paths are on NFS rather than accepted local NVMe; this is
  correctly a hard preflight failure, not production acceptance.
- K001's FA4 numerical path has one-GPU evidence, but the fixed upstream
  commit/tree/license provenance gate remains blocked
  (`docs/model-architecture/progress/tasks/K001.md:3-6`, `:44-47`).
- Immutable production manifest/scans, the exact governed real-shard parser run,
  formal 1,000-update S0, quality evaluation, long-run/endurance, four-GPU DDP/NCCL,
  and all multi-GPU stage gates remain pending or blocked.
