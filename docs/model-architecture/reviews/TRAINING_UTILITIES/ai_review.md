# Training Utilities package AI/model correctness review

Reviewer: independent agent `/root/training_utils_ai_review`

Date: 2026-08-02

Scope: the current T051-T053 CPU and bounded single-GPU implementation at
`HEAD=d79c4d3817462d739f6220bc0eca3549899c743e`, including the atomic task
commits below and the current canonical decisions, open items, observability
contract, roadmap, task files, trace mappings, implementation reports, tests, and
remediation evidence.

- T051: `91aa14d4d0281ebcd09aafae7b221d6734cf5334`
- C002 integration consumed by T051/T052:
  `48734143ca95cf9b0c72eaa227ab3b6a758fa047`
- T052: `1dfeb181d554821ebcd2a9f7a2b05168023301ed`
- T053: `d79c4d3817462d739f6220bc0eca3549899c743e`

Overall verdict: **PASS** for each task's implemented CPU and bounded single-GPU
scope. No blocking AI/model-correctness finding remains.

## Per-task verdicts

| Task | Verdict | Independent conclusion |
|---|---|---|
| T051 | **PASS** | The fixed numeric schema preserves total and `t<0.95`/`t>=0.95` bucket losses with explicit bucket counts, exact T050 successful-update identity, clipping facts, timestep statistics, token/FLOP/memory/queue facts, all dropout counters, and the fixed phase vocabulary. The adapter consumes detached successful-update facts and cannot change loss, optimizer, scheduler, checkpoint, batch, or backend behavior. Local JSONL publication precedes bounded asynchronous W&B submission; timing uses monotonic CPU clocks and deferred CUDA events without phase-boundary synchronization. C002 now supplies the strict production construction, context provider, paths, cadence, timing vocabulary, W&B identity, replay, and close ordering, so the older T051 wording that calls C002 pending is historical and is not an active blocker. |
| T052 | **PASS** | Jobs are derived only from strict successful-update/stage-end schedules. The canonical, non-symlinked prompt manifest is hash-checked and consumed as an immutable ordered prefix with sufficient cases. Checkpoint kind/ID/config/update, trigger update, prompt plan, reference sampling/CFG contract, extractor/preprocess/stat identities, IS splits, GPU, and pause policy all participate in the content-addressed job identity. Future checkpoints fail, raw latest must match the trigger, and an older accepted checkpoint remains valid. FID uses CPU float64 statistics with symmetry/PSD checks; IS uses deterministic exactly divisible splits. FID/IS, manual quality, and VAE artifacts are type-isolated, three-checkpoint comparisons require a shared metric/generation protocol, and no metric can automatically release a checkpoint. |
| T053 | **PASS** | The current remediation changes evidence governance only; the previously reviewed harness remains unchanged. It enforces exactly 100 warmup plus 500 candidate measured successful updates, or at least 1,000 final measured updates, through `SingleGpuStepBenchmarkAdapter`. Actual sample/shape streams are hashed across warmup and measured windows and must match the workload identity. Real step boundaries cover DiT, loss, backward, clip, optimizer, zero-grad, data, and checkpoint facts; checkpoint cadence, measured compile/recompile/fallback counters, profiler trace range, and publication all fail closed. The two short GPU cases prove only synthetic matmul and BF16 linear update mechanics, not model performance or capacity. |

## Independent validation

Focused package CPU tests:

```text
uv run --frozen pytest -q -p no:cacheprovider --basetemp=/tmp/sakuramoon-training-utils-ai-review-20260802 tests/unit/telemetry/test_metrics.py tests/unit/telemetry/test_timers_nvtx.py tests/unit/telemetry/test_wandb_sink.py tests/unit/telemetry/test_observer.py tests/unit/telemetry/test_profiler.py tests/unit/train/test_benchmark_adapter.py tests/unit/config/test_benchmark_config.py tests/unit/eval tests/unit/config/test_eval_jobs.py
```

Result: **159 passed, 17 warnings in 15.79s**.

Targeted Ruff:

```text
uv run --frozen ruff check src/sakuramoon/telemetry src/sakuramoon/eval src/sakuramoon/train/benchmark.py src/sakuramoon/cli/eval.py src/sakuramoon/cli/benchmark.py tests/unit/telemetry tests/unit/eval tests/unit/train/test_benchmark_adapter.py tests/unit/config/test_eval_jobs.py tests/unit/config/test_benchmark_config.py
```

Result: **All checks passed**.

Targeted strict Pyright:

```text
uv run --frozen pyright src/sakuramoon/telemetry src/sakuramoon/eval src/sakuramoon/train/benchmark.py src/sakuramoon/cli/eval.py src/sakuramoon/cli/benchmark.py
```

Result: **0 errors, 0 warnings, 0 informations**.

`git diff --check` passed before this review file was added. Historical final task
evidence was also checked: T052 records focused 89, targeted 180, full CPU 933,
trace unit 40, and one bounded RTX 5090 test; T053 records focused 52, config
regression 218, trace unit 40, and two bounded RTX 5090 tests. The live registry at
T053 completion was revision 109 with 237 requirements, 109 production modules,
907 runtime keys, and zero errors. This reviewer did not rerun GPU, DDP/NCCL,
multi-GPU, long-run, or formal-stage work.

## Residual gates

- T051 still requires real full-chain phase coverage and an independent four-GPU
  timing/DDP overhead result. Its prior 0.8057% single-GPU matmul instrumentation
  result is focused overhead evidence only, not training throughput evidence.
- T052 still requires S000-qualified immutable prompt/extractor/preprocess/real-stat
  identities and budgets, a real training checkpoint and real extractor execution,
  an approved evaluator GPU/pause plan, the authorized 10k/50k runs, and manual
  quality review. No FID/IS numeric release threshold may be invented from the
  synthetic plumbing result.
- T053 still requires authorized real data/Qwen/VAE/DiT/loss/checkpoint 16/20/24-layer
  100+500 runs, the final 24L/512 1,000-measured-update run, retained formal traces,
  and capacity conclusions. NCU remains blocked by `ERR_NVGPUCTRPERM`.
- No CPU or single-GPU evidence here closes production throughput, quality, formal
  stage, long-run, regional-compile, DDP/NCCL, rank-failure, or four-GPU gates.

## Post-remediation independent rereview

Reviewer: independent agent `/root/training_utils_ai_rereview_v2`

Date: 2026-08-02

Review base: `HEAD=433702f2d64a3ca4a367d377a0362a87b3691a76`, including
T052 remediation commit `617d4b8041c1e7f4c173447a749cb879245802a4`, T053
remediation commit `433702f2d64a3ca4a367d377a0362a87b3691a76`, and the
uncommitted revision-112 package-review bindings. The historical review above is
retained unchanged.

Overall verdict: **PASS** for T051, T052, and T053 within their implemented CPU and
bounded single-GPU evidence boundaries. No AI/model-correctness blocker remains for
the Training Utilities package review. This verdict does not close any formal
evaluation, production benchmark, long-run, stage, DDP/NCCL, or four-GPU gate.

### T051: PASS for implemented CPU and bounded 1GPU scope

The implementation remains anchored at `91aa14d4d0281ebcd09aafae7b221d6734cf5334`
with C002 production assembly at `48734143ca95cf9b0c72eaa227ab3b6a758fa047`;
the metrics, observer, and local/W&B publication paths have not changed since the
earlier package review. The fixed record retains total and `t<0.95`/`t>=0.95` losses
and counts, clipping facts, timestep statistics, effective samples, tokens/FLOPs,
memory/queue facts, all fixed dropout counters, and the complete timing vocabulary.
The bounded observer consumes detached successful-update facts, waits for CUDA events
by query rather than synchronization, and cannot mutate model loss, optimizer,
scheduler, checkpoint, batch, backend, or training control flow. Local JSONL remains
ordered before bounded remote submission and remote failure remains fail-visible and
durably retryable.

Focused rereview coverage included `test_metrics.py`, `test_observer.py`,
`test_timers_nvtx.py`, and `test_wandb_sink.py`. **Residual status: BLOCKED/PENDING**
for real full-chain phase coverage, current production-run overhead, and independent
four-GPU timing/DDP overhead. The historical 0.8057% RTX 5090 matmul instrumentation
result remains a narrow single-GPU measurement, not model throughput evidence.

### T052: PASS for remediated CPU and synthetic-1GPU plumbing scope

Commit `617d4b8041c1e7f4c173447a749cb879245802a4` closes the package Infra finding at
the immutable publication boundary: `EvaluationArtifact` now rejects non-finite
values, `FID < 0`, and `IS < 1` before any artifact can be constructed and published.
The existing job contract already binds metric to artifact kind, requires a
content-addressed job identity, keeps IS split standard deviation explicit, and
forbids automatic release. Negative tests cover both metric lower bounds and assert
that no output path is created. This fail-closed validation does not alter prompt,
checkpoint, sampler, extractor/stat, comparison, or manual-quality semantics.

Focused rereview coverage included the evaluation artifact, FID/IS, schedule, and
strict evaluation-config suites. **Residual status: BLOCKED/PENDING** for frozen
S000-qualified prompt/extractor/preprocess/real-stat identities and budgets, a real
COMPLETE checkpoint and real extractor execution, an approved evaluator GPU/pause
plan, authorized 10k/50k runs, and manual quality review. The retained short 1GPU
result is synthetic solver/metric plumbing only and supplies no quality threshold or
formal FID/IS result.

### T053: PASS for remediated CPU harness and bounded 1GPU mechanics scope

Commit `433702f2d64a3ca4a367d377a0362a87b3691a76` closes all four package Infra
accounting findings without changing successful-update, objective, backward, clip,
optimizer, scheduler, sample/shape identity, or checkpoint-cadence semantics:

- Aggregate sample/token/FLOP rates and phase/checkpoint shares use one positive,
  explicitly synchronized measured-window elapsed time. Per-update host/CUDA spans
  remain only for p50/p95/p99, an impossible update/window relation fails closed, and
  Chrome-trace serialization occurs after the measured window.
- Trace accounting unions kernels with governed GPU memcpy/memset intervals, assigns
  uncovered unknown `gpu_*` work to unattributed time, and leaves launch/group/gap/NCCL
  accounting kernel-specific without double-counting overlapping intervals.
- A regional-compile transition must isolate exactly
  `compile.regional_enabled`, preserve the attention backend, and add exactly the
  `regional_compile` feature. A combined backend/compile mutation fails before a
  retention decision.
- Comparison policy and output include host swap, policy cannot allow swap, and any
  baseline or candidate swap fails before retention.

Focused rereview coverage included profiler accounting, benchmark configuration, the
production step adapter, and T050 step contracts. **Residual status: BLOCKED/PENDING**
for approved real service/data/Qwen/VAE/16L-DiT/JLT/checkpoint 100+500 runs, real
20/24-layer and final 24L/512 1,000-measured-update runs, retained formal traces,
capacity and `perf_baseline`/`perf_after` conclusions, and all DDP/NCCL/four-GPU
evidence. NCU remains blocked by `ERR_NVGPUCTRPERM`. The two retained RTX 5090 cases
remain short matmul/BF16-linear mechanics evidence only.

### Independent validation and evidence boundary

Archive-free focused CPU validation:

```text
CUDA_VISIBLE_DEVICES='' uv run --frozen pytest -q -p no:cacheprovider \
  --basetemp=/tmp/sakuramoon-training-utils-ai-rereview-v2-20260802 \
  tests/unit/telemetry/test_metrics.py \
  tests/unit/telemetry/test_observer.py \
  tests/unit/telemetry/test_timers_nvtx.py \
  tests/unit/telemetry/test_wandb_sink.py \
  tests/unit/telemetry/test_profiler.py \
  tests/unit/eval/test_artifacts_quality.py \
  tests/unit/eval/test_fid_is_metrics.py \
  tests/unit/eval/test_spec_schedule.py \
  tests/unit/config/test_eval_jobs.py \
  tests/unit/config/test_benchmark_config.py \
  tests/unit/train/test_benchmark_adapter.py \
  tests/unit/train/test_step.py
```

Result: **156 passed, 3 warnings in 13.78s**. Targeted Ruff passed; targeted strict
Pyright reported **0 errors, 0 warnings, 0 informations**. No GPU, DDP/NCCL, long-run,
or formal-stage command was run by this reviewer.

An archive-free `tomllib` parse confirmed registry revision **112**, 237 globally
unique requirement IDs, unchanged fingerprints for T051 `OBS-002..004`, T052
`OBS-006..012`, and T053 `OPEN-067/068`, and exact bindings of all 12 entries to both
Training Utilities package reports. The uncommitted docs-test fixture addition was
inspected but not executed.

The historical traceability unit/live results recorded in task evidence traverse
`docs/model-architecture/archive/`; they are **inadmissible for this rereview and were
not used or rerun**. No archive-dependent result contributes to the PASS verdicts
above.
