# Training Utilities Infra/performance review

Reviewer: `/root/training_utils_infra_review`

Review base: `d79c4d3817462d739f6220bc0eca3549899c743e`

Verdict: **FAIL pending task-local remediation in T052 and T053.** T051 passes
the implemented CPU and bounded single-GPU scope. This review does not close any
production, formal-stage, long-run, DDP/NCCL, four-GPU, throughput, capacity, or
quality gate.

## Findings

### T053: FAIL - measured throughput lacks a true window elapsed time

Each update records CPU submission time and a CUDA-event span, then selects their
per-update maximum. `summarize_benchmark` sums those maxima and uses the sum as every
throughput denominator. CPU work for update N+1 can overlap the GPU tail of update N,
so the sum can double-count overlapped time and is not the elapsed duration between
the synchronized measured-window boundaries. The resulting samples/token/FLOP rates
and phase/checkpoint shares are not a governed end-to-end throughput result.

Evidence: `src/sakuramoon/telemetry/profiler.py:885-964`, `:985-1001`, and
`:1121-1188`. Required remediation is to record and publish the actual complete
measured-window elapsed time between explicit boundaries, use it for aggregate rates,
and retain per-update timings only for p50/p95/p99. Tests must cover deliberate
CPU/GPU overlap and prove the aggregate denominator is the window elapsed time.

### T053: FAIL - GPU memcpy/memset activity is reported as idle

Trace derivation accepts only events whose category contains `kernel`. Device memcpy,
memset, and other GPU activity are discarded; all remaining trace time is then
reported as GPU idle while `gpu_unattributed_seconds` is fixed to zero. This makes GPU
active/idle, gap, and hotspot attribution unreliable for the data/H2D-heavy workload
the harness is intended to diagnose.

Evidence: `src/sakuramoon/telemetry/profiler.py:682-746`. Required remediation is an
explicit device-activity classification that includes governed copy/set categories,
keeps kernel launch/group metrics kernel-specific, and assigns unknown device work to
unattributed rather than idle. Synthetic trace tests must cover overlapping kernels,
copies, and unknown device categories.

### T053: FAIL - compile gain can be credited while changing attention backend

The variant allowlist permits `compile.regional_enabled` and
`kernels.attention_backend` in the same comparison. The comparison accepts disclosed
backend drift, while the compile gate only checks that the compile key/feature is
present. An adversarial four-GPU-shaped comparison changed `native` to `fa4`, enabled
regional compile, and supplied all three hash-bound evidence artifacts; it returned
`regional_compile_allowed=true`. The >=3% gain can therefore come entirely from the
backend change rather than compile.

Evidence: `src/sakuramoon/cli/benchmark.py:42-79` and
`src/sakuramoon/telemetry/profiler.py:1361-1424`. Required remediation is a compile
comparison whose only changed config key is `compile.regional_enabled`, with identical
attention backend and all other workload fields. A combined compile/backend mutation
must fail before a retention decision.

### T053: FAIL - host swap is measured but omitted from comparison policy

`BenchmarkReport` records `max_host_swap_bytes`, but `ComparisonPolicy`,
`BenchmarkComparison`, and `compare_benchmarks` cover only CUDA allocated/reserved,
host RSS, and pinned RAM. An adversarial comparison with zero permitted resource
growth accepted a new 4,096-byte swap peak and exposed no swap delta. Consequently a
future regional-compile comparison can satisfy the implemented gain/tail/evidence
gate despite swap, contrary to the no-swap operational contract.

Evidence: `src/sakuramoon/telemetry/profiler.py:1323-1424` and
`src/sakuramoon/cli/benchmark.py:170-188`. Required remediation is a published
host-swap field and a fail-closed zero-swap gate for both reports, with negative tests
showing any swap prevents retention/compile enablement. Production 16/20/24-layer
runs and four-GPU evidence remain separately pending after these code fixes.

### T052: FAIL - immutable scalar artifacts accept impossible metric values

`EvaluationArtifact` requires a finite scalar but does not enforce metric range. The
artifact boundary accepted an immutable FID result with `value=-1.0`; an IS artifact
can likewise accept a value below its mathematical lower bound. The metric functions
currently return governed values, but the public immutable publication contract does
not bind artifacts to that invariant, allowing corrupt trend/formal evidence to be
published. `automatic_release=false` limits blast radius but does not make the metric
evidence valid.

Evidence: `src/sakuramoon/eval/artifacts.py:19-47`. Required remediation is strict
metric-specific scalar validation (`FID >= 0`, `IS >= 1`) plus negative publication
tests. Real extractor/checkpoint execution, 10k/50k runs, and evaluator resource
coordination remain pending after this contract fix.

## Per-ID conclusions

### T051: PASS for current CPU and bounded 1GPU scope

Local JSONL publication precedes the bounded remote queue; append, file/directory
`fsync`, private regular-file checks, redacted retry, at-least-once replay, and close
failure propagation are fail-visible. CUDA event timing uses deferred `query()` and
does not synchronize at phase boundaries. The T050 observer keeps queue/event waits
bounded and rejects incomplete or inconsistent update facts. C002 commit `4873414`
now binds the exact config, paths, retry/startup sequence, context provider, and close
order; the older "C002 pending" wording in the T051 task/evidence is stale and is not
a current implementation blocker.

The historical RTX 5090 instrumentation benchmark reports 0.8057407532% median
overhead for a narrow 1024x1024 matmul workload. It is not training-throughput or
four-GPU overhead evidence. Full-chain phase coverage and four-GPU/DDP timing remain
pending/blocked.

### T052: FAIL for the implemented artifact contract

Scheduling, prompt/checkpoint provenance, canonical ordered prompt loading, exact IS
split divisibility, artifact-kind isolation, no-clobber publication, explicit GPU and
pause cost, and no-auto-release behavior otherwise passed review. The scalar-range
finding above prevents package closure. The retained single-GPU evidence is synthetic
Heun/metric plumbing only and provides no FID/IS quality or production-cost claim.

### T053: FAIL for the implemented comparison contract

Warmup/measured counts, runtime data/shape identity, real successful-update adapter,
checkpoint cadence, compile/recompile/fallback failure, atomic trace publication, and
external-trace fail-closed behavior otherwise passed review. The four measurement and
comparison findings above prevent package closure. The two short RTX 5090 tests are
mechanics evidence only; NCU remains blocked by
`ERR_NVGPUCTRPERM`, and no production trace, capacity result, or before/after artifact
is claimed.

## Independent verification

- `uv run --frozen pytest -q -p no:cacheprovider ... tests/unit/telemetry
  tests/unit/eval tests/unit/config/test_eval_jobs.py
  tests/unit/config/test_benchmark_config.py
  tests/unit/config/test_c002_production_config.py
  tests/unit/train/test_benchmark_adapter.py tests/unit/train/test_step.py`:
  **214 passed**, 17 warnings.
- Focused Ruff: **passed**.
- Focused strict Pyright: **0 errors, 0 warnings, 0 informations**.
- Traceability unit suite: **40 passed**.
- Live trace verifier: **237 requirements, 237 source nodes, 109 production
  modules, 907 runtime config keys, 16 read-only archive hashes, 0 errors**;
  registry revision 109.
- `git diff --check` for commits `91aa14d`, `1dfeb18`, and `d79c4d3`: **passed**.
- All T051/T052/T053 JSON evidence parsed successfully. No GPU test was rerun for
  this review; the bounded historical single-GPU evidence was inspected without
  extending it to production or four-GPU conclusions.

## Remaining operational gates

T051 still needs full-chain and four-GPU timing overhead. T052 still needs immutable
production prompt/extractor/preprocess/real-stat identities, a real checkpoint,
resource coordination, and bounded/formal metric execution. T053 still needs approved
real data/Qwen/VAE/DiT workloads, retained production traces, 16/20/24-layer and final
24L/512 runs, capacity evidence, and four-GPU DDP/NCCL validation. Synthetic or
single-GPU evidence must not close those gates.

## Post-remediation independent rereview (2026-08-02)

Reviewer: `/root/training_utils_infra_rereview`

Review base: `433702f2d64a3ca4a367d377a0362a87b3691a76`, including
T052 remediation commit `617d4b8041c1e7f4c173447a749cb879245802a4` and
T053 remediation commit `433702f2d64a3ca4a367d377a0362a87b3691a76`.

Overall verdict: **PASS for T051, T052, and T053 within the implemented CPU and
bounded single-GPU mechanics scope.** The initial FAIL and its five findings above
are retained as historical review evidence. Each finding is closed by the cited
remediation; no production throughput, capacity, quality, formal-stage, long-run,
DDP/NCCL, regional-compile, or four-GPU gate is closed by this rereview.

### T051: PASS for the implemented telemetry scope

The current code still preserves local-before-remote publication, explicit append and
directory durability, private regular-file retry queues, redacted at-least-once replay,
bounded asynchronous queues, visible close failures, monotonic CPU timing, and deferred
CUDA-event collection without phase-boundary synchronization. The T050 observer rejects
nonconsecutive or inconsistent successful-update facts and keeps event waits bounded.
C002 commit `48734143ca95cf9b0c72eaa227ab3b6a758fa047` binds the strict telemetry
configuration, paths, startup replay, context-provider boundary, and lifecycle order;
T051 commit `91aa14d4d0281ebcd09aafae7b221d6734cf5334` provides the current adapter and
durability implementation.

Inspected implementation and tests include `telemetry/metrics.py`, `timers.py`,
`nvtx.py`, `wandb_sink.py`, `observer.py`, `config/assembly.py`, and the corresponding
unit suites. The historical 0.8057407532% RTX 5090 matmul instrumentation result remains
a narrow single-GPU timing result only. Full real-chain phase coverage and four-GPU/DDP
overhead remain pending/blocked.

### T052: PASS after artifact-range remediation

Commit `617d4b8041c1e7f4c173447a749cb879245802a4` closes the original artifact
contract finding. `EvaluationArtifact.__post_init__` now rejects `FID < 0` and `IS < 1`
before any publisher can create an artifact, while preserving the existing finite-value,
metric/kind, IS-standard-deviation, pause-cost, and no-auto-release checks. The new
negative tests cover both metric kinds and assert that no destination is created.

The broader scheduler, prompt/checkpoint identity, ordered manifest, exact IS split,
artifact isolation, no-clobber publication, and cost contracts were also rechecked at
the current HEAD. The original implementation commit
`1dfeb181d554821ebcd2a9f7a2b05168023301ed` and the package remediation report/test
evidence were inspected. Production prompt/extractor/preprocess/real-stat identities,
a real COMPLETE checkpoint, approved evaluator placement, real extractor execution,
authorized 10k/50k runs, and manual quality review remain pending; no numeric quality
threshold is inferred from synthetic plumbing.

### T053: PASS after benchmark-accounting remediation

Commit `433702f2d64a3ca4a367d377a0362a87b3691a76` closes all four original T053
findings:

- Aggregate samples/token/FLOP rates and phase/checkpoint shares now use one positive,
  synchronized boundary-to-boundary `measured_window_seconds`; per-update spans remain
  the p50/p95/p99 source, and trace serialization occurs after the window closes.
- Trace accounting unions kernels with governed `gpu_memcpy`/`gpu_memset` intervals,
  assigns uncovered unknown `gpu_*` work to unattributed time, and keeps kernel launch,
  group, gap, and NCCL statistics kernel-specific.
- A regional-compile transition must change only `compile.regional_enabled`, preserve
  the attention backend, and add exactly the regional-compile feature. A combined
  backend/compile comparison fails before retention is evaluated.
- Comparison policy requires zero allowed swap, both baseline and candidate must report
  zero swap, and the published comparison includes the swap delta.

The relevant current paths are `telemetry/profiler.py`, `cli/benchmark.py`,
`tests/unit/telemetry/test_profiler.py`, and
`tests/unit/config/test_benchmark_config.py`; the original implementation commit
`d79c4d3817462d739f6220bc0eca3549899c743e` and the benchmark-accounting remediation
reports were also inspected. The short historical RTX 5090 cases remain synthetic
matmul/BF16-linear mechanics evidence only. Real data/Qwen/VAE/16/20/24-layer DiT
100+500 runs, the final 24L/512 1,000-update run, retained formal profiler/Nsight
traces, capacity conclusions, NCU access, and four-GPU DDP/NCCL remain pending or
blocked.

### Independent verification

Focused archive-free CPU command:

```text
uv run --frozen pytest -q -p no:cacheprovider --basetemp=/tmp/sakuramoon-training-utils-infra-rereview-20260802-focused tests/unit/telemetry/test_metrics.py tests/unit/telemetry/test_timers_nvtx.py tests/unit/telemetry/test_wandb_sink.py tests/unit/telemetry/test_observer.py tests/unit/eval/test_artifacts_quality.py tests/unit/eval/test_fid_is_metrics.py tests/unit/telemetry/test_profiler.py tests/unit/config/test_benchmark_config.py
```

Result: **97 passed, 17 dependency deprecation warnings**.

- Focused Ruff over the corresponding source/tests: **passed**.
- Focused strict Pyright over telemetry/eval/benchmark sources: **0 errors, 0 warnings,
  0 informations**.
- `git diff --check` for commits `617d4b8` and `433702f`: **passed**.
- Archive-free `tomllib` parse of the prepared registry: revision **112**; all 12
  affected stable IDs (`OBS-002..004`, `OBS-006..012`, `OPEN-067`, `OPEN-068`) retain
  valid 64-hex fingerprints and bind this package Infra report. The worktree diff only
  increments the registry revision and adds the package AI/Infra paths for those IDs.
- The only rereview basetemp,
  `/tmp/sakuramoon-training-utils-infra-rereview-20260802-focused`, was moved intact to
  Trash after the run.

The repository traceability unit test and live verifier are intentionally excluded:
both traverse `docs/model-architecture/archive/`, which is prohibited for this rereview.
Their historical results in earlier task/package evidence are not used for this PASS.
No GPU test/kernel workload, DDP/NCCL, multi-GPU, long-run, or formal stage was
executed. Notion, `.env`, `docs/model-architecture/archive/`, and `reference/` were
not accessed during this rereview.
