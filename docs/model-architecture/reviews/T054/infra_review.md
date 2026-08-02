# T054 Infra/performance review

Status: PASS for bounded CPU/1GPU process, durability, and resource-cleanup behavior
after remediation acceptance. As documented in the AI review, a fresh independent
post-remediation reviewer was unavailable after two direct launch failures; no
independent PASS is claimed.

Both subprocess drivers now construct the same fixed environment allowlist. A negative
contract sets ModelScope and generic cloud credential variables in the parent and proves
they are absent in the child. Workers use inherited pipe readiness, process-group
`SIGKILL`, bounded waits, discarded output, and unconditional reaping.

Cache, state, diagnostic, scenario-evidence, and matrix writers use unique temporary
entries, file and directory fsync, no-clobber publication, and cleanup on injected
`ENOSPC`. The leased pipeline does not claim completion after a DataLoader worker exit.
The CUDA OOM process restores a fixed complete parent first, exits without retry, and is
followed by both a fresh recovery process and a parent-process allocation probe.

Targeted results: 16 CPU fault tests passed; 87 associated CPU regressions passed; 7
bounded RTX 5090 tests passed in 89.30 seconds. No `perf_baseline.json` or
`perf_after.json` was created because this is a correctness task, not an optimization.
No long run, DDP/NCCL command, four-GPU run, 100+500 benchmark, or formal stage canary
was executed.

## Main-agent post-remediation readiness self-review

The initial archive-free CPU revalidation reported 156 passed and three failed because
all real-SIGKILL workers exceeded the five-second readiness timeout during package
import. The measured import was 12.847739 seconds. Lazy export resolution removes the
unused checkpoint/Torch initialization from the readiness child; a fresh
credential-free probe measured 0.030985 seconds.

Post-fix results are 10/10 focused control-plane tests, 159/159 targeted CPU tests, and
7/7 bounded RTX 5090 tests in 113.11 seconds. Child reaping, fixed environment
allowlisting, explicit COMPLETE-parent restore, and GPU context recovery remain covered.
Ruff, Pyright, and `git diff --check` pass. Main-agent Infra self-review is PASS for this
bounded CPU/1GPU readiness correction only. No independent rereview, performance
capacity result, long run, multi-GPU result, or formal-stage evidence is claimed.

## Independent Infra/performance post-remediation review (2026-08-02)

Reviewer authority: fresh independent Infra reviewer. This review inspected commit
`a8d16a62032e08f53fafc908a11c516811bc3c73` and reran archive-free CPU/static checks.
The reviewer did not modify implementation, run GPU workloads, stage files, or commit.

### Verdict

PASS for the implemented CPU and bounded-single-GPU post-remediation scope. The earlier
independent Infra findings are closed within that scope. No performance capacity,
formal-stage, or multi-GPU conclusion is implied.

### Post-remediation conclusions

- Lazy package exports remove the readiness-path Torch/checkpoint initialization
  regression. A fresh credential-empty import of `signal_ready_from_environment`
  completed in 0.028055 seconds; the 23 public exports were unique and matched the lazy
  export map.
- The real-SIGKILL and expected-exit drivers use the same fixed credential-free child
  environment allowlist, bounded readiness/reaping, and process-group termination.
- `WebDatasetPipeline.iter_leased_shards()` now holds the durable lease across actual
  iteration: abnormal generator or worker exit preserves active state, while normal
  exhaustion completes it.
- The scenario binder requires every CPU/1GPU record in canonical order, validates the
  strict schema, binds the test-report hash, and verifies exact evidence bytes by
  SHA-256.
- The OOM worker requires an explicit complete parent, validates protected recovery
  controls before allocation, and is followed by fresh-process recovery. Raw recovery
  remains exact and exposes no latest, PMA, model-only, or silent fallback path.

### Independent checks

- T054 archive-free CPU selector: 161 passed, 5 warnings.
- Scoped Ruff passed.
- Scoped Pyright reported 0 errors.
- `git diff --check` passed before this report append.

### Open boundary

DDP reduction kill, stochastic-rounding rank divergence, NCCL rank failure, all-rank
synchronized stop, four-rank checkpoint recovery/state equality, formal-stage fault
canary, and long-run behavior remain blocked or pending. Existing CPU and bounded 1GPU
evidence cannot close any of those gates.

## Final independent Infra/performance review (2026-08-02)

Reviewer authority: fresh independent `s000_infra_reviewer_final`. This review inspected both
the lazy-readiness remediation commit
`a8d16a62032e08f53fafc908a11c516811bc3c73` and the independent post-remediation review
evidence commit `a5771b9c26ab6725bae32bbc4a82726bdaa2bd66`. It did not modify
implementation, use a GPU, stage files, or create commits.

### Verdict

PASS for T054's implemented CPU and bounded-single-GPU reliability scope. The prior
post-remediation conclusions are supported; no new T054 Infra finding was identified.

### Conclusions

- Lazy public exports keep the fault-injection API available without importing the heavy
  checkpoint/Torch dependency graph before the subprocess readiness barrier.
- Credential-free child allowlisting, bounded readiness and reaping, process-group kill,
  durable lease replay, exact complete-parent recovery and no latest/PMA/model-only
  fallback remain structurally enforced.
- The post-remediation evidence commit appends reviews without rewriting the historical
  self-review or changing implementation. Its conclusions preserve the CPU/1GPU evidence
  boundary and all four-rank blockers.

### Independent checks

- Current archive-free T054 CPU selector: 156 passed, 19 dependency/runtime warnings.
- The additional training/runtime/stage/fault selector: 66 passed.
- Scoped Ruff passed; affected-path Pyright reported 0 errors.
- The retained seven-test RTX 5090 result was inspected as historical bounded evidence;
  this reviewer did not rerun it or infer a capacity, long-run, formal-stage, DDP/NCCL or
  four-GPU conclusion.
