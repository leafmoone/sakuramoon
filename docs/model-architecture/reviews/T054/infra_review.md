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
