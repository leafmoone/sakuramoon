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
