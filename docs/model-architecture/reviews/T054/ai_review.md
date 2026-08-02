# T054 AI/model correctness review

Status: PASS for the implemented CPU and bounded 1GPU semantics after remediation
acceptance. This is not an independent post-remediation PASS: the independent audit
found the blockers below, while two direct attempts to start a fresh reviewer failed
with `agent thread limit reached` and the user directed continuation without agents.

## Independent audit findings and disposition

1. The shard lease was isolated in a unit helper and did not protect the real
   `WebDatasetPipeline` consumption boundary. Resolved by per-shard leased iteration;
   abnormal termination preserves `active`, normal exhaustion alone completes, and
   restart counts exact replayed shards and manifest samples.
2. The matrix could be constructed from synthetic aggregate fixtures. Resolved by a
   strict per-scenario evidence loader and binder that requires all 12 CPU/1GPU records,
   verifies the test-report hash, hashes the exact parsed bytes, and rejects missing,
   stale, malformed, or mislabeled evidence before publication.
3. The OOM worker did not restore an explicit parent. Resolved by mandatory
   `--parent`, `COMPLETE` and control validation before allocation, plus a second fresh
   recovery process after OOM.

The protected controls remain identical across every passed record. The raw recovery
selector still requires one exact checkpoint ID and successful update, and the worker
uses `torch.load(..., weights_only=True)`. No fallback changes batch, accumulation,
backend, world size, optimizer, LR, feature gates, or checkpoint cadence.

DDP reduction kill, SR rank divergence, NCCL rank failure, synchronized all-rank stop,
four-rank raw recovery/state equality, and formal fault canaries remain blocked on
`FOUR-GPU-AVAILABLE`; none is inferred from 1GPU evidence.

## Main-agent post-remediation readiness self-review

The 2026-08-02 readiness run exposed a package-import regression before any fault was
injected: the readiness-only worker spent 12.847739 seconds loading transitive
checkpoint and Torch modules and missed the fixed five-second barrier. Converting the
package exports to the repository's established lazy-public-API pattern preserves all
symbols and type-only imports while reducing the fresh credential-free import to
0.030985 seconds.

The three phase-specific SIGKILL barriers now pass, as do the complete 159-test CPU
selector and seven bounded single-GPU tests. This validates the control-plane timing
needed to reach the intended fault boundary; it does not add training semantics,
fallbacks, or production-stage claims. Main-agent AI/model self-review is PASS for the
implemented CPU and bounded 1GPU scope only. No independent post-remediation conclusion
is claimed under the user's no-agent direction.

## Independent post-remediation review (2026-08-02)

Reviewer authority: independent `s000_ai_reviewer`, reviewing commit
`a8d16a62032e08f53fafc908a11c516811bc3c73` and the preserved T054 evidence. The
reviewer ran CPU/static checks only.

### Verdict

PASS for the implemented CPU and bounded single-GPU T054 scope. The lazy public API
preserves all 23 exports, imports only the owning module on first access, and removes
the checkpoint/Torch import path from the readiness-only child. No training control,
recovery-parent selection, or fault outcome is changed by this remediation.

The credential-empty fresh import of `signal_ready_from_environment` completed in
0.028010 seconds, below the fixed five-second readiness contract. The focused
control-plane selector passed 10 tests, and the current full T054 CPU selector passed
161 tests with no failures. Ruff passed for the lazy initializer and control-plane
test. Existing bounded single-GPU evidence was inspected but not rerun by this reviewer.

DDP reduction kill, SR rank divergence, NCCL rank failure, synchronized all-rank stop,
four-rank raw recovery/state equality, and formal stage fault canaries remain blocked
on real four-GPU resources and the prohibited long-run/formal-stage scope. This PASS
does not close or reinterpret those retained blockers.
