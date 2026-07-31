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
