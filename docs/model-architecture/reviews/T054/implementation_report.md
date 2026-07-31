# T054 implementation report

The fault harness defines a canonical CPU/1GPU/4GPU matrix rather than allowing an
incomplete scenario list to be labeled complete. Every passed outcome binds identical
pre/post control snapshots and every single-GPU outcome binds the explicit recovery
parent and replay state. Multi-GPU outcomes accept only an explicit blocked result.

After the independent implementation audit, the matrix path was strengthened so it can
no longer be instantiated solely from an aggregate unit-test fixture. Each of the 12
executed CPU/1GPU scenarios has a strict no-clobber evidence record bound to the final
test report and to the matrix by SHA-256 over the exact bytes that were parsed. Missing,
stale, mislabeled, malformed, or reordered records fail before matrix publication. The
five 4GPU records are constructed only as `blocked:FOUR-GPU-AVAILABLE`.

The process driver waits on an inherited pipe, sends a real SIGKILL, enforces a bounded
timeout, discards worker output, reaps the child, and passes only a fixed non-credential
environment allowlist. The expected-exit path now uses the same allowlist, with a
negative contract covering ModelScope and generic cloud credential names. The download
case leaves no final shard and restarts a residual
partial from byte zero. A coordinator lease marks completion only on normal return, so
a real DataLoader worker `os._exit()` inside `WebDatasetPipeline.iter_leased_shards()`
leaves the active shard recoverable and increments the exact manifest sample count once
on restart.

State files use unique temporary names and fsync their parent directory. Dataset shard
publication fsyncs its parent and removes an unacknowledged final file on publication
failure. Diagnostic bundles use unique temporary directories, fsync payload, marker,
directory, and parent, and retain both original and diagnostic failures through the
existing loop boundary. Injected ENOSPC failures leave no claimed complete cache,
checkpoint, state, diagnostic report, or matrix artifact.

On the RTX 5090, each phase worker performs a real BF16 forward/backward with the locked
TorchAO AdamW8bit policy. Microbatch and optimizer workers are killed at their explicit
barriers; checkpoint kill occurs after one update and a fsynced temporary state write
but before a COMPLETE marker. A fresh process accepts only `parent_0/COMPLETE`, restores
model/optimizer/SR state, performs one update, and emits identical protected controls.
The production raw selector was separately exercised against a real raw checkpoint and
the established fresh-process next-step equality test passed.

The allocator OOM case first restores the explicit `parent_0/COMPLETE`, requests more
than CUDA-reported free memory in a subprocess, catches only `torch.OutOfMemoryError`,
records unchanged controls and parent identity, and exits with the expected nonzero
code. A second fresh process restores the same parent and performs the next update; the
parent then allocates successfully, showing process-local context cleanup. The host
needs an explicit matching NVML 580.105.08 preload because
the default userspace library is mismatched; that test command does not alter runtime
training behavior.

No performance optimization or production capacity claim is made, so no
`perf_baseline.json` or `perf_after.json` was generated. DDP reduction kill, SR rank
divergence, NCCL rank failure, synchronized all-rank stop, four-rank raw recovery/state
equality, and formal multi-GPU fault canaries remain blocked pending 4x RTX 5090.

The independent pre-remediation audit found three blockers: expected-exit environment
inheritance, lease isolation from the real pipeline boundary, and an aggregate-only
matrix/OOM result without explicit parent restoration. All three are remediated above.
Two direct attempts to start a fresh reviewer failed with `agent thread limit reached`;
the user directed continuation without agents. The final acceptance is therefore a
main-agent remediation review, not a falsely claimed independent rereview.
