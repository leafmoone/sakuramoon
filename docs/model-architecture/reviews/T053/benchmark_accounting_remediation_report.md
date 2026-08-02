# T053 benchmark-accounting remediation report

Status: the four Infra findings from the post-commit implementation self-review are
remediated in the CPU harness and bounded single-GPU mechanics scope. Independent
Training Utilities package rereview remains required before package closure.

The aggregate samples, image-token, text-token, and DiT-FLOP rates now use one
explicitly synchronized boundary-to-boundary measured-window wall clock. Per-update
host/CUDA spans remain only for p50/p95/p99. Phase and checkpoint shares use the same
window denominator, trace serialization occurs after that window closes, and invalid
or shorter-than-an-update windows fail closed.

Trace accounting now unions kernel plus governed `gpu_memcpy` and `gpu_memset`
intervals as active device time. Unknown `gpu_*` work contributes only uncovered time
to `gpu_unattributed_seconds`; idle is the remainder after all observed device work.
Kernel launches, groups, gaps, and NCCL remain kernel-specific.

Regional compile retention now requires the sole changed config key to be
`compile.regional_enabled`, an unchanged attention backend, and exactly one added
`regional_compile` feature. Combined backend/compile comparisons fail before a
retention result. Comparison policy and output include host swap, the policy cannot
permit swap, and either baseline or candidate swap hard-fails comparison.

Historical T053 implementation, timing, artifact, AI, Infra, and first remediation
evidence was not rewritten. No production 16/20/24-layer benchmark, retained formal
trace, performance baseline/after, capacity conclusion, formal stage, DDP/NCCL, or
four-GPU evidence is claimed. NCU remains blocked by `ERR_NVGPUCTRPERM`.

The repository trace unit/live verifiers traverse the read-only architecture archive.
Their implementation-agent runs are therefore inadmissible under this turn's stricter
no-archive-read constraint and are not used for acceptance. Main-agent acceptance uses
an archive-free TOML check limited to revision 111 and stable OPEN-067/OPEN-068
identity/evidence preservation.

## Implementation self-review

AI/model correctness: PASS for the CPU harness and bounded single-GPU mechanics scope.
The remediation changes measurement boundaries, trace classification, and comparison
validation only; successful-update order, model/loss/backward/update semantics, sample
and shape identities, scheduler advancement, and checkpoint cadence remain unchanged.

Infra/performance: PASS for the remediated control-plane scope. Aggregate accounting is
bound to one synchronized window, device intervals are partitioned without overlap,
compile retention is isolated, and zero swap is enforced for both sides of comparison.
Focused CPU, config/manifest regression, short RTX 5090 mechanics, Ruff, strict Pyright,
and archive-free registry checks pass. The trace unit/live results are excluded under
the current no-archive-read constraint. Formal performance and capacity gates remain
pending or blocked as listed above.
