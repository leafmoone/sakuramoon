# K001 independent AI/model correctness rereview

Reviewer: independent agent `/root/k001_ai_rereview`.

Scope: the current accepted-boundary K001 remediation and its committed T024
dependencies at `HEAD=d396029cf0a2f95ab8ccf7d27d1cc9422a960973`. The uncommitted
roadmap changes in the shared worktree were excluded. This review only writes this
file; it did not modify implementation, tests, benchmark output, traceability, or
the historical Infra review. No `.env`, `reference/`, or archived architecture file
was read or executed.

## Verdict

**PASS for AI/model correctness in the implemented K001 scope, with the fixed
upstream provenance audit BLOCKED.** No implementation-level finding remains in the
accepted-boundary remediation. K001 must not claim the governed four-column upstream
algorithm audit is complete until a fixed FA4 repository commit, tree identity, and
license evidence are established.

## Findings

### K001-AI-BLOCKED-001: no governed fixed FA4 upstream repository commit

The runtime is reproducibly pinned to `flash-attn-4==4.0.0b24` and its wheel SHA-256
(`uv.lock:207-223`), but `docs/model-architecture/reviews/R001/reference_manifest.json`
contains only HDM, JLT, and krea-2 repository locks (`:6-55`). No FlashAttention
repository URL/commit/tree/license lock is present. A wheel pin establishes the
installed artifact, not the requested fixed-commit algorithm provenance, so the FA4
upstream audit and any claim that this provenance gate is closed remain **BLOCKED**.
This is a remaining governance scope item, not a failure of the local numerical
implementation. No network or `reference/` access was used to fill the gap.

## Accepted-boundary correctness

- `accept_fa4_boundaries` performs one host/CUDA content comparison at the
  PackedDiT entry, then rematerializes private CUDA offsets from the canonical host
  lengths (`src/sakuramoon/model/attention.py:82-130`). A forged host `(2,2)` with
  CUDA `[0,3,4]` and post-construction mutation are rejected before native FA4
  import (`tests/gpu/fa4/test_varlen_attention.py:265-333`).
- Only the accepted capability reaches the native call; FA4 checks native 20Q/5KV
  shapes, CUDA BF16, contiguity, and uses `pack_gqa=True`, `causal=False`, and the
  same accepted offsets (`src/sakuramoon/model/attention.py:174-242`). No K/V head
  repetition or silent dense fallback is present.
- `PackedDiT.forward_packed_features` accepts exactly once per packed forward and
  passes that handle to every active block (`src/sakuramoon/model/dit.py:505-543`).
  `accepted_sample_indices` expands the same host `sequence_lengths` tuple rather
  than rereading CUDA offsets (`src/sakuramoon/model/attention.py:133-153`), so
  sample routing and FA4 share one boundary identity. The 16-block test and
  instrumentation report one entry D2H and zero per-block D2H
  (`tests/gpu/model/test_packed_dit.py:172-264`; `reviews/K001/fa4_benchmark.json`).
- Q/K head-dimension RMSNorm runs before shared-frequency 2D RoPE, and the native
  attention output is gated and projected only after FA4 (`src/sakuramoon/model/attention.py:483-494`,
  `src/sakuramoon/conditioning/rope.py:104-130`). The locked `32/48/48`,
  `position_scale=16`, `theta=1000`, noncausal, native 20Q/5KV contract is preserved.
- The dense path is an explicit reference. Performance uses separate per-sample
  SDPA calls with `attn_mask=None`; the all-True mask is limited to numerical
  correctness (`benchmarks/attention/benchmark_fa4_varlen.py:130-153`). The current
  benchmark/task/implementation/test artifacts contain no import or description of
  the removed `validate_cu_seqlens` API. Older historical review records retain that
  term as the prior finding and were intentionally not rewritten.

## Numerical and performance evidence

The identical-state FA4-vs-dense module contract compares output, loss, every named
parameter gradient, and one BF16 SGD update (`tests/gpu/fa4/test_varlen_attention.py:375-479`).
The current accepted-boundary evidence records all seven gradients and all seven
updates passing, with every parameter changing (`reviews/K001/test_report.json:49-79`).
The same test also verifies valid-boundary cross-sample isolation; the targeted
RTX 5090 run below exercises the forged/mutated negative cases, 17 image buckets,
native FA4 forward/backward, and full-module comparisons.

Current evidence reports 20 warmups, 100 measured calls, mask-free dense timing,
16-block entry-inclusive timing, allocated/reserved memory, and profiler counts of
16 FA4 kernels plus one D2H and one H2D per packed forward with no per-block boundary
copy (`reviews/K001/fa4_benchmark.json`; `benchmarks/attention/benchmark_fa4_varlen.py:444-645`).
The recorded current measurements are FA4/dense CUDA Event `0.216326/0.285460 ms`
(`1.3196x`) and synchronized wall `0.288413/0.317405 ms` (`1.1005x`); these are
single-GPU measurements only.

## Independent verification

- `PYTHONPATH=src .venv/bin/python -m pytest -q --basetemp=/tmp/sakuramoon-K001-ai-review tests/unit/model/test_fa4.py tests/gpu/fa4/test_varlen_attention.py`
  -> **22 passed** in 17.31s on one NVIDIA GeForce RTX 5090 (2 NVML warnings only).
- Targeted Ruff: clean.
- Targeted Pyright: `0 errors, 0 warnings`.
- `tools/verify_traceability.py --format json`: `222/222` requirements/source nodes,
  zero errors.

## Remaining blocked scope

- Governed fixed FA4 upstream repository commit/tree/license provenance and the
  associated four-column algorithm audit remain blocked; the wheel hash cannot close
  this item.
- This rereview does not run or close four-GPU integration, DDP, NCCL, formal stage
  canaries, 1,000-step canaries, endurance, or production training/quality gates.
  Single-GPU FA4 correctness and timing cannot be extrapolated to those gates.
