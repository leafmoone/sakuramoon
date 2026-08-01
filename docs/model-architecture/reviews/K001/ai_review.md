# K001 independent AI/model provenance closure review

Reviewer: independent agent `/root/k001_ai_provenance_review`.

Scope: the staged K001 upstream-provenance remediation at
`HEAD=4143a605d70491ab39365b8b2e3fc2017862ed4a`. The shared worktree's DATA,
ENCODERS, and M037 review outputs were excluded. This review only updates this file;
it did not modify implementation, traceability, task, lock, test, benchmark, or
historical evidence and did not create a commit. No `.env`, `reference/`, or archived
architecture file was read, imported, or executed. The governed upstream audit source
was inspected only through static Git/HTTP reads and was never imported, installed, or
executed; the separately locked runtime wheel was already present in the environment.

## Verdict

**PASS for AI/model correctness of the K001 provenance closure.** The prior governed
fixed-commit provenance blocker is closed by the new immutable lock and four-column
comparison. K001's AI verification may be set to `verified` for the implemented CPU
and single-RTX-5090 scope after the independent Infra review also passes.

No blocking AI/model finding remains. The first matrix row cites
`pyproject.toml:1-37` although the exact `setuptools_scm` tag regex is at lines 50-53;
this is a non-blocking locator imprecision because the fixed file blob/SHA-256 and the
independent static read both establish the stated rule.

## Governed source identity

- Official `git ls-remote` maps lightweight tag `fa4-v4.0.0.beta24` exactly to
  `849f660f73b176e5ad5670e7f822c7fa9f3eaf8b`. GitHub's commit object maps that commit
  to root tree `dbc07053f34000ba50274ad7fbb51ff5411f9ff0`; the root `flash_attn` tree then
  maps `cute` to subtree `ac02fb1b8e90985e7b88ff0916fa326f4e0d4227`.
- The subtree independently returns the exact locked blob IDs for `pyproject.toml`,
  `interface.py`, `flash_fwd.py`, `pack_gqa.py`, `seqlen_info.py`, and `LICENSE`.
  Static raw-file SHA-256 values match every digest in
  `upstream_provenance_lock.json`, including BSD-3-Clause license
  `8c9ccb96c065e706135b6cbad279b721da6156e51f3a5f27c6b3329af9416d73`.
- The installed locked wheel's `interface.py`, `flash_fwd.py`, `pack_gqa.py`,
  `seqlen_info.py`, and `flash_bwd.py` statically match the governed commit byte for
  byte. Installed metadata reports `flash-attn-4==4.0.0b24`, the official repository,
  and BSD-3-Clause; its installed `LICENSE` and `AUTHORS` digests also match the
  governed source. This is a direct source comparison, not an inference from the wheel
  hash alone.
- The fixed `pyproject.toml` names `flash-attn-4`, uses `setuptools_scm` rooted at the
  repository root, and accepts `fa4-v*` tags. Because the tag points exactly at the
  governed commit, `fa4-v4.0.0.beta24-0-g849f660` normalizes to the separately locked
  distribution version `4.0.0b24`.

## Algorithm and local-contract comparison

- The fixed upstream interface consumes flat Q/K/V plus CUDA int32
  `cu_seqlens_q/k`, requires Q-head divisibility by KV heads, derives the GQA ratio,
  and carries `pack_gqa` into the packed layout without K/V head expansion. Its
  autograd entry routes through `FlashAttnVarlenFunc`, and the installed backward
  source also matches the governed commit.
- SakuraMoon narrows that interface to contiguous CUDA BF16 `[T,20,128]` Q and
  `[T,5,128]` K/V, one accepted boundary identity, `causal=False`, and
  `pack_gqa=True`. Q/K FP32-accumulating RMSNorm and `32/48/48` 2D RoPE precede FA4;
  the sigmoid content gate and output projection follow it. Native import or contract
  failure hard-stops, and dense SDPA remains a separately selected reference rather
  than a silent fallback.
- Existing RTX 5090 evidence covers real forward/backward, cross-sample isolation,
  malformed and mutated boundaries, all 17 image shapes at three text boundaries,
  identical-state full-module output/loss/all-seven-gradient/all-seven-update
  comparison, and one accepted handle reused across 16 blocks without per-block D2H.
  The provenance remediation changes no implementation or numerical tolerance.

## Historical evidence and trace state

The historical top-level
`"upstream_repository_commit_provenance": "blocked_not_governed"` and prior
independent-review strings remain intact in `test_report.json`. The remediation is
appended in a separate object with both new reviews still pending. Trace requirements
`C05-003`, `C06-006`, and `OPEN-043` retain their stable identities and
`implemented` status; review/evidence fields remain empty pending completion of both
new independent reviews. Registry revision advances only from 100 to 101 for this
staged remediation.

## Independent verification

- Official static Git/HTTP identity audit: tag, commit, root tree, FA4 subtree, six
  governed blobs, source SHA-256, license, authors, and installed source equality all
  passed.
- `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/verify_traceability.py --format json`
  -> **237/237** requirements/source nodes, zero errors.
- `git diff --cached --check` and JSON parsing of the K001 lock/test/benchmark
  artifacts passed; no archived path is staged.
- The existing targeted K001 evidence remains **22 passed** on one RTX 5090, with
  Ruff clean and Pyright `0 errors, 0 warnings`. This provenance-only review did not
  rerun the GPU suite or rewrite its recorded results.

## Remaining blocked scope

This closure does not establish four-GPU FA4/DDP/NCCL correctness or performance,
1,000-step/endurance evidence, a formal stage canary, production training throughput,
or quality/FID/IS release evidence. Those gates remain pending/blocked and cannot be
closed from the current single-GPU results.

---

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
