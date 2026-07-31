# K001 independent Infra/performance rereview

Verdict: **PASS for the accepted-boundary remediation and the recorded 1GPU
evidence.** The FA4 production gate remains **BLOCKED** until a governed fixed
upstream repository commit and license/provenance lock exists. The locked wheel
hash is reproducible environment evidence, but it is not repository provenance.

## Scope and review basis

This rereview covers the current K001 worktree remediation, the T024 accepted
boundary implementation, the benchmark source, and the K001 task and evidence
files. The concurrent Data, T020/T021, and AI-review work was not modified. No
GPU command was run by this reviewer while the independent AI reviewer used the
RTX 5090; the reported RTX 5090 measurements below were checked against the
source, JSON schema, and static call chain.

Static checks run by this reviewer:

- `uv run --frozen ruff check benchmarks/attention/benchmark_fa4_varlen.py src/sakuramoon/model/attention.py src/sakuramoon/model/dit.py src/sakuramoon/conditioning/packing.py` - passed.
- `uv run --frozen pyright benchmarks/attention/benchmark_fa4_varlen.py src/sakuramoon/model/attention.py src/sakuramoon/model/dit.py src/sakuramoon/conditioning/packing.py` - 0 errors, 0 warnings.
- `jq empty` on the current K001 benchmark, test, and timing JSON - passed.
- Source scan confirms the deleted `validate_cu_seqlens` API is absent from the
  benchmark and production source. Remaining references are historical finding
  text in prior review records only.

## Finding disposition

### K001-INFRA-001 - mutable CUDA boundary isolation

**Resolved.** `PackedDiT.forward_packed_features()` accepts the public
`ValidatedCuSeqlens` exactly once before the block loop
(`src/sakuramoon/model/dit.py:518-525`). `accept_fa4_boundaries()` validates
host metadata and performs one CUDA-to-host content comparison
(`src/sakuramoon/model/attention.py:82-129`), then rematerializes private CUDA
offsets behind the accepted capability. Later public-tensor mutation cannot
reach the native kernel.

`accepted_sample_indices()` derives routing from the same accepted host length
tuple (`src/sakuramoon/model/attention.py:133-153`). Every active block receives
that same accepted handle (`src/sakuramoon/model/dit.py:533-543`), and
`FA4VarlenGQAAttention` passes only its private tensor to FA4
(`src/sakuramoon/model/attention.py:194-237`). Static inspection found one
intentional boundary `.to(device="cpu")` at the packed entry and no block-level
`.to(cpu)`, `.tolist()`, or `.item()` use.

The current tests/evidence cover forged static metadata, forged `[0,3,4]`,
post-construction mutation, and rejection of an unaccepted public handle before
native import. The valid-boundary cross-sample isolation and full PackedDiT
16-block path are also covered by the recorded current GPU suite.

### K001-INFRA-002 - stale benchmark and missing remediation evidence

**Resolved.** `benchmarks/attention/benchmark_fa4_varlen.py` constructs and
accepts boundaries through the current API and no longer imports the deleted
validator. `fa4_benchmark.json` retains the schema-v2 historical measurements
and appends a distinct accepted-boundary remediation section, rather than
rewriting history. The current `test_report.json`, `implementation_report.md`,
and `K001.md` describe the current 6 CPU plus 16 GPU test split and the new
entry/16-block evidence.

### K001-INFRA-003 - upstream repository commit provenance

**Still blocked as required.** `flash-attn-4==4.0.0b24` and its SHA-256 are
recorded, but no governed fixed upstream repository commit/license lock is
present. The benchmark and review evidence explicitly retain
`blocked_not_governed`; no source from `reference/` was read, imported, or
executed. This blocks the requested four-column upstream algorithm audit and
FA4 production release, but it is not an implementation defect in this
remediation.

## Infra/performance evidence

The current formal command is:

`PYTHONPATH=src uv run python benchmarks/attention/benchmark_fa4_varlen.py --warmup 20 --repeats 100`

Recorded RTX 5090 results at sequence lengths `(1028, 1540)` and 2,568 total
tokens:

| Measurement | FA4 | Dense reference | Result |
| --- | ---: | ---: | ---: |
| CUDA Event per call | 0.216326 ms | 0.285460 ms | 1.31958x FA4 speedup |
| Synchronized wall mean | 0.288413 ms | 0.317405 ms | 1.10052x FA4 speedup |
| Peak allocated | 31.349 MiB | 43.888 MiB | FA4 lower |
| Peak reserved | 36.0 MiB | 50.0 MiB | FA4 lower |

The dense performance reference is two separate per-sample SDPA calls with
`attn_mask=None`. The full-true-mask dense path is retained for numerical
correctness only and is excluded from timing comparisons.

The accepted-boundary 50-sample timing reports:

- entry acceptance p50: `0.036929 ms`;
- 16-block accepted-handle hot p50: `3.561176 ms`;
- 16-block entry-inclusive p50: `3.625846 ms`;
- entry-inclusive minus hot p50: `0.064669 ms`.

The five-forward profiler recorded 80 non-copy FA4 kernel events and 10
boundary-copy events: 16 kernels plus two entry copies per packed forward. The
copy names are exactly one `Memcpy DtoH (Device -> Pageable)` and one
`Memcpy HtoD (Pageable -> Device)` per entry path, with no copy between blocks.
The within-forward FA4 kernel-gap p95 is `0.289 us`. The benchmark source also
hard-fails on any unexpected kernel/copy event count, so a missing block kernel
or extra boundary transfer cannot silently produce the report.

The benchmark's 16-block sequence intentionally measures repeated FA4 calls
with one accepted handle and one final synchronization; it is a boundary/kernel
overhead probe, not a claim of a complete 16-block DiT end-to-end throughput
benchmark. The full PackedDiT 16-block forward/backward contract remains in the
targeted GPU evidence.

## Numerical and backend contract

The recorded remediation controls report true for output, loss, all seven
named parameter gradients, all seven parameter updates, and all parameters
updated against an identical-state dense module. Static review confirms native
20-query/5-KV head layout, `pack_gqa=true`, BF16 CUDA hard failures, Q/K
head-dimension normalization before 2D RoPE, noncausal attention, and no KV
head repetition or dense fallback in the FA4 production path. Import failure is
raised as an error; dense SDPA remains an explicit separately selected
reference path.

## Remaining blocked/pending gates

- Governed fixed FA4 upstream repository commit, license, and four-column
  algorithm audit remain blocked.
- Current evidence is single RTX 5090 evidence. Four-GPU DDP/NCCL equivalence,
  rank-failure behavior, and any multi-GPU throughput gate remain pending.
- No 1,000-step canary, formal stage, endurance, or long-run validation was
  performed. These remain pending and must not be inferred from this benchmark.
- The broader production data/cache cold-run throughput, ready-wait, RSS/swap,
  and final worker/queue sweep remain Data milestone gates.
- Existing all-17-shape current GPU contract evidence is retained, while this
  remediation benchmark itself measures the registered `(1028,1540)` shape pair;
  this is not a new full 17x8 performance sweep.

No new Infra finding requires a code change for the accepted-boundary
remediation. K001 may be committed as a remediation evidence update, with the
FA4 production status explicitly remaining blocked on the provenance gate above.
