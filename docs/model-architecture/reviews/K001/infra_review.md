# K001 independent Infra/license/reproducibility provenance rereview

Reviewer: independent agent `/root/k001_infra_provenance_review`.

Scope: the uncommitted K001 upstream-provenance remediation at
`HEAD=4143a605d70491ab39365b8b2e3fc2017862ed4a`. This review covers the official
tag/commit/tree identity, fixed source blobs and digests, BSD-3-Clause license,
tag-to-distribution version binding, PyPI artifact binding, static upstream-to-local
contract matrix, and trace/review reproducibility. It does not modify the implementation,
task, traceability registry, benchmark, test report, AI review, or concurrent DATA,
ENCODERS, and M037 review work.

No `.env`, Notion MCP, or `reference/` content was read. No upstream checkout was
imported, installed, or executed; the audit used a temporary bare Git object database
and static archive reads only. No archived architecture file was modified.

## Verdict

**PASS.** The governed fixed-upstream provenance remediation is internally consistent
and independently reproducible. The prior `K001-INFRA-003` blocker is resolved: the
official FA4 tag is fixed to an exact commit, root tree, FA4 subtree, source blobs,
license digest, and locked distribution artifacts. No Infra, license, or reproducibility
finding remains in this remediation.

This PASS closes only the K001 fixed-upstream provenance gap. It does not close the
separate four-GPU DDP/NCCL, formal-stage, 1,000-step, endurance, production data, or
quality gates.

## Immutable upstream identity

Official remote observation:

- repository: `https://github.com/Dao-AILab/flash-attention.git`;
- tag: `fa4-v4.0.0.beta24`;
- `git ls-remote --refs`: tag resolves exactly to
  `849f660f73b176e5ad5670e7f822c7fa9f3eaf8b`;
- commit root tree: `dbc07053f34000ba50274ad7fbb51ff5411f9ff0`;
- `flash_attn/cute` subtree: `ac02fb1b8e90985e7b88ff0916fa326f4e0d4227`;
- `git describe --tags --long`: `fa4-v4.0.0.beta24-0-g849f660`.

A depth-one fetch into a bare repository reproduced all three object identities. The
five governed source blob IDs and SHA-256 digests in
`upstream_provenance_lock.json` also reproduced exactly:

| Fixed path | Git blob | SHA-256 result |
|---|---|---|
| `flash_attn/cute/pyproject.toml` | `174f7db046120d8598555d338ce83ba68d5748de` | `1d73ea9937c404a0fb5948f4a3898c2fa671a490b40f90f4e9f229feb6b3dedc` |
| `flash_attn/cute/interface.py` | `0300179f173bc9759a1524e66750cfd536b432af` | `812a3fcc84ce0cd34401cd25c59f9dce5ef55a52b69a6e8659dca79155d8c40b` |
| `flash_attn/cute/flash_fwd.py` | `7d1593d7412a29268f192d933f41c44b4c34c5c6` | `7c50c12e46209270f47a2f687ea00788eb58026c9d13324fbe4382cf6426029d` |
| `flash_attn/cute/pack_gqa.py` | `5b481b5e6fc4d7f2d0d3d49f30aeb856e123da7a` | `82ad3a7c44ab4d7b0cffc248aebfedeeadacade8ae2eabb231fddf343e8755cd` |
| `flash_attn/cute/seqlen_info.py` | `7110c8f2b783e033140ae03e64fbd0dc9b8bf760` | `640d2433702635d68d4f0b94e9fa2d6b81da48b7f62056270b1b1fd480c72f30` |

The review did not rely on the lock's historical `checked_out_worktree_clean`
observation. A bare object fetch has no worktree and independently establishes the
immutable objects needed for this audit.

## License and distribution binding

The fixed `flash_attn/cute/LICENSE` is the complete BSD 3-Clause text. Its Git blob is
`5860e4b33f3d9d85fc636137c559331d51783a5b`; its independently recomputed SHA-256 is
`8c9ccb96c065e706135b6cbad279b721da6156e51f3a5f27c6b3329af9416d73`.
The AUTHORS digest also matches the lock at
`4627841c206c9bf990d37cc2ecbfa778a632d85731aafb4ccb59238334d2821d`.
The license is compatible with the current use, and no upstream source was vendored or
copied into SakuraMoon. Both locked distribution archives carry LICENSE and AUTHORS.

The fixed upstream `pyproject.toml` names `flash-attn-4`, declares BSD 3-Clause, uses
`setuptools_scm` with root `../..`, and restricts recognized tags with
`^fa4-v(?P<version>.+)$`. The exact tag version `4.0.0.beta24` normalizes under the
declared packaging rule to the locked `4.0.0b24` distribution version.

Official PyPI JSON independently returned the same repository URLs, BSD license,
version, filenames, URLs, and hashes recorded in the lock and `uv.lock`:

- sdist SHA-256: `6c4b981ef433882871ded48317deaa18ea22f731ca4f8b9387804bfa2e8078e2`;
- wheel SHA-256: `c1dcf0dfcf37c4496728547dcb3c1e66d7dcaa07cfedef0f26ccc4d74453951f`.

Static reads of those downloaded, hash-verified archives provided an additional binding:
LICENSE, AUTHORS, `interface.py`, `flash_fwd.py`, `pack_gqa.py`, and
`seqlen_info.py` in both the official sdist and wheel are byte-identical to the fixed
Git tag; the sdist `pyproject.toml` is also byte-identical. Thus the audit is not merely
inferring repository provenance from a wheel version or wheel hash.

## Static contract comparison

The four-column `upstream_algorithm_matrix.md` is accurate for the governed source:

- `flash_attn_varlen_func` accepts flat Q/K/V plus `cu_seqlens_q/k` and maximum
  lengths, then dispatches through `FlashAttnVarlenFunc.apply` for autograd.
- Upstream validates matching float dtypes, contiguous int32 sequence boundaries,
  CUDA residency, and Q-head divisibility by KV heads. Its lower kernel boundary
  restricts the production Q/K/V/O path to FP16 or BF16.
- The upstream head ratio is derived without expanding K/V. `pack_gqa` is explicit,
  and `flash_fwd.py` applies the packed Q/O layout while retaining native KV heads.
- `causal` is an explicit interface argument whose default is false. SakuraMoon fixes
  `causal=False`, `pack_gqa=True`, CUDA BF16 `[T,20,128]` Q and `[T,5,128]` K/V,
  with no K/V repeat or silent dense fallback.
- SakuraMoon performs Q/K normalization and 2D RoPE before the call, uses the same
  accepted host identity for routing and FA4, and reuses one private accepted boundary
  handle through all active blocks. This matches the matrix's locally governed
  packed-entry extension.

The upstream package contains the corresponding backward sources, while K001's existing
real RTX 5090 output/loss/all-gradient/update evidence establishes runtime behavior.
This provenance rereview did not rerun GPU work and does not broaden that evidence.

## Independent verification

- `uv run --frozen pytest -q --basetemp=cache/.pytest-K001-infra-provenance-review-20260801 tests/unit/docs/test_verify_traceability.py`
  -> **40 passed** in 73.85 seconds.
- `uv run --frozen python tools/verify_traceability.py --format json`
  -> `ok=true`, **237 requirements**, **237 source nodes**, zero errors.
- Targeted Ruff on the trace verifier and its tests -> **passed**.
- Targeted Pyright on the trace verifier and its tests -> **0 errors, 0 warnings**.
- `jq` parse of `upstream_provenance_lock.json`, `test_report.json`,
  `fa4_benchmark.json`, and `timing.json` -> **passed**.
- `git diff --check --cached` and `git diff --check` -> **passed**.
- Tracked production/test/benchmark/config/tool import scan -> no `reference/` import.

All temporary bare Git, sdist, wheel, and pytest directories used by this review were
task-private and removed after verification. No commit was created by this reviewer.

---

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
