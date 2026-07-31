# K001 independent Infra/performance review

Verdict: **CHANGES_REQUIRED**

Reviewed HEAD: `c7b2e90a293aaaddb39015d0ab4c86f6b7c9af39`. Concurrent Data,
Encoders, and K001 AI review worktree changes were excluded. This review only adds this
file.

## Findings

### K001-INFRA-001 (high): mutable device boundaries bypass the isolation contract

`ValidatedCuSeqlens` is a frozen dataclass, but its public CUDA tensor remains mutable
(`src/sakuramoon/conditioning/packing.py:25-56`). The production guard checks the host
length tuple and only the device tensor's rank, shape, dtype, contiguity, and device
(`src/sakuramoon/model/attention.py:61-75`). It never verifies the CUDA offset values.
Both the native kernel and `PackedDiT._sample_indices()` then consume those mutable
values (`src/sakuramoon/model/attention.py:90-102` and
`src/sakuramoon/model/dit.py:496-507`).

A real RTX 5090 probe built the valid host identity `(2,2)`, mutated only
`boundaries.tensor[1]` from 2 to 3, and observed device offsets `[0,3,4]`. The call
still passed `max_seqlen=2` even though the device-derived maximum was 3, the native
FA4 call completed, and production sample routing became `[0,0,0,1]`. With valid
boundaries, changing all K/V values in sample two changed sample-one output by exactly
`0.0`; with the mutated boundary, the same operation changed sample-one output by a
maximum absolute `27.265625`. This is a direct cross-sample conditioning and attention
isolation failure, not merely an adversarial constructor issue.

The issue can be fixed without a per-block D2H synchronization. The packed-batch entry
must establish one canonical boundary identity, derive both sample routing and native
offsets from it, and pass a locally scoped accepted handle through all blocks. The
current K001 review contract explicitly permits exactly one D2H content validation at
that entry. A host-authoritative design may instead rematerialize private CUDA offsets
once and never consume the externally reachable tensor, but it must still make the
post-construction mutation case fail before corrupted values reach the native kernel.
In either design, `_sample_indices()` must use the same host-validated lengths rather
than rereading device offsets.

A reviewer-only timing diagnostic over 16 FA4 calls at lengths `(1028,1540)` measured
the following RTX 5090 p50 wall times: no validation `3.5521 ms`, one entry D2H
`3.6175 ms`, and a D2H before every block `4.7486 ms`. A single three-entry D2H check
had `0.0146 ms` p50. These are not formal end-to-end performance evidence, but they
show why one entry validation is viable and why per-block validation is not. The final
remediation still needs a current small multi-block benchmark and profiler check.

### K001-INFRA-002 (medium): the recorded evidence is stale and its benchmark cannot run

The claimed reproducible command currently fails at import time because
`benchmarks/attention/benchmark_fa4_varlen.py:17-23` still imports the removed
`validate_cu_seqlens` API. The implementation report nevertheless says that API
performs one D2H validation (`docs/model-architecture/reviews/K001/implementation_report.md:5`),
and `test_report.json` still claims that boundaries are content-validated once and
malformed boundaries are rejected
(`docs/model-architecture/reviews/K001/test_report.json:32-33`). The current split is
4 CPU plus 12 GPU tests, not the recorded 7 plus 9, although all 16 current tests pass.

`fa4_benchmark.json` remains useful historical kernel data, but it is not reproducible
from the reviewed HEAD and predates the current boundary handoff. It also omits the
roadmap-required allocated/reserved memory fields. After K001-INFRA-001 is fixed, the
benchmark must use the current API, include the one-time boundary cost in a multi-block
measurement, rerun the numerical/output/loss/all-parameter-gradient/update controls,
record allocated and reserved memory, and update the evidence without deleting the
historical results.

### K001-INFRA-003 (medium): no fixed upstream repository commit is governed for FA4

R001's reference lock contains only HDM, JLT, and krea-2
(`docs/model-architecture/reviews/R001/reference_manifest.json:6-55`). The runtime
dependency is reproducibly pinned as the `flash-attn-4==4.0.0b24` wheel with SHA-256
`c1dcf0dfcf37c4496728547dcb3c1e66d7dcaa07cfedef0f26ccc4d74453951f`
(`uv.lock:207-223`), but neither R001 nor K001 records the corresponding upstream
repository commit. Therefore the required fixed-commit algorithm audit is blocked.
The wheel hash is sufficient for environment reproduction, but it is not a substitute
for the requested source-commit provenance. Establish that lock before claiming the
upstream four-column audit complete; do not import or execute `reference/`.

## Upstream and local contract matrix

The installed locked wheel was read statically. No code under `reference/` was read,
imported, or executed.

| Plan formula/contract | Fixed upstream commit/source | SakuraMoon implementation | Golden/contract test |
|---|---|---|---|
| Flat varlen BF16 Q `[T,20,128]`, K/V `[T,5,128]`, native GQA | **Commit provenance blocked.** Locked wheel source `flash_attn/cute/interface.py:2819-2856` accepts flat Q/K/V and `(batch+1)` boundaries | `src/sakuramoon/model/attention.py:49-60,90-101` passes 20Q/5KV and `pack_gqa=True` without KV repeat | `tests/unit/model/test_fa4.py:31-47`; `tests/gpu/fa4/test_varlen_attention.py:116-182` |
| CUDA boundaries isolate samples and are strictly increasing from zero to total tokens | Locked wheel `flash_attn/cute/seqlen_info.py:24-45` reads offsets and lengths directly from CUDA entries; its interface checks static shape/dtype/stride, not contents | Host metadata and CUDA values can diverge at `packing.py:25-56`; `attention.py:61-75` misses the divergence | Valid-input isolation passes at `test_varlen_attention.py:184-199`; the real mutation probe fails the contract and has no checked-in regression |
| `max_seqlen` equals the longest device-described sequence | Locked interface accepts explicit `max_seqlen_q/k` at `interface.py:2819-2848`, while device lengths come from CUDA offsets | `attention.py:67,96-97` validates host max only, so mutation produced host max 2 versus device max 3 | Existing forged tests at `test_varlen_attention.py:272-297` cover static metadata only and miss value mutation |
| Dense reference is explicit, noncausal, native GQA; performance dense calls are per-sample and mask-free | N/A: local comparison contract | `attention.py:190-256`; benchmark helper uses `attn_mask=None`, but the runner is currently broken | Current full-module dense correctness test passes; historical timing JSON cannot be regenerated from this HEAD |

No discrepancy was found in the normal-input native 20Q/5KV layout, BF16 hard failure,
Q/K norm then RoPE ordering, content-gate/output ordering, or explicit separation from
dense SDPA. Those passing paths do not mitigate the mutable-boundary failure.

## Independent verification

- `PYTHONPATH=src .venv/bin/python -m pytest -q tests/unit/model/test_fa4.py tests/gpu/fa4/test_varlen_attention.py` -> `16 passed in 18.22s` on one NVIDIA GeForce RTX 5090.
- Current benchmark smoke command -> import failure for removed `validate_cu_seqlens` before any benchmark executes.
- Real FA4 mutation probe -> accepted host `(2,2)` with CUDA `[0,3,4]`, stale max 2 versus actual max 3, and first-declared-sample boundary-change delta `3.1328125`.
- Real FA4 cross-sample probe -> valid-boundary sample-one delta `0.0`; corrupted-boundary delta `27.265625`; all four native calls completed.
- Exact production `_sample_indices()` probe with the mutated handle -> `[0,0,0,1]`.
- Static review covered packing, attention, block and PackedDiT routing, current GPU tests, benchmark source/results, dependency lock, R001 reference lock, K001/T024 task evidence, and trace mappings.

No long run, 1,000-step canary, multi-GPU, DDP, NCCL, or formal stage test was run.
Single-GPU correctness and timing cannot close any four-GPU or formal stage gate.
