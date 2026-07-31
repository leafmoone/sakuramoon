# K001 independent AI/model correctness review

Verdict: **CHANGES_REQUIRED**

Reviewed HEAD: `c7b2e90a293aaaddb39015d0ab4c86f6b7c9af39`. M037 and the concurrent Data
worktree changes are outside K001 and were not modified. This review only changes this
file.

## Findings

### K001-AI-001 (high): mutable CUDA boundaries bypass the claimed validation

`ValidatedCuSeqlens` is a frozen dataclass, but its `tensor` remains mutable in place
(`src/sakuramoon/conditioning/packing.py:25-56`). The production guard checks host
metadata plus tensor shape/dtype/contiguity, but never checks that the CUDA values still
equal the cumulative `sequence_lengths` (`src/sakuramoon/model/attention.py:61-75`).
The locked FA4 interface likewise validates shape/dtype/contiguity only; its kernel reads
offset and length directly from `cu_seqlens[batch]` and `cu_seqlens[batch+1]`.

A real RTX 5090 call constructed the valid host contract `(2, 2)`, mutated only
`boundaries.tensor[1]` from `2` to `3`, and then called `fa4_varlen_attention`. The call
was accepted with host lengths `(2, 2)`, CUDA offsets `[0, 3, 4]`, and stale
`max_seqlen=2`; the first declared sample's two output tokens changed by a maximum
absolute `3.046875`. This silently changes cross-sample isolation and can also pass a
`max_seqlen` smaller than the actual CUDA sequence length to the native kernel.

The forged-handle tests do not cover this case: they only corrupt dtype, shape,
contiguity, or internally inconsistent host metadata
(`tests/gpu/fa4/test_varlen_attention.py:255-297`). The required remediation is to
validate CUDA boundary contents against host-derived cumulative offsets once at the
packed-batch entry, reuse only that accepted handle across blocks, derive token/sample
indexing from the same host-validated lengths, and add a content-mutation regression
that hard-fails before the native kernel. It must not add a per-block D2H sync.

### K001-AI-002 (medium): current evidence describes a removed validation path

The implementation report claims that `validate_cu_seqlens` performs one D2H content
validation and rejects nonzero starts, wrong terminals, non-increasing boundaries, and
wrong `max_seqlen` (`docs/model-architecture/reviews/K001/implementation_report.md:5`).
That function was removed by T024; the current factory creates CUDA offsets from host
lengths and the runtime only applies the static checks described above. The review task
repeats the obsolete D2H claim
(`docs/model-architecture/reviews/K001/task.md:5`), while `test_report.json` still says
`cu_seqlens_validated_once_before_blocks=true` and `malformed_boundaries_rejected=true`
(`docs/model-architecture/reviews/K001/test_report.json:32-33`). These claims currently
overstate the protection that exists and must be corrected with the boundary fix.

The test split is also stale after T024: the current suite is 4 CPU plus 12 GPU tests,
not the recorded 7 plus 9, although the total remains 16.

## Correctness matrix

The FA4 runtime dependency is the locked `flash-attn-4==4.0.0b24` wheel (SHA-256
`c1dcf0dfcf37c4496728547dcb3c1e66d7dcaa07cfedef0f26ccc4d74453951f` in
`uv.lock:208-222`). R001 does not register a separate FlashAttention reference-repository
commit, so this review makes no fixed-reference-commit equivalence claim and did not
read or execute `reference/`. The package source below was read statically from that
locked installed wheel; production/GPU tests execute the normal locked dependency.

| Plan formula/contract | Locked upstream package source | SakuraMoon implementation | Golden/contract test |
|---|---|---|---|
| Flat varlen Q `[T,20,128]`, K/V `[T,5,128]`, native GQA | `flash_attn/cute/interface.py:2819-2856` accepts total-token Q/K/V and `(batch+1)` boundaries | `src/sakuramoon/model/attention.py:49-54,90-99` passes native 20Q/5KV and `pack_gqa=True`; no KV repeat | `tests/unit/model/test_fa4.py:31-47`; `tests/gpu/fa4/test_varlen_attention.py:116-182` |
| Full bidirectional attention within each sample; no cross-sample attention | locked interface exposes `causal=False` at `interface.py:2834`; kernel derives offsets/lengths from CUDA entries at `seqlen_info.py:32-42` | `src/sakuramoon/model/attention.py:94-101` uses the same Q/K boundaries and `causal=False` | Valid-boundary isolation passes at `tests/gpu/fa4/test_varlen_attention.py:184-199`; mutable-boundary negative case is missing and fails review |
| CUDA BF16 production inputs; no silent dense fallback | locked interface accepts BF16 at `interface.py:403-423` | `src/sakuramoon/model/attention.py:55-88,289-292,334-344` hard-fails wrong dtype/device/import; dense SDPA is a separate module | `tests/unit/model/test_fa4.py:50-76`; real GPU suite below |
| Q/K FP32-accumulating RMSNorm, then 2D RoPE, then FA4; content gate/output projection after attention | FA4 accepts already projected Q/K/V; it does not own SakuraMoon's normalization/RoPE contract | `src/sakuramoon/conditioning/rope.py:104-130`; call order at `src/sakuramoon/model/attention.py:346-357` | Identical-state full-module output/loss/all-parameter-grad/update comparison at `tests/gpu/fa4/test_varlen_attention.py:311-415` |
| Dense SDPA is an explicit numerical reference, with noncausal GQA and valid-query masking | N/A: SakuraMoon reference contract | `src/sakuramoon/model/attention.py:190-256`; production module never calls it | Core and full-module dense comparisons at `tests/gpu/fa4/test_varlen_attention.py:116-182,311-415` |

No disagreement was found in native GQA head layout, BF16 Q/K/V, Q/K norm-before-RoPE,
noncausal attention, content-gate ordering, explicit fallback separation, or the valid
boundary output/loss/gradient/update comparisons. Those checks do not compensate for
K001-AI-001.

## Verification

- `PYTHONPATH=src .venv/bin/python -m pytest -q tests/unit/model/test_fa4.py tests/gpu/fa4/test_varlen_attention.py` -> `16 passed in 17.70s` on one NVIDIA GeForce RTX 5090.
- `PYTHONPATH=src .venv/bin/python -m pytest -q tests/unit/model/test_fa4.py` -> `4 passed in 2.65s`.
- GPU collection confirms 12 current tests in `tests/gpu/fa4/test_varlen_attention.py`.
- Targeted real-kernel mutation diagnostic -> call accepted with host lengths `(2,2)`, CUDA offsets `[0,3,4]`, and first-declared-sample max absolute output difference `3.046875`.
- `PYTHONPATH=src .venv/bin/python tools/verify_traceability.py --format json` -> `ok=true`, 222/222 requirements/source nodes, zero errors.
- Static review covered the locked wheel interface/signature, varlen sequence-length reads, local attention/packing/RoPE/block/DiT call chain, K001 benchmark and test reports, K001 trace mappings, and the current architecture/roadmap/open-item contracts.

## Remaining blocked scope

This review does not run or close four-GPU integration, DDP, NCCL, T041 equivalence,
formal stage canaries, 1,000-step canaries, endurance, or training long runs. Existing
single-GPU numerical and performance evidence remains single-GPU evidence only and
cannot replace any four-GPU gate. Independent Infra review is still required after the
affected implementation/evidence is remediated.
