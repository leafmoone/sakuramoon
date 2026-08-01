# K001 FA4 upstream algorithm and interface comparison

Governed source: official Dao-AILab FlashAttention repository tag
`fa4-v4.0.0.beta24`, commit
`849f660f73b176e5ad5670e7f822c7fa9f3eaf8b`, root tree
`dbc07053f34000ba50274ad7fbb51ff5411f9ff0`, and FA4 package subtree
`ac02fb1b8e90985e7b88ff0916fa326f4e0d4227`. The fixed BSD-3-Clause
license digest and relevant source blob/digests are recorded in
`upstream_provenance_lock.json`.

The upstream checkout was used only for static `git` and text reads. It was not
imported, executed, installed, vendored, or copied into SakuraMoon. The ignored
`reference/` directory was not read or used. Runtime reproducibility remains the
separate `uv.lock` pin for `flash-attn-4==4.0.0b24` and its sdist/wheel hashes.

| Contract | Fixed upstream implementation | SakuraMoon implementation | Evidence and result |
|---|---|---|---|
| Release identity | `flash_attn/cute/pyproject.toml:1-37,50-53` names `flash-attn-4`, uses `setuptools_scm`, and accepts only `fa4-v*` tags. Exact `git describe` is `fa4-v4.0.0.beta24-0-g849f660`, which normalizes to distribution version `4.0.0b24`. | `pyproject.toml` and `uv.lock` require exactly `flash-attn-4==4.0.0b24`; the locked wheel and sdist hashes remain unchanged. | Official `git ls-remote` maps the tag to the governed commit; commit, root tree, subtree, source blobs, and license digest are fixed in the lock. **PASS** |
| Padding-free varlen boundary | `flash_attn/cute/interface.py:2819-2906` exposes `flash_attn_varlen_func` with flat Q/K/V, `cu_seqlens_q/k`, maximum sequence lengths, and no dense adapter. | `src/sakuramoon/model/attention.py:174-239` passes the private accepted CUDA int32 offsets directly as both Q/K boundaries with exact maximum length. | Forged/mutated public boundaries fail before native import; valid real FA4 output and cross-sample isolation pass. **PASS** |
| Native 20Q/5KV GQA and `pack_gqa` | `flash_attn/cute/interface.py:377-463` accepts distinct Q and KV head counts, requires Q heads divisible by KV heads, derives the ratio, and enables packed GQA for ratios greater than one. `flash_attn/cute/flash_fwd.py:33-93,363-370,676-678` carries the packed layout without expanding K/V heads. | `src/sakuramoon/model/attention.py:182-214,227-239` requires `[T,20,128]` Q and `[T,5,128]` K/V and calls upstream with `pack_gqa=True`; there is no KV repeat path. | Real RTX 5090 forward/backward, all seven parameter gradients, one update, 17 bucket shapes, and explicit `pack_gqa=true/false` timing favoring true are recorded. **PASS** |
| BF16 CUDA and int32 offsets | `flash_attn/cute/interface.py:393-450` validates varlen shapes, matching float dtypes, contiguous int32 boundaries, CUDA residency, and head divisibility. `flash_attn/cute/flash_fwd.py:177-203` restricts kernel Q/K/V/O to FP16/BF16 and boundaries to int32. | `src/sakuramoon/model/attention.py:182-214` narrows production to contiguous CUDA BF16 Q/K/V and accepted same-device int32 boundaries, with no fallback. | Unit/GPU negative contracts cover dtype/device/shape/contiguity and boundary corruption. **PASS** |
| Noncausal full attention and autograd | `flash_attn/cute/interface.py:2833-2840,2877-2905` forwards explicit `causal`, `pack_gqa`, and autograd inputs through `FlashAttnVarlenFunc`; the governed package contains matching backward kernels. | `src/sakuramoon/model/attention.py:227-239` fixes `causal=False`, `pack_gqa=True`; Q/K normalization plus 2D RoPE run before the call and content gate/output projection run after it at lines 483-494. | Identical-state dense comparison passes output, loss, every named gradient, and every BF16 SGD update. No silent dense fallback exists. **PASS** |
| One accepted boundary per packed forward | Upstream consumes the supplied boundaries per call and does not govern SakuraMoon's public-to-private capability boundary. | `src/sakuramoon/model/dit.py:505-543` accepts once, derives routing from the same host identity, and reuses the private handle through all active blocks. | The 16-block profile records one D2H plus one H2D at entry and zero per-block boundary transfers. **PASS (SakuraMoon boundary extension)** |

This comparison closes the fixed-upstream repository provenance gap only. Existing
single-GPU numerical and performance evidence remains bounded to one RTX 5090. It
does not close DDP/NCCL, four-GPU, 1,000-step, endurance, or formal stage gates.
