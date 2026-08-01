# K001 implementation report

The production attention module projects flat `[T,2560]` BF16 tokens directly to `[T,20,128]` Q and `[T,5,128]` K/V, applies Q/K normalization plus 2D RoPE, and calls FA4 varlen with the original CUDA int32 cumulative sequence boundaries. It requests packed GQA in the kernel and never expands K/V heads. Import or contract failures stop execution; dense SDPA remains a separately selected reference path.

T024 now constructs the public `ValidatedCuSeqlens` only from a nonempty tuple of
positive host lengths. At the PackedDiT entry, `accept_fa4_boundaries` checks its
host/static metadata and performs the one required D2H content comparison against
host-derived cumulative offsets. A mismatch such as host `(2,2)` with CUDA
`[0,3,4]` hard-fails before native import. Acceptance rematerializes private CUDA
offsets from the same host tuple behind a capability-protected handle. Sample routing
uses that tuple, while all blocks reuse the accepted handle, so there is no per-block
D2H and later mutation of the discarded public tensor cannot alter FA4 boundaries.

The full `FA4VarlenGQAAttention` module is tested against `DenseGQAAttention` loaded with the identical state. The test covers output, loss, every named parameter gradient, and one real BF16 SGD update, with a same-FA4-backend repeat control used to derive p99 tolerance terms. Q projection weights are asserted to change after the update.

The reproducible benchmark command and raw results are recorded in
`fa4_benchmark.json`, with the earlier schema-v2 measurements retained as historical
data. Cold compilation is excluded, followed by 20 warmups. Pure device timing uses
one CUDA Event interval around 100 calls; end-to-end call timing uses `perf_counter`
and a synchronization after every call. Both methods compare FA4 with separate
per-sample SDPA calls using `attn_mask=None`. A full-True-mask SDPA path remains
correctness-only. Current evidence adds allocated/reserved memory and 50 synchronized
16-block packed-forward measurements. A five-forward profiler contract observed 80
FA4 kernels, 5 D2H and 5 H2D copies: exactly 16 kernels plus the two entry copies per
forward, with no boundary copy between blocks.

Environment reproducibility is supplied by the locked `flash-attn-4==4.0.0b24` wheel
and SHA-256. The separate governance layer now pins the official Dao-AILab tag
`fa4-v4.0.0.beta24` to commit
`849f660f73b176e5ad5670e7f822c7fa9f3eaf8b`, root tree
`dbc07053f34000ba50274ad7fbb51ff5411f9ff0`, FA4 package subtree
`ac02fb1b8e90985e7b88ff0916fa326f4e0d4227`, relevant source blobs, and the
BSD-3-Clause license digest. Upstream `setuptools_scm` proves the tag normalizes to
the locked distribution version. The fixed-source comparison covers the padding-free
varlen interface, native divisible GQA, explicit packed GQA, BF16/CUDA/int32 contracts,
noncausal execution, and autograd. The upstream checkout was never imported,
executed, installed, or copied into production.

Independent AI/model and Infra/license/reproducibility reviewers reproduced the
governed source identity, source and license digests, locked distribution binding, and
static algorithm contracts and returned PASS for the implemented CPU/single-GPU scope.
Historical blocked evidence remains unchanged; the completed provenance remediation is
recorded separately in the current lock, test report, reviews, and trace mappings.
