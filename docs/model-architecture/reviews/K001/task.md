# K001 review task

Review direct varlen boundary handling, native 20Q/5KV GQA without K/V repetition, CUDA BF16 hard failures, Q/K norm and RoPE ordering, explicit separation from dense SDPA reference, cross-sample isolation, numerical output/loss/gradient/update evidence, and compile-versus-steady performance reporting.

Remediation review must also confirm that the packed-batch entry performs exactly one
D2H content validation of the public `ValidatedCuSeqlens`, rematerializes a private
accepted handle from the canonical host lengths, and reuses that handle across all
blocks without per-block D2H. Sample routing and FA4 must share that host identity.
The full FA4 module must be compared to an identical-state dense module for output,
loss, every parameter gradient, and one update. `fa4_benchmark.json` must retain
historical results while recording current batched CUDA Event and per-call synchronized
wall timing, 16-block entry-inclusive timing, allocated/reserved memory, and profiler
copy/kernel/gap evidence. Performance dense SDPA must be per-sample and mask-free; an
all-True mask is numerical-reference-only. The fixed-upstream-commit audit must verify
the official tag-to-version rule, exact commit and tree, relevant source blobs,
BSD-3-Clause license digest, and the static varlen/GQA/BF16/noncausal/autograd
comparison. It must not infer repository provenance from the wheel hash, import or
execute upstream code, or read/import/execute `reference/`.
