# K001 review task

Review direct varlen boundary handling, native 20Q/5KV GQA without K/V repetition, CUDA BF16 hard failures, Q/K norm and RoPE ordering, explicit separation from dense SDPA reference, cross-sample isolation, numerical output/loss/gradient/update evidence, and compile-versus-steady performance reporting.

Remediation review must also confirm that `ValidatedCuSeqlens` performs exactly one D2H validation at the packed-batch entry, all blocks reuse the immutable-by-contract handle without a per-block D2H sync, the full FA4 module is compared to an identical-state dense module for every parameter gradient and one update, and `fa4_benchmark.json` records reproducible batched CUDA Event and per-call synchronized wall timing plus profiler kernel/gap evidence. Performance dense SDPA must be per-sample and mask-free; an all-True mask is numerical-reference-only.
