# Dense Model package AI/model correctness review

Reviewer: independent agent `/root/dense_package_review`

Date: 2026-07-31

Scope: M030-M034 frozen CPU/one-GPU implementation and evidence. The reviewer did not edit or commit repository files.

## Verdicts

| Task | Verdict | Basis |
|---|---|---|
| M030 | PASS | Fixed global-condition embeddings and FP32 MLP, shared 6d modulation, active-slot bias, independent final path, and required zero initialization are preserved. |
| M031 | PASS | Dense SDPA reference preserves native 20Q/5KV GQA, Q/K normalization before RoPE, unnormalized V, content gating, residual gating, and padding isolation. |
| M032 | PASS | Stable 16/20/24 slot FQNs, growth alpha-zero equivalence, direct 128-channel latent output, image-only zero-initialized head, and packed-sample isolation remain covered. |
| M033 | PASS | Strict shared-clamp JLT conversion, inverse-square weighting, sample-first reduction, t=0.5 observation buckets, and per-branch x-to-v before CFG are exact. |
| M034 | PASS | The three canonical profiles, derived NFE, explicit selection, formal reference binding, no t=1 model evaluation, and resolved/evaluation/generation identities match the locked decision. |

## Remaining boundaries

- T050 must bind the strict M033 objective into a real 1-10 step data/Qwen/VAE/DiT/loss/checkpoint engineering smoke; the current training loop still accepts an external loss function.
- T041 retains DDP global-sample-mean equivalence and all multi-GPU conclusions.
- M034's CUDA test executes the real solvers with a synthetic velocity function. A checkpoint-driven conditional/unconditional DiT, CFG, decoder, and quality workflow remains pending.
- K001 and formal stage/quality gates retain their independent reviews and evidence requirements.

No AI/model-correctness finding blocks the M034 atomic commit.
