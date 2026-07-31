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

## M035-M036 remediation review

Reviewer: independent agent `/root/dense_remediation_review`

Date: 2026-07-31

Scope: committed M035/M036 remediation evidence and the M033 objective/M034 sampler
code that evidence qualifies. The review used static `git show` reads of the R001-locked
JLT commit `aca236efa97aab3b7d865fd3d99a270431cf6ae5`; no reference code was imported or
executed.

| Task | Verdict | Basis |
|---|---|---|
| M035 | PASS | The four-column comparison checks the actual timestep distribution, noise-to-clean direction, x-prediction parameterization, shared `max(1-t,0.05)` conversion, inverse-square loss, sample-first reduction, FP32 override, observation boundary, and per-branch x-to-v CFG order. The cited locked-reference and local lines support each claim. It also correctly limits the M033 CUDA evidence to a standalone Parameter/SGD/component solver smoke. |
| M036 | PASS | The non-autonomous `dz/dt=t` golden forces the Heun corrector to use `t_next`; for 50 intervals with a final Euler update it correctly expects 99 NFE, no model evaluation at `t=1`, and `z(1)=0.5-1/(2*50^2)=0.4998`. The CPU and CUDA tests exercise the same analytic contract with FP32 solver state. |

No blocking AI/model-correctness finding was found. M035 does not close T050's real
data/Qwen/VAE/DiT/loss/checkpoint smoke, and M036's analytic velocity function does not
constitute checkpoint-driven generation or formal quality evidence. T041 and every
multi-GPU, DDP, and NCCL conclusion remain pending/blocked under their owning tasks.
