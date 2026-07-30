# T041 Infra review

Status: PASS; no blocking findings.

The independent review confirmed that the microbatch path has no `.item()`, explicit synchronization, packed-to-dense conversion or distributed call. The update boundary order is total-sample gradient scaling, FP32 global clip at 1.0, one optimizer step, then `zero_grad(set_to_none=True)`. The composite introduces no Qwen/VAE ownership, DDP/NCCL stub or cross-sample attention path.

The reviewer ran all 10 CPU tests and the real RTX 5090 TorchAO test, plus Ruff and strict Pyright; all passed. The production 16-layer composite has 2.574 GB of gradient tensors. CUDA Event measurement was 3.3434 ms for direct per-parameter scaling and 3.3431 ms for `torch._foreach_mul_`, showing that the operation is memory-bandwidth-bound and that foreach does not justify extra implementation complexity. Its end-to-end share remains T050/T053 work.

Four-GPU DDP/NCCL behavior, all-rank state equality and rank failure remain pending.
