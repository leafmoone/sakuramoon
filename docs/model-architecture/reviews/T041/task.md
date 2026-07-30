# T041 review task

Review the single-GPU trainable boundary, unequal-microbatch sample weighting, gradient normalization order, FP32 clip, optimizer/zero-grad boundary and immutable counters. Confirm no Qwen/VAE ownership, flat packed-token averaging, DDP/NCCL stub or per-microbatch host synchronization was introduced. Keep all four-rank claims pending.
