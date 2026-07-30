# T041 AI review

Status: PASS; no blocking findings.

The independent review confirmed that every microbatch backpropagates the FP32 sum of its already per-sample-mean losses and that gradients are divided exactly once by the total effective sample count at the update boundary. Unequal microbatch and variable-element tests exclude mean-of-means, packed-token weighting and element weighting.

`TrainableComposite` registers only `dit`, `text` and `style`; Qwen and Mage-VAE remain outside the module, optimizer and checkpoint boundary. Attempted updates advance before the optimizer call, while successful updates and lifetime effective samples advance only after optimizer step and gradient clearing both succeed. A failed attempt terminates the step object.

The reviewer ran the 10 T041 CPU tests, a 25-test objective/clip/train regression, Ruff, strict Pyright and `git diff --check`; all passed. The existing RTX 5090 TorchAO evidence was inspected without rerunning it concurrently with the Infra reviewer. Four-rank global mean, state equality, rank failure and NCCL remain pending.
