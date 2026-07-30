# T041 implementation report

`TrainableComposite` registers only `dit`, `text`, and `style`, accepts frozen Qwen outputs and VAE latents as input values, and returns the per-sample image-span predictions already preserved by `PackedDiT`. Its full canonical parameter schema is identical to T040.

`SingleGpuStep.backward()` accepts only nonempty one-dimensional FP32 per-sample losses with a gradient graph. It calls backward on each microbatch sum without dividing by microbatch size, accumulates the detached loss sum and sample count on device, and rejects too few or too many microbatches.

At the update boundary it divides every present gradient once by the total effective sample count, calls the T040 FP32 clip, performs one optimizer step and clears gradients with `set_to_none=True`. Attempted state advances before the optimizer call; successful state and lifetime effective samples advance only after the full boundary succeeds. A failed attempt is terminal for that step object.

AI/model self-check: unequal microbatch and variable-element tests directly guard against mean-of-means and token-weighted loss. Infra self-check: the accumulator uses no `.item()` or synchronization per microbatch, does not materialize dense packed tokens, and adds no distributed path. The only existing host synchronization remains the once-per-update finite/clip check from T040.
