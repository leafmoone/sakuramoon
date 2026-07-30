# T040 implementation report

Parameter grouping resolves each FQN to its owning module and explicit sensitive role. It rejects unrecognized ranked parameters, FP32 projection matrices outside the condition/output-head exceptions, BF16 sensitive parameters, aliases, and any BF16 decay matrix that cannot use 256-block quantized TorchAO state. Param groups carry canonical `param_names` in the same order as their parameters.

Text and style constructors now require explicit linear and sensitive dtypes. Their large projections are allocated as BF16; gate weights/biases, learned queries/layer embeddings/null tokens, norms, scalar gates and layer scale remain FP32. FP32 gate computations cast their result back to the BF16 activation path before BF16 projections.

The optimizer wrapper validates all gradients before entering an RNG guard. The guard saves the training CUDA RNG, installs the isolated SR state, runs exactly one TorchAO step, advances the SR state and restores the training RNG. State loading checks the canonical schema and requires the exact CPU uint8 RNG state shape. `audit_state()` reports lazy/initialized status, state class, physical bytes, step and block size by FQN and rejects any BF16 decay state fallback.

The original independent reviews blocked the first implementation because it inferred roles from dtype/ndim, omitted the full trainable composite, used final training loss instead of held-out validation EMA, weakly tested state restore and accepted corrupt SR state. All five findings have targeted regression tests in the remediated implementation, and both independent remediation rereviews passed without blockers.
