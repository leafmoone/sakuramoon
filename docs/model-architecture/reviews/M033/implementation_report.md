# M033 implementation report

The objective keeps model prediction semantics as clean latent x and derives velocity only for the loss and CFG. All velocity arithmetic and reductions are FP32. The sampler enters inference mode internally, evaluates 49 predictor/corrector intervals plus one final Euler interval, and checks the final state once instead of synchronizing after every model evaluation.
