# M030 implementation report

`GlobalConditioner` implements the global canvas/time condition without an asset layer or runtime fallback. It accepts the target-canvas scalars already defined by T024, keeps all condition parameters and math in FP32, and exposes named block and final modulation tensors. Block and final projections are separate and zero-initialized. Its per-forward contract checks tensor metadata without a GPU `.item()` synchronization; value range is validated once by the training or solver boundary that creates timesteps.

The evidence remediation did not change production code. It added hard-coded nonzero
frequency goldens, an exact T024 `canvas_condition` to M030 384-dimensional input
integration golden with equal conditional/unconditional canvas rows, explicit coverage
of the six modulation chunks in their locked order, and an autocast test proving that
condition parameters and outputs remain FP32. A fresh single-RTX-5090
production-dimension forward/backward also passed with finite gradients. Package review
remains pending; no multi-GPU or long-running training evidence was claimed.
