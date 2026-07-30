# M030 implementation report

`GlobalConditioner` implements the global canvas/time condition without an asset layer or runtime fallback. It accepts the target-canvas scalars already defined by T024, keeps all condition parameters and math in FP32, and exposes named block and final modulation tensors. Block and final projections are separate and zero-initialized.
