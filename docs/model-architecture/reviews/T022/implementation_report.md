# T022 implementation report

The text conditioner implements the approved seven-layer grouped mixer and one bidirectional attention-only refinement. Unlocked head count and initialization values remain explicit constructor inputs rather than hidden defaults. Real RTX 5090 BF16 forward/backward passed with finite outputs and adapter gradients while the Qwen input stayed detached.

Masked main-token indices are now replaced with a safe gather index before the frozen
Qwen tensor is read; active negative or oversized indices still hard fail. A production
factory locks the approved 2048/1024/2560 dimensions, eight groups, FP32 normalization,
and BF16/FP32 parameter precision. MHA heads, mix-gate initialization, LayerScale
initialization, and projection bias remain required keyword-only inputs because the
current decisions do not lock them. CPU contracts and a production-shape RTX 5090
BF16 forward/backward passed. Encoders/Conditioning package review remains pending.
