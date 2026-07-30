# T022 implementation report

The text conditioner implements the approved seven-layer grouped mixer and one bidirectional attention-only refinement. Unlocked head count and initialization values remain explicit constructor inputs rather than hidden defaults. Real RTX 5090 BF16 forward/backward passed with finite outputs and adapter gradients while the Qwen input stayed detached.
