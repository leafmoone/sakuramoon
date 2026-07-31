# T023 implementation report

The style resampler consumes Artist indices from the shared Qwen output and emits exactly four always-valid style tokens. Missing/dropout samples bypass attention and use the learned null tokens directly. Real RTX 5090 BF16 forward/backward found and fixed a null/active dtype mismatch; output now follows the Qwen input dtype with FP32 parameter masters retained.

Mixed batches now select active samples before gather, normalization, and input
projection. Inactive Artist positions are replaced with a safe gather index, while only
unmasked indices belonging to non-null samples are range checked. Null-routed samples
therefore perform no style projection, and masked large placeholder indices cannot
trigger an out-of-range gather. CPU hooks and a production-size RTX 5090 BF16
forward/backward prove the active projection batch size and null-token gradient path.

All constructor values remain explicit and self-describing in checkpoint metadata.
Resolved-config construction of the full production composite remains owned by T050;
this task adds no code defaults. Encoders/Conditioning package review is pending.
