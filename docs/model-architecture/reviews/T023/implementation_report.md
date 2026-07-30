# T023 implementation report

The style resampler consumes Artist indices from the shared Qwen output and emits exactly four always-valid style tokens. Missing/dropout samples bypass attention and use the learned null tokens directly. Real RTX 5090 BF16 forward/backward found and fixed a null/active dtype mismatch; output now follows the Qwen input dtype with FP32 parameter masters retained.
