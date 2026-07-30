# T020 implementation report

The repository contains the checkpoint-compatible Mage encoder/decoder subset and a
frozen wrapper. The loader accepts only the prepared local safetensors file. Strict CPU
loading and real RTX 5090 BF16 encode/decode passed at 32x32 and 512x512.

The reconstruction evaluator now requires exactly 2,000 unique observations split into
1,600 stratified and 400 risk samples. It rejects non-finite or incorrectly typed
metrics and applies the fixed LPIPS, SSIM, severe-error, and detail-loss thresholds.
Synthetic contracts prove the aggregator; the actual quality cohort has not been run.
