# T020 implementation report

The repository contains the checkpoint-compatible Mage encoder/decoder subset and a
frozen wrapper. The loader accepts only the prepared local safetensors file. Strict CPU
loading and real RTX 5090 BF16 encode/decode passed at 32x32 and 512x512.

The reconstruction evaluator now requires exactly 2,000 unique observations split into
1,600 stratified and 400 risk samples. It rejects non-finite or incorrectly typed
metrics and applies the fixed LPIPS, SSIM, severe-error, and detail-loss thresholds.
Synthetic contracts prove the aggregator; the actual quality cohort has not been run.

The executor now consumes explicit tensor batches, runs each through `encode` and
`decode` under inference mode, requires an injected LPIPS/SSIM engine to return finite
per-sample vectors, and builds the strict observations. Cohort hash and duplicate IDs
fail before affected model forwards. There is deliberately no built-in approximate
metric, auto-download, or fallback because the current decisions do not lock a metric
implementation or weights.

The artifact records the fixed local VAE path, cohort-manifest SHA-256, LPIPS/SSIM
implementation/version/weights/parameters, every observation, and the recomputed
aggregate report. It uses unique temporary publication, file and parent fsync, and an
atomic no-clobber hard link. Tests inject a synthetic metric engine only to exercise the
executor; no production quality result is claimed. Fresh independent review is pending.
