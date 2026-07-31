# T020 implementation report

The repository contains the checkpoint-compatible Mage encoder/decoder subset and a
frozen wrapper. The loader accepts only the prepared local safetensors file. Strict CPU
loading and real RTX 5090 BF16 encode/decode passed at 32x32 and 512x512.

The reconstruction evaluator now requires an immutable canonical manifest of exactly
2,000 unique members split into 1,600 stratified and 400 risk samples, including the
locked 1,500 base-512 and 500 extended resolution classes. The manifest owns each exact
sample ID, cohort, height, and width. Its canonical bytes derive the report SHA-256, so
callers cannot substitute an unrelated digest. Synthetic contracts prove this binding
and the aggregator; the actual quality cohort has not been run.

The executor preflights an immutable batch plan against every canonical member before
model work, then checks each actual tensor batch against that plan before its forward.
It runs `encode` and `decode` under inference mode and requires the injected LPIPS/SSIM
engine to return finite per-sample vectors. Metric implementation, versions, weights,
and parameters have one source: the engine's validated identity property. There is no
built-in approximate metric, auto-download, caller-supplied identity, or fallback.

The artifact records the fixed local VAE path, derived cohort-manifest SHA-256,
LPIPS/SSIM identity, every observation, and the recomputed aggregate report. As a
reproducible report, it uses a unique temporary file, file fsync, and `os.replace`.
Injected failure proves an old report survives and the temporary file is cleaned when
replacement fails. Tests inject a synthetic metric engine only to exercise plumbing;
no production quality result is claimed. Encoders/Conditioning package review remains
pending until T020-T024 are frozen.
