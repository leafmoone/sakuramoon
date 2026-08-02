# T052 checkpoint-driven evaluator executable integration

The evaluator now has an executable, fail-closed checkpoint path rather than only job
contracts. It accepts explicit absolute raw, model-only, PMA and accepted checkpoint
roles; model-only inputs require explicit objective provenance, raw inputs must match
the trigger update, and an accepted release requires its own exact source-PMA path.
Job identity binds checkpoint role/kind/config/update,
ordered prompt plan, batch size, Heun-50/99 NFE, CFG 2.9, extractor/version/local
file/SHA-256, preprocess file/SHA-256, real-stat file/SHA-256, IS splits, GPU placement
and training-pause policy.

Preflight rejects relative, non-canonical or symlinked repository/checkpoint/output/
identity paths. Before CUDA initialization it verifies the governed NFS identity,
atomic publication behavior and explicit evaluator output reservation. It verifies
local bytes and hashes before model loading, requires the canonical prompt manifest to
cover every complete batch, validates safe real feature statistics, and reports each
missing identity as a structured blocker. It performs no download and has no
extractor, preprocess, real-stat or checkpoint fallback.

Raw objective provenance is derived from the checkpoint's checksummed resolved TOML,
not from its CLI label. Stage-end preflight requires exactly raw/PMA/accepted roles and
binds the current PMA to ten source identities ending at the raw trigger. For an older
accepted release it separately verifies the release sidecar and the explicitly selected
source PMA's ten-source sidecar against the same strict-JLT lineage/topology. Formal
stage-end remains hard-blocked until the prompt-condition contract is governed; the
runner repeats that guard rather than trusting a caller-built plan. Initial Gaussian
solver state is generated directly in FP32. Checkpoint cost now includes metric
finalization after generator teardown, while final tree publication is reported
separately as post-commit latency and total wall time includes it.

Execution loads the selected checkpoint read-only, generates with the locked reference
sampling profile, streams CPU float64 FID/IS aggregation, writes a manual-quality image
index, and atomically publishes a complete no-clobber artifact tree with file and
directory fsync. Metrics and manual review cannot automatically release a checkpoint.

The retained post-remediation bounded runner summary at
`artifacts/s000-evaluator-bounded-remediation-20260802/test_runner_publishes_metrics_0/artifacts/evaluation-test/summary.json`
has SHA-256 `f6b773c23a2e7cfcba7e243d9b36417ea6da32b6ff8a6ad1398b39b479bd1728`
and permanent classification `synthetic_bounded_engineering_only`. A separate RTX 5090
test executed seeded CUDA FP32 initial noise and the Heun-50/99 NFE feature/FID/IS
path. Neither result is a formal FID/IS run.

Formal checkpoint evaluation remains blocked because the prompt-condition contract,
prompt/extractor/preprocess/real-stat identities, output reservation,
sample-count/cadence, GPU/pause policy and checkpoint identities are unresolved. No
assets were downloaded or invented, and no 10k/50k job was run.
