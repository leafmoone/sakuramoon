# S000 S001 Manual Handoff Readiness

Classification: `synthetic_bounded_engineering_only`

S001 has not started. This report records the bounded engineering evidence required
to hand fresh-start and exact-RAW-resume control to the user; it is not a formal stage,
throughput, FID/IS, long-run, or four-GPU result.

## Frozen S0 Configuration

- Resolved config identity:
  `f92da9eae4cfab8aaba9dc1cc715f85e29a9187b81300546f39e2c244e6eb62a`.
  This digest is generated from the redacted resolved config; it is not a user-supplied
  data/model/evaluator SHA.
- Seed 44; one GPU; local batch 2; accumulation 4; global batch 8.
- Planned updates 1,000; updates 1..1,000 linearly warm to `2e-5`, then remain
  constant. These values are TOML fields, not code defaults.
- RAW cadence 1,000 successful updates; two retained RAW checkpoints; storage accounts
  for one additional publishing copy.
- Metrics append every successful update and S0 uses `flush_every_updates=1`.
- The operational manifest initializes from the current ModelScope `master` tar listing
  only when missing; an existing operational snapshot is loaded locally without
  relisting mutable `master`. Validation is fixed to
  `data/2_2026.1/shard-000509.tar` and
  `data/2_2026.1/shard-000060.tar`, containing 2,099 image+JSON pairs.

## Single-GPU Engineering Evidence

The six one-update rows all executed the real data-service, local Qwen/Mage-VAE,
16-layer DiT, strict loss, backward, clip, TorchAO update, metrics, and RAW publication.

| row | samples/s | ready wait (s) | reserved bytes |
| --- | ---: | ---: | ---: |
| w1-b1 | 2.340489 | 0.314834 | 12,169,773,056 |
| w1-b2 | 2.467178 | 1.962169 | 16,661,872,640 |
| w2-b1 | 2.247280 | 0.372827 | 12,169,773,056 |
| w2-b2 | 3.914727 | 0.745925 | 17,483,956,224 |
| w3-b1 | 2.226535 | 0.406372 | 12,169,773,056 |
| w3-b2 | 3.869306 | 0.767402 | 17,483,956,224 |

`w2-b2` is the current bounded engineering selection. These one-update measurements
do not establish steady-state throughput, a maximum batch, or a formal optimum.

Fresh-process resume used the exact absolute RAW update-1 path
`checkpoints/s0-engineering-resume/w2-b1/ckpt_1_raw-1-update-cadence-8544056930b1`,
restored before connecting the service, completed update 2, and published
`ckpt_2_raw-2-stage-finalize-c51db97eee56`. Metrics contain updates `[1,2]`, learning
rates approximately `[2e-8,4e-8]`, and the final directory retains exactly those two
RAW checkpoints. A prior interrupted update-2 diagnostic remains in the ignored runtime
tree and is not selected as final evidence.

## Evaluator Boundary

Bounded evaluator plan `evaluation-508d6ccb19d4bbcd362ee637` published two real
generated images, one manual-quality index and an atomic `COMPLETE` tree from one RAW.
It records `automatic_release=false` and is permanently engineering-only.

Formal `eval.toml` currently fails before CUDA with 13 explicit unresolved bindings:
extractor name/version/path, preprocess path, real-stat path, FID trend/acceptance
samples, IS trend/acceptance samples and splits, evaluation batch/output reserve, and
manual-quality samples. No formal FID/IS result is claimed.

## Final CPU And Static Validation

- Core config/data/checkpoint/telemetry/contracts/fault group: 652 passed.
- Train/engineering/CLI group: 175 passed.
- Evaluator group: 143 passed.
- Ruff: passed.
- Affected-path Pyright: 0 errors. Full-project Pyright retains two unrelated baseline
  errors in `tests/unit/telemetry/test_profiler.py:366`.
- `git diff --check`: passed.

Post-review remediation removed the duplicated runtime acceptance gates for
`warmup_updates=1000` and `max_lr=2e-5`: both remain explicit positive TOML fields,
with scheduler maximum LR required to equal optimizer LR. Current TOML values and the
resolved config identity above are unchanged. The evaluator publisher now reserves its
final directory with atomic no-clobber `mkdir`, hard-links fsynced staged payloads, and
hard-links `COMPLETE` last; independent normal/crash/race probes passed on the configured
NFSv3 mount.

Independent remediation rereviews resolve `S000-AI-005/006` and
`S000-INFRA-003/004/005` with no new finding. Their final CPU evidence includes 227 AI
focused tests, complete data/evaluator suites of 268/144 tests, and scoped Ruff/Pyright
passes. The main-agent post-fix train selector passed 153 tests, the combined focused
selector passed 255 tests, and the corrected manifest snapshot selector passed 12 tests.

The next state transition is exclusively a user action: start the data service, then
invoke either the fresh S001 command or an exact absolute RAW resume command. No
automatic launch, latest-checkpoint selection, PMA/model-only resume, or fallback is
implemented.
