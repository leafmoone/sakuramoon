# T052 Bounded Evaluator Remediation

Classification: `synthetic_bounded_engineering_only`

The explicit engineering-only path now permits one exact RAW checkpoint and a bounded
manual-quality job without weakening the formal stage-end contract. Formal stage-end
still requires exactly raw, PMA and accepted roles from valid lineage, while accepted
release provenance remains explicit.

Validation prompts are derived from the configured two validation tar files and reuse
the production typed caption parser/serializer. Structured nsfw/character/copyright/
general/artist groups are preserved; tags-only records, NL identity mismatch and invalid
conditions fail before generation.

Plan `evaluation-508d6ccb19d4bbcd362ee637` completed from one exact RAW checkpoint,
generated two images, published one manual-quality index and atomically published its
final tree with `COMPLETE`. It records `automatic_release=false`; no FID/IS, extractor,
preprocess or real-stat substitution occurred.

The post-review final publisher uses the configured NFSv3-compatible no-clobber
protocol: atomic final-directory reservation, hard-linked fsynced payloads, and an
atomic `COMPLETE` hard link as the last commit operation. Independent normal,
pre-COMPLETE failure and concurrent-owner probes passed without overwrite.

CUDA-vs-wall cost validation permits only the calculable CUDA-event/perf-counter
quantization boundary. Tests cover exact 1, 60 and 1,200 second tolerances and reject
the next representable value above each boundary. Runner-level fake events cover both
checkpoint and overall scopes, and artifact JSON retains the original wall/GPU values
without clamping.

Final complete evaluator CPU validation: 144 passed. Ruff, scoped Pyright and diff
checks pass.
Formal `eval.toml` remains fail-closed on 13 explicit local identity/sample/resource
bindings, so this remediation does not claim formal FID/IS or start S001.
