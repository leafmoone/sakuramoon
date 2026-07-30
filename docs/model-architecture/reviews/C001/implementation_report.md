# C001 implementation report

## Scope

C001 implements strict runtime configuration mechanics only. It does not create a runnable stage overlay, select an unresolved dropout value, load an asset, execute a kernel, or perform training.

## Implementation

- Defined frozen, exact-type Pydantic v2 tables with `extra="forbid"` for the current Foundation runtime surface. All present training-semantic fields are required; no semantic field has a code default. TOML float fields reject integer syntax and all non-finite values. S000 owns the later target-machine stage budget extension.
- Locked approved architecture/protocol values and cross-table invariants, including caption ordering, token buckets, model dimensions, x-pred/CFG/sampling, optimizer, stage topology, selected-stage enablement, and H1/H2 disablement. FID/IS acceptance sample counts remain required TOML values; 50,000 is only the current example and is benchmark-revisable. The remote dataset revision requires an immutable 40-character lowercase hexadecimal commit and its repo is fixed by the data contract.
- Implemented deterministic `extends` resolution relative to the including file. Tables merge recursively; scalars and arrays replace atomically. Traversal, symlink components, cycles, duplicate includes, and table/scalar conflicts hard-fail.
- Recorded SHA-256 for every input and produced deterministic redacted resolved TOML plus an exact SHA-256. The writer checks lexical ancestors before creating directories and atomically replaces only a non-symlink destination.
- Configuration retains only `MODELSCOPE_API_TOKEN`/W&B-style environment-variable identifiers. The loader unconditionally validates named variables without reading `.env`; transient resolution returns Pydantic `SecretStr`, and safe validation errors omit input values. The public API has no switch that can bypass this check.
- Boundary-aware redaction masks credential fields while preserving non-secret fixed local paths and dataset provenance in resolved configuration and hashes.
- Qwen and VAE configuration contains only fixed local paths and required loading semantics. It does not contain model repo, revision, tokenizer SHA, manifest, capability, or asset identity fields.

## Undecided dropout handling

Only `all_condition=0.10` is encoded as a fixed numeric value. The six component probabilities and five NL probabilities are `DECISION_REQUIRED` strings in `all_options.example.toml`, so the runtime loader rejects the example. Unit tests use clearly labeled synthetic values solely to exercise range and equality checks; they are not persisted as runtime configuration, defaults, or recommendations.

## AI/model-correctness self-check

- Fixed values were taken only from current canonical documents and the roadmap contract; archive candidates were not consulted as configuration sources.
- Unknown/missing/wrong-type/range errors, fixed-value changes, NL inequality, invalid stage transitions, backend/world-size mismatches, and unauthorized H1/H2 enablement fail before runtime work.
- `run.stage`, distributed topology, and growth enablement are cross-validated; no silent batch/backend/world-size/LR/config adjustment exists.
- `OPEN-010` remains blocked by `DECISION-DROPOUT-VALUES`; schema mechanics do not close the user decision.

## Infra/performance self-check

- Loading and hashing are startup-only CPU work and are not imported into a training hot path.
- Lexical path checks run before `resolve()` or file reads. Output parent checks run before directory creation, preventing a symlinked parent from creating directories outside the intended tree.
- Resolved serialization is deterministic and excludes credential values. No `.env`, asset, dataset, cache, checkpoint, reference repository, or GPU was accessed.
- No performance baseline/after placeholders were created because C001 is not a performance task.

## Review result

The independent Foundation review found that the schema implementation was correct but this report and the canonical task still described withdrawn model identity fields. Those evidence-only claims were removed without restoring any withdrawn schema field. Direct independent re-review startup did not return a valid task name; per the user's instruction, the main agent completed remediation acceptance without representing it as an independent re-review.
