# A001 Implementation Report

## Scope

A001 implements an immutable asset-manifest boundary and read-only preflight. It does not load model tensors, inspect database rows, download assets, or run a GPU.

## Implementation

- `assets/manifest.toml` records two runtime model assets, three optional local DB/Parquet reference assets, and the three reference repositories.
- The strict Pydantic schema rejects unknown fields, unsafe relative paths, duplicate IDs/files, incomplete `ready` assets, a non-Microsoft ready Mage-VAE, missing required model file kinds, and summary/file hash disagreement.
- The inspector rejects missing files, byte or SHA drift, symlinks, config architecture drift, reference origin/commit drift, tracked reference changes, and license drift.
- `require_assets_ready` has no force, skip, metadata-only, or fallback control. Any issue fails before model or DB loading.
- Runtime model configuration is bound exactly to manifest repo, revision, path, manifest hash, tokenizer hash, and architecture summary.
- A blocked production preflight does not hash any large payload. During A001 evidence collection, the three database files were hashed once sequentially and schemas were read only from the DuckDB catalog or Parquet footer; model weights were not reread.
- Reference checks invoke `git` without a shell and verify only local metadata and declared license files.

## AI / Model Correctness Self-Check

- Qwen config observes 24 text layers and hidden size 2048. The manifest fixes the approved custom ModelScope repo and does not substitute official Qwen.
- Mage-VAE config observes 128 latent channels, downsample factor 16, and `sample_posterior=false`. The schema only allows a ready Mage-VAE from a `microsoft/` repo.
- The manifest records `posterior_mean_required=true`, but A001 does not misrepresent a static config assertion as a real posterior API or round-trip test; that evidence remains assigned to T020.
- The user-confirmed Qwen and Mage-VAE assets are locked to immutable upstream revisions and declared file SHA-256 values without rereading the model payloads.

## Infra / Performance Self-Check

- The blocked production inspection performs `stat`, small config JSON reads, reference Git metadata calls, and small license hashes only. It does not hash the 4.43 GB Qwen or 345 MB VAE weights. A001 separately hashed approximately 25.15 GB of DB/Parquet payloads once in a background session.
- A fully ready manifest intentionally hashes every runtime-required payload sequentially before load. Optional database references are excluded because the user confirmed WebDataset JSON captions are the runtime source.
- All asset roots remain ignored, and `git ls-files model db reference` is empty.

## Result

Implementation and targeted CPU verification are complete. Declared runtime assets are fully locked; the databases are audit-only references and do not gate model or WebDataset use. Foundation AI/Infra review and main-agent acceptance remain pending.
