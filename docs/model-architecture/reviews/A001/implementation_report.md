# A001 Implementation Report

## Scope

A001 implements an immutable asset-manifest boundary and read-only preflight. It does not load model tensors, inspect database rows, download assets, or run a GPU.

## Implementation

- `assets/manifest.toml` records two runtime model assets, three optional local DB/Parquet reference assets, and the three reference repositories.
- The strict Pydantic schema rejects unknown fields, unsafe relative paths, duplicate IDs/files, incomplete `ready` assets, a non-Microsoft ready Mage-VAE, missing required model file kinds, and summary/file hash disagreement.
- Runtime, selected-database, and development-reference inspections are separate scopes. Missing ignored reference repositories and optional databases cannot block Qwen/VAE runtime readiness.
- `require_runtime_assets_ready` is the only public hard-pass runtime entry point. It reads one root-confined, non-symlink manifest snapshot and uses those exact bytes for schema validation, digest, required-file inspection, and exact runtime config binding.
- Runtime readiness returns verified file identities. Consumers must obtain paths through `verified_path`, which revalidates the manifest plus every selected file and hard-fails on post-check identity or symlink drift.
- `require_databases_ready` accepts an explicit, unique list of database asset IDs and validates missing/bytes/SHA before returning a similarly revalidated identity; it never opens a database or reads rows.
- Reference auditing verifies origin/commit/clean tracked worktree/licenses independently. Origin mismatch and dirty-worktree diagnostics never serialize observed URLs, credentials, or path names.
- An obvious byte-size mismatch stops before SHA-256, avoiding a needless full read of an already-invalid multi-GiB payload.
- A blocked production preflight does not hash any large payload. During initial A001 evidence collection, the three database files were hashed once sequentially and schemas were read only from the DuckDB catalog or Parquet footer; model weights were not reread.

## AI / Model Correctness Self-Check

- Qwen config observes 24 text layers and hidden size 2048. The manifest fixes the approved custom ModelScope repo and does not substitute official Qwen.
- Mage-VAE config observes 128 latent channels, downsample factor 16, and `sample_posterior=false`. The schema only allows a ready Mage-VAE from a `microsoft/` repo.
- The manifest records `posterior_mean_required=true`, but A001 does not misrepresent a static config assertion as a real posterior API or round-trip test; that evidence remains assigned to T020.
- The user-confirmed Qwen and Mage-VAE assets are locked to immutable upstream revisions and declared file SHA-256 values without rereading the model payloads.

## Infra / Performance Self-Check

- A blocked runtime inspection performs metadata and small config JSON reads only. A fully ready runtime inspection intentionally hashes every runtime model payload sequentially before load; optional DB and reference audits do not share that path.
- Size mismatches skip SHA-256. Selected database files are validated sequentially, and no audit opens them through DuckDB/Arrow.
- All asset roots remain ignored, and `git ls-files model db reference` is empty.

## Result

Review remediation and targeted CPU verification are complete. Declared runtime models are fully locked through one readiness contract; databases and development references are explicit audit-only scopes. Independent Foundation AI/Infra re-review and main-agent acceptance remain pending.
