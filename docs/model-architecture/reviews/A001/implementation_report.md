# A001 Implementation Report

## Scope

A001 implements an immutable asset-manifest boundary and read-only preflight. It does not load model tensors, inspect database rows, download assets, or run a GPU.

## Implementation

- `assets/manifest.toml` records two runtime model assets, three optional local DB/Parquet reference assets, and the three reference repositories.
- The strict Pydantic schema rejects unknown fields, unsafe relative paths, duplicate IDs/files, incomplete `ready` assets, a non-Microsoft ready Mage-VAE, missing required model file kinds, and summary/file hash disagreement.
- Runtime, selected-database, and development-reference inspections are separate scopes. Missing ignored reference repositories and optional databases cannot block Qwen/VAE runtime readiness.
- `require_runtime_assets_ready` is the only public hard-pass runtime entry point. It reads one root-confined, non-symlink manifest snapshot and uses those exact bytes for schema validation, digest, required-file inspection, and exact runtime config binding.
- Runtime readiness returns verified file identities. Consumers must obtain paths through `verified_path`, which revalidates the manifest plus every selected file and hard-fails on post-check identity or symlink drift.
- Capability narrowing accepts only exact `VerifiedAssetSelection` and nested `VerifiedAssetFile` instances issued by the successful preflight factories. Both are Python 3.12 slotted, weak-referenceable frozen dataclasses without an instance `__dict__`, so even `object.__setattr__` cannot install a same-name attribute ahead of `require_unchanged`, `verified_path`, or `verified_root`. Separate identity-keyed weak registries store immutable issuance fingerprints, use `weakref() is object`, never dataclass equality, and compare every slot field with the issued snapshot on every gate. Fingerprinting uses direct `object.__getattribute__` slot reads rather than a consumer method. The selection snapshot also records the ordered object identities of its original files. Any mismatch revokes the record before class-level revalidation dispatch, preventing field retargeting, issued DB-file grafting, direct constructors, `object.__new__`, equal-field copies, and nested subclass overrides. Locks protect issue/check/GC cleanup without retaining capabilities.
- Verified consumption rehashes the small manifest and rechecks its filesystem identity before returning a path/root. This detects same-inode, same-size content drift even when an NFS stat result is stale; already-verified model payloads receive identity checks only and are not repeatedly hashed.
- `require_databases_ready` accepts an explicit, unique list of database asset IDs and validates missing/bytes/SHA before returning a similarly revalidated identity; it never opens a database or reads rows.
- Reference auditing verifies origin/commit/clean tracked worktree/licenses independently. Origin mismatch and dirty-worktree diagnostics never serialize observed URLs, credentials, or path names.
- An obvious byte-size mismatch stops before SHA-256, avoiding a needless full read of an already-invalid multi-GiB payload.
- Root/manifest resolution errors and hash/config read failures are fail-closed and redacted. The inspection CLI emits one JSON object with exit code 0 for success, 1 for a completed failing asset report, and 2 for invocation/manifest/preflight errors; no expected asset I/O path emits a traceback.
- A safe argparse boundary converts invalid choices, unknown options, missing option values, and sensitive argument mixtures to one fixed JSON error with empty stderr and exit 2. It never renders usage, argv, or raw parser exceptions on failure; help remains a normal stdout/exit-0 path.
- `DatabaseAsset.required_for_runtime` is a strict bool with a field validator that accepts only `false`. The current audit-only database decision is enforced during manifest parsing; `true`, integer `0/1`, strings, and null/`None` cannot be coerced into an accepted value.
- A blocked production preflight does not hash any large payload. During initial A001 evidence collection, the three database files were hashed once sequentially and schemas were read only from the DuckDB catalog or Parquet footer; model weights were not reread.

## AI / Model Correctness Self-Check

- Qwen config observes 24 text layers and hidden size 2048. The manifest fixes the approved custom ModelScope repo and does not substitute official Qwen.
- Mage-VAE config observes 128 latent channels, downsample factor 16, and `sample_posterior=false`. The schema only allows a ready Mage-VAE from a `microsoft/` repo.
- The manifest records `posterior_mean_required=true`, but A001 does not misrepresent a static config assertion as a real posterior API or round-trip test; that evidence remains assigned to T020.
- The user-confirmed Qwen and Mage-VAE assets are locked to immutable upstream revisions and declared file SHA-256 values without rereading the model payloads.

## Infra / Performance Self-Check

- A blocked runtime inspection performs metadata and small config JSON reads only. A fully ready runtime inspection intentionally hashes every runtime model payload sequentially before load; optional DB and reference audits do not share that path.
- A ready selection rehashes only `assets/manifest.toml` at each consumer gate. Registry checks are constant-time identity lookups and do not repeat the approximately 4.784 GB runtime-model payload hash before every `verified_path` or `verified_root` call.
- Size mismatches skip SHA-256. Selected database files are validated sequentially, and no audit opens them through DuckDB/Arrow.
- All asset roots remain ignored, and `git ls-files model db reference` is empty.

## Result

Fourth-review remediation and isolated CPU verification are complete. The A001-only candidate based on `f7ceb2229874986492f807863b33f803ebec9566` passes 115 targeted tests, 260 full-suite tests, full Ruff, strict Pyright, and traceability. Every public file/selection consumer is covered through instance and class invocation, shadow attempts, field mutation, and runtime/database graft cases; the pre-existing identity-registry concurrency and GC suite also passes. The current stricter A002 scanner independently reports 28 sources and 0 violations for the isolated candidate. Declared runtime models are fully locked through one readiness contract; databases and development references are explicit audit-only scopes. This work remains static/hash readiness evidence only and does not close T020/T021 real-load gates. Independent Foundation AI/Infra re-review and main-agent acceptance remain pending.
