# A001 Review Remediation

The independent AI and Infra reviews requested changes. This remediation is limited to A001-owned code, tests, and evidence while D001 owns the shared traceability registry.

## Closed Findings

- Runtime readiness, selected optional-database audit, and reference-repository audit are explicit, separate scopes. Ignored reference checkouts and the three optional databases do not gate Qwen/VAE runtime readiness.
- `require_runtime_assets_ready` uses one root-confined manifest byte snapshot for strict parsing, SHA-256, runtime file inspection, and `AssetsConfig` binding. The package no longer exports a binding-only entry point.
- Successful runtime and database checks return `VerifiedAssetSelection`; consumers revalidate the manifest and all selected filesystem identities before obtaining a path. Manifest or asset drift after inspection hard-fails.
- Database audit requires explicit, unique manifest asset IDs and validates missing files, bytes, and SHA-256 before any database library can open the payload.
- Reference origin and worktree diagnostics redact observed values. Synthetic credential-bearing origins are absent from both JSON and exception/report representations.
- A byte mismatch stops before hashing. Tests guard the mismatched file with a hash function that raises if called.
- Test evidence now records 13 production modules.

## Verification

- A001 unit/contract: 38 passed.
- Full live worktree: 121 passed.
- Ruff: passed.
- strict Pyright: 0 errors, 0 warnings.
- Traceability checker: 219 requirements, 219 source nodes, 13 production modules, 235 runtime config keys, 16 archive files, no errors.

No model payload, database row, `.env`, GPU, or performance placeholder was read or created during this remediation. Main-agent integration verification passed after D001 revision 4 was committed. Registry revision 5 records `task:A001` implementation paths for `C02-001` and `C03-001`; both remain below `verified`, so T020/T021 retain the real model execution gates.
