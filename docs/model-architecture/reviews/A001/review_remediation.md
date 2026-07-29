# A001 Review Remediation

The independent AI and Infra reviews requested changes. This remediation is limited to A001-owned code, tests, and evidence; it updates only A001's registry mappings and does not include the concurrent D001 checker work.

## Closed Findings

- Runtime readiness, selected optional-database audit, and reference-repository audit are explicit, separate scopes. Ignored reference checkouts and the three optional databases do not gate Qwen/VAE runtime readiness.
- `require_runtime_assets_ready` uses one root-confined manifest byte snapshot for strict parsing, SHA-256, runtime file inspection, and `AssetsConfig` binding. The package no longer exports a binding-only entry point.
- Successful runtime and database checks return `VerifiedAssetSelection`; consumers revalidate the manifest and all selected filesystem identities before obtaining a path. Manifest or asset drift after inspection hard-fails.
- Database audit requires explicit, unique manifest asset IDs and validates missing files, bytes, and SHA-256 before any database library can open the payload.
- Reference origin and worktree diagnostics redact observed values. Synthetic credential-bearing origins are absent from both JSON and exception/report representations.
- A byte mismatch stops before hashing. Tests guard the mismatched file with a hash function that raises if called.
- CLI/preflight exceptions are normalized to a stable, redacted JSON contract with explicit 0/1/2 exit codes. Missing manifest/root and raw I/O negative tests assert empty stderr and no traceback.
- File hash/config read failures report only a stable issue code and manifest-relative path; injected sensitive exception text is absent from reports and CLI output.
- Selection revalidation hashes only the small manifest, catches revalidation I/O, and deterministically rejects both atomic replacement and same-inode/same-size content drift under simulated stale NFS stat metadata. Model payloads are not rehashed at the consumer gate.
- The selection and every nested file require both an exact capability type and identity issuance by the successful preflight factories. Identity-keyed weak registries reject direct construction, `object.__new__`, structurally equal copies, and nested subclass dispatch while remaining thread-safe and reclaimable by GC. This closes the capability-forgery prerequisite found by the A002 reviewer, with task ownership retained by A001.
- Test evidence records the 12 production modules present in the isolated `80692a7`-based commit candidate.

## Verification

- A001-only isolated asset unit/contract suite: 55 passed in 7.04 seconds.
- A001-only isolated full suite: 200 passed in 17.55 seconds.
- Full repository Ruff passed; strict Pyright reported 0 errors and 0 warnings.
- Traceability passed with 221 requirements, 221 source nodes, 16 archive files, 12 production modules, and 235 runtime config keys.
- Manifest binding suite: 13 passed in each of 5 consecutive runs.
- The 32-thread issuance validation plus weak-registry GC cleanup test passed in each of 10 consecutive runs.

The latest clean candidate was cloned at base `80692a7` and received only the explicit A001 sealed-capability diff. No D001 checker/test diff, D010 data diff, or A002 scanner diff entered it. No model payload, database row, `.env`, GPU, or performance placeholder was read or created during this remediation. The affected implementation remains on the existing `C02-001`/`C03-001` paths, so registry revision 11 is unchanged; both requirements remain below `verified`, and T020/T021 retain the real model loading, posterior/forward, round-trip, and 1GPU gates.
