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
- The selection and every nested file require exact capability type, identity issuance by the successful preflight factories, and a complete match to the immutable issuance fingerprint. File fingerprints cover all declared identity/path/hash/stat fields; selection fingerprints cover every manifest/root/path/stat field plus ordered original file object identities. Mismatch revokes before class-level dispatch, so `object.__setattr__` retargeting, runtime-to-DB file grafting, instance method replacement, direct construction, `object.__new__`, equal copies, and nested subclasses fail closed.
- CLI parser failures now use a fixed redacted JSON contract with no stderr/usage/argv/raw exception; help remains exit 0. Semantic requests containing a sensitive asset ID are also redacted.
- `DatabaseAsset.required_for_runtime=true` is rejected by a strict `Literal[False]` schema and both unit and contract negative tests.
- Test evidence records the 12 production modules present in the isolated `72bd362`-based commit candidate.

## Verification

- A001-only isolated asset unit/contract suite: 84 passed in 9.27 seconds.
- A001-only isolated full suite: 229 passed in 24.93 seconds.
- Full repository Ruff passed; strict Pyright reported 0 errors and 0 warnings.
- Traceability passed with 221 requirements, 221 source nodes, 16 archive files, 12 production modules, and 235 runtime config keys.
- Manifest binding suite: 13 passed in each of 5 consecutive runs.
- Eighteen field-mutation, graft, 32-thread issuance, and weak-registry GC tests passed in each of 10 consecutive runs.
- The current A002 scanner independently scanned 28 isolated-candidate sources with 0 violations.

The latest clean candidate was cloned at base `72bd362` and received only the explicit A001 third-review diff. No D001 checker/test diff, D010 data diff, or A002 scanner diff entered it. No model payload, database row, `.env`, GPU, or performance placeholder was read or created during this remediation. The affected implementation remains on the existing `C02-001`/`C03-001` paths, so registry revision 11 is unchanged; both requirements remain below `verified`, and T020/T021 retain the real model loading, posterior/forward, round-trip, and 1GPU gates.
