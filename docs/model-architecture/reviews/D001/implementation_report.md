# D001 implementation report

## Scope

D001 adds the repository-local requirement registry and its fail-closed checker. It does not implement runtime configuration, model code, asset loading, or any unresolved dropout value.

## Implementation

- Registered all 221 current normative AST nodes from the three fixed sources using stable IDs and source fingerprints; the original 219 bootstrap identities remain separately anchored.
- Stored config, module, reference module, test, benchmark, and artifact mappings directly on every requirement, with explicit not-applicable dimensions.
- Added strict status, alias, supersession, blocker, hardware, implementation, review, artifact, inventory, path, symlink, archive checksum, changelog, and local-link validation.
- Anchored the original archive manifest SHA-256 in the checker, so changing archive payloads and `SHA256SUMS` together cannot redefine the trust root.
- Anchored all 219 bootstrap requirement ID-to-source bindings with a trusted canonical locator digest, including in shallow or no-Git validation. Every available committed registry revision plus the worktree candidate is also checked. History must retain every issued ID, increment `registry_revision` exactly once, preserve locator ownership, and allocate new serials above the prior prefix maximum.
- Excluded the repository-root `src/sakuramoon/**` cross-cutting boundary glob from reverse module ownership and included specific profile mappings, so a new production module must have a domain-specific mapping.
- Kept `all_condition=0.10` as confirmed and all other dropout probabilities blocked by `DECISION-DROPOUT-VALUES`.
- Added repository rules that forbid re-bootstrap/re-numbering and require traceability updates in each implementation commit.

## AI/model correctness self-check

- Canonical source precedence is encoded without reading archive candidates as implementation inputs.
- The flowchart, normative callouts, fenced protocols, nested list items, and bullets with source suffixes are independently registered.
- Alias and supersession edges must terminate at live requirements; 1GPU evidence cannot close a 4GPU requirement.
- Stable IDs may follow a legitimately revised source node, but an existing historical locator cannot be reassigned to another ID; removal and later reuse remain invalid because every history transition is checked.
- No model architecture values or undecided dropout probabilities were invented.

## Infra/performance self-check

- Verification is CPU-only and scans tracked documentation plus production Python/TOML inventory; it does not read `.env`, weights, datasets, caches, checkpoints, or reference repositories.
- Git inspection is limited to the tracked traceability registry. The trusted bootstrap digest remains effective when Git history is unavailable; when history is present, the number of parsed snapshots grows with registry commits and remains outside runtime/training hot paths.
- Repository-relative path and symlink checks run before configured file reads. Archive symlinks and inventory ignore patterns are rejected.
- D001 creates no performance baseline placeholders and uses no GPU time.

## Review status

The Foundation independent AI and Infra review found two D001 blockers: bootstrap IDs could be rebound without full Git history, and a repository-root module glob could satisfy reverse ownership. Both were remediated with negative contracts. Direct attempts to start independent re-review did not return a valid task name; per the user's instruction, the main agent completed remediation acceptance without representing it as independent re-review. See the task-specific review records for the exact boundary.
