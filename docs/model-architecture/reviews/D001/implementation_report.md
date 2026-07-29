# D001 implementation report

## Scope

D001 adds the repository-local requirement registry and its fail-closed checker. It does not implement runtime configuration, model code, asset loading, or any unresolved dropout value.

## Implementation

- Registered 219 normative AST nodes from the three fixed current sources using stable IDs and source fingerprints.
- Stored config, module, reference module, test, benchmark, and artifact mappings directly on every requirement, with explicit not-applicable dimensions.
- Added strict status, alias, supersession, blocker, hardware, implementation, review, artifact, inventory, path, symlink, archive checksum, changelog, and local-link validation.
- Anchored the original archive manifest SHA-256 in the checker, so changing archive payloads and `SHA256SUMS` together cannot redefine the trust root.
- Anchored all 219 bootstrap requirement ID-to-source bindings and validated every committed registry revision plus the worktree candidate. History must retain every issued ID, increment `registry_revision` exactly once, preserve locator ownership, and allocate new serials above the prior prefix maximum.
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
- Git inspection is limited to the tracked traceability registry. The number of parsed snapshots grows with registry commits and remains outside runtime/training hot paths.
- Repository-relative path and symlink checks run before configured file reads. Archive symlinks and inventory ignore patterns are rejected.
- D001 creates no performance baseline placeholders and uses no GPU time.

## Review status

Implementation self-check passed. Independent AI and Infra conclusions remain pending until the Foundation package review, as required by `AGENTS.md`.
