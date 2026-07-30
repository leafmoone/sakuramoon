# A001 minimal local-model boundary implementation report

## Scope

The original asset manifest, identity, capability, hash, and hostile-local-environment
system remains withdrawn. This remediation only makes the retained fixed-path file
checks component-specific.

## Implementation

- Added Qwen-only and Mage-VAE-only required-file checks for their fixed local paths.
- Kept the aggregate check for callers that explicitly preflight both components.
- Changed each real loader to call only its component check and removed Qwen's duplicate
  file loop.
- Kept local-only upstream loading flags, missing-file hard failures, and the absence of
  download or fallback behavior.
- Corrected the stale `.gitignore` manifest comment without changing ignore behavior.

## Self-check

The change does not inspect file contents, bytes, hashes, local repositories, licenses,
databases, `.env`, or `reference/`. It does not add runtime configuration or training
work. Component checks are startup-only filesystem metadata operations.

The independent Foundation review found the cross-component coupling. Direct
independent re-review startup did not return a valid task name; the main agent completed
remediation acceptance without representing it as independent re-review.
