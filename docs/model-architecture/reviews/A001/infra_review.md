# A001 Infra / Performance Review

Review commit: `7b7d28ff42e96ad9ea2be34ef44f568d5c2954ac`

Verdict: **changes required**. The manifest and model binding are strict in the successful path, and the committed production manifest correctly excludes all three database assets from runtime inspection. However, the exported runtime preflight currently depends on ignored development reference repositories, so A001 cannot receive an Infra pass yet.

## Findings

### High: development reference repositories incorrectly gate runtime preflight

`src/sakuramoon/assets/inspect.py:249-255` skips databases whose `required_for_runtime` is false, but unconditionally inspects every `reference/` repository. `require_assets_ready()` at `inspect.py:260-266` then turns a missing reference checkout, dirty tracked reference file, remote mismatch, or reference-license mismatch into a model-loading hard failure. This conflicts with `docs/model-architecture/progress/asset-policy.md:20,32-34`: `reference/` is ignored, is not a submodule/vendor dependency, and is only a development/audit input. A clean production clone cannot reconstruct these directories from Git and would fail before loading otherwise valid Qwen/VAE assets.

Separate runtime-required model inspection from optional audit inspection. Runtime preflight must require the two locked models only; reference identity/license checks should remain an explicit A001 audit command or carry an explicit non-runtime flag. Add a contract test that removes all three reference directories and proves runtime model preflight still succeeds while the explicit reference audit reports them missing.

### Medium: runtime binding is not a single root-confined readiness contract

`src/sakuramoon/assets/bindings.py:20-32` accepts any `manifest_path`, calls `load_manifest()` directly, and hashes the file in a second read. It neither enforces the repository-root/symlink checks in `inspect_assets()` nor proves that `require_assets_ready()` ran against the same manifest. The public API therefore permits a caller to perform config binding without filesystem integrity inspection, or to bind against an external/symlinked manifest. The separate parse and hash reads also permit a manifest replacement race.

Expose one runtime entry point that reads the root-confined manifest bytes once, derives the digest and validated model identities from those same bytes, verifies required payloads, and binds `AssetsConfig`. Keep lower-level audit helpers private or clearly non-sufficient. Add integration tests for an external manifest, a final-component symlink, binding without readiness, and manifest replacement between phases.

### Medium: an invalid local Git origin can leak credentials in diagnostics

The expected manifest URL is credential-free, but `src/sakuramoon/assets/inspect.py:188-190` places the observed `git remote get-url origin` output directly into `InspectionIssue.detail`; `InspectionReport.to_json()` serializes that detail. If a local ignored reference repo has an origin such as `https://TOKEN@host/repo`, the failure report prints the token. Report only `origin_mismatch` or a redacted host/path, never the observed remote string. Add a credential-bearing synthetic remote test that asserts the secret is absent from exceptions and JSON.

### Low: obvious size drift still triggers a full payload hash

`src/sakuramoon/assets/inspect.py:95-107` records `byte_size_mismatch` and then continues hashing. For a multi-GiB file on the NFS mount, an already-conclusive size failure can still consume a full sequential read before preflight fails. Return after the size mismatch, or record that SHA verification was skipped because the byte contract already failed. The existing test at `tests/unit/assets/test_inspect.py:28-36` currently requires both errors and should be adjusted to assert fail-fast behavior.

### Low: A001 evidence has a stale module count

`docs/model-architecture/reviews/A001/test_report.json:28` records 12 production modules. The committed tree and live checker report 13; A001 added four `assets` modules and two `cli` modules to the prior seven. Correct the evidence before package closure.

## Verified

- The production manifest records exact 40-character revisions, file bytes/SHA-256, config/tokenizer summaries, licenses, and credential-free source metadata for Qwen and Mage-VAE.
- Qwen runtime fields bind to the approved custom ModelScope repo; Mage-VAE binds to the committed `microsoft/Mage-Flow` identity. Revision, path, whole-manifest hash, tokenizer hash, dtype, frozen state, and architecture summary mismatches are rejected.
- All three database assets have `required_for_runtime=false`; `inspect_assets()` does not stat, hash, or open them on the committed runtime path. This matches the confirmed WebDataset JSON caption source and the user's ownership/source statement for `dan_5_9.db`.
- `git ls-files model db reference .env` is empty. Git ignore checks cover the declared roots, local payloads are regular files with manifest-matching byte sizes, and no model/DB/reference payload is present in commit `7b7d28f`.
- Artifact file hashes in `docs/model-architecture/reviews/A001/artifacts.json` match the committed files.
- Independent rerun: 25 targeted tests passed in 3.02 seconds; Ruff and strict Pyright passed; traceability checker passed with 219 requirements, 219 source nodes, 13 production modules, 235 config keys, and 16 archive files.

## Residual Risk And Gates

This review did not hash or load the real model payloads. The production preflight implementation will intentionally perform approximately 4.77 GB of sequential model hashing, so its cold-NFS wall time and check/load race remain unmeasured. T020/T021 must still perform real model load, Mage-VAE posterior-mean/round-trip validation, Qwen tokenizer/config assertions, and failure-before-forward tests. A001 evidence must not close those gates, any GPU kernel gate, or any four-GPU/NVMe training gate.
