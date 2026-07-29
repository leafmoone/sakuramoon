# A002 Independent Review Remediation

## Initial verdicts

- AI/model correctness: **FAIL / changes required**. The first contract lived entirely in a test helper, accepted any `from_pretrained` local-looking argument without provenance from A001, gave broad dataset-path download exceptions, and did not close aliases, `getattr`, `partial`, cache paths, or reference taint flows.
- Infra/performance: **FAIL / changes required**. The first scan skipped itself and could be bypassed through symlinks, composed paths, read-then-exec, process/search-path APIs, and hostile reference-repository Git configuration. Git metadata inspection did not disable fsmonitor, hooks, pager, external diff, interactive filters, and prompts.

These verdicts remain failures until the original independent reviewers rereview this remediation. This file is not review approval.

## Remediation

- Moved policy logic into `tools/asset_execution_boundary.py`, scanned that tool together with every Python file below `src/`, `tests/`, and `tools/`, and made enumeration/read fail closed on out-of-root paths, symlinked files/directories, non-regular files, decode errors, and before/during-read identity drift.
- Added AST import/call provenance and reference taint propagation. Parameterized negative contracts cover loader aliases, `getattr`, `partial`, remote repo/cache/arbitrary paths, missing `local_files_only`, `trust_remote_code`, download aliases, reference static/dynamic imports, composed paths, read-then-exec, `subprocess`/`os`/`asyncio`, `sys.path`, and `site`; positive contracts cover the A001-bound loader and unrelated method/text names.
- Added `VerifiedAssetSelection.verified_root(asset_id)`. A model loader source is accepted only when it comes from `require_runtime_assets_ready` or an exact `sakuramoon.assets.VerifiedAssetSelection` and names `qwen_text_encoder` or `mage_vae`; every locked file and manifest identity is revalidated before returning the root.
- Removed all model-download exceptions. Only `src/sakuramoon/data/modelscope.py::fetch_dataset_shard` may call the ModelScope download API, and only with repo `leafmoone/webdataset_danbooru`, a literal 40-lowercase-hex revision, and `repo_type="dataset"`.
- Restricted reference Git inspection to exact read-only `rev-parse --verify HEAD^{commit}`, `remote get-url origin`, and `status --porcelain=v1 --untracked-files=no`. The subprocess disables system/global config, fsmonitor, hooks, pager, external diff, interactive diff filters, optional locks, terminal prompts, and stdin; hostile local-config tests assert no sentinel executes. Unlisted commands return before subprocess creation.

## Scope and remaining gates

- CPU/static remediation only. It did not read `.env`, model, DB, dataset, cache, or ignored reference contents; did not access the network; and did not use a GPU.
- It proves execution-boundary enforcement, not real Qwen/Mage-VAE load, forward, VAE posterior mean/round-trip, performance, or any four-GPU requirement. Those gates remain assigned to their later tasks.
- Status: remediation complete, pending original independent AI and Infra rereview and main-agent acceptance.
