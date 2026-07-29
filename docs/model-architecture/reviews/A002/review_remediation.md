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

## Second independent review verdicts

- AI/model correctness: **FAIL / changes required**. The first remediation kept global, flow-insensitive root sets, so reassignment could retain stale trust; unknown expanded kwargs and ModelScope loader paths were incomplete. The dataset exception and evidence inventory also needed stricter provenance and clean-commit counts.
- Infra/performance: **FAIL / changes required**. The first remediation could be spoofed by annotations/casts, missed computed loader forms and several reference flows, allowed overly broad test Git shapes, and conflated same-name variables across functions.

These verdicts also remain failures until the original independent reviewers rereview this second remediation. No local status or evidence file represents approval.

## Second remediation

- Replaced global provenance sets with per-module/per-function environments evaluated in statement order. Assignment and reassignment kill a target and its descendants; conservative branch merges retain only valid facts. Function summaries propagate symbolic parameter and return taint across local calls without leaking same-name facts between scopes.
- Added a runtime capability gate, `require_verified_selection`, which accepts only a real `VerifiedAssetSelection` and immediately calls `require_unchanged()`. The static gate trusts only `require_runtime_assets_ready` results or narrow wrapper parameters passed through this exact gate; annotations and casts do not create selection/root provenance.
- Expanded model-loader recognition to Transformers, Diffusers, and ModelScope, including aliases, computed `getattr`, `functools.partial`, and `.from_pretrained.__call__`. `trust_remote_code`, `cache_dir`, non-literal policy values, and unknown expanded kwargs fail closed. Hugging Face download APIs include their qualified module paths and remain forbidden.
- Restricted the dataset exception to the exact `modelscope.hub.snapshot_download.snapshot_download` callable inside `src/sakuramoon/data/modelscope.py::fetch_dataset_shard`, with the exact dataset repo, revision obtained from the locked `load_config().config.data.source.revision` path, and literal `repo_type="dataset"`. Hugging Face lookalikes receive no exception.
- Extended reference taint through shell command strings, attributes, subscripts, list/dict containers, mutator calls, helper returns/parameters, computed process calls, and `sys.path` slice operations. Reassignment and cross-function same-name benign cases demonstrate the intended kill and scope isolation.
- Restricted the test-only Git exception to exact test fixture/function identities and exact allowed argv shapes. Calls containing `-c`, alias setup, unknown argv, or appended commands are rejected; the synthetic repository fixture now configures identity through explicit allowed calls.
- Excluded ignored notebook checkpoint files from scanner inventory so evidence matches a clean tracked checkout: 28 scanned Python sources and 12 tracked production modules. The governance checker may observe an ignored local checkpoint in a dirty developer workspace, but that file is neither scanned nor counted as clean-commit evidence.

## Second-remediation scope and remaining gates

- CPU/static work only. It did not read `.env`, model, DB, dataset, cache, or reference payloads; did not access the network; and did not use a GPU.
- It still proves only enforcement and provenance. Real Qwen/Mage-VAE load, forward, VAE posterior/round-trip, performance, canary, and all multi-GPU gates remain outside A002.
- Status: second remediation complete, pending original independent AI and Infra rereview and main-agent acceptance.
